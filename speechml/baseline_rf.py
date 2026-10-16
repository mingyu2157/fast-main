"""옛 feature 레시피(10 feature) + RandomForest baseline 재학습.

왜 이걸 만드는가:
  기존 모델 아티팩트(rf_pipe.joblib, dtw_refs.pkl)를 확보할 수 없어서
  옛 모델을 그대로 돌려 비교하는 건 불가능하다. 대신 preprocess.py의
  feature 레시피를 TORGO에 **같은 fold / 같은 seed**로 재학습해 비교 상대를 만든다.

핵심 설계 3가지:
  1) DTW 참조는 **그 fold의 train 정상 화자에서만** 뽑는다.
     전체 데이터에서 뽑으면 test 화자 음성이 train feature에 새어들어
     지금까지 지켜온 화자 독립 평가가 무의미해진다.
     → fold마다 DTW를 다시 계산해야 하므로, 비싼 MFCC 슬라이스는 1회만 계산해 캐시한다.
  2) VAD는 energy 모드. preprocess.py가 "학습 전처리와 일치 필요 시 권장"이라고
     직접 적어둔 경로이고(pyin은 서빙 쪽 후속 변경 = train/serve skew),
     실측 455배 빠르다.
  3) 10초 미만 발화는 타일링으로 채운다 — 원래 서비스 동작 그대로.
     TORGO 중앙값이 2.5초라 대부분 4번쯤 반복되지만, 레시피를 고치면
     "같은 레시피" 비교가 아니게 되므로 유지한다.

실행:
  python -m speechml.baseline_rf --limit 20   # 소규모 검증
  python -m speechml.baseline_rf              # 전량 (약 40분)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from speechml import config, dataset, train_w2v

warnings.filterwarnings('ignore')
os.environ.setdefault('SPEECH_VAD_MODE', config.BASELINE_VAD_MODE)

# preprocess.py / quantile_clipper.py를 재구현하지 않고 그대로 import한다.
# 다시 구현하면 "같은 레시피"라는 주장이 깨진다.
sys.path.insert(0, str(config.BACKEND_DIR))
import librosa  # noqa: E402
from app.services.speech import preprocess as pp  # noqa: E402
from app.services.speech.quantile_clipper import QuantileClipper  # noqa: E402

FEATURE_NAMES = (
    [f'dtw_slice{i}_mean' for i in range(1, 6)]
    + ['dtw_mean_slope', 'zcr_mean', 'zcr_std', 'rms_mean', 'rms_std']
)


# ---------------------------------------------------------------- 1) 전처리 캐시

def extract_one(path) -> tuple[np.ndarray, np.ndarray]:
    """발화 1개 → (5개 MFCC 슬라이스, ZCR/RMS 4개).

    DTW는 참조가 fold마다 달라지므로 여기서 계산하지 않는다.
    """
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE)
    yv = pp.voiced_concat(y, sr=config.SAMPLE_RATE)
    yv = pp.trim_voiced_to_target(yv, 10.0, sr=config.SAMPLE_RATE)
    slices = np.stack([
        pp.mfcc_cmvn(yv[s:e], sr=config.SAMPLE_RATE, n_mfcc=13)
        for s, e in pp.slice_edges(len(yv), 5)
    ])                                      # (5, 13, T)
    zr = np.array(pp.zcr_rms(yv, sr=config.SAMPLE_RATE), dtype=np.float32)
    return slices.astype(np.float32), zr


def build_cache(d, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    print(f'[cache] MFCC 슬라이스 추출 {len(d)}개 (VAD={os.environ["SPEECH_VAD_MODE"]}, n_jobs={n_jobs})')
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(extract_one)(config.AUDIO_ROOT / rel) for rel in d['rel_path']
    )
    slices = np.stack([o[0] for o in out])
    zr = np.stack([o[1] for o in out])
    print(f'[cache] slices={slices.shape} zcr_rms={zr.shape}  {time.perf_counter() - t0:.1f}s')
    return slices, zr


# ---------------------------------------------------------------- 2) fold별 DTW

def pick_refs(d, train_idx: np.ndarray, slices: np.ndarray,
              n_refs: int, rng: np.random.Generator) -> tuple[list[np.ndarray], list[int]]:
    """fold의 train 정상 화자에서 참조 발화를 고른다.

    화자를 골고루 쓰도록 정상 화자별로 번갈아 뽑는다 — 한 사람의 발화만
    참조가 되면 그 사람 목소리와의 유사도를 재는 셈이 된다.
    """
    sub = d.iloc[train_idx]
    normals = sub[sub[config.LABEL_COL] != config.POSITIVE_LABEL]
    speakers = sorted(normals[config.GROUP_COL].unique())
    chosen: list[int] = []
    for k in range(n_refs):
        spk = speakers[k % len(speakers)]
        pool = normals.index[normals[config.GROUP_COL] == spk].to_numpy()
        chosen.append(int(rng.choice(pool)))
    # 누수 방지 검증: 참조는 반드시 train의 '정상' 화자에서만 나와야 한다.
    train_set = set(int(i) for i in train_idx)
    for i in chosen:
        assert i in train_set, f'참조 {i}가 train 밖에서 선택됨 (누수)'
        assert d.iloc[i][config.LABEL_COL] != config.POSITIVE_LABEL, \
            f'참조 {i}가 구음장애 화자 (정상 참조여야 함)'
    return [slices[i] for i in chosen], chosen


def dtw_feats_one(slice_set: np.ndarray, refs: list[np.ndarray]) -> np.ndarray:
    """발화 1개 → DTW 거리 5개 + 기울기 1개. preprocess의 함수를 그대로 쓴다."""
    means = pp.dtw_slice_means(list(slice_set), refs, scale_mode='never')
    return np.array(means + [pp.dtw_mean_slope(means)], dtype=np.float32)


def fold_features(slices: np.ndarray, zr: np.ndarray,
                  refs: list[np.ndarray], n_jobs: int) -> np.ndarray:
    """이 fold의 참조로 전체 샘플의 10 feature를 만든다."""
    dtw = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(dtw_feats_one)(slices[i], refs) for i in range(len(slices))
    )
    return np.hstack([np.stack(dtw), zr]).astype(np.float32)   # (N, 6+4)


# ---------------------------------------------------------------- 3) 학습·평가

def make_model() -> Pipeline:
    """옛 파이프라인 재구성.

    QuantileClipper는 원본 아티팩트 역직렬화에 필요했던 커스텀 트랜스포머이므로
    파이프라인에 있었음이 확실하다. Imputer는 DTW가 NaN을 낼 수 있어서 필요하다.
    RF 하이퍼파라미터는 sklearn 기본값 — 7단계에서 "표본이 작아 튜닝이 역효과"임을
    확인했으므로 양쪽 모델 모두 튜닝 없이 비교하는 게 일관된다.
    """
    return Pipeline([
        ('clip', QuantileClipper(low=0.5, high=99.5)),
        ('impute', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(random_state=config.SEED, n_jobs=-1)),
    ])


def run(d, y, folds, slices, zr, n_jobs: int) -> dict:
    """fold마다 참조를 새로 뽑고, feature를 다시 만들어 학습·평가한다."""
    rng = np.random.default_rng(config.SEED)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_score = np.full(len(y), np.nan)
    rows = []

    for i, (tr, te) in enumerate(folds, 1):
        t0 = time.perf_counter()
        refs, ref_idx = pick_refs(d, tr, slices, config.N_DTW_REFS, rng)
        ref_spk = sorted(set(d.iloc[ref_idx][config.GROUP_COL]))
        X = fold_features(slices, zr, refs, n_jobs)

        model = make_model()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        oof_score[te] = proba
        oof_pred[te] = (proba >= config.DECISION_THRESHOLD).astype(int)

        m = train_w2v.fold_metrics(y[te], oof_pred[te])
        m['fold'], m['n_test'] = i, len(te)
        rows.append(m)
        print(f'  fold{i}: 참조화자={ref_spk} '
              f'Recall {m["recall"]:.3f} Acc {m["accuracy"]:.3f} '
              f'({time.perf_counter() - t0:.0f}s)')

    assert (oof_pred >= 0).all()
    return {'oof_pred': oof_pred, 'oof_score': oof_score, 'folds': rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='앞 N개만 (검증용, 저장 안 함)')
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--refresh-cache', action='store_true')
    args = ap.parse_args()

    d, y, folds = dataset.build()
    if args.limit:
        # head()로 자르면 안 된다 — data.csv 앞부분은 정상 화자 3명뿐이어서
        # fold에 한 클래스만 들어가고 평가가 불가능해진다.
        # 화자 15명이 모두 등장하도록 화자별로 균등 샘플링한다.
        # pandas 3.0부터 groupby.apply가 그룹 컬럼을 결과에서 빼므로 head()를 쓴다.
        per = max(2, args.limit // d[config.GROUP_COL].nunique())
        d = d.groupby(config.GROUP_COL).head(per).reset_index(drop=True)
        y = dataset.get_labels(d)
        folds = dataset.speaker_folds(
            d[config.GROUP_COL].to_numpy(), y, config.N_SPLITS, config.SEED)
        print(f'[limit] 화자별 {per}개씩 → {len(d)}개 '
              f'(양성 {int(y.sum())} / 음성 {len(y) - int(y.sum())})')

    print(f'\n[baseline] 옛 레시피 10 feature + RandomForest | {len(d)}개 | seed {config.SEED}')

    if args.limit or args.refresh_cache or not config.SLICES_NPZ.exists():
        slices, zr = build_cache(d, args.n_jobs)
        if not args.limit:
            config.OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(config.SLICES_NPZ, slices=slices, zcr_rms=zr,
                                rel_path=d['rel_path'].to_numpy())
            print(f'[cache] 저장 → {config.SLICES_NPZ}')
    else:
        z = np.load(config.SLICES_NPZ, allow_pickle=True)
        assert (z['rel_path'] == d['rel_path'].to_numpy()).all(), '캐시 순서 불일치'
        slices, zr = z['slices'], z['zcr_rms']
        print(f'[cache] 재사용 → {config.SLICES_NPZ}  slices={slices.shape}')

    print(f'[dtw] 참조 {config.N_DTW_REFS}개 × fold {len(folds)}개 (fold별 train 정상 화자에서만)')
    res = run(d, y, folds, slices, zr, args.n_jobs)

    if not args.limit:
        np.savez_compressed(config.OUT_DIR / 'baseline_rf_oof.npz',
                            oof_pred=res['oof_pred'], oof_score=res['oof_score'],
                            y=y, rel_path=d['rel_path'].to_numpy())
        print(f'[save] OOF 예측 저장 완료')

    train_w2v.report('baseline RF (옛 레시피 10 feature)', y, res,
                     d[config.GROUP_COL].to_numpy())


if __name__ == '__main__':
    main()
