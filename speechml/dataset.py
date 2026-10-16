"""데이터셋 로드 · 필터 · 화자 단위 fold 생성.

실행:  python -m speechml.dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from speechml import config


def load_meta() -> pd.DataFrame:
    """build_meta.py가 만든 메타 테이블을 읽는다."""
    if not config.META_CSV.exists():
        raise FileNotFoundError(
            f'{config.META_CSV} 가 없습니다. 먼저 build_meta.py 를 실행하세요.'
        )
    return pd.read_csv(config.META_CSV)


def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    """EDA 결론에 따른 필터를 적용하고, 무엇을 왜 버렸는지 출력한다."""
    before = len(d)
    dropped = d[d['duration_sec'] < config.MIN_DURATION_SEC]
    kept = d[d['duration_sec'] >= config.MIN_DURATION_SEC].reset_index(drop=True)

    print(f'[filter] duration_sec >= {config.MIN_DURATION_SEC}s')
    print(f'  {before} -> {len(kept)}  (제거 {len(dropped)}건)')
    if len(dropped):
        print(f'  제거된 화자: {dropped["speaker_id"].value_counts().to_dict()}')
        print(f'  제거된 클래스: {dropped[config.LABEL_COL].value_counts().to_dict()}')
    return kept


def get_labels(d: pd.DataFrame) -> np.ndarray:
    """문자열 라벨을 0/1로 변환. 양성(1) = 구음장애 = 놓치면 안 되는 쪽."""
    return (d[config.LABEL_COL] == config.POSITIVE_LABEL).astype(int).to_numpy()


def speaker_table(d: pd.DataFrame) -> pd.DataFrame:
    """화자 1명 = 1행인 테이블. TORGO는 화자마다 라벨이 하나로 고정이다."""
    return (
        d.groupby(config.GROUP_COL)[config.LABEL_COL]
         .first()
         .reset_index()
    )


def make_folds(d: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """화자 단위 층화 K-fold 인덱스.

    핵심: 층화를 '발화'가 아니라 '화자' 수준에서 한다.
    TORGO는 화자 한 명이 전부 같은 클래스이므로, 발화 단위로 층화를 요구하면
    제약이 사실상 충족 불가능해져서 한쪽 클래스만 담긴 fold가 생긴다.
    15명을 라벨로 층화해 나눈 뒤, 그 화자들의 발화를 전부 끌어오면
    (1) 화자 누수가 원천적으로 불가능하고 (2) 모든 fold에 두 클래스가 들어간다.
    """
    return speaker_folds(
        d[config.GROUP_COL].to_numpy(), get_labels(d), config.N_SPLITS, config.SEED
    )


def speaker_folds(sample_speakers: np.ndarray, sample_y: np.ndarray,
                  n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """샘플별 화자 배열 + 라벨로부터 화자 단위 층화 fold를 만든다.

    make_folds()와 중첩 교차검증(내부 fold)에서 공통으로 쓴다.
    """
    speakers, first_idx = np.unique(sample_speakers, return_index=True)
    spk_y = sample_y[first_idx]          # 화자 1명 = 라벨 1개

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for _, te_s in skf.split(speakers.reshape(-1, 1), spk_y):
        in_test = np.isin(sample_speakers, speakers[te_s])
        folds.append((np.where(~in_test)[0], np.where(in_test)[0]))
    return folds


def describe_folds(d: pd.DataFrame, folds: list) -> None:
    """fold 구성을 출력하고 화자 누수가 있으면 즉시 중단한다."""
    y = get_labels(d)
    print(f'\n[folds] 화자 단위 StratifiedKFold(n_splits={config.N_SPLITS}, seed={config.SEED})')
    seen_in_test: set[str] = set()
    for i, (tr, te) in enumerate(folds, 1):
        tr_spk = set(d.iloc[tr][config.GROUP_COL])
        te_spk = set(d.iloc[te][config.GROUP_COL])
        leaked = tr_spk & te_spk
        if leaked:
            raise AssertionError(f'fold {i}: 화자 누수 발생 {sorted(leaked)}')
        seen_in_test |= te_spk
        pos = int(y[te].sum())
        neg = len(te) - pos
        if pos == 0 or neg == 0:
            raise AssertionError(
                f'fold {i}: test에 한 클래스만 있음 (양성 {pos} / 음성 {neg}). '
                'Recall 또는 Precision을 계산할 수 없다.'
            )
        print(f'  fold{i}: train {len(tr):4d} / test {len(te):4d}'
              f' | test 양성 {pos:3d} 음성 {neg:3d} ({pos / len(te):5.1%})'
              f' | test 화자 {sorted(te_spk)}')
    total_spk = d[config.GROUP_COL].nunique()
    print(f'  화자 누수 없음 ✅ / test에 등장한 화자 {len(seen_in_test)}명 (전체 {total_spk}명)')


def build() -> tuple[pd.DataFrame, np.ndarray, list]:
    """다른 스크립트가 쓰는 진입점. (데이터, 라벨, fold) 반환."""
    d = apply_filters(load_meta())
    return d, get_labels(d), make_folds(d)


def main() -> None:
    d, y, folds = build()
    pos = int(y.sum())
    print(f'\n[data] {len(d)}개 | 양성(구음장애) {pos} / 음성(정상) {len(y) - pos}')
    describe_folds(d, folds)


if __name__ == '__main__':
    main()
