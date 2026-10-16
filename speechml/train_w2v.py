"""wav2vec2 임베딩 + 선형 분류기 — 화자 단위 5-fold 교차검증.

보고 방식 두 가지를 함께 낸다:
  (1) OOF 풀링 — 1,991개 전원이 정확히 한 번씩 평가된 혼동행렬 1개. 대표 수치.
      fold별 샘플 수 편차(화자당 발화 수가 7.5배까지 차이)가 이 값을 왜곡하지 않는다.
  (2) fold별 평균 ± 표준편차 — 어떤 화자가 test에 들어가느냐에 따른 변동폭.
      화자가 15명뿐이라 이 분산을 숨기면 안 된다.

평가 우선순위는 Recall (실제 구음장애를 정상으로 판단하는 미탐지를 줄이는 게 최우선).

실행: python -m speechml.train_w2v
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from speechml import config, dataset

MODELS = {
    # 768차원 × 약 1,500 학습샘플 = 샘플 대비 고차원.
    # 강한 정규화를 걸 수 있는 선형 모델이 적합하고, 확률을 주므로 7단계 threshold 조정에 쓸 수 있다.
    'logreg': lambda: Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=config.SEED)),
    ]),
    # 비선형 비교군. 커널 SVM이 선형보다 나은지 확인용.
    'svm_rbf': lambda: Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=config.SEED)),
    ]),
}


def load_embeddings():
    if not config.EMBED_NPZ.exists():
        raise FileNotFoundError(
            f'{config.EMBED_NPZ} 가 없습니다. 먼저 `python -m speechml.embed` 를 실행하세요.'
        )
    z = np.load(config.EMBED_NPZ, allow_pickle=True)
    return z['X'], z['y'], z['rel_path'], z['speaker_id']


def fold_metrics(y_true, y_pred) -> dict:
    return {
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }


def run_cv(X, y, folds, make_model, threshold: float | None = None) -> dict:
    """fold별로 학습·평가하고 OOF 예측을 모은다.

    threshold: 양성 판정 기준. None이면 config.DECISION_THRESHOLD(0.15)를 쓴다.
      기본 0.5(predict())가 아니라 낮춘 값을 쓰는 이유는 미탐지를 줄이기 위함.
      decision_function만 제공하는 모델(SVC 등)은 확률 스케일이 아니므로
      threshold를 적용할 수 없고 predict()를 그대로 쓴다.
    """
    thr = config.DECISION_THRESHOLD if threshold is None else threshold
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_score = np.full(len(y), np.nan, dtype=float)
    rows = []

    for i, (tr, te) in enumerate(folds, 1):
        model = make_model() if callable(make_model) else MODELS[make_model]()
        # StandardScaler를 Pipeline 안에 둔 이유: fold의 train으로만 fit 해야 한다.
        # 전체 데이터로 미리 스케일링하면 test 통계가 train에 새어든다(data leakage).
        model.fit(X[tr], y[tr])
        clf = model.named_steps['clf']
        if hasattr(clf, 'predict_proba'):
            proba = model.predict_proba(X[te])[:, 1]
            oof_score[te] = proba
            pred = (proba >= thr).astype(int)
        else:
            oof_score[te] = model.decision_function(X[te])
            pred = model.predict(X[te])
        oof_pred[te] = pred
        m = fold_metrics(y[te], pred)
        m['fold'] = i
        m['n_test'] = len(te)
        rows.append(m)

    assert (oof_pred >= 0).all(), 'OOF 예측이 비어 있는 샘플이 있습니다'
    return {'oof_pred': oof_pred, 'oof_score': oof_score, 'folds': rows}


def report(name: str, y, res: dict, spk) -> None:
    print(f'\n{"=" * 68}\n[{name}]\n{"=" * 68}')

    print('  fold별 성능')
    print(f'  {"fold":>4} {"n_test":>7} {"Recall":>8} {"Precision":>10} {"F1":>8} {"Acc":>8}')
    for r in res['folds']:
        print(f'  {r["fold"]:>4} {r["n_test"]:>7} {r["recall"]:>8.3f} '
              f'{r["precision"]:>10.3f} {r["f1"]:>8.3f} {r["accuracy"]:>8.3f}')

    for key in ('recall', 'precision', 'f1', 'accuracy'):
        vals = np.array([r[key] for r in res['folds']])
        print(f'  {key:>10}: {vals.mean():.3f} ± {vals.std():.3f}')

    pred = res['oof_pred']
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f'\n  OOF 풀링 (n={len(y)})')
    print(f'    혼동행렬        예측:정상  예측:구음장애')
    print(f'      실제 정상     {tn:>8}  {fp:>12}')
    print(f'      실제 구음장애 {fn:>8}  {tp:>12}   ← fn={fn} 이 미탐지')
    m = fold_metrics(y, pred)
    auc = roc_auc_score(y, res['oof_score'])
    print(f'    Recall {m["recall"]:.3f} | Precision {m["precision"]:.3f} | '
          f'F1 {m["f1"]:.3f} | Acc {m["accuracy"]:.3f} | ROC-AUC {auc:.3f}')

    print('\n  화자별 정확도 (OOF)')
    order = sorted(set(spk.tolist()))
    for s in order:
        mask = spk == s
        acc = (pred[mask] == y[mask]).mean()
        label = 'dys ' if y[mask][0] == 1 else 'norm'
        bar = '█' * int(acc * 20)
        print(f'    {s:>5} {label} n={mask.sum():>3}  {acc:.3f}  {bar}')


def main() -> None:
    np.random.seed(config.SEED)
    d, y_df, folds = dataset.build()
    X, y, rel, spk = load_embeddings()
    assert (d['rel_path'].to_numpy() == rel).all(), '임베딩과 메타 순서가 다릅니다'
    assert (y_df == y).all(), '라벨 불일치'

    print(f'\n[train] X={X.shape} | 양성 {int(y.sum())} / 음성 {len(y) - int(y.sum())}'
          f' | fold {len(folds)}개 | seed {config.SEED}')

    for name in MODELS:
        res = run_cv(X, y, folds, name)
        report(name, y, res, spk)


if __name__ == '__main__':
    main()
