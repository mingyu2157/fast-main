"""8단계 최종 비교 — wav2vec2(layer8+mean) vs 옛 레시피 RF baseline.

동일 조건: 같은 1,991개 / 같은 화자 단위 5-fold / 같은 seed(42) / 같은 OOF 풀링.
보고 지표 3종:
  - ROC-AUC         threshold와 무관 → 가장 공정한 단일 비교
  - thr=0.5         기본 설정
  - thr=0.15        새 모델의 운용 설정
threshold는 새 모델 기준으로 고른 값이므로, 그것만으로 비교하면 baseline에 불리하다.
그래서 AUC와 각 모델의 threshold 스윕을 함께 본다.

실행: python -m speechml.compare
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from speechml import config, dataset, sweep_layers, train_w2v


def metrics_at(y, score, thr: float) -> dict:
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        'recall': recall_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'acc': accuracy_score(y, pred),
        'fn': int(fn), 'fp': int(fp),
    }


def main() -> None:
    d, y, folds = dataset.build()
    spk = d[config.GROUP_COL].to_numpy()

    # 새 모델 — 임베딩 캐시가 있어 즉시 재현된다
    Xl, y2, rel, _ = sweep_layers.load_layers()
    assert (y == y2).all() and (d['rel_path'].to_numpy() == rel).all()
    X = sweep_layers.POOLINGS[config.W2V_POOLING](Xl[:, config.W2V_LAYER, :])
    w2v = train_w2v.run_cv(X, y, folds, sweep_layers.make_model)
    s_w2v = w2v['oof_score']

    # baseline — baseline_rf.py가 저장한 OOF
    z = np.load(config.OUT_DIR / 'baseline_rf_oof.npz', allow_pickle=True)
    assert (z['rel_path'] == rel).all() and (z['y'] == y).all(), 'baseline OOF 정렬 불일치'
    s_rf = z['oof_score']

    models = {
        f'wav2vec2 layer{config.W2V_LAYER}+{config.W2V_POOLING} (768d)': s_w2v,
        '옛 레시피 RF (10 feature)': s_rf,
    }

    print(f'\n{"=" * 76}')
    print(f'8단계 비교 — 동일 데이터 {len(y)}개 / 화자 단위 {len(folds)}-fold / seed {config.SEED}')
    print(f'{"=" * 76}')

    print('\n■ ROC-AUC (threshold 무관 — 가장 공정한 비교)')
    for name, s in models.items():
        print(f'    {name:<44} {roc_auc_score(y, s):.3f}')

    for thr in (0.5, config.DECISION_THRESHOLD):
        print(f'\n■ threshold = {thr}')
        print(f'    {"모델":<44} {"Recall":>7} {"Prec":>7} {"F1":>7} {"Acc":>7} {"FN":>5} {"FP":>5}')
        for name, s in models.items():
            m = metrics_at(y, s, thr)
            print(f'    {name:<44} {m["recall"]:>7.3f} {m["precision"]:>7.3f} '
                  f'{m["f1"]:>7.3f} {m["acc"]:>7.3f} {m["fn"]:>5} {m["fp"]:>5}')

    print('\n■ threshold 스윕 — baseline을 유리한 지점에 놓아도 되는지 확인')
    print(f'    {"thr":>5} {"w2v Recall":>11} {"w2v Acc":>9} {"RF Recall":>11} {"RF Acc":>9}')
    for thr in (0.70, 0.60, 0.50, 0.35, 0.25, 0.15):
        a, b = metrics_at(y, s_w2v, thr), metrics_at(y, s_rf, thr)
        print(f'    {thr:>5.2f} {a["recall"]:>11.3f} {a["acc"]:>9.3f} '
              f'{b["recall"]:>11.3f} {b["acc"]:>9.3f}')

    print('\n■ 화자별 OOF 정확도 (thr=0.15)')
    print(f'    {"화자":>6} {"라벨":>5} {"n":>4} {"w2v":>7} {"RF":>7}  차이')
    pw = (s_w2v >= config.DECISION_THRESHOLD).astype(int)
    pr = (s_rf >= config.DECISION_THRESHOLD).astype(int)
    for sp in sorted(set(spk.tolist())):
        m = spk == sp
        aw = (pw[m] == y[m]).mean()
        ar = (pr[m] == y[m]).mean()
        lab = 'dys' if y[m][0] == 1 else 'norm'
        mark = '▲' if aw > ar + 0.05 else ('▼' if ar > aw + 0.05 else '=')
        print(f'    {sp:>6} {lab:>5} {m.sum():>4} {aw:>7.3f} {ar:>7.3f}  {mark} {aw - ar:+.3f}')


if __name__ == '__main__':
    main()
