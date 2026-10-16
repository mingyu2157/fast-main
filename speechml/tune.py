"""정규화 · 차원축소 스윕 — 임베딩 재추출 없이 즉시 할 수 있는 개선 탐색.

6단계 결과의 진단은 "768차원 선형 모델이 1,500샘플에 과적합" 이었다.
근거: OOF 확률이 0 또는 1로 극단화되어 threshold 조정 여지가 거의 없었다.
그래서 먼저 정규화 세기(C)와 차원 축소(PCA)만 바꿔 어디까지 개선되는지 본다.

⚠️ 주의: 같은 fold로 여러 설정을 비교하면 그 fold에 맞춰 고르게 되어
평가치가 낙관적으로 부풀 수 있다(selection bias). 그래서 마지막에
중첩 교차검증으로 "정직한 수치"를 따로 낸다.

실행: python -m speechml.tune
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from speechml import config, dataset, train_w2v

C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]
PCA_GRID = [32, 64, 128, 256]


def make_logreg(C: float, pca: int | None = None) -> Pipeline:
    steps = [('scale', StandardScaler())]
    if pca:
        steps.append(('pca', PCA(n_components=pca, random_state=config.SEED)))
    steps.append(('clf', LogisticRegression(max_iter=5000, C=C,
                                            random_state=config.SEED)))
    return Pipeline(steps)


def summarize(y, res: dict) -> dict:
    pred, score = res['oof_pred'], res['oof_score']
    per_fold_recall = np.array([r['recall'] for r in res['folds']])
    return {
        'recall': recall_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'acc': accuracy_score(y, pred),
        'auc': roc_auc_score(y, score),
        'fold_std': per_fold_recall.std(),
        'fn': int(((pred == 0) & (y == 1)).sum()),
    }


def print_header(title: str) -> None:
    print(f'\n{title}')
    print(f'  {"설정":>14} {"Recall":>8} {"Prec":>7} {"F1":>7} {"Acc":>7} '
          f'{"AUC":>7} {"fold±":>7} {"FN":>5}')


def print_row(label: str, m: dict) -> None:
    print(f'  {label:>14} {m["recall"]:>8.3f} {m["precision"]:>7.3f} {m["f1"]:>7.3f} '
          f'{m["acc"]:>7.3f} {m["auc"]:>7.3f} {m["fold_std"]:>7.3f} {m["fn"]:>5}')


def sweep_C(X, y, folds) -> float:
    print_header('[1] 정규화 세기 (C가 작을수록 강한 정규화)')
    best_c, best_auc = None, -1.0
    for C in C_GRID:
        m = summarize(y, train_w2v.run_cv(X, y, folds, lambda C=C: make_logreg(C)))
        print_row(f'C={C:g}', m)
        if m['auc'] > best_auc:
            best_auc, best_c = m['auc'], C
    print(f'  → AUC 최고: C={best_c:g} (AUC {best_auc:.3f})')
    return best_c


def sweep_pca(X, y, folds, C: float) -> None:
    print_header(f'[2] PCA 차원 축소 (C={C:g} 고정)')
    for n in PCA_GRID:
        m = summarize(y, train_w2v.run_cv(X, y, folds, lambda n=n: make_logreg(C, n)))
        print_row(f'PCA={n}', m)


def nested_cv(X, y, folds, d) -> None:
    """fold마다 train 화자만으로 C를 고르고, 그 C로 test를 평가한다.

    바깥 fold의 test는 C 선택에 일절 쓰이지 않으므로 selection bias가 없다.
    """
    oof_score = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1, dtype=int)
    chosen = []

    for tr, te in folds:
        tr_spk = d.iloc[tr][config.GROUP_COL].to_numpy()
        tr_y = y[tr]
        inner = dataset.speaker_folds(tr_spk, tr_y, n_splits=3, seed=config.SEED)

        best_c, best_auc = None, -1.0
        for C in C_GRID:
            aucs = []
            for itr, ite in inner:
                m = make_logreg(C)
                m.fit(X[tr][itr], tr_y[itr])
                aucs.append(roc_auc_score(tr_y[ite], m.predict_proba(X[tr][ite])[:, 1]))
            a = float(np.mean(aucs))
            if a > best_auc:
                best_auc, best_c = a, C

        final = make_logreg(best_c)
        final.fit(X[tr], tr_y)
        oof_score[te] = final.predict_proba(X[te])[:, 1]
        oof_pred[te] = final.predict(X[te])
        chosen.append(best_c)

    print('\n[3] 중첩 교차검증 — selection bias 없는 정직한 수치')
    print(f'  fold별로 고른 C: {[f"{c:g}" for c in chosen]}')
    fn = int(((oof_pred == 0) & (y == 1)).sum())
    print(f'  Recall {recall_score(y, oof_pred):.3f} | '
          f'Precision {precision_score(y, oof_pred, zero_division=0):.3f} | '
          f'F1 {f1_score(y, oof_pred):.3f} | '
          f'Acc {accuracy_score(y, oof_pred):.3f} | '
          f'AUC {roc_auc_score(y, oof_score):.3f} | FN {fn}')


def main() -> None:
    np.random.seed(config.SEED)
    d, y, folds = dataset.build()
    X, y2, rel, spk = train_w2v.load_embeddings()
    assert (y == y2).all() and (d['rel_path'].to_numpy() == rel).all()

    print(f'\n[tune] X={X.shape} | fold {len(folds)}개 | seed {config.SEED}')
    print('  기준(6단계): Recall 0.641 | AUC 0.710 | FN 356  (C=1.0, PCA 없음)')

    best_c = sweep_C(X, y, folds)
    sweep_pca(X, y, folds, best_c)
    nested_cv(X, y, folds, d)


if __name__ == '__main__':
    main()
