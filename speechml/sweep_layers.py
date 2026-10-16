"""레이어 × 풀링 비교 — 어느 표현이 구음장애 신호를 가장 잘 담는가.

배경:
  6단계는 마지막 레이어 + mean 풀링만 썼고 AUC 0.710에 그쳤다.
  7단계 [1]에서 정규화·PCA로는 개선되지 않음을 확인했으므로(가설 기각),
  남은 가능성은 "표현을 잘못 골랐다"는 것이다.

  wav2vec2의 마지막 레이어는 사전학습 과제(마스킹된 양자화 단위 예측)에
  특화되어 있어, 음향·병리 정보는 중간 레이어에 더 많이 남는다는 보고가 많다.
  여기서는 13개 레이어 × 3가지 풀링을 전부 비교해 실제로 그런지 확인한다.

C는 1.0으로 고정한다 — 7단계 [3]의 중첩 교차검증에서 C 튜닝이
표본 부족으로 역효과임을 확인했기 때문이다.

실행: python -m speechml.sweep_layers
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from speechml import config, dataset, train_w2v

POOLINGS = {
    'mean': lambda v: v[:, :768],
    'std': lambda v: v[:, 768:],
    'mean+std': lambda v: v,
}


def make_model() -> Pipeline:
    return Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=1.0, random_state=config.SEED)),
    ])


def load_layers():
    if not config.EMBED_LAYERS_NPZ.exists():
        raise FileNotFoundError(
            f'{config.EMBED_LAYERS_NPZ} 가 없습니다. '
            '먼저 `python -m speechml.embed --all-layers --device cpu` 를 실행하세요.'
        )
    z = np.load(config.EMBED_LAYERS_NPZ, allow_pickle=True)
    return z['X'], z['y'], z['rel_path'], z['speaker_id']


def evaluate(X, y, folds) -> dict:
    res = train_w2v.run_cv(X, y, folds, make_model)
    pred, score = res['oof_pred'], res['oof_score']
    return {
        'recall': recall_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'acc': accuracy_score(y, pred),
        'auc': roc_auc_score(y, score),
        'fold_std': np.std([r['recall'] for r in res['folds']]),
        'fn': int(((pred == 0) & (y == 1)).sum()),
        'res': res,
    }


def main() -> None:
    np.random.seed(config.SEED)
    d, y, folds = dataset.build()
    Xl, y2, rel, spk = load_layers()
    assert (y == y2).all() and (d['rel_path'].to_numpy() == rel).all()

    n_layers = Xl.shape[1]
    print(f'\n[sweep] X={Xl.shape} (샘플, 레이어, mean|std) | fold {len(folds)}개')
    print('  기준(6단계): 마지막 레이어 + mean → Recall 0.641 | AUC 0.710 | FN 356\n')

    print(f'  {"레이어":>6} {"풀링":>9} {"Recall":>8} {"Prec":>7} {"F1":>7} '
          f'{"Acc":>7} {"AUC":>7} {"fold±":>7} {"FN":>5}')
    best = None
    for layer in range(n_layers):
        for pname, pfn in POOLINGS.items():
            m = evaluate(pfn(Xl[:, layer, :]), y, folds)
            flag = ''
            if best is None or m['auc'] > best[0]['auc']:
                best = (m, layer, pname)
                flag = '  ←'
            print(f'  {layer:>6} {pname:>9} {m["recall"]:>8.3f} {m["precision"]:>7.3f} '
                  f'{m["f1"]:>7.3f} {m["acc"]:>7.3f} {m["auc"]:>7.3f} '
                  f'{m["fold_std"]:>7.3f} {m["fn"]:>5}{flag}')
        print()

    m, layer, pname = best
    print(f'  ★ 최고: layer={layer}, pooling={pname} → AUC {m["auc"]:.3f} '
          f'(기준 0.710 대비 {m["auc"] - 0.710:+.3f})')

    print('\n  최고 설정의 화자별 정확도')
    train_w2v.report(f'layer{layer}-{pname}', y, m['res'], spk)


if __name__ == '__main__':
    main()
