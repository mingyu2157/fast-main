"""최종 모델 학습 및 서빙 아티팩트 저장 (9단계 선행 작업).

지금까지의 학습은 전부 교차검증이었다 — fold마다 모델을 만들고 버렸다.
서비스에 올릴 단일 모델은 아직 없으므로, **전체 1,991개로 학습**해 저장한다.

저장물:
  clf.joblib   StandardScaler + LogisticRegression (768차원 입력 → 확률)
  meta.json    서빙 계약서 — 레이어/풀링/임계값/샘플레이트/전처리 규칙/CV 성능

주의: wav2vec2 인코더는 여기 포함되지 않는다. 서비스는 인코더를 따로 로드해
768차원 벡터를 만들어 clf.joblib에 넣는 구조다(관심사 분리).

실행: python -m speechml.train_final
"""
from __future__ import annotations

import json
import platform
from datetime import date

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from speechml import config, dataset, sweep_layers, train_w2v

OUT_DIR = config.PROJECT_ROOT / 'back-end' / 'app' / 'assets' / 'models' / 'speech' / 'v2'


def main() -> None:
    np.random.seed(config.SEED)
    d, y, folds = dataset.build()
    Xl, y2, rel, spk = sweep_layers.load_layers()
    assert (y == y2).all() and (d['rel_path'].to_numpy() == rel).all()

    X = sweep_layers.POOLINGS[config.W2V_POOLING](Xl[:, config.W2V_LAYER, :])
    print(f'\n[final] X={X.shape} | layer={config.W2V_LAYER} | pooling={config.W2V_POOLING}')

    # 1) 기대 성능은 '전량 학습 모델'이 아니라 교차검증에서 나온다.
    #    전량 학습 모델의 train 성능은 일반화 추정치가 아니므로 참고용으로만 찍는다.
    cv = train_w2v.run_cv(X, y, folds, sweep_layers.make_model)
    pred, score = cv['oof_pred'], cv['oof_score']
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    cv_perf = {
        'protocol': f'화자 단위 StratifiedKFold(n_splits={config.N_SPLITS}, seed={config.SEED}), OOF 풀링',
        'threshold': config.DECISION_THRESHOLD,
        'recall': round(float(recall_score(y, pred)), 4),
        'precision': round(float(precision_score(y, pred)), 4),
        'f1': round(float(f1_score(y, pred)), 4),
        'accuracy': round(float(accuracy_score(y, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y, score)), 4),
        'false_negatives': int(fn),
        'false_positives': int(fp),
        'per_fold_recall_mean': round(float(np.mean([r['recall'] for r in cv['folds']])), 4),
        'per_fold_recall_std': round(float(np.std([r['recall'] for r in cv['folds']])), 4),
    }
    print(f'[final] 기대 성능(OOF): Recall {cv_perf["recall"]} | AUC {cv_perf["roc_auc"]} | FN {cv_perf["false_negatives"]}')

    # 2) 전량 학습
    model = sweep_layers.make_model()
    model.fit(X, y)
    train_proba = model.predict_proba(X)[:, 1]
    train_acc = accuracy_score(y, (train_proba >= config.DECISION_THRESHOLD).astype(int))
    print(f'[final] 전량 학습 완료. train 정확도 {train_acc:.4f} '
          f'(⚠️ 일반화 추정치 아님 — 같은 데이터로 재채점)')

    # 3) 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clf_path = OUT_DIR / 'clf.joblib'
    joblib.dump(model, clf_path)

    meta = {
        'model_name': 'wav2vec2-base(frozen) + StandardScaler + LogisticRegression',
        'created_at': date.today().isoformat(),
        'seed': config.SEED,

        # --- 서빙이 반드시 맞춰야 하는 계약 ---
        'embedding': {
            'w2v_model_id': config.W2V_MODEL_ID,
            'w2v_local_dir_hint': str(config.W2V_LOCAL_DIR.relative_to(config.PROJECT_ROOT)),
            'layer': config.W2V_LAYER,
            'layer_note': 'output_hidden_states=True 의 hidden_states[8]. 마지막 레이어 아님',
            'pooling': config.W2V_POOLING,
            'embed_dim': int(X.shape[1]),
            'sample_rate': config.SAMPLE_RATE,
        },
        'preprocessing': {
            'audio_load': 'librosa.load(path, sr=16000) — 원본 전체 길이',
            'vad': 'none',
            'crop': 'none',
            'tiling': 'none',
            'warning': 'VAD / 10초 크롭 / 타일링을 넣으면 학습 입력과 달라져 성능이 무효가 된다',
            'min_duration_sec': config.MIN_DURATION_SEC,
            'min_duration_note': '학습에서 0.3초 미만 8건 제외. 서빙에서는 전처리가 아니라 입력 검증으로 처리할 것',
        },
        'decision': {
            'threshold': config.DECISION_THRESHOLD,
            'positive_label': config.POSITIVE_LABEL,
            'positive_index': 1,
            'rule': 'predict_proba(X)[:, 1] >= threshold → 구음장애',
            'threshold_note': '기본 0.5가 아니다. 미탐지를 줄이기 위해 0.15로 낮춤',
        },

        'training_data': {
            'dataset': 'TORGO (Kaggle iamhungundji/dysarthria-detection)',
            'n_samples': int(len(y)),
            'n_speakers': int(d[config.GROUP_COL].nunique()),
            'positive': int(y.sum()),
            'negative': int(len(y) - y.sum()),
            'license': 'academic use only — 재배포 금지',
            'limitations': [
                '영어 발화 (서비스는 한국어 대상)',
                '병인이 뇌성마비·ALS로 뇌졸중이 아님',
                '경증 구음장애 탐지율 낮음 (F03 화자 OOF 정확도 0.401)',
            ],
        },
        'expected_performance': cv_perf,
        'train_fit_accuracy': round(float(train_acc), 4),
        'train_fit_note': '같은 데이터로 재채점한 값. 일반화 성능이 아니며 비교에 쓰지 말 것',
        'baseline_comparison': {
            'old_recipe_rf_roc_auc': 0.562,
            'new_model_roc_auc': cv_perf['roc_auc'],
            'caveat': '옛 레시피의 DTW는 고정 프롬프트를 전제하므로 TORGO에서 불리하다. 과대 주장 금지',
        },
        'environment': {
            'python': platform.python_version(),
            'scikit_learn': sklearn.__version__,
            'joblib': joblib.__version__,
            'platform': platform.platform(),
        },
    }
    meta_path = OUT_DIR / 'meta.json'
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    # 4) 저장물 왕복 검증 — 다시 읽어 같은 확률이 나오는지
    reloaded = joblib.load(clf_path)
    assert np.allclose(reloaded.predict_proba(X[:50])[:, 1], train_proba[:50]), '저장/로드 불일치'

    print(f'\n[save] {clf_path.relative_to(config.PROJECT_ROOT)}  '
          f'({clf_path.stat().st_size / 1024:.0f} KB)')
    print(f'[save] {meta_path.relative_to(config.PROJECT_ROOT)}')
    print('[check] joblib 왕복 후 확률 일치 ✅')


if __name__ == '__main__':
    main()
