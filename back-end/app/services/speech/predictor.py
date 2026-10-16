# 분류기 - 768차원 임베딩을 받아 위험도와 판정을 냄
#
# threshold는 코드에 박지 않고 meta.json에서 읽는다.
# 기본값 0.5가 아니라 0.15다 (미탐지를 줄이기 위해 낮춤).

from __future__ import annotations
import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# 프론트(SpeechResult.jsx)와 DB(speech.result_text)가 기대하는 문자열.
# 바꾸면 프론트가 판정 색을 초록으로 표시하고 사용자에게 그대로 노출된다.
LABEL_ABNORMAL = 'Abnormal'
LABEL_NORMAL = 'Normal'


class SpeechPredictor:
    def __init__(self, model_dir: Path, threshold: float | None = None):
        """clf.joblib과 meta.json을 로드한다.

        threshold 우선순위: 인자(환경변수 override) > meta.json > 코드 기본값.
        meta.json은 모델과 함께 다니는 진실이고, 인자는 운영자의 조정 수단이다.
        어느 쪽이 적용됐는지 로그에 남겨 나중에 추적할 수 있게 한다.
        """
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.clf = joblib.load(model_dir / 'clf.joblib')

        meta = json.loads((model_dir / 'meta.json').read_text(encoding='utf-8'))
        self.meta = meta
        meta_threshold = meta.get('decision', {}).get('threshold')

        if threshold is not None:
            self.threshold = float(threshold)
            source = 'override(인자/환경변수)'
        elif meta_threshold is not None:
            self.threshold = float(meta_threshold)
            source = 'meta.json'
        else:
            self.threshold = 0.5
            source = '코드 기본값'

        logger.info(
            '[speech] 모델 로드 %s | threshold=%.4f (출처: %s)',
            model_dir.name, self.threshold, source,
        )

    def predict(self, vec: np.ndarray) -> dict:
        """(768,) 벡터 → {"pred", "risk", "decision", "threshold"}

        clf.predict()를 쓰지 않는 이유: sklearn의 predict()는 threshold가
        0.5로 고정되어 있다. 우리는 0.15를 써야 하므로 확률을 받아 직접 비교한다.
        """
        # sklearn은 (샘플수, 특성수) 2차원을 요구한다
        vec_2d = np.asarray(vec).reshape(1, -1)

        # [:, 1] 이 양성(구음장애) 확률. 인덱스를 뒤집으면 1-p 가 되는데
        # 에러는 나지 않으므로 조용히 틀린다. meta.json 의 positive_index 참고.
        risk = float(self.clf.predict_proba(vec_2d)[0][1])

        pred = int(risk >= self.threshold)
        return {
            'pred': pred,
            'risk': risk,
            'decision': LABEL_ABNORMAL if pred else LABEL_NORMAL,
            'threshold': self.threshold,
        }


def predictor_from_meta(model_dir: Path, threshold: float | None = None) -> SpeechPredictor:
    """embedder_from_meta 와 짝을 이루는 생성 함수."""
    return SpeechPredictor(model_dir, threshold=threshold)
