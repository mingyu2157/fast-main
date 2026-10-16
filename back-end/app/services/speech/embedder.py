# wav2vec2 인코더 - 오디오를 768차원 벡터로 변환
#
# 학습 코드(speechml/embed.py)와 완전히 같은 경로여야 한다.
# 한 군데라도 다르면 모델이 학습 때 본 적 없는 입력을 받게 되고,
# 교차검증에서 측정한 AUC 0.818은 의미를 잃는다.

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model


class SpeechEmbedder:
    def __init__(self, model_dir: Path, layer: int, pooling: str = 'mean',
                 sample_rate: int = 16000, embed_dim: int | None = None):
        # 인코더를 1회 로드해서 재사용 (380MB라 요청마다 로드하면 안 됨)
        self.model_dir = model_dir
        self.layer = layer
        self.pooling = pooling
        self.sample_rate = sample_rate
        self.embed_dim = embed_dim

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        self.model = Wav2Vec2Model.from_pretrained(self.model_dir)
        self.model.eval()
        # embed()에 @torch.no_grad()가 걸려 있어 기능상으론 중복이지만,
        # 학습 코드(speechml/embed.py)와 상태를 맞춰둔다.
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """16kHz mono float32 오디오를 입력받아 768차원 벡터를 반환.

        배치는 1로 고정한다. facebook/wav2vec2-base는 attention mask 없이
        학습된 모델이라, 길이가 다른 발화를 패딩해 묶으면 패딩 구간이
        시간축 평균에 섞여 들어간다.
        """
        # feature_extractor로 입력 텐서 만들기
        # padding 인자는 넘기지 않는다 - 단일 입력에선 무의미하고, 학습 때도 없었다
        inputs = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        # 중간 레이어를 받으려면 output_hidden_states=True가 필요하다
        outputs = self.model(inputs.input_values, output_hidden_states=True)

        # hidden_states는 12층 모델인데 13개가 들어 있다.
        #   [0]  트랜스포머 이전 (CNN feature → projection 출력)
        #   [1]~[12]  각 트랜스포머 층,  [12] == last_hidden_state
        # layer 8을 쓰는 이유: 39조합 스윕에서 AUC가 역U자를 그렸고
        # last_hidden_state(0.710)보다 layer 8(0.818)이 뚜렷하게 높았다.
        layer_output = outputs.hidden_states[self.layer]   # (1, 프레임, 768)

        # 시간축(dim=1) 풀링
        if self.pooling == 'mean':
            pooled = layer_output.mean(dim=1)
        elif self.pooling == 'max':
            pooled, _ = layer_output.max(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        vec = pooled.squeeze(0).float().cpu().numpy()       # (768,)

        # 레이어나 풀링을 잘못 지정하면 여기서 바로 걸린다
        if self.embed_dim is not None and vec.shape[0] != self.embed_dim:
            raise ValueError(
                f"임베딩 차원 불일치: {vec.shape[0]} != meta.json의 {self.embed_dim}"
            )
        return vec


def embedder_from_meta(model_dir: Path, w2v_dir: Path) -> SpeechEmbedder:
    """meta.json의 embedding 블록을 읽어 SpeechEmbedder를 생성.

    layer / pooling / sample_rate를 코드에 하드코딩하지 않고 모델과 함께
    다니는 meta.json에서 읽는다 - 모델을 교체하면 설정도 따라온다.

    model_dir: clf.joblib과 meta.json이 있는 디렉터리 (예: .../speech/v2)
    w2v_dir:   wav2vec2 가중치 디렉터리 (환경변수로 주입)
    """
    meta = json.loads((Path(model_dir) / 'meta.json').read_text(encoding='utf-8'))
    cfg = meta['embedding']
    return SpeechEmbedder(
        model_dir=Path(w2v_dir),
        layer=cfg['layer'],
        pooling=cfg['pooling'],
        sample_rate=cfg['sample_rate'],
        embed_dim=cfg.get('embed_dim'),
    )
