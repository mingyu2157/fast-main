"""wav2vec2 임베딩 추출 — 발화 1개 → 768차원 벡터 1개.

인코더는 freeze(추론만 수행)한다. 무거운 forward를 한 번만 돌려 결과를 저장하고,
이후 분류기 실험에서는 이 npz 파일만 읽는다. 이게 CPU/MPS 환경에서
full fine-tuning 대신 선택한 전략의 핵심이다.

실행:
  python -m speechml.embed --limit 10          # 시간 측정 (저장하지 않음)
  python -m speechml.embed --device cpu --limit 10
  python -m speechml.embed                     # 전량 추출 후 저장
"""
from __future__ import annotations

import argparse
import time

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from speechml import config, dataset


def pick_device(requested: str) -> torch.device:
    """auto면 MPS를 우선 시도하고, 없으면 CPU."""
    if requested != 'auto':
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(device: torch.device):
    """feature extractor + 인코더를 로드하고 freeze한다."""
    if not (config.W2V_LOCAL_DIR / 'model.safetensors').exists():
        raise FileNotFoundError(
            f'{config.W2V_LOCAL_DIR} 에 변환된 가중치가 없습니다. '
            '먼저 `python -m speechml.fetch_model` 을 실행하세요.'
        )
    fe = AutoFeatureExtractor.from_pretrained(config.W2V_LOCAL_DIR)
    model = Wav2Vec2Model.from_pretrained(config.W2V_LOCAL_DIR)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return fe, model


@torch.no_grad()
def embed_all_layers(path, fe, model, device: torch.device) -> np.ndarray:
    """오디오 1개 → (n_layers, 1536). 레이어마다 [mean | std] 를 이어붙인다.

    std를 함께 담는 이유: 5단계 EDA에서 "구음장애의 신호는 평균이 아니라 분산"
    (길이 std 2.88 vs 1.30)이라는 관찰이 나왔다. 시간축 표준편차는 그 직관을
    임베딩 수준에서 재현해보는 feature다.
    """
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE)
    inputs = fe(y, sampling_rate=config.SAMPLE_RATE, return_tensors='pt')
    out = model(inputs.input_values.to(device), output_hidden_states=True)
    vecs = []
    for h in out.hidden_states:                 # (1, T', 768) × (레이어+1)개
        m = h.mean(dim=1).squeeze(0)
        sd = h.std(dim=1).squeeze(0)
        vecs.append(torch.cat([m, sd]).float().cpu().numpy())
    return np.stack(vecs)                       # (n_layers, 1536)


@torch.no_grad()
def embed_one(path, fe, model, device: torch.device) -> np.ndarray:
    """오디오 1개 → 768차원 벡터.

    배치 크기를 1로 고정한 이유: 발화 길이가 0.3~25초로 가변인데
    facebook/wav2vec2-base는 attention mask 없이 학습된 모델이라
    패딩을 섞으면 mean pooling에 패딩 구간이 끼어든다. 1개씩 돌리면 그 문제가 없다.
    """
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE)
    inputs = fe(y, sampling_rate=config.SAMPLE_RATE, return_tensors='pt')
    hidden = model(inputs.input_values.to(device)).last_hidden_state  # (1, T', 768)
    return hidden.mean(dim=1).squeeze(0).float().cpu().numpy()        # 시간축 평균


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='앞 N개만 처리하고 저장하지 않는다 (시간 측정용)')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps'])
    ap.add_argument('--all-layers', action='store_true',
                    help='13개 레이어 × mean·std 풀링을 모두 저장한다')
    args = ap.parse_args()

    d, y, _ = dataset.build()
    total_all = len(d)
    if args.limit:
        d = d.head(args.limit)
        y = y[:args.limit]

    device = pick_device(args.device)
    print(f'\n[embed] device={device} | model={config.W2V_MODEL_ID} | 대상 {len(d)}개')

    fe, model = load_model(device)

    extract = embed_all_layers if args.all_layers else embed_one
    if args.all_layers:
        print('  모드: 전체 레이어 × [mean|std]')

    vecs: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i, rel in enumerate(d['rel_path'], 1):
        vecs.append(extract(config.AUDIO_ROOT / rel, fe, model, device))
        if i % 100 == 0 or i == len(d):
            el = time.perf_counter() - t0
            print(f'  {i}/{len(d)}  경과 {el:6.1f}s  ({el / i:.3f}s/건)')

    X = np.stack(vecs)
    per = (time.perf_counter() - t0) / len(d)
    print(f'[embed] X shape={X.shape}  평균 {per:.3f}s/건')

    if args.limit:
        print(f'[추정] 전량 {total_all}개 → {per * total_all / 60:.1f}분 (device={device})')
        print('[embed] --limit 지정이므로 저장하지 않음')
        return

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EMBED_LAYERS_NPZ if args.all_layers else config.EMBED_NPZ
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        rel_path=d['rel_path'].to_numpy(),
        speaker_id=d[config.GROUP_COL].to_numpy(),
    )
    print(f'[embed] 저장 → {out_path}')


if __name__ == '__main__':
    main()
