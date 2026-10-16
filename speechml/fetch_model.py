"""facebook/wav2vec2-base 가중치를 로컬 safetensors로 변환한다.

왜 필요한가:
  transformers 4.56은 torch<2.6에서 .bin(pickle) 로딩을 거부한다 (CVE-2025-32434).
  macOS x86_64는 torch 2.2.2가 상한이라 업그레이드가 불가능하고,
  facebook/wav2vec2-base 저장소에는 safetensors가 없다.
  → 공식 저장소에서 한 번만 내려받아 safetensors로 변환해두고 로컬에서 로드한다.
  transformers를 구버전으로 내려 보안 가드를 끄는 대신 이 방법을 택했다.

실행:  python -m speechml.fetch_model   (1회성, 약 380MB 다운로드)
"""
from __future__ import annotations

import shutil

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from speechml import config

CONFIG_FILES = ['config.json', 'preprocessor_config.json']


def main() -> None:
    dst = config.W2V_LOCAL_DIR
    target = dst / 'model.safetensors'
    if target.exists():
        print(f'[fetch] 이미 변환되어 있음 → {target}')
        return

    dst.mkdir(parents=True, exist_ok=True)

    for name in CONFIG_FILES:
        src = hf_hub_download(config.W2V_MODEL_ID, name)
        shutil.copy(src, dst / name)
        print(f'[fetch] {name} 복사')

    bin_path = hf_hub_download(config.W2V_MODEL_ID, 'pytorch_model.bin')
    print('[fetch] 가중치 로드 중 (weights_only=True)...')
    state = torch.load(bin_path, map_location='cpu', weights_only=True)

    # safetensors는 저장소를 공유하는 텐서를 허용하지 않으므로 복사해 분리한다.
    tensors = {
        k: v.contiguous().clone()
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }
    save_file(tensors, target)
    size_mb = target.stat().st_size / 1e6
    print(f'[fetch] 변환 완료 → {target}  ({len(tensors)} 텐서, {size_mb:.0f}MB)')


if __name__ == '__main__':
    main()
