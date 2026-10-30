import hashlib, sys
import numpy as np
from app.core.config import model_dir, w2v_dir
from app.services.speech.embedder import embedder_from_meta

# 양쪽에서 동일하게 재현되는 입력
rng = np.random.default_rng(42)
audio = (rng.standard_normal(16000 * 3) * 0.1).astype(np.float32)

# 임베딩
emb = embedder_from_meta(model_dir("speech"), w2v_dir())
vec = emb.embed(audio)

# 지문
print("shape : ", vec.shape, vec.dtype)
print("sha256 : ", hashlib.sha256(vec.tobytes()).hexdigest())
print("head : ", np.array2string(vec[:4], precision=12))

# 요청 시 저장
if len(sys.argv) > 1:
    np.save(sys.argv[1], vec)
    
