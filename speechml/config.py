"""speechml 공통 설정 — 경로 / seed / 필터 기준 / 모델을 한곳에서 관리한다.

여기 있는 값들은 전부 "왜 이 값인지" 근거를 주석으로 남긴다.
실험 조건이 바뀌면 이 파일만 수정하면 전체 파이프라인에 반영된다.
"""
from pathlib import Path

# --- 경로 ---
# __file__ 기준으로 루트를 찾으므로 어느 디렉터리에서 실행해도 동작한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / 'torgo_data' / 'data.csv'
AUDIO_ROOT = PROJECT_ROOT          # data.csv의 filename이 'torgo_data/...' 로 시작
OUT_DIR = PROJECT_ROOT / 'speechml' / 'outputs'
META_CSV = OUT_DIR / 'meta.csv'
EMBED_NPZ = OUT_DIR / 'embed_wav2vec2_base.npz'

# --- 재현성 ---
SEED = 42

# --- 데이터 필터 (5단계 EDA 결론) ---
# 0.3s = 4,800샘플. MFCC 기본 프레임(2,048)이 2개 들어가는 하한.
# 이보다 짧은 8개는 전부 M05의 8~43ms 녹음 사고(M05 중앙값은 3.25s로 오히려 긴 쪽).
# 0.5s 이상으로 올리면 "짧은 발화"라는 구음장애의 실제 증상을 지우게 된다.
MIN_DURATION_SEC = 0.3

# --- split ---
# 화자가 15명뿐이라 단일 hold-out은 test가 2~4명에 의존해 성능 추정 분산이 크다.
# 5-fold면 모든 화자가 정확히 한 번 test에 들어간다.
# LOSO(15-fold)는 fold당 test에 한 클래스만 들어가서 fold별 Recall 계산이 불가능.
N_SPLITS = 5
GROUP_COL = 'speaker_id'       # 화자 누수 차단: 같은 화자는 train/test에 동시 등장 금지
LABEL_COL = 'is_dysarthria'
POSITIVE_LABEL = 'dysarthria'  # Recall을 측정할 양성 클래스 (미탐지 최소화가 최우선)

# --- 모델 ---
W2V_MODEL_ID = 'facebook/wav2vec2-base'
SAMPLE_RATE = 16000            # TORGO 전부 16kHz(EDA 확인) = wav2vec2 요구 입력과 일치

# 로컬 모델 디렉터리.
# transformers 4.56은 torch<2.6 환경에서 .bin(pickle) 가중치 로딩을 거부한다(CVE-2025-32434).
# macOS x86_64는 torch 2.2.2가 상한이고 facebook/wav2vec2-base는 safetensors를 제공하지 않으므로,
# fetch_model.py로 한 번 변환해 여기에 두고 이후에는 로컬에서 로드한다.
W2V_LOCAL_DIR = PROJECT_ROOT / 'speechml' / 'models' / 'wav2vec2-base'

# 레이어별 임베딩 (13개 레이어 × mean·std 풀링).
# 한 번 추출해두면 "어느 레이어가 병리 신호를 가장 잘 담는가" 실험이 공짜가 된다.
EMBED_LAYERS_NPZ = OUT_DIR / 'embed_w2v_layers.npz'

# --- 7단계에서 확정한 모델 설정 ---
# 레이어 8: 13개 레이어 × 3가지 풀링(39조합) 스윕 결과 AUC가 뚜렷한 역U자를 그렸고
#   layer 0=0.63 → layer 8=0.82 → layer 12=0.71. 마지막 레이어는 사전학습 과제
#   (마스킹된 양자화 단위 예측)에 특화되어 병리 신호가 깎여 있다.
W2V_LAYER = 8
# mean 단독: layer8에서 mean+std와 AUC는 동급(0.818 vs 0.820)이지만
#   Recall 0.727>0.710, FN 271<287, fold 분산 0.220<0.247, 차원은 절반.
#   평가 우선순위가 Recall이므로 mean을 택한다. (std 풀링은 하위 레이어에서만 유효했다)
W2V_POOLING = 'mean'
# C는 튜닝하지 않고 기본값 고정. 중첩 교차검증에서 fold별로 고른 C가
#   0.0001~1로 널뛰고 그 결과가 고정값보다 나빴다(AUC 0.619 vs 0.710).
#   내부 fold에 화자가 4~5명뿐이라 안정적 선택이 불가능하다.
LOGREG_C = 1.0
# 결정 임계값. 기본 0.5 대신 0.15를 쓴다.
# 근거: OOF threshold 스윕에서 0.5→0.15로 내리면
#   미탐지(FN) 271 → 221건 (-50), Recall 0.727 → 0.777,
#   대가로 Precision 0.773 → 0.703.
# 뇌졸중 경고 서비스에서 "실제 환자를 정상이라 판단"하는 비용이
# "정상인에게 병원 방문을 권하는" 비용보다 훨씬 크므로 Recall을 택했다.
DECISION_THRESHOLD = 0.15

# --- 8단계: 옛 레시피 RF baseline ---
BACKEND_DIR = PROJECT_ROOT / 'back-end'       # preprocess.py를 그대로 import하기 위한 경로
SLICES_NPZ = OUT_DIR / 'baseline_slices.npz'  # MFCC 슬라이스 + ZCR/RMS 캐시 (1회 계산)
# DTW 참조 개수. fold의 train 정상 화자에서만 뽑는다(누수 방지) → fold마다 DTW 재계산.
# 3개=22분 / 5개=37분 / 10개=75분 중 5개를 택했다.
N_DTW_REFS = 5
# VAD 모드. preprocess.py 주석이 "학습 전처리와 일치 필요 시 energy 권장"이라고 명시한다.
# pyin은 서빙 쪽에만 나중에 추가된 변경(2단계에서 진단한 train/serve skew)이고,
# 실측상 pyin 1,512ms vs energy 3.3ms로 455배 차이가 난다.
BASELINE_VAD_MODE = 'energy'
