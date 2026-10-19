# Speech API 엔드포인트
# 오디오 파일을 받아 wav2vec2 임베딩 → 로지스틱 회귀로 구음장애 위험도를 반환한다.
#
# 모델: wav2vec2-base(frozen) layer 8 mean-pooling → StandardScaler + LogisticRegression
# 설정: app/core/config.py 의 settings (모델 경로/버전/임계값은 환경변수로 주입 가능)
#
# ⚠️ 전처리 주의: 학습은 librosa.load(sr=16000) 원본 전체를 그대로 사용했다.
#    VAD / 10초 크롭 / 타일링을 넣으면 학습 입력과 달라져 성능이 무효가 된다.
#    (meta.json 의 preprocessing 블록 참고)

import os
import uuid
import base64
import logging
import subprocess
from pathlib import Path

import numpy as np
import librosa
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.services.speech.preprocess import load_audio_16k
from app.services.speech.embedder import embedder_from_meta
from app.services.speech.predictor import SpeechPredictor
from app.services.speech.viz import build_waveform_png
from app.core.config import model_dir, w2v_dir, settings
from app.db.session import get_db
from app.crud import speech as crud_speech
from app.crud import user as crud_user
from app.core.security import get_user_id_from_cookie

logger = logging.getLogger(__name__)
router = APIRouter()

# settings 에서 유도. 환경변수(SPEECH_MODEL_VERSION 등)로 바뀌며 절대 경로다.
MODEL_DIR = model_dir("speech")
W2V_DIR = w2v_dir()

# 무음 판정 하한. TORGO 학습 데이터의 rms_mean 최솟값이 0.00089 이므로,
# 그보다 조용하면 학습 분포 밖이고 발화로 볼 수 없다. (1% 분위는 0.0062)
MIN_RMS = 0.001

# 발화 여부 게이트 — 스펙트럼 평탄도(median)
#
# 이 모델에는 "발화가 아님" 클래스가 없다. 구음장애/정상 둘 중 하나로만 학습됐으므로
# 학습 분포 밖의 입력(잡음·무음·험)은 억지로 한쪽에 배정되고, 실제로 무음 녹음이
# 0.97로 Abnormal 판정된 사례가 있었다. 모델이 판단한 게 아니라 임베딩이 우연히
# 중증 구음장애 영역에 떨어진 결과였다.
#
# 근거 (TORGO 실제 발화 249개 샘플 측정):
#   실제 발화     0.00067 ~ 0.05817   (1% 0.00133 / 50% 0.01402 / 99% 0.04407)
#   백색잡음      0.5615   ↑          에어컨형 저역잡음  0.5622  ↑
#   완전 무음     1.0000   ↑          60Hz 험/순음/핑크  0.0000  ↓
# 사람 음성은 배음 구조가 있어 스펙트럼이 뾰족하고(평탄도 중간), 광대역 잡음은
# 고르게 퍼져(높음), 순음은 한 곳에 몰린다(낮음). 양쪽을 모두 막는다.
FLATNESS_MAX = 0.15     # 실제 발화 최댓값 0.058 의 약 2.6배 여유
FLATNESS_MIN = 0.0002   # 실제 발화 최솟값 0.00067 의 약 1/3

# ⚠️ 한계: 이 게이트는 "발화처럼 들리지 않는 입력"만 막는다.
# 사람 목소리와 섞인 잡음, 타인의 말소리는 걸러내지 못한다. 근본 해결은
# 학습 단계에서 "발화 아님" 클래스를 두거나 OOD 탐지를 붙여야 한다.


# JSON 직렬화/DB 저장을 위한 안전 변환 (numpy → python 기본형)
def _to_py(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    return obj


# --- 모델 캐시 -------------------------------------------------------------
# 인코더는 380MB다. 요청마다 로드하면 안 되므로 프로세스당 1회만 만든다.
_embedder = None
_predictor = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = embedder_from_meta(Path(MODEL_DIR), Path(W2V_DIR))
    return _embedder


def get_predictor():
    global _predictor
    if _predictor is None:
        # threshold 를 넘기면 환경변수가 meta.json 을 이긴다.
        # 넘기지 않으려면 settings.SPEECH_THRESHOLD 를 Optional 로 바꾸고 None 을 전달한다.
        _predictor = SpeechPredictor(Path(MODEL_DIR), threshold=settings.SPEECH_THRESHOLD)
    return _predictor


def _median_flatness(audio: np.ndarray) -> float:
    """스펙트럼 평탄도의 중앙값. 발화 여부 판별에 쓴다."""
    try:
        f = librosa.feature.spectral_flatness(y=audio)[0]
        return float(np.nanmedian(f)) if len(f) else 1.0
    except Exception:
        return 1.0   # 계산 실패 시 '발화 아님' 쪽으로 보수적 처리


def _save_speech_row(db, user_id, file, data: dict) -> None:
    """DB 저장. 실패해도 응답은 돌려준다."""
    try:
        crud_speech.create(db, {
            "user_id": user_id,
            "wav_filename": getattr(file, "filename", None),
            "model_tag": os.path.basename(MODEL_DIR),
            "dtw_slice1": None, "dtw_slice2": None, "dtw_slice3": None,
            "dtw_slice4": None, "dtw_slice5": None, "dtw_slope": None,
            "x_cover": None, "features_json": None,
            "dtw_graph_url": None, "feature_graph_url": None,
            **data,
        })
    except Exception as e:
        logger.warning("[speech] DB 저장 실패: %s", e)


def _no_speech_response(db, user_id, file, duration, rms, flatness, sr, audio):
    """발화가 감지되지 않은 경우의 응답.

    모델을 돌리지 않는다. 학습 분포 밖의 입력이라 확률에 의미가 없기 때문이다.
    risk 를 null 로 두는 것이 '0.97 이라는 근거 없는 숫자'를 보여주는 것보다 정직하다.
    판정은 Abnormal 로 둔다 — 말을 하지 못하는 상태 자체가 경고 신호이므로.
    """
    logger.warning(
        "[speech] 발화 미감지 | flatness=%.5f (허용 %.4f~%.2f) rms=%.4f dur=%.2fs file=%s",
        flatness, FLATNESS_MIN, FLATNESS_MAX, rms, duration, getattr(file, "filename", None),
    )
    result = {
        "pred": 1,
        "risk": None,                 # 모델을 돌리지 않았다
        "threshold": get_predictor().threshold,
        "decision": "Abnormal",
        "reason": "no_speech_detected",
        "personalized_text": (
            "녹음에서 사람의 말소리가 감지되지 않았습니다.\n\n"
            "말을 하려 해도 소리가 나오지 않는 상태라면 뇌졸중의 경고 신호일 수 있으니 "
            "즉시 주변에 도움을 요청하세요.\n"
            "마이크가 꺼져 있거나 주변 소음만 녹음됐을 수도 있습니다. "
            "마이크 상태를 확인한 뒤 조용한 곳에서 다시 시도해 주세요."
        ),
        "debug_info": {
            "model_version": settings.SPEECH_MODEL_VERSION,
            "audio_duration": duration,
            "rms": rms,
            "flatness": flatness,
            "flatness_range": [FLATNESS_MIN, FLATNESS_MAX],
            "gate": "no_speech_detected",
            "filename": getattr(file, "filename", None),
        },
    }
    try:
        g = build_waveform_png(y=audio, meta={"sr": sr},
                               extras={"audio_length": len(audio), "audio_duration": duration})
        result["graph"] = f"data:image/png;base64,{g}" if g else None
    except Exception:
        result["graph"] = None

    _save_speech_row(db, user_id, file, {
        "risk_score": None,
        "threshold": result["threshold"],
        "result_text": "Abnormal",
        "voiced_sec": duration,
        "debug_json": _to_py(result["debug_info"]),
        "waveform_graph_url": None,
    })
    return JSONResponse(_to_py(result))


def check_ffmpeg() -> bool:
    """FFmpeg 설치 확인. webm 녹음을 디코딩할 때 필요하다."""
    try:
        return subprocess.run(["ffmpeg", "-version"],
                              capture_output=True, text=True).returncode == 0
    except FileNotFoundError:
        return False


# --- 점검용 엔드포인트 (인증 없음) ------------------------------------------

@router.get("/system-check")
async def check_system():
    """운영 중 가장 먼저 확인하는 엔드포인트. 모델 파일과 외부 도구 존재 여부."""
    import platform
    w2v = Path(W2V_DIR)
    return {
        "system": platform.system(),
        "ffmpeg_installed": check_ffmpeg(),
        "model_dir": MODEL_DIR,
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "model_version": settings.SPEECH_MODEL_VERSION,
        "model_files": {
            "clf": os.path.exists(os.path.join(MODEL_DIR, "clf.joblib")),
            "meta": os.path.exists(os.path.join(MODEL_DIR, "meta.json")),
        },
        "w2v_dir": W2V_DIR,
        "w2v_files": {
            "weights": (w2v / "model.safetensors").exists(),
            "config": (w2v / "config.json").exists(),
            "preprocessor": (w2v / "preprocessor_config.json").exists(),
        },
    }


@router.get("/model-info")
async def get_model_info():
    """meta.json 의 서빙 계약과 기대 성능을 노출한다."""
    try:
        p = get_predictor()
        meta = p.meta
        return {
            "model_name": meta.get("model_name"),
            "created_at": meta.get("created_at"),
            "embedding": meta.get("embedding"),
            "preprocessing": meta.get("preprocessing"),
            "decision": {**meta.get("decision", {}), "threshold_in_use": p.threshold},
            "expected_performance": meta.get("expected_performance"),
        }
    except Exception as e:
        logger.exception("[speech] model-info 실패")
        return {"error": str(e)}


# --- 예측 -----------------------------------------------------------------

@router.post("/predict")
async def predict_speech(
    file: UploadFile = File(...),
    debug: bool = Form(False),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_user_id_from_cookie),
):
    try:
        # 1) 파일 수신
        b = await file.read()
        if not b:
            return JSONResponse(status_code=400,
                                content={"error": "업로드된 오디오 파일이 비어 있습니다."})

        # 2) 디코딩 (webm/mp3/m4a/ogg/wav) → 16kHz mono float32
        #    전처리가 아니라 디코딩이다. 학습 때도 16kHz 원본을 그대로 썼다.
        y_16k = load_audio_16k(b, filename=file.filename)
        sr = 16000
        duration = len(y_16k) / sr

        # 3) 입력 검증
        #    예전에는 무음일 때 확률을 0.65로 끌어올렸지만(SILENCE_VETO),
        #    그건 모델 출력을 왜곡하는 것이다. 입력 단계에서 정직하게 거부한다.
        meta = get_predictor().meta
        min_sec = meta.get("preprocessing", {}).get("min_duration_sec", 0.3)
        if duration < min_sec:
            return JSONResponse(status_code=400, content={
                "error": f"녹음이 너무 짧습니다 ({duration:.2f}초). 조금 더 길게 말씀해 주세요."})

        rms = float(np.sqrt(np.mean(np.square(y_16k)))) if len(y_16k) else 0.0
        if rms < MIN_RMS:
            return JSONResponse(status_code=400, content={
                "error": "소리가 거의 녹음되지 않았습니다. 마이크를 확인하고 다시 시도해 주세요."})

        # 발화 여부 게이트.
        # 거부(400)가 아니라 "발화 없음"을 근거로 한 Abnormal 을 반환한다.
        # 뇌졸중으로 말을 하지 못하는 상태라면 그 자체가 경고 신호이기 때문이다.
        # 다만 모델 확률을 조작하지 않는다 — risk 는 null 로 두고 판정 근거를 명시한다.
        # (옛 SILENCE_VETO 는 확률을 0.65 로 끌어올려 모델 출력을 왜곡했다.)
        flatness = _median_flatness(y_16k)
        if flatness > FLATNESS_MAX or flatness < FLATNESS_MIN:
            return _no_speech_response(
                db=db, user_id=user_id, file=file,
                duration=duration, rms=rms, flatness=flatness, sr=sr, audio=y_16k,
            )

        # 4) 임베딩 → 예측
        vec = get_embedder().embed(y_16k)
        out = get_predictor().predict(vec)

        result = {
            "pred": out["pred"],
            "risk": out["risk"],
            "threshold": out["threshold"],
            "decision": out["decision"],
            "debug_info": {
                "model_dir": MODEL_DIR,
                "model_version": settings.SPEECH_MODEL_VERSION,
                "layer": get_embedder().layer,
                "pooling": get_embedder().pooling,
                "embed_dim": int(vec.shape[0]),
                "audio_duration": duration,
                "rms": rms,
                "filename": getattr(file, "filename", None),
            },
        }

        # 5) 파형 그림 (DTW 그래프는 새 모델에 해당 feature 가 없어 제거)
        wav_b64 = None
        try:
            wav_b64 = build_waveform_png(
                y=y_16k, meta={"sr": sr},
                extras={"audio_length": len(y_16k), "audio_duration": duration},
            )
        except Exception as e:
            logger.warning("[speech] 파형 그림 생성 실패: %s", e)
        result["graph"] = f"data:image/png;base64,{wav_b64}" if wav_b64 else None

        # 6) 사용자 안내 문구
        try:
            u = crud_user.get_user_by_id(db, user_id) if user_id else None
            name_txt = getattr(u, "name", None) or "사용자"
        except Exception:
            name_txt = "사용자"
        result["personalized_text"] = (
            "언어 장애 평가는 사전학습 음성 모델(wav2vec2)이 발화에서 추출한 표현을 바탕으로 "
            "정상 발화와의 차이를 학습한 분류기로 위험도를 산출합니다.\n\n"
            f"{name_txt}님의 위험도는 {out['risk']:.2f}로 기준치 {out['threshold']:.2f} "
            f"{'미만' if out['pred'] == 0 else '이상'}으로 위험이 "
            f"{'없' if out['pred'] == 0 else '있'}다고 판단됩니다."
        )
        result["debug_info"]["personalized_text"] = result["personalized_text"]

        # 7) DB 저장 (실패해도 응답은 돌려준다)
        try:
            app_root = Path(__file__).resolve().parents[3]          # back-end/app
            static_dir = app_root / "static" / "speech"
            static_dir.mkdir(parents=True, exist_ok=True)

            wav_name = None
            if wav_b64:
                wav_name = f"wave_{uuid.uuid4().hex}.png"
                (static_dir / wav_name).write_bytes(base64.b64decode(wav_b64))

            crud_speech.create(db, {
                "user_id": user_id,
                "wav_filename": getattr(file, "filename", None),
                "model_tag": os.path.basename(MODEL_DIR),           # "v2"
                "risk_score": out["risk"],
                "threshold": out["threshold"],
                "result_text": out["decision"],
                # DTW 계열 컬럼은 새 모델에 해당 feature 가 없다. 스키마는 A단계에서
                # 건드리지 않고 비워둔다(nullable). 정리는 B단계에서.
                "dtw_slice1": None, "dtw_slice2": None, "dtw_slice3": None,
                "dtw_slice4": None, "dtw_slice5": None, "dtw_slope": None,
                "x_cover": None,
                "voiced_sec": duration,                              # VAD 미사용 → 전체 길이
                "features_json": None,                               # 768차원은 저장하지 않는다
                "debug_json": _to_py(result["debug_info"]),
                "waveform_graph_url": (f"/static/speech/{wav_name}" if wav_name else None),
                "dtw_graph_url": None,
                "feature_graph_url": None,
            })
        except Exception as e:
            logger.warning("[speech] DB 저장 실패: %s", e)

        return JSONResponse(_to_py(result))

    except Exception as e:
        logger.exception("[speech] 예측 실패")
        raise HTTPException(status_code=500, detail=f"Speech analysis failed: {str(e)}")


@router.get("/list")
def list_speech_results(
    limit: int = 50,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_user_id_from_cookie),
):
    rows = crud_speech.list_by_user(db, user_id=user_id, limit=limit)
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "result": r.result_text,
            "risk": r.risk_score,
            "threshold": r.threshold,
            "wav_filename": r.wav_filename,
            "model_tag": r.model_tag,
        } for r in rows
    ]
