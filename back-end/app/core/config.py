# back-end/app/core/config.py

from typing import Literal
from pydantic import BaseSettings, Field

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]   # back-end/
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # fast-main/

# 모달리티 타입 (face | arm | speech)
Modality = Literal["face", "arm", "speech"]


class Settings(BaseSettings):
    # === DB (회원/인증 등 기존 기능에서 사용) ===
    DB_USER: str = Field("fastteamjpsm", env="DB_USER")
    DB_PASSWORD: str = Field("fast!jpsm", env="DB_PASSWORD")
    DB_HOST: str = Field("localhost", env="DB_HOST")
    DB_PORT: int = Field(3306, env="DB_PORT")
    DB_NAME: str = Field("fast_db", env="DB_NAME")

    # JWT/보안
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # === 모델/추론 관련 (WebTest 기능 이식) ===
    # 모델 파일 기본 디렉토리: app/assets/models/{modality}/{version}/(TabNet_final.zip, scaler.save)
    MODEL_DIR_BASE: str = Field("app/assets/models", env="MODEL_DIR_BASE")

    # 모달리티별 현재 사용 버전 (버저닝 운영 시 .env에서 v2 등으로 교체)
    FACE_MODEL_VERSION: str = Field("v1", env="FACE_MODEL_VERSION")
    ARM_MODEL_VERSION: str = Field("v1", env="ARM_MODEL_VERSION")
    SPEECH_MODEL_VERSION: str = Field("v2", env="SPEECH_MODEL_VERSION")

    # 모달리티별 임계치 (TabNet 확률 → 라벨 변환)
    FACE_THRESHOLD: float = Field(0.2278, env="FACE_THRESHOLD")
    ARM_THRESHOLD: float = Field(0.5, env="ARM_THRESHOLD")
    SPEECH_THRESHOLD: float = Field(0.15, env="SPEECH_THRESHOLD")
    SPEECH_W2V_DIR: str = Field("speechml/models/wav2vec2-base", env="SPEECH_W2V_DIR")


    # 애플리케이션 로그 레벨. uvicorn 은 자기 로거만 설정하므로
    # root 로거가 WARNING 으로 남아 app 의 logger.info() 가 전부 버려진다.
    # main.py 에서 이 값으로 root 를 설정한다. 운영에서는 WARNING/ERROR 로 올린다.
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # 개발 편의를 위한 CORS 기본값(프론트 로컬)
    FRONTEND_ORIGIN: str = Field("http://localhost:5173", env="FRONTEND_ORIGIN")

    class Config:
        env_file = BACKEND_DIR / ".env"
        case_sensitive = True

    # SQLAlchemy 연결 URL (pymysql)
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


# === 헬퍼 함수들 ===
def model_dir(modality: Modality) -> str:
    """
    모달리티/버전 조합으로 실제 모델 디렉토리 경로 반환.
    예: app/assets/models/face/v1
    """
    ver = {
        "face": settings.FACE_MODEL_VERSION,
        "arm": settings.ARM_MODEL_VERSION,
        "speech": settings.SPEECH_MODEL_VERSION,
    }[modality]
    return str(BACKEND_DIR / settings.MODEL_DIR_BASE / modality / ver)

def threshold(modality: Modality) -> float:
    """
    모달리티별 임계치 반환.
    """
    return {
        "face": settings.FACE_THRESHOLD,
        "arm": settings.ARM_THRESHOLD,
        "speech": settings.SPEECH_THRESHOLD,
    }[modality]

def w2v_dir() -> str:
    """
    wav2vec2 가중치 디렉터리. 프로젝트 루트 기준"""
    return str(PROJECT_ROOT / settings.SPEECH_W2V_DIR)

# 싱글톤 설정 인스턴스
settings = Settings()
