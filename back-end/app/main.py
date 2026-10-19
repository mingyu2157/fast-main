# back-end/app/main.py
from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routers import api_router
from app.core.config import settings
from app.db.base import Base


# .env 로드
load_dotenv()

# --- 로깅 설정 -------------------------------------------------------------
# uvicorn 은 uvicorn / uvicorn.access 로거만 구성하고 root 는 건드리지 않는다.
# root 기본 레벨이 WARNING 이라, 설정하지 않으면 애플리케이션의 logger.info() 가
# 전부 버려진다(모델 로드 정보, threshold 출처 등이 안 보임).
# 운영에서 파드 로그가 1차 디버깅 수단이므로 여기서 root 를 명시적으로 구성한다.
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info(
    "FAST API 기동 | speech 모델 %s | LOG_LEVEL=%s",
    settings.SPEECH_MODEL_VERSION, settings.LOG_LEVEL,
)

app = FastAPI(
    title="FAST API",
    description="뇌졸중 조기 진단 시스템 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# API 라우터 등록
app.include_router(api_router, prefix="/api/v1")

# 데이터베이스 테이블 생성
# Base.metadata.create_all(bind=engine)  # 이 부분 제거 또는 수정


@app.get("/")
async def root():
    return {"message": "FAST API Server is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
