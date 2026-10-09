# back-end/app/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routers import api_router
from app.api.v1.endpoints.arm_predict import router as arm_predict_router
from app.core.config import settings
# from app.db.base import Base
# from app.db.session import engine, ping_db


def _resolve_cors_origins() -> list[str]:
    """
    CORS_ORIGINS > FRONTEND_ORIGIN/FRONTEND_URL > 기본 dev 오리진들
    """
    origins: list[str] = []

    cors = getattr(settings, "CORS_ORIGINS", None)
    if cors:
        if isinstance(cors, str):
            origins += [o.strip() for o in cors.split(",") if o.strip()]
        elif isinstance(cors, (list, tuple)):
            origins += [str(o).strip() for o in cors if str(o).strip()]

    single = getattr(settings, "FRONTEND_ORIGIN", None) or getattr(settings, "FRONTEND_URL", None)
    if single:
        origins.append(str(single).strip())

    for d in ["http://localhost:5173", "http://127.0.0.1:5173"]:
        if d not in origins:
            origins.append(d)

    # dedup
    out: list[str] = []
    for o in origins:
        if o and o not in out:
            out.append(o)
    return out


def create_app() -> FastAPI:
    app = FastAPI(title="FAST 프로젝트 백엔드")

    # FastAPI/Starlette가 슬래시 자동 리다이렉트 하도록 명시
    app.router.redirect_slashes = True  # 기본값 True지만, 명시해 두면 안전

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # /api/v1/* 엔드포인트
    app.include_router(api_router, prefix="/api/v1")
    # /api/v1/arm/* (모듈 내부 prefix 가정)
    app.include_router(arm_predict_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # 라우트 디버깅용 핑 (선택)
    @app.get("/api/v1/ping")
    def ping():
        return {"pong": True}

    return app


app = create_app()
