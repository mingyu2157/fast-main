# back-end/app/api/v1/routers.py
from fastapi import APIRouter
from app.api.v1.endpoints import auth, measure, face  # face 모듈 쓰는 경우 포함


api_router = APIRouter()

# /api/v1/auth/...
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# /api/v1/measure/...
api_router.include_router(measure.router, prefix="/measure", tags=["measure"])

# (선택) /api/v1/face/...
api_router.include_router(face.router, prefix="/face", tags=["face"])
