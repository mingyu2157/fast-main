# backend Dockerfile
# --------builder stage ----------
FROM python:3.12-slim AS builder

# 일부 패키지(fastdtw 등)는 sdist 라 컴파일러가 필요하다.
# 이 스테이지에만 두고 최종 이미지에는 넣지 않는다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 가상환경에 설치하면 "결과물 한 덩어리"를 그대로 옮길 수 있다
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt

# torch 를 CPU 전용 인덱스에서 먼저 설치한다.
# PyPI 기본 torch 는 CUDA 빌드라 nvidia-*/triton 으로 4.3GB 가 따라온다.
# 이 서비스는 CPU 추론(측정: CPU 0.267s vs MPS 1.271s)이므로 전부 사용되지 않는다.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2

# torch 가 이미 충족돼 pip 이 재설치하지 않는다
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# --------runtime stage ----------
FROM python:3.12-slim AS runtime

# 실행에만 필요한 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# wav2vec2 인코더(363MB). C단계에서 MLflow/MinIO 로 분리할 예정.
COPY speechml/models/wav2vec2-base /models/wav2vec2-base
ENV SPEECH_W2V_DIR=/models/wav2vec2-base

COPY back-end/ /app/

# 비권한 사용자로 실행
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# 워커는 1개. 모델이 워커마다 808MB 올라가므로 수평 확장은 replica 로 한다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]