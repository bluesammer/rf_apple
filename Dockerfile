FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_VISIBLE_DEVICES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade \
    pip==24.2 \
    setuptools==69.5.1 \
    wheel==0.44.0

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75"]
