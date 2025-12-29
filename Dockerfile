ARG CUDA_BASE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
FROM ${CUDA_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

# PyTorch (CUDA) for translation on GPU
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=
RUN if [ -n "$TORCH_VERSION" ]; then \
      pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}; \
    else \
      pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch; \
    fi

# ASR + API + upload
RUN pip install --no-cache-dir \
      faster-whisper==1.2.1 \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.32.1" \
      python-multipart==0.0.20 \
      transformers==4.46.2 \
      sentencepiece==0.2.0

WORKDIR /app
COPY app /app

RUN mkdir -p /work/in /work/out /hf

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
