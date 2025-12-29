FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

# ASR + API + upload
RUN pip install --no-cache-dir \
      faster-whisper==1.2.1 \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.32.1" \
      python-multipart==0.0.20

WORKDIR /app
COPY app /app

RUN mkdir -p /work/in /work/out /hf

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
