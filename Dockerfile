# Dockerfile
FROM python:3.11-slim

# System deps for FastAI/Pillow/OpenCV-style image ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Spaces sets $PORT. Uvicorn must listen on it.
ENV PORT=7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
