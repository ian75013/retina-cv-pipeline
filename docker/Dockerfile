# =============================================================================
# RetinAI — Image Docker de production
# Multi-stage build pour une image minimale
# =============================================================================

# -- Stage 1 : Build --
FROM python:3.10-slim as builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -- Stage 2 : Production --
FROM python:3.10-slim

LABEL maintainer="RetinAI Team"
LABEL description="API d'inférence pour la détection de rétinopathie diabétique"

# Dépendances système minimales (OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances Python depuis le builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copie du code source
COPY src/ src/
COPY configs/ configs/

# Copie du modèle optimisé (monté en volume en production)
# COPY models/optimized/ models/optimized/

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/optimized/model_dynamic.tflite
ENV LOG_LEVEL=INFO

# Port de l'API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Démarrage de l'API FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
