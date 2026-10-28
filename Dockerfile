# ============================================================
# Dockerfile — API FastAPI + MLflow (Atelier 6)
# Image multi-stage pour réduire la taille finale
# ============================================================

# ---------- Stage 1 : Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Copier uniquement requirements.txt d'abord (cache Docker)
COPY requirements.txt .

# Installer les dépendances dans un environnement virtuel isolé
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2 : Runtime ----------
FROM python:3.12-slim

WORKDIR /app

# Copier le venv depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier les fichiers nécessaires UNIQUEMENT (pas venv, pas mlruns, pas tests)
COPY app.py .
COPY model_pipeline.py .
COPY classifier.joblib .

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
