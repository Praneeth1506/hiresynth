# HireSynth — candidate ranking pipeline (CPU image)
#
#   docker build -t hiresynth .
#   docker run --rm hiresynth pytest
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/output:/app/output" \
#       hiresynth python rank.py --candidates data/candidates.jsonl --jd jd.docx --out output/submission.csv
#   docker run --rm --env-file .env -v "$PWD/output:/app/output" hiresynth python run_pipeline.py

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

RUN useradd --create-home --uid 1000 app

WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers doesn't pull CUDA wheels.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-dev.txt ./
# spacy==3.7.6 ships no linux/aarch64 wheel (e.g. Apple Silicon), so a compiler
# is needed there; it is removed again in the same layer to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install -r requirements.txt -r requirements-dev.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bake the embedding model into the image so Layers 1–2 need no network at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chown -R app:app "$HF_HOME"

COPY --chown=app:app . .

USER app

CMD ["python", "run_pipeline.py", "--help"]
