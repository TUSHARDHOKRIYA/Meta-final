FROM python:3.11-slim

WORKDIR /app

# Install only essential system deps (git needed by some pip packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──
# Copy requirements first for Docker layer caching
COPY backend/requirements.txt ./requirements.txt

# Install PyTorch CPU wheel (pre-built, no compilation) + all deps in one step
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ──
COPY backend/ ./backend/
COPY inference.py .
COPY gym_wrapper.py .
COPY train.py .
COPY evaluate.py .
COPY ablation.py .
COPY hrl_train.py .
COPY tasks/ ./tasks/
COPY dashboard/ ./dashboard/
COPY openenv.yaml .
COPY README.md .
COPY server/ ./server/

# Create dirs for models and results
RUN mkdir -p ./models ./results ./backend/data/trajectory

# Copy pre-trained models if they exist
COPY models/ ./models/

# Required env vars (set as HF Space Secrets at runtime)
ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.3-70b-versatile"
ENV HF_TOKEN=""
ENV GROQ_API_KEY=""
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""
ENV PYTHONPATH="/app/backend"

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run from /app so both backend and inference.py are accessible
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
