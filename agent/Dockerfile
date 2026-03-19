# Build context must be the project root:
#   docker build -f agent/Dockerfile -t ml4fin-agent .
#
# Run (model + data files are mounted, not baked into the image):
#   docker run -p 8000:8000 \
#     -v $(pwd)/modeling:/app/modeling \
#     -v $(pwd)/data:/app/data \
#     --env-file agent/.env \
#     ml4fin-agent

FROM python:3.11-slim

WORKDIR /app

# Install OS-level deps needed by some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent/ ./agent/
COPY modeling/vectorizer.py ./modeling/vectorizer.py

EXPOSE 8000

CMD ["uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
