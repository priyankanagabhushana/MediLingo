FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    TOKENIZERS_PARALLELISM=false

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY scripts ./scripts
COPY ui ./ui
COPY config ./config
COPY artifacts/rag ./artifacts/rag
COPY artifacts/rag_100k ./artifacts/rag_100k
COPY artifacts/examples.json ./artifacts/examples.json
COPY models/qwen3-4b-medical-lora ./models/qwen3-4b-medical-lora
COPY models/qwen3-4b-medical-lora-100k ./models/qwen3-4b-medical-lora-100k
COPY models/gemma4-e2b-medical-lora ./models/gemma4-e2b-medical-lora

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/_stcore/health', timeout=4)"

EXPOSE 8080

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8080"]
