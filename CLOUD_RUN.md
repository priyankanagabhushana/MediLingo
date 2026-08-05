# Cloud Run handoff

This project is packaged for a later Cloud Run deployment. No GCP project, billing
account, region, or credentials were supplied, so deployment is intentionally not
performed by the Thor run.

## Local validation

From the project root:

    docker build --progress=plain -t medilingo:local .
    docker run --rm -p 8080:8080 medilingo:local

The Streamlit health endpoint is:

    http://localhost:8080/_stcore/health

The current local image contains the 50k and expanded 100k Qwen LoRA adapters plus the Gemma adapter under
/app/models/qwen3-4b-medical-lora, /app/models/qwen3-4b-medical-lora-100k, and /app/models/gemma4-e2b-medical-lora. It also includes the 50k and 100k local retrieval indexes. It does
not contain the base model weights; the first model request may download them into
the container cache unless a serving image or mounted model storage is prepared.

## Deployment outline

Use a project-specific Artifact Registry repository and replace the placeholders:

    gcloud auth configure-docker REGION-docker.pkg.dev
    docker tag medilingo:local REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/medilingo:TAG
    docker push REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/medilingo:TAG

    gcloud run deploy medilingo       --image REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/medilingo:TAG       --region REGION       --port 8080       --set-env-vars RAG_INDEX_DIR=/app/artifacts/rag

Provide HF_TOKEN through Secret Manager if the container must access a gated
checkpoint. Never put a token in this repository or an image layer.

A 4-billion-parameter model may need a GPU-capable Cloud Run configuration or a
different serving platform. Measure cold-start time, memory, and latency first.
The project-local Streamlit app is a demonstration UI; a production service should
add authentication, request limits, audit logging, data retention controls, and
human review.

The application is an administrative translation prototype. Keep human review,
data governance, and healthcare governance requirements in the deployment
decision.
