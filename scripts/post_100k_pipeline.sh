#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/home/sreenath/research-space/SovereignMedTranslate"
cd "$PROJECT_ROOT"
LOG="$PROJECT_ROOT/logs/post_100k_pipeline.log"
TRAIN_PID_FILE="$PROJECT_ROOT/logs/qwen3_training_100k.pid"
ADAPTER="$PROJECT_ROOT/models/qwen3-4b-medical-lora-100k"
INDEX="$PROJECT_ROOT/artifacts/rag_100k"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

log "post-100k pipeline started"
if [[ ! -f "$TRAIN_PID_FILE" ]]; then
  log "training PID file missing; stopping"
  exit 1
fi

train_pid="$(<"$TRAIN_PID_FILE")"
while kill -0 "$train_pid" 2>/dev/null; do
  log "waiting for Qwen 100k training PID $train_pid"
  sleep 60
done
log "training process exited"

for _ in $(seq 1 60); do
  if [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]]; then
    break
  fi
  log "waiting for adapter files to finish flushing"
  sleep 10
done
if [[ ! -f "$ADAPTER/adapter_config.json" || ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  log "adapter files are missing after training"
  exit 1
fi

TEST_JSON="$PROJECT_ROOT/artifacts/qwen3_100k_gated400_test_comparison.json"
EMEA_JSON="$PROJECT_ROOT/artifacts/qwen3_100k_gated400_emea_comparison.json"
if [[ ! -f "$TEST_JSON" ]]; then
  log "starting 100k test evaluation"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/evaluate.py \
    --model qwen3 --condition base,sft,rag --dataset test \
    --max-examples 400 --max-new-tokens 256 \
    --run-name qwen3_100k_gated400 \
    --adapter-dir "$ADAPTER" --index-dir "$INDEX" \
    --similarity-threshold 0.65 \
    > logs/qwen3_100k_gated400_test.log 2>&1
else
  log "100k test JSON already exists; keeping it"
fi

if [[ ! -f "$EMEA_JSON" ]]; then
  log "starting 100k EMEA evaluation"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/evaluate.py \
    --model qwen3 --condition base,sft,rag --dataset emea \
    --max-examples 400 --max-new-tokens 256 \
    --run-name qwen3_100k_gated400 \
    --adapter-dir "$ADAPTER" --index-dir "$INDEX" \
    --similarity-threshold 0.65 \
    > logs/qwen3_100k_gated400_emea.log 2>&1
else
  log "100k EMEA JSON already exists; keeping it"
fi

log "building data-scaling comparisons"
.venv/bin/python scripts/compare_scaling.py \
  --baseline artifacts/qwen3_50k_gated400_test_comparison.json \
  --expanded artifacts/qwen3_100k_gated400_test_comparison.json \
  --output-json artifacts/qwen3_data_scaling_test.json \
  --output-md reports/qwen3_data_scaling_test.md \
  > logs/qwen3_data_scaling_test.log 2>&1
.venv/bin/python scripts/compare_scaling.py \
  --baseline artifacts/qwen3_50k_gated400_emea_comparison.json \
  --expanded artifacts/qwen3_100k_gated400_emea_comparison.json \
  --output-json artifacts/qwen3_data_scaling_emea.json \
  --output-md reports/qwen3_data_scaling_emea.md \
  > logs/qwen3_data_scaling_emea.log 2>&1

log "restarting project UI on port 8511"
if [[ -f logs/streamlit.pid ]]; then
  old_pid="$(<logs/streamlit.pid)"
  old_args="$(ps -o args= -p "$old_pid" 2>/dev/null || true)"
  if [[ "$old_args" == *"streamlit run ui/app.py"* && "$old_args" == *"--server.port 8511"* ]]; then
    kill "$old_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$old_pid" 2>/dev/null; then break; fi
      sleep 1
    done
  fi
fi
nohup .venv/bin/python -m streamlit run ui/app.py \
  --server.address 127.0.0.1 --server.port 8511 \
  > logs/streamlit_medilingo.log 2>&1 < /dev/null &
echo $! > logs/streamlit.pid
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8511/_stcore/health >/dev/null 2>&1; then
    log "project UI health check passed"
    break
  fi
  sleep 2
done

if command -v docker >/dev/null 2>&1; then
  log "building medilingo:local image"
  if docker build --progress=plain -t medilingo:local . > logs/docker_build_medilingo.log 2>&1; then
    log "Docker build passed"
    test_name="medilingo-test-100k"
    if docker ps -a --format '{{.Names}}' | grep -Fxq "$test_name"; then
      log "Docker test container name already exists; skipped container test"
    else
      container_id="$(docker run --rm --name "$test_name" -d -p 18084:8080 medilingo:local)"
      sleep 15
      if curl -fsS http://127.0.0.1:18084/_stcore/health > logs/docker_health_100k.txt 2>&1; then
        log "Docker health check passed"
      else
        log "Docker health check failed"
      fi
      docker stop "$container_id" >/dev/null 2>&1 || true
    fi
  else
    log "Docker build failed; see logs/docker_build_medilingo.log"
  fi
else
  log "Docker command unavailable; skipped image build"
fi

.venv/bin/python - <<'PY2'
import json
from pathlib import Path
root = Path('/home/sreenath/research-space/SovereignMedTranslate')
status = {
    'status': 'complete',
    'adapter': str(root / 'models/qwen3-4b-medical-lora-100k'),
    'test_comparison': str(root / 'artifacts/qwen3_100k_gated400_test_comparison.json'),
    'emea_comparison': str(root / 'artifacts/qwen3_100k_gated400_emea_comparison.json'),
    'test_scaling_report': str(root / 'reports/qwen3_data_scaling_test.md'),
    'emea_scaling_report': str(root / 'reports/qwen3_data_scaling_emea.md'),
    'ui_health': 'http://127.0.0.1:8511/_stcore/health',
}
(root / 'artifacts/post_100k_pipeline_status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
PY2
log "post-100k pipeline complete"
