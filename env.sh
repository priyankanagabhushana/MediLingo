#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT=/home/sreenath/research-space/SovereignMedTranslate
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export HF_DATASETS_CACHE=$PROJECT_ROOT/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=$PROJECT_ROOT/.cache/huggingface/transformers
export TORCH_HOME=$PROJECT_ROOT/.cache/torch
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
