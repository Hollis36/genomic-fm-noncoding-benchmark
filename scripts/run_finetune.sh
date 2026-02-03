#!/bin/bash
# ============================================
# Run LoRA fine-tuning evaluation for all models
# ============================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== LoRA Fine-tuning Pipeline ==="

python -m scripts.run_experiment \
    --mode lora \
    --models dnabert2 nucleotide_transformer_500m hyenadna caduceus \
    --negative-sets N1 N3 \
    --context-size 1024 \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-4 \
    --output-dir results/lora

echo "=== LoRA fine-tuning complete ==="
