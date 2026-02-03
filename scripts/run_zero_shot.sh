#!/bin/bash
# ============================================
# Run zero-shot evaluation for all models
# ============================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Zero-shot Evaluation Pipeline ==="
echo "Working directory: $PROJECT_DIR"

# Run for each model × negative set combination
python -m scripts.run_experiment \
    --mode zero_shot \
    --models dnabert2 nucleotide_transformer_500m hyenadna caduceus evo1 \
    --negative-sets N1 N3 \
    --context-size 1024 \
    --output-dir results/zero_shot

echo "=== Zero-shot evaluation complete ==="
