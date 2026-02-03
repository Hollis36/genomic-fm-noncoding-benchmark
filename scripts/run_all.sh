#!/bin/bash
# ============================================
# Full pipeline: data preparation → evaluation → visualization
# ============================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================"
echo "  Genomic FM Non-coding Variant Benchmark"
echo "============================================"
echo ""

# --- Phase 1: Data Preparation ---
echo ">>> Phase 1: Data Preparation"
bash data/scripts/01_download_clinvar.sh
python data/scripts/02_filter_noncoding.py
python data/scripts/03_build_negative_sets.py \
    --positive data/processed/positive_noncoding.tsv \
    --ref-fasta data/reference/GRCh38.fa

echo ">>> Annotating regions..."
python data/scripts/04_annotate_regions.py \
    --variants \
        data/processed/positive_noncoding.tsv \
        data/processed/negative_N1_benign.tsv \
        data/processed/negative_N3_matched_random.tsv

echo ">>> Extracting sequences..."
python data/scripts/05_extract_sequences.py \
    --variants data/processed/positive_noncoding_annotated.tsv \
    --context-size 1024

echo ""

# --- Phase 2: Zero-shot Evaluation ---
echo ">>> Phase 2: Zero-shot Evaluation"
bash scripts/run_zero_shot.sh

echo ""

# --- Phase 3: LoRA Fine-tuning ---
echo ">>> Phase 3: LoRA Fine-tuning"
bash scripts/run_finetune.sh

echo ""

# --- Phase 4: Generate Figures ---
echo ">>> Phase 4: Generating Figures"
python -c "
from src.visualization import *
from src.visualization.plots import load_results

results = load_results('results/zero_shot') + load_results('results/lora')
plot_auroc_heatmap(results)
plot_region_bars(results)
plot_zeroshot_vs_finetune(results)
plot_pareto_front(results)
print('All figures generated.')
"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in: results/"
echo "============================================"
