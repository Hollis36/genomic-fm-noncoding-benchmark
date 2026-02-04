# Usage Guide

This guide covers common usage patterns for the genomic foundation model benchmark.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Running Experiments](#running-experiments)
4. [Analyzing Results](#analyzing-results)
5. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/genomic-fm-benchmark.git
cd genomic-fm-benchmark

# Create conda environment
conda env create -f environment.yml
conda activate genomic-fm-benchmark
```

### 2. Download Data

```bash
# Download ClinVar and reference genome
bash data/scripts/01_download_clinvar.sh
```

### 3. Prepare Datasets

```bash
# Filter non-coding variants
python data/scripts/02_filter_noncoding.py

# Build negative sets
python data/scripts/03_build_negative_sets.py \
    --positive data/processed/positive_noncoding.tsv \
    --ref-fasta data/reference/GRCh38.fa

# Annotate region categories
python data/scripts/04_annotate_regions.py \
    --variants data/processed/positive_noncoding.tsv \
               data/processed/negative_N1_benign.tsv \
               data/processed/negative_N3_matched_random.tsv

# Extract DNA sequences
python data/scripts/05_extract_sequences.py \
    --variants data/processed/positive_noncoding_annotated.tsv \
    --context-size 1024
```

### 4. Run a Simple Experiment

```bash
# Zero-shot evaluation with DNABERT-2
python -m scripts.run_experiment \
    --mode zero_shot \
    --models dnabert2 \
    --negative-sets N1 N3
```

## Data Preparation

### Filtering Variants

The `02_filter_noncoding.py` script filters ClinVar variants:

```bash
python data/scripts/02_filter_noncoding.py \
    --vcf data/raw/clinvar.vcf.gz \
    --output-dir data/processed
```

**Filters applied**:
- Non-coding molecular consequences only
- Pathogenic/Likely_pathogenic for positive set
- Benign/Likely_benign for negative set (N1)
- Review status ≥1 star

### Building Negative Sets

Three negative set strategies:

**N1: ClinVar Benign** (easiest baseline)
```bash
# Automatically created by 02_filter_noncoding.py
```

**N2: gnomAD Common** (intermediate difficulty)
```bash
python data/scripts/03_build_negative_sets.py \
    --positive data/processed/positive_noncoding.tsv \
    --gnomad-vcf data/raw/gnomad_v4_chr*.vcf.gz \
    --min-maf 0.05
```

**N3: Position-Matched Random** (hardest baseline)
```bash
python data/scripts/03_build_negative_sets.py \
    --positive data/processed/positive_noncoding.tsv \
    --ref-fasta data/reference/GRCh38.fa
```

### Annotating Regions

Classify variants by non-coding region type:

```bash
python data/scripts/04_annotate_regions.py \
    --variants data/processed/*.tsv \
    --encode-ccres data/reference/GRCh38-cCREs.bed \
    --gencode data/reference/gencode.v44.annotation.gtf
```

**Region categories**:
- `splice_proximal`: Within 20bp of splice site
- `utr_5prime`: 5' UTR
- `utr_3prime`: 3' UTR
- `promoter`: TSS ±2kb
- `enhancer`: ENCODE cCRE distal enhancer-like signature
- `deep_intronic`: >50bp from exon boundary
- `intergenic`: Outside gene bodies and cCREs

## Running Experiments

### Zero-Shot Evaluation

Evaluate models without fine-tuning:

```bash
python -m scripts.run_experiment \
    --mode zero_shot \
    --models dnabert2 nucleotide_transformer_500m hyenadna \
    --negative-sets N1 N3 \
    --context-size 1024 \
    --output-dir results/zero_shot
```

**Scoring methods**:
- Encoder models (DNABERT-2, NT-v2, Caduceus): Cosine distance between ref/alt embeddings
- Causal models (HyenaDNA, Evo-1): Log-likelihood ratio

### Linear Probe

Train a logistic regression on frozen embeddings:

```bash
python -m scripts.run_experiment \
    --mode linear_probe \
    --models dnabert2 \
    --negative-sets N3 \
    --output-dir results/linear_probe
```

**Process**:
1. Extract embeddings for all sequences
2. 5-fold stratified cross-validation
3. Train logistic regression (balanced class weights)
4. Evaluate on held-out folds

### LoRA Fine-Tuning

Parameter-efficient fine-tuning:

```bash
python -m scripts.run_experiment \
    --mode lora \
    --models dnabert2 nucleotide_transformer_500m \
    --negative-sets N3 \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-4 \
    --output-dir results/lora
```

**LoRA parameters** (configurable in `configs/models.yaml`):
- Rank `r`: 8
- Alpha: 16
- Dropout: 0.05
- Target modules: Model-specific (see config)

### Running All Experiments

Full pipeline:

```bash
bash scripts/run_all.sh
```

This runs:
1. Data preparation
2. Zero-shot evaluation (all models)
3. LoRA fine-tuning (selected models)
4. Figure generation

## Analyzing Results

### Viewing Results

Results are saved as JSON files in the output directory:

```bash
ls results/
# zeroshot_DNABERT-2-117M_N1.json
# zeroshot_DNABERT-2-117M_N3.json
# lora_DNABERT-2-117M_N3.json
# scores_DNABERT-2-117M_N3.parquet  # Detailed scores
```

### Using Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

**Available notebooks**:
- `01_data_exploration.ipynb`: Explore variant datasets
- `02_results_analysis.ipynb`: Analyze and compare model performance

### Generating Figures

```python
from src.visualization.plots import *

# Load results
results = load_results('results/')

# Generate figures
plot_auroc_heatmap(results)
plot_region_bars(results)
plot_zeroshot_vs_finetune(results)
plot_pareto_front(results)
```

### Command-Line Summary

```bash
# Quick summary of results
python -c "
import json
from pathlib import Path

for path in Path('results').glob('*.json'):
    with open(path) as f:
        data = json.load(f)
        print(f\"{data['model']} | {data['method']} | {data['negative_set']}\")
        print(f\"  AUROC: {data['overall']['auroc']:.4f}\")
        print()
"
```

## Advanced Usage

### Custom Models

Add your own model:

```python
# src/models/my_model.py
from .base import GenomicModelBase

class MyModel(GenomicModelBase):
    def load(self):
        # Load your model
        pass

    def get_embedding(self, sequence: str):
        # Return embedding
        pass

    def score_variant(self, ref_seq: str, alt_seq: str):
        # Return variant score
        pass
```

Register in `src/models/registry.py`:

```python
from .my_model import MyModel

MODEL_REGISTRY["my_model"] = MyModel
```

Add configuration to `configs/models.yaml`:

```yaml
my_model:
  name: "huggingface/my-model"
  type: encoder
  embedding_dim: 1024
  max_length: 512
  # ...
```

### Baseline Comparisons

Compare against conservation scores:

```python
from src.baselines import CADDScorer, PhyloPScorer

# Load baseline method
cadd = CADDScorer(data_path="data/baselines/CADD_v1.7.bw")

# Score variants
import pandas as pd
df = pd.read_csv("data/processed/positive_noncoding_annotated.tsv", sep="\t")
scores = cadd.score_variants(df)

# Evaluate
from src.evaluation.metrics import compute_all_metrics
metrics = compute_all_metrics(df["label"], scores)
print(f"CADD AUROC: {metrics['auroc']:.4f}")
```

### Experiment Tracking

Use the experiment tracker:

```python
from src.utils.progress import ExperimentTracker

with ExperimentTracker("my_experiment", output_dir="experiments/") as tracker:
    tracker.log_config({"model": "dnabert2", "lr": 1e-4})

    for epoch in range(10):
        # Training
        loss = train_epoch()
        tracker.log_metric("train_loss", loss, step=epoch)

        # Validation
        metrics = validate()
        tracker.log_metrics(metrics, step=epoch)

    tracker.log_status("completed")
```

### Caching Embeddings

Enable caching for faster repeated evaluations:

```python
from src.utils.caching import EmbeddingCache

cache = EmbeddingCache(cache_dir=".cache/embeddings", max_size_gb=10.0)

# Check cache before computing
embedding = cache.get(model_name, sequence)
if embedding is None:
    embedding = model.get_embedding(sequence)
    cache.set(model_name, sequence, embedding)
```

### Statistical Comparisons

Compare models statistically:

```python
from src.utils.statistical_analysis import bootstrap_auroc_comparison

result = bootstrap_auroc_comparison(
    y_true=labels,
    scores_a=model_a_scores,
    scores_b=model_b_scores,
    n_bootstrap=1000
)

print(f"Model A AUROC: {result['auroc_a_mean']:.4f}")
print(f"Model B AUROC: {result['auroc_b_mean']:.4f}")
print(f"Difference: {result['auroc_diff_mean']:.4f} "
      f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
print(f"Significant: {result['significant']}")
```

## Troubleshooting

### Out of Memory

**Problem**: CUDA out of memory during inference

**Solutions**:
1. Reduce batch size: `--batch-size 8`
2. Use 4-bit quantization (Evo-1): Set `quantize: "4bit"` in config
3. Use gradient checkpointing for LoRA training
4. Process sequences in smaller chunks

### Slow Inference

**Problem**: Inference is too slow

**Solutions**:
1. Enable embedding caching (see above)
2. Use smaller models (DNABERT-2, NT-v2-500M)
3. Reduce context size: `--context-size 512`
4. Use GPU if available
5. Parallelize across multiple GPUs

### Missing Data Files

**Problem**: `FileNotFoundError` for data files

**Solutions**:
1. Run data preparation scripts in order (01 → 05)
2. Check file paths in error message
3. Verify downloads completed successfully
4. Check `data/processed/` directory

### Model Download Failures

**Problem**: HuggingFace model download fails

**Solutions**:
1. Check internet connection
2. Set HuggingFace cache: `export HF_HOME=/path/to/cache`
3. Use HuggingFace CLI: `huggingface-cli login`
4. Download manually and set local path in config

## Performance Tips

1. **Use GPUs**: Essential for acceptable inference speed
2. **Cache embeddings**: Reuse embeddings across experiments
3. **Batch processing**: Process multiple sequences together
4. **Context size**: Use smallest context that captures the variant effect
5. **Model selection**: Start with smaller models for prototyping

## Citation

If you use this benchmark, please cite:

```bibtex
@article{genomic_fm_noncoding_benchmark,
  title={Systematic Benchmarking of Genomic Foundation Models for Non-coding Variant Pathogenicity Prediction},
  year={2025},
}
```
