# Genomic Foundation Model Benchmark for Non-coding Variant Pathogenicity Prediction

Systematic benchmarking of genomic foundation models on non-coding variant pathogenicity prediction, evaluating zero-shot performance, linear probe, and LoRA fine-tuning across multiple negative set strategies and non-coding region categories.

## Key Features

- **6 genomic foundation models** evaluated in a unified framework: DNABERT-2, Nucleotide Transformer v2 (500M & 2.5B), HyenaDNA, Caduceus, and Evo-1
- **3 negative set strategies** to expose benchmark inflation: ClinVar Benign, gnomAD Common (MAF>5%), Position-matched Random
- **7 non-coding region categories**: splice-proximal, 5'UTR, 3'UTR, promoter, enhancer, deep intronic, intergenic
- **3 evaluation modes**: zero-shot scoring, linear probe on frozen embeddings, LoRA fine-tuning
- All experiments reproducible on a **single RTX 4090 (24GB)**

## Project Structure

```
├── configs/
│   ├── models.yaml              # Model configurations and hyperparameters
│   └── datasets.yaml            # Dataset sources and filtering criteria
├── data/
│   ├── scripts/
│   │   ├── 01_download_clinvar.sh    # Download ClinVar, reference genome, annotations
│   │   ├── 02_filter_noncoding.py    # Filter non-coding pathogenic/benign variants
│   │   ├── 03_build_negative_sets.py # Build N2 (gnomAD) and N3 (matched random) sets
│   │   ├── 04_annotate_regions.py    # Classify variants by non-coding region type
│   │   └── 05_extract_sequences.py   # Extract ref/alt DNA sequences with context
│   └── processed/                    # Generated datasets (not tracked by git)
├── src/
│   ├── models/
│   │   ├── base.py                   # Abstract base class for all models
│   │   ├── dnabert2.py               # DNABERT-2 (117M, encoder)
│   │   ├── nucleotide_transformer.py # NT-v2 (500M/2.5B, encoder)
│   │   ├── hyenadna.py              # HyenaDNA (causal, sub-quadratic)
│   │   ├── caduceus.py              # Caduceus (bidirectional Mamba)
│   │   └── evo.py                   # Evo-1 (7B, 4-bit quantized)
│   ├── evaluation/
│   │   ├── zero_shot.py             # Zero-shot variant scoring
│   │   ├── linear_probe.py          # Linear probe (logistic regression on embeddings)
│   │   ├── lora_finetune.py         # LoRA fine-tuning with K-fold CV
│   │   └── metrics.py              # AUROC, AUPRC, MCC, DeLong test, Bootstrap CI
│   └── visualization/
│       └── plots.py                 # Publication-quality figures
├── scripts/
│   ├── run_experiment.py            # Main experiment runner
│   ├── run_zero_shot.sh             # Run zero-shot evaluation
│   ├── run_finetune.sh              # Run LoRA fine-tuning
│   └── run_all.sh                   # Full pipeline
├── results/                          # Output directory for results and figures
├── environment.yml                   # Conda environment specification
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate genomic-fm-benchmark
```

### 2. Download Data

```bash
bash data/scripts/01_download_clinvar.sh
```

### 3. Prepare Datasets

```bash
python data/scripts/02_filter_noncoding.py
python data/scripts/03_build_negative_sets.py
python data/scripts/04_annotate_regions.py \
    --variants data/processed/positive_noncoding.tsv \
               data/processed/negative_N1_benign.tsv \
               data/processed/negative_N3_matched_random.tsv
python data/scripts/05_extract_sequences.py \
    --variants data/processed/positive_noncoding_annotated.tsv \
    --context-size 1024
```

### 4. Run Experiments

```bash
# Zero-shot evaluation (all models)
python -m scripts.run_experiment \
    --mode zero_shot \
    --models dnabert2 nucleotide_transformer_500m hyenadna caduceus evo1 \
    --negative-sets N1 N3

# LoRA fine-tuning
python -m scripts.run_experiment \
    --mode lora \
    --models dnabert2 nucleotide_transformer_500m \
    --negative-sets N3 \
    --epochs 10
```

Or run the full pipeline:

```bash
bash scripts/run_all.sh
```

## Models

| Model | Parameters | Type | VRAM (Inference) | Zero-shot Method |
|-------|-----------|------|-----------------|-----------------|
| DNABERT-2 | 117M | Encoder (BPE) | ~2 GB | Embedding cosine distance |
| NT-v2-500M | 500M | Encoder (6-mer) | ~3 GB | Embedding cosine distance |
| NT-v2-2.5B | 2.5B | Encoder (6-mer) | ~6 GB | Embedding cosine distance |
| HyenaDNA | ~1.6B | Causal (character) | ~4 GB | Log-likelihood ratio |
| Caduceus | ~200M | Bidirectional Mamba | ~3 GB | Embedding cosine distance |
| Evo-1 | 7B | Causal (character) | ~10 GB (4-bit) | Log-likelihood ratio |

## Evaluation Metrics

- **AUROC**: Area Under Receiver Operating Characteristic curve
- **AUPRC**: Area Under Precision-Recall curve (robust to class imbalance)
- **MCC**: Matthews Correlation Coefficient
- **F1**: F1 score at optimal (Youden's J) threshold
- **DeLong test**: Statistical comparison of AUROC between models
- **Bootstrap CI**: 95% confidence intervals via 1000 bootstrap samples

## Data Sources

| Source | Description | URL |
|--------|------------|-----|
| ClinVar | Variant-phenotype associations | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/ |
| gnomAD v4 | Population allele frequencies | https://gnomad.broadinstitute.org/ |
| ENCODE cCREs | Candidate cis-regulatory elements | https://screen.encodeproject.org/ |
| GENCODE v44 | Gene annotation | https://www.gencodegenes.org/ |
| GRCh38 | Reference genome | https://www.ensembl.org/ |

## Hardware Requirements

- **Minimum**: 1x NVIDIA RTX 4090 (24GB VRAM)
- **Recommended**: 1x NVIDIA A100 (80GB) for running all models without quantization
- **Storage**: ~50 GB for reference genome + annotations; ~5 GB for processed datasets

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{genomic_fm_noncoding_benchmark,
  title={Systematic Benchmarking of Genomic Foundation Models for Non-coding Variant Pathogenicity Prediction},
  year={2025},
}
```

## License

MIT License
