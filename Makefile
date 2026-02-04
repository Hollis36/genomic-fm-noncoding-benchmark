.PHONY: help install test lint format clean data experiments all

help:
	@echo "Genomic Foundation Model Benchmark - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-fast       Run tests (skip slow/gpu/model tests)"
	@echo "  make test-cov        Run tests with coverage report"
	@echo "  make test-unit       Run unit tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run all linters"
	@echo "  make format          Format code with black and isort"
	@echo "  make type-check      Run mypy type checking"
	@echo ""
	@echo "Data:"
	@echo "  make download-data   Download ClinVar and reference genome"
	@echo "  make prepare-data    Prepare all datasets"
	@echo ""
	@echo "Experiments:"
	@echo "  make zero-shot       Run zero-shot evaluation"
	@echo "  make finetune        Run LoRA fine-tuning"
	@echo "  make all-experiments Run full benchmark pipeline"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove build artifacts and cache"
	@echo "  make clean-data      Remove processed data (keep raw)"
	@echo "  make clean-results   Remove experiment results"

# === Setup ===

install:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Environment created/updated. Activate with: conda activate genomic-fm-benchmark"

install-dev: install
	pip install pytest pytest-cov pytest-xdist black isort ruff mypy
	@echo "Development dependencies installed"

# === Testing ===

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow and not gpu and not model" -n auto

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-unit:
	pytest tests/unit/ -v -n auto

# === Code Quality ===

lint:
	@echo "Running ruff..."
	ruff check src/ tests/ data/scripts/
	@echo "Running black (check)..."
	black --check src/ tests/ data/scripts/
	@echo "Running isort (check)..."
	isort --check-only src/ tests/ data/scripts/

format:
	@echo "Formatting with black..."
	black src/ tests/ data/scripts/
	@echo "Sorting imports with isort..."
	isort src/ tests/ data/scripts/
	@echo "Done!"

type-check:
	mypy src/ --ignore-missing-imports

# === Data Preparation ===

download-data:
	bash data/scripts/01_download_clinvar.sh

prepare-data:
	@echo "Step 1: Filtering ClinVar..."
	python data/scripts/02_filter_noncoding.py
	@echo "Step 2: Building negative sets..."
	python data/scripts/03_build_negative_sets.py \
		--positive data/processed/positive_noncoding.tsv \
		--ref-fasta data/reference/GRCh38.fa
	@echo "Step 3: Annotating regions..."
	python data/scripts/04_annotate_regions.py \
		--variants data/processed/positive_noncoding.tsv \
		           data/processed/negative_N1_benign.tsv \
		           data/processed/negative_N3_matched_random.tsv
	@echo "Step 4: Extracting sequences..."
	python data/scripts/05_extract_sequences.py \
		--variants data/processed/positive_noncoding_annotated.tsv \
		--context-size 1024
	@echo "Data preparation complete!"

# === Experiments ===

zero-shot:
	python -m scripts.run_experiment \
		--mode zero_shot \
		--models dnabert2 nucleotide_transformer_500m hyenadna caduceus \
		--negative-sets N1 N3 \
		--output-dir results/zero_shot

finetune:
	python -m scripts.run_experiment \
		--mode lora \
		--models dnabert2 nucleotide_transformer_500m \
		--negative-sets N3 \
		--epochs 10 \
		--output-dir results/lora

all-experiments:
	bash scripts/run_all.sh

# === Cleanup ===

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage .cache/
	@echo "Cleaned build artifacts and cache"

clean-data:
	rm -rf data/processed/*
	@echo "Removed processed data files"

clean-results:
	rm -rf results/*
	@echo "Removed experiment results"

# === Documentation ===

docs:
	@echo "Building documentation..."
	mkdocs build
	@echo "Documentation built in site/"

docs-serve:
	mkdocs serve

# === Quick Commands ===

quick-test: format test-fast
	@echo "Quick test complete!"

ci: lint test-fast
	@echo "CI checks passed!"

all: install prepare-data all-experiments
	@echo "Full pipeline complete!"
