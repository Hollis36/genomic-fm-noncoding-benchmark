# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial project structure and codebase
- Support for 6 genomic foundation models:
  - DNABERT-2 (117M)
  - Nucleotide Transformer v2 (500M / 2.5B)
  - HyenaDNA (1.6B)
  - Caduceus (~200M)
  - Evo-1 (7B)
- Three evaluation modes:
  - Zero-shot variant scoring
  - Linear probe on frozen embeddings
  - LoRA parameter-efficient fine-tuning
- Three negative set strategies:
  - N1: ClinVar Benign
  - N2: gnomAD Common (MAF > 5%)
  - N3: Position-matched Random
- Seven non-coding region categories:
  - Splice-proximal
  - 5' UTR
  - 3' UTR
  - Promoter
  - Enhancer
  - Deep intronic
  - Intergenic
- Comprehensive evaluation metrics:
  - AUROC with DeLong test
  - AUPRC
  - MCC
  - F1 score
  - Bootstrap confidence intervals
- Data processing pipeline:
  - ClinVar filtering
  - Negative set generation
  - Region annotation
  - Sequence extraction
- Model registry and factory pattern for extensibility
- Logging infrastructure with colored console output
- Progress tracking and experiment monitoring
- Embedding caching system
- Statistical analysis utilities:
  - Paired t-test
  - Wilcoxon test
  - McNemar's test
  - Effect size calculation
  - Bootstrap AUROC comparison
- Baseline methods:
  - CADD
  - PhyloP
  - PhastCons
- Comprehensive testing infrastructure:
  - Unit tests
  - Integration tests
  - Test fixtures and mocks
  - 80%+ code coverage requirement
- CI/CD pipeline with GitHub Actions:
  - Automated testing
  - Code linting and formatting
  - Security scanning
- Jupyter notebooks:
  - Data exploration
  - Results analysis
- Publication-quality visualization:
  - AUROC heatmaps
  - Per-region bar charts
  - Zero-shot vs fine-tuned comparison
  - Pareto efficiency frontier
  - ROC curves
- Comprehensive documentation:
  - Usage guide
  - Model descriptions
  - Contributing guidelines
  - API documentation

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- Added bandit security scanning in CI pipeline
- Input validation for all user-provided data
- Safe file handling with pathlib

## [0.1.0] - 2025-XX-XX

### Added

- Initial release
- Core benchmarking functionality
- Support for 6 genomic foundation models
- Three evaluation modes
- Comprehensive documentation

[Unreleased]: https://github.com/yourusername/genomic-fm-benchmark/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/genomic-fm-benchmark/releases/tag/v0.1.0
