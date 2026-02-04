# Contributing to Genomic Foundation Model Benchmark

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/genomic-fm-benchmark.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment: `conda env create -f environment.yml`
5. Activate the environment: `conda activate genomic-fm-benchmark`

## Development Workflow

### 1. Code Style

We follow PEP 8 style guidelines with the following tools:

- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Linting
- **mypy**: Type checking (optional but recommended)

Run formatters before committing:

```bash
black src/ tests/ data/scripts/
isort src/ tests/ data/scripts/
ruff check src/ tests/ data/scripts/
```

### 2. Testing

All new features must include tests. We use pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_metrics.py

# Run tests with markers
pytest -m "not slow and not gpu"
```

Test markers:
- `slow`: Tests that take >5 seconds
- `gpu`: Tests that require GPU
- `integration`: Integration tests
- `model`: Tests that download models

### 3. Adding a New Model

To add a new genomic foundation model:

1. Create a new file in `src/models/` (e.g., `your_model.py`)
2. Implement the `GenomicModelBase` interface:

```python
from .base import GenomicModelBase

class YourModel(GenomicModelBase):
    def load(self):
        # Load model and tokenizer
        pass

    def get_embedding(self, sequence: str) -> np.ndarray:
        # Return embedding vector
        pass

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        # Return variant effect score
        pass
```

3. Register the model in `src/models/registry.py`:

```python
from .your_model import YourModel

MODEL_REGISTRY["your_model"] = YourModel
```

4. Add model configuration to `configs/models.yaml`
5. Write tests in `tests/unit/test_your_model.py`
6. Update documentation

### 4. Adding a New Evaluation Method

To add a new evaluation method:

1. Create a new file in `src/evaluation/` (e.g., `your_method.py`)
2. Implement the evaluation logic
3. Add tests in `tests/unit/test_your_method.py`
4. Update `scripts/run_experiment.py` to support the new method
5. Document the method in `docs/`

### 5. Adding Baseline Methods

To add a new baseline method:

1. Implement `BaselineMethod` interface in `src/baselines/`
2. Add data download instructions in `docs/DATA.md`
3. Write tests
4. Update documentation

## Commit Guidelines

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or modifying tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add support for Nucleotide Transformer v3
fix: correct AUROC calculation for imbalanced datasets
docs: update installation instructions
test: add unit tests for linear probe evaluation
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass: `pytest`
4. Run linters: `black`, `isort`, `ruff`
5. Update CHANGELOG.md (if applicable)
6. Create a pull request with a clear description
7. Link relevant issues
8. Wait for review and address feedback

## Code Review

All submissions require code review. We review for:

- Code quality and style
- Test coverage (aim for >80%)
- Documentation completeness
- Performance considerations
- Security implications

## Documentation

- Update `README.md` for user-facing changes
- Add docstrings to all functions and classes (Google style)
- Create tutorials/examples for major features
- Update configuration documentation in `docs/`

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
