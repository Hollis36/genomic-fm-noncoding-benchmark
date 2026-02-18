"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_dna_sequence():
    """Generate a sample DNA sequence for testing."""
    return "ATCGATCGATCGATCGATCG" * 10


@pytest.fixture
def sample_variant_df():
    """Generate a sample variant DataFrame for testing."""
    return pd.DataFrame({
        "chrom": ["chr1", "chr2", "chr3", "chr1", "chr2"],
        "pos": [1000, 2000, 3000, 1500, 2500],
        "ref": ["A", "C", "G", "T", "A"],
        "alt": ["T", "G", "A", "C", "G"],
        "label": [1, 1, 0, 0, 1],
        "ref_seq": ["ATCG" * 50] * 5,
        "alt_seq": ["TTCG" * 50] * 5,
        "region_category": ["promoter", "enhancer", "intergenic", "utr_5prime", "splice_proximal"],
    })


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(100, 768).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample binary labels for testing."""
    np.random.seed(42)
    return np.random.binomial(2, 0.5, size=100)


@pytest.fixture
def sample_scores():
    """Generate sample prediction scores for testing."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def mock_model_config():
    """Generate a mock model configuration."""
    return {
        "name": "test-model",
        "type": "encoder",
        "params": "100M",
        "embedding_dim": 768,
        "max_length": 512,
        "max_bp": 3000,
        "tokenizer_type": "BPE",
        "quantize": None,
        "pooling": "mean",
        "zero_shot_method": "embedding_distance",
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["query", "value"],
        },
        "estimated_vram_gb": {
            "inference": 2,
            "lora_finetune": 5,
        },
    }


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
