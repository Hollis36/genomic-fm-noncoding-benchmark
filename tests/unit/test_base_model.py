"""Unit tests for base model class."""

import numpy as np
import pytest
import torch

from src.models.base import GenomicModelBase


class MockGenomicModel(GenomicModelBase):
    """Mock implementation of GenomicModelBase for testing."""

    def load(self):
        """Mock load method."""
        self.model = torch.nn.Linear(10, self.config.get("embedding_dim", 768))
        self.tokenizer = "mock_tokenizer"
        return self

    def get_embedding(self, sequence: str) -> np.ndarray:
        """Mock get_embedding method."""
        return np.random.randn(self.config.get("embedding_dim", 768))

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        """Mock score_variant method."""
        emb_ref = self.get_embedding(ref_seq)
        emb_alt = self.get_embedding(alt_seq)
        cos_sim = np.dot(emb_ref, emb_alt) / (
            np.linalg.norm(emb_ref) * np.linalg.norm(emb_alt) + 1e-8
        )
        return float(1.0 - cos_sim)


class TestGenomicModelBase:
    """Test suite for GenomicModelBase abstract class."""

    def test_initialization(self, mock_model_config):
        """Test model initialization."""
        model = MockGenomicModel(mock_model_config)

        assert model.config == mock_model_config
        assert model.model_name == mock_model_config["name"]
        assert model.model_type == mock_model_config["type"]
        assert model.max_length == mock_model_config["max_length"]
        assert model.model is None
        assert model.tokenizer is None

    def test_load(self, mock_model_config):
        """Test model loading."""
        model = MockGenomicModel(mock_model_config)
        model.load()

        assert model.model is not None
        assert model.tokenizer is not None

    def test_get_embedding_shape(self, mock_model_config, sample_dna_sequence):
        """Test embedding shape."""
        model = MockGenomicModel(mock_model_config)
        model.load()

        embedding = model.get_embedding(sample_dna_sequence)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (mock_model_config["embedding_dim"],)
        assert embedding.dtype == np.float64

    def test_score_variant_returns_float(self, mock_model_config, sample_dna_sequence):
        """Test variant scoring returns float."""
        model = MockGenomicModel(mock_model_config)
        model.load()

        ref_seq = sample_dna_sequence
        alt_seq = "T" + sample_dna_sequence[1:]

        score = model.score_variant(ref_seq, alt_seq)

        assert isinstance(score, float)
        assert 0 <= score <= 2  # Cosine distance range

    def test_get_embeddings_batch(self, mock_model_config, sample_dna_sequence):
        """Test batch embedding extraction."""
        model = MockGenomicModel(mock_model_config)
        model.load()

        sequences = [sample_dna_sequence] * 5
        embeddings = model.get_embeddings_batch(sequences, batch_size=2)

        assert embeddings.shape == (5, mock_model_config["embedding_dim"])

    def test_score_variants_batch(self, mock_model_config, sample_dna_sequence):
        """Test batch variant scoring."""
        model = MockGenomicModel(mock_model_config)
        model.load()

        ref_seqs = [sample_dna_sequence] * 5
        alt_seqs = ["T" + sample_dna_sequence[1:]] * 5
        scores = model.score_variants_batch(ref_seqs, alt_seqs, batch_size=2)

        assert scores.shape == (5,)
        assert all(isinstance(s, (float, np.floating)) for s in scores)

    def test_num_parameters(self, mock_model_config):
        """Test parameter counting."""
        model = MockGenomicModel(mock_model_config)

        assert model.num_parameters == 0

        model.load()
        assert model.num_parameters > 0

    def test_repr(self, mock_model_config):
        """Test string representation."""
        model = MockGenomicModel(mock_model_config)
        repr_str = repr(model)

        assert "MockGenomicModel" in repr_str
        assert mock_model_config["name"] in repr_str
