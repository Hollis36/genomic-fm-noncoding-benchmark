"""Integration tests for model registry and loading."""

import numpy as np
import pytest
import torch

from src.models.base import GenomicModelBase
from src.models.registry import (
    MODEL_REGISTRY,
    create_model,
    list_available_models,
    load_model,
    register_model,
    unregister_model,
)


class StubGenomicModel(GenomicModelBase):
    """Lightweight stub model for integration testing without real downloads."""

    def load(self):
        self.model = torch.nn.Linear(10, self.config.get("embedding_dim", 768))
        self.tokenizer = "stub_tokenizer"

    def get_embedding(self, sequence: str) -> np.ndarray:
        dim = self.config.get("embedding_dim", 768)
        return np.zeros(dim, dtype=np.float32)

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        return 0.5


STUB_CONFIG = {
    "name": "stub-model",
    "type": "encoder",
    "embedding_dim": 256,
    "max_length": 512,
    "params": "1M",
}


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Test that the model registry correctly creates, loads, and manages models."""

    def test_registry_has_expected_models(self):
        """All expected model keys should be registered at import time."""
        models = list_available_models()
        expected = {"dnabert2", "nucleotide_transformer_500m", "hyenadna", "caduceus", "evo1"}
        assert expected.issubset(set(models))

    def test_register_and_create_custom_model(self):
        """Register a custom model class, create an instance, and clean up."""
        key = "_test_stub_integration"
        try:
            register_model(key, StubGenomicModel)
            assert key in list_available_models()

            model = create_model(key, STUB_CONFIG)
            assert isinstance(model, StubGenomicModel)
            assert model.model_name == STUB_CONFIG["name"]
        finally:
            if key in MODEL_REGISTRY:
                unregister_model(key)

    def test_register_duplicate_raises(self):
        """Registering a duplicate key should raise ValueError."""
        key = "_test_dup_integration"
        try:
            register_model(key, StubGenomicModel)
            with pytest.raises(ValueError, match="already registered"):
                register_model(key, StubGenomicModel)
        finally:
            if key in MODEL_REGISTRY:
                unregister_model(key)

    def test_create_unknown_model_raises(self):
        """Creating a model with an unknown key should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model key"):
            create_model("nonexistent_model_xyz", STUB_CONFIG)

    def test_load_model_end_to_end(self):
        """Register, load, and verify a model is fully functional."""
        key = "_test_load_integration"
        try:
            register_model(key, StubGenomicModel)
            model = load_model(key, STUB_CONFIG)

            assert model.model is not None
            assert model.tokenizer is not None

            emb = model.get_embedding("ATCGATCG")
            assert emb.shape == (STUB_CONFIG["embedding_dim"],)

            score = model.score_variant("ATCGATCG", "TTCGATCG")
            assert isinstance(score, float)
        finally:
            if key in MODEL_REGISTRY:
                unregister_model(key)

    def test_load_with_retry_succeeds(self):
        """load_with_retry should succeed on first attempt for a working model."""
        key = "_test_retry_integration"
        try:
            register_model(key, StubGenomicModel)
            model = create_model(key, STUB_CONFIG)
            result = model.load_with_retry()

            assert result is model
            assert model.model is not None
        finally:
            if key in MODEL_REGISTRY:
                unregister_model(key)

    def test_batch_operations_after_load(self):
        """Batch embedding and scoring should work on a loaded model."""
        key = "_test_batch_integration"
        try:
            register_model(key, StubGenomicModel)
            model = load_model(key, STUB_CONFIG)

            seqs = ["ATCG" * 50] * 4
            embeddings = model.get_embeddings_batch(seqs, batch_size=2)
            assert embeddings.shape == (4, STUB_CONFIG["embedding_dim"])

            ref_seqs = ["ATCG" * 50] * 3
            alt_seqs = ["TTCG" * 50] * 3
            scores = model.score_variants_batch(ref_seqs, alt_seqs, batch_size=2)
            assert scores.shape == (3,)
        finally:
            if key in MODEL_REGISTRY:
                unregister_model(key)
