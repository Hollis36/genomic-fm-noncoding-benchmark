"""DNABERT-2 model wrapper (117M parameters, encoder-based)."""

import sys
from types import ModuleType

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import GenomicModelBase

# Mock triton for Mac compatibility (DNABERT-2 requires it but we don't use flash-attn on Mac)
if 'triton' not in sys.modules:
    class MockTriton(ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    sys.modules['triton'] = MockTriton('triton')
    sys.modules['triton.language'] = MockTriton('triton.language')


class DNABERT2Model(GenomicModelBase):
    """Wrapper for DNABERT-2 (zhihan1996/DNABERT-2-117M)."""

    def load(self):
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        # Disable flash attention on non-CUDA devices (e.g., MPS on Mac)
        import os
        os.environ["FLASH_ATTENTION_SKIP_CUDA_CHECK"] = "1"

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention instead of flash
        ).to(self.device)
        self.model.eval()
        print(f"  Loaded. Parameters: {self.num_parameters:,}")
        return self

    @torch.no_grad()
    def get_embedding(self, sequence: str) -> np.ndarray:
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

        # Mean pooling over tokens (excluding padding)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
        masked_hidden = hidden_states * attention_mask
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

        return embedding.cpu().numpy().flatten()

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        emb_ref = self.get_embedding(ref_seq)
        emb_alt = self.get_embedding(alt_seq)

        # Cosine distance: 1 - cosine_similarity
        cos_sim = np.dot(emb_ref, emb_alt) / (
            np.linalg.norm(emb_ref) * np.linalg.norm(emb_alt) + 1e-8
        )
        return float(1.0 - cos_sim)
