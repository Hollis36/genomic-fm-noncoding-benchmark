"""Caduceus model wrapper (bidirectional Mamba for DNA)."""

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base import GenomicModelBase


class CaduceusModel(GenomicModelBase):
    """Wrapper for Caduceus (kuleshov-group/caduceus-ph_seqlen-131k_d256_n4)."""

    def load(self):
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
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

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, dim]

        # Mean pooling
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            embedding = hidden_states.mean(dim=1)

        return embedding.cpu().float().numpy().flatten()

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        emb_ref = self.get_embedding(ref_seq)
        emb_alt = self.get_embedding(alt_seq)

        cos_sim = np.dot(emb_ref, emb_alt) / (
            np.linalg.norm(emb_ref) * np.linalg.norm(emb_alt) + 1e-8
        )
        return float(1.0 - cos_sim)
