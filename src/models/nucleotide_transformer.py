"""Nucleotide Transformer v2 model wrapper (500M / 2.5B, encoder-based)."""

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base import GenomicModelBase


class NucleotideTransformerModel(GenomicModelBase):
    """Wrapper for Nucleotide Transformer v2 models."""

    def load(self):
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"  Loaded. Parameters: {self.num_parameters:,}")
        return self

    @torch.no_grad()
    def get_embedding(self, sequence: str) -> np.ndarray:
        tokens = self.tokenizer.batch_encode_plus(
            [sequence],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, dim]

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

        return embedding.cpu().numpy().flatten()

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        emb_ref = self.get_embedding(ref_seq)
        emb_alt = self.get_embedding(alt_seq)

        cos_sim = np.dot(emb_ref, emb_alt) / (
            np.linalg.norm(emb_ref) * np.linalg.norm(emb_alt) + 1e-8
        )
        return float(1.0 - cos_sim)
