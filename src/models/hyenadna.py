"""HyenaDNA model wrapper (causal, sub-quadratic long-range)."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import GenomicModelBase


class HyenaDNAModel(GenomicModelBase):
    """Wrapper for HyenaDNA (LongSafari/hyenadna-large-1m-seqlen-hf)."""

    def load(self):
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
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
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, dim]

        # Mean pooling
        embedding = hidden_states.mean(dim=1)
        return embedding.cpu().float().numpy().flatten()

    @torch.no_grad()
    def _compute_log_likelihood(self, sequence: str) -> float:
        """Compute total log-likelihood of a sequence under the causal model."""
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Shift for causal: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum().item()

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        ll_ref = self._compute_log_likelihood(ref_seq)
        ll_alt = self._compute_log_likelihood(alt_seq)
        # Positive score = alt less likely = potentially pathogenic
        return float(ll_ref - ll_alt)
