"""Evo model wrapper (7B causal, requires 4-bit quantization on RTX 4090)."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import GenomicModelBase


class EvoModel(GenomicModelBase):
    """Wrapper for Evo-1 (togethercomputer/evo-1-131k-base)."""

    def load(self):
        print(f"Loading {self.model_name} with 4-bit quantization ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        quantize = self.config.get("quantize", "4bit")

        if quantize == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

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
        )
        # Move to model device (handles device_map="auto")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, 4096]

        embedding = hidden_states.mean(dim=1)
        return embedding.cpu().float().numpy().flatten()

    @torch.no_grad()
    def _compute_log_likelihood(self, sequence: str) -> float:
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum().item()

    def score_variant(self, ref_seq: str, alt_seq: str) -> float:
        ll_ref = self._compute_log_likelihood(ref_seq)
        ll_alt = self._compute_log_likelihood(alt_seq)
        return float(ll_ref - ll_alt)
