"""
LoRA fine-tuning evaluation for genomic foundation models.

Uses HuggingFace PEFT library for parameter-efficient fine-tuning
on the variant pathogenicity classification task.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class VariantDataset(Dataset):
    """Dataset for variant pathogenicity classification."""

    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class ClassificationHead(torch.nn.Module):
    """Simple classification head on top of frozen/LoRA encoder."""

    def __init__(self, hidden_dim, num_labels=1, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


class LoRAFineTuner:
    """LoRA fine-tuning for variant effect prediction."""

    def __init__(
        self,
        model_config: dict,
        n_splits: int = 5,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 16,
        seed: int = 42,
        output_dir: str = "results",
    ):
        self.model_config = model_config
        self.n_splits = n_splits
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        """Build encoder + LoRA + classification head."""
        model_name = self.model_config["name"]
        lora_cfg = self.model_config.get("lora", {})

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        )
        encoder = get_peft_model(encoder, peft_config)
        encoder.print_trainable_parameters()

        # Classification head
        hidden_dim = self.model_config.get("embedding_dim", 768)
        head = ClassificationHead(hidden_dim)

        return tokenizer, encoder, head

    def _train_one_fold(
        self,
        train_seqs: list[str],
        train_labels: np.ndarray,
        val_seqs: list[str],
        val_labels: np.ndarray,
    ) -> dict:
        """Train for one CV fold and return validation metrics."""
        try:
            return self._train_one_fold_inner(train_seqs, train_labels, val_seqs, val_labels)
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during training fold. Clearing cache and returning zero metrics.")
            torch.cuda.empty_cache()
            return {"auroc": 0.0, "auprc": 0.0, "mcc": 0.0, "f1": 0.0, "optimal_threshold": 0.5}

    def _train_one_fold_inner(
        self,
        train_seqs: list[str],
        train_labels: np.ndarray,
        val_seqs: list[str],
        val_labels: np.ndarray,
    ) -> dict:
        """Inner training logic for one CV fold."""
        tokenizer, encoder, head = self._build_model()
        encoder = encoder.to(self.device)
        head = head.to(self.device)

        max_length = self.model_config.get("max_length", 512)

        train_ds = VariantDataset(train_seqs, train_labels, tokenizer, max_length)
        val_ds = VariantDataset(val_seqs, val_labels, tokenizer, max_length)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # Optimizer: LoRA params + head params
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(head.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader),
            num_training_steps=len(train_loader) * self.epochs,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        # Training loop
        best_auroc = 0
        best_metrics = {}

        for epoch in range(self.epochs):
            encoder.train()
            head.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state

                # Mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                logits = head(pooled.float()).squeeze(-1)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            # Validation
            encoder.eval()
            head.eval()
            val_probs = []
            val_true = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                    hidden = outputs.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                    logits = head(pooled.float()).squeeze(-1)
                    probs = torch.sigmoid(logits)

                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(batch["label"].numpy())

            metrics = compute_all_metrics(np.array(val_true), np.array(val_probs))

            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_metrics = metrics

        # Cleanup
        del encoder, head
        torch.cuda.empty_cache()

        return best_metrics

    def evaluate(
        self,
        df: pd.DataFrame,
        negative_set_name: str = "N1",
    ) -> dict:
        """Run full LoRA fine-tuning evaluation with K-fold CV."""
        model_name = self.model_config["name"].split("/")[-1]
        print(f"\n{'='*60}")
        print(f"LoRA fine-tuning: {model_name} | {negative_set_name}")
        print(f"{'='*60}")

        sequences = df["alt_seq"].tolist()
        labels = df["label"].values

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )

        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(sequences, labels)):
            print(f"\n--- Fold {fold+1}/{self.n_splits} ---")
            train_seqs = [sequences[i] for i in train_idx]
            val_seqs = [sequences[i] for i in test_idx]

            metrics = self._train_one_fold(
                train_seqs, labels[train_idx],
                val_seqs, labels[test_idx],
            )
            metrics["fold"] = fold
            fold_results.append(metrics)
            print(f"  Fold {fold+1} AUROC: {metrics['auroc']:.4f}")

        # Aggregate
        overall = {
            metric: {
                "mean": np.mean([f[metric] for f in fold_results]),
                "std": np.std([f[metric] for f in fold_results]),
            }
            for metric in ["auroc", "auprc", "mcc", "f1"]
        }

        print(f"\n  Overall AUROC: {overall['auroc']['mean']:.4f} "
              f"± {overall['auroc']['std']:.4f}")

        result = {
            "model": model_name,
            "negative_set": negative_set_name,
            "method": "lora_finetune",
            "n_variants": len(df),
            "overall": overall,
            "folds": fold_results,
            "hyperparameters": {
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "lora_r": self.model_config.get("lora", {}).get("r", 8),
                "lora_alpha": self.model_config.get("lora", {}).get("alpha", 16),
            },
        }

        output_path = self.output_dir / f"lora_{model_name}_{negative_set_name}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Results saved to {output_path}")

        return result
