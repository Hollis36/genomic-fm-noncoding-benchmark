#!/usr/bin/env python3
"""
Main experiment runner for the benchmark.

Usage:
    python -m scripts.run_experiment --mode zero_shot --models dnabert2 --negative-sets N1 N3
    python -m scripts.run_experiment --mode lora --models dnabert2 --negative-sets N3
    python -m scripts.run_experiment --mode linear_probe --models dnabert2 --negative-sets N1
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_model
from src.evaluation import ZeroShotEvaluator, LinearProbeEvaluator, LoRAFineTuner


def load_dataset(
    positive_path: str,
    negative_path: str,
    seq_parquet: str = None,
) -> pd.DataFrame:
    """Load and combine positive + negative variants into a single DataFrame."""
    pos_df = pd.read_csv(positive_path, sep="\t")
    neg_df = pd.read_csv(negative_path, sep="\t")

    pos_df["label"] = 1
    neg_df["label"] = 0

    # Balance: downsample larger set
    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=n, random_state=42)
    neg_df = neg_df.sample(n=n, random_state=42)

    df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)

    # If sequence parquet exists, merge
    if seq_parquet and Path(seq_parquet).exists():
        seq_df = pd.read_parquet(seq_parquet)
        df = df.merge(seq_df[["chrom", "pos", "ref_seq", "alt_seq"]], on=["chrom", "pos"], how="inner")

    print(f"Dataset: {len(df)} variants ({pos_df.shape[0]} pos + {neg_df.shape[0]} neg)")
    return df


NEGATIVE_SET_FILES = {
    "N1": "data/processed/negative_N1_benign_annotated.tsv",
    "N2": "data/processed/negative_N2_gnomad_common_annotated.tsv",
    "N3": "data/processed/negative_N3_matched_random_annotated.tsv",
}


def main():
    parser = argparse.ArgumentParser(description="Run benchmark experiments")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "linear_probe", "lora"],
        required=True,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dnabert2"],
        help="Model keys from configs/models.yaml",
    )
    parser.add_argument(
        "--negative-sets",
        nargs="+",
        default=["N1", "N3"],
    )
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Load model configs
    with open("configs/models.yaml") as f:
        all_configs = yaml.safe_load(f)

    positive_path = "data/processed/positive_noncoding_annotated.tsv"

    for model_key in args.models:
        if model_key not in all_configs:
            print(f"WARNING: Unknown model '{model_key}', skipping")
            continue

        config = all_configs[model_key]

        for neg_set in args.negative_sets:
            neg_path = NEGATIVE_SET_FILES.get(neg_set)
            if neg_path is None or not Path(neg_path).exists():
                print(f"WARNING: Negative set file not found for {neg_set}, skipping")
                continue

            seq_parquet = f"data/processed/sequences_positive_noncoding_annotated_{args.context_size}bp.parquet"

            df = load_dataset(positive_path, neg_path, seq_parquet)

            if "ref_seq" not in df.columns:
                print("ERROR: No sequence data. Run 05_extract_sequences.py first.")
                sys.exit(1)

            if args.mode == "zero_shot":
                model = load_model(model_key, config)
                model.load()
                evaluator = ZeroShotEvaluator(model, output_dir=args.output_dir)
                evaluator.evaluate(df, negative_set_name=neg_set)

            elif args.mode == "linear_probe":
                model = load_model(model_key, config)
                model.load()
                evaluator = LinearProbeEvaluator(model, output_dir=args.output_dir)
                evaluator.evaluate(df, negative_set_name=neg_set, use_ref_seq=True)
                evaluator.evaluate(df, negative_set_name=neg_set, use_ref_seq=False)

            elif args.mode == "lora":
                tuner = LoRAFineTuner(
                    config,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    output_dir=args.output_dir,
                )
                tuner.evaluate(df, negative_set_name=neg_set)


if __name__ == "__main__":
    main()
