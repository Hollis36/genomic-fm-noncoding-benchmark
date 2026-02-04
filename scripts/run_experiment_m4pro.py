#!/usr/bin/env python3
"""
Optimized experiment runner for Apple M4 Pro.

Runs models that work well on Apple Silicon with MPS acceleration.
For large models, recommend using cloud GPU.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_model
from src.evaluation import ZeroShotEvaluator, LinearProbeEvaluator, LoRAFineTuner
from src.utils import setup_logging, get_logger

setup_logging(log_level="INFO", log_file="experiment_m4pro.log")
logger = get_logger(__name__)


# Models optimized for M4 Pro (24GB unified memory)
M4PRO_COMPATIBLE_MODELS = {
    "dnabert2": {
        "recommended": True,
        "batch_size": 8,
        "notes": "Runs excellently on M4 Pro"
    },
    "nucleotide_transformer_500m": {
        "recommended": True,
        "batch_size": 4,
        "notes": "Good performance on M4 Pro"
    },
    "caduceus": {
        "recommended": False,
        "batch_size": 4,
        "notes": "May work, test carefully"
    },
}

# Models NOT recommended for M4 Pro
NOT_RECOMMENDED_MODELS = {
    "nucleotide_transformer_2500m": "Too large, use cloud GPU",
    "hyenadna": "Better on NVIDIA GPU",
    "evo1": "Requires NVIDIA GPU for quantization",
}


def check_device():
    """Check available compute device and provide recommendations."""
    logger.info("Checking device compatibility...")

    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("✅ MPS (Metal) available - will use GPU acceleration")
    else:
        device = "cpu"
        logger.warning("⚠️  MPS not available - will use CPU (slower)")

    # Test MPS
    if device == "mps":
        try:
            test_tensor = torch.randn(100, 100, device="mps")
            _ = torch.matmul(test_tensor, test_tensor)
            logger.info("✅ MPS computation test successful")
        except Exception as e:
            logger.error(f"❌ MPS test failed: {e}")
            logger.warning("Falling back to CPU")
            device = "cpu"

    return device


def validate_model_choice(models):
    """Validate and warn about model choices."""
    for model in models:
        if model in NOT_RECOMMENDED_MODELS:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"⚠️  Model '{model}' is NOT recommended for M4 Pro")
            logger.warning(f"Reason: {NOT_RECOMMENDED_MODELS[model]}")
            logger.warning(f"{'='*60}\n")

            response = input(f"Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                logger.info(f"Skipping model '{model}'")
                models.remove(model)

        elif model in M4PRO_COMPATIBLE_MODELS:
            info = M4PRO_COMPATIBLE_MODELS[model]
            if info["recommended"]:
                logger.info(f"✅ Model '{model}': {info['notes']}")
            else:
                logger.warning(f"⚠️  Model '{model}': {info['notes']}")


def get_optimal_batch_size(model_key):
    """Get optimal batch size for model on M4 Pro."""
    if model_key in M4PRO_COMPATIBLE_MODELS:
        return M4PRO_COMPATIBLE_MODELS[model_key]["batch_size"]
    return 2  # Conservative default


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments optimized for Apple M4 Pro"
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "linear_probe", "lora"],
        required=True,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dnabert2"],
        help="Model keys (default: dnabert2)",
    )
    parser.add_argument(
        "--negative-sets",
        nargs="+",
        default=["N1", "N3"],
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help="Sequence context size (reduce if OOM)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size",
    )
    parser.add_argument("--output-dir", default="results/m4pro")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip compatibility warnings",
    )
    args = parser.parse_args()

    # Device check
    device = check_device()

    logger.info("\n" + "="*60)
    logger.info("Apple M4 Pro Optimized Experiment Runner")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Device: {device}")
    logger.info(f"Negative sets: {args.negative_sets}")
    logger.info("="*60 + "\n")

    # Validate model choices
    if not args.force:
        validate_model_choice(args.models)

    # Load configs
    with open(PROJECT_ROOT / "configs/models.yaml") as f:
        all_configs = yaml.safe_load(f)

    # Run experiments
    for model_key in args.models:
        if model_key not in all_configs:
            logger.warning(f"Unknown model '{model_key}', skipping")
            continue

        config = all_configs[model_key]

        # Get optimal batch size
        if args.batch_size:
            batch_size = args.batch_size
        else:
            batch_size = get_optimal_batch_size(model_key)

        logger.info(f"\n{'='*60}")
        logger.info(f"Running {model_key} with batch_size={batch_size}")
        logger.info(f"{'='*60}\n")

        for neg_set in args.negative_sets:
            try:
                # Load dataset
                from scripts.run_experiment import load_dataset, NEGATIVE_SET_FILES

                positive_path = "data/processed/positive_noncoding_annotated.tsv"
                neg_path = NEGATIVE_SET_FILES.get(neg_set)

                if not neg_path or not Path(neg_path).exists():
                    logger.warning(f"Negative set {neg_set} not found, skipping")
                    continue

                seq_parquet = (
                    f"data/processed/sequences_positive_noncoding_annotated_"
                    f"{args.context_size}bp.parquet"
                )

                df = load_dataset(positive_path, neg_path, seq_parquet)

                # Run evaluation
                if args.mode == "zero_shot":
                    logger.info(f"Zero-shot evaluation: {model_key} | {neg_set}")
                    model = load_model(model_key, config)
                    model.load()

                    evaluator = ZeroShotEvaluator(
                        model,
                        output_dir=args.output_dir
                    )
                    evaluator.evaluate(df, negative_set_name=neg_set)

                elif args.mode == "linear_probe":
                    logger.info(f"Linear probe: {model_key} | {neg_set}")
                    model = load_model(model_key, config)
                    model.load()

                    evaluator = LinearProbeEvaluator(
                        model,
                        output_dir=args.output_dir
                    )
                    # Run both ref and alt
                    evaluator.evaluate(df, negative_set_name=neg_set, use_ref_seq=True)
                    evaluator.evaluate(df, negative_set_name=neg_set, use_ref_seq=False)

                elif args.mode == "lora":
                    logger.info(f"LoRA fine-tuning: {model_key} | {neg_set}")
                    tuner = LoRAFineTuner(
                        config,
                        epochs=args.epochs,
                        batch_size=batch_size,
                        lr=args.lr,
                        output_dir=args.output_dir,
                    )
                    tuner.evaluate(df, negative_set_name=neg_set)

                logger.info(f"✅ Completed {model_key} | {neg_set}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"❌ Out of memory for {model_key}")
                    logger.error("Try reducing --batch-size or --context-size")
                else:
                    logger.error(f"❌ Error: {e}")
                continue

            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                continue

    logger.info("\n" + "="*60)
    logger.info("All experiments complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
