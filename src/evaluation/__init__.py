from .zero_shot import ZeroShotEvaluator
from .linear_probe import LinearProbeEvaluator
from .metrics import compute_all_metrics, delong_test


def __getattr__(name):
    if name == "LoRAFineTuner":
        from .lora_finetune import LoRAFineTuner

        return LoRAFineTuner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
