"""Progress tracking and monitoring utilities."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from .logging_config import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Track experiment progress and metrics."""

    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.metrics = {}
        self.config = {}
        self.status = "initialized"

        self.log_file = self.output_dir / "experiment.jsonl"

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._write_log({
            "type": "config",
            "timestamp": time.time(),
            "config": config,
        })
        logger.info(f"Logged configuration for experiment: {self.experiment_name}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            "value": value,
            "step": step,
            "timestamp": time.time(),
        })

        self._write_log({
            "type": "metric",
            "name": name,
            "value": value,
            "step": step,
            "timestamp": time.time(),
        })

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/iteration number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_status(self, status: str, message: Optional[str] = None) -> None:
        """
        Log experiment status change.

        Args:
            status: New status (e.g., "running", "completed", "failed")
            message: Optional status message
        """
        self.status = status

        self._write_log({
            "type": "status",
            "status": status,
            "message": message,
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
        })

        logger.info(f"Experiment {self.experiment_name} status: {status}")

    def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to JSONL file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_final_results(self) -> None:
        """Save final experiment results as JSON."""
        results = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "metrics": self.metrics,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_duration": time.time() - self.start_time,
        }

        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved final results to {results_file}")

    def __enter__(self):
        """Context manager entry."""
        self.log_status("running")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.log_status("completed")
        else:
            self.log_status("failed", message=str(exc_val))

        self.save_final_results()


class ProgressBar:
    """Wrapper around tqdm with additional features."""

    def __init__(
        self,
        total: int,
        desc: str = "",
        unit: str = "it",
        leave: bool = True,
        **kwargs,
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of iterations
            desc: Description/prefix for progress bar
            unit: Unit name for iterations
            leave: Whether to leave the progress bar after completion
            **kwargs: Additional arguments for tqdm
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            **kwargs,
        )

    def update(self, n: int = 1, **kwargs) -> None:
        """
        Update progress bar.

        Args:
            n: Number of iterations to increment
            **kwargs: Additional tqdm.update arguments
        """
        self.pbar.update(n)
        if kwargs:
            self.pbar.set_postfix(**kwargs)

    def set_description(self, desc: str) -> None:
        """Update progress bar description."""
        self.pbar.set_description(desc)

    def close(self) -> None:
        """Close progress bar."""
        self.pbar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_progress_callback(total: int, desc: str = "Processing"):
    """
    Create a simple callback function for progress tracking.

    Args:
        total: Total number of steps
        desc: Description for progress bar

    Returns:
        Callback function that updates progress
    """
    pbar = tqdm(total=total, desc=desc)

    def callback(step: int = 1):
        """Update progress."""
        pbar.update(step)

    return callback
