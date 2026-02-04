"""Caching utilities for embeddings and model outputs."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Cache for model embeddings to avoid recomputation."""

    def __init__(self, cache_dir: str = ".cache/embeddings", max_size_gb: float = 10.0):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)

        logger.info(f"Initialized embedding cache at {self.cache_dir}")

    def _get_cache_key(self, model_name: str, sequence: str) -> str:
        """
        Generate cache key for a sequence.

        Args:
            model_name: Name of the model
            sequence: DNA sequence

        Returns:
            Cache key (hash)
        """
        key_str = f"{model_name}:{sequence}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        # Use first 2 chars for subdirectory (avoid too many files in one dir)
        subdir = self.cache_dir / cache_key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{cache_key}.npy"

    def get(self, model_name: str, sequence: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.

        Args:
            model_name: Name of the model
            sequence: DNA sequence

        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(model_name, sequence)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                logger.debug(f"Cache hit for {model_name}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                cache_path.unlink()  # Remove corrupted cache

        return None

    def set(self, model_name: str, sequence: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding.

        Args:
            model_name: Name of the model
            sequence: DNA sequence
            embedding: Embedding vector
        """
        cache_key = self._get_cache_key(model_name, sequence)
        cache_path = self._get_cache_path(cache_key)

        try:
            np.save(cache_path, embedding)
            logger.debug(f"Cached embedding for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

        # Check cache size
        self._enforce_size_limit()

    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0

        for cache_file in self.cache_dir.rglob("*.npy"):
            total_size += cache_file.stat().st_size

        return total_size

    def _enforce_size_limit(self) -> None:
        """Remove oldest cache files if size limit is exceeded."""
        current_size = self._get_cache_size()

        if current_size > self.max_size_bytes:
            logger.info(f"Cache size ({current_size / 1024**3:.2f} GB) exceeds limit, cleaning...")

            # Get all cache files sorted by modification time
            cache_files = sorted(
                self.cache_dir.rglob("*.npy"),
                key=lambda p: p.stat().st_mtime,
            )

            # Remove oldest files until under limit
            for cache_file in cache_files:
                if current_size <= self.max_size_bytes * 0.9:  # Keep 90% of limit
                    break

                file_size = cache_file.stat().st_size
                cache_file.unlink()
                current_size -= file_size
                logger.debug(f"Removed cache file: {cache_file}")

            logger.info(f"Cache cleaned. New size: {current_size / 1024**3:.2f} GB")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.rglob("*.npy"):
            cache_file.unlink()

        logger.info("Cleared embedding cache")


def memoize(func: Callable) -> Callable:
    """
    Decorator for memoizing function results.

    Args:
        func: Function to memoize

    Returns:
        Memoized function
    """
    cache = {}

    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = (args, tuple(sorted(kwargs.items())))

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrapper


class ResultCache:
    """Persistent cache for experiment results."""

    def __init__(self, cache_dir: str = ".cache/results"):
        """
        Initialize result cache.

        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, experiment_id: str) -> Path:
        """Get cache file path for an experiment."""
        return self.cache_dir / f"{experiment_id}.pkl"

    def get(self, experiment_id: str) -> Optional[Any]:
        """
        Get cached result.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Cached result or None
        """
        cache_path = self._get_cache_path(experiment_id)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                logger.info(f"Loaded cached result for {experiment_id}")
                return result
            except Exception as e:
                logger.warning(f"Failed to load cached result: {e}")

        return None

    def set(self, experiment_id: str, result: Any) -> None:
        """
        Cache a result.

        Args:
            experiment_id: Unique experiment identifier
            result: Result to cache
        """
        cache_path = self._get_cache_path(experiment_id)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Cached result for {experiment_id}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def clear(self, experiment_id: Optional[str] = None) -> None:
        """
        Clear cached results.

        Args:
            experiment_id: If provided, clear only this experiment; otherwise clear all
        """
        if experiment_id:
            cache_path = self._get_cache_path(experiment_id)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for {experiment_id}")
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cleared all cached results")
