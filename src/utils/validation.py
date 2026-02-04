"""Input validation utilities."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_dna_sequence(sequence: str) -> bool:
    """
    Validate that a string is a valid DNA sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        True if valid

    Raises:
        ValidationError: If sequence is invalid
    """
    if not sequence:
        raise ValidationError("DNA sequence cannot be empty")

    valid_bases = set("ACGTN")
    sequence_upper = sequence.upper()

    invalid_bases = set(sequence_upper) - valid_bases
    if invalid_bases:
        raise ValidationError(
            f"Invalid bases in sequence: {invalid_bases}. "
            f"Valid bases are: {valid_bases}"
        )

    return True


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file

    Returns:
        Path object

    Raises:
        ValidationError: If file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    return path


def validate_directory_exists(dir_path: str, create: bool = False) -> Path:
    """
    Validate that a directory exists, optionally creating it.

    Args:
        dir_path: Path to directory
        create: Whether to create directory if it doesn't exist

    Returns:
        Path object

    Raises:
        ValidationError: If directory doesn't exist and create=False
    """
    path = Path(dir_path)

    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        else:
            raise ValidationError(f"Directory not found: {dir_path}")

    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {dir_path}")

    return path


def validate_variant_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> bool:
    """
    Validate that a DataFrame has required columns for variant data.

    Args:
        df: Variant DataFrame
        required_columns: List of required column names

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if df is None or df.empty:
        raise ValidationError("DataFrame is empty")

    if required_columns is None:
        required_columns = ["chrom", "pos", "ref", "alt"]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate chromosome format
    if "chrom" in df.columns:
        invalid_chroms = df["chrom"].isna().sum()
        if invalid_chroms > 0:
            raise ValidationError(f"Found {invalid_chroms} rows with missing chromosome")

    # Validate position
    if "pos" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["pos"]):
            raise ValidationError("Position column must be numeric")

        invalid_pos = (df["pos"] <= 0).sum()
        if invalid_pos > 0:
            raise ValidationError(f"Found {invalid_pos} rows with invalid position (<= 0)")

    # Validate ref and alt alleles
    for col in ["ref", "alt"]:
        if col in df.columns:
            empty_alleles = df[col].isna().sum()
            if empty_alleles > 0:
                raise ValidationError(f"Found {empty_alleles} rows with missing {col} allele")

    return True


def validate_labels(labels: pd.Series) -> bool:
    """
    Validate binary labels.

    Args:
        labels: Series of binary labels

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    unique_values = set(labels.unique())

    if not unique_values.issubset({0, 1}):
        raise ValidationError(
            f"Labels must be binary (0/1). Found: {unique_values}"
        )

    if len(unique_values) < 2:
        raise ValidationError(
            f"Labels must contain both classes. Found only: {unique_values}"
        )

    return True


def validate_model_config(config: dict) -> bool:
    """
    Validate model configuration dictionary.

    Args:
        config: Model configuration

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    required_keys = ["name", "type", "embedding_dim", "max_length"]

    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValidationError(
            f"Missing required configuration keys: {missing_keys}"
        )

    # Validate model type
    valid_types = ["encoder", "causal"]
    if config["type"] not in valid_types:
        raise ValidationError(
            f"Invalid model type: {config['type']}. "
            f"Must be one of: {valid_types}"
        )

    # Validate numeric values
    if not isinstance(config["embedding_dim"], int) or config["embedding_dim"] <= 0:
        raise ValidationError("embedding_dim must be a positive integer")

    if not isinstance(config["max_length"], int) or config["max_length"] <= 0:
        raise ValidationError("max_length must be a positive integer")

    return True
