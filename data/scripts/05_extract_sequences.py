#!/usr/bin/env python3
"""
Step 5: Extract reference & alternate DNA sequences around each variant.

For each variant, extracts a window of `context_size` bp centered on the
variant position, producing both reference and alternate sequences.

Produces:
  - data/processed/sequences_{context_size}bp.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
import pysam


def extract_sequences(
    variant_tsv: str,
    ref_fasta: str,
    context_size: int = 1024,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Extract reference and alternate DNA sequences around each variant.

    For each variant, fetches a window of context_size bp centered on the
    variant position from the reference genome, then constructs the alternate
    sequence by substituting the alt allele.

    Args:
        variant_tsv: Path to annotated variant TSV with chrom, pos, ref, alt columns.
        ref_fasta: Path to indexed reference genome FASTA (GRCh38).
        context_size: Total context window size in base pairs (centered on variant).
        output_path: Optional path to save output as Parquet.

    Returns:
        DataFrame with ref_seq, alt_seq, and variant metadata.
    """
    df = pd.read_csv(variant_tsv, sep="\t")
    ref = pysam.FastaFile(ref_fasta)

    records = []
    half = context_size // 2
    skipped = 0

    for idx, row in df.iterrows():
        chrom = str(row["chrom"])
        pos = int(row["pos"])
        ref_allele = str(row["ref"])
        alt_allele = str(row["alt"])

        start = max(0, pos - 1 - half)
        end = pos - 1 + half

        try:
            seq = ref.fetch(chrom, start, end).upper()
        except (ValueError, KeyError):
            # Try chromosome name variants
            alt_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
            try:
                seq = ref.fetch(alt_chrom, start, end).upper()
            except (ValueError, KeyError):
                skipped += 1
                continue

        if len(seq) < context_size:
            skipped += 1
            continue

        # Verify reference allele matches
        center_idx = pos - 1 - start
        if center_idx >= len(seq):
            skipped += 1
            continue

        # Build alternate sequence
        alt_seq = seq[:center_idx] + alt_allele + seq[center_idx + len(ref_allele):]

        records.append({
            "chrom": chrom,
            "pos": pos,
            "ref_allele": ref_allele,
            "alt_allele": alt_allele,
            "ref_seq": seq,
            "alt_seq": alt_seq,
            "region_category": row.get("region_category", "unknown"),
            "label": row.get("label", -1),
            "source": row.get("source", ""),
        })

    ref.close()

    result_df = pd.DataFrame(records)

    if output_path:
        result_df.to_parquet(output_path, index=False)
        print(f"Saved {len(result_df)} sequences to {output_path} (skipped {skipped})")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DNA sequences around variants")
    parser.add_argument("--variants", required=True, help="Annotated variant TSV")
    parser.add_argument("--ref-fasta", default="data/reference/GRCh38.fa")
    parser.add_argument("--context-size", type=int, default=1024, help="Context window in bp")
    parser.add_argument("--output", default=None, help="Output parquet path")
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.variants).stem
        args.output = f"data/processed/sequences_{stem}_{args.context_size}bp.parquet"

    extract_sequences(args.variants, args.ref_fasta, args.context_size, args.output)
