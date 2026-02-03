#!/usr/bin/env python3
"""
Step 3: Build additional negative sets (N2: gnomAD common, N3: matched random).

Requires:
  - data/processed/positive_noncoding.tsv (from step 02)
  - gnomAD v4 sites VCF (user must download separately due to size)
  - Reference genome FASTA

Produces:
  - data/processed/negative_N2_gnomad_common.tsv
  - data/processed/negative_N3_matched_random.tsv
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pysam


def build_n2_gnomad(positive_df: pd.DataFrame, gnomad_vcf: str, output_path: str):
    """
    N2: Extract common non-coding variants from gnomAD (MAF > 5%).

    NOTE: gnomAD v4 genome VCF is very large (~750 GB total).
    Users should pre-filter by chromosome or use the Hail framework.
    This script processes a pre-filtered gnomAD VCF.
    """
    print("Building N2: gnomAD common variants ...")

    # Collect positive positions to exclude
    pos_set = set(zip(positive_df["chrom"], positive_df["pos"]))

    records = []
    vcf = pysam.VariantFile(gnomad_vcf)

    for rec in vcf:
        # Skip if overlaps with positive set
        if (rec.chrom, rec.pos) in pos_set:
            continue

        # Check allele frequency
        af = rec.info.get("AF", [0])
        if isinstance(af, tuple):
            af = af[0]
        if af < 0.05:
            continue

        # Only SNVs
        if len(rec.ref) != 1 or len(rec.alts) == 0 or len(rec.alts[0]) != 1:
            continue

        records.append({
            "chrom": rec.chrom,
            "pos": rec.pos,
            "ref": rec.ref,
            "alt": rec.alts[0],
            "af": af,
            "source": "gnomAD_v4_common",
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"N2 variants: {len(df)}, saved to {output_path}")
    return df


def build_n3_matched_random(
    positive_df: pd.DataFrame,
    ref_fasta: str,
    output_path: str,
    seed: int = 42,
):
    """
    N3: For each positive variant, create a random substitution at the same
    genomic position with a different alt allele.
    """
    print("Building N3: position-matched random variants ...")
    random.seed(seed)
    np.random.seed(seed)

    ref = pysam.FastaFile(ref_fasta)
    records = []

    for _, row in positive_df.iterrows():
        chrom = str(row["chrom"])
        pos = int(row["pos"])

        try:
            ref_base = ref.fetch(chrom, pos - 1, pos).upper()
        except (ValueError, KeyError):
            # Try adding/removing 'chr' prefix
            alt_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
            try:
                ref_base = ref.fetch(alt_chrom, pos - 1, pos).upper()
            except (ValueError, KeyError):
                continue

        if ref_base not in "ACGT":
            continue

        # Pick a random alt that differs from both ref and the original pathogenic alt
        candidates = [b for b in "ACGT" if b != ref_base and b != row.get("alt", "")]
        if not candidates:
            candidates = [b for b in "ACGT" if b != ref_base]

        alt_base = random.choice(candidates)

        records.append({
            "chrom": chrom,
            "pos": pos,
            "ref": ref_base,
            "alt": alt_base,
            "source": "matched_random",
        })

    ref.close()

    df = pd.DataFrame(records)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"N3 variants: {len(df)}, saved to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build negative variant sets")
    parser.add_argument(
        "--positive",
        default="data/processed/positive_noncoding.tsv",
        help="Positive variants TSV from step 02",
    )
    parser.add_argument(
        "--gnomad-vcf",
        default=None,
        help="Pre-filtered gnomAD v4 VCF (optional, for N2)",
    )
    parser.add_argument(
        "--ref-fasta",
        default="data/reference/GRCh38.fa",
        help="Reference genome FASTA",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_df = pd.read_csv(args.positive, sep="\t")
    print(f"Loaded {len(positive_df)} positive variants")

    # N2: gnomAD common (optional — requires large download)
    if args.gnomad_vcf:
        build_n2_gnomad(
            positive_df,
            args.gnomad_vcf,
            str(output_dir / "negative_N2_gnomad_common.tsv"),
        )
    else:
        print("Skipping N2 (no gnomAD VCF provided). See README for download instructions.")

    # N3: Matched random (always possible)
    build_n3_matched_random(
        positive_df,
        args.ref_fasta,
        str(output_dir / "negative_N3_matched_random.tsv"),
        seed=args.seed,
    )
