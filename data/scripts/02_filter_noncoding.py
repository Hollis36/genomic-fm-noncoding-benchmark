#!/usr/bin/env python3
"""
Step 2: Filter ClinVar VCF for non-coding pathogenic / benign variants.

Produces:
  - data/processed/positive_noncoding.tsv  (Pathogenic/Likely_pathogenic)
  - data/processed/negative_N1_benign.tsv  (Benign/Likely_benign)
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

# Non-coding molecular consequence keywords
NONCODING_MC = {
    "intron_variant",
    "non-coding_transcript_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "intergenic_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "splice_region_variant",
}

PATHOGENIC_TERMS = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN_TERMS = {"Benign", "Likely_benign", "Benign/Likely_benign"}


def parse_info(info_str: str) -> dict[str, str | bool]:
    """Parse VCF INFO field into a dictionary.

    Args:
        info_str: Semicolon-delimited INFO string from a VCF record.
            Key=value pairs become string entries; standalone flags become True.

    Returns:
        Parsed INFO dictionary.
    """
    if not info_str:
        return {}
    info: dict[str, str | bool] = {}
    for item in info_str.split(";"):
        if "=" in item:
            key, val = item.split("=", 1)
            info[key] = val
        else:
            info[item] = True
    return info


def get_molecular_consequences(info: dict[str, str | bool]) -> set[str]:
    """Extract molecular consequence terms from the VCF MC (Molecular Consequence) field.

    The MC field format is: "SO:accession|term,SO:accession|term,...".
    This function extracts the human-readable term from each entry.

    Args:
        info: Parsed INFO dictionary (output of parse_info).

    Returns:
        Set of molecular consequence term strings.
    """
    mc = info.get("MC", "")
    if not mc:
        return set()
    consequences: set[str] = set()
    for entry in str(mc).split(","):
        parts = entry.split("|")
        if len(parts) >= 2:
            consequences.add(parts[1])
    return consequences


def is_noncoding(consequences: set[str]) -> bool:
    """Check if any consequence is non-coding."""
    return bool(consequences & NONCODING_MC)


def process_clinvar(vcf_path: str, output_dir: str) -> None:
    """Process ClinVar VCF and output positive/negative variant sets.

    Reads a ClinVar VCF (optionally gzipped), filters for non-coding SNVs,
    and splits them into pathogenic (positive) and benign (negative N1) sets.

    Args:
        vcf_path: Path to ClinVar VCF file (.vcf or .vcf.gz).
        output_dir: Directory to write output TSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive = []
    negative_n1 = []

    opener = gzip.open if vcf_path.endswith(".gz") else open

    print(f"Processing {vcf_path} ...")
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info_str = fields[:8]

            # Skip multi-allelic for simplicity
            if "," in alt:
                continue

            info = parse_info(info_str)
            clnsig = info.get("CLNSIG", "")
            consequences = get_molecular_consequences(info)

            if not is_noncoding(consequences):
                continue

            # Get review stars
            clnrevstat = info.get("CLNREVSTAT", "")
            stars = clnrevstat.count("criteria_provided")

            record = {
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "clnsig": clnsig,
                "consequences": "|".join(sorted(consequences)),
                "review_stars": stars,
                "variant_id": vid,
            }

            if clnsig in PATHOGENIC_TERMS:
                positive.append(record)
            elif clnsig in BENIGN_TERMS:
                negative_n1.append(record)

    # Save to TSV
    pos_df = pd.DataFrame(positive)
    neg_df = pd.DataFrame(negative_n1)

    pos_path = output_dir / "positive_noncoding.tsv"
    neg_path = output_dir / "negative_N1_benign.tsv"

    pos_df.to_csv(pos_path, sep="\t", index=False)
    neg_df.to_csv(neg_path, sep="\t", index=False)

    print(f"Positive (pathogenic) non-coding variants: {len(pos_df)}")
    print(f"Negative N1 (benign) non-coding variants:  {len(neg_df)}")
    print(f"Saved to {pos_path} and {neg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter ClinVar for non-coding variants")
    parser.add_argument(
        "--vcf",
        default="data/raw/clinvar.vcf.gz",
        help="Path to ClinVar VCF (default: data/raw/clinvar.vcf.gz)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    args = parser.parse_args()
    process_clinvar(args.vcf, args.output_dir)
