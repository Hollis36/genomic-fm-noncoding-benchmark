#!/usr/bin/env python3
"""
Step 4: Annotate each variant with its non-coding region category.

Categories:
  - splice_proximal: within 20bp of splice site
  - utr_5prime / utr_3prime: UTR regions
  - promoter: TSS +/- 2kb
  - enhancer: ENCODE cCRE distal enhancer-like signature
  - deep_intronic: intronic, >50bp from exon boundary
  - intergenic: outside gene body and cCREs

Requires:
  - GENCODE GTF annotation
  - ENCODE cCREs BED file

Produces:
  - Adds 'region_category' column to each variant TSV
"""

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_gencode_features(gtf_path: str) -> dict[str, dict[str, list]]:
    """Parse GENCODE GTF to extract genomic features for region classification.

    Extracts exon boundaries, 5'/3' UTRs, transcription start sites, and gene
    body intervals from a GENCODE annotation file. Intervals are sorted per
    chromosome for efficient lookup.

    Args:
        gtf_path: Path to GENCODE GTF file (.gtf or .gtf.gz).

    Returns:
        Nested dict keyed by feature type ("exons", "utr5", "utr3", "tss",
        "gene_body"), each mapping chromosome names to sorted interval lists.
    """
    print(f"Loading GENCODE annotation from {gtf_path} ...")
    features = {
        "exons": defaultdict(list),       # chrom -> [(start, end)]
        "utr5": defaultdict(list),
        "utr3": defaultdict(list),
        "tss": defaultdict(list),         # chrom -> [pos]
        "gene_body": defaultdict(list),   # chrom -> [(start, end)]
    }

    opener = gzip.open if gtf_path.endswith(".gz") else open

    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            chrom, source, feature, start, end = (
                fields[0], fields[1], fields[2], int(fields[3]), int(fields[4])
            )
            strand = fields[6]

            if feature == "exon":
                features["exons"][chrom].append((start, end))
            elif feature == "five_prime_UTR":
                features["utr5"][chrom].append((start, end))
            elif feature == "three_prime_UTR":
                features["utr3"][chrom].append((start, end))
            elif feature == "transcript":
                tss = start if strand == "+" else end
                features["tss"][chrom].append(tss)
            elif feature == "gene":
                features["gene_body"][chrom].append((start, end))

    # Sort intervals for binary search
    for key in features:
        for chrom in features[key]:
            features[key][chrom].sort()

    print(f"  Loaded exons on {len(features['exons'])} chromosomes")
    return features


def load_encode_ccres(bed_path: str) -> dict[str, list[tuple[int, int]]]:
    """Load ENCODE cCREs and extract distal enhancer-like signatures (dELS).

    Reads an ENCODE candidate cis-Regulatory Elements BED file and filters
    for entries annotated as distal enhancer-like signatures.

    Args:
        bed_path: Path to ENCODE cCREs BED file (.bed or .bed.gz).

    Returns:
        Dict mapping chromosome names to sorted lists of (start, end) intervals.
    """
    print(f"Loading ENCODE cCREs from {bed_path} ...")
    enhancers = defaultdict(list)

    opener = gzip.open if bed_path.endswith(".gz") else open

    with opener(bed_path, "rt") as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 6:
                continue
            chrom = fields[0]
            start, end = int(fields[1]), int(fields[2])
            ccre_type = fields[5] if len(fields) > 5 else ""

            # dELS = distal enhancer-like signature
            if "dELS" in ccre_type or "Enhancer" in ccre_type:
                enhancers[chrom].append((start, end))

    for chrom in enhancers:
        enhancers[chrom].sort()

    print(f"  Loaded enhancers on {len(enhancers)} chromosomes")
    return enhancers


def overlaps(pos: int, intervals: list[tuple[int, int]], window: int = 0) -> bool:
    """Check if position overlaps any interval (with optional window)."""
    import bisect
    # Simple linear scan for now; optimize with interval tree for large datasets
    for start, end in intervals:
        if start - window <= pos <= end + window:
            return True
        if start - window > pos:
            break
    return False


def near_splice_site(pos: int, exons: list[tuple[int, int]], distance: int = 20) -> bool:
    """Check if position is within `distance` bp of an exon boundary."""
    for start, end in exons:
        if abs(pos - start) <= distance or abs(pos - end) <= distance:
            return True
        if start - distance > pos:
            break
    return False


def classify_region(
    chrom: str,
    pos: int,
    gencode: dict[str, dict[str, list]],
    enhancers: dict[str, list[tuple[int, int]]],
) -> str:
    """Classify a variant position into a non-coding region category.

    Priority order: splice_proximal > utr_5prime > utr_3prime > promoter >
    enhancer > deep_intronic > intergenic.

    Args:
        chrom: Chromosome name (e.g., "chr1").
        pos: 1-based genomic position.
        gencode: GENCODE features from load_gencode_features().
        enhancers: ENCODE enhancer intervals from load_encode_ccres().

    Returns:
        Region category string.
    """
    # 1. Splice-proximal
    if near_splice_site(pos, gencode["exons"].get(chrom, []), distance=20):
        return "splice_proximal"

    # 2. 5' UTR
    if overlaps(pos, gencode["utr5"].get(chrom, [])):
        return "utr_5prime"

    # 3. 3' UTR
    if overlaps(pos, gencode["utr3"].get(chrom, [])):
        return "utr_3prime"

    # 4. Promoter (TSS +/- 2kb)
    for tss in gencode["tss"].get(chrom, []):
        if abs(pos - tss) <= 2000:
            return "promoter"

    # 5. Enhancer (ENCODE cCRE-dELS)
    if overlaps(pos, enhancers.get(chrom, [])):
        return "enhancer"

    # 6. Deep intronic (in gene body but not near exon)
    in_gene = overlaps(pos, gencode["gene_body"].get(chrom, []))
    if in_gene:
        return "deep_intronic"

    # 7. Intergenic
    return "intergenic"


def annotate_variants(
    variant_tsv: str,
    gencode: dict[str, dict[str, list]],
    enhancers: dict[str, list[tuple[int, int]]],
    output_path: str,
) -> None:
    """Add region_category column to variant TSV."""
    df = pd.read_csv(variant_tsv, sep="\t")
    print(f"Annotating {len(df)} variants from {variant_tsv} ...")

    categories = []
    for _, row in df.iterrows():
        chrom = str(row["chrom"])
        pos = int(row["pos"])
        cat = classify_region(chrom, pos, gencode, enhancers)
        categories.append(cat)

    df["region_category"] = categories

    # Summary
    print(df["region_category"].value_counts())

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved annotated variants to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate variants with region categories")
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Variant TSV file(s) to annotate",
    )
    parser.add_argument(
        "--gencode-gtf",
        default="data/raw/gencode.v44.annotation.gtf.gz",
    )
    parser.add_argument(
        "--ccre-bed",
        default="data/raw/GRCh38-cCREs.bed.gz",
    )
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    gencode = load_gencode_features(args.gencode_gtf)
    enhancers = load_encode_ccres(args.ccre_bed)

    output_dir = Path(args.output_dir)

    for variant_file in args.variants:
        stem = Path(variant_file).stem
        output_path = str(output_dir / f"{stem}_annotated.tsv")
        annotate_variants(variant_file, gencode, enhancers, output_path)
