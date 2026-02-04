#!/usr/bin/env python3
"""
Create synthetic test data for quick model testing.
This allows testing the pipeline without downloading large ClinVar files.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_dna_sequence(length=1024):
    """Generate random DNA sequence."""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases, size=length))

def create_test_variants(n_samples=100):
    """Create synthetic variant dataset."""

    chromosomes = ['chr1', 'chr2', 'chr3', 'chr7', 'chr17']
    regions = ['promoter', 'enhancer', 'utr_5prime', 'utr_3prime',
               'splice_proximal', 'deep_intronic', 'intergenic']

    variants = []

    for i in range(n_samples):
        chrom = np.random.choice(chromosomes)
        pos = np.random.randint(10000, 1000000)
        ref = np.random.choice(['A', 'C', 'G', 'T'])

        # Ensure alt is different from ref
        alt_options = [b for b in ['A', 'C', 'G', 'T'] if b != ref]
        alt = np.random.choice(alt_options)

        region = np.random.choice(regions)

        variants.append({
            'chrom': chrom,
            'pos': pos,
            'ref': ref,
            'alt': alt,
            'variant_id': f'test_variant_{i}',
            'region_category': region,
            'clnsig': 'Pathogenic' if i < n_samples // 2 else 'Benign',
        })

    return pd.DataFrame(variants)

def create_sequences(df, context_size=1024):
    """Create sequences for variants."""
    sequences = []

    for _, row in df.iterrows():
        # Generate reference sequence
        ref_seq = generate_dna_sequence(context_size)

        # Create alternate sequence (change middle base)
        mid = context_size // 2
        alt_seq = ref_seq[:mid] + row['alt'] + ref_seq[mid+1:]

        sequences.append({
            'chrom': row['chrom'],
            'pos': row['pos'],
            'ref_seq': ref_seq,
            'alt_seq': alt_seq,
        })

    return pd.DataFrame(sequences)

def main():
    print("=" * 60)
    print("Creating Test Dataset")
    print("=" * 60)

    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Create positive (pathogenic) variants
    print("\n📝 Creating positive variants...")
    positive_df = create_test_variants(n_samples=50)
    positive_df = positive_df[positive_df['clnsig'] == 'Pathogenic']
    positive_path = "data/processed/positive_noncoding_annotated.tsv"
    positive_df.to_csv(positive_path, sep='\t', index=False)
    print(f"✅ Created {len(positive_df)} positive variants")
    print(f"   Saved to: {positive_path}")

    # Create negative sets
    print("\n📝 Creating negative set N1 (benign)...")
    negative_n1_df = create_test_variants(n_samples=50)
    negative_n1_df = negative_n1_df[negative_n1_df['clnsig'] == 'Benign']
    n1_path = "data/processed/negative_N1_benign_annotated.tsv"
    negative_n1_df.to_csv(n1_path, sep='\t', index=False)
    print(f"✅ Created {len(negative_n1_df)} N1 variants")
    print(f"   Saved to: {n1_path}")

    print("\n📝 Creating negative set N3 (matched random)...")
    negative_n3_df = create_test_variants(n_samples=50)
    negative_n3_df = negative_n3_df[negative_n3_df['clnsig'] == 'Benign']
    negative_n3_df['source'] = 'matched_random'
    n3_path = "data/processed/negative_N3_matched_random_annotated.tsv"
    negative_n3_df.to_csv(n3_path, sep='\t', index=False)
    print(f"✅ Created {len(negative_n3_df)} N3 variants")
    print(f"   Saved to: {n3_path}")

    # Create sequences
    print("\n📝 Creating DNA sequences (1024bp context)...")
    all_variants = pd.concat([positive_df, negative_n1_df, negative_n3_df],
                              ignore_index=True)
    sequences_df = create_sequences(all_variants, context_size=1024)

    seq_path = "data/processed/sequences_positive_noncoding_annotated_1024bp.parquet"
    sequences_df.to_parquet(seq_path, index=False)
    print(f"✅ Created {len(sequences_df)} sequences")
    print(f"   Saved to: {seq_path}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Test Dataset Created Successfully!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Positive variants: {len(positive_df)}")
    print(f"  Negative N1:       {len(negative_n1_df)}")
    print(f"  Negative N3:       {len(negative_n3_df)}")
    print(f"  Total variants:    {len(all_variants)}")
    print(f"\nRegion distribution:")
    print(all_variants['region_category'].value_counts().to_string())

    print("\n💡 Note: This is synthetic test data for quick validation.")
    print("   For real experiments, download actual ClinVar data.")
    print("\n🚀 Ready to test! Run:")
    print("   python scripts/run_experiment_m4pro.py --mode zero_shot --models dnabert2 --negative-sets N1")
    print("=" * 60)

if __name__ == "__main__":
    main()
