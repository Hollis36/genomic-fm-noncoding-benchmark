#!/bin/bash
# ============================================
# Step 1: Download ClinVar VCF and Reference Genome
# ============================================
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${DATA_DIR}/raw"
REF_DIR="${DATA_DIR}/reference"

mkdir -p "${RAW_DIR}" "${REF_DIR}"

echo "=== Downloading ClinVar VCF (GRCh38) ==="
wget -nc -P "${RAW_DIR}" \
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
wget -nc -P "${RAW_DIR}" \
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi

echo "=== Downloading GRCh38 Reference Genome ==="
if [ ! -f "${REF_DIR}/GRCh38.fa" ]; then
    wget -nc -P "${REF_DIR}" \
        https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    echo "Decompressing reference genome..."
    gunzip -k "${REF_DIR}/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    mv "${REF_DIR}/Homo_sapiens.GRCh38.dna.primary_assembly.fa" "${REF_DIR}/GRCh38.fa"
    echo "Indexing reference genome..."
    samtools faidx "${REF_DIR}/GRCh38.fa"
fi

echo "=== Downloading GENCODE annotation ==="
wget -nc -P "${RAW_DIR}" \
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz

echo "=== Downloading ENCODE cCREs ==="
wget -nc -P "${RAW_DIR}" \
    "https://api.wenglab.org/screen_v13/fdownloads/GRCh38-cCREs.bed.gz"

echo "=== All downloads complete ==="
