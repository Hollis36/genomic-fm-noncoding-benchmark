"""Integration tests for data filtering and annotation pipeline."""

import pytest

from data.scripts.filter_noncoding import (
    BENIGN_TERMS,
    NONCODING_MC,
    PATHOGENIC_TERMS,
    get_molecular_consequences,
    is_noncoding,
    parse_info,
)
from data.scripts.annotate_regions import (
    classify_region,
    near_splice_site,
    overlaps,
)


@pytest.mark.integration
class TestFilterAnnotatePipelineIntegration:
    """Test the data filtering and annotation pipeline components together."""

    def test_parse_info_then_classify_noncoding(self):
        """Parse a VCF INFO string and check if its consequences are non-coding."""
        info_str = (
            "CLNSIG=Pathogenic;"
            "MC=SO:0001627|intron_variant,SO:0001619|non-coding_transcript_variant"
        )
        info = parse_info(info_str)
        consequences = get_molecular_consequences(info)

        assert is_noncoding(consequences)
        assert info["CLNSIG"] == "Pathogenic"
        assert "intron_variant" in consequences
        assert "non-coding_transcript_variant" in consequences

    def test_coding_variant_rejected(self):
        """A purely coding variant should not be classified as non-coding."""
        info_str = "CLNSIG=Pathogenic;MC=SO:0001583|missense_variant"
        info = parse_info(info_str)
        consequences = get_molecular_consequences(info)

        assert not is_noncoding(consequences)

    def test_benign_splice_variant_flow(self):
        """A benign splice variant should be classified as non-coding."""
        info_str = "CLNSIG=Benign;MC=SO:0001630|splice_region_variant"
        info = parse_info(info_str)
        clnsig = info.get("CLNSIG", "")
        consequences = get_molecular_consequences(info)

        assert clnsig in BENIGN_TERMS
        assert is_noncoding(consequences)
        assert "splice_region_variant" in consequences

    def test_region_classification_splice_proximal(self):
        """A position near an exon boundary should be classified as splice_proximal."""
        gencode = {
            "exons": {"chr1": [(1000, 1200), (2000, 2500)]},
            "utr5": {"chr1": []},
            "utr3": {"chr1": []},
            "tss": {"chr1": []},
            "gene_body": {"chr1": [(500, 3000)]},
        }
        enhancers = {"chr1": []}

        result = classify_region("chr1", 1010, gencode, enhancers)
        assert result == "splice_proximal"

    def test_region_classification_promoter(self):
        """A position near a TSS should be classified as promoter."""
        gencode = {
            "exons": {"chr1": [(5000, 5200)]},
            "utr5": {"chr1": []},
            "utr3": {"chr1": []},
            "tss": {"chr1": [10000]},
            "gene_body": {"chr1": [(9000, 15000)]},
        }
        enhancers = {"chr1": []}

        result = classify_region("chr1", 10500, gencode, enhancers)
        assert result == "promoter"

    def test_region_classification_enhancer(self):
        """A position inside an enhancer region should be classified as enhancer."""
        gencode = {
            "exons": {"chr2": [(100000, 100200)]},
            "utr5": {"chr2": []},
            "utr3": {"chr2": []},
            "tss": {"chr2": []},
            "gene_body": {"chr2": []},
        }
        enhancers = {"chr2": [(50000, 51000)]}

        result = classify_region("chr2", 50500, gencode, enhancers)
        assert result == "enhancer"

    def test_region_classification_deep_intronic(self):
        """A position in a gene body far from exons should be deep_intronic."""
        gencode = {
            "exons": {"chr1": [(1000, 1200)]},
            "utr5": {"chr1": []},
            "utr3": {"chr1": []},
            "tss": {"chr1": []},
            "gene_body": {"chr1": [(500, 5000)]},
        }
        enhancers = {"chr1": []}

        result = classify_region("chr1", 3000, gencode, enhancers)
        assert result == "deep_intronic"

    def test_region_classification_intergenic(self):
        """A position outside all annotations should be intergenic."""
        gencode = {
            "exons": {"chr1": [(1000, 1200)]},
            "utr5": {"chr1": []},
            "utr3": {"chr1": []},
            "tss": {"chr1": []},
            "gene_body": {"chr1": [(500, 2000)]},
        }
        enhancers = {"chr1": []}

        result = classify_region("chr1", 99999, gencode, enhancers)
        assert result == "intergenic"

    def test_region_classification_utr5(self):
        """A position in a 5' UTR should be classified as utr_5prime."""
        gencode = {
            "exons": {"chr1": [(5000, 5200)]},
            "utr5": {"chr1": [(3000, 3500)]},
            "utr3": {"chr1": []},
            "tss": {"chr1": []},
            "gene_body": {"chr1": [(2000, 6000)]},
        }
        enhancers = {"chr1": []}

        result = classify_region("chr1", 3250, gencode, enhancers)
        assert result == "utr_5prime"

    def test_overlaps_with_window(self):
        """overlaps should respect the window parameter."""
        intervals = [(1000, 2000)]

        assert overlaps(999, intervals, window=0) is False
        assert overlaps(999, intervals, window=5) is True
        assert overlaps(2001, intervals, window=0) is False
        assert overlaps(2001, intervals, window=5) is True

    def test_near_splice_site_boundary(self):
        """near_splice_site should detect positions at exactly the distance threshold."""
        exons = [(1000, 2000)]

        assert near_splice_site(980, exons, distance=20) is True
        assert near_splice_site(979, exons, distance=20) is False
        assert near_splice_site(2020, exons, distance=20) is True
        assert near_splice_site(2021, exons, distance=20) is False

    def test_full_variant_classification_pipeline(self):
        """Run a variant through parse -> classify flow end to end."""
        info_str = "CLNSIG=Pathogenic;MC=SO:0001627|intron_variant"
        info = parse_info(info_str)
        consequences = get_molecular_consequences(info)

        assert is_noncoding(consequences)
        assert info["CLNSIG"] in PATHOGENIC_TERMS

        gencode = {
            "exons": {"chr7": [(50000, 50200)]},
            "utr5": {"chr7": []},
            "utr3": {"chr7": []},
            "tss": {"chr7": []},
            "gene_body": {"chr7": [(49000, 55000)]},
        }
        enhancers = {"chr7": []}

        region = classify_region("chr7", 52000, gencode, enhancers)
        assert region == "deep_intronic"
