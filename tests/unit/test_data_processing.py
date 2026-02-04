"""Unit tests for data processing functions."""

import pytest

from data.scripts.filter_noncoding import (
    BENIGN_TERMS,
    NONCODING_MC,
    PATHOGENIC_TERMS,
    get_molecular_consequences,
    is_noncoding,
    parse_info,
)


class TestParseInfo:
    """Test suite for VCF INFO field parsing."""

    def test_parse_simple_info(self):
        """Test parsing simple INFO field."""
        info_str = "AF=0.5;DP=100;QUAL=30"
        result = parse_info(info_str)

        assert result["AF"] == "0.5"
        assert result["DP"] == "100"
        assert result["QUAL"] == "30"

    def test_parse_info_with_flags(self):
        """Test parsing INFO with boolean flags."""
        info_str = "SOMATIC;AF=0.3;DB"
        result = parse_info(info_str)

        assert result["SOMATIC"] is True
        assert result["AF"] == "0.3"
        assert result["DB"] is True

    def test_parse_empty_info(self):
        """Test parsing empty INFO field."""
        info_str = ""
        result = parse_info(info_str)

        assert result == {}

    def test_parse_info_with_semicolons_in_value(self):
        """Test parsing INFO with complex values."""
        info_str = "CLNSIG=Pathogenic;MC=SO:0001583|missense_variant"
        result = parse_info(info_str)

        assert result["CLNSIG"] == "Pathogenic"
        assert "MC" in result


class TestGetMolecularConsequences:
    """Test suite for molecular consequence extraction."""

    def test_extract_single_consequence(self):
        """Test extracting a single molecular consequence."""
        info = {"MC": "SO:0001583|missense_variant"}
        result = get_molecular_consequences(info)

        assert "missense_variant" in result

    def test_extract_multiple_consequences(self):
        """Test extracting multiple molecular consequences."""
        info = {"MC": "SO:0001627|intron_variant,SO:0001792|non-coding_transcript_variant"}
        result = get_molecular_consequences(info)

        assert "intron_variant" in result
        assert "non-coding_transcript_variant" in result

    def test_missing_mc_field(self):
        """Test handling missing MC field."""
        info = {}
        result = get_molecular_consequences(info)

        assert len(result) == 0

    def test_malformed_mc_field(self):
        """Test handling malformed MC field."""
        info = {"MC": "invalid_format"}
        result = get_molecular_consequences(info)

        assert len(result) == 0


class TestIsNoncoding:
    """Test suite for non-coding variant classification."""

    def test_noncoding_variant(self):
        """Test identification of non-coding variant."""
        consequences = {"intron_variant", "synonymous_variant"}
        assert is_noncoding(consequences) is True

    def test_coding_variant(self):
        """Test identification of coding variant."""
        consequences = {"missense_variant", "stop_gained"}
        assert is_noncoding(consequences) is False

    def test_mixed_consequences(self):
        """Test mixed coding/non-coding consequences."""
        consequences = {"intron_variant", "missense_variant"}
        # Should be True if any non-coding consequence is present
        assert is_noncoding(consequences) is True

    def test_empty_consequences(self):
        """Test empty consequence set."""
        consequences = set()
        assert is_noncoding(consequences) is False


class TestClinVarTerms:
    """Test suite for ClinVar term definitions."""

    def test_pathogenic_terms_defined(self):
        """Test pathogenic terms are properly defined."""
        assert len(PATHOGENIC_TERMS) > 0
        assert "Pathogenic" in PATHOGENIC_TERMS
        assert "Likely_pathogenic" in PATHOGENIC_TERMS

    def test_benign_terms_defined(self):
        """Test benign terms are properly defined."""
        assert len(BENIGN_TERMS) > 0
        assert "Benign" in BENIGN_TERMS
        assert "Likely_benign" in BENIGN_TERMS

    def test_no_overlap_pathogenic_benign(self):
        """Test no overlap between pathogenic and benign terms."""
        overlap = PATHOGENIC_TERMS & BENIGN_TERMS
        assert len(overlap) == 0

    def test_noncoding_mc_defined(self):
        """Test non-coding molecular consequences are defined."""
        assert len(NONCODING_MC) > 0
        assert "intron_variant" in NONCODING_MC
        assert "intergenic_variant" in NONCODING_MC
