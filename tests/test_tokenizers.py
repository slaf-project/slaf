"""
Tests for SLAFTokenizer with simplified fragment-based processing.

This module tests the simplified fragment-based tokenization approach that uses
the FragmentLoader in datasets.py instead of complex tokenizer-specific processors.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from slaf.core.slaf import SLAFArray
from slaf.ml.tokenizers import SLAFTokenizer


class TestSLAFTokenizerFragmentBased:
    """Test the SLAFTokenizer with simplified fragment-based processing."""

    def test_tokenizer_initialization_with_fragment_processing(self):
        """Test SLAFTokenizer initialization with fragment processing enabled."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        # Test initialization with fragment processing
        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
            use_fragment_processing=True,
        )

        assert tokenizer.use_fragment_processing is True
        assert len(tokenizer.gene_vocab) == 3
        assert tokenizer.special_tokens["CLS"] == 0
        assert tokenizer.special_tokens["SEP"] == 1
        assert tokenizer.special_tokens["PAD"] == 2
        assert tokenizer.special_tokens["UNK"] == 3

    def test_tokenizer_initialization_without_fragment_processing(self):
        """Test SLAFTokenizer initialization with fragment processing disabled."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        # Test initialization without fragment processing
        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
            use_fragment_processing=False,
        )

        assert tokenizer.use_fragment_processing is False
        assert len(tokenizer.gene_vocab) == 3

    def test_deprecated_fragment_based_methods(self):
        """Test that deprecated fragment-based methods raise NotImplementedError."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
            use_fragment_processing=True,
        )

        # Test that deprecated methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            tokenizer.tokenize_scgpt_fragment_based((0, 32))

        with pytest.raises(NotImplementedError):
            tokenizer.tokenize_geneformer_fragment_based((0, 32))

    def test_convert_gene_sequence_to_scgpt_tokens(self):
        """Test conversion of gene sequences to scGPT tokens."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
            n_expression_bins=5,
        )

        # Test gene sequence conversion - use integer gene IDs that will be converted to strings
        # The method converts int to str for vocabulary lookup, so we need to use integers
        # that will match the vocabulary keys when converted to strings
        gene_sequence = [1, 2, 3]  # These will be converted to "1", "2", "3" for lookup
        expr_sequence = [1.0, 2.0, 3.0]
        max_genes = 1024

        tokens = tokenizer._convert_gene_sequence_to_scgpt_tokens(
            gene_sequence, expr_sequence, max_genes
        )

        # Check token format: [CLS] gene1 expr1 gene2 expr2 ... [SEP] [PAD...]
        assert tokens[0] == tokenizer.special_tokens["CLS"]  # Start token
        assert tokens[-1] == tokenizer.special_tokens["PAD"]  # End with padding
        assert len(tokens) == max_genes * 2 + 2  # CLS + (gene,expr)*max_genes + SEP

        # Check that gene tokens are in vocabulary (they should be UNK tokens since "1", "2", "3" aren't in vocab)
        gene_tokens = tokens[1::2]  # Every other token starting from index 1
        for token in gene_tokens[:3]:  # First 3 gene tokens
            # Since "1", "2", "3" aren't in the vocabulary, they should be UNK tokens
            assert token == tokenizer.special_tokens["UNK"]

    def test_convert_gene_sequence_to_geneformer_tokens(self):
        """Test conversion of gene sequences to Geneformer tokens."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
        )

        # Test gene sequence conversion - use integer gene IDs that will be converted to strings
        gene_sequence = [1, 2, 3]  # These will be converted to "1", "2", "3" for lookup
        max_genes = 2048

        tokens = tokenizer._convert_gene_sequence_to_geneformer_tokens(
            gene_sequence, max_genes
        )

        # Check token format: ranked gene tokens [PAD...]
        assert len(tokens) == max_genes
        assert tokens[-1] == tokenizer.special_tokens["PAD"]  # End with padding

        # Check that gene tokens are in vocabulary (they should be UNK tokens since "1", "2", "3" aren't in vocab)
        gene_tokens = [t for t in tokens if t != tokenizer.special_tokens["PAD"]]
        for token in gene_tokens[:3]:  # First 3 gene tokens
            # Since "1", "2", "3" aren't in the vocabulary, they should be UNK tokens
            assert token == tokenizer.special_tokens["UNK"]

    def test_expression_binning_edge_cases(self):
        """Test expression binning with edge cases"""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
            n_expression_bins=5,
        )

        # Test zero and negative values
        assert tokenizer._expression_to_bin(0.0) == tokenizer.special_tokens["PAD"]
        assert tokenizer._expression_to_bin(-1.0) == tokenizer.special_tokens["PAD"]

        # Test vectorized binning

        expr_values = np.array([0.0, 1.0, 10.0, -1.0])
        bins = tokenizer._expression_to_bin_vectorized(expr_values)
        assert bins[0] == tokenizer.special_tokens["PAD"]  # 0.0
        assert bins[1] >= tokenizer.expr_bin_start  # 1.0
        assert bins[2] >= tokenizer.expr_bin_start  # 10.0
        assert bins[3] == tokenizer.special_tokens["PAD"]  # -1.0

    def test_vocabulary_edge_cases(self):
        """Test vocabulary building with edge cases"""
        # Mock SLAFArray with empty gene list
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index([])  # Empty gene list
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
        )

        # Should handle empty vocabulary gracefully
        assert len(tokenizer.gene_vocab) == 0
        assert tokenizer.special_tokens["UNK"] == 3

    def test_token_conversion_edge_cases(self):
        """Test token conversion with edge cases"""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
        )

        # Test with empty gene sequence
        empty_sequence = []
        tokens = tokenizer._convert_gene_sequence_to_geneformer_tokens(
            empty_sequence, max_genes=10
        )
        assert len(tokens) == 10
        assert all(token == tokenizer.special_tokens["PAD"] for token in tokens)

        # Test with gene IDs not in vocabulary
        unknown_genes = [999, 1000, 1001]  # Not in vocabulary
        tokens = tokenizer._convert_gene_sequence_to_geneformer_tokens(
            unknown_genes, max_genes=10
        )
        assert len(tokens) == 10
        # Should be UNK tokens for unknown genes
        assert tokens[0] == tokenizer.special_tokens["UNK"]
        assert tokens[1] == tokenizer.special_tokens["UNK"]
        assert tokens[2] == tokenizer.special_tokens["UNK"]


class TestBackwardCompatibility:
    """Test backward compatibility with existing tokenizer methods."""

    def test_existing_methods_still_work(self):
        """Test that existing SQL-based methods still work."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["GENE_001", "GENE_002", "GENE_003"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            vocab_size=1000,
            use_fragment_processing=False,  # Use SQL-based approach
        )

        # Test that basic tokenizer functionality still works
        assert len(tokenizer.gene_vocab) == 3
        assert tokenizer.special_tokens["CLS"] == 0
        assert tokenizer.special_tokens["SEP"] == 1
        assert tokenizer.special_tokens["PAD"] == 2
        assert tokenizer.special_tokens["UNK"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
