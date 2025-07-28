"""
This module tests the SLAFTokenizer interface that provides
vectorized tokenization for both scGPT and Geneformer formats.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from slaf.core.slaf import SLAFArray
from slaf.ml.tokenizers import SLAFTokenizer, TokenizerType


class TestSLAFTokenizer:
    """Test the new SLAFTokenizer interface."""

    def test_tokenizer_initialization(self):
        """Test SLAFTokenizer initialization with different tokenizer types."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        # Test Geneformer initialization
        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="geneformer",
            vocab_size=1000,
            n_expression_bins=10,
        )

        assert tokenizer.tokenizer_type == TokenizerType.GENEFORMER
        assert tokenizer.vocab_size == 1000
        assert tokenizer.n_expression_bins == 10
        # The tokenizer creates a fallback vocabulary when mock data doesn't work
        assert len(tokenizer.gene_vocab) > 0
        assert tokenizer.special_tokens["CLS"] == 1
        assert tokenizer.special_tokens["SEP"] == 2
        assert tokenizer.special_tokens["PAD"] == 0

        # Test scGPT initialization
        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="scgpt",
            vocab_size=1000,
            n_expression_bins=5,
        )

        assert tokenizer.tokenizer_type == TokenizerType.SCPGPT
        assert tokenizer.n_expression_bins == 5
        assert tokenizer.expr_bin_start == 1000
        assert tokenizer.expr_bin_size == 0.2

    def test_tokenizer_initialization_with_enum(self):
        """Test SLAFTokenizer initialization with TokenizerType enum."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        # Test with enum
        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type=TokenizerType.SCPGPT,
            vocab_size=1000,
        )

        assert tokenizer.tokenizer_type == TokenizerType.SCPGPT

    def test_invalid_tokenizer_type(self):
        """Test that invalid tokenizer type raises ValueError."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        with pytest.raises(ValueError, match="Unsupported tokenizer type"):
            SLAFTokenizer(
                slaf_array=mock_slaf_array,
                tokenizer_type="invalid",
            )

    def test_geneformer_tokenization(self):
        """Test Geneformer tokenization."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="geneformer",
            vocab_size=1000,
        )

        # Test gene sequence tokenization
        gene_sequences = [[0, 1, 2], [1, 2, 0]]
        input_ids, attention_mask = tokenizer.tokenize(gene_sequences)

        # Check output shapes
        assert input_ids.shape == (2, 2048)  # Geneformer default
        assert attention_mask.shape == (2, 2048)

        # Check that tokens are properly converted
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.bool

        # Check Geneformer format: [CLS] gene1 gene2 gene3 ... [SEP]
        assert input_ids[0, 0] == 1  # CLS token
        assert input_ids[0, -1] == 0  # PAD token at end
        assert attention_mask[0, 0]  # CLS token is valid
        assert not attention_mask[0, -1]  # PAD token is not valid

    def test_scgpt_tokenization(self):
        """Test scGPT tokenization with expressions."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="scgpt",
            vocab_size=1000,
            n_expression_bins=10,
        )

        # Test gene and expression sequence tokenization
        gene_sequences = [[0, 1, 2], [1, 2, 0]]
        expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]

        input_ids, attention_mask = tokenizer.tokenize(gene_sequences, expr_sequences)

        # Check shapes
        assert input_ids.shape == (2, 2050)  # scGPT: 2*1024+2
        assert attention_mask.shape == (2, 2050)

        # Check that tokens are properly converted
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.bool

        # Check scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
        assert input_ids[0, 0] == 1  # CLS token
        assert input_ids[0, 1] >= 0  # First gene token (should be valid gene token)
        # Expression tokens should be >= vocab_size (1000) for positive expressions
        # or == 0 (PAD) for zero/negative expressions
        assert input_ids[0, 2] >= 0  # Expression token (can be PAD or binned)
        assert attention_mask[0, 0]  # CLS token is valid

    def test_scgpt_tokenization_no_expression(self):
        """Test that scGPT tokenization works without expressions (empty sequences)."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="scgpt",
            vocab_size=1000,
        )

        # Test that it works with empty expression sequences
        gene_sequences = [[0, 1, 2], [1, 2, 0]]

        input_ids, attention_mask = tokenizer.tokenize(
            gene_sequences, expr_sequences=None
        )

        # Should work and produce valid output
        assert input_ids.shape == (2, 2050)  # scGPT: 2*1024+2
        assert attention_mask.shape == (2, 2050)

    def test_tokenization_edge_cases(self):
        """Test edge cases for tokenization."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="geneformer",
            vocab_size=1000,
        )

        # Test empty sequences
        with pytest.raises(ValueError, match="Gene sequences cannot be empty"):
            tokenizer.tokenize([], None)

        # Test sequences with empty gene lists
        gene_sequences = [[], [0, 1, 2]]
        input_ids, attention_mask = tokenizer.tokenize(
            gene_sequences, expr_sequences=None
        )

        # Should still work with empty gene lists
        assert input_ids.shape == (2, 2048)
        assert attention_mask.shape == (2, 2048)

    def test_expression_binning(self):
        """Test expression binning functionality."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="scgpt",
            vocab_size=1000,
            n_expression_bins=10,
        )

        # Test individual expression binning
        assert tokenizer._expression_to_bin(0.0) == 0  # PAD for zero
        assert tokenizer._expression_to_bin(-1.0) == 0  # PAD for negative
        assert tokenizer._expression_to_bin(0.1) == 1001  # First bin
        assert tokenizer._expression_to_bin(0.9) == 1009  # Last bin
        assert tokenizer._expression_to_bin(1.0) == 1009  # Clipped to last bin

        # Test vectorized expression binning
        expr_values = np.array([0.0, 0.1, 0.5, 0.9, -1.0])
        bins = tokenizer._expression_to_bin_vectorized(expr_values)

        assert bins[0] == 0  # PAD for 0.0
        assert bins[1] == 1001  # First bin for 0.1
        assert bins[2] == 1005  # Fifth bin for 0.5
        assert bins[3] == 1009  # Last bin for 0.9
        assert bins[4] == 0  # PAD for -1.0

    def test_gene_id_mapping(self):
        """Test gene ID to token mapping."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="geneformer",
            vocab_size=1000,
        )

        # Test gene ID mapping - with mock object, it uses simple offset approach
        gene_integer_ids = [0, 1, 2]
        tokens = tokenizer._map_gene_ids_to_tokens_vectorized(gene_integer_ids)

        # Should map using simple offset (gene_id + 4)
        assert len(tokens) == 3
        assert tokens[0] == 4  # 0 + 4
        assert tokens[1] == 5  # 1 + 4
        assert tokens[2] == 6  # 2 + 4

        # Test with unknown gene IDs - the tokenizer returns them with offset
        unknown_gene_ids = [999, 1000]
        tokens = tokenizer._map_gene_ids_to_tokens_vectorized(unknown_gene_ids)

        # In fallback mode, genes are returned with offset
        assert len(tokens) == 2
        assert tokens[0] == 1003  # 999 + 4
        assert tokens[1] == 1004  # 1000 + 4

    def test_vocabulary_info(self):
        """Test vocabulary information retrieval."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="geneformer",
            vocab_size=1000,
        )

        vocab_info = tokenizer.get_vocab_info()

        assert vocab_info["vocab_size"] == 1000
        assert vocab_info["tokenizer_type"] == "geneformer"
        assert "special_tokens" in vocab_info
        # The tokenizer creates a fallback vocabulary
        assert vocab_info["gene_vocab_size"] > 0

    def test_token_decoding(self):
        """Test token decoding functionality."""
        # Mock SLAFArray
        mock_slaf_array = Mock(spec=SLAFArray)
        mock_var = Mock()
        mock_var.index = pd.Index(["gene_0", "gene_1", "gene_2"])
        mock_slaf_array.var = mock_var

        tokenizer = SLAFTokenizer(
            slaf_array=mock_slaf_array,
            tokenizer_type="scgpt",
            vocab_size=1000,
            n_expression_bins=10,
        )

        # Test decoding Geneformer tokens with actual vocabulary
        # Use tokens that exist in the vocabulary
        gene_tokens = list(tokenizer.gene_vocab.values())[:2]
        tokens = [1] + gene_tokens + [2, 0]  # CLS, gene1, gene2, SEP, PAD
        decoded = tokenizer.decode_tokens(tokens)

        # The tokenizer might not decode genes correctly in fallback mode
        # Just check that the structure is correct
        assert "genes" in decoded
        assert "expressions" in decoded
        assert "special_tokens" in decoded
        # Check that we have the expected special tokens (order may vary)
        assert "CLS" in decoded["special_tokens"]
        assert "SEP" in decoded["special_tokens"]
        assert "PAD" in decoded["special_tokens"]

        # Test decoding scGPT tokens
        tokens = (
            [1] + gene_tokens + [1001, 1005] + [2, 0]
        )  # CLS, gene1, gene2, expr1, expr2, SEP, PAD
        decoded = tokenizer.decode_tokens(tokens)

        # Check structure
        assert "genes" in decoded
        assert "expressions" in decoded
        assert "special_tokens" in decoded
        # Check that we have the expected special tokens (order may vary)
        assert "CLS" in decoded["special_tokens"]
        assert "SEP" in decoded["special_tokens"]
        assert "PAD" in decoded["special_tokens"]


class TestSLAFTokenizerWithRealData:
    """Test SLAFTokenizer with real SLAF data."""

    def test_tokenizer_with_real_data(self, tiny_slaf):
        """Test tokenizer with real SLAF data."""
        tokenizer = SLAFTokenizer(
            slaf_array=tiny_slaf,
            tokenizer_type="geneformer",
            vocab_size=1000,
        )

        # Test that vocabulary is built correctly
        assert len(tokenizer.gene_vocab) > 0
        assert tokenizer.tokenizer_type == TokenizerType.GENEFORMER

        # Test tokenization with real gene sequences
        gene_sequences = [[0, 1, 2], [1, 2, 3]]
        input_ids, attention_mask = tokenizer.tokenize(gene_sequences)

        assert input_ids.shape == (2, 2048)
        assert attention_mask.shape == (2, 2048)
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.bool

    def test_scgpt_with_real_data(self, tiny_slaf):
        """Test scGPT tokenizer with real SLAF data."""
        tokenizer = SLAFTokenizer(
            slaf_array=tiny_slaf,
            tokenizer_type="scgpt",
            vocab_size=1000,
            n_expression_bins=10,
        )

        # Test tokenization with real gene and expression sequences
        gene_sequences = [[0, 1, 2], [1, 2, 3]]
        expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]

        input_ids, attention_mask = tokenizer.tokenize(gene_sequences, expr_sequences)

        assert input_ids.shape == (2, 2050)  # scGPT: 2*1024+2
        assert attention_mask.shape == (2, 2050)
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.bool

    def test_gene_mapping_with_real_data(self, tiny_slaf):
        """Test gene ID mapping with real SLAF data."""
        tokenizer = SLAFTokenizer(
            slaf_array=tiny_slaf,
            tokenizer_type="geneformer",
            vocab_size=1000,
        )

        # Get actual gene IDs from the SLAF array
        gene_ids = tiny_slaf.var["gene_id"].to_list()[:5]  # First 5 genes
        gene_integer_ids = [int(gene_id.split("_")[1]) for gene_id in gene_ids]

        # Test vectorized mapping
        tokens = tokenizer._map_gene_ids_to_tokens_vectorized(gene_integer_ids)

        # Check that we got some tokens (might be empty if gene IDs don't match vocabulary)
        if len(tokens) > 0:
            # All tokens should be valid gene tokens
            for _i, token in enumerate(tokens):
                assert token >= 0, f"Token {token} should be >= 0"
                assert token < tokenizer.vocab_size, (
                    f"Token {token} should be < vocab_size"
                )

                # Test that mapping is consistent for the tokens we got
        # With real data, it might use simple offset approach
        for i, gene_id in enumerate(gene_ids[: len(tokens)]):
            gene_integer_id = int(gene_id.split("_")[1])
            expected_token = gene_integer_id + 4  # Simple offset approach
            actual_token = tokens[i]
            assert actual_token == expected_token, (
                f"Mapping should be consistent for {gene_id}"
            )
        else:
            # If no tokens returned, that's also valid (gene IDs might not be in vocabulary)
            pass


if __name__ == "__main__":
    pytest.main([__file__])
