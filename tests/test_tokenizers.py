from slaf.ml.tokenizers import SLAFTokenizer


class TestSLAFTokenizer:
    """Test suite for SLAFTokenizer class"""

    def test_tokenizer_initialization(self, tiny_slaf):
        """Test SLAFTokenizer initialization"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        # Check basic attributes
        assert tokenizer.slaf_array is tiny_slaf
        assert tokenizer.vocab_size == 50000
        assert tokenizer.n_expression_bins == 10
        assert tokenizer.chunk_size == 2048

        # Check special tokens
        assert tokenizer.special_tokens["CLS"] == 0
        assert tokenizer.special_tokens["SEP"] == 1
        assert tokenizer.special_tokens["PAD"] == 2
        assert tokenizer.special_tokens["UNK"] == 3

        # Check expression bin start
        assert tokenizer.expr_bin_start == 4  # After 4 special tokens

        # Check gene vocabulary
        assert len(tokenizer.gene_vocab) > 0
        assert len(tokenizer.gene_vocab) <= tokenizer.vocab_size

    def test_tokenizer_initialization_custom_params(self, tiny_slaf):
        """Test SLAFTokenizer initialization with custom parameters"""
        tokenizer = SLAFTokenizer(
            tiny_slaf,
            vocab_size=1000,
            n_expression_bins=5,
            chunk_size=512,
        )

        assert tokenizer.vocab_size == 1000
        assert tokenizer.n_expression_bins == 5
        assert tokenizer.chunk_size == 512
        assert tokenizer.expr_bin_start == 4

    def test_gene_vocabulary_building(self, tiny_slaf):
        """Test gene vocabulary building"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Check vocabulary size is limited
        assert len(tokenizer.gene_vocab) <= 10

        # Check token IDs are sequential and start after special tokens + expression bins
        expected_start = 4 + 10  # special tokens + expression bins
        for i, (_, token_id) in enumerate(tokenizer.gene_vocab.items()):
            assert token_id == expected_start + i

        # Check reverse mapping
        for _, token_id in tokenizer.gene_vocab.items():
            assert (
                tokenizer.token_to_gene[token_id] == tokenizer.token_to_gene[token_id]
            )

    def test_expression_to_bin_conversion(self, tiny_slaf):
        """Test expression value to bin conversion"""
        tokenizer = SLAFTokenizer(tiny_slaf, n_expression_bins=5)

        # Test zero expression
        assert tokenizer._expression_to_bin(0.0) == tokenizer.special_tokens["PAD"]

        # Test positive expressions
        assert tokenizer._expression_to_bin(1.0) >= tokenizer.expr_bin_start
        assert tokenizer._expression_to_bin(10.0) >= tokenizer.expr_bin_start

        # Test binning logic
        bins = []
        for expr in [0.1, 1.0, 5.0, 10.0, 50.0]:
            bin_id = tokenizer._expression_to_bin(expr)
            if bin_id != tokenizer.special_tokens["PAD"]:
                bins.append(bin_id - tokenizer.expr_bin_start)

        # Should have different bins for different expression levels
        assert len(set(bins)) > 1

    def test_chunk_range_splitting(self, tiny_slaf):
        """Test chunk range splitting"""
        tokenizer = SLAFTokenizer(tiny_slaf, chunk_size=3)

        # Test small range
        chunks = tokenizer._chunk_range(0, 5)
        assert chunks == [(0, 3), (3, 5)]

        # Test exact chunk size
        chunks = tokenizer._chunk_range(0, 3)
        assert chunks == [(0, 3)]

        # Test empty range
        chunks = tokenizer._chunk_range(5, 5)
        assert chunks == []

    def test_scgpt_tokenization_basic(self, tiny_slaf):
        """Test basic scGPT tokenization"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize a small range
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 2), max_genes=5)

        assert len(tokens) == 2  # 2 cells
        assert all(isinstance(seq, list) for seq in tokens)

        # Check scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
        for seq in tokens:
            assert seq[0] == tokenizer.special_tokens["CLS"]  # Start with CLS
            assert seq[-1] == tokenizer.special_tokens["SEP"]  # End with SEP
            assert len(seq) >= 2  # At least CLS and SEP

    def test_scgpt_tokenization_with_sql_binning(self, tiny_slaf):
        """Test scGPT tokenization with SQL binning"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize with SQL binning
        tokens = tokenizer.tokenize_scgpt(
            cell_integer_id_range=(0, 2), max_genes=5, use_sql_binning=True
        )

        assert len(tokens) == 2
        assert all(isinstance(seq, list) for seq in tokens)

        # Check format
        for seq in tokens:
            assert seq[0] == tokenizer.special_tokens["CLS"]
            assert seq[-1] == tokenizer.special_tokens["SEP"]

    def test_geneformer_tokenization_basic(self, tiny_slaf):
        """Test basic Geneformer tokenization"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize a small range
        tokens = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 2), max_genes=5
        )

        assert len(tokens) == 2  # 2 cells
        assert all(isinstance(seq, list) for seq in tokens)

        # Check Geneformer format: ranked gene tokens (no CLS/SEP)
        for seq in tokens:
            assert len(seq) <= 5  # Max genes limit
            # Should not have CLS/SEP tokens in Geneformer format
            assert tokenizer.special_tokens["CLS"] not in seq
            assert tokenizer.special_tokens["SEP"] not in seq

    def test_geneformer_tokenization_with_percentile(self, tiny_slaf):
        """Test Geneformer tokenization with percentile filtering"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize with percentile filter
        tokens = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 2), max_genes=5, min_percentile=10
        )

        assert len(tokens) == 2
        assert all(isinstance(seq, list) for seq in tokens)

        # Check format
        for seq in tokens:
            assert len(seq) <= 5

    def test_tokenization_edge_cases(self, tiny_slaf):
        """Test tokenization edge cases"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Test empty range
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 0), max_genes=5)
        assert tokens == []

        tokens = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 0), max_genes=5
        )
        assert tokens == []

        # Test single cell
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 1), max_genes=5)
        assert len(tokens) == 1

        tokens = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 1), max_genes=5
        )
        assert len(tokens) == 1

    def test_vocabulary_info(self, tiny_slaf):
        """Test vocabulary information retrieval"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10, n_expression_bins=5)

        vocab_info = tokenizer.get_vocab_info()

        assert "vocab_size" in vocab_info
        assert "n_genes" in vocab_info
        assert "n_expression_bins" in vocab_info
        assert "n_special_tokens" in vocab_info
        assert "total_vocab_size" in vocab_info
        assert "special_tokens" in vocab_info
        assert "expr_bin_start" in vocab_info
        assert "chunk_size" in vocab_info

        # Check values
        assert vocab_info["vocab_size"] == 10
        assert vocab_info["n_expression_bins"] == 5
        assert vocab_info["n_special_tokens"] == 4
        assert vocab_info["expr_bin_start"] == 4
        assert vocab_info["chunk_size"] == 2048

    def test_token_decoding(self, tiny_slaf):
        """Test token sequence decoding"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Create a test token sequence
        test_tokens = [
            tokenizer.special_tokens["CLS"],  # CLS
            list(tokenizer.gene_vocab.values())[0],  # Gene token
            tokenizer.expr_bin_start,  # Expression bin
            tokenizer.special_tokens["SEP"],  # SEP
        ]

        decoded = tokenizer.decode_tokens(test_tokens)

        assert "special_tokens" in decoded
        assert "genes" in decoded
        assert "expression_bins" in decoded

        # Check decoded values
        assert "CLS" in decoded["special_tokens"]
        assert "SEP" in decoded["special_tokens"]
        assert len(decoded["genes"]) == 1
        assert len(decoded["expression_bins"]) == 1

    def test_tokenization_consistency(self, tiny_slaf):
        """Test that tokenization produces consistent results"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize same range multiple times
        tokens1 = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 2), max_genes=5)
        tokens2 = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 2), max_genes=5)

        assert tokens1 == tokens2

        tokens3 = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 2), max_genes=5
        )
        tokens4 = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 2), max_genes=5
        )

        assert tokens3 == tokens4

    def test_large_batch_handling(self, tiny_slaf):
        """Test handling of larger batches"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10, chunk_size=2)

        # Test batch that requires multiple chunks
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 5), max_genes=5)
        assert len(tokens) == 5  # Should process all cells

        tokens = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 5), max_genes=5
        )
        assert len(tokens) == 5

    def test_max_genes_limiting(self, tiny_slaf):
        """Test that max_genes parameter properly limits token sequences"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Test with very small max_genes
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 2), max_genes=1)
        for seq in tokens:
            # scGPT: CLS + (gene + expr) * max_genes + SEP
            max_expected = 1 + 2 * 1 + 1  # CLS + (gene+expr) + SEP
            assert len(seq) <= max_expected

        tokens = tokenizer.tokenize_geneformer(
            cell_integer_id_range=(0, 2), max_genes=1
        )
        for seq in tokens:
            # Geneformer: just gene tokens, padded to max_genes
            assert len(seq) == 1

    def test_unknown_gene_handling(self, tiny_slaf):
        """Test handling of unknown genes"""
        tokenizer = SLAFTokenizer(
            tiny_slaf, vocab_size=5
        )  # Small vocab to force unknown genes

        # Tokenize and check for UNK tokens
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 2), max_genes=10)

        # Should have some UNK tokens due to small vocabulary
        all_tokens = [token for seq in tokens for token in seq]
        assert tokenizer.special_tokens["UNK"] in all_tokens

    def test_expression_binning_edge_cases(self, tiny_slaf):
        """Test expression binning edge cases"""
        tokenizer = SLAFTokenizer(tiny_slaf, n_expression_bins=3)

        # Test very small expressions
        bin1 = tokenizer._expression_to_bin(0.001)
        bin2 = tokenizer._expression_to_bin(0.1)

        # Test very large expressions
        bin3 = tokenizer._expression_to_bin(1000.0)
        bin4 = tokenizer._expression_to_bin(10000.0)

        # All should be valid bins (not PAD)
        assert bin1 != tokenizer.special_tokens["PAD"]
        assert bin2 != tokenizer.special_tokens["PAD"]
        assert bin3 != tokenizer.special_tokens["PAD"]
        assert bin4 != tokenizer.special_tokens["PAD"]

        # Should be within valid range
        assert bin1 >= tokenizer.expr_bin_start
        assert bin1 < tokenizer.expr_bin_start + tokenizer.n_expression_bins

    def test_scgpt_tokenization_consistent_length(self, tiny_slaf):
        """Test that scGPT tokenization returns sequences of consistent length"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize a range with max_genes=5
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 3), max_genes=5)

        assert len(tokens) == 3  # 3 cells

        # All sequences should have the same length
        # scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
        # Max length = max_genes * 2 + 2 (CLS + SEP)
        expected_length = 5 * 2 + 2  # 12 tokens

        for seq in tokens:
            assert len(seq) == expected_length
            assert seq[0] == tokenizer.special_tokens["CLS"]  # Start with CLS
            assert seq[-1] == tokenizer.special_tokens["SEP"]  # End with SEP

    def test_scgpt_tokenization_padding(self, tiny_slaf):
        """Test that scGPT tokenization properly pads shorter sequences"""
        tokenizer = SLAFTokenizer(tiny_slaf, vocab_size=10)

        # Tokenize with a large max_genes to ensure some cells have fewer genes
        tokens = tokenizer.tokenize_scgpt(cell_integer_id_range=(0, 2), max_genes=20)

        assert len(tokens) == 2

        # Check that sequences are padded to the same length
        expected_length = 20 * 2 + 2  # 42 tokens

        for seq in tokens:
            assert len(seq) == expected_length
            assert seq[0] == tokenizer.special_tokens["CLS"]
            # SEP should be present
            assert tokenizer.special_tokens["SEP"] in seq
            sep_pos = seq.index(tokenizer.special_tokens["SEP"])
            # All tokens after SEP should be PAD
            if sep_pos < len(seq) - 1:
                assert all(
                    t == tokenizer.special_tokens["PAD"] for t in seq[sep_pos + 1 :]
                )
