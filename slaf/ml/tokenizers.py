from typing import Any

import numpy as np

from slaf.core.slaf import SLAFArray


class SLAFTokenizer:
    """Tokenizer for single-cell RNA-seq data in SLAF format

    Supports scGPT and Geneformer tokenization styles with various configurations.
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        chunk_size: int = 2048,
    ):
        """Initialize SLAFTokenizer with SLAF array and vocabulary settings

        Args:
            slaf_array: Initialized SLAFArray instance
            vocab_size: Maximum size of gene vocabulary
            n_expression_bins: Number of expression bins for scGPT tokenization
            chunk_size: Number of cells to process in each chunk
        """
        self.slaf_array = slaf_array
        self.vocab_size = vocab_size
        self.n_expression_bins = n_expression_bins
        self.chunk_size = chunk_size

        # Define special tokens first
        self.special_tokens = {
            "CLS": 0,  # Start of sequence
            "SEP": 1,  # End of sequence
            "PAD": 2,  # Padding token
            "UNK": 3,  # Unknown gene
        }

        # Expression bin tokens start after special tokens
        self.expr_bin_start = len(self.special_tokens)

        # Build gene vocabulary from var DataFrame
        self._build_gene_vocabulary()

    def _build_gene_vocabulary(self):
        """Build gene vocabulary from SLAF var DataFrame"""
        # Use gene_integer_id as token IDs, but limit to vocab_size
        gene_ids = self.slaf_array.var.index.tolist()[: self.vocab_size]

        # Simple dictionary comprehension - faster than pandas for this use case
        self.gene_vocab = {
            gene_id: i + self.n_expression_bins + len(self.special_tokens)
            for i, gene_id in enumerate(gene_ids)
        }

        # Create reverse mapping for debugging
        self.token_to_gene = {v: k for k, v in self.gene_vocab.items()}

    def _expression_to_bin(self, expression_value: float) -> int:
        """Convert expression value to bin token ID"""
        if expression_value <= 0:
            return self.special_tokens["PAD"]

        # Log transform and bin
        log_expr = np.log1p(expression_value)

        # Simple binning: assume log expression ranges from 0 to ~10
        # This could be made more sophisticated with dynamic binning
        bin_id = min(
            self.n_expression_bins - 1, int(log_expr * self.n_expression_bins / 10)
        )

        return self.expr_bin_start + bin_id

    def _expression_to_bin_vectorized(
        self, expression_values: np.ndarray
    ) -> np.ndarray:
        """Vectorized version of expression binning"""
        # Handle zero/negative values
        result = np.full_like(expression_values, self.special_tokens["PAD"], dtype=int)

        # Only process positive values
        positive_mask = expression_values > 0
        if not np.any(positive_mask):
            return result

        # Log transform positive values
        log_expr = np.log1p(expression_values[positive_mask])

        # Vectorized binning
        bin_ids = np.clip(
            np.floor(log_expr * self.n_expression_bins / 10),
            0,
            self.n_expression_bins - 1,
        ).astype(int)

        # Assign results
        result[positive_mask] = self.expr_bin_start + bin_ids

        return result

    def _map_gene_ids_to_tokens_vectorized(self, gene_ids) -> np.ndarray:
        """Vectorized mapping of gene IDs to token IDs"""
        # Direct dictionary lookup is faster than pandas for small arrays
        return np.array(
            [
                self.gene_vocab.get(gene_id, self.special_tokens["UNK"])
                for gene_id in gene_ids
            ],
            dtype=int,
        )

    def _chunk_range(self, start: int, end: int) -> list[tuple[int, int]]:
        """Split a range into chunks"""
        chunks = []
        for i in range(start, end, self.chunk_size):
            chunk_end = min(i + self.chunk_size, end)
            chunks.append((i, chunk_end))
        return chunks

    def tokenize_scgpt(
        self,
        cell_integer_id_range: tuple[int, int],
        max_genes: int = 1024,
        use_sql_binning: bool = False,
    ) -> list[list[int]]:
        """Tokenize cells using scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]

        Args:
            cell_integer_id_range: Range of cell integer IDs (start, end)
            max_genes: Maximum number of genes per cell
            use_sql_binning: Whether to use SQL-based expression binning

        Returns:
            List of token sequences, one per cell
        """
        start, end = cell_integer_id_range
        chunks = self._chunk_range(start, end)

        if len(chunks) == 1:
            # Single chunk - process directly
            return self._tokenize_scgpt_chunk(chunks[0], max_genes, use_sql_binning)
        else:
            # Multiple chunks - process sequentially
            return self._tokenize_scgpt_sequential(chunks, max_genes, use_sql_binning)

    def _tokenize_scgpt_sequential(
        self, chunks: list[tuple[int, int]], max_genes: int, use_sql_binning: bool
    ) -> list[list[int]]:
        """Process scGPT tokenization sequentially across chunks"""
        all_tokens = []
        for chunk in chunks:
            chunk_tokens = self._tokenize_scgpt_chunk(chunk, max_genes, use_sql_binning)
            all_tokens.extend(chunk_tokens)
        return all_tokens

    def _tokenize_scgpt_chunk(
        self, chunk_range: tuple[int, int], max_genes: int, use_sql_binning: bool
    ) -> list[list[int]]:
        """Tokenize a single chunk of cells using scGPT format"""
        if use_sql_binning:
            return self._tokenize_scgpt_sql_binning(chunk_range, max_genes)
        else:
            return self._tokenize_scgpt_python_binning(chunk_range, max_genes)

    def _tokenize_scgpt_python_binning(
        self, cell_integer_id_range: tuple[int, int], max_genes: int
    ) -> list[list[int]]:
        """scGPT tokenization with Python-based expression binning"""
        start, end = cell_integer_id_range

        # Query expression data sorted by expression
        sql = f"""
        WITH ranked_genes AS (
            SELECT
                cell_id,
                gene_id,
                value,
                ROW_NUMBER() OVER (
                    PARTITION BY cell_id
                    ORDER BY value DESC, gene_id ASC
                ) as gene_rank
            FROM expression
            WHERE cell_integer_id BETWEEN {start} AND {end - 1}
        ),
        limited_genes AS (
            SELECT cell_id, gene_id, value, gene_rank
            FROM ranked_genes
            WHERE gene_rank <= {max_genes}
        )
        SELECT
            cell_id,
            array_agg(gene_id ORDER BY gene_rank) as gene_sequence,
            array_agg(value ORDER BY gene_rank) as expr_sequence
        FROM limited_genes
        GROUP BY cell_id
        ORDER BY cell_id
        """

        batch_data = self.slaf_array.query(sql)

        # Convert to token sequences
        token_sequences = []
        max_seq_length = max_genes * 2 + 2  # CLS + (gene,expr)*max_genes + SEP

        for _, row in batch_data.iterrows():
            tokens = [self.special_tokens["CLS"]]

            # Vectorized expression binning for the entire sequence
            expr_sequence = np.array(row["expr_sequence"])
            expr_tokens = self._expression_to_bin_vectorized(expr_sequence)

            # Process gene tokens and interleave with expression tokens
            for i, gene_id in enumerate(row["gene_sequence"]):
                # Gene token
                gene_token = self.gene_vocab.get(gene_id, self.special_tokens["UNK"])
                tokens.append(gene_token)

                # Expression bin token (already computed)
                tokens.append(expr_tokens[i])

            tokens.append(self.special_tokens["SEP"])
            # Now pad after SEP if needed
            if len(tokens) < max_seq_length:
                tokens.extend(
                    [self.special_tokens["PAD"]] * (max_seq_length - len(tokens))
                )
            else:
                tokens = tokens[:max_seq_length]

            token_sequences.append(tokens)

        return token_sequences

    def _tokenize_scgpt_sql_binning(
        self, cell_integer_id_range: tuple[int, int], max_genes: int
    ) -> list[list[int]]:
        """scGPT tokenization with SQL-based expression binning"""
        start, end = cell_integer_id_range

        # Calculate expression bins and create tokens in SQL
        sql = f"""
        WITH ranked_genes AS (
            SELECT
                cell_id,
                gene_id,
                value,
                -- Calculate expression bin in SQL using window functions
                LEAST(
                    {self.n_expression_bins - 1},
                    FLOOR(
                        (ln(1 + value) - MIN(ln(1 + value)) OVER ()) * {self.n_expression_bins} /
                        NULLIF(MAX(ln(1 + value)) OVER () - MIN(ln(1 + value)) OVER (), 0)
                    )
                ) as expr_bin,
                ROW_NUMBER() OVER (
                    PARTITION BY cell_id
                    ORDER BY value DESC, gene_id ASC
                ) as gene_rank
            FROM expression
            WHERE cell_integer_id BETWEEN {start} AND {end - 1}
        )
        SELECT
            cell_id,
            array_agg(gene_id ORDER BY gene_rank) as gene_ids,
            array_agg(expr_bin ORDER BY gene_rank) as expr_bins
        FROM ranked_genes
        WHERE gene_rank <= {max_genes}
        GROUP BY cell_id
        ORDER BY cell_id
        """

        batch_data = self.slaf_array.query(sql)

        # Convert to token sequences
        token_sequences = []
        max_seq_length = max_genes * 2 + 2  # CLS + (gene,expr)*max_genes + SEP

        for _, row in batch_data.iterrows():
            tokens = [self.special_tokens["CLS"]]

            # Vectorized expression token calculation
            expr_bins = np.array(row["expr_bins"])
            expr_tokens = expr_bins + self.expr_bin_start

            # Process gene tokens and interleave with expression tokens
            for i, gene_id in enumerate(row["gene_ids"]):
                # Gene token
                gene_token = self.gene_vocab.get(gene_id, self.special_tokens["UNK"])
                tokens.append(gene_token)

                # Expression bin token (already computed)
                tokens.append(expr_tokens[i])

            tokens.append(self.special_tokens["SEP"])
            # Now pad after SEP if needed
            if len(tokens) < max_seq_length:
                tokens.extend(
                    [self.special_tokens["PAD"]] * (max_seq_length - len(tokens))
                )
            else:
                tokens = tokens[:max_seq_length]

            token_sequences.append(tokens)

        return token_sequences

    def tokenize_geneformer(
        self,
        cell_integer_id_range: tuple[int, int],
        max_genes: int = 2048,
        min_percentile: float | None = None,
    ) -> list[list[int]]:
        """Tokenize cells using Geneformer format: ranked gene tokens

        Args:
            cell_integer_id_range: Range of cell integer IDs (start, end)
            max_genes: Maximum number of genes per cell
            min_percentile: Optional percentile filter (0-100) for expression

        Returns:
            List of token sequences, one per cell
        """
        start, end = cell_integer_id_range
        chunks = self._chunk_range(start, end)

        if len(chunks) == 1:
            # Single chunk - process directly
            return self._tokenize_geneformer_chunk(chunks[0], max_genes, min_percentile)
        else:
            # Multiple chunks - process sequentially
            return self._tokenize_geneformer_sequential(
                chunks, max_genes, min_percentile
            )

    def _tokenize_geneformer_sequential(
        self,
        chunks: list[tuple[int, int]],
        max_genes: int,
        min_percentile: float | None,
    ) -> list[list[int]]:
        """Process Geneformer tokenization sequentially across chunks"""
        all_tokens = []
        for chunk in chunks:
            chunk_tokens = self._tokenize_geneformer_chunk(
                chunk, max_genes, min_percentile
            )
            all_tokens.extend(chunk_tokens)
        return all_tokens

    def _tokenize_geneformer_chunk(
        self,
        chunk_range: tuple[int, int],
        max_genes: int,
        min_percentile: float | None,
    ) -> list[list[int]]:
        """Tokenize a single chunk of cells using Geneformer format"""
        if min_percentile is not None:
            return self._tokenize_geneformer_percentile(
                chunk_range, max_genes, min_percentile
            )
        else:
            return self._tokenize_geneformer_standard(chunk_range, max_genes)

    def _tokenize_geneformer_standard(
        self, cell_integer_id_range: tuple[int, int], max_genes: int
    ) -> list[list[int]]:
        """Standard Geneformer tokenization with expression ranking"""
        start, end = cell_integer_id_range

        # Rank genes by expression within each cell
        sql = f"""
        WITH ranked_expression AS (
            SELECT
                cell_id,
                gene_id,
                value,
                RANK() OVER (
                    PARTITION BY cell_id
                    ORDER BY value DESC, gene_id ASC
                ) as expression_rank
            FROM expression
            WHERE cell_integer_id BETWEEN {start} AND {end - 1}
        )
        SELECT
            cell_id,
            array_agg(gene_id ORDER BY expression_rank) as ranked_genes
        FROM ranked_expression
        WHERE expression_rank <= {max_genes}
        GROUP BY cell_id
        ORDER BY cell_id
        """

        batch_data = self.slaf_array.query(sql)

        # Convert to token sequences
        token_sequences = []
        for _, row in batch_data.iterrows():
            # Vectorized gene token mapping
            gene_tokens = self._map_gene_ids_to_tokens_vectorized(row["ranked_genes"])

            # Convert to list and pad/truncate
            tokens = gene_tokens.tolist()
            if len(tokens) < max_genes:
                tokens.extend([self.special_tokens["PAD"]] * (max_genes - len(tokens)))
            else:
                tokens = tokens[:max_genes]

            token_sequences.append(tokens)

        return token_sequences

    def _tokenize_geneformer_percentile(
        self,
        cell_integer_id_range: tuple[int, int],
        max_genes: int,
        min_percentile: float,
    ) -> list[list[int]]:
        """Geneformer tokenization with expression percentile filtering"""
        start, end = cell_integer_id_range

        # Filter by expression percentile within each cell
        sql = f"""
        WITH cell_percentiles AS (
            SELECT
                cell_id,
                gene_id,
                value,
                PERCENT_RANK() OVER (
                    PARTITION BY cell_id
                    ORDER BY value
                ) * 100 as expr_percentile
            FROM expression
            WHERE cell_integer_id BETWEEN {start} AND {end - 1}
        ),
        filtered_genes AS (
            SELECT
                cell_id,
                gene_id,
                value,
                RANK() OVER (
                    PARTITION BY cell_id
                    ORDER BY value DESC, gene_id ASC
                ) as final_rank
            FROM cell_percentiles
            WHERE expr_percentile >= {min_percentile}
        )
        SELECT
            cell_id,
            array_agg(gene_id ORDER BY final_rank) as ranked_genes
        FROM filtered_genes
        WHERE final_rank <= {max_genes}
        GROUP BY cell_id
        ORDER BY cell_id
        """

        batch_data = self.slaf_array.query(sql)

        # Convert to token sequences
        token_sequences = []
        for _, row in batch_data.iterrows():
            # Vectorized gene token mapping
            gene_tokens = self._map_gene_ids_to_tokens_vectorized(row["ranked_genes"])

            # Convert to list and pad/truncate
            tokens = gene_tokens.tolist()
            if len(tokens) < max_genes:
                tokens.extend([self.special_tokens["PAD"]] * (max_genes - len(tokens)))
            else:
                tokens = tokens[:max_genes]

            token_sequences.append(tokens)

        return token_sequences

    def get_vocab_info(self) -> dict[str, Any]:
        """Get vocabulary information"""
        return {
            "vocab_size": self.vocab_size,
            "n_genes": len(self.gene_vocab),
            "n_expression_bins": self.n_expression_bins,
            "n_special_tokens": len(self.special_tokens),
            "total_vocab_size": len(self.special_tokens)
            + self.n_expression_bins
            + len(self.gene_vocab),
            "special_tokens": self.special_tokens,
            "expr_bin_start": self.expr_bin_start,
            "chunk_size": self.chunk_size,
        }

    def decode_tokens(self, tokens: list[int]) -> dict[str, Any]:
        """Decode token sequence back to interpretable format (for debugging)"""
        tokens_array = np.array(tokens)
        decoded = {"special_tokens": [], "genes": [], "expression_bins": []}

        # Create reverse mapping for special tokens for O(1) lookup
        special_token_reverse = {v: k for k, v in self.special_tokens.items()}

        # Vectorized token classification
        special_mask = np.isin(tokens_array, list(self.special_tokens.values()))
        expr_bin_mask = (tokens_array >= self.expr_bin_start) & (
            tokens_array < self.expr_bin_start + self.n_expression_bins
        )
        gene_mask = np.isin(tokens_array, list(self.token_to_gene.keys()))

        # Process special tokens
        special_tokens = tokens_array[special_mask]
        for token in special_tokens:
            decoded["special_tokens"].append(special_token_reverse.get(token, "UNK"))

        # Process expression bin tokens
        expr_bin_tokens = tokens_array[expr_bin_mask]
        decoded["expression_bins"] = (expr_bin_tokens - self.expr_bin_start).tolist()

        # Process gene tokens
        gene_tokens = tokens_array[gene_mask]
        decoded["genes"] = [self.token_to_gene[token] for token in gene_tokens]

        return decoded
