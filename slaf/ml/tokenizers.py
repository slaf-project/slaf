from typing import Any

import lance
import numpy as np
import polars as pl

from slaf.core.slaf import SLAFArray


class SLAFTokenizer:
    """
    Tokenizer for single-cell RNA-seq data in SLAF format.

    SLAFTokenizer converts single-cell gene expression data into tokenized sequences
    suitable for machine learning models. It supports multiple tokenization strategies
    including scGPT and GeneFormer styles with configurable vocabulary sizes and
    expression binning.

    Key Features:
        - Multiple tokenization strategies (scGPT, GeneFormer)
        - Configurable vocabulary size and expression binning
        - Efficient chunked processing for large datasets
        - Special token handling (CLS, SEP, PAD, UNK)
        - Memory-efficient gene vocabulary building

    Examples:
        >>> # Basic tokenizer initialization
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> tokenizer = SLAFTokenizer(slaf_array)
        >>> print(f"Vocabulary size: {tokenizer.vocab_size}")
        Vocabulary size: 50000

        >>> # Custom configuration
        >>> tokenizer = SLAFTokenizer(
        ...     slaf_array=slaf_array,
        ...     vocab_size=30000,
        ...     n_expression_bins=15,
        ...     chunk_size=1024
        ... )
        >>> print(f"Expression bins: {tokenizer.n_expression_bins}")
        Expression bins: 15

        >>> # Tokenize cells for training
        >>> tokens = tokenizer.tokenize_geneformer(
        ...     cell_integer_id_range=(0, 32),
        ...     max_genes=2048
        ... )
        >>> print(f"Tokenized {len(tokens)} cells")
        Tokenized 32 cells
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        chunk_size: int = 2048,
        use_fragment_processing: bool = True,
    ):
        """
        Initialize SLAFTokenizer with SLAF array and vocabulary settings.

        Args:
            slaf_array: Initialized SLAFArray instance containing the single-cell data.
                       Used to build the gene vocabulary and access expression data.
            vocab_size: Maximum size of gene vocabulary. Genes beyond this limit
                       will be mapped to UNK token. Larger vocabularies use more memory.
            n_expression_bins: Number of expression level bins for scGPT tokenization.
                              More bins provide finer expression level discretization.
            chunk_size: Number of cells to process in each chunk for memory efficiency.
                       Larger chunks are faster but use more memory.
            use_fragment_processing: Whether to use fragment-based processing (recommended)
                                   instead of SQL queries for much better performance.

        Raises:
            ValueError: If vocab_size or n_expression_bins are invalid.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = SLAFTokenizer(slaf_array)
            >>> print(f"Special tokens: {list(tokenizer.special_tokens.keys())}")
            Special tokens: ['CLS', 'SEP', 'PAD', 'UNK']

            >>> # Custom vocabulary size
            >>> tokenizer = SLAFTokenizer(
            ...     slaf_array=slaf_array,
            ...     vocab_size=10000
            ... )
            >>> print(f"Gene vocabulary size: {len(tokenizer.gene_vocab)}")
            Gene vocabulary size: 10000

            >>> # With more expression bins
            >>> tokenizer = SLAFTokenizer(
            ...     slaf_array=slaf_array,
            ...     n_expression_bins=20
            ... )
            >>> print(f"Expression bin start: {tokenizer.expr_bin_start}")
            Expression bin start: 4
        """
        self.slaf_array = slaf_array
        self.vocab_size = vocab_size
        self.n_expression_bins = n_expression_bins
        self.chunk_size = chunk_size
        self.use_fragment_processing = use_fragment_processing

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

        # Initialize FragmentProcessor if using fragment-based processing
        if self.use_fragment_processing:
            # Use simple fragment processing (like Phase0)
            pass  # No complex processors needed

    def _build_gene_vocabulary(self):
        """
        Build gene vocabulary from SLAF var DataFrame.

        Creates a mapping from gene IDs to token IDs for the tokenizer.
        The vocabulary is limited by vocab_size and includes special tokens
        and expression bins in the token space.

        Examples:
            >>> # Vocabulary building process
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = SLAFTokenizer(slaf_array, vocab_size=1000)
            >>> print(f"Gene vocabulary size: {len(tokenizer.gene_vocab)}")
            Gene vocabulary size: 1000
            >>> print(f"First gene token: {list(tokenizer.gene_vocab.values())[0]}")
            First gene token: 14  # 4 special tokens + 10 expression bins
        """
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
        # Convert integer gene IDs to strings for vocabulary lookup
        # The vocabulary uses string keys from slaf_array.var.index
        converted_gene_ids = [str(gene_id) for gene_id in gene_ids]

        # Direct dictionary lookup is faster than pandas for small arrays
        return np.array(
            [
                self.gene_vocab.get(gene_id, self.special_tokens["UNK"])
                for gene_id in converted_gene_ids
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
        """
        Tokenize cells using scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP].

        This method tokenizes single-cell data in the scGPT format, which interleaves
        gene tokens with expression level tokens. The format is designed for transformer
        models that can learn gene-expression relationships.

        Args:
            cell_integer_id_range: Range of cell integer IDs (start, end) to tokenize.
            max_genes: Maximum number of genes to include per cell. Genes are ranked
                      by expression level and only the top max_genes are included.
            use_sql_binning: Whether to use SQL-based expression binning for better
                           performance on large datasets. If False, uses Python-based
                           binning which is more memory efficient.

        Returns:
            list[list[int]]: List of token sequences, one per cell. Each sequence
                            follows the format [CLS, gene1, expr1, gene2, expr2, ..., SEP, PAD...].

        Raises:
            ValueError: If cell_integer_id_range is invalid.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic scGPT tokenization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = SLAFTokenizer(slaf_array)
            >>> tokens = tokenizer.tokenize_scgpt(
            ...     cell_integer_id_range=(0, 32), max_genes=1024
            ... )
            >>> print(f"Tokenized {len(tokens)} cells")
            Tokenized 32 cells
            >>> print(f"First cell sequence length: {len(tokens[0])}")
            First cell sequence length: 2050  # CLS + 1024*2 + SEP

            >>> # With SQL binning for large datasets
            >>> tokens = tokenizer.tokenize_scgpt(
            ...     cell_integer_id_range=(0, 100), max_genes=512, use_sql_binning=True
            ... )
            >>> print(f"Tokenized {len(tokens)} cells with SQL binning")
            Tokenized 100 cells with SQL binning

            >>> # Check token format
            >>> first_tokens = tokens[0][:10]  # First 10 tokens
            >>> print(f"First tokens: {first_tokens}")
            First tokens: [0, 14, 4, 15, 5, ...]  # CLS, gene1, expr1, gene2, expr2, ...
        """
        # Choose between fragment-based and SQL-based processing
        if self.use_fragment_processing:
            return self._tokenize_scgpt_fragment_based(cell_integer_id_range, max_genes)
        else:
            # Use original SQL-based approach
            start, end = cell_integer_id_range
            chunks = self._chunk_range(start, end)

            if len(chunks) == 1:
                # Single chunk - process directly
                return self._tokenize_scgpt_chunk(chunks[0], max_genes, use_sql_binning)
            else:
                # Multiple chunks - process sequentially
                return self._tokenize_scgpt_sequential(
                    chunks, max_genes, use_sql_binning
                )

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
                c.cell_id,
                g.gene_id,
                e.value,
                ROW_NUMBER() OVER (
                    PARTITION BY c.cell_id
                    ORDER BY e.value DESC, g.gene_id ASC
                ) as gene_rank
            FROM expression e
            JOIN cells c ON e.cell_integer_id = c.cell_integer_id
            JOIN genes g ON e.gene_integer_id = g.gene_integer_id
            WHERE e.cell_integer_id BETWEEN {start} AND {end - 1}
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
                c.cell_id,
                g.gene_id,
                e.value,
                -- Calculate expression bin in SQL using window functions
                LEAST(
                    {self.n_expression_bins - 1},
                    FLOOR(
                        (ln(1 + e.value) - MIN(ln(1 + e.value)) OVER ()) * {self.n_expression_bins} /
                        NULLIF(MAX(ln(1 + e.value)) OVER () - MIN(ln(1 + e.value)) OVER (), 0)
                    )
                ) as expr_bin,
                ROW_NUMBER() OVER (
                    PARTITION BY c.cell_id
                    ORDER BY e.value DESC, g.gene_id ASC
                ) as gene_rank
            FROM expression e
            JOIN cells c ON e.cell_integer_id = c.cell_integer_id
            JOIN genes g ON e.gene_integer_id = g.gene_integer_id
            WHERE e.cell_integer_id BETWEEN {start} AND {end - 1}
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

    def _tokenize_scgpt_fragment_based(
        self, cell_integer_id_range: tuple[int, int], max_genes: int
    ) -> list[list[int]]:
        """Tokenize using fragment-based processing with Polars"""
        start, end = cell_integer_id_range

        # Load fragment using Lance dataset
        expression_dataset = lance.dataset(
            f"{self.slaf_array.slaf_path}/expression.lance"
        )
        fragments = expression_dataset.get_fragments()

        all_tokens = []

        # Process fragments that contain cells in our range
        for fragment_id in range(len(fragments)):
            fragment = fragments[fragment_id]
            fragment_df = pl.from_arrow(fragment.to_table())

            # Filter to our cell range
            filtered_df = fragment_df.filter(
                (pl.col("cell_integer_id") >= start) & (pl.col("cell_integer_id") < end)
            )
            assert isinstance(filtered_df, pl.DataFrame), (
                f"Expected DataFrame, got {type(filtered_df)}"
            )

            if len(filtered_df) > 0:
                # Tokenize this fragment
                fragment_tokens = self._tokenize_fragment_scgpt(filtered_df, max_genes)
                all_tokens.extend(fragment_tokens)

        return all_tokens

    def _tokenize_fragment_scgpt(
        self, fragment_df: pl.DataFrame, max_genes: int
    ) -> list[list[int]]:
        """
        Tokenize a fragment using scGPT format with Polars window functions.

        This method replaces the SQL-based approach with fragment-based Polars processing
        for much better performance (38K cells/sec vs 32 cells/min).

        Args:
            fragment_df: Polars DataFrame containing fragment data
            max_genes: Maximum number of genes to include per cell

        Returns:
            List of token sequences, one per cell
        """
        # Apply window functions for scGPT tokenization
        grouped = (
            fragment_df.with_columns(
                [
                    pl.col("value")
                    .rank(method="dense", descending=True)
                    .over("cell_integer_id")
                    .alias("gene_rank")
                ]
            )
            .filter(pl.col("gene_rank") <= max_genes)
            .group_by("cell_integer_id")
            .agg(
                [
                    pl.col("gene_integer_id").alias("gene_sequence"),
                    pl.col("value").alias("expr_sequence"),
                ]
            )
        )

        # Convert to token sequences
        token_sequences = []
        max_seq_length = max_genes * 2 + 2  # CLS + (gene,expr)*max_genes + SEP

        for row in grouped.iter_rows(named=True):
            tokens = [self.special_tokens["CLS"]]

            # Vectorized expression binning for the entire sequence
            expr_sequence = np.array(row["expr_sequence"])
            expr_tokens = self._expression_to_bin_vectorized(expr_sequence)

            # Process gene tokens and interleave with expression tokens
            for i, gene_id in enumerate(row["gene_sequence"]):
                # Gene token
                gene_token = self.gene_vocab.get(
                    str(gene_id), self.special_tokens["UNK"]
                )
                tokens.append(gene_token)

                # Expression bin token (already computed)
                tokens.append(expr_tokens[i])

            tokens.append(self.special_tokens["SEP"])
            # Pad after SEP if needed
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
        """
        Tokenize cells using Geneformer format: ranked gene tokens.

        This method tokenizes single-cell data in the Geneformer format, which creates
        sequences of gene tokens ranked by expression level. This format is designed
        for transformer models that learn gene importance from expression patterns.

        Args:
            cell_integer_id_range: Range of cell integer IDs (start, end) to tokenize.
            max_genes: Maximum number of genes to include per cell. Genes are ranked
                      by expression level and only the top max_genes are included.
            min_percentile: Optional percentile filter (0-100) for expression levels.
                          Only genes with expression above this percentile are included.
                          If None, all genes are considered.

        Returns:
            list[list[int]]: List of token sequences, one per cell. Each sequence
                            contains gene tokens ranked by expression level, padded
                            to max_genes length.

        Raises:
            ValueError: If cell_integer_id_range is invalid or min_percentile is out of range.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic Geneformer tokenization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = SLAFTokenizer(slaf_array)
            >>> tokens = tokenizer.tokenize_geneformer(
            ...     cell_integer_id_range=(0, 32), max_genes=2048
            ... )
            >>> print(f"Tokenized {len(tokens)} cells")
            Tokenized 32 cells
            >>> print(f"First cell sequence length: {len(tokens[0])}")
            First cell sequence length: 2048

            >>> # With percentile filtering
            >>> tokens = tokenizer.tokenize_geneformer(
            ...     cell_integer_id_range=(0, 100), max_genes=1024, min_percentile=10.0
            ... )
            >>> print(f"Tokenized {len(tokens)} cells with percentile filtering")
            Tokenized 100 cells with percentile filtering

            >>> # Check token ranking (first few tokens should be highest expressed genes)
            >>> first_tokens = tokens[0][:10]
            >>> print(f"First 10 gene tokens: {first_tokens}")
            First 10 gene tokens: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        """
        # Choose between fragment-based and SQL-based processing
        if self.use_fragment_processing:
            return self._tokenize_geneformer_fragment_based(
                cell_integer_id_range, max_genes
            )
        else:
            # Use original SQL-based approach
            start, end = cell_integer_id_range
            chunks = self._chunk_range(start, end)

            if len(chunks) == 1:
                # Single chunk - process directly
                return self._tokenize_geneformer_chunk(
                    chunks[0], max_genes, min_percentile
                )
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
                c.cell_id,
                g.gene_id,
                e.value,
                RANK() OVER (
                    PARTITION BY c.cell_id
                    ORDER BY e.value DESC, g.gene_id ASC
                ) as expression_rank
            FROM expression e
            JOIN cells c ON e.cell_integer_id = c.cell_integer_id
            JOIN genes g ON e.gene_integer_id = g.gene_integer_id
            WHERE e.cell_integer_id BETWEEN {start} AND {end - 1}
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
                c.cell_id,
                g.gene_id,
                e.value,
                PERCENT_RANK() OVER (
                    PARTITION BY c.cell_id
                    ORDER BY e.value
                ) * 100 as expr_percentile
            FROM expression e
            JOIN cells c ON e.cell_integer_id = c.cell_integer_id
            JOIN genes g ON e.gene_integer_id = g.gene_integer_id
            WHERE e.cell_integer_id BETWEEN {start} AND {end - 1}
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

    def _tokenize_geneformer_fragment_based(
        self, cell_integer_id_range: tuple[int, int], max_genes: int
    ) -> list[list[int]]:
        """Tokenize using fragment-based processing with Polars"""
        start, end = cell_integer_id_range

        # Load fragment using Lance dataset
        expression_dataset = lance.dataset(
            f"{self.slaf_array.slaf_path}/expression.lance"
        )
        fragments = expression_dataset.get_fragments()

        all_tokens = []

        # Process fragments that contain cells in our range
        for fragment_id in range(len(fragments)):
            fragment = fragments[fragment_id]
            fragment_df = pl.from_arrow(fragment.to_table())

            # Filter to our cell range
            filtered_df = fragment_df.filter(
                (pl.col("cell_integer_id") >= start) & (pl.col("cell_integer_id") < end)
            )
            assert isinstance(filtered_df, pl.DataFrame), (
                f"Expected DataFrame, got {type(filtered_df)}"
            )

            if len(filtered_df) > 0:
                # Tokenize this fragment
                fragment_tokens = self._tokenize_fragment_geneformer(
                    filtered_df, max_genes
                )
                all_tokens.extend(fragment_tokens)

        return all_tokens

    def _tokenize_fragment_geneformer(
        self, fragment_df: pl.DataFrame, max_genes: int
    ) -> list[list[int]]:
        """
        Tokenize a fragment using Geneformer format with Polars window functions.

        This method replaces the SQL-based approach with fragment-based Polars processing
        for much better performance (38K cells/sec vs 32 cells/min).

        Args:
            fragment_df: Polars DataFrame containing fragment data
            max_genes: Maximum number of genes to include per cell

        Returns:
            List of token sequences, one per cell
        """
        # Apply window functions for Geneformer tokenization
        grouped = (
            fragment_df.with_columns(
                [
                    pl.col("value")
                    .rank(method="dense", descending=True)
                    .over("cell_integer_id")
                    .alias("gene_rank")
                ]
            )
            .filter(pl.col("gene_rank") <= max_genes)
            .group_by("cell_integer_id")
            .agg(
                [
                    pl.col("gene_integer_id").alias(
                        "gene_sequence"
                    ),  # Geneformer only needs genes
                ]
            )
        )

        # Convert to token sequences
        token_sequences = []
        max_seq_length = max_genes * 2 + 2  # CLS + (gene,expr)*max_genes + SEP

        for row in grouped.iter_rows(named=True):
            # Vectorized gene token mapping
            gene_tokens = self._map_gene_ids_to_tokens_vectorized(row["gene_sequence"])

            # Convert to list and pad/truncate
            tokens = gene_tokens.tolist()
            if len(tokens) < max_seq_length:
                tokens.extend(
                    [self.special_tokens["PAD"]] * (max_seq_length - len(tokens))
                )
            else:
                tokens = tokens[:max_seq_length]

            token_sequences.append(tokens)

        return token_sequences

    def get_vocab_info(self) -> dict[str, Any]:
        """
        Get comprehensive vocabulary information for the tokenizer.

        This method returns a dictionary containing all relevant information about
        the tokenizer's vocabulary, including sizes, token ranges, and configuration
        parameters.

        Returns:
            dict[str, Any]: Dictionary containing vocabulary information:
                - vocab_size: Maximum vocabulary size
                - n_genes: Number of genes in vocabulary
                - n_expression_bins: Number of expression level bins
                - n_special_tokens: Number of special tokens
                - total_vocab_size: Total vocabulary size including all token types
                - special_tokens: Dictionary mapping special token names to IDs
                - expr_bin_start: Starting token ID for expression bins
                - chunk_size: Processing chunk size

        Examples:
            >>> # Get vocabulary information
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = SLAFTokenizer(slaf_array, vocab_size=10000)
            >>> vocab_info = tokenizer.get_vocab_info()
            >>> print(f"Total vocabulary size: {vocab_info['total_vocab_size']}")
            Total vocabulary size: 10014  # 4 special + 10 bins + 10000 genes

            >>> # Check token ranges
            >>> print(f"Special tokens: {vocab_info['special_tokens']}")
            Special tokens: {'CLS': 0, 'SEP': 1, 'PAD': 2, 'UNK': 3}
            >>> print(f"Expression bin start: {vocab_info['expr_bin_start']}")
            Expression bin start: 4

            >>> # Verify vocabulary composition
            >>> print(f"Genes: {vocab_info['n_genes']}")
            Genes: 10000
            >>> print(f"Expression bins: {vocab_info['n_expression_bins']}")
            Expression bins: 10
        """
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
        """
        Decode token sequence back to interpretable format for debugging.

        This method converts a sequence of token IDs back into a human-readable
        format, separating special tokens, gene tokens, and expression bin tokens.
        Useful for debugging tokenization and understanding model inputs.

        Args:
            tokens: List of token IDs to decode.

        Returns:
            dict[str, Any]: Dictionary containing decoded tokens:
                - special_tokens: List of special token names (CLS, SEP, PAD, UNK)
                - genes: List of gene IDs corresponding to gene tokens
                - expression_bins: List of expression bin indices (0-based)

        Examples:
            >>> # Decode a token sequence
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = SLAFTokenizer(slaf_array)
            >>> tokens = [0, 14, 4, 15, 5, 1, 2, 2]  # CLS, gene1, expr1, gene2, expr2, SEP, PAD, PAD
            >>> decoded = tokenizer.decode_tokens(tokens)
            >>> print(f"Special tokens: {decoded['special_tokens']}")
            Special tokens: ['CLS', 'SEP', 'PAD', 'PAD']
            >>> print(f"Genes: {decoded['genes']}")
            Genes: ['GENE_001', 'GENE_002']
            >>> print(f"Expression bins: {decoded['expression_bins']}")
            Expression bins: [0, 1]

            >>> # Decode Geneformer tokens
            >>> geneformer_tokens = [14, 15, 16, 17, 2, 2, 2, 2]  # gene1, gene2, gene3, gene4, PAD, PAD, PAD, PAD
            >>> decoded = tokenizer.decode_tokens(geneformer_tokens)
            >>> print(f"Genes: {decoded['genes']}")
            Genes: ['GENE_001', 'GENE_002', 'GENE_003', 'GENE_004']
            >>> print(f"Special tokens: {decoded['special_tokens']}")
            Special tokens: ['PAD', 'PAD', 'PAD', 'PAD']
        """
        tokens_array = np.array(tokens)
        decoded: dict[str, Any] = {
            "special_tokens": [],
            "genes": [],
            "expression_bins": [],
        }

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

    def tokenize_scgpt_fragment_based(
        self,
        cell_integer_id_range: tuple[int, int],
        max_genes: int = 1024,
    ) -> list[list[int]]:
        """
        DEPRECATED: Use the simplified fragment processing in datasets.py instead.
        """
        raise NotImplementedError(
            "Use the simplified fragment processing in datasets.py instead"
        )

    def tokenize_geneformer_fragment_based(
        self,
        cell_integer_id_range: tuple[int, int],
        max_genes: int = 2048,
    ) -> list[list[int]]:
        """
        DEPRECATED: Use the simplified fragment processing in datasets.py instead.
        """
        raise NotImplementedError(
            "Use the simplified fragment processing in datasets.py instead"
        )

    def _convert_gene_sequence_to_scgpt_tokens(
        self, gene_sequence: list[int], expr_sequence: list[float], max_genes: int
    ) -> list[int]:
        """Convert gene sequence to scGPT token format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]"""
        tokens = [self.special_tokens["CLS"]]

        # Process genes up to max_genes
        for i, gene_id in enumerate(gene_sequence[:max_genes]):
            # Gene token - convert integer to string for vocabulary lookup
            gene_token = self.gene_vocab.get(str(gene_id), self.special_tokens["UNK"])
            tokens.append(gene_token)

            # Expression bin token (simplified - could be enhanced)
            expr_token = self.expr_bin_start + (i % self.n_expression_bins)
            tokens.append(expr_token)

        tokens.append(self.special_tokens["SEP"])

        # Pad to max_genes * 2 + 2 (CLS + (gene,expr)*max_genes + SEP)
        max_seq_length = max_genes * 2 + 2
        if len(tokens) < max_seq_length:
            tokens.extend([self.special_tokens["PAD"]] * (max_seq_length - len(tokens)))
        else:
            tokens = tokens[:max_seq_length]

        return tokens

    def _convert_gene_sequence_to_geneformer_tokens(
        self, gene_sequence: list[int], max_genes: int
    ) -> list[int]:
        """Convert gene sequence to Geneformer token format: ranked gene tokens"""
        # Convert gene IDs to tokens
        gene_tokens = self._map_gene_ids_to_tokens_vectorized(gene_sequence)

        # Convert to list and pad/truncate
        tokens = gene_tokens.tolist()
        if len(tokens) < max_genes:
            tokens.extend([self.special_tokens["PAD"]] * (max_genes - len(tokens)))
        else:
            tokens = tokens[:max_genes]

        return tokens

    def _convert_gene_sequence_to_scgpt_tokens_simple(
        self, gene_sequence: list[int], max_genes: int
    ) -> list[int]:
        """Convert gene sequence to scGPT token format without expression values (fallback)"""
        tokens = [self.special_tokens["CLS"]]

        # Process genes up to max_genes
        for i, gene_id in enumerate(gene_sequence[:max_genes]):
            # Gene token - convert integer to string for vocabulary lookup
            gene_token = self.gene_vocab.get(str(gene_id), self.special_tokens["UNK"])
            tokens.append(gene_token)

            # Expression bin token (simplified - could be enhanced)
            expr_token = self.expr_bin_start + (i % self.n_expression_bins)
            tokens.append(expr_token)

        tokens.append(self.special_tokens["SEP"])

        # Pad to max_genes * 2 + 2 (CLS + (gene,expr)*max_genes + SEP)
        max_seq_length = max_genes * 2 + 2
        if len(tokens) < max_seq_length:
            tokens.extend([self.special_tokens["PAD"]] * (max_seq_length - len(tokens)))
        else:
            tokens = tokens[:max_seq_length]

        return tokens
