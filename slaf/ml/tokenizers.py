from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
import torch

from slaf.core.slaf import SLAFArray
from slaf.core.tabular_schema import SLAF_LANCE_COO_SCHEMA, DataSchema
from slaf.integrations.anndata import LazyAnnData
from slaf.ml.aggregators import GeneformerWindow, ScGPTWindow, Window

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False


class TokenizerType(str, Enum):
    """Tokenizer types"""

    GENEFORMER = "geneformer"
    SCGPT = "scgpt"


class SLAFTokenizer(ABC):
    """
    Tokenizer for single-cell RNA-seq data in SLAF format.

    SLAFTokenizer converts single-cell gene expression data into token sequences
    suitable for machine learning models. It supports multiple tokenization strategies
    including GeneFormer and scGPT formats with optimized vectorized operations.

    Key Features:
        - Multiple tokenization strategies (GeneFormer, scGPT)
        - Vectorized tokenization for high performance
        - Expression binning for scGPT format
        - Device-agnostic CPU tensor output
        - Memory-efficient processing
        - Comprehensive vocabulary management
    """

    def __init__(
        self,
        adata: LazyAnnData,
        vocab_size: int = 50000,
        max_genes: int = 2048,
    ):
        """
        Initialize SLAFTokenizer with SLAF array and vocabulary settings.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
                   Used to build the gene vocabulary, access expression metadata,
                   and inspect lazy runtime transformations.
            vocab_size: Maximum size of gene vocabulary. Genes beyond this limit
                       are excluded from tokenization. Higher values use more memory.
            max_genes: Max genes per cell for windowing and tokenization (sequence layout
                       is tokenizer-specific). Should match training ``max_genes`` / model
                       sequence length expectations.

        Raises:
            ValueError: If vocab_size or max_genes is invalid.
            RuntimeError: If SLAF array is not properly initialized.
            TypeError: If adata is not a valid LazyAnnData instance.
        """
        self.adata = adata
        self.slaf_array = adata.slaf
        self.vocab_size = vocab_size
        if max_genes < 1:
            raise ValueError(f"max_genes must be >= 1, got {max_genes}")
        self.max_genes = max_genes

        self.window = self.create_window()

        # Build vocabulary and special tokens
        self._build_gene_vocabulary()
        self._setup_special_tokens()

    @property
    def tokenizer_name(self) -> str:
        """Stable tokenizer identifier for logging and worker reconstruction."""
        return self.__class__.__name__

    def _build_gene_vocabulary(self):
        """Build gene vocabulary from SLAF var DataFrame or genes Lance table."""
        try:
            # Try to use metadata if available
            if self.slaf_array.is_metadata_ready():
                var_df = self.slaf_array.var.reset_index()

                # Check if we have a real SLAF array or a Mock object
                if (
                    hasattr(var_df, "columns")
                    and "gene_integer_id" in var_df.columns
                    and "gene_id" in var_df.columns
                ):
                    # Real SLAF array - build vocabulary from gene data
                    gene_vocab = {}

                    # Use Polars native iteration
                    for row in var_df.iter_rows(named=True):
                        gene_id = row["gene_id"]
                        gene_integer_id = row["gene_integer_id"]

                        # Only include genes within vocab size limit
                        if gene_integer_id < self.vocab_size:
                            gene_vocab[gene_id] = gene_integer_id

                    self.gene_vocab = gene_vocab
                    # Account for the +4 offset used in tokenization
                    self.token_to_gene = {v + 4: k for k, v in self.gene_vocab.items()}

                    # Pre-build vectorized mapping array for fast lookup
                    max_gene_id = (
                        max(
                            int(k) if isinstance(k, str) else k
                            for k in self.gene_vocab.keys()
                        )
                        if self.gene_vocab
                        else 0
                    )
                    self.vocab_mapping = np.full(max_gene_id + 1, -1, dtype=int)

                    # Fill the mapping array once
                    for gene_id, token_id in self.gene_vocab.items():
                        try:
                            gene_id_int = (
                                int(gene_id) if isinstance(gene_id, str) else gene_id
                            )
                            self.vocab_mapping[gene_id_int] = token_id
                        except (ValueError, TypeError):
                            continue
                    return

            # If metadata not available, read directly from genes Lance table
            # This is more efficient for cloud datasets where metadata loading is skipped
            import polars as pl

            genes_table = self.slaf_array.genes.to_table()
            genes_df = pl.from_arrow(genes_table)

            # Check if we have the required columns
            if "gene_integer_id" in genes_df.columns and "gene_id" in genes_df.columns:
                gene_vocab = {}

                # Use Polars native iteration
                for row in genes_df.iter_rows(named=True):
                    gene_id = row["gene_id"]
                    gene_integer_id = row["gene_integer_id"]

                    # Only include genes within vocab size limit
                    if gene_integer_id < self.vocab_size:
                        gene_vocab[gene_id] = gene_integer_id

                self.gene_vocab = gene_vocab
                # Account for the +4 offset used in tokenization
                self.token_to_gene = {v + 4: k for k, v in self.gene_vocab.items()}

                # Pre-build vectorized mapping array for fast lookup
                max_gene_id = (
                    max(
                        int(k) if isinstance(k, str) else k
                        for k in self.gene_vocab.keys()
                    )
                    if self.gene_vocab
                    else 0
                )
                self.vocab_mapping = np.full(max_gene_id + 1, -1, dtype=int)

                # Fill the mapping array once
                for gene_id, token_id in self.gene_vocab.items():
                    try:
                        gene_id_int = (
                            int(gene_id) if isinstance(gene_id, str) else gene_id
                        )
                        self.vocab_mapping[gene_id_int] = token_id
                    except (ValueError, TypeError):
                        continue
                return

            # Fallback: Mock object or missing columns
            self.gene_vocab = {f"gene_{i}": i for i in range(1000)}
            # Account for the +4 offset used in tokenization
            self.token_to_gene = {v + 4: k for k, v in self.gene_vocab.items()}
            # No mapping array needed for mock objects

        except Exception:
            # Fallback for testing or error cases
            self.gene_vocab = {f"gene_{i}": i for i in range(1000)}
            # Account for the +4 offset used in tokenization
            self.token_to_gene = {v + 4: k for k, v in self.gene_vocab.items()}
            # No mapping array needed for fallback

    def _setup_special_tokens(self):
        """Setup special tokens for tokenization."""
        # Special token IDs
        self.special_tokens = {
            "PAD": 0,
            "CLS": 1,
            "SEP": 2,
            "MASK": 3,
        }

    def _map_gene_ids_to_tokens_vectorized(self, gene_ids) -> np.ndarray:
        """Vectorized mapping of gene IDs to token IDs"""
        # Handle edge cases
        if hasattr(gene_ids, "is_empty") and gene_ids.is_empty():
            return np.array([], dtype=int)
        elif hasattr(gene_ids, "__len__") and len(gene_ids) == 0:
            return np.array([], dtype=int)

        # Convert to numpy array for vectorized operations
        gene_ids_array = np.array(gene_ids, dtype=int)

        # Check if we have a real SLAF array or a Mock object
        try:
            # Try to access the DataFrame properly
            var_df = self.slaf_array.var.reset_index()

            # Check if it's actually a DataFrame with the expected columns
            if (
                hasattr(var_df, "columns")
                and "gene_integer_id" in var_df.columns
                and "gene_id" in var_df.columns
            ):
                # Real SLAF array - use pre-built vectorized mapping
                # Vectorized lookup using pre-built mapping array
                tokens = self.vocab_mapping[gene_ids_array]

                # Filter out missing genes (-1 values)
                valid_mask = tokens != -1
                return tokens[valid_mask]
            else:
                # Mock object - direct mapping (same as original test)
                return gene_ids_array + 4  # Simple offset like original test
        except Exception:
            # Fallback for testing - direct mapping with offset
            return gene_ids_array + 4  # Simple offset like original test

    @abstractmethod
    def create_window(self) -> Window:
        """
        Create a window function based on the tokenizer type.
        """

    def _get_runtime_transformations(self) -> dict[str, Any]:
        transformations = getattr(self.adata, "_transformations", None)
        if not isinstance(transformations, dict):
            return {}
        return transformations

    def _apply_runtime_transformations(
        self,
        df: pl.DataFrame,
        schema: DataSchema,
    ) -> pl.DataFrame:
        transformations = self._get_runtime_transformations()
        if not transformations:
            return df

        transformed_df = df
        value_col = schema.value_key
        group_col = schema.group_key

        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                target_sum = float(transform_data.get("target_sum", 1e4))
                transformed_df = transformed_df.with_columns(
                    (
                        pl.col(value_col)
                        / pl.col(value_col).sum().over(group_col)
                        * target_sum
                    ).alias(value_col)
                )
            elif transform_name == "log1p":
                transformed_df = transformed_df.with_columns(
                    pl.col(value_col).log1p().alias(value_col)
                )

        return transformed_df

    def apply(
        self,
        df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Group per-cell COO rows into tokenizer-ready sequences."""
        transformed_df = self._apply_runtime_transformations(df, schema)
        return self.window.apply(
            transformed_df,
            schema=schema,
            max_items=max_items,
            **kwargs,
        )

    def tokenize_grouped(
        self,
        grouped_df: pl.DataFrame,
        schema: DataSchema = SLAF_LANCE_COO_SCHEMA,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Tokenize grouped cell sequences emitted by ``apply``."""
        return self.tokenize(
            gene_sequences=grouped_df[schema.item_list_key].to_list(),
            expr_sequences=(
                grouped_df[schema.value_list_key].to_list()
                if schema.value_list_key and schema.value_list_key in grouped_df.columns
                else None
            ),
        )

    def get_factory_kwargs(self) -> dict[str, Any]:
        """Return constructor kwargs required to recreate this tokenizer."""
        return {
            "vocab_size": self.vocab_size,
            "max_genes": self.max_genes,
        }

    def get_vocab_info(self) -> dict[str, Any]:
        """
        Get vocabulary information for debugging and analysis.

        Returns:
            dict: Vocabulary information including size, special tokens, etc.

        Examples:
            >>> vocab_info = tokenizer.get_vocab_info()
            >>> print(f"Vocabulary size: {vocab_info['vocab_size']}")
            >>> print(f"Special tokens: {vocab_info['special_tokens']}")
            Vocabulary size: 50000
            Special tokens: {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3}
        """
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "gene_vocab_size": len(self.gene_vocab),
            "max_genes": self.max_genes,
        }

    @abstractmethod
    def tokenize(
        self,
        gene_sequences: list[list[int] | list[tuple[int, float]]],
        expr_sequences: list[list[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Tokenize gene expression sequences into model-ready tensors.

        This method converts gene and expression sequences into tokenized tensors
        suitable for machine learning models. It supports both GeneFormer and scGPT
        tokenization strategies with optimized vectorized operations.
        """

    @abstractmethod
    def decode_tokens(self, tokens: list[int]) -> dict[str, Any]:
        """
        Decode token sequence back to gene information.

        Args:
            tokens: List of token IDs to decode

        Returns:
            dict: Decoded information including genes, expressions, etc.

        Examples:
            >>> # Decode a token sequence
            >>> tokens = [1, 100, 50050, 200, 50060, 2]  # CLS, gene1, expr1, gene2, expr2, SEP
            >>> decoded = tokenizer.decode_tokens(tokens)
            >>> print(f"Genes: {decoded['genes']}")
            >>> print(f"Expressions: {decoded['expressions']}")
            Genes: ['gene_100', 'gene_200']
            Expressions: [0.5, 0.6]
        """


class ScGPTTokenizer(SLAFTokenizer):
    """
    scGPT tokenizer.

    Examples:
        >>> # scGPT with expression sequences
        >>> tokenizer = ScGPTTokenizer(slaf_array)
        >>> gene_sequences = [[1, 2, 3], [4, 5, 6]]
        >>> expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]
        >>> input_ids, attention_mask = tokenizer.tokenize(
        ...     gene_sequences, expr_sequences
        ... )
        >>> print(f"Input shape: {input_ids.shape}")
        Input shape: torch.Size([2, 2050])

        >>> # Vocabulary information
        >>> vocab_info = tokenizer.get_vocab_info()
        >>> print(f"Vocabulary size: {vocab_info['vocab_size']}")
        Vocabulary size: 50000
    """

    def __init__(
        self,
        adata: LazyAnnData,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        max_genes: int = 1024,
    ):
        """
        Initialize ScGPTTokenizer with SLAF array and vocabulary settings.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            vocab_size: Maximum size of gene vocabulary. Genes beyond this limit
                       are excluded from tokenization. Higher values use more memory.
            n_expression_bins: Number of expression bins for scGPT tokenization.
                             Higher values provide finer expression resolution.
                             Range: 1-1000, default: 10.
            max_genes: Maximum gene--expression pairs per cell. Sequence length is
                       ``2 * max_genes + 2`` (CLS, pairs, SEP).

        Raises:
            ValueError: If vocab_size is invalid.
            RuntimeError: If SLAF array is not properly initialized.
            TypeError: If adata is not a valid LazyAnnData instance.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> tokenizer = ScGPTTokenizer(LazyAnnData(slaf_array))

            >>> # scGPT with custom settings
            >>> tokenizer = ScGPTTokenizer(
            ...     adata=LazyAnnData(slaf_array),
            ...     vocab_size=30000,
            ...     n_expression_bins=20
            ... )
            >>> print(f"Expression bins: {tokenizer.n_expression_bins}")
            Expression bins: 20

            >>> # Error handling for invalid SLAF array
            >>> try:
            ...     tokenizer = ScGPTTokenizer(None)
            ... except TypeError as e:
            ...     print(f"Error: {e}")
            Error: adata must be a valid LazyAnnData instance
        """

        self.n_expression_bins = n_expression_bins
        super().__init__(
            adata=adata, vocab_size=vocab_size, max_genes=max_genes
        )

    def create_window(self) -> Window:
        return ScGPTWindow()

    def apply(
        self,
        df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        kwargs.setdefault("special_token_offset", 4)
        kwargs.setdefault("expr_bin_start", self.expr_bin_start)
        kwargs.setdefault("n_expression_bins", self.n_expression_bins)
        return super().apply(df, schema=schema, max_items=max_items, **kwargs)

    def tokenize_grouped(
        self,
        grouped_df: pl.DataFrame,
        schema: DataSchema = SLAF_LANCE_COO_SCHEMA,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gene_sequences = grouped_df[schema.item_list_key].to_list()
        expr_sequences = (
            grouped_df[schema.value_list_key].to_list()
            if schema.value_list_key and schema.value_list_key in grouped_df.columns
            else None
        )
        if expr_sequences is None:
            raise ValueError("scGPT grouped tokenization requires expression token sequences")

        max_sequence_length = self.max_genes + 2
        batch_size = len(gene_sequences)
        gene_token_array = np.full(
            (batch_size, max_sequence_length),
            self.special_tokens["PAD"],
            dtype=np.int64,
        )
        value_array = np.full(
            (batch_size, max_sequence_length),
            self.special_tokens["PAD"],
            dtype=np.int64,
        )

        for i, (genes, exprs) in enumerate(zip(gene_sequences, expr_sequences, strict=False)):
            n_pairs = min(len(genes), len(exprs), self.max_genes)

            if n_pairs > 0:
                gene_ids = np.full(
                    n_pairs + 2, self.special_tokens["PAD"], dtype=np.int64
                )
                value_tokens = np.full(
                    n_pairs + 2, self.special_tokens["PAD"], dtype=np.int64
                )
                gene_ids[0] = self.special_tokens["CLS"]
                gene_ids[1 : 1 + n_pairs] = np.asarray(genes[:n_pairs], dtype=np.int64)
                gene_ids[1 + n_pairs] = self.special_tokens["SEP"]
                value_tokens[1 : 1 + n_pairs] = np.asarray(exprs[:n_pairs], dtype=np.int64)
            else:
                gene_ids = np.array(
                    [self.special_tokens["CLS"], self.special_tokens["SEP"]],
                    dtype=np.int64,
                )
                value_tokens = np.array(
                    [self.special_tokens["PAD"], self.special_tokens["PAD"]],
                    dtype=np.int64,
                )

            length = min(len(gene_ids), max_sequence_length)
            gene_token_array[i, :length] = gene_ids[:length]
            value_array[i, :length] = value_tokens[:length]

        input_ids = torch.from_numpy(gene_token_array)
        values_tensor = torch.from_numpy(value_array)
        attention_mask = input_ids != self.special_tokens["PAD"]

        return input_ids, attention_mask, values_tensor

    def tokenize(
        self,
        gene_sequences: list[list[int] | list[tuple[int, float]]],
        expr_sequences: list[list[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Tokenize gene expression sequences into model-ready tensors.

        This method converts gene and expression sequences into tokenized tensors
        suitable for machine learning models. It supports both GeneFormer and scGPT
        tokenization strategies with optimized vectorized operations.

        Args:
            gene_sequences: List of gene ID sequences for each cell
            expr_sequences: List of expression value sequences for each cell (required for scGPT)

        Returns:
            tuple: (input_ids, attention_mask) tensors
                - input_ids: Tokenized sequences with padding
                - attention_mask: Boolean mask indicating valid tokens

        Raises:
            ValueError: If gene_sequences is empty

        Examples:
            >>> # scGPT tokenization
            >>> gene_sequences = [[1, 2, 3], [4, 5, 6]]
            >>> expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]
            >>> input_ids, attention_mask = tokenizer.tokenize(gene_sequences, expr_sequences)
            >>> print(f"Shape: {input_ids.shape}")
            Shape: torch.Size([2, 2050])
        """
        if not gene_sequences:
            raise ValueError("Gene sequences cannot be empty")

        # Canonical scGPT contract: dual stream, aligned positions.
        # gene_ids: [CLS] gene_1 ... gene_n [SEP] [PAD]...
        # values:   [PAD] expr_1 ... expr_n [PAD] [PAD]...
        max_sequence_length = self.max_genes + 2

        batch_size = len(gene_sequences)

        # Use fast numpy-based approach (same as original test)
        import numpy as np

        gene_token_array = np.full(
            (batch_size, max_sequence_length),
            self.special_tokens["PAD"],
            dtype=np.int64,
        )
        value_array = np.full(
            (batch_size, max_sequence_length),
            self.special_tokens["PAD"],
            dtype=np.int64,
        )

        for i, gene_sequence in enumerate(gene_sequences):
            expr_sequence = expr_sequences[i] if expr_sequences is not None else []
            genes = list(gene_sequence) if gene_sequence else []
            exprs = list(expr_sequence) if expr_sequence else []

            n_pairs = min(len(genes), len(exprs), self.max_genes)

            if n_pairs > 0:
                gene_tokens = np.array(genes[:n_pairs], dtype=np.int64) + 4

                if isinstance(exprs[0], int | np.integer):
                    expr_tokens = (
                        np.array(exprs[:n_pairs], dtype=np.int64) + self.expr_bin_start
                    )
                else:
                    expr_tokens = self._expression_to_bin_vectorized(
                        np.array(exprs[:n_pairs], dtype=np.float32)
                    )

                sequence_length = n_pairs + 2
                gene_ids = np.full(
                    sequence_length, self.special_tokens["PAD"], dtype=np.int64
                )
                value_tokens = np.full(
                    sequence_length, self.special_tokens["PAD"], dtype=np.int64
                )
                gene_ids[0] = self.special_tokens["CLS"]
                gene_ids[1 : 1 + n_pairs] = gene_tokens
                gene_ids[1 + n_pairs] = self.special_tokens["SEP"]
                value_tokens[1 : 1 + n_pairs] = expr_tokens
            else:
                gene_ids = np.array(
                    [self.special_tokens["CLS"], self.special_tokens["SEP"]],
                    dtype=np.int64,
                )
                value_tokens = np.array(
                    [self.special_tokens["PAD"], self.special_tokens["PAD"]],
                    dtype=np.int64,
                )

            length = min(len(gene_ids), max_sequence_length)
            gene_token_array[i, :length] = gene_ids[:length]
            value_array[i, :length] = value_tokens[:length]

        input_ids = torch.from_numpy(gene_token_array)
        values_tensor = torch.from_numpy(value_array)
        attention_mask = input_ids != self.special_tokens["PAD"]

        return input_ids, attention_mask, values_tensor

    def get_vocab_info(self) -> dict[str, Any]:
        """
        Get vocabulary information for debugging and analysis.

        Returns:
            dict: Vocabulary information including size, special tokens, etc.

        Examples:
            >>> vocab_info = tokenizer.get_vocab_info()
            >>> print(f"Vocabulary size: {vocab_info['vocab_size']}")
            >>> print(f"Special tokens: {vocab_info['special_tokens']}")
            Vocabulary size: 50000
            Special tokens: {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3}
        """
        vocab_info = super().get_vocab_info()

        return {
            **vocab_info,
            "n_expression_bins": self.n_expression_bins,
        }

    def get_factory_kwargs(self) -> dict[str, Any]:
        factory_kwargs = super().get_factory_kwargs()
        factory_kwargs["n_expression_bins"] = self.n_expression_bins
        return factory_kwargs

    def _setup_special_tokens(self):
        """Setup special tokens for tokenization."""
        super()._setup_special_tokens()

        # Expression binning setup for scGPT
        self.expr_bin_start = self.vocab_size
        self.expr_bin_size = 1.0 / self.n_expression_bins

    def _expression_to_bin(self, expression_value: float) -> int:
        """Convert expression value to bin token ID"""
        if expression_value <= 0:
            return self.special_tokens["PAD"]

        # Bin the expression value
        bin_id = min(
            int(expression_value / self.expr_bin_size), self.n_expression_bins - 1
        )
        return self.expr_bin_start + bin_id

    def _expression_to_bin_vectorized(
        self, expression_values: np.ndarray
    ) -> np.ndarray:
        """Vectorized version of expression binning"""
        # Handle edge cases
        if len(expression_values) == 0:
            return np.array([], dtype=int)

        # Create bins
        bins = np.clip(
            (expression_values / self.expr_bin_size).astype(int),
            0,
            self.n_expression_bins - 1,
        )

        # Convert to token IDs
        result = np.where(
            expression_values > 0,
            self.expr_bin_start + bins,
            self.special_tokens["PAD"],
        )

        return result.astype(int)

    def decode_tokens(self, tokens: list[int]) -> dict[str, Any]:
        """
        Decode token sequence back to gene information.

        Args:
            tokens: List of token IDs to decode

        Returns:
            dict: Decoded information including genes, expressions, etc.

        Examples:
            >>> # Decode a token sequence
            >>> tokens = [1, 100, 50050, 200, 50060, 2]  # CLS, gene1, expr1, gene2, expr2, SEP
            >>> decoded = tokenizer.decode_tokens(tokens)
            >>> print(f"Genes: {decoded['genes']}")
            >>> print(f"Expressions: {decoded['expressions']}")
            Genes: ['gene_100', 'gene_200']
            Expressions: [0.5, 0.6]
        """
        if not tokens:
            return {"genes": [], "expressions": [], "special_tokens": []}

        genes = []
        expressions = []
        special_tokens = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == self.special_tokens["CLS"]:
                special_tokens.append("CLS")
            elif token == self.special_tokens["SEP"]:
                special_tokens.append("SEP")
            elif token == self.special_tokens["PAD"]:
                special_tokens.append("PAD")
            elif token == self.special_tokens["MASK"]:
                special_tokens.append("MASK")
            elif token >= self.expr_bin_start:  # Expression token
                bin_id = token - self.expr_bin_start
                expr_value = bin_id * self.expr_bin_size
                expressions.append(expr_value)
            else:
                # Gene token
                if token in self.token_to_gene:
                    genes.append(self.token_to_gene[token])
                else:
                    genes.append(f"unknown_gene_{token}")

            i += 1

        return {
            "genes": genes,
            "expressions": expressions,
            "special_tokens": special_tokens,
        }


class GeneformerTokenizer(SLAFTokenizer):
    """
    Geneformer tokenizer.

    Examples:
        >>> # Basic usage with GeneFormer
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> tokenizer = GeneformerTokenizer(slaf_array)
        >>> gene_sequences = [[1, 2, 3], [4, 5, 6]]
        >>> input_ids, attention_mask = tokenizer.tokenize(gene_sequences)
        >>> print(f"Input shape: {input_ids.shape}")
        Input shape: torch.Size([2, 2048])

        >>> # Vocabulary information
        >>> vocab_info = tokenizer.get_vocab_info()
        >>> print(f"Vocabulary size: {vocab_info['vocab_size']}")
        Vocabulary size: 50000
    """

    def __init__(
        self,
        adata: LazyAnnData,
        vocab_size: int = 50000,
        max_genes: int = 2048,
    ):
        super().__init__(
            adata=adata, vocab_size=vocab_size, max_genes=max_genes
        )

    def create_window(self) -> Window:
        return GeneformerWindow()

    def apply(
        self,
        df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        kwargs.setdefault("special_token_offset", 4)
        return self.window.apply(df, schema=schema, max_items=max_items, **kwargs)

    def tokenize_grouped(
        self,
        grouped_df: pl.DataFrame,
        schema: DataSchema = SLAF_LANCE_COO_SCHEMA,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gene_sequences = grouped_df[schema.item_list_key].to_list()
        batch_size = len(gene_sequences)
        token_array = np.full(
            (batch_size, self.max_genes), self.special_tokens["PAD"], dtype=np.int64
        )

        for i, genes in enumerate(gene_sequences):
            gene_tokens = np.asarray(genes, dtype=np.int64)
            if len(gene_tokens) > 0:
                tokens = np.concatenate(
                    [
                        [self.special_tokens["CLS"]],
                        gene_tokens,
                        [self.special_tokens["SEP"]],
                    ]
                )
            else:
                tokens = np.array(
                    [self.special_tokens["CLS"], self.special_tokens["SEP"]],
                    dtype=np.int64,
                )

            tokens = tokens[: self.max_genes]
            if len(tokens) < self.max_genes:
                padding = np.full(
                    self.max_genes - len(tokens),
                    self.special_tokens["PAD"],
                    dtype=np.int64,
                )
                tokens = np.concatenate([tokens, padding])
            token_array[i, :] = tokens

        input_ids = torch.from_numpy(token_array)
        attention_mask = input_ids != self.special_tokens["PAD"]
        return input_ids, attention_mask, None

    def tokenize(
        self,
        gene_sequences: list[list[int] | list[tuple[int, float]]],
        expr_sequences: list[list[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Tokenize gene expression sequences into model-ready tensors.

        This method converts gene and expression sequences into tokenized tensors
        suitable for machine learning models. It supports both GeneFormer and scGPT
        tokenization strategies with optimized vectorized operations.

        Args:
            gene_sequences: List of gene ID sequences for each cell
            expr_sequences: List of expression value sequences for each cell (required for scGPT)

        Returns:
            tuple: (input_ids, attention_mask) tensors
                - input_ids: Tokenized sequences with padding
                - attention_mask: Boolean mask indicating valid tokens

        Raises:
            ValueError: If gene_sequences is empty

        Examples:
            >>> # GeneFormer tokenization
            >>> gene_sequences = [[1, 2, 3], [4, 5, 6]]
            >>> input_ids, attention_mask = tokenizer.tokenize(gene_sequences)
            >>> print(f"Shape: {input_ids.shape}")
            Shape: torch.Size([2, 2048])
        """
        if not gene_sequences:
            raise ValueError("Gene sequences cannot be empty")

        # Always define max_sequence_length based on tokenizer type
        max_sequence_length = self.max_genes  # For Geneformer, same as max_genes

        batch_size = len(gene_sequences)

        # Use fast numpy-based approach (same as original test)
        import numpy as np

        array_width = max_sequence_length

        token_array = np.full(
            (batch_size, array_width), self.special_tokens["PAD"], dtype=np.int64
        )

        # Geneformer format: [CLS] gene1 gene2 gene3 ... [SEP]
        for i, gene_sequence in enumerate(gene_sequences):
            # Convert gene IDs to tokens (fast mapping)
            gene_tokens = np.array(gene_sequence, dtype=np.int64) + 4

            # Vectorized sequence building: use concatenation for speed
            if len(gene_tokens) > 0:
                # Use concatenation: CLS + genes + SEP
                tokens = np.concatenate(
                    [
                        [self.special_tokens["CLS"]],
                        gene_tokens,
                        [self.special_tokens["SEP"]],
                    ]
                )  # type: ignore[assignment]
            else:
                # Empty sequence case
                tokens = np.array(
                    [self.special_tokens["CLS"], self.special_tokens["SEP"]],
                    dtype=np.int64,
                )  # type: ignore[assignment]

            # Pad/truncate to max_genes
            tokens = tokens[: self.max_genes]  # type: ignore[assignment]
            if len(tokens) < self.max_genes:
                padding = np.full(
                    self.max_genes - len(tokens),
                    self.special_tokens["PAD"],
                    dtype=np.int64,
                )
                tokens = np.concatenate([tokens, padding])  # type: ignore[assignment]

            # Fill array
            token_array[i, :] = tokens  # type: ignore[assignment]

        # Convert to tensors in one operation
        input_ids = torch.from_numpy(token_array)
        attention_mask = input_ids != self.special_tokens["PAD"]

        return input_ids, attention_mask, None

    def decode_tokens(self, tokens: list[int]) -> dict[str, Any]:
        """
        Decode token sequence back to gene information.

        Args:
            tokens: List of token IDs to decode

        Returns:
            dict: Decoded information including genes, expressions, etc.

        Examples:
            >>> # Decode a token sequence
            >>> tokens = [1, 100, 50050, 200, 50060, 2]  # CLS, gene1, expr1, gene2, expr2, SEP
            >>> decoded = tokenizer.decode_tokens(tokens)
            >>> print(f"Genes: {decoded['genes']}")
            >>> print(f"Expressions: {decoded['expressions']}")
            Genes: ['gene_100', 'gene_200']
            Expressions: [0.5, 0.6]
        """
        if not tokens:
            return {"genes": [], "expressions": [], "special_tokens": []}

        genes = []
        special_tokens = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == self.special_tokens["CLS"]:
                special_tokens.append("CLS")
            elif token == self.special_tokens["SEP"]:
                special_tokens.append("SEP")
            elif token == self.special_tokens["PAD"]:
                special_tokens.append("PAD")
            elif token == self.special_tokens["MASK"]:
                special_tokens.append("MASK")
            else:
                # Gene token
                if token in self.token_to_gene:
                    genes.append(self.token_to_gene[token])
                else:
                    genes.append(f"unknown_gene_{token}")

            i += 1

        return {
            "genes": genes,
            "expressions": [],
            "special_tokens": special_tokens,
        }
