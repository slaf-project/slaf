"""
Window functions for SLAF data processing.

This module provides window function implementations for different tokenization strategies.
Each window function defines how to apply Polars window operations to raw COO data from a PyArrow batch.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

import polars as pl


class WindowType(str, Enum):
    """Window function types"""

    SCPGPT = "scgpt"
    GENEFORMER = "geneformer"


class Window(ABC):
    """
    Base class for window function implementations.

    Window functions define how to apply Polars window operations to COO data from a PyArrow batch
    for different tokenization strategies (scGPT, Geneformer, etc.).
    """

    @abstractmethod
    def apply(
        self, fragment_df: pl.DataFrame, max_genes: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply window function to COO data from a PyArrow batch.

        Args:
            fragment_df: Polars DataFrame containing COO data from a PyArrow batch
            max_genes: Maximum number of genes to include per cell
            **kwargs: Additional strategy-specific parameters

        Returns:
            Processed DataFrame with window function results
        """
        raise NotImplementedError


class ScGPTWindow(Window):
    """
    scGPT window function: ROW_NUMBER() with expression values for binning.

    This window function ranks genes by expression within each cell and includes
    expression values for scGPT's gene-expression interleaved format.
    """

    def apply(
        self, fragment_df: pl.DataFrame, max_genes: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply scGPT window function.

        Args:
            fragment_df: Polars DataFrame containing raw COO data from a PyArrow batch
            max_genes: Maximum number of genes to include per cell
            **kwargs: Additional parameters:
                - n_expression_bins: Number of expression bins (default: 10)
                - use_binned_expressions: Whether to return binned expression values (default: False)

        Returns:
            DataFrame with gene sequences and expression sequences (raw or binned) for scGPT format
        """
        n_expression_bins = kwargs.get("n_expression_bins", 10)
        use_binned_expressions = kwargs.get(
            "use_binned_expressions", True
        )  # Default to True for scGPT

        if use_binned_expressions:
            # Optimized version - single with_columns chain with early filtering
            grouped = (
                fragment_df
                # Early filter to reduce data volume for subsequent operations
                .with_columns(
                    pl.col("value")
                    .rank(method="dense", descending=True)
                    .over("cell_integer_id")
                    .alias("gene_rank")
                )
                .filter(pl.col("gene_rank") <= max_genes)  # Filter early!
                # Now compute everything else on the reduced dataset
                .with_columns(
                    [
                        # Log transform for binning
                        pl.col("value").log1p().alias("log_value"),
                    ]
                )
                .with_columns(
                    # Expression binning based on actual log values
                    pl.when(pl.col("log_value") > 0)
                    .then(
                        (
                            pl.col("log_value")
                            * n_expression_bins
                            / pl.col("log_value").max().over("cell_integer_id")
                        )
                        .floor()
                        .clip(0, n_expression_bins - 1)
                    )
                    .otherwise(0)
                    .alias("expr_bin")
                )
                .group_by("cell_integer_id")
                .agg(
                    [
                        pl.col("gene_integer_id").alias("gene_sequence"),
                        pl.col("expr_bin").alias("expr_sequence"),
                    ]
                )
            )
        else:
            # Fast path: simple ranking without expression binning (matches SimpleWindow)
            result = fragment_df.with_columns(
                [
                    pl.col("value")
                    .rank(method="dense", descending=True)
                    .over("cell_integer_id")
                    .alias("gene_rank")
                ]
            ).filter(pl.col("gene_rank") <= max_genes)

            # Group by cell and create separate columns
            grouped = result.group_by("cell_integer_id").agg(
                [
                    pl.col("gene_integer_id").alias("gene_sequence"),
                    pl.col("value").alias("expr_sequence"),
                ]
            )

        return grouped


class GeneformerWindow(Window):
    """
    Geneformer window function: RANK() with optional percentile filtering.

    This window function ranks genes by expression within each cell for
    Geneformer's ranked gene token format.
    """

    def apply(
        self, fragment_df: pl.DataFrame, max_genes: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply Geneformer window function.

        Args:
            fragment_df: Polars DataFrame containing raw COO data from a PyArrow batch
            max_genes: Maximum number of genes to include per cell
            **kwargs: Additional parameters:
                - min_percentile: Optional percentile filter (0-100)

        Returns:
            DataFrame with ranked gene sequences for Geneformer format
        """
        min_percentile = kwargs.get("min_percentile", None)

        if min_percentile is not None:
            # Optimized version - early filtering to reduce data volume
            grouped = (
                fragment_df
                # Compute both ranks in one pass
                .with_columns(
                    [
                        pl.col("value")
                        .rank(method="dense", descending=True)
                        .over("cell_integer_id")
                        .alias("gene_rank"),
                        pl.col("value")
                        .rank(method="dense", descending=False)
                        .over("cell_integer_id")
                        .alias("percentile_rank"),
                    ]
                )
                # Early filter to reduce data volume for subsequent operations
                .filter(
                    (pl.col("gene_rank") <= max_genes)
                    & (
                        pl.col("percentile_rank")
                        >= min_percentile
                        * pl.col("percentile_rank").max().over("cell_integer_id")
                        / 100
                    )
                )
                .group_by("cell_integer_id")
                .agg(
                    [
                        pl.col("gene_integer_id").alias("gene_sequence"),
                    ]
                )
            )
        else:
            # Standard ranking without percentile filtering
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
                    ]
                )
            )

        return grouped


class SimpleWindow(Window):
    """
    Simple, fast window function that matches the original performance.

    This uses the same approach as the test file for maximum performance.
    """

    def apply(
        self, fragment_df: pl.DataFrame, max_genes: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply simple window function for maximum performance.

        Args:
            fragment_df: Polars DataFrame containing raw COO data
            max_genes: Maximum number of genes to include per cell
            **kwargs: Additional parameters (ignored for performance)

        Returns:
            DataFrame with gene sequences
        """
        # Simple ranking approach (exactly same as test file)
        result = fragment_df.with_columns(
            [
                pl.col("value")
                .rank(method="dense", descending=True)
                .over("cell_integer_id")
                .alias("gene_rank")
            ]
        ).filter(pl.col("gene_rank") <= max_genes)

        # Group by cell and create gene sequences (exactly same as test file)
        grouped = result.group_by("cell_integer_id").agg(
            [
                pl.col("gene_integer_id").alias("gene_sequence"),
                pl.col("value").alias("expr_sequence"),
            ]
        )

        return grouped


# Factory function for creating window functions
def create_window(window_type: WindowType | Literal["scgpt", "geneformer"]) -> Window:
    """
    Create a window function based on the specified type.

    Args:
        window_type: Type of window function to create

    Returns:
        Window function instance

    Raises:
        ValueError: If window_type is not supported
    """
    if isinstance(window_type, str):
        try:
            window_type = WindowType(window_type.lower())
        except ValueError as err:
            raise ValueError(f"Unsupported window type: {window_type}") from err

    if window_type == WindowType.SCPGPT:
        return ScGPTWindow()  # Now optimized with fast path
    elif window_type == WindowType.GENEFORMER:
        return GeneformerWindow()  # Use proper Geneformer window
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
