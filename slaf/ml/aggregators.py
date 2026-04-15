"""
Window functions for SLAF data processing.

This module provides window function implementations for different tokenization strategies.
Each window function defines how to apply Polars window operations to raw COO data from a PyArrow batch.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import polars as pl

from slaf.core.tabular_schema import DataSchema


class Window(ABC):
    """
    Base class for window function implementations.

    Window functions define how to apply Polars window operations to COO data from a PyArrow batch
    for different tokenization strategies (scGPT, Geneformer, etc.).
    """

    @abstractmethod
    def apply(
        self,
        fragment_df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Apply window function to COO data from a PyArrow batch.

        Matches ``WindowProtocol`` in ``slaf.distributed.processor`` (same contract as
        ``slaf.distributed.window.Window``).

        Args:
            fragment_df: Polars DataFrame with ``schema.group_key``, ``schema.item_key``,
                ``schema.value_key``.
            schema: Column names for input and aggregated list outputs.
            max_items: Maximum items per group after ranking/filtering.
            **kwargs: Strategy-specific options (e.g. ``n_expression_bins``).

        Returns:
            Grouped DataFrame; list columns use ``schema.item_list_key`` and optionally
            ``schema.value_list_key``.
        """
        raise NotImplementedError


class ScGPTWindow(Window):
    """
    scGPT window: rank genes by expression per cell, then aggregate to paired lists.

    Output is two list columns (gene ids and expression bins or raw values)—the same
    layout ``ScGPTTokenizer`` expects for its dual-stream ``input_ids`` / ``values``
    tensors (aligned positions, not a single interleaved token sequence).
    """

    def apply(
        self,
        fragment_df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Apply scGPT window function.

        **kwargs:
            n_expression_bins: Number of expression bins (default: 10)
            use_binned_expressions: Whether to return binned expression values (default: True)
        """
        gk = schema.group_key
        ik = schema.item_key
        vk = schema.value_key
        item_out = schema.item_list_key
        value_out = schema.value_list_key or "expr_sequence"

        n_expression_bins = kwargs.get("n_expression_bins", 10)
        use_binned_expressions = kwargs.get(
            "use_binned_expressions", True
        )  # Default to True for scGPT

        if use_binned_expressions:
            grouped = (
                fragment_df.with_columns(
                    pl.col(vk)
                    .rank(method="dense", descending=True)
                    .over(gk)
                    .alias("gene_rank")
                )
                .filter(pl.col("gene_rank") <= max_items)
                .with_columns(pl.col(vk).log1p().alias("log_value"))
                .with_columns(
                    pl.when(pl.col("log_value") > 0)
                    .then(
                        (
                            pl.col("log_value")
                            * n_expression_bins
                            / pl.col("log_value").max().over(gk)
                        )
                        .floor()
                        .clip(0, n_expression_bins - 1)
                    )
                    .otherwise(0)
                    .alias("expr_bin")
                )
                .group_by(gk, maintain_order=True)
                .agg(
                    [
                        pl.col(ik).alias(item_out),
                        pl.col("expr_bin").alias(value_out),
                    ]
                )
            )
        else:
            result = fragment_df.with_columns(
                [
                    pl.col(vk)
                    .rank(method="dense", descending=True)
                    .over(gk)
                    .alias("gene_rank")
                ]
            ).filter(pl.col("gene_rank") <= max_items)

            grouped = result.group_by(gk, maintain_order=True).agg(
                [
                    pl.col(ik).alias(item_out),
                    pl.col(vk).alias(value_out),
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
        self,
        fragment_df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Apply Geneformer window function.

        **kwargs:
            min_percentile: Optional percentile filter (0-100)
        """
        gk = schema.group_key
        ik = schema.item_key
        vk = schema.value_key
        item_out = schema.item_list_key

        min_percentile = kwargs.get("min_percentile", None)

        if min_percentile is not None:
            grouped = (
                fragment_df.with_columns(
                    [
                        pl.col(vk)
                        .rank(method="dense", descending=True)
                        .over(gk)
                        .alias("gene_rank"),
                        pl.col(vk)
                        .rank(method="dense", descending=False)
                        .over(gk)
                        .alias("percentile_rank"),
                    ]
                )
                .filter(
                    (pl.col("gene_rank") <= max_items)
                    & (
                        pl.col("percentile_rank")
                        >= min_percentile
                        * pl.col("percentile_rank").max().over(gk)
                        / 100
                    )
                )
                .group_by(gk, maintain_order=True)
                .agg(
                    [
                        pl.col(ik).alias(item_out),
                    ]
                )
            )
        else:
            grouped = (
                fragment_df.with_columns(
                    [
                        pl.col(vk)
                        .rank(method="dense", descending=True)
                        .over(gk)
                        .alias("gene_rank")
                    ]
                )
                .filter(pl.col("gene_rank") <= max_items)
                .group_by(gk, maintain_order=True)
                .agg(
                    [
                        pl.col(ik).alias(item_out),
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
        self,
        fragment_df: pl.DataFrame,
        schema: DataSchema,
        max_items: int,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Rank by value, keep top ``max_items``, aggregate gene and raw value lists."""
        gk = schema.group_key
        ik = schema.item_key
        vk = schema.value_key
        item_out = schema.item_list_key
        value_out = schema.value_list_key or "expr_sequence"

        result = fragment_df.with_columns(
            [
                pl.col(vk)
                .rank(method="dense", descending=True)
                .over(gk)
                .alias("gene_rank")
            ]
        ).filter(pl.col("gene_rank") <= max_items)

        grouped = result.group_by(gk, maintain_order=True).agg(
            [
                pl.col(ik).alias(item_out),
                pl.col(vk).alias(value_out),
            ]
        )

        return grouped


class WindowType(str, Enum):
    """Window implementation selector (mirrors tokenizer strategy names)."""

    GENEFORMER = "geneformer"
    SCPGPT = "scgpt"
    SIMPLE = "simple"


def create_window(window_type: WindowType | str) -> Window:
    """Factory for ML window implementations."""
    if isinstance(window_type, str):
        window_type = WindowType(window_type.lower())
    if window_type is WindowType.GENEFORMER:
        return GeneformerWindow()
    if window_type is WindowType.SCPGPT:
        return ScGPTWindow()
    if window_type is WindowType.SIMPLE:
        return SimpleWindow()
    raise ValueError(f"Unknown window type: {window_type!r}")
