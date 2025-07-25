"""
Shuffling strategies for SLAF data processing.

This module provides shuffling strategy implementations for different sampling approaches.
Each shuffle strategy defines how to shuffle cells within fragments.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

import polars as pl


class ShuffleType(str, Enum):
    """Shuffle strategy types"""

    RANDOM = "random"
    STRATIFIED = "stratified"


class Shuffle(ABC):
    """
    Base class for shuffling strategy implementations.

    Shuffle strategies define how to shuffle cells within fragments for
    different sampling approaches (random, stratified, etc.).
    """

    @abstractmethod
    def apply(
        self,
        df: pl.DataFrame,
        seed: int,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame | list[pl.DataFrame]:
        """
        Apply shuffling strategy to a Polars DataFrame.

        Args:
            df: Polars DataFrame to shuffle
            seed: Random seed for reproducible shuffling
            batch_size: If provided, return list of chunked DataFrames. If None, return single DataFrame.
            **kwargs: Additional strategy-specific parameters

        Returns:
            If batch_size is provided: List of pre-chunked Polars DataFrames
            If batch_size is None: Single shuffled Polars DataFrame
        """
        raise NotImplementedError


class RandomShuffle(Shuffle):
    """
    Random shuffling: shuffle cells randomly within fragments.

    This strategy randomly shuffles cells within each fragment, which is
    useful for training to avoid bias from data ordering.
    """

    def apply(
        self,
        df: pl.DataFrame,
        seed: int,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame | list[pl.DataFrame]:
        """
        Apply random shuffling using Polars operations for performance.

        Args:
            df: Polars DataFrame to shuffle
            seed: Random seed for reproducible shuffling
            batch_size: If provided, return list of chunked DataFrames. If None, return single DataFrame.
            **kwargs: Additional parameters (unused)

        Returns:
            If batch_size is provided: List of pre-chunked Polars DataFrames
            If batch_size is None: Single shuffled Polars DataFrame
        """
        import random

        # Handle empty DataFrame
        if len(df) == 0:
            return [] if batch_size is not None else df

        # Set seed for reproducible shuffling
        random.seed(seed)

        # Efficient block shuffling for pre-sorted data
        # Partition by cell_integer_id (fast since data is pre-sorted)
        chunks = df.partition_by("cell_integer_id", as_dict=False)

        # Shuffle the list of chunks
        random.shuffle(chunks)

        if batch_size is not None:
            # Chunked mode: return list of DataFrames
            return [
                pl.concat(chunks[i : i + batch_size])
                for i in range(0, len(chunks), batch_size)
            ]
        else:
            # Single DataFrame mode: concatenate all chunks
            return pl.concat(chunks)


class StratifiedShuffle(Shuffle):
    """
    Stratified shuffling: shuffle while maintaining cell type balance.

    This strategy shuffles cells while maintaining the balance of cell types
    within each batch. Requires cell type information.
    """

    def apply(
        self,
        df: pl.DataFrame,
        seed: int,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame | list[pl.DataFrame]:
        """
        Apply stratified shuffling.

        Args:
            df: Polars DataFrame to shuffle
            seed: Random seed for reproducible shuffling
            batch_size: If provided, return list of chunked DataFrames. If None, return single DataFrame.
            **kwargs: Additional parameters:
                - cell_type_column: Column name containing cell types

        Returns:
            If batch_size is provided: List of pre-chunked Polars DataFrames
            If batch_size is None: Single stratified shuffled Polars DataFrame
        """
        import random

        # Handle empty DataFrame
        if len(df) == 0:
            return [] if batch_size is not None else df

        cell_type_column = kwargs.get("cell_type_column", "cell_type")

        if cell_type_column not in df.columns:
            # Fall back to random shuffling if cell type column not found
            return RandomShuffle().apply(df, seed, batch_size, **kwargs)

        # Set seed for reproducible shuffling
        random.seed(seed)

        # Group by cell type and shuffle within each group
        shuffled_groups = []

        for cell_type in df[cell_type_column].unique():
            group_df = df.filter(pl.col(cell_type_column) == cell_type)
            if len(group_df) > 0:
                # Shuffle this group
                shuffle_keys = [random.random() for _ in range(len(group_df))]
                shuffled_group = (
                    group_df.with_columns(pl.lit(shuffle_keys).alias("shuffle_key"))
                    .sort("shuffle_key")
                    .drop("shuffle_key")
                )
                shuffled_groups.append(shuffled_group)

        # Interleave groups to maintain balance
        if not shuffled_groups:
            return [] if batch_size is not None else df

        max_rows = max(len(group) for group in shuffled_groups)
        interleaved_rows = []

        for i in range(max_rows):
            for group in shuffled_groups:
                if i < len(group):
                    interleaved_rows.append(group.row(i, named=True))

        result_df = pl.DataFrame(interleaved_rows)

        if batch_size is not None:
            # Chunked mode: partition by cell_integer_id and return list of DataFrames
            chunks = result_df.partition_by("cell_integer_id", as_dict=False)
            return [
                pl.concat(chunks[i : i + batch_size])
                for i in range(0, len(chunks), batch_size)
            ]
        else:
            # Single DataFrame mode: return the interleaved DataFrame
            return result_df


# Factory function for creating shuffle strategies
def create_shuffle(
    shuffle_type: ShuffleType | Literal["random", "stratified"],
) -> Shuffle:
    """
    Create a shuffle strategy based on the specified type.

    Args:
        shuffle_type: Type of shuffle strategy to create

    Returns:
        Shuffle strategy instance

    Raises:
        ValueError: If shuffle_type is not supported
    """
    if isinstance(shuffle_type, str):
        try:
            shuffle_type = ShuffleType(shuffle_type.lower())
        except ValueError as err:
            raise ValueError(f"Unsupported shuffle type: {shuffle_type}") from err

    if shuffle_type == ShuffleType.RANDOM:
        return RandomShuffle()  # Now uses optimized Polars approach
    elif shuffle_type == ShuffleType.STRATIFIED:
        return StratifiedShuffle()
    else:
        raise ValueError(f"Unsupported shuffle type: {shuffle_type}")
