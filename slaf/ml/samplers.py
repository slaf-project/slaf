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
    def apply(self, cell_integer_ids: list[int], seed: int, **kwargs: Any) -> list[int]:
        """
        Apply shuffling strategy to cell integer IDs.

        Args:
            cell_integer_ids: List of cell integer IDs to shuffle
            seed: Random seed for reproducible shuffling
            **kwargs: Additional strategy-specific parameters

        Returns:
            Shuffled list of cell integer IDs
        """
        raise NotImplementedError


class RandomShuffle(Shuffle):
    """
    Random shuffling: shuffle cells randomly within fragments.

    This strategy randomly shuffles cells within each fragment, which is
    useful for training to avoid bias from data ordering.
    """

    def apply(self, cell_integer_ids: list[int], seed: int, **kwargs: Any) -> list[int]:
        """
        Apply random shuffling using Polars operations for performance.

        Args:
            cell_integer_ids: List of cell integer IDs to shuffle
            seed: Random seed for reproducible shuffling
            **kwargs: Additional parameters (unused)

        Returns:
            Randomly shuffled list of cell integer IDs
        """
        import random

        # Set seed for reproducible shuffling
        random.seed(seed)

        # Create shuffle keys (same as original test)
        shuffle_keys = [random.random() for _ in range(len(cell_integer_ids))]

        # Create DataFrame with cell IDs and shuffle keys
        shuffle_df = pl.DataFrame(
            {"cell_integer_id": cell_integer_ids, "shuffle_key": shuffle_keys}
        )

        # Sort by shuffle key and return cell IDs
        shuffled_df = shuffle_df.sort("shuffle_key")
        return shuffled_df["cell_integer_id"].to_list()


class StratifiedShuffle(Shuffle):
    """
    Stratified shuffling: shuffle while maintaining cell type balance.

    This strategy shuffles cells while maintaining the balance of cell types
    within each batch. Requires cell type information.
    """

    def apply(self, cell_integer_ids: list[int], seed: int, **kwargs: Any) -> list[int]:
        """
        Apply stratified shuffling.

        Args:
            cell_integer_ids: List of cell integer IDs to shuffle
            seed: Random seed for reproducible shuffling
            **kwargs: Additional parameters:
                - cell_types: List of cell types corresponding to cell_integer_ids
                - n_strata: Number of strata to maintain balance

        Returns:
            Stratified shuffled list of cell integer IDs
        """
        import random

        cell_types = kwargs.get("cell_types", None)
        # n_strata = kwargs.get("n_strata", 10)  # Unused variable

        if cell_types is None or len(cell_types) != len(cell_integer_ids):
            # Fall back to random shuffling if cell types not provided
            return RandomShuffle().apply(cell_integer_ids, seed, **kwargs)

        # Set seed for reproducible shuffling
        random.seed(seed)

        # Group cells by type
        cell_type_groups: dict[Any, list[int]] = {}
        for cell_id, cell_type in zip(cell_integer_ids, cell_types, strict=False):
            if cell_type not in cell_type_groups:
                cell_type_groups[cell_type] = []
            cell_type_groups[cell_type].append(cell_id)

        # Shuffle within each cell type group
        for cell_type in cell_type_groups:
            random.shuffle(cell_type_groups[cell_type])

        # Interleave cells from different types to maintain balance
        shuffled_ids: list[int] = []

        # Handle empty cell type groups
        if not cell_type_groups:
            return shuffled_ids

        max_cells_per_type = max(len(cells) for cells in cell_type_groups.values())

        for i in range(max_cells_per_type):
            for cell_type in cell_type_groups:
                if i < len(cell_type_groups[cell_type]):
                    shuffled_ids.append(cell_type_groups[cell_type][i])

        return shuffled_ids


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
