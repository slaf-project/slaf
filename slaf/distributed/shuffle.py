"""
Generic Shuffle implementation for distributed dataloading.

This is a generic implementation that doesn't depend on slaf.ml or any
domain-specific concepts.
"""

import random
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from slaf.distributed.processor import DataSchema


class Shuffle:
    """
    Generic shuffle function.

    Randomly shuffles groups in the DataFrame.
    Works with any tabular data structure using DataSchema.
    """

    def apply(
        self, df: pl.DataFrame, schema: "DataSchema", seed: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply generic shuffle: randomly shuffle groups.

        Args:
            df: Input DataFrame
            schema: Data schema (used to identify group_key)
            seed: Random seed for reproducibility
            **kwargs: Ignored (for compatibility with protocol)

        Returns:
            Shuffled DataFrame
        """
        # Handle empty DataFrame
        if len(df) == 0:
            return df

        # Partition by group_key (fast for pre-sorted data)
        chunks = df.partition_by(schema.group_key, as_dict=False)

        # Shuffle the list of chunks
        random.seed(seed)
        random.shuffle(chunks)

        # Concatenate back
        return pl.concat(chunks, how="vertical")
