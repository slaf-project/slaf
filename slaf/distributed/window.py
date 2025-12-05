"""
Generic Window implementation for distributed dataloading.

This is a generic implementation that doesn't depend on slaf.ml or any
domain-specific concepts.
"""

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from slaf.distributed.processor import DataSchema


class Window:
    """
    Generic window function.

    Performs: group_by → rank by value → filter top N → aggregate into lists.
    Works with any tabular data structure using DataSchema.
    """

    def apply(
        self, df: pl.DataFrame, schema: "DataSchema", max_items: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply generic window function: rank items by value within each group.

        Args:
            df: Input DataFrame with group_key, item_key, value_key columns
            schema: Data schema configuration
            max_items: Maximum items to keep per group
            **kwargs: Ignored (for compatibility with protocol)

        Returns:
            DataFrame with group_key and aggregated item_list (and optionally value_list)
        """
        # Rank items by value within each group
        ranked = df.with_columns(
            [
                pl.col(schema.value_key)
                .rank(method="dense", descending=True)
                .over(schema.group_key)
                .alias("rank")
            ]
        )

        # Filter to top N items per group
        filtered = ranked.filter(pl.col("rank") <= max_items)

        # Aggregate into lists (items and optionally values in single group_by)
        agg_exprs = [pl.col(schema.item_key).alias(schema.item_list_key)]
        if schema.value_list_key:
            agg_exprs.append(pl.col(schema.value_key).alias(schema.value_list_key))

        grouped = filtered.group_by(schema.group_key).agg(agg_exprs)

        return grouped
