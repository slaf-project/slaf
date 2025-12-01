"""
Generic batch processor for distributed dataloading.

Uses Protocols (not concrete classes) to avoid circular imports.
"""

from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

if TYPE_CHECKING:
    from slaf.distributed.boundary import GroupBoundaryHandler


class DataSchema:
    """
    Generic schema configuration for tabular data processing.

    Describes the structure of input data and output aggregations.
    The input data must be tabular with at least three columns:
    - group_key: Column to group by
    - item_key: Column for items within groups
    - value_key: Column for values

    Output columns after aggregation:
    - group_key_out: Output group key (defaults to group_key if None)
    - item_list_key: Aggregated list of items per group
    - value_list_key: Optional aggregated list of values per group
    """

    # Input columns (required)
    group_key: str
    item_key: str
    value_key: str

    # Output columns (after window/aggregation)
    group_key_out: str | None
    item_list_key: str
    value_list_key: str | None

    def __init__(
        self,
        group_key: str,
        item_key: str,
        value_key: str,
        group_key_out: str | None = None,
        item_list_key: str = "item_list",
        value_list_key: str | None = None,
    ):
        """
        Initialize data schema.

        Args:
            group_key: Column to group by
            item_key: Column for items within groups
            value_key: Column for values
            group_key_out: Output group key (defaults to group_key if None)
            item_list_key: Aggregated list of items per group
            value_list_key: Optional aggregated list of values per group
        """
        self.group_key = group_key
        self.item_key = item_key
        self.value_key = value_key
        self.group_key_out = group_key_out
        self.item_list_key = item_list_key
        self.value_list_key = value_list_key


class WindowProtocol(Protocol):
    """
    Protocol for window functions - no import needed.

    Window functions perform: group_by → rank/filter → aggregate
    Input: Tabular DataFrame with group_key, item_key, value_key columns
    Output: DataFrame with group_key and aggregated item_list (and optionally value_list)
    """

    def apply(
        self, df: pl.DataFrame, schema: DataSchema, max_items: int, **kwargs: Any
    ) -> pl.DataFrame:
        """
        Apply window function to tabular data.

        Args:
            df: Input DataFrame with columns from schema (group_key, item_key, value_key)
            schema: Data schema configuration
            max_items: Maximum items to keep per group after ranking/filtering
            **kwargs: Additional window-specific parameters

        Returns:
            DataFrame with group_key and aggregated columns (item_list, optionally value_list)
        """
        ...


class ShuffleProtocol(Protocol):
    """
    Protocol for shuffle functions - no import needed.

    Shuffle functions randomize the order of groups in the DataFrame.
    """

    def apply(
        self, df: pl.DataFrame, schema: DataSchema, seed: int, **kwargs: Any
    ) -> pl.DataFrame | list[pl.DataFrame]:
        """
        Apply shuffling to DataFrame.

        Args:
            df: Input DataFrame
            schema: Data schema (used to identify group_key for shuffling)
            seed: Random seed for reproducibility
            **kwargs: Additional shuffle-specific parameters

        Returns:
            Shuffled DataFrame (or list of DataFrames if chunking)
        """
        ...


class TokenizerProtocol(Protocol):
    """
    Protocol for tokenizers - no import needed.

    Tokenizers convert aggregated sequences into model inputs.
    """

    def __call__(self, grouped_df: pl.DataFrame, schema: DataSchema) -> dict[str, Any]:
        """
        Tokenize grouped DataFrame.

        Args:
            grouped_df: DataFrame with group_key and aggregated columns
            schema: Data schema (used to identify columns)

        Returns:
            Dictionary with tokenized data (format depends on implementation)
        """
        ...


class BatchProcessor:
    """
    Generic batch processor for distributed dataloading.

    Performs: Boundary Handling → Shuffle → Window → Tokenize pipeline on tabular data.
    Completely generic - works with any data structure as long as schema is specified.
    """

    def __init__(
        self,
        schema: DataSchema,
        window: WindowProtocol | None = None,
        shuffle: ShuffleProtocol | None = None,
        tokenizer: TokenizerProtocol | None = None,
        boundary_handler: "GroupBoundaryHandler | None" = None,
        max_items: int = 1024,  # Max items per group after window function
        seed: int = 42,
        continuity_check: str = "sequential",  # How to detect continuity between batches
        **window_kwargs: Any,
    ):
        self.schema = schema
        self.window = window
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_items = max_items
        self.seed = seed
        self.window_kwargs = window_kwargs
        self.batch_id = 0
        self.current_epoch = 0

        # Initialize boundary handler (or use provided one)
        if boundary_handler is None:
            from slaf.distributed.boundary import GroupBoundaryHandler

            self.boundary_handler = GroupBoundaryHandler(
                schema=schema,
                continuity_check=continuity_check,
            )
        else:
            self.boundary_handler = boundary_handler

        # Track partial groups across batches
        self.partial_groups: dict[Any, pl.DataFrame] = {}

    def process_batch(
        self,
        raw_batches: list[pl.DataFrame],  # Raw tabular data with schema columns
        epoch: int = 0,
        partition_id: int | None = None,  # Optional partition ID for boundary tracking
        is_partition_exhausted: bool = False,  # If True, all groups in this batch are complete
    ) -> list[dict[str, Any]]:
        """
        Process raw batches through pipeline: Boundary Handling → Shuffle → Window → Tokenize.

        Args:
            raw_batches: List of DataFrames with tabular data (group_key, item_key, value_key)
            epoch: Current epoch number (for seed variation)
            partition_id: Optional partition/reader ID for tracking boundaries across partitions

        Returns:
            List of sample dictionaries (one per group). Each sample contains:
            - For tokenized: input_ids, attention_mask, group_key
            - For raw: grouped (DataFrame), group_key
        """
        # Combine raw batches
        combined_df = pl.concat(raw_batches, how="vertical")

        # Handle group boundaries: merge partial groups and track new partials
        # Pass is_partition_exhausted to boundary handler
        complete_df, self.partial_groups = self.boundary_handler.merge_partial_data(
            self.partial_groups,
            combined_df,
            partition_id=partition_id,
            is_partition_exhausted=is_partition_exhausted,
        )

        # If no complete groups, return empty list (partial groups will be handled in next batch)
        if len(complete_df) == 0:
            return []  # Return empty list instead of {"empty": True}

        # Apply shuffle (if provided) - shuffles groups
        if self.shuffle:
            shuffled_df = self.shuffle.apply(
                complete_df,  # Use complete_df (after boundary handling)
                schema=self.schema,
                seed=self.seed + self.batch_id + epoch * 10000,
            )
            if isinstance(shuffled_df, list):
                shuffled_df = pl.concat(shuffled_df, how="vertical")
        else:
            shuffled_df = complete_df  # Use complete_df (after boundary handling)

        # Apply window (if provided) - groups, ranks, filters, aggregates
        if self.window:
            grouped = self.window.apply(
                shuffled_df,
                schema=self.schema,
                max_items=self.max_items,
                **self.window_kwargs,
            )
        else:
            # No window - simple group by and aggregate
            group_key = self.schema.group_key
            item_key = self.schema.item_key
            item_list_key = self.schema.item_list_key

            # Aggregate items and optionally values in a single group_by operation
            agg_exprs = [pl.col(item_key).alias(item_list_key)]
            if self.schema.value_list_key and self.schema.value_key:
                agg_exprs.append(
                    pl.col(self.schema.value_key).alias(self.schema.value_list_key)
                )

            grouped = shuffled_df.group_by(group_key).agg(agg_exprs)

        # Apply tokenizer (if provided)
        group_key_out = self.schema.group_key_out or self.schema.group_key
        group_keys = grouped[group_key_out].to_list()

        if self.tokenizer:
            # Tokenize all groups at once
            tokenized = self.tokenizer(grouped, self.schema)
            # Split into individual samples (one per group)
            # tokenized has input_ids and attention_mask as lists/tensors
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Convert to list of samples using list comprehension (vectorized)
            # Handle both list/tensor and scalar cases
            is_indexable = hasattr(input_ids, "__getitem__") and hasattr(
                attention_mask, "__getitem__"
            )

            if is_indexable:
                # Vectorized: create all samples at once
                # Pre-extract keys to avoid repeated iteration
                other_keys = [
                    key
                    for key in tokenized.keys()
                    if key not in ["input_ids", "attention_mask"]
                ]
                other_values = [tokenized[key] for key in other_keys]

                samples = [
                    {
                        "input_ids": input_ids[idx],
                        "attention_mask": attention_mask[idx],
                        "group_key": group_key,
                        **{
                            key: (
                                value[idx]
                                if hasattr(value, "__getitem__") and len(value) > idx
                                else value
                            )
                            for key, value in zip(other_keys, other_values, strict=True)
                        },
                    }
                    for idx, group_key in enumerate(group_keys)
                ]
            else:
                # Scalar case: all groups share the same tokenized output
                sample_base = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    **{
                        key: value
                        for key, value in tokenized.items()
                        if key not in ["input_ids", "attention_mask"]
                    },
                }
                samples = [
                    {**sample_base, "group_key": group_key} for group_key in group_keys
                ]
        else:
            # Return raw grouped data - split into individual samples
            # Use partition_by for efficient splitting (vectorized)
            # This is more efficient than filtering per group
            if len(group_keys) > 0:
                # Use partition_by to split DataFrame by group_key efficiently
                # This creates a dict mapping group_key -> DataFrame
                # Note: partition_by returns tuple keys (even for single column)
                partitioned = grouped.partition_by(
                    group_key_out, as_dict=True, maintain_order=True
                )
                samples = [
                    {
                        "grouped": partitioned.get((group_key,), pl.DataFrame()),
                        "group_key": group_key,
                    }
                    for group_key in group_keys
                    if (group_key,) in partitioned
                ]
            else:
                samples = []

        self.batch_id += 1
        return samples  # Return list of samples instead of single dict
