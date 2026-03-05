"""
Unit tests for BatchProcessor pipeline.

Tests focus on the full pipeline: boundary → shuffle → window → tokenize.
"""

import polars as pl

from slaf.distributed.processor import (
    BatchProcessor,
    DataSchema,
    ShuffleProtocol,
    TokenizerProtocol,
    WindowProtocol,
)
from slaf.distributed.shuffle import Shuffle
from slaf.distributed.window import Window


def create_test_dataframe(
    group_key: str = "group_id",
    item_key: str = "item_id",
    value_key: str = "value",
    n_groups: int = 5,
    items_per_group: int = 10,
) -> pl.DataFrame:
    """Create a test DataFrame with specified structure."""
    data = []
    for group_id in range(n_groups):
        for item_id in range(items_per_group):
            # Create values in descending order
            value = items_per_group - item_id
            data.append(
                {
                    group_key: group_id,
                    item_key: item_id,
                    value_key: value,
                }
            )
    return pl.DataFrame(data)


class MockWindow(WindowProtocol):
    """Mock window function for testing."""

    def apply(
        self, df: pl.DataFrame, schema: DataSchema, max_items: int, **kwargs
    ) -> pl.DataFrame:
        # Simple aggregation without ranking
        return df.group_by(schema.group_key).agg(
            [pl.col(schema.item_key).alias(schema.item_list_key)]
        )


class MockShuffle(ShuffleProtocol):
    """Mock shuffle function for testing."""

    def apply(
        self, df: pl.DataFrame, schema: DataSchema, seed: int, **kwargs
    ) -> pl.DataFrame:
        # Return as-is (no shuffling for testing)
        return df


class MockTokenizer(TokenizerProtocol):
    """Mock tokenizer for testing."""

    def __call__(self, grouped_df: pl.DataFrame, schema: DataSchema) -> dict[str, any]:
        # Return mock tokenized data
        n_groups = len(grouped_df)
        return {
            "input_ids": [[1, 2, 3] for _ in range(n_groups)],
            "attention_mask": [[1, 1, 1] for _ in range(n_groups)],
        }


class TestBatchProcessor:
    """Test cases for BatchProcessor pipeline."""

    def test_processor_initialization(self):
        """Test processor creation."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        processor = BatchProcessor(schema=schema)

        assert processor.schema == schema
        assert processor.window is None
        assert processor.shuffle is None
        assert processor.tokenizer is None
        assert processor.max_items == 1024
        assert processor.seed == 42

    def test_process_batch_basic(self):
        """Test basic batch processing (boundary → shuffle → window → tokenize)."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        shuffle = Shuffle()
        tokenizer = MockTokenizer()
        processor = BatchProcessor(
            schema=schema,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            max_items=5,
        )

        batch = create_test_dataframe(n_groups=3, items_per_group=10)
        # Set is_partition_exhausted=True so all groups are complete
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Should return list of samples (one per group)
        assert isinstance(samples, list)
        assert len(samples) == 3

        # Each sample should have tokenized format
        for sample in samples:
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert "group_key" in sample

    def test_process_batch_raw_mode(self):
        """Test raw mode (no tokenization)."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(
            schema=schema, window=window, tokenizer=None, max_items=5
        )

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        # Set is_partition_exhausted=True so all groups are complete
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Should return list of samples
        assert isinstance(samples, list)
        assert len(samples) == 2

        # Each sample should have raw format
        for sample in samples:
            assert "grouped" in sample
            assert "group_key" in sample
            assert isinstance(sample["grouped"], pl.DataFrame)

    def test_process_batch_with_tokenizer(self):
        """Test with tokenizer."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        tokenizer = MockTokenizer()
        processor = BatchProcessor(
            schema=schema, window=window, tokenizer=tokenizer, max_items=5
        )

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0)

        # Should have tokenized format
        for sample in samples:
            assert "input_ids" in sample
            assert "attention_mask" in sample

    def test_process_batch_empty_input(self):
        """Test edge case with empty input."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        processor = BatchProcessor(schema=schema)

        empty_batch = pl.DataFrame(
            {
                "group_id": [],
                "item_id": [],
                "value": [],
            }
        )
        samples = processor.process_batch([empty_batch], epoch=0)

        # Should return empty list
        assert samples == []

    def test_process_batch_all_partial_groups(self):
        """Test when all groups are partial."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        processor = BatchProcessor(schema=schema)

        # Create batch with single group (will be partial)
        batch = create_test_dataframe(n_groups=1, items_per_group=5)
        samples = processor.process_batch(
            [batch], epoch=0, is_partition_exhausted=False
        )

        # Should return empty list (all groups are partial)
        assert samples == []

    def test_process_batch_partition_exhausted(self):
        """Test with is_partition_exhausted=True."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(schema=schema, window=window, max_items=5)

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Should return samples (all groups are complete when partition exhausted)
        assert len(samples) == 2

    def test_process_batch_multiple_samples(self):
        """Test that returns list of samples (one per group)."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(schema=schema, window=window, max_items=5)

        batch = create_test_dataframe(n_groups=5, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Should have 5 samples (one per group)
        assert len(samples) == 5

    def test_process_batch_sample_format_tokenized(self):
        """Test tokenized sample format."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        tokenizer = MockTokenizer()
        processor = BatchProcessor(
            schema=schema, window=window, tokenizer=tokenizer, max_items=5
        )

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Check format
        for sample in samples:
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert "group_key" in sample
            assert isinstance(sample["input_ids"], list)
            assert isinstance(sample["attention_mask"], list)

    def test_process_batch_sample_format_raw(self):
        """Test raw sample format."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(
            schema=schema, window=window, tokenizer=None, max_items=5
        )

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Check format
        for sample in samples:
            assert "grouped" in sample
            assert "group_key" in sample
            assert isinstance(sample["grouped"], pl.DataFrame)

    def test_process_batch_without_window(self):
        """Test without window function."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        processor = BatchProcessor(schema=schema, window=None)

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Should still work (simple aggregation)
        assert len(samples) == 2

    def test_process_batch_without_shuffle(self):
        """Test without shuffle function."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(
            schema=schema, window=window, shuffle=None, max_items=5
        )

        batch = create_test_dataframe(n_groups=2, items_per_group=10)
        samples = processor.process_batch([batch], epoch=0, is_partition_exhausted=True)

        # Should still work
        assert len(samples) == 2

    def test_process_batch_boundary_merging(self):
        """Test boundary merging integration."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(schema=schema, window=window, max_items=5)

        # First batch: groups 0, 1 (group 1 is partial)
        batch1 = create_test_dataframe(n_groups=2, items_per_group=5)
        samples1 = processor.process_batch(
            [batch1], epoch=0, is_partition_exhausted=False
        )

        # Second batch: group 1 continues
        batch2 = create_test_dataframe(n_groups=2, items_per_group=5)
        # Adjust group IDs to continue from batch1
        batch2 = batch2.with_columns((pl.col("group_id") + 1).alias("group_id"))
        samples2 = processor.process_batch(
            [batch2], epoch=0, is_partition_exhausted=True
        )

        # Should process both batches
        assert isinstance(samples1, list)
        assert isinstance(samples2, list)

    def test_process_batch_multiple_batches(self):
        """Test processing multiple batches at once."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        processor = BatchProcessor(schema=schema, window=window, max_items=5)

        batch1 = create_test_dataframe(n_groups=2, items_per_group=5)
        batch2 = create_test_dataframe(n_groups=2, items_per_group=5)
        # Adjust group IDs
        batch2 = batch2.with_columns((pl.col("group_id") + 2).alias("group_id"))

        samples = processor.process_batch(
            [batch1, batch2], epoch=0, is_partition_exhausted=True
        )

        # Should process all groups from both batches
        assert len(samples) == 4
