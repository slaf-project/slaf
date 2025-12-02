"""
Unit tests for Shuffle function logic.

Tests focus on shuffling behavior: randomize order of groups.
"""

import polars as pl

from slaf.distributed.processor import DataSchema
from slaf.distributed.shuffle import Shuffle


def create_test_dataframe(
    group_key: str = "group_id",
    n_groups: int = 10,
    items_per_group: int = 5,
) -> pl.DataFrame:
    """Create a test DataFrame with specified structure."""
    data = []
    for group_id in range(n_groups):
        for item_id in range(items_per_group):
            data.append(
                {
                    group_key: group_id,
                    "item_id": item_id,
                    "value": item_id,
                }
            )
    return pl.DataFrame(data)


class TestShuffle:
    """Test cases for Shuffle function logic."""

    def test_shuffle_basic(self):
        """Test basic shuffling of groups."""
        df = create_test_dataframe(n_groups=10, items_per_group=3)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()
        seed = 42

        result = shuffle.apply(df, schema, seed)

        # Should have same number of rows
        assert len(result) == len(df)

        # Should have same groups (just reordered)
        original_groups = df["group_id"].unique().sort().to_list()
        result_groups = result["group_id"].unique().sort().to_list()
        assert original_groups == result_groups

        # Groups should be in different order (with high probability)
        original_order = df["group_id"].head(3).to_list()
        result_order = result["group_id"].head(3).to_list()
        # With 10 groups, probability of same order is very low
        assert original_order != result_order

    def test_shuffle_deterministic(self):
        """Test that same seed produces same shuffle."""
        df = create_test_dataframe(n_groups=10, items_per_group=3)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()
        seed = 42

        result1 = shuffle.apply(df, schema, seed)
        result2 = shuffle.apply(df, schema, seed)

        # Should produce identical results
        assert result1.equals(result2)

    def test_shuffle_different_seeds(self):
        """Test that different seeds produce different shuffles."""
        df = create_test_dataframe(n_groups=10, items_per_group=3)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()

        result1 = shuffle.apply(df, schema, seed=42)
        result2 = shuffle.apply(df, schema, seed=123)

        # Should produce different results (with high probability)
        order1 = result1["group_id"].head(10).to_list()
        order2 = result2["group_id"].head(10).to_list()
        assert order1 != order2

    def test_shuffle_single_group(self):
        """Test edge case with single group."""
        df = create_test_dataframe(n_groups=1, items_per_group=5)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()
        seed = 42

        result = shuffle.apply(df, schema, seed)

        # Should have same data (only one group, can't shuffle)
        assert result.equals(df)

    def test_shuffle_empty_dataframe(self):
        """Test edge case with empty input."""
        df = pl.DataFrame(
            {
                "group_id": [],
                "item_id": [],
                "value": [],
            }
        )
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()
        seed = 42

        result = shuffle.apply(df, schema, seed)

        # Should return empty DataFrame
        assert len(result) == 0
        assert result.equals(df)

    def test_shuffle_preserves_groups(self):
        """Test that groups remain intact (no items moved between groups)."""
        df = create_test_dataframe(n_groups=5, items_per_group=4)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()
        seed = 42

        result = shuffle.apply(df, schema, seed)

        # Check that each group has the same items
        for group_id in range(5):
            original_items = set(
                df.filter(pl.col("group_id") == group_id)["item_id"].to_list()
            )
            result_items = set(
                result.filter(pl.col("group_id") == group_id)["item_id"].to_list()
            )
            assert original_items == result_items

    def test_shuffle_preserves_data(self):
        """Test that all data is preserved (no loss)."""
        df = create_test_dataframe(n_groups=10, items_per_group=5)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        shuffle = Shuffle()
        seed = 42

        result = shuffle.apply(df, schema, seed)

        # Should have same number of rows
        assert len(result) == len(df)

        # Should have same total number of groups
        assert result["group_id"].n_unique() == df["group_id"].n_unique()

        # Should have same items per group
        for group_id in range(10):
            original_count = len(df.filter(pl.col("group_id") == group_id))
            result_count = len(result.filter(pl.col("group_id") == group_id))
            assert original_count == result_count
