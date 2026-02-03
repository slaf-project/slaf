"""
Unit tests for Window function logic.

Tests focus on windowing behavior: rank by value, filter top N, aggregate.
"""

import polars as pl

from slaf.distributed.processor import DataSchema
from slaf.distributed.window import Window


def create_test_dataframe(
    group_key: str = "group_id",
    item_key: str = "item_id",
    value_key: str = "value",
    n_groups: int = 10,
    items_per_group: int = 5,
) -> pl.DataFrame:
    """Create a test DataFrame with specified structure."""
    data = []
    for group_id in range(n_groups):
        for item_id in range(items_per_group):
            # Create values in descending order so we can test ranking
            value = items_per_group - item_id
            data.append(
                {
                    group_key: group_id,
                    item_key: item_id,
                    value_key: value,
                }
            )
    return pl.DataFrame(data)


class TestWindow:
    """Test cases for Window function logic."""

    def test_window_basic(self):
        """Test basic windowing (rank by value, filter top N)."""
        df = create_test_dataframe(n_groups=3, items_per_group=5)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should have 3 groups
        assert len(result) == 3

        # Each group should have max_items items
        for row in result.iter_rows(named=True):
            assert len(row["item_list"]) == max_items
            # Items should be in descending order of value (top 3)
            # Since values are 5, 4, 3, 2, 1, top 3 should be items 0, 1, 2
            assert row["item_list"] == [0, 1, 2]

    def test_window_max_items(self):
        """Test max_items filtering."""
        df = create_test_dataframe(n_groups=2, items_per_group=10)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        max_items = 5

        result = window.apply(df, schema, max_items)

        # Each group should have exactly max_items items
        for row in result.iter_rows(named=True):
            assert len(row["item_list"]) == max_items

    def test_window_multiple_groups(self):
        """Test windowing with multiple groups."""
        df = create_test_dataframe(n_groups=5, items_per_group=7)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should have 5 groups
        assert len(result) == 5

        # All groups should have max_items items
        for row in result.iter_rows(named=True):
            assert len(row["item_list"]) == max_items

    def test_window_empty_dataframe(self):
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
            item_list_key="item_list",
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should return empty DataFrame
        assert len(result) == 0

    def test_window_single_group(self):
        """Test edge case with single group."""
        df = create_test_dataframe(n_groups=1, items_per_group=5)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should have 1 group
        assert len(result) == 1
        assert len(result["item_list"][0]) == max_items

    def test_window_with_value_list(self):
        """Test windowing when value_list_key is specified."""
        df = create_test_dataframe(n_groups=2, items_per_group=5)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
            value_list_key="value_list",
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should have value_list column
        assert "value_list" in result.columns

        # Each group should have max_items values
        for row in result.iter_rows(named=True):
            assert len(row["value_list"]) == max_items
            # Values should be in descending order (top 3: 5, 4, 3)
            assert row["value_list"] == [5, 4, 3]

    def test_window_without_value_list(self):
        """Test windowing when value_list_key is None."""
        df = create_test_dataframe(n_groups=2, items_per_group=5)
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
            value_list_key=None,
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should not have value_list column
        assert "value_list" not in result.columns

    def test_window_ties(self):
        """Test handling of tied values (same rank)."""
        # Create data with tied values
        data = []
        for group_id in range(2):
            # Group 0: values [5, 5, 4, 4, 3] - ties at top
            # Group 1: values [5, 4, 4, 3, 3] - ties in middle
            values = [5, 5, 4, 4, 3] if group_id == 0 else [5, 4, 4, 3, 3]
            for item_id, value in enumerate(values):
                data.append(
                    {
                        "group_id": group_id,
                        "item_id": item_id,
                        "value": value,
                    }
                )
        df = pl.DataFrame(data)

        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            item_list_key="item_list",
        )
        window = Window()
        max_items = 3

        result = window.apply(df, schema, max_items)

        # Should have 2 groups
        assert len(result) == 2

        # Group 0: top 3 should include both items with value 5 and one with value 4
        # With dense ranking and ties, we might get more than 3 items
        group_0_row = result.filter(pl.col("group_id") == 0)
        if len(group_0_row) > 0:
            group_0_items = group_0_row["item_list"][0]
            # Create a lookup dict for values
            value_lookup = {
                row["item_id"]: row["value"]
                for row in df.filter(pl.col("group_id") == 0).iter_rows(named=True)
            }
            group_0_values = [value_lookup[item_id] for item_id in group_0_items]
            # With ties, we might get more than max_items, but all should be top-ranked
            # All items should have value >= 3 (top 3 ranks include values 5, 4, 3)
            assert all(v >= 3 for v in group_0_values)
            # Should have at least max_items items (might be more due to ties)
            assert len(group_0_values) >= max_items

        # Group 1: top 3 should include item with value 5 and items with value 4
        group_1_row = result.filter(pl.col("group_id") == 1)
        if len(group_1_row) > 0:
            group_1_items = group_1_row["item_list"][0]
            # Create a lookup dict for values
            value_lookup = {
                row["item_id"]: row["value"]
                for row in df.filter(pl.col("group_id") == 1).iter_rows(named=True)
            }
            group_1_values = [value_lookup[item_id] for item_id in group_1_items]
            # All items should have value >= 3 (top 3 ranks include values 5, 4, 3)
            assert all(v >= 3 for v in group_1_values)
            # Should have at least max_items items (might be more due to ties)
            assert len(group_1_values) >= max_items
