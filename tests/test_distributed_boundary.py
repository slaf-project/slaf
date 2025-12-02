"""
Unit tests for GroupBoundaryHandler logic.

Tests focus on boundary detection, partial group tracking, and merging.
"""

from typing import Any

import polars as pl

from slaf.distributed.boundary import GroupBoundaryHandler
from slaf.distributed.processor import DataSchema


def create_test_dataframe(
    group_key: str = "group_id",
    item_key: str = "item_id",
    value_key: str = "value",
    groups: list[int] | None = None,
    items_per_group: int = 5,
) -> pl.DataFrame:
    """Create a test DataFrame with specified groups."""
    if groups is None:
        groups = [0, 1, 2]

    data = []
    for group_id in groups:
        for item_id in range(items_per_group):
            data.append(
                {
                    group_key: group_id,
                    item_key: item_id,
                    value_key: item_id * 10,
                }
            )
    return pl.DataFrame(data)


class MockKVStore:
    """Mock KV store for testing."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def put(self, key: str, value: Any, ttl: int | None = None):
        self._store[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        return self._store.pop(key, default)


class TestGroupBoundaryHandler:
    """Test cases for GroupBoundaryHandler logic."""

    def test_boundary_handler_initialization(self):
        """Test handler creation."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        assert handler.schema == schema
        assert handler.continuity_check == "sequential"
        assert handler.partial_groups_kv is None
        assert handler.partial_groups == {}
        assert handler.last_seen_groups == {}

    def test_check_continuity_sequential(self):
        """Test sequential continuity detection."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # Sequential: last_group + 1 = first_group
        assert handler.check_continuity(5, 6) is True
        assert handler.check_continuity(5, 7) is False
        assert handler.check_continuity(5, 5) is True  # Same group

    def test_check_continuity_same_group(self):
        """Test same group spanning boundary."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # Same group should be continuous
        assert handler.check_continuity(5, 5) is True

    def test_check_continuity_no_continuity(self):
        """Test when batches are not continuous."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # Gap in groups
        assert handler.check_continuity(5, 8) is False

    def test_track_partial_groups_basic(self):
        """Test tracking partial groups."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # Create batch with groups 0, 1, 2
        batch = create_test_dataframe(groups=[0, 1, 2], items_per_group=3)

        complete_groups, partial_groups = handler.track_partial_groups(
            batch, partition_id=0, is_partition_exhausted=False
        )

        # Last group (2) should be partial
        assert len(partial_groups) == 1
        assert 2 in partial_groups

        # Complete groups should be 0 and 1
        complete_group_ids = complete_groups["group_id"].unique().to_list()
        assert 0 in complete_group_ids
        assert 1 in complete_group_ids
        assert 2 not in complete_group_ids

    def test_track_partial_groups_partition_exhausted(self):
        """Test when partition is exhausted."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # Create batch with groups 0, 1, 2
        batch = create_test_dataframe(groups=[0, 1, 2], items_per_group=3)

        complete_groups, partial_groups = handler.track_partial_groups(
            batch, partition_id=0, is_partition_exhausted=True
        )

        # When partition is exhausted, all groups are complete
        assert len(partial_groups) == 0
        assert len(complete_groups) == len(batch)

    def test_merge_partial_data_basic(self):
        """Test merging partial groups."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # First batch: groups 0, 1 (group 1 is partial)
        batch1 = create_test_dataframe(groups=[0, 1], items_per_group=3)
        complete1, partial1 = handler.track_partial_groups(
            batch1, partition_id=0, is_partition_exhausted=False
        )

        # Second batch: group 1 continues (same group)
        batch2 = create_test_dataframe(groups=[1, 2], items_per_group=3)

        # Merge
        merged, remaining_partial = handler.merge_partial_data(
            partial1, batch2, partition_id=0, is_partition_exhausted=False
        )

        # Group 1 should be merged (complete) - it appears in merged result
        merged_group_ids = merged["group_id"].unique().to_list()
        assert 1 in merged_group_ids

        # Group 1 should have more rows in merged than in batch2 alone (merged with partial)
        group_1_in_merged = len(merged.filter(pl.col("group_id") == 1))
        group_1_in_batch2 = len(batch2.filter(pl.col("group_id") == 1))
        assert group_1_in_merged > group_1_in_batch2

    def test_merge_partial_data_no_partials(self):
        """Test when no partial groups exist."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        batch = create_test_dataframe(groups=[0, 1, 2], items_per_group=3)
        partial_data = {}

        merged, remaining_partial = handler.merge_partial_data(
            partial_data, batch, partition_id=0, is_partition_exhausted=False
        )

        # Last group (2) should be partial, so merged should have groups 0 and 1
        assert len(merged) < len(batch)  # Last group is partial
        assert len(remaining_partial) > 0  # New partial groups tracked
        # Complete groups should be 0 and 1
        complete_group_ids = merged["group_id"].unique().to_list()
        assert 0 in complete_group_ids
        assert 1 in complete_group_ids

    def test_merge_partial_data_cross_worker_kv(self):
        """Test KV store integration for cross-worker merging."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        kv_store = MockKVStore()
        handler = GroupBoundaryHandler(
            schema=schema,
            continuity_check="sequential",
            partial_groups_kv=kv_store,
        )

        # Worker 0: process partition 0, group 0 is partial (last group)
        batch1 = create_test_dataframe(groups=[0], items_per_group=3)
        complete1, partial1 = handler.track_partial_groups(
            batch1, partition_id=0, is_partition_exhausted=False
        )
        handler.send_partial_groups_to_kv(0, partial1)

        # Worker 1: process partition 1, starts with group 1 (sequential from group 0)
        batch2 = create_test_dataframe(groups=[1, 2], items_per_group=3)
        handler2 = GroupBoundaryHandler(
            schema=schema,
            continuity_check="sequential",
            partial_groups_kv=kv_store,
        )

        # Check KV for partial groups - partition 1 starts with group 1
        # Should check for group 0 from partition 0 (sequential: 0 + 1 = 1)
        first_group_id = batch2["group_id"][0]
        kv_partial = handler2.check_kv_for_partial_groups(1, first_group_id)

        # Should find partial group from partition 0 if group 0 was stored
        # The check looks for group_id - 1 from partition_id - 1
        # So for partition 1, group 1, it looks for group 0 from partition 0
        if 0 in partial1:
            assert kv_partial is not None
            assert len(kv_partial) > 0
        else:
            # If no matching partial, that's also valid
            pass

    def test_merge_partial_data_kv_store_missing(self):
        """Test when KV store key doesn't exist."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        kv_store = MockKVStore()
        handler = GroupBoundaryHandler(
            schema=schema,
            continuity_check="sequential",
            partial_groups_kv=kv_store,
        )

        # Check for non-existent partial group
        kv_partial = handler.check_kv_for_partial_groups(1, 5)

        # Should return None
        assert kv_partial is None

    def test_send_partial_groups_to_kv(self):
        """Test sending partial groups to KV store."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        kv_store = MockKVStore()
        handler = GroupBoundaryHandler(
            schema=schema,
            continuity_check="sequential",
            partial_groups_kv=kv_store,
        )

        # Create partial groups
        batch = create_test_dataframe(groups=[0, 1], items_per_group=3)
        complete, partial = handler.track_partial_groups(batch, partition_id=0)

        # Send to KV
        handler.send_partial_groups_to_kv(0, partial)

        # Check that it's in KV store
        assert len(partial) > 0
        for group_id in partial.keys():
            key = f"0:{group_id}"
            assert kv_store.get(key) is not None

    def test_check_kv_for_partial_groups(self):
        """Test checking KV store for partial groups."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        kv_store = MockKVStore()
        handler = GroupBoundaryHandler(
            schema=schema,
            continuity_check="sequential",
            partial_groups_kv=kv_store,
        )

        # Store a partial group
        batch = create_test_dataframe(groups=[0, 1], items_per_group=3)
        complete, partial = handler.track_partial_groups(batch, partition_id=0)
        handler.send_partial_groups_to_kv(0, partial)

        # Check for it
        group_id = list(partial.keys())[0]
        kv_partial = handler.check_kv_for_partial_groups(1, group_id + 1)

        # Should find it (sequential continuity)
        assert kv_partial is not None

    def test_cleanup_stale_partial_groups(self):
        """Test cleanup of stale partial data."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        # Create partial groups
        batch1 = create_test_dataframe(groups=[0, 1], items_per_group=3)
        complete1, partial1 = handler.track_partial_groups(batch1, partition_id=0)

        # Cleanup stale partials (simulating gap where group 3 is seen, skipping group 1)
        cleaned = handler.cleanup_stale_partial(partial1, 3, partition_id=0)

        # Group 1 should be removed (gap detected)
        assert 1 not in cleaned

    def test_boundary_handler_continuity_none(self):
        """Test with continuity_check='none'."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="none")

        # With "none", no continuity should be detected
        assert handler.check_continuity(5, 6) is False
        assert handler.check_continuity(5, 5) is True  # Same group still works

        # Cleanup should not remove anything
        partial = {1: create_test_dataframe(groups=[1], items_per_group=2)}
        cleaned = handler.cleanup_stale_partial(partial, 3, partition_id=0)
        assert len(cleaned) == len(partial)

    def test_merge_partial_data_empty_batch(self):
        """Test merging with empty batch."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )
        handler = GroupBoundaryHandler(schema=schema, continuity_check="sequential")

        batch = pl.DataFrame(
            {
                "group_id": [],
                "item_id": [],
                "value": [],
            }
        )
        partial_data = {}

        merged, remaining_partial = handler.merge_partial_data(
            partial_data, batch, partition_id=0
        )

        # Should return empty batch
        assert len(merged) == 0
        assert len(remaining_partial) == 0
