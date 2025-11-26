"""
Generic Group Boundary Handler for distributed dataloading.

Handles tracking and merging of groups that span partition boundaries.

Supports cross-worker boundary merging via KV store-like object:
- When a worker finishes a partition, it stores partial groups in a shared KV store
- When a worker starts a partition, it checks the KV store for matching partial groups
- Partial groups are matched based on partition adjacency and group continuity
- Uses key format: f"{partition_id}:{group_id}" for efficient lookups

Framework-agnostic - accepts any KV store-like object with get, put, and pop methods.
"""

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from slaf.distributed.processor import DataSchema


class GroupBoundaryHandler:
    """
    Generic handler for tracking groups across partition boundaries.

    Works with any tabular data structure using DataSchema.
    Handles:
    - Tracking partial groups from previous batches
    - Detecting continuity between batches
    - Merging partial data when continuity is detected
    - Cleaning up stale partial data when gaps are detected
    """

    def __init__(
        self,
        schema: "DataSchema",
        continuity_check: str = "sequential",
        partial_groups_kv: Any | None = None,
    ):
        """
        Initialize boundary handler.

        Args:
            schema: Data schema (used to identify group_key)
            continuity_check: How to detect continuity between batches.
                             Options:
                             - "sequential": Groups are sequential integers (last_group + 1 = next_first_group)
                             - "ordered": Groups are ordered but not necessarily sequential
                             - "none": No continuity checking (assume groups are complete within partitions)
            partial_groups_kv: Optional KV store-like object for cross-worker partial group sharing.
                              If provided, enables boundary merging across workers.
                              Must have methods: get(key), put(key, value, ttl=None), pop(key, default=None)
                              Keys are formatted as f"{partition_id}:{group_id}".
                              Framework-agnostic - works with Modal Dict, Redis, or any KV store.
        """
        self.schema = schema
        self.continuity_check = continuity_check
        self.partial_groups_kv = partial_groups_kv
        self.partial_groups: dict[Any, pl.DataFrame] = {}  # Store incomplete groups
        self.last_seen_groups: dict[
            int, Any
        ] = {}  # Track last group per partition/reader

    def check_continuity(
        self, last_group_id: Any, first_group_id: Any, partition_id: int | None = None
    ) -> bool:
        """
        Check if batches are continuous (same group spans both).

        Args:
            last_group_id: Last group ID from previous batch
            first_group_id: First group ID from current batch
            partition_id: Optional partition/reader ID for tracking

        Returns:
            True if batches are continuous (same group), False otherwise
        """
        if self.continuity_check == "sequential":
            # Sequential: check if first_group = last_group + 1
            # Assumes group_key is integer and groups are sequential
            try:
                return int(first_group_id) == int(last_group_id) + 1
            except (ValueError, TypeError):
                return False
        elif self.continuity_check == "ordered":
            # Ordered: check if first_group > last_group (assumes sorted data)
            try:
                return first_group_id > last_group_id
            except TypeError:
                return False
        else:  # "none"
            return False

    def track_partial_groups(
        self, batch: pl.DataFrame, partition_id: int | None = None
    ) -> tuple[pl.DataFrame, dict[Any, pl.DataFrame]]:
        """
        Track partial groups in a batch and return complete groups.

        Args:
            batch: Input DataFrame
            partition_id: Optional partition/reader ID for tracking

        Returns:
            Tuple of (complete_groups_df, partial_groups_dict)
        """
        if len(batch) == 0:
            return batch, {}

        group_key = self.schema.group_key

        # Find the last complete group (all groups except the last one)
        group_counts = batch.group_by(group_key).len()
        if len(group_counts) == 0:
            return batch, {}

        # Get all unique group IDs
        unique_groups = batch[group_key].unique().sort()
        if len(unique_groups) == 0:
            return batch, {}

        # Last group might be incomplete (spans partition boundary)
        last_group_id = unique_groups[-1]

        # Split into complete and partial
        complete_groups = batch.filter(pl.col(group_key) != last_group_id)
        partial_group = batch.filter(pl.col(group_key) == last_group_id)

        # Store partial group
        partial_dict = {}
        if len(partial_group) > 0:
            partial_dict[last_group_id] = partial_group
            # Track last seen group for this partition
            if partition_id is not None:
                self.last_seen_groups[partition_id] = last_group_id

        return complete_groups, partial_dict

    def check_kv_for_partial_groups(
        self, partition_id: int, first_group_id: Any
    ) -> pl.DataFrame | None:
        """
        Check the KV store for matching partial data from previous partition.

        Args:
            partition_id: Current partition ID
            first_group_id: First group ID in current batch

        Returns:
            Matching partial group DataFrame, or None if not found
        """
        if self.partial_groups_kv is None:
            return None

        if partition_id == 0:
            # First partition, no previous partition to check
            return None

        try:
            # Check for partial group from previous partition
            # Key format: f"{partition_id - 1}:{group_id}"
            # We need to check if there's a partial group from partition_id - 1
            # that would be continuous with first_group_id

            # For sequential continuity, check if there's a partial group
            # with group_id = first_group_id - 1 from partition_id - 1
            if self.continuity_check == "sequential":
                try:
                    prev_group_id = int(first_group_id) - 1
                    key = f"{partition_id - 1}:{prev_group_id}"
                    partial_data = self.partial_groups_kv.get(key)
                    if partial_data is not None:
                        # Found matching partial group! Convert back to DataFrame
                        return pl.from_arrow(partial_data)
                except (ValueError, TypeError):
                    # first_group_id is not an integer, can't check sequential
                    pass
            elif self.continuity_check == "ordered":
                # For ordered, we need to check all partial groups from previous partition
                # This is less efficient but necessary for non-sequential groups
                # We'll iterate through possible keys (this could be optimized)
                # For now, we'll check a range around first_group_id
                try:
                    # Check a small range around first_group_id
                    first_id_int = int(first_group_id)
                    for offset in range(-10, 1):  # Check 10 groups before
                        check_group_id = first_id_int + offset
                        key = f"{partition_id - 1}:{check_group_id}"
                        partial_data = self.partial_groups_kv.get(key)
                        if partial_data is not None:
                            # Found a partial group, check if it's continuous
                            if self.check_continuity(
                                check_group_id, first_group_id, None
                            ):
                                return pl.from_arrow(partial_data)
                except (ValueError, TypeError):
                    # Can't convert to int, skip ordered check
                    pass

        except Exception as e:
            # KV operations failed, log and fall back to local-only
            print(f"Error checking KV store for partial groups: {e}")

        return None

    def send_partial_groups_to_kv(
        self, partition_id: int, partial_data: dict[Any, pl.DataFrame]
    ):
        """
        Store partial groups in the shared KV store for cross-worker merging.

        Args:
            partition_id: Partition ID that produced these partial groups
            partial_data: Dictionary of partial groups to store
        """
        if self.partial_groups_kv is None:
            return

        if not partial_data:
            return

        # Store each partial group in the KV store
        for group_id, partial_df in partial_data.items():
            try:
                # Convert DataFrame to Arrow for serialization
                arrow_table = partial_df.to_arrow()

                # Key format: f"{partition_id}:{group_id}"
                key = f"{partition_id}:{group_id}"

                # Store in KV store
                # Note: Modal Dict entries expire after 7 days of inactivity automatically
                self.partial_groups_kv.put(key, arrow_table)
            except Exception as e:
                # Log error but continue (don't fail worker)
                print(f"Error storing partial group in KV store: {e}")

    def cleanup_partial_groups_from_kv(self, partition_id: int):
        """
        Clean up partial groups from KV store after partition is fully processed.

        This is optional but helps prevent KV store from accumulating stale data.

        Args:
            partition_id: Partition ID to clean up
        """
        if self.partial_groups_kv is None:
            return

        try:
            # Remove all keys for this partition
            # We don't know all group_ids, so we can't clean up perfectly
            # The TTL will handle cleanup automatically
            # But we can try to remove known keys if we track them
            pass  # Optional cleanup - TTL handles it
        except Exception as e:
            print(f"Error cleaning up partial groups from KV store: {e}")

    def merge_partial_data(
        self,
        partial_data: dict[Any, pl.DataFrame],
        new_batch: pl.DataFrame,
        partition_id: int | None = None,
    ) -> tuple[pl.DataFrame, dict[Any, pl.DataFrame]]:
        """
        Merge partial group data with new batch when continuity is detected.

        Supports both local (within-worker) and cross-worker (via KV store) merging.

        Args:
            partial_data: Dictionary of partial groups from previous batches (local)
            new_batch: New batch DataFrame
            partition_id: Optional partition/reader ID for tracking

        Returns:
            Tuple of (merged_batch, remaining_partial_data)
        """
        if len(new_batch) == 0:
            return new_batch, partial_data

        group_key = self.schema.group_key

        # Get first group in new batch
        first_group_id = new_batch[group_key].item(0)

        merged_batch = new_batch
        remaining_partial = partial_data.copy()

        # First, check local partial data
        last_group_id = None
        if partition_id is not None and partition_id in self.last_seen_groups:
            last_group_id = self.last_seen_groups[partition_id]

        # Check continuity and merge if detected (local)
        if last_group_id is not None and self.check_continuity(
            last_group_id, first_group_id, partition_id
        ):
            if last_group_id in partial_data:
                # Merge: prepend partial data to new batch
                partial_df = partial_data[last_group_id]
                merged_batch = pl.concat([partial_df, new_batch], how="vertical")
                # Remove merged partial data
                remaining_partial.pop(last_group_id)

        # Then, check KV store for cross-worker partial groups
        if partition_id is not None and self.partial_groups_kv is not None:
            kv_partial = self.check_kv_for_partial_groups(partition_id, first_group_id)
            if kv_partial is not None:
                # Merge KV partial data
                merged_batch = pl.concat([kv_partial, merged_batch], how="vertical")
                # Remove the partial group from KV store since we've merged it
                try:
                    if self.continuity_check == "sequential":
                        prev_group_id = int(first_group_id) - 1
                        key = f"{partition_id - 1}:{prev_group_id}"
                        self.partial_groups_kv.pop(key, None)
                except Exception:
                    pass  # Ignore cleanup errors

        # Track new partial groups from this batch
        complete_groups, new_partial = self.track_partial_groups(
            merged_batch, partition_id
        )
        remaining_partial.update(new_partial)

        # Store new partial groups in KV store for cross-worker merging
        if partition_id is not None and new_partial:
            self.send_partial_groups_to_kv(partition_id, new_partial)

        return complete_groups, remaining_partial

    def cleanup_stale_partial(
        self,
        partial_data: dict[Any, pl.DataFrame],
        last_seen_group_id: Any,
        partition_id: int | None = None,
    ) -> dict[Any, pl.DataFrame]:
        """
        Remove stale partial data when gaps are detected.

        Args:
            partial_data: Dictionary of partial groups
            last_seen_group_id: Last group ID seen (for gap detection)
            partition_id: Optional partition/reader ID

        Returns:
            Cleaned partial data dictionary
        """
        if self.continuity_check == "none":
            return partial_data

        # Remove partial groups that can't be completed (gap detected)
        keys_to_remove = []
        for group_id in partial_data.keys():
            if self.continuity_check == "sequential":
                # Remove if group_id <= last_seen (gap means we'll never see it again)
                try:
                    if int(group_id) <= int(last_seen_group_id):
                        keys_to_remove.append(group_id)
                except (ValueError, TypeError):
                    pass
            elif self.continuity_check == "ordered":
                # Remove if group_id < last_seen (gap means we'll never see it again)
                try:
                    if group_id < last_seen_group_id:
                        keys_to_remove.append(group_id)
                except TypeError:
                    pass

        cleaned = partial_data.copy()
        for key in keys_to_remove:
            cleaned.pop(key, None)

        return cleaned
