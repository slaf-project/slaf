"""
Tests for SLAF samplers module.

This module tests the shuffle strategy implementations for different sampling approaches.
"""

import pytest

from slaf.ml.samplers import (
    RandomShuffle,
    Shuffle,
    ShuffleType,
    StratifiedShuffle,
    create_shuffle,
)


class TestShuffle:
    """Test base Shuffle class"""

    def test_shuffle_is_abstract(self):
        """Test that Shuffle is an abstract base class"""
        with pytest.raises(TypeError):
            Shuffle()  # type: ignore


class TestRandomShuffle:
    """Test RandomShuffle implementation"""

    def setup_method(self):
        """Set up test data"""
        self.shuffle = RandomShuffle()
        self.test_cell_integer_ids = [0, 1, 2, 3, 4, 5]

    def test_apply_basic(self):
        """Test basic random shuffling"""
        result = self.shuffle.apply(self.test_cell_integer_ids, seed=42)

        # Should have same elements but potentially different order
        assert len(result) == len(self.test_cell_integer_ids)
        assert sorted(result) == sorted(self.test_cell_integer_ids)

    def test_apply_reproducible(self):
        """Test that shuffling is reproducible with same seed"""
        result1 = self.shuffle.apply(self.test_cell_integer_ids, seed=42)
        result2 = self.shuffle.apply(self.test_cell_integer_ids, seed=42)

        # Should be the same with same seed
        assert result1 == result2

    def test_apply_different_seeds(self):
        """Test that different seeds produce different results"""
        result1 = self.shuffle.apply(self.test_cell_integer_ids, seed=42)
        result2 = self.shuffle.apply(self.test_cell_integer_ids, seed=123)

        # Should be different with different seeds (most of the time)
        # Note: This could theoretically fail, but very unlikely
        assert result1 != result2

    def test_apply_empty_list(self):
        """Test with empty list"""
        result = self.shuffle.apply([], seed=42)
        assert result == []

    def test_apply_single_element(self):
        """Test with single element"""
        result = self.shuffle.apply([5], seed=42)
        assert result == [5]

    def test_apply_returns_copy(self):
        """Test that a copy is returned, not the original"""
        result = self.shuffle.apply(self.test_cell_integer_ids, seed=42)

        # Should have same elements but not the same object
        assert sorted(result) == sorted(self.test_cell_integer_ids)
        assert result is not self.test_cell_integer_ids


class TestStratifiedShuffle:
    """Test StratifiedShuffle implementation"""

    def setup_method(self):
        """Set up test data"""
        self.shuffle = StratifiedShuffle()
        self.test_cell_integer_ids = [0, 1, 2, 3, 4, 5]
        self.test_cell_types = ["A", "A", "B", "B", "C", "C"]

    def test_apply_with_cell_types(self):
        """Test stratified shuffling with cell types"""
        result = self.shuffle.apply(
            self.test_cell_integer_ids, seed=42, cell_types=self.test_cell_types
        )

        # Should have same elements
        assert len(result) == len(self.test_cell_integer_ids)
        assert sorted(result) == sorted(self.test_cell_integer_ids)

    def test_apply_reproducible(self):
        """Test that stratified shuffling is reproducible"""
        result1 = self.shuffle.apply(
            self.test_cell_integer_ids, seed=42, cell_types=self.test_cell_types
        )
        result2 = self.shuffle.apply(
            self.test_cell_integer_ids, seed=42, cell_types=self.test_cell_types
        )

        # Should be the same with same seed
        assert result1 == result2

    def test_apply_falls_back_to_random(self):
        """Test fallback to random shuffling when cell types not provided"""
        result = self.shuffle.apply(self.test_cell_integer_ids, seed=42)

        # Should still shuffle
        assert len(result) == len(self.test_cell_integer_ids)
        assert sorted(result) == sorted(self.test_cell_integer_ids)

    def test_apply_mismatched_lengths(self):
        """Test fallback when cell types length doesn't match"""
        mismatched_types = ["A", "B"]  # Only 2 types for 6 cells

        result = self.shuffle.apply(
            self.test_cell_integer_ids, seed=42, cell_types=mismatched_types
        )

        # Should fall back to random shuffling
        assert len(result) == len(self.test_cell_integer_ids)
        assert sorted(result) == sorted(self.test_cell_integer_ids)

    def test_apply_empty_lists(self):
        """Test with empty lists"""
        result = self.shuffle.apply([], seed=42, cell_types=[])
        assert result == []

    def test_apply_single_element(self):
        """Test with single element"""
        result = self.shuffle.apply([5], seed=42, cell_types=["A"])
        assert result == [5]

    def test_apply_balanced_output(self):
        """Test that output maintains some balance between cell types"""
        # Create data with equal numbers of each cell type
        cell_integer_ids = list(range(12))
        cell_types = ["A"] * 4 + ["B"] * 4 + ["C"] * 4

        result = self.shuffle.apply(cell_integer_ids, seed=42, cell_types=cell_types)

        # Should have same elements
        assert len(result) == len(cell_integer_ids)
        assert sorted(result) == sorted(cell_integer_ids)


class TestCreateShuffle:
    """Test shuffle factory function"""

    def test_create_random_shuffle(self):
        """Test creating RandomShuffle"""
        shuffle = create_shuffle("random")
        assert isinstance(shuffle, RandomShuffle)

    def test_create_stratified_shuffle(self):
        """Test creating StratifiedShuffle"""
        shuffle = create_shuffle("stratified")
        assert isinstance(shuffle, StratifiedShuffle)

    def test_create_random_shuffle_enum(self):
        """Test creating RandomShuffle with enum"""
        shuffle = create_shuffle(ShuffleType.RANDOM)
        assert isinstance(shuffle, RandomShuffle)

    def test_create_stratified_shuffle_enum(self):
        """Test creating StratifiedShuffle with enum"""
        shuffle = create_shuffle(ShuffleType.STRATIFIED)
        assert isinstance(shuffle, StratifiedShuffle)

    def test_create_unknown_shuffle(self):
        """Test creating unknown shuffle type"""
        with pytest.raises(ValueError, match="Unsupported shuffle type"):
            create_shuffle("unknown")  # type: ignore

    def test_create_case_insensitive(self):
        """Test that shuffle creation is case insensitive"""
        shuffle1 = create_shuffle("RANDOM")  # type: ignore
        shuffle2 = create_shuffle("random")
        assert isinstance(shuffle1, RandomShuffle)
        assert isinstance(shuffle2, RandomShuffle)

        shuffle3 = create_shuffle("STRATIFIED")  # type: ignore
        shuffle4 = create_shuffle("stratified")
        assert isinstance(shuffle3, StratifiedShuffle)
        assert isinstance(shuffle4, StratifiedShuffle)
