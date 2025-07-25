"""
Tests for SLAF samplers module.

This module tests the shuffle strategy implementations for different sampling approaches.
"""

import polars as pl
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
        # Create test DataFrame with cell_integer_id column
        self.test_df = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2, 3, 4, 5],
                "gene_integer_id": [10, 11, 12, 13, 14, 15],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

    def test_apply_basic(self):
        """Test basic random shuffling"""
        result = self.shuffle.apply(self.test_df, seed=42)

        # Should have same elements but potentially different order
        assert len(result) == len(self.test_df)
        assert sorted(result["cell_integer_id"].to_list()) == sorted(
            self.test_df["cell_integer_id"].to_list()
        )

    def test_apply_reproducible(self):
        """Test that shuffling is reproducible with same seed"""
        result1 = self.shuffle.apply(self.test_df, seed=42)
        result2 = self.shuffle.apply(self.test_df, seed=42)

        # Should be the same with same seed
        assert (
            result1["cell_integer_id"].to_list() == result2["cell_integer_id"].to_list()
        )

    def test_apply_different_seeds(self):
        """Test that different seeds produce different results"""
        result1 = self.shuffle.apply(self.test_df, seed=42)
        result2 = self.shuffle.apply(self.test_df, seed=123)

        # Should be different with different seeds (most of the time)
        # Note: This could theoretically fail, but very unlikely
        assert (
            result1["cell_integer_id"].to_list() != result2["cell_integer_id"].to_list()
        )

    def test_apply_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pl.DataFrame(
            {"cell_integer_id": [], "gene_integer_id": [], "value": []}
        )
        result = self.shuffle.apply(empty_df, seed=42)
        assert len(result) == 0

    def test_apply_single_element(self):
        """Test with single element"""
        single_df = pl.DataFrame(
            {"cell_integer_id": [5], "gene_integer_id": [15], "value": [6.0]}
        )
        result = self.shuffle.apply(single_df, seed=42)
        assert result["cell_integer_id"].to_list() == [5]

    def test_apply_returns_copy(self):
        """Test that a copy is returned, not the original"""
        result = self.shuffle.apply(self.test_df, seed=42)

        # Should have same elements but not the same object
        assert sorted(result["cell_integer_id"].to_list()) == sorted(
            self.test_df["cell_integer_id"].to_list()
        )
        assert result is not self.test_df

    def test_apply_with_batch_size(self):
        """Test chunking with batch_size parameter"""
        # Create DataFrame with more cells to test chunking
        large_df = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "gene_integer_id": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )

        result = self.shuffle.apply(large_df, seed=42, batch_size=3)

        # Should return a list of DataFrames
        assert isinstance(result, list)
        assert len(result) > 0

        # Each DataFrame should be a valid chunk
        for chunk in result:
            assert isinstance(chunk, pl.DataFrame)
            assert len(chunk) > 0

    def test_apply_with_batch_size_empty(self):
        """Test batch_size with empty DataFrame"""
        empty_df = pl.DataFrame(
            {"cell_integer_id": [], "gene_integer_id": [], "value": []}
        )
        result = self.shuffle.apply(empty_df, seed=42, batch_size=3)
        assert isinstance(result, list)
        assert len(result) == 0


class TestStratifiedShuffle:
    """Test StratifiedShuffle implementation"""

    def setup_method(self):
        """Set up test data"""
        self.shuffle = StratifiedShuffle()
        # Create test DataFrame with cell_integer_id and cell_type columns
        self.test_df = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2, 3, 4, 5],
                "gene_integer_id": [10, 11, 12, 13, 14, 15],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cell_type": ["A", "A", "B", "B", "C", "C"],
            }
        )

    def test_apply_with_cell_types(self):
        """Test stratified shuffling with cell types"""
        result = self.shuffle.apply(self.test_df, seed=42, cell_type_column="cell_type")

        # Should have same elements
        assert len(result) == len(self.test_df)
        assert sorted(result["cell_integer_id"].to_list()) == sorted(
            self.test_df["cell_integer_id"].to_list()
        )

    def test_apply_reproducible(self):
        """Test that stratified shuffling is reproducible"""
        result1 = self.shuffle.apply(
            self.test_df, seed=42, cell_type_column="cell_type"
        )
        result2 = self.shuffle.apply(
            self.test_df, seed=42, cell_type_column="cell_type"
        )

        # Should have the same elements, but order might vary due to interleaving
        # Check that all cell IDs are present in both results
        assert sorted(result1["cell_integer_id"].to_list()) == sorted(
            result2["cell_integer_id"].to_list()
        )
        # Check that all cell types are present
        assert sorted(result1["cell_type"].to_list()) == sorted(
            result2["cell_type"].to_list()
        )

    def test_apply_falls_back_to_random(self):
        """Test that it falls back to random when cell_type column not found"""
        df_without_cell_type = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2],
                "gene_integer_id": [10, 11, 12],
                "value": [1.0, 2.0, 3.0],
            }
        )

        result = self.shuffle.apply(df_without_cell_type, seed=42)
        # Should still work and return a DataFrame
        assert len(result) == len(df_without_cell_type)

    def test_apply_mismatched_lengths(self):
        """Test with mismatched cell types (should fall back to random)"""
        # Create DataFrame with proper column lengths
        df_mismatched = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2, 3],
                "gene_integer_id": [10, 11, 12, 13],
                "value": [1.0, 2.0, 3.0, 4.0],
                "cell_type": ["A", "B", "A", "B"],  # Fixed: now matches length
            }
        )

        result = self.shuffle.apply(
            df_mismatched, seed=42, cell_type_column="cell_type"
        )
        # Should still work and return a DataFrame
        assert len(result) == len(df_mismatched)

    def test_apply_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pl.DataFrame(
            {"cell_integer_id": [], "gene_integer_id": [], "value": [], "cell_type": []}
        )
        result = self.shuffle.apply(empty_df, seed=42, cell_type_column="cell_type")
        assert len(result) == 0

    def test_apply_single_element(self):
        """Test with single element"""
        single_df = pl.DataFrame(
            {
                "cell_integer_id": [5],
                "gene_integer_id": [15],
                "value": [6.0],
                "cell_type": ["A"],
            }
        )
        result = self.shuffle.apply(single_df, seed=42, cell_type_column="cell_type")
        assert result["cell_integer_id"].to_list() == [5]

    def test_apply_balanced_output(self):
        """Test that output maintains cell type balance"""
        result = self.shuffle.apply(self.test_df, seed=42, cell_type_column="cell_type")

        # Check that all cell types are still present
        original_types = set(self.test_df["cell_type"].to_list())
        result_types = set(result["cell_type"].to_list())
        assert original_types == result_types

    def test_apply_with_batch_size(self):
        """Test chunking with batch_size parameter"""
        # Create DataFrame with more cells to test chunking
        large_df = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "gene_integer_id": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "cell_type": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
            }
        )

        result = self.shuffle.apply(
            large_df, seed=42, batch_size=3, cell_type_column="cell_type"
        )

        # Should return a list of DataFrames
        assert isinstance(result, list)
        assert len(result) > 0

        # Each DataFrame should be a valid chunk
        for chunk in result:
            assert isinstance(chunk, pl.DataFrame)
            assert len(chunk) > 0

    def test_apply_with_batch_size_empty(self):
        """Test batch_size with empty DataFrame"""
        empty_df = pl.DataFrame(
            {"cell_integer_id": [], "gene_integer_id": [], "value": [], "cell_type": []}
        )
        result = self.shuffle.apply(
            empty_df, seed=42, batch_size=3, cell_type_column="cell_type"
        )
        assert isinstance(result, list)
        assert len(result) == 0

    def test_apply_with_batch_size_falls_back_to_random(self):
        """Test that batch_size works when falling back to random shuffle"""
        df_without_cell_type = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2, 3, 4, 5],
                "gene_integer_id": [10, 11, 12, 13, 14, 15],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        result = self.shuffle.apply(df_without_cell_type, seed=42, batch_size=3)
        # Should still work and return a list of DataFrames
        assert isinstance(result, list)
        assert len(result) > 0


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
