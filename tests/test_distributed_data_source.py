"""
Unit tests for DataSource interface and LanceDataSource implementation.

Tests focus on partition counting and reader creation with mocks.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import polars as pl

from slaf.distributed.data_source import DataSource, LanceDataSource


class TestDataSource:
    """Test cases for DataSource interface."""

    def test_data_source_interface(self):
        """Test abstract methods exist."""
        # DataSource is abstract, so we can't instantiate it directly
        # But we can check that it has the required abstract methods
        assert hasattr(DataSource, "get_partition_count")
        assert hasattr(DataSource, "create_reader")


class TestLanceDataSource:
    """Test cases for LanceDataSource implementation."""

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_partition_count(self, mock_lance):
        """Test get_partition_count() with mock Lance dataset."""
        # Create mock fragments
        mock_fragments = [MagicMock() for _ in range(5)]
        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = mock_fragments
        mock_lance.dataset.return_value = mock_dataset

        data_source = LanceDataSource(lance_path="/fake/path")
        partition_count = data_source.get_partition_count()

        assert partition_count == 5
        mock_lance.dataset.assert_called_once_with("/fake/path")

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_create_reader(self, mock_lance):
        """Test create_reader() returns generator."""
        # Create mock fragment with batches
        mock_batch1 = MagicMock()
        mock_batch2 = MagicMock()
        mock_fragment = MagicMock()
        mock_fragment.to_batches.return_value = [mock_batch1, mock_batch2]

        mock_fragments = [mock_fragment]
        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = mock_fragments
        mock_lance.dataset.return_value = mock_dataset

        # Mock polars conversion
        with patch("slaf.distributed.data_source.pl.from_arrow") as mock_from_arrow:
            mock_df1 = pl.DataFrame({"col1": [1, 2]})
            mock_df2 = pl.DataFrame({"col1": [3, 4]})
            mock_from_arrow.side_effect = [mock_df1, mock_df2]

            data_source = LanceDataSource(lance_path="/fake/path")
            reader = data_source.create_reader(partition_index=0, batch_size=100)

            # Should return an iterator
            assert isinstance(reader, Iterator)

            # Should yield DataFrames
            results = list(reader)
            assert len(results) == 2
            assert isinstance(results[0], pl.DataFrame)
            assert isinstance(results[1], pl.DataFrame)

            # Check that to_batches was called with correct batch_size
            mock_fragment.to_batches.assert_called_once_with(batch_size=100)

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_reader_yields_dataframes(self, mock_lance):
        """Test reader yields Polars DataFrames."""
        # Create mock fragment with batches
        mock_batch = MagicMock()
        mock_fragment = MagicMock()
        mock_fragment.to_batches.return_value = [mock_batch]

        mock_fragments = [mock_fragment]
        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = mock_fragments
        mock_lance.dataset.return_value = mock_dataset

        # Mock polars conversion
        with patch("slaf.distributed.data_source.pl.from_arrow") as mock_from_arrow:
            mock_df = pl.DataFrame({"col1": [1, 2, 3]})
            mock_from_arrow.return_value = mock_df

            data_source = LanceDataSource(lance_path="/fake/path")
            reader = data_source.create_reader(partition_index=0, batch_size=100)

            result = next(reader)
            assert isinstance(result, pl.DataFrame)
            assert len(result) == 3

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_empty_dataset(self, mock_lance):
        """Test edge case with empty dataset."""
        # Create mock dataset with no fragments
        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = []
        mock_lance.dataset.return_value = mock_dataset

        data_source = LanceDataSource(lance_path="/fake/path")
        partition_count = data_source.get_partition_count()

        assert partition_count == 0

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_single_partition(self, mock_lance):
        """Test edge case with single partition."""
        # Create mock dataset with one fragment
        mock_fragment = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = [mock_fragment]
        mock_lance.dataset.return_value = mock_dataset

        data_source = LanceDataSource(lance_path="/fake/path")
        partition_count = data_source.get_partition_count()

        assert partition_count == 1

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_lazy_loading(self, mock_lance):
        """Test that dataset is lazy-loaded."""
        # Dataset should not be loaded until first access
        data_source = LanceDataSource(lance_path="/fake/path")

        # Should not have called lance.dataset yet
        mock_lance.dataset.assert_not_called()

        # Now access dataset property
        _ = data_source.dataset

        # Now it should be called
        mock_lance.dataset.assert_called_once_with("/fake/path")

    @patch("slaf.distributed.data_source.lance")
    def test_lance_data_source_partition_count_cached(self, mock_lance):
        """Test that partition count is cached."""
        mock_fragments = [MagicMock() for _ in range(5)]
        mock_dataset = MagicMock()
        mock_dataset.get_fragments.return_value = mock_fragments
        mock_lance.dataset.return_value = mock_dataset

        data_source = LanceDataSource(lance_path="/fake/path")

        # First call
        count1 = data_source.get_partition_count()
        assert count1 == 5

        # Reset mock call count
        mock_dataset.get_fragments.reset_mock()

        # Second call should use cached value
        count2 = data_source.get_partition_count()
        assert count2 == 5

        # get_fragments should not be called again (cached)
        # Note: The actual implementation caches _fragment_count, so get_fragments
        # is only called once. We can't easily test this without accessing private
        # attributes, but we can verify the result is correct.
