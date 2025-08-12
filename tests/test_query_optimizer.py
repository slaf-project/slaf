"""Tests for SLAF query optimization."""

import numpy as np
import polars as pl
import pytest

from slaf.core.query_optimizer import PerformanceMetrics, QueryOptimizer, RowIndexMapper


class TestRowIndexMapper:
    """Test RowIndexMapper functionality."""

    @pytest.fixture
    def mock_slaf_array(self):
        """Create a mock SLAFArray for testing RowIndexMapper."""

        class MockSLAFArray:
            def __init__(self):
                self.shape = (5, 3)  # 5 cells, 3 genes
                # Create mock obs with n_genes column
                self.obs = pl.DataFrame(
                    {
                        "cell_integer_id": [0, 1, 2, 3, 4],
                        "cell_id": ["cell_0", "cell_1", "cell_2", "cell_3", "cell_4"],
                        "n_genes": [2, 3, 1, 4, 2],  # Number of genes per cell
                    }
                )
                # Mock cell start index (cumulative sum with prepended 0)
                cumsum = self.obs["n_genes"].cum_sum()
                self._cell_start_index = pl.concat([pl.Series([0]), cumsum])

        return MockSLAFArray()

    def test_row_index_mapper_initialization(self, mock_slaf_array):
        """Test RowIndexMapper initialization."""
        mapper = RowIndexMapper(mock_slaf_array)

        assert mapper.slaf_array == mock_slaf_array
        assert mapper._cell_count == 5
        assert mapper._gene_count == 3
        # RowIndexMapper references slaf_array._cell_start_index, doesn't have its own

    def test_get_cell_start_index_via_slaf_array(self, mock_slaf_array):
        """Test accessing cell start indices via slaf_array."""
        mapper = RowIndexMapper(mock_slaf_array)

        # The start index should be accessible via slaf_array
        start_indices = mapper.slaf_array._cell_start_index

        # Should be [0, 2, 5, 6, 10, 12] (cumulative sum with prepended 0)
        expected = [0, 2, 5, 6, 10, 12]
        assert start_indices.to_list() == expected

    def test_get_cell_row_ranges_single_cell(self, mock_slaf_array):
        """Test getting row ranges for a single cell."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Cell 0 should have row indices [0, 1] (2 genes)
        row_indices = mapper.get_cell_row_ranges([0])
        assert row_indices == [0, 1]

        # Cell 1 should have row indices [2, 3, 4] (3 genes)
        row_indices = mapper.get_cell_row_ranges([1])
        assert row_indices == [2, 3, 4]

        # Cell 2 should have row indices [5] (1 gene)
        row_indices = mapper.get_cell_row_ranges([2])
        assert row_indices == [5]

    def test_get_cell_row_ranges_multiple_cells(self, mock_slaf_array):
        """Test getting row ranges for multiple cells."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Cells 0 and 2 should combine their ranges
        row_indices = mapper.get_cell_row_ranges([0, 2])
        expected = [0, 1] + [5]  # Cell 0: [0,1], Cell 2: [5]
        assert sorted(row_indices) == sorted(expected)

    def test_get_cell_row_ranges_by_selector_none(self, mock_slaf_array):
        """Test getting row ranges for None selector (all cells)."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Should return all row indices (total of 12 based on n_genes sum)
        row_indices = mapper.get_cell_row_ranges_by_selector(None)
        assert len(row_indices) == 12  # Sum of n_genes: 2+3+1+4+2=12

    def test_get_cell_row_ranges_by_selector_int(self, mock_slaf_array):
        """Test getting row ranges for integer selector."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Cell index 1 should give same result as cell_integer_id 1
        row_indices = mapper.get_cell_row_ranges_by_selector(1)
        expected = mapper.get_cell_row_ranges([1])
        assert row_indices == expected

    def test_get_cell_row_ranges_by_selector_slice(self, mock_slaf_array):
        """Test getting row ranges for slice selector."""
        mapper = RowIndexMapper(mock_slaf_array)

        # First 2 cells (indices 0, 1)
        row_indices = mapper.get_cell_row_ranges_by_selector(slice(0, 2))
        expected = mapper.get_cell_row_ranges([0, 1])  # cell_integer_ids
        assert sorted(row_indices) == sorted(expected)

    def test_get_cell_row_ranges_by_selector_list(self, mock_slaf_array):
        """Test getting row ranges for list selector."""
        mapper = RowIndexMapper(mock_slaf_array)

        # List of cell indices [0, 2]
        row_indices = mapper.get_cell_row_ranges_by_selector([0, 2])
        expected = mapper.get_cell_row_ranges([0, 2])  # cell_integer_ids
        assert sorted(row_indices) == sorted(expected)

    def test_get_cell_row_ranges_by_selector_empty_list(self, mock_slaf_array):
        """Test getting row ranges for empty list selector."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Empty list should return empty result
        row_indices = mapper.get_cell_row_ranges_by_selector([])
        assert row_indices == []

    def test_get_cell_row_ranges_by_selector_boolean_mask(self, mock_slaf_array):
        """Test getting row ranges for boolean mask selector."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Boolean mask for first 3 cells
        mask = np.array([True, True, True, False, False])
        row_indices = mapper.get_cell_row_ranges_by_selector(mask)
        expected = mapper.get_cell_row_ranges([0, 1, 2])  # cell_integer_ids
        assert sorted(row_indices) == sorted(expected)

    def test_get_cell_row_ranges_by_selector_negative_index(self, mock_slaf_array):
        """Test getting row ranges for negative index selector."""
        mapper = RowIndexMapper(mock_slaf_array)

        # -1 should be the last cell (index 4)
        row_indices = mapper.get_cell_row_ranges_by_selector(-1)
        expected = mapper.get_cell_row_ranges([4])  # last cell_integer_id
        assert row_indices == expected

    def test_get_cell_row_ranges_invalid_cell_id(self, mock_slaf_array):
        """Test error handling for invalid cell IDs."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Non-existent cell_integer_id should raise ValueError
        with pytest.raises(ValueError, match="Cell integer ID .* not found"):
            mapper.get_cell_row_ranges([999])

    def test_get_cell_row_ranges_by_selector_out_of_bounds(self, mock_slaf_array):
        """Test error handling for out of bounds selectors."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Out of bounds index should raise ValueError
        with pytest.raises(ValueError, match="Cell index .* out of bounds"):
            mapper.get_cell_row_ranges_by_selector(5)  # Only 5 cells (0-4)

    def test_get_cell_row_ranges_by_selector_invalid_boolean_mask(
        self, mock_slaf_array
    ):
        """Test error handling for invalid boolean mask length."""
        mapper = RowIndexMapper(mock_slaf_array)

        # Wrong length boolean mask should raise ValueError
        mask = np.array([True, False])  # Only 2 elements for 5 cells
        with pytest.raises(ValueError, match="Boolean mask length .* doesn't match"):
            mapper.get_cell_row_ranges_by_selector(mask)


class TestQueryOptimizer:
    """Test QueryOptimizer functionality."""

    def test_is_consecutive_empty(self):
        """Test is_consecutive with empty list."""
        assert QueryOptimizer.is_consecutive([]) is True

    def test_is_consecutive_single(self):
        """Test is_consecutive with single element."""
        assert QueryOptimizer.is_consecutive([5]) is True

    def test_is_consecutive_consecutive(self):
        """Test is_consecutive with consecutive numbers."""
        assert QueryOptimizer.is_consecutive([1, 2, 3, 4, 5]) is True

    def test_is_consecutive_non_consecutive(self):
        """Test is_consecutive with non-consecutive numbers."""
        assert QueryOptimizer.is_consecutive([1, 3, 5, 7]) is False

    def test_is_consecutive_unsorted(self):
        """Test is_consecutive with unsorted consecutive numbers."""
        assert QueryOptimizer.is_consecutive([3, 1, 2, 4]) is True

    def test_adaptive_batch_ids_small(self):
        """Test adaptive_batch_ids with small list."""
        ids = [1, 2, 3, 4, 5]
        batches = QueryOptimizer.adaptive_batch_ids(ids, max_batch_size=10)
        assert batches == [[1, 2, 3, 4, 5]]

    def test_adaptive_batch_ids_consecutive(self):
        """Test adaptive_batch_ids with consecutive numbers."""
        ids = list(range(1, 21))  # 1-20
        batches = QueryOptimizer.adaptive_batch_ids(ids, max_batch_size=10)
        assert len(batches) == 2
        assert batches[0] == list(range(1, 11))
        assert batches[1] == list(range(11, 21))

    def test_adaptive_batch_ids_with_gaps(self):
        """Test adaptive_batch_ids with gaps."""
        ids = [1, 2, 3, 10, 11, 12, 20, 21, 22]
        batches = QueryOptimizer.adaptive_batch_ids(
            ids, max_batch_size=5, gap_threshold=5
        )
        assert len(batches) == 3
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [10, 11, 12]
        assert batches[2] == [20, 21, 22]

    def test_adaptive_batch_ids_large_gap(self):
        """Test adaptive_batch_ids with large gap."""
        ids = [1, 2, 3, 50, 51, 52, 100, 101, 102, 103, 104, 105]
        batches = QueryOptimizer.adaptive_batch_ids(
            ids, max_batch_size=5, gap_threshold=5
        )
        # The algorithm creates 4 batches: [1,2,3], [50,51,52], [100,101,102,103,104], [105]
        assert len(batches) == 4
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [50, 51, 52]
        assert batches[2] == [100, 101, 102, 103, 104]
        assert batches[3] == [105]

    def test_build_optimized_query_empty(self):
        """Test build_optimized_query with empty list."""
        query = QueryOptimizer.build_optimized_query([], "cell")
        assert "WHERE FALSE" in query

    def test_build_optimized_query_consecutive(self):
        """Test build_optimized_query with consecutive IDs."""
        query = QueryOptimizer.build_optimized_query([1, 2, 3, 4, 5], "cell")
        assert "BETWEEN 1 AND 5" in query
        assert "cell_integer_id" in query

    def test_build_optimized_query_non_consecutive(self):
        """Test build_optimized_query with non-consecutive IDs."""
        query = QueryOptimizer.build_optimized_query([1, 3, 5, 7], "cell")
        assert "IN (1,3,5,7)" in query
        assert "cell_integer_id" in query

    def test_build_optimized_query_gene_type(self):
        """Test build_optimized_query with gene type."""
        query = QueryOptimizer.build_optimized_query([1, 2, 3], "gene")
        assert "gene_integer_id" in query

    def test_build_optimized_query_with_batching(self):
        """Test build_optimized_query with adaptive batching."""
        ids = list(range(1, 201))  # 200 consecutive IDs
        query = QueryOptimizer.build_optimized_query(
            ids, "cell", use_adaptive_batching=True, max_batch_size=50
        )
        assert "UNION ALL" in query
        assert "BETWEEN" in query

    def test_build_optimized_query_without_batching(self):
        """Test build_optimized_query without adaptive batching."""
        ids = [1, 3, 5, 7, 9]
        query = QueryOptimizer.build_optimized_query(
            ids, "cell", use_adaptive_batching=False
        )
        assert "IN (1,3,5,7,9)" in query

    def test_normalize_slice_indices_basic(self):
        """Test _normalize_slice_indices with basic slice."""
        start, stop, step = QueryOptimizer._normalize_slice_indices(slice(10, 20), 100)
        assert start == 10
        assert stop == 20
        assert step == 1

    def test_normalize_slice_indices_none_values(self):
        """Test _normalize_slice_indices with None values."""
        start, stop, step = QueryOptimizer._normalize_slice_indices(
            slice(None, None), 100
        )
        assert start == 0
        assert stop == 100
        assert step == 1

    def test_normalize_slice_indices_negative(self):
        """Test _normalize_slice_indices with negative indices."""
        start, stop, step = QueryOptimizer._normalize_slice_indices(slice(-10, -5), 100)
        assert start == 90
        assert stop == 95
        assert step == 1

    def test_normalize_slice_indices_with_step(self):
        """Test _normalize_slice_indices with step."""
        start, stop, step = QueryOptimizer._normalize_slice_indices(
            slice(0, 20, 2), 100
        )
        assert start == 0
        assert stop == 20
        assert step == 2

    def test_normalize_slice_indices_bounds_clamping(self):
        """Test _normalize_slice_indices with bounds clamping."""
        start, stop, step = QueryOptimizer._normalize_slice_indices(
            slice(-10, 200), 100
        )
        assert start == 90  # -10 becomes 90 (100 + (-10))
        assert stop == 100  # Clamped to 100
        assert step == 1

    def test_build_slice_condition_step_one(self):
        """Test _build_slice_condition with step=1."""
        condition = QueryOptimizer._build_slice_condition(10, 20, 1, "cell")
        assert condition == "cell_integer_id >= 10 AND cell_integer_id < 20"

    def test_build_slice_condition_small_step(self):
        """Test _build_slice_condition with small step."""
        condition = QueryOptimizer._build_slice_condition(0, 10, 2, "cell")
        assert "cell_integer_id >= 0 AND cell_integer_id < 10" in condition
        assert "modulo" in condition.lower() or "%" in condition

    def test_build_slice_condition_large_step(self):
        """Test _build_slice_condition with large step."""
        condition = QueryOptimizer._build_slice_condition(0, 100, 20, "cell")
        assert "IN" in condition or "modulo" in condition.lower()

    def test_build_list_condition(self):
        """Test _build_list_condition."""
        condition = QueryOptimizer._build_list_condition([1, 3, 5], "cell", 100)
        assert "cell_integer_id IN (1,3,5)" in condition

    def test_build_list_condition_numpy_array(self):
        """Test _build_list_condition with numpy array."""
        condition = QueryOptimizer._build_list_condition(
            np.array([1, 3, 5]), "cell", 100
        )
        assert "cell_integer_id IN (1,3,5)" in condition

    def test_build_list_condition_boolean_mask(self):
        """Test _build_list_condition with boolean mask."""
        mask = np.array([True, False, True, False, True])
        condition = QueryOptimizer._build_list_condition(mask, "cell", 100)
        assert "cell_integer_id IN (0,2,4)" in condition

    def test_build_single_index_condition(self):
        """Test _build_single_index_condition."""
        condition = QueryOptimizer._build_single_index_condition(5, "cell", 100)
        assert condition == "cell_integer_id = 5"

    def test_build_submatrix_query_no_selectors(self):
        """Test build_submatrix_query with no selectors."""
        query = QueryOptimizer.build_submatrix_query()
        assert "SELECT" in query
        assert "FROM expression" in query

    def test_build_submatrix_query_with_selectors(self):
        """Test build_submatrix_query with selectors."""
        query = QueryOptimizer.build_submatrix_query(
            cell_selector=[1, 2, 3], gene_selector=[4, 5, 6], cell_count=3, gene_count=3
        )
        assert "SELECT" in query
        assert "FROM expression" in query
        assert "WHERE" in query

    def test_build_cte_query(self):
        """Test build_cte_query."""
        query = QueryOptimizer.build_cte_query([1, 2, 3], "cell")
        assert "WITH" in query
        assert "SELECT" in query
        assert "cell_integer_id" in query

    def test_estimate_query_strategy_consecutive(self):
        """Test estimate_query_strategy for consecutive IDs."""
        strategy = QueryOptimizer.estimate_query_strategy([1, 2, 3, 4, 5])
        assert "consecutive" in strategy.lower()

    def test_estimate_query_strategy_scattered(self):
        """Test estimate_query_strategy for scattered IDs."""
        strategy = QueryOptimizer.estimate_query_strategy([1, 10, 100, 1000])
        assert "scattered" in strategy.lower()

    def test_estimate_query_strategy_batching(self):
        """Test estimate_query_strategy for batching."""
        strategy = QueryOptimizer.estimate_query_strategy(list(range(1, 201)))
        # For 200 consecutive IDs, it should return "consecutive" not "batching"
        assert "consecutive" in strategy.lower()


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""

    def test_init(self):
        """Test PerformanceMetrics initialization."""
        metrics = PerformanceMetrics()
        assert hasattr(metrics, "query_times")

    def test_record_query(self):
        """Test record_query."""
        metrics = PerformanceMetrics()
        metrics.record_query("consecutive", 100, 0.5)
        assert len(metrics.query_times) == 1

    def test_record_multiple_queries(self):
        """Test recording multiple queries."""
        metrics = PerformanceMetrics()
        metrics.record_query("consecutive", 100, 0.5)
        metrics.record_query("scattered", 50, 0.3)
        metrics.record_query("consecutive", 200, 1.0)
        assert len(metrics.query_times) == 3

    def test_get_average_time_existing(self):
        """Test get_average_time for existing strategy and count."""
        metrics = PerformanceMetrics()
        metrics.record_query("consecutive", 100, 0.5)
        metrics.record_query("consecutive", 100, 0.7)
        avg_time = metrics.get_average_time("consecutive", 100)
        assert avg_time == 0.6

    def test_get_average_time_nonexistent(self):
        """Test get_average_time for nonexistent strategy/count."""
        metrics = PerformanceMetrics()
        avg_time = metrics.get_average_time("nonexistent", 100)
        assert avg_time is None

    def test_get_performance_summary(self):
        """Test get_performance_summary."""
        metrics = PerformanceMetrics()
        metrics.record_query("consecutive", 100, 0.5)
        metrics.record_query("scattered", 50, 0.3)
        summary = metrics.get_performance_summary()
        assert isinstance(summary, dict)
        assert "consecutive" in summary
        assert "scattered" in summary

    def test_get_performance_summary_empty(self):
        """Test get_performance_summary with no recorded queries."""
        metrics = PerformanceMetrics()
        summary = metrics.get_performance_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 0
