"""Tests for SLAF query optimization."""

import numpy as np

from slaf.core.query_optimizer import PerformanceMetrics, QueryOptimizer


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
