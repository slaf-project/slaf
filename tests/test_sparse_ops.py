"""Tests for SLAF sparse operations."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from slaf.core.sparse_ops import LazySparseMixin


class TestLazySparseMixin:
    """Test LazySparseMixin functionality."""

    class TestMixin(LazySparseMixin):
        """Test implementation of LazySparseMixin."""

        __test__ = False  # Tell pytest this is not a test class

        def __init__(
            self, shape: tuple[int, int], slaf_array, obs_names=None, var_names=None
        ):
            super().__init__()
            self._shape = shape
            self.slaf_array = slaf_array
            self._obs_names = obs_names
            self._var_names = var_names

        @property
        def shape(self) -> tuple[int, int]:
            return self._shape

        @property
        def obs_names(self):
            return self._obs_names

        @property
        def var_names(self):
            return self._var_names

    @pytest.fixture
    def mock_slaf_array(self):
        """Create a mock SLAFArray for testing."""
        mock_array = Mock()
        mock_array.get_submatrix.return_value = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2],
                "gene_integer_id": [0, 1, 2],
                "expression": [1.0, 2.0, 3.0],
            }
        )
        return mock_array

    @pytest.fixture
    def sparse_mixin(self, mock_slaf_array):
        """Create a LazySparseMixin instance for testing."""
        mixin = self.TestMixin(
            shape=(100, 200),
            slaf_array=mock_slaf_array,
            obs_names=pd.Index([f"cell_{i}" for i in range(100)]),
            var_names=pd.Index([f"gene_{i}" for i in range(200)]),
        )
        return mixin

    def test_parse_key_single_index(self, sparse_mixin):
        """Test parsing single index key."""
        cell_sel, gene_sel = sparse_mixin._parse_key(5)
        assert cell_sel == 5
        assert gene_sel == slice(None)

    def test_parse_key_tuple(self, sparse_mixin):
        """Test parsing tuple key."""
        cell_sel, gene_sel = sparse_mixin._parse_key((10, 20))
        assert cell_sel == 10
        assert gene_sel == 20

    def test_parse_key_single_tuple(self, sparse_mixin):
        """Test parsing single element tuple."""
        cell_sel, gene_sel = sparse_mixin._parse_key((5,))
        assert cell_sel == 5
        assert gene_sel == slice(None)

    def test_parse_key_np_ix(self, sparse_mixin):
        """Test parsing numpy ix_ style indexing."""
        row_selector = np.array([[0, 1], [2, 3]])
        col_selector = np.array([[0], [1]])
        cell_sel, gene_sel = sparse_mixin._parse_key((row_selector, col_selector))
        assert np.array_equal(cell_sel, np.array([0, 1, 2, 3]))
        assert np.array_equal(gene_sel, np.array([0, 1]))

    def test_parse_key_too_many_indices(self, sparse_mixin):
        """Test error for too many indices."""
        with pytest.raises(IndexError, match="Too many indices"):
            sparse_mixin._parse_key((1, 2, 3))

    def test_selector_to_sql_condition_none(self, sparse_mixin):
        """Test SQL condition for None selector."""
        condition = sparse_mixin._selector_to_sql_condition(None, 0, "cell")
        assert condition == "TRUE"

    def test_selector_to_sql_condition_slice_all(self, sparse_mixin):
        """Test SQL condition for slice(None)."""
        condition = sparse_mixin._selector_to_sql_condition(slice(None), 0, "cell")
        assert condition == "TRUE"

    def test_selector_to_sql_condition_slice(self, sparse_mixin):
        """Test SQL condition for slice."""
        condition = sparse_mixin._selector_to_sql_condition(slice(10, 20), 0, "cell")
        assert condition == "cell_integer_id >= 10 AND cell_integer_id < 20"

    def test_selector_to_sql_condition_slice_with_step(self, sparse_mixin):
        """Test SQL condition for slice with step."""
        condition = sparse_mixin._selector_to_sql_condition(slice(0, 10, 2), 0, "cell")
        assert "cell_integer_id IN (0,2,4,6,8)" in condition

    def test_selector_to_sql_condition_list(self, sparse_mixin):
        """Test SQL condition for list selector."""
        condition = sparse_mixin._selector_to_sql_condition([1, 3, 5], 0, "cell")
        assert condition == "cell_integer_id IN (1,3,5)"

    def test_selector_to_sql_condition_array(self, sparse_mixin):
        """Test SQL condition for numpy array."""
        condition = sparse_mixin._selector_to_sql_condition(
            np.array([1, 3, 5]), 0, "cell"
        )
        assert condition == "cell_integer_id IN (1,3,5)"

    def test_selector_to_sql_condition_boolean_mask(self, sparse_mixin):
        """Test SQL condition for boolean mask."""
        mask = np.array([True, False, True, False, True])
        condition = sparse_mixin._selector_to_sql_condition(mask, 0, "cell")
        assert condition == "cell_integer_id IN (0,2,4)"

    def test_selector_to_sql_condition_boolean_mask_all_true(self, sparse_mixin):
        """Test SQL condition for all-True boolean mask."""
        mask = np.array([True, True, True])
        condition = sparse_mixin._selector_to_sql_condition(mask, 0, "cell")
        assert condition == "TRUE"

    def test_selector_to_sql_condition_boolean_mask_all_false(self, sparse_mixin):
        """Test SQL condition for all-False boolean mask."""
        mask = np.array([False, False, False])
        condition = sparse_mixin._selector_to_sql_condition(mask, 0, "cell")
        assert condition == "FALSE"

    def test_selector_to_sql_condition_integer(self, sparse_mixin):
        """Test SQL condition for integer selector."""
        condition = sparse_mixin._selector_to_sql_condition(5, 0, "cell")
        assert condition == "cell_integer_id = 5"

    def test_selector_to_sql_condition_gene_type(self, sparse_mixin):
        """Test SQL condition for gene type selector."""
        condition = sparse_mixin._selector_to_sql_condition(slice(10, 20), 1, "gene")
        assert condition == "gene_integer_id >= 10 AND gene_integer_id < 20"

    def test_selector_to_sql_condition_unsupported_type(self, sparse_mixin):
        """Test error for unsupported selector type."""
        with pytest.raises(TypeError, match="Unsupported selector type"):
            sparse_mixin._selector_to_sql_condition("invalid", 0, "cell")

    def test_selector_to_range_none(self, sparse_mixin):
        """Test range conversion for None selector."""
        start, stop = sparse_mixin._selector_to_range(None, 0)
        assert start == 0
        assert stop == 100

    def test_selector_to_range_slice(self, sparse_mixin):
        """Test range conversion for slice selector."""
        start, stop = sparse_mixin._selector_to_range(slice(10, 20), 0)
        assert start == 10
        assert stop == 20

    def test_selector_to_range_list(self, sparse_mixin):
        """Test range conversion for list selector."""
        start, stop = sparse_mixin._selector_to_range([1, 3, 5], 0)
        assert start == 1
        assert stop == 6

    def test_selector_to_range_boolean_mask(self, sparse_mixin):
        """Test range conversion for boolean mask."""
        mask = np.array([False, True, False, True, False])
        start, stop = sparse_mixin._selector_to_range(mask, 0)
        assert start == 1
        assert stop == 4

    def test_selector_to_range_empty_list(self, sparse_mixin):
        """Test range conversion for empty list."""
        start, stop = sparse_mixin._selector_to_range([], 0)
        assert start == 0
        assert stop == 0

    def test_estimate_selected_count_none(self, sparse_mixin):
        """Test count estimation for None selector."""
        count = sparse_mixin._estimate_selected_count(None, 0)
        assert count == 100

    def test_estimate_selected_count_slice(self, sparse_mixin):
        """Test count estimation for slice selector."""
        count = sparse_mixin._estimate_selected_count(slice(10, 20), 0)
        assert count == 10

    def test_estimate_selected_count_list(self, sparse_mixin):
        """Test count estimation for list selector."""
        count = sparse_mixin._estimate_selected_count([1, 3, 5], 0)
        assert count == 3

    def test_estimate_selected_count_boolean_mask(self, sparse_mixin):
        """Test count estimation for boolean mask."""
        mask = np.array([True, False, True, False, True])
        count = sparse_mixin._estimate_selected_count(mask, 0)
        assert count == 3

    def test_estimate_selected_count_integer(self, sparse_mixin):
        """Test count estimation for integer selector."""
        count = sparse_mixin._estimate_selected_count(5, 0)
        assert count == 1

    def test_build_submatrix_query(self, sparse_mixin):
        """Test building submatrix query."""
        result = sparse_mixin._build_submatrix_query(slice(0, 10), slice(0, 20), None)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_integer_id" in result.columns
        assert "gene_integer_id" in result.columns
        assert "expression" in result.columns

    def test_get_result_shape(self, sparse_mixin):
        """Test result shape calculation."""
        shape = sparse_mixin._get_result_shape(slice(0, 10), slice(0, 20))
        assert shape == (10, 20)

    def test_get_result_shape_boolean_mask(self, sparse_mixin):
        """Test result shape calculation with boolean mask."""
        mask = np.array([True, False, True, False, True])
        shape = sparse_mixin._get_result_shape(mask, slice(0, 20))
        assert shape == (3, 20)

    def test_sql_aggregation(self, sparse_mixin):
        """Test SQL aggregation."""
        # Mock the query method to return a polars DataFrame
        sparse_mixin.slaf_array.query.return_value = pl.DataFrame({"result": [42.0]})

        result = sparse_mixin._sql_aggregation("SUM", None)
        assert isinstance(result, np.ndarray)
        assert result[0] == 42.0

    def test_sql_mean_aggregation(self, sparse_mixin):
        """Test SQL mean aggregation."""
        # Mock the query method to return a polars DataFrame
        sparse_mixin.slaf_array.query.return_value = pl.DataFrame(
            {"total_sum": [100.0]}
        )

        result = sparse_mixin._sql_mean_aggregation(None)
        assert isinstance(result, np.ndarray)
        assert result[0] == 100.0 / (100 * 200)  # total_sum / (n_cells * n_genes)

    def test_sql_variance_aggregation(self, sparse_mixin):
        """Test SQL variance aggregation."""
        # Mock the query method to return a polars DataFrame
        sparse_mixin.slaf_array.query.return_value = pl.DataFrame(
            {"total_sum": [100.0], "sum_squares": [200.0]}
        )

        result = sparse_mixin._sql_variance_aggregation(None)
        assert isinstance(result, np.ndarray)
        # Variance calculation: (sum_squares / n) - (mean^2)
        n = 100 * 200
        mean = 100.0 / n
        expected_variance = (200.0 / n) - (mean * mean)
        assert abs(result[0] - expected_variance) < 1e-10

    def test_sql_other_aggregation(self, sparse_mixin):
        """Test SQL other aggregation."""
        # Mock the query method to return a polars DataFrame
        sparse_mixin.slaf_array.query.return_value = pl.DataFrame({"result": [42.0]})

        result = sparse_mixin._sql_other_aggregation("MAX", None)
        assert isinstance(result, np.ndarray)
        assert result[0] == 42.0

    def test_sql_multi_aggregation(self, sparse_mixin):
        """Test SQL multi-aggregation."""
        # Mock the query method to return a polars DataFrame
        sparse_mixin.slaf_array.query.return_value = pl.DataFrame(
            {
                "mean_sum": [100.0],
                "var_sum": [200.0],
                "var_sum_squares": [400.0],
                "max_result": [42.0],
            }
        )

        result = sparse_mixin._sql_multi_aggregation(["mean", "var", "max"], None)
        assert isinstance(result, dict)
        assert "mean" in result
        assert "var" in result
        assert "max" in result

    def test_reconstruct_sparse_matrix(self, sparse_mixin):
        """Test sparse matrix reconstruction."""
        # Create a mock polars DataFrame
        records = pl.DataFrame(
            {
                "cell_integer_id": [0, 1, 2],
                "gene_integer_id": [0, 1, 2],
                "value": [1.0, 2.0, 3.0],
            }
        )

        # Mock the shape attribute properly
        sparse_mixin.slaf_array.shape = (10, 20)

        result = sparse_mixin._reconstruct_sparse_matrix(
            records, slice(0, 10), slice(0, 20)
        )

        # Check that we got a sparse matrix
        import scipy.sparse

        assert isinstance(result, scipy.sparse.csr_matrix)
        assert result.shape == (10, 20)

    def test_create_id_mapping(self, sparse_mixin):
        """Test ID mapping creation."""
        mapping = sparse_mixin._create_id_mapping(slice(0, 10), 0)
        assert isinstance(mapping, dict)
        assert len(mapping) == 10
        assert mapping[0] == 0
        assert mapping[9] == 9

    def test_boolean_mask_to_sql(self, sparse_mixin):
        """Test boolean mask to SQL conversion."""
        mask = np.array([True, False, True, False, True])
        condition = sparse_mixin._boolean_mask_to_sql(mask, "cell_id")
        assert condition == "cell_id IN (0,2,4)"

    def test_boolean_mask_to_sql_all_true(self, sparse_mixin):
        """Test boolean mask to SQL conversion for all True."""
        mask = np.array([True, True, True])
        condition = sparse_mixin._boolean_mask_to_sql(mask, "cell_id")
        assert condition == "TRUE"

    def test_boolean_mask_to_sql_all_false(self, sparse_mixin):
        """Test boolean mask to SQL conversion for all False."""
        mask = np.array([False, False, False])
        condition = sparse_mixin._boolean_mask_to_sql(mask, "cell_id")
        assert condition == "FALSE"
