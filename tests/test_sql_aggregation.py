#!/usr/bin/env python3
"""
Test script to demonstrate the SQL aggregation fix for mean calculations.
This script shows the difference between the old and new behavior.
"""

import json

import numpy as np
import pandas as pd
import pytest


# Mock the SLAFArray class for testing
class MockSLAFArray:
    def __init__(self, shape=(100, 50), seed=42):
        self.shape = shape
        # Set random seed for reproducible tests
        np.random.seed(seed)
        # Create mock expression data with some sparse entries
        self.mock_expression_data = self._create_mock_data()

    def _create_mock_data(self):
        """Create mock expression data where only some cells have expression"""
        data = []
        # Only 30 out of 100 cells have expression data
        for cell_id in range(30):
            # Each cell has expression for 10-20 genes
            n_genes = np.random.randint(10, 21)
            gene_ids = np.random.choice(self.shape[1], n_genes, replace=False)
            expression_values = np.random.uniform(0.1, 5.0, n_genes)

            sparse_data = {
                str(gene_id): float(expr)
                for gene_id, expr in zip(gene_ids, expression_values, strict=False)
            }
            data.append({"cell_id": cell_id, "sparse_data": json.dumps(sparse_data)})
        return data

    def query(self, sql):
        """Mock query method that simulates the SQL aggregation behavior"""
        if "genes g" in sql and "json_extract" in sql:
            # New gene-wise behavior: use genes view with json_extract
            results = []
            for gene_id in range(self.shape[1]):
                # Calculate sum across all cells for this gene
                total_expr = 0.0
                for cell_data in self.mock_expression_data:
                    sparse_data = json.loads(cell_data["sparse_data"])
                    if str(gene_id) in sparse_data:
                        total_expr += sparse_data[str(gene_id)]
                mean_expr = total_expr / self.shape[0]  # Divide by total cells
                results.append({"gene_id": gene_id, "result": mean_expr})

            df = pd.DataFrame(results)
            return df

        elif "cells c" in sql and "json_each" in sql:
            # New cell-wise behavior: use cells view with json_each
            results = []
            for cell_id in range(self.shape[0]):
                # Find if this cell has expression data
                cell_data = next(
                    (d for d in self.mock_expression_data if d["cell_id"] == cell_id),
                    None,
                )
                if cell_data:
                    sparse_data = json.loads(cell_data["sparse_data"])
                    total_expr = sum(sparse_data.values())
                    mean_expr = total_expr / self.shape[1]  # Divide by total genes
                else:
                    mean_expr = 0.0  # No expression data
                results.append({"cell_id": cell_id, "result": mean_expr})

            df = pd.DataFrame(results)
            return df

        elif "COALESCE(" in sql and "cell_id" in sql:
            # New cell-wise behavior: include all cells
            results = []
            for cell_id in range(self.shape[0]):
                # Find if this cell has expression data
                cell_data = next(
                    (d for d in self.mock_expression_data if d["cell_id"] == cell_id),
                    None,
                )
                if cell_data:
                    sparse_data = json.loads(cell_data["sparse_data"])
                    if "SUM" in sql:
                        total_expr = sum(sparse_data.values())
                        result = (
                            total_expr / self.shape[1]
                        )  # Divide by total genes for mean
                    elif "MAX" in sql:
                        result = max(sparse_data.values())
                    elif "MIN" in sql:
                        result = min(sparse_data.values())
                    else:
                        result = 0.0
                else:
                    result = 0.0  # No expression data
                results.append({"cell_id": cell_id, "result": result})

            df = pd.DataFrame(results)
            return df

        elif "AVG" in sql and "cell_id" in sql:
            # Old behavior: only cells with expression data
            results = []
            for cell_data in self.mock_expression_data:
                cell_id = cell_data["cell_id"]
                sparse_data = json.loads(cell_data["sparse_data"])
                # Old behavior: only consider non-zero genes
                mean_expr = np.mean(list(sparse_data.values()))
                results.append({"cell_id": cell_id, "result": mean_expr})
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()


# Mock the LazySparseMixin for testing
class MockLazySparseMixin:
    def __init__(self, slaf_array):
        self.slaf_array = slaf_array
        self.shape = slaf_array.shape
        # Mock obs_names as integer range
        self.obs_names = pd.Index(range(self.shape[0]))

    def _sql_aggregation(self, operation, axis=None):
        """Test the fixed aggregation logic"""
        if axis == 1:  # Cell-wise aggregation
            if operation.upper() == "AVG":
                # New behavior: include all cells
                sql = f"""
                SELECT
                    cell_id,
                    COALESCE(SUM(CAST(j.value AS FLOAT)), 0.0) / {self.shape[1]} as result
                FROM expression AS e, LATERAL json_each(e.sparse_data) AS j
                GROUP BY cell_id
                ORDER BY cell_id
                """
            else:
                sql = f"""
                SELECT
                    cell_id,
                    COALESCE({operation.upper()}(CAST(j.value AS FLOAT)), 0.0) as result
                FROM expression AS e, LATERAL json_each(e.sparse_data) AS j
                GROUP BY cell_id
                ORDER BY cell_id
                """

            result_df = self.slaf_array.query(sql)

            # Create result array for ALL cells
            full_result = np.zeros(self.shape[0])

            # Get all cell IDs to ensure we have results for all cells
            all_cell_ids = self.obs_names

            # Create a mapping from cell IDs to their results
            result_map = {}
            for _, row in result_df.iterrows():
                cell_id = row["cell_id"]
                result_map[cell_id] = row["result"]

            # Fill in results for all cells
            for i, cell_id in enumerate(all_cell_ids):
                if cell_id in result_map:
                    full_result[i] = result_map[cell_id]
                else:
                    # Cell has no expression data, result depends on operation
                    if operation.upper() == "AVG":
                        full_result[i] = 0.0  # Mean of all zeros is 0
                    elif operation.upper() == "SUM":
                        full_result[i] = 0.0  # Sum of no values is 0
                    elif operation.upper() == "MAX":
                        full_result[i] = 0.0  # Max of no values is 0
                    elif operation.upper() == "MIN":
                        full_result[i] = 0.0  # Min of no values is 0
                    else:
                        full_result[i] = 0.0

            return full_result
        else:
            # For other axes or global aggregation, return a simple result
            return np.array([0.0])


# Pytest fixtures
@pytest.fixture
def mock_slaf_array():
    """Fixture to create a mock SLAFArray with reproducible data."""
    return MockSLAFArray(shape=(100, 50), seed=42)


@pytest.fixture
def mock_lazy_sparse_mixin(mock_slaf_array):
    """Fixture to create a mock LazySparseMixin with the mock SLAFArray."""
    return MockLazySparseMixin(mock_slaf_array)


@pytest.fixture
def expected_stats():
    """Fixture with expected statistics for the test data."""
    return {
        "total_cells": 100,
        "cells_with_data": 30,
        "cells_without_data": 70,
        "total_genes": 50,
    }


# Test functions
def test_mock_data_creation(mock_slaf_array, expected_stats):
    """Test that mock data is created correctly."""
    assert mock_slaf_array.shape == (100, 50)
    assert (
        len(mock_slaf_array.mock_expression_data) == expected_stats["cells_with_data"]
    )

    # Check that only cells 0-29 have data
    cell_ids_with_data = [d["cell_id"] for d in mock_slaf_array.mock_expression_data]
    assert set(cell_ids_with_data) == set(range(30))

    # Check that all cells with data have sparse_data
    for cell_data in mock_slaf_array.mock_expression_data:
        assert "sparse_data" in cell_data
        sparse_data = json.loads(cell_data["sparse_data"])
        assert len(sparse_data) > 0


def test_cell_wise_mean_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test the fixed cell-wise mean calculation."""
    cell_means = mock_lazy_sparse_mixin._sql_aggregation("avg", axis=1)

    # Basic assertions
    assert len(cell_means) == expected_stats["total_cells"]
    assert np.sum(cell_means > 0) == expected_stats["cells_with_data"]
    assert np.sum(cell_means == 0) == expected_stats["cells_without_data"]

    # Check that overall average is lower than non-zero average
    non_zero_means = cell_means[cell_means > 0]
    assert len(non_zero_means) > 0
    assert cell_means.mean() < non_zero_means.mean()


def test_cell_wise_sum_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise sum aggregation."""
    cell_sums = mock_lazy_sparse_mixin._sql_aggregation("sum", axis=1)

    assert len(cell_sums) == expected_stats["total_cells"]
    assert np.sum(cell_sums > 0) == expected_stats["cells_with_data"]
    assert np.sum(cell_sums == 0) == expected_stats["cells_without_data"]


def test_cell_wise_max_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise max aggregation."""
    cell_maxs = mock_lazy_sparse_mixin._sql_aggregation("max", axis=1)

    assert len(cell_maxs) == expected_stats["total_cells"]
    assert np.sum(cell_maxs > 0) == expected_stats["cells_with_data"]
    assert np.sum(cell_maxs == 0) == expected_stats["cells_without_data"]


def test_cell_wise_min_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise min aggregation."""
    cell_mins = mock_lazy_sparse_mixin._sql_aggregation("min", axis=1)

    assert len(cell_mins) == expected_stats["total_cells"]
    assert np.sum(cell_mins > 0) == expected_stats["cells_with_data"]
    assert np.sum(cell_mins == 0) == expected_stats["cells_without_data"]


def test_gene_wise_aggregation(mock_lazy_sparse_mixin):
    """Test gene-wise aggregation (currently returns simple result)."""
    gene_means = mock_lazy_sparse_mixin._sql_aggregation("avg", axis=0)

    # Currently returns simple result for gene-wise aggregation
    assert len(gene_means) == 1
    assert gene_means[0] == 0.0


def test_aggregation_statistics(mock_lazy_sparse_mixin, expected_stats):
    """Test that aggregation statistics are reasonable."""
    cell_means = mock_lazy_sparse_mixin._sql_aggregation("avg", axis=1)

    # Check statistics
    assert cell_means.min() >= 0.0  # All means should be non-negative
    assert cell_means.max() > 0.0  # Should have some positive means

    # Check that the distribution makes sense
    non_zero_means = cell_means[cell_means > 0]
    if len(non_zero_means) > 0:
        assert (
            non_zero_means.mean() > cell_means.mean()
        )  # Non-zero average > overall average


def test_reproducibility():
    """Test that the same seed produces the same results."""
    # Create two arrays with the same seed
    array1 = MockSLAFArray(shape=(100, 50), seed=42)
    array2 = MockSLAFArray(shape=(100, 50), seed=42)

    mixin1 = MockLazySparseMixin(array1)
    mixin2 = MockLazySparseMixin(array2)

    # Get results
    result1 = mixin1._sql_aggregation("avg", axis=1)
    result2 = mixin2._sql_aggregation("avg", axis=1)

    # Results should be identical
    np.testing.assert_array_equal(result1, result2)


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    # Create two arrays with different seeds
    array1 = MockSLAFArray(shape=(100, 50), seed=42)
    array2 = MockSLAFArray(shape=(100, 50), seed=123)

    mixin1 = MockLazySparseMixin(array1)
    mixin2 = MockLazySparseMixin(array2)

    # Get results
    result1 = mixin1._sql_aggregation("avg", axis=1)
    result2 = mixin2._sql_aggregation("avg", axis=1)

    # Results should be different
    assert not np.array_equal(result1, result2)


def test_sql_query_patterns(mock_slaf_array):
    """Test that SQL queries are generated and processed correctly."""
    # Test different SQL patterns
    test_cases = [
        ("genes g", "json_extract"),
        ("cells c", "json_each"),
        ("COALESCE(SUM", "cell_id"),
        ("AVG", "cell_id"),
    ]

    for pattern1, pattern2 in test_cases:
        # Create a mock SQL query that would match the pattern
        mock_sql = f"SELECT something FROM table WHERE {pattern1} AND {pattern2}"
        result = mock_slaf_array.query(mock_sql)

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
