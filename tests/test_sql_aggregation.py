#!/usr/bin/env python3
"""
Test script to demonstrate the SQL aggregation fix for mean calculations.
This script shows the difference between the old and new behavior.
"""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest


class MockSLAFArray:
    """Mock SLAFArray for testing SQL aggregation."""

    def __init__(self, shape=(100, 50), seed=42):
        self.shape = shape
        np.random.seed(seed)
        self._create_mock_data()

    def _create_mock_data(self):
        """Create mock expression data."""
        self.mock_expression_data = []
        n_cells, n_genes = self.shape

        # Create sparse expression data for each cell
        for cell_id in range(n_cells):
            # Randomly select genes to express
            n_expressed_genes = np.random.randint(1, min(10, n_genes + 1))
            expressed_genes = np.random.choice(
                n_genes, n_expressed_genes, replace=False
            )

            # Create sparse data as JSON
            sparse_data = {}
            for gene_id in expressed_genes:
                sparse_data[str(gene_id)] = np.random.uniform(0.1, 5.0)

            self.mock_expression_data.append(
                {"cell_id": cell_id, "sparse_data": json.dumps(sparse_data)}
            )

    def query(self, sql):
        """Mock query method that returns polars DataFrames."""
        if "genes AS g" in sql and "json_each" in sql:
            # Gene-wise behavior: use genes view with json_each
            results = []
            for gene_id in range(self.shape[1]):
                # Find if this gene has expression data
                total_expr = 0.0
                for cell_data in self.mock_expression_data:
                    sparse_data = json.loads(cell_data["sparse_data"])
                    if str(gene_id) in sparse_data:
                        total_expr += sparse_data[str(gene_id)]
                mean_expr = total_expr / self.shape[0]  # Divide by total cells
                results.append({"gene_id": gene_id, "result": mean_expr})

            df = pl.DataFrame(results)
            return df

        elif "expression AS e" in sql and "json_each" in sql:
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

            df = pl.DataFrame(results)
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

            df = pl.DataFrame(results)
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
            return pl.DataFrame(results)

        # Handle simple gene queries
        elif "FROM genes" in sql and "gene_id" in sql:
            results = []
            for gene_id in range(self.shape[1]):
                # Find if this gene has expression data
                total_expr = 0.0
                for cell_data in self.mock_expression_data:
                    sparse_data = json.loads(cell_data["sparse_data"])
                    if str(gene_id) in sparse_data:
                        total_expr += sparse_data[str(gene_id)]
                mean_expr = total_expr / self.shape[0]  # Divide by total cells
                results.append({"gene_id": gene_id, "result": mean_expr})

            df = pl.DataFrame(results)
            return df

        # Handle simple cell queries
        elif "FROM cells" in sql and "cell_id" in sql:
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

            df = pl.DataFrame(results)
            return df
        else:
            return pl.DataFrame()


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
        else:  # Gene-wise aggregation
            sql = """
            SELECT
                gene_id,
                AVG(CAST(j.value AS FLOAT)) as result
            FROM genes AS g, LATERAL json_each(g.sparse_data) AS j
            GROUP BY gene_id
            ORDER BY gene_id
            """

        return self.slaf_array.query(sql)

    def _sql_mean_aggregation(self, axis=None):
        """Test mean aggregation specifically."""
        return self._sql_aggregation("AVG", axis)

    def _sql_variance_aggregation(self, axis=None):
        """Test variance aggregation specifically."""
        if axis == 1:  # Cell-wise aggregation
            sql = """
            SELECT
                cell_id,
                COALESCE(VAR(CAST(j.value AS FLOAT)), 0.0) as result
            FROM expression AS e, LATERAL json_each(e.sparse_data) AS j
            GROUP BY cell_id
            ORDER BY cell_id
            """
        else:  # Gene-wise aggregation
            sql = """
            SELECT
                gene_id,
                VAR(CAST(j.value AS FLOAT)) as result
            FROM genes AS g, LATERAL json_each(g.sparse_data) AS j
            GROUP BY gene_id
            ORDER BY gene_id
            """

        return self.slaf_array.query(sql)

    def _sql_other_aggregation(self, operation, axis=None):
        """Test other aggregation operations."""
        return self._sql_aggregation(operation, axis)


@pytest.fixture
def mock_slaf_array():
    """Create a mock SLAFArray for testing."""
    return MockSLAFArray(shape=(100, 50), seed=42)


@pytest.fixture
def mock_lazy_sparse_mixin(mock_slaf_array):
    """Create a mock LazySparseMixin for testing."""
    return MockLazySparseMixin(mock_slaf_array)


@pytest.fixture
def expected_stats():
    """Expected statistics for the mock data."""
    # These are approximate values based on the mock data generation
    return {
        "cell_count": 100,
        "gene_count": 50,
        "min_cell_mean": 0.0,  # Some cells might have no expression
        "max_cell_mean": 5.0,  # Maximum expression value
        "min_gene_mean": 0.0,  # Some genes might have no expression
        "max_gene_mean": 5.0,  # Maximum expression value
    }


def test_mock_data_creation(mock_slaf_array, expected_stats):
    """Test that mock data is created correctly."""
    assert mock_slaf_array.shape == (100, 50)
    assert len(mock_slaf_array.mock_expression_data) == expected_stats["cell_count"]

    # Check that each cell has some expression data
    for cell_data in mock_slaf_array.mock_expression_data:
        assert "cell_id" in cell_data
        assert "sparse_data" in cell_data
        sparse_data = json.loads(cell_data["sparse_data"])
        assert isinstance(sparse_data, dict)
        assert len(sparse_data) > 0  # Each cell should have some expression


def test_cell_wise_mean_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise mean aggregation."""
    result = mock_lazy_sparse_mixin._sql_mean_aggregation(axis=1)

    # Check that we got a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check that it has the expected columns
    assert "cell_id" in result.columns
    assert "result" in result.columns

    # Check that we got results for all cells
    assert len(result) == expected_stats["cell_count"]

    # Check that all results are within expected range
    if len(result) > 0:
        assert all(result["result"] >= expected_stats["min_cell_mean"])
        assert all(result["result"] <= expected_stats["max_cell_mean"])


def test_cell_wise_sum_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise sum aggregation."""
    result = mock_lazy_sparse_mixin._sql_other_aggregation("SUM", axis=1)

    # Check that we got a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check that it has the expected columns
    assert "cell_id" in result.columns
    assert "result" in result.columns

    # Check that we got results for all cells
    assert len(result) == expected_stats["cell_count"]


def test_cell_wise_max_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise max aggregation."""
    result = mock_lazy_sparse_mixin._sql_other_aggregation("MAX", axis=1)

    # Check that we got a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check that it has the expected columns
    assert "cell_id" in result.columns
    assert "result" in result.columns

    # Check that we got results for all cells
    assert len(result) == expected_stats["cell_count"]


def test_cell_wise_min_aggregation(mock_lazy_sparse_mixin, expected_stats):
    """Test cell-wise min aggregation."""
    result = mock_lazy_sparse_mixin._sql_other_aggregation("MIN", axis=1)

    # Check that we got a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check that it has the expected columns
    assert "cell_id" in result.columns
    assert "result" in result.columns

    # Check that we got results for all cells
    assert len(result) == expected_stats["cell_count"]


def test_gene_wise_aggregation(mock_lazy_sparse_mixin):
    """Test gene-wise aggregation."""
    result = mock_lazy_sparse_mixin._sql_mean_aggregation(axis=0)

    # Check that we got a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check that it has the expected columns
    assert "gene_id" in result.columns
    assert "result" in result.columns

    # Check that we got results for all genes
    assert len(result) == 50


def test_aggregation_statistics(mock_lazy_sparse_mixin, expected_stats):
    """Test that aggregation produces reasonable statistics."""
    # Test cell-wise mean
    cell_means = mock_lazy_sparse_mixin._sql_mean_aggregation(axis=1)
    assert len(cell_means) == expected_stats["cell_count"]

    # Test gene-wise mean
    gene_means = mock_lazy_sparse_mixin._sql_mean_aggregation(axis=0)
    assert len(gene_means) == expected_stats["gene_count"]

    # Test variance aggregation
    cell_vars = mock_lazy_sparse_mixin._sql_variance_aggregation(axis=1)
    assert len(cell_vars) == expected_stats["cell_count"]

    gene_vars = mock_lazy_sparse_mixin._sql_variance_aggregation(axis=0)
    assert len(gene_vars) == expected_stats["gene_count"]


def test_reproducibility():
    """Test that the same seed produces the same results."""
    # Create two mock arrays with the same seed
    mock1 = MockSLAFArray(shape=(10, 5), seed=42)
    mock2 = MockSLAFArray(shape=(10, 5), seed=42)

    # Query both with the same SQL
    result1 = mock1.query("SELECT cell_id, result FROM cells")
    result2 = mock2.query("SELECT cell_id, result FROM cells")

    # Results should be identical
    assert result1.equals(result2)


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    # Create two mock arrays with different seeds
    mock1 = MockSLAFArray(shape=(10, 5), seed=42)
    mock2 = MockSLAFArray(shape=(10, 5), seed=123)

    # Query both with the same SQL
    result1 = mock1.query("SELECT cell_id, result FROM cells")
    result2 = mock2.query("SELECT cell_id, result FROM cells")

    # Results should be different (not equal)
    assert not result1.equals(result2)


def test_sql_query_patterns(mock_slaf_array):
    """Test different SQL query patterns."""
    # Test gene-wise query
    gene_result = mock_slaf_array.query("SELECT gene_id, result FROM genes")
    assert isinstance(gene_result, pl.DataFrame)
    assert "gene_id" in gene_result.columns
    assert "result" in gene_result.columns

    # Test cell-wise query
    cell_result = mock_slaf_array.query("SELECT cell_id, result FROM cells")
    assert isinstance(cell_result, pl.DataFrame)
    assert "cell_id" in cell_result.columns
    assert "result" in cell_result.columns

    # Test empty result
    empty_result = mock_slaf_array.query("SELECT * FROM nonexistent")
    assert isinstance(empty_result, pl.DataFrame)
    assert len(empty_result) == 0
