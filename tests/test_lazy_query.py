"""Tests for LazyQuery composition functionality."""

import duckdb
import pytest

from slaf.core.lazy_query import LazyQuery


class TestLazyQuery:
    """Test suite for LazyQuery class"""

    @pytest.fixture
    def duckdb_conn(self):
        """Create a DuckDB connection for testing."""
        conn = duckdb.connect()

        # Create test data
        conn.execute(
            """
            CREATE TABLE cells AS SELECT * FROM (
                VALUES
                    ('cell_1', 'T-cell', 1000),
                    ('cell_2', 'B-cell', 1500),
                    ('cell_3', 'T-cell', 800),
                    ('cell_4', 'B-cell', 1200)
            ) AS t(cell_id, cell_type, total_counts)
        """
        )

        return conn

    def test_basic_query(self, duckdb_conn):
        """Test basic LazyQuery creation and execution."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        result = query.compute()

        assert len(result) == 4
        assert list(result.columns) == ["cell_id", "cell_type", "total_counts"]

    def test_filter_composition(self, duckdb_conn):
        """Test query composition with filter."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        filtered = query.filter("cell_type = 'T-cell'")
        result = filtered.compute()

        assert len(result) == 2
        assert all(result["cell_type"] == "T-cell")

    def test_select_composition(self, duckdb_conn):
        """Test query composition with select."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        selected = query.select("cell_id, cell_type")
        result = selected.compute()

        assert len(result) == 4
        assert list(result.columns) == ["cell_id", "cell_type"]
        assert "total_counts" not in result.columns

    def test_group_by_composition(self, duckdb_conn):
        """Test query composition with group by."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        grouped = query.group_by("cell_type").select("cell_type, COUNT(*) as count")
        result = grouped.compute()

        assert len(result) == 2
        assert "count" in result.columns
        assert result[result["cell_type"] == "T-cell"]["count"].iloc[0] == 2

    def test_multiple_compositions(self, duckdb_conn):
        """Test multiple query compositions."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        composed = (
            query.filter("total_counts >= 800")  # Changed to include both cell types
            .select("cell_type, AVG(total_counts) as avg_counts")
            .group_by("cell_type")
            .order_by("avg_counts DESC")
        )

        result = composed.compute()

        assert len(result) == 2
        assert "avg_counts" in result.columns
        # B-cells should have higher average counts
        b_cell_avg = result[result["cell_type"] == "B-cell"]["avg_counts"].iloc[0]
        t_cell_avg = result[result["cell_type"] == "T-cell"]["avg_counts"].iloc[0]
        assert b_cell_avg > t_cell_avg

    def test_limit_composition(self, duckdb_conn):
        """Test query composition with limit."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        limited = query.limit(2)
        result = limited.compute()

        assert len(result) == 2

    def test_immutability(self, duckdb_conn):
        """Test that LazyQuery operations are immutable."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        original_result = query.compute()

        # Apply filter to original query
        filtered = query.filter("cell_type = 'T-cell'")
        filtered_result = filtered.compute()

        # Original query should be unchanged
        unchanged_result = query.compute()

        assert len(original_result) == 4
        assert len(filtered_result) == 2
        assert len(unchanged_result) == 4

    def test_sql_building(self, duckdb_conn):
        """Test that SQL is built correctly."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        composed = query.filter("cell_type = 'T-cell'").select("cell_id, cell_type")

        # Check the built SQL
        sql = composed._build_sql()
        assert "WHERE cell_type = 'T-cell'" in sql
        assert "SELECT cell_id, cell_type" in sql

    def test_repr(self, duckdb_conn):
        """Test string representation."""
        query = LazyQuery(duckdb_conn, "SELECT * FROM cells")
        composed = query.filter("cell_type = 'T-cell'")

        repr_str = repr(composed)
        assert "LazyQuery" in repr_str
        assert "SELECT" in repr_str
