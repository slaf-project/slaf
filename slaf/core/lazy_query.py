"""
LazyQuery class for composable SQL queries.

This module provides a LazyQuery class that wraps DuckDB query objects
and enables query composition without materialization.
"""

import duckdb
import pandas as pd


class LazyQuery:
    """
    A lazy query wrapper that supports composition without materialization.

    LazyQuery objects can be composed using methods like filter(), select(),
    group_by(), etc. The actual query is only executed when compute() is called.

    Examples:
        >>> # Create a lazy query
        >>> query = LazyQuery(conn, "SELECT * FROM cells")
        >>>
        >>> # Compose operations
        >>> filtered = query.filter("cell_type = 'T-cell'")
        >>> selected = filtered.select("cell_id, cell_type, total_counts")
        >>> grouped = selected.group_by("cell_type").select("cell_type, COUNT(*) as count")
        >>>
        >>> # Execute the composed query
        >>> result = grouped.compute()
    """

    def __init__(
        self,
        duckdb_conn: duckdb.DuckDBPyConnection,
        sql_or_query: str,
        lance_datasets: dict | None = None,
    ):
        """
        Initialize a LazyQuery.

        Args:
            duckdb_conn: DuckDB connection object
            sql_or_query: SQL query string or existing query
            lance_datasets: Dictionary of Lance datasets for table references
        """
        self.conn = duckdb_conn
        self.base_sql = sql_or_query
        self.operations: list[tuple[str, str]] = []
        self.lance_datasets = lance_datasets or {}

    def filter(self, condition: str) -> "LazyQuery":
        """
        Add a WHERE clause to the query.

        Args:
            condition: SQL WHERE condition (e.g., "cell_type = 'T-cell'")

        Returns:
            New LazyQuery with the filter applied
        """
        new_query = LazyQuery(self.conn, self.base_sql, self.lance_datasets)
        new_query.operations = self.operations + [("WHERE", condition)]
        return new_query

    def select(self, columns: str | list[str]) -> "LazyQuery":
        """
        Modify the SELECT clause of the query.

        Args:
            columns: Column names to select (string or list)

        Returns:
            New LazyQuery with the select applied
        """
        if isinstance(columns, list):
            columns = ", ".join(columns)

        new_query = LazyQuery(self.conn, self.base_sql, self.lance_datasets)
        new_query.operations = self.operations + [("SELECT", columns)]
        return new_query

    def group_by(self, columns: str | list[str]) -> "LazyQuery":
        """
        Add a GROUP BY clause to the query.

        Args:
            columns: Column names to group by (string or list)

        Returns:
            New LazyQuery with the group by applied
        """
        if isinstance(columns, list):
            columns = ", ".join(columns)

        new_query = LazyQuery(self.conn, self.base_sql, self.lance_datasets)
        new_query.operations = self.operations + [("GROUP BY", columns)]
        return new_query

    def order_by(self, columns: str | list[str]) -> "LazyQuery":
        """
        Add an ORDER BY clause to the query.

        Args:
            columns: Column names to order by (string or list)

        Returns:
            New LazyQuery with the order by applied
        """
        if isinstance(columns, list):
            columns = ", ".join(columns)

        new_query = LazyQuery(self.conn, self.base_sql, self.lance_datasets)
        new_query.operations = self.operations + [("ORDER BY", columns)]
        return new_query

    def limit(self, n: int) -> "LazyQuery":
        """
        Add a LIMIT clause to the query.

        Args:
            n: Number of rows to limit

        Returns:
            New LazyQuery with the limit applied
        """
        new_query = LazyQuery(self.conn, self.base_sql, self.lance_datasets)
        new_query.operations = self.operations + [("LIMIT", str(n))]
        return new_query

    def _build_sql(self) -> str:
        """
        Build the final SQL query from base SQL and operations.

        Returns:
            Composed SQL query string
        """
        sql = self.base_sql

        for op_type, op_value in self.operations:
            if op_type == "SELECT":
                # Replace the SELECT clause
                if "SELECT" in sql.upper():
                    # Find the SELECT clause and replace it
                    select_start = sql.upper().find("SELECT")
                    from_start = sql.upper().find("FROM", select_start)
                    if from_start != -1:
                        sql = f"SELECT {op_value} {sql[from_start:]}"
                    else:
                        sql = f"SELECT {op_value} FROM ({sql})"
                else:
                    sql = f"SELECT {op_value} FROM ({sql})"
            elif op_type == "WHERE":
                if "WHERE" in sql.upper():
                    sql += f" AND {op_value}"
                else:
                    sql += f" WHERE {op_value}"
            elif op_type == "GROUP BY":
                sql += f" GROUP BY {op_value}"
            elif op_type == "ORDER BY":
                sql += f" ORDER BY {op_value}"
            elif op_type == "LIMIT":
                sql += f" LIMIT {op_value}"

        return sql

    def compute(self) -> pd.DataFrame:
        """
        Execute the composed query and return results.

        Returns:
            DataFrame containing the query results
        """
        final_sql = self._build_sql()

        # Reference Lance datasets in local scope so DuckDB can find them
        # Use the same approach as query() method
        if self.lance_datasets:
            # Create local variables that DuckDB can find
            locals_dict = {}
            for name, dataset in self.lance_datasets.items():
                locals_dict[name] = dataset

            # Execute in a context where these variables are available
            def execute_with_datasets():
                # Reference the datasets in local scope
                for name, dataset in locals_dict.items():
                    if name == "expression":
                        expression = dataset  # noqa: F841
                    elif name == "cells":
                        cells = dataset  # noqa: F841
                    elif name == "genes":
                        genes = dataset  # noqa: F841

                return self.conn.execute(final_sql).fetchdf()

            return execute_with_datasets()

        return self.conn.execute(final_sql).fetchdf()

    def __repr__(self) -> str:
        """String representation of the LazyQuery."""
        sql = self._build_sql()
        return f"LazyQuery(sql='{sql[:50]}{'...' if len(sql) > 50 else ''}')"
