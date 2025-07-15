import json
from pathlib import Path
from typing import Any

import duckdb
import lance
import pandas as pd

from .query_optimizer import QueryOptimizer


class SLAFArray:
    """
    High-performance single-cell data storage and querying format.

    SLAFArray provides SQL-native access to single-cell data with lazy evaluation.
    Data is stored in a relational format with three main tables: cells, genes, and expression.
    The class enables direct SQL queries, efficient filtering, and seamless integration
    with the single-cell analysis ecosystem.

    Key Features:
        - SQL-native querying with DuckDB integration
        - Lazy evaluation for memory efficiency
        - Direct access to cell and gene metadata
        - High-performance storage with Lance format
        - Scanpy/AnnData compatibility

    Examples:
        >>> # Load a SLAF dataset
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> print(f"Dataset shape: {slaf_array.shape}")
        Dataset shape: (1000, 20000)

        >>> # Access metadata
        >>> print(f"Cell metadata columns: {list(slaf_array.obs.columns)}")
        Cell metadata columns: ['cell_type', 'total_counts', 'batch']
        >>> print(f"Gene metadata columns: {list(slaf_array.var.columns)}")
        Gene metadata columns: ['gene_type', 'chromosome']

        >>> # Filter cells by metadata
        >>> t_cells = slaf_array.filter_cells(cell_type="T cells")
        >>> print(f"Found {len(t_cells)} T cells")
        Found 250 T cells

        >>> # Execute SQL query
        >>> results = slaf_array.query("
        ...     SELECT cell_type, AVG(total_counts) as avg_counts
        ...     FROM cells
        ...     GROUP BY cell_type
        ...     ORDER BY avg_counts DESC
        ... ")
        >>> print(results)
           cell_type  avg_counts
        0  T cells      1250.5
        1  B cells      1100.2
        2  Monocytes     950.8

        >>> # Get expression data
        >>> expression = slaf_array.get_cell_expression(["cell_001", "cell_002"])
        >>> print(f"Expression matrix shape: {expression.shape}")
        Expression matrix shape: (2, 20000)
    """

    def __init__(self, slaf_path: str | Path):
        """
        Initialize SLAF array from a SLAF dataset directory.

        Args:
            slaf_path: Path to SLAF directory containing config.json and .lance files.
                       The directory should contain the dataset configuration and Lance tables.

        Raises:
            FileNotFoundError: If the SLAF config file is not found at the specified path.
            ValueError: If the config file is invalid or missing required tables.
            KeyError: If required configuration keys are missing.

        Examples:
            >>> # Load from local directory
            >>> slaf_array = SLAFArray("./data/pbmc3k.slaf")
            >>> print(f"Loaded dataset: {slaf_array.shape}")
            Loaded dataset: (2700, 32738)

            >>> # Load from cloud storage
            >>> slaf_array = SLAFArray("s3://bucket/data.slaf")
            >>> print(f"Cloud dataset: {slaf_array.shape}")
            Cloud dataset: (5000, 25000)

            >>> # Error handling for missing directory
            >>> try:
            ...     slaf_array = SLAFArray("nonexistent/path")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
            Error: SLAF config not found at nonexistent/path/config.json
        """
        self.slaf_path = Path(slaf_path)

        # Load configuration
        config_path = self.slaf_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"SLAF config not found at {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize shape
        self.shape = tuple(self.config["array_shape"])

        # Setup Lance datasets and LanceDB connection
        self._setup_datasets()
        self._setup_lancedb()

        # Load metadata into memory for fast access
        self._load_metadata()

    def _setup_lancedb(self):
        """Setup LanceDB connection"""
        import lancedb

        self.lancedb_conn = lancedb.connect(str(self.slaf_path))
        self.expression_table = self.lancedb_conn.open_table("expression")
        self.cells_table = self.lancedb_conn.open_table("cells")
        self.genes_table = self.lancedb_conn.open_table("genes")

    def _setup_datasets(self):
        """Setup Lance datasets for the new table structure"""
        # Load all Lance datasets
        self.expression = lance.dataset(
            str(self.slaf_path / self.config["tables"]["expression"])
        )
        self.cells = lance.dataset(str(self.slaf_path / self.config["tables"]["cells"]))
        self.genes = lance.dataset(str(self.slaf_path / self.config["tables"]["genes"]))

    def _load_metadata(self):
        """Load cell and gene metadata into memory, restoring dtypes if available"""
        # Load cell metadata
        self.obs = (
            self.cells_table.search()
            .to_pandas()
            .sort_values("cell_integer_id")
            .set_index("cell_id")
        )

        # Load gene metadata
        self.var = (
            self.genes_table.search()
            .to_pandas()
            .sort_values("gene_integer_id")
            .set_index("gene_id")
        )

        # Restore dtypes for obs
        obs_dtypes = self.config.get("obs_dtypes", {})
        for col, dtype_info in obs_dtypes.items():
            if col in self.obs.columns:
                if dtype_info["dtype"] == "category":
                    self.obs[col] = pd.Categorical(
                        self.obs[col],
                        categories=dtype_info.get("categories"),
                        ordered=dtype_info.get("ordered", False),
                    )
                else:
                    self.obs[col] = self.obs[col].astype(dtype_info["dtype"])

        # Restore dtypes for var
        var_dtypes = self.config.get("var_dtypes", {})
        for col, dtype_info in var_dtypes.items():
            if col in self.var.columns:
                if dtype_info["dtype"] == "category":
                    self.var[col] = pd.Categorical(
                        self.var[col],
                        categories=dtype_info.get("categories"),
                        ordered=dtype_info.get("ordered", False),
                    )
                else:
                    self.var[col] = self.var[col].astype(dtype_info["dtype"])

        # Infer categorical columns if not in config
        self._infer_categorical_columns()

    def _infer_categorical_columns(self):
        """Infer categorical columns based on data characteristics"""
        # For obs: infer categoricals for object columns with few unique values
        for col in self.obs.columns:
            if col not in self.config.get("obs_dtypes", {}):
                if self.obs[col].dtype == "object":
                    unique_ratio = self.obs[col].nunique() / len(self.obs)
                    # If less than 20% unique values, likely categorical
                    if unique_ratio < 0.2 and self.obs[col].nunique() < 50:
                        self.obs[col] = pd.Categorical(self.obs[col])

        # For var: infer categoricals for object columns with few unique values
        for col in self.var.columns:
            if col not in self.config.get("var_dtypes", {}):
                if self.var[col].dtype == "object":
                    unique_ratio = self.var[col].nunique() / len(self.var)
                    # If less than 20% unique values, likely categorical
                    if unique_ratio < 0.2 and self.var[col].nunique() < 50:
                        self.var[col] = pd.Categorical(self.var[col])

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query on the SLAF dataset.

        Executes SQL queries directly on the underlying Lance tables using DuckDB.
        The query can reference three tables: 'cells', 'genes', and 'expression'.
        This enables complex aggregations, joins, and filtering operations.

        Args:
            sql: SQL query string to execute. Can reference tables: cells, genes, expression.
                 Supports standard SQL operations including WHERE, GROUP BY, ORDER BY, etc.

        Returns:
            DataFrame containing the query results.

        Raises:
            ValueError: If the SQL query is malformed or references non-existent tables.
            RuntimeError: If the query execution fails.

        Examples:
            >>> # Basic query to count cells
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> result = slaf_array.query("SELECT COUNT(*) as total_cells FROM cells")
            >>> print(f"Total cells: {result['total_cells'].iloc[0]}")
            Total cells: 1000

            >>> # Complex aggregation query
            >>> result = slaf_array.query("
            ...     SELECT cell_type,
            ...            COUNT(*) as cell_count,
            ...            AVG(total_counts) as avg_counts
            ...     FROM cells
            ...     WHERE total_counts > 500
            ...     GROUP BY cell_type
            ...     ORDER BY avg_counts DESC
            ... ")
            >>> print(result)
               cell_type  cell_count  avg_counts
            0  T cells         250      1250.5
            1  B cells         200      1100.2
            2  Monocytes       150       950.8

            >>> # Join query across tables
            >>> result = slaf_array.query("
            ...     SELECT c.cell_type, g.gene_type, AVG(e.value) as avg_expression
            ...     FROM cells c
            ...     JOIN expression e ON c.cell_integer_id = e.cell_integer_id
            ...     JOIN genes g ON e.gene_integer_id = g.gene_integer_id
            ...     WHERE c.cell_type = 'T cells'
            ...     GROUP BY c.cell_type, g.gene_type
            ... ")
            >>> print(f"Found {len(result)} expression patterns")
            Found 5 expression patterns
        """
        # Reference Lance datasets in local scope so DuckDB can find them
        expression = self.expression  # noqa: F841
        cells = self.cells  # noqa: F841
        genes = self.genes  # noqa: F841

        # Use global duckdb to query Lance datasets directly
        duckdb.query("SET enable_progress_bar = true;")
        return duckdb.query(sql).fetchdf()

    def filter_cells(self, **filters: Any) -> pd.DataFrame:
        """
        Filter cells based on metadata columns.

        Provides a convenient interface for filtering cells using metadata columns.
        Supports exact matches, list values, and range queries with operators.
        Uses in-memory pandas filtering when metadata is loaded, falls back to SQL otherwise.

        Args:
            **filters: Column name and filter value pairs. Supports:
                - Exact matches: cell_type="T cells"
                - List values: cell_type=["T cells", "B cells"]
                - Range queries: total_counts=">1000", total_counts="<=2000"
                - Multiple conditions: cell_type="T cells", total_counts=">500"

        Returns:
            DataFrame containing filtered cell metadata.

        Raises:
            ValueError: If a specified column is not found in cell metadata.
            TypeError: If filter values are of unsupported types.

        Examples:
            >>> # Filter by cell type
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> t_cells = slaf_array.filter_cells(cell_type="T cells")
            >>> print(f"Found {len(t_cells)} T cells")
            Found 250 T cells

            >>> # Filter by multiple criteria
            >>> high_quality_t_cells = slaf_array.filter_cells(
            ...     cell_type="T cells",
            ...     total_counts=">1000",
            ...     batch=["batch1", "batch2"]
            ... )
            >>> print(f"Found {len(high_quality_t_cells)} high-quality T cells")
            Found 180 high-quality T cells

            >>> # Range query
            >>> medium_counts = slaf_array.filter_cells(
            ...     total_counts=">=500",
            ...     total_counts="<=2000"
            ... )
            >>> print(f"Found {len(medium_counts)} cells with medium counts")
            Found 450 cells with medium counts

            >>> # Error handling for invalid column
            >>> try:
            ...     result = slaf_array.filter_cells(invalid_column="value")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Column 'invalid_column' not found in cell metadata
        """
        return self._filter("cells", **filters)

    def filter_genes(self, **filters: Any) -> pd.DataFrame:
        """
        Filter genes based on metadata columns.

        Provides a convenient interface for filtering genes using metadata columns.
        Supports exact matches, list values, and range queries with operators.
        Uses in-memory pandas filtering when metadata is loaded, falls back to SQL otherwise.

        Args:
            **filters: Column name and filter value pairs. Supports:
                - Exact matches: gene_type="protein_coding"
                - List values: gene_type=["protein_coding", "lncRNA"]
                - Range queries: expression_mean=">5.0", expression_mean="<=10.0"
                - Multiple conditions: gene_type="protein_coding", chromosome="chr1"

        Returns:
            DataFrame containing filtered gene metadata.

        Raises:
            ValueError: If a specified column is not found in gene metadata.
            TypeError: If filter values are of unsupported types.

        Examples:
            >>> # Filter by gene type
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> protein_coding = slaf_array.filter_genes(gene_type="protein_coding")
            >>> print(f"Found {len(protein_coding)} protein-coding genes")
            Found 15000 protein-coding genes

            >>> # Filter by multiple criteria
            >>> high_expr_proteins = slaf_array.filter_genes(
            ...     gene_type="protein_coding",
            ...     expression_mean=">5.0",
            ...     chromosome=["chr1", "chr2"]
            ... )
            >>> print(f"Found {len(high_expr_proteins)} high-expression protein genes")
            Found 2500 high-expression protein genes

            >>> # Range query for expression
            >>> medium_expr = slaf_array.filter_genes(
            ...     expression_mean=">=2.0",
            ...     expression_mean="<=8.0"
            ... )
            >>> print(f"Found {len(medium_expr)} genes with medium expression")
            Found 8000 genes with medium expression

            >>> # Error handling for invalid column
            >>> try:
            ...     result = slaf_array.filter_genes(invalid_column="value")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Column 'invalid_column' not found in gene metadata
        """
        return self._filter("genes", **filters)

    def _filter(self, table_name: str, **filters: Any) -> pd.DataFrame:
        """
        Generic filtering method that chooses between pandas and SQL based on metadata availability.

        Args:
            table_name: Either "cells" or "genes"
            **filters: Filter conditions

        Returns:
            Filtered DataFrame
        """
        if not filters:
            # Return all metadata if no filters
            if table_name == "cells":
                return (
                    self.obs.copy()
                    if self.obs is not None
                    else self.query("SELECT * FROM cells")
                )
            else:
                return (
                    self.var.copy()
                    if self.var is not None
                    else self.query("SELECT * FROM genes")
                )

        # Choose filtering method based on metadata availability
        if table_name == "cells" and self.obs is not None:
            return self._filter_with_pandas(self.obs, **filters)
        elif table_name == "genes" and self.var is not None:
            return self._filter_with_pandas(self.var, **filters)
        else:
            return self._filter_with_sql(table_name, **filters)

    def _filter_with_pandas(
        self, metadata_df: pd.DataFrame, **filters: Any
    ) -> pd.DataFrame:
        """
        Filter metadata using in-memory pandas operations.

        Args:
            metadata_df: DataFrame to filter (self.obs or self.var)
            **filters: Filter conditions

        Returns:
            Filtered DataFrame
        """
        # Start with all rows
        mask = pd.Series(True, index=metadata_df.index)

        for column, value in filters.items():
            if column not in metadata_df.columns:
                raise ValueError(f"Column '{column}' not found in metadata")

            if isinstance(value, str) and value.startswith((">", "<", ">=", "<=")):
                # Handle range queries
                operator = value[:2] if value.startswith((">=", "<=")) else value[0]
                filter_value = (
                    value[2:] if value.startswith((">=", "<=")) else value[1:]
                )

                # Convert to numeric for comparison
                try:
                    numeric_filter_value = float(filter_value)
                    col_data = pd.to_numeric(metadata_df[column], errors="coerce")

                    # Create boolean mask for the comparison
                    if operator == ">":
                        comparison_mask = col_data > numeric_filter_value  # type: ignore
                    elif operator == "<":
                        comparison_mask = col_data < numeric_filter_value  # type: ignore
                    elif operator == ">=":
                        comparison_mask = col_data >= numeric_filter_value  # type: ignore
                    elif operator == "<=":
                        comparison_mask = col_data <= numeric_filter_value  # type: ignore
                    else:
                        raise ValueError(f"Unsupported operator: {operator}")

                    # Apply the comparison mask, excluding NaN values
                    mask &= comparison_mask & col_data.notna()  # type: ignore
                except Exception as err:
                    raise ValueError(
                        f"Cannot perform numeric comparison on non-numeric column '{column}'"
                    ) from err

            elif isinstance(value, list):
                # Handle list values
                mask &= metadata_df[column].isin(value)
            else:
                # Handle exact matches
                mask &= metadata_df[column] == value

        result = metadata_df[mask].copy()
        return result  # type: ignore

    def _filter_with_sql(self, table_name: str, **filters: Any) -> pd.DataFrame:
        """
        Filter metadata using SQL queries against disk-based tables.

        Args:
            table_name: Table name ("cells" or "genes")
            **filters: Filter conditions

        Returns:
            Filtered DataFrame
        """
        if not filters:
            return self.query(f"SELECT * FROM {table_name}")

        # Build filter conditions
        conditions = []
        for column, value in filters.items():
            if isinstance(value, str) and value.startswith((">", "<", ">=", "<=")):
                # Handle range queries
                operator = value[:2] if value.startswith((">=", "<=")) else value[0]
                filter_value = (
                    value[2:] if value.startswith((">=", "<=")) else value[1:]
                )
                conditions.append(f"{column} {operator} {filter_value}")
            elif isinstance(value, list):
                # Handle list values
                value_list = "', '".join(map(str, value))
                conditions.append(f"{column} IN ('{value_list}')")
            else:
                # Handle exact matches
                conditions.append(f"{column} = '{value}'")

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM {table_name} WHERE {where_clause}"

        return self.query(sql)

    def _normalize_entity_ids(
        self, entity_ids: str | list[str], entity_type: str
    ) -> list[int]:
        """Convert string entity IDs to integer IDs using metadata mappings"""
        if entity_type == "cell":
            metadata = self.obs
            id_col = "cell_integer_id"
        else:
            metadata = self.var
            id_col = "gene_integer_id"

        # Convert to list if single ID
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]

        # Map string IDs to integer IDs
        integer_ids = []
        for entity_id in entity_ids:
            if entity_id in metadata.index:
                integer_ids.append(metadata.loc[entity_id][id_col])
            else:
                raise ValueError(f"{entity_type} ID '{entity_id}' not found")

        return integer_ids

    def get_cell_expression(self, cell_ids: str | list[str]) -> pd.DataFrame:
        """
        Get expression data for specific cells using optimized query strategies.

        Retrieves expression data for specified cells using optimized SQL queries.
        The method automatically converts string cell IDs to integer IDs and uses
        query optimization for efficient data retrieval.

        Args:
            cell_ids: Single cell ID (string) or list of cell IDs to retrieve.
                     Can be string identifiers or integer IDs.

        Returns:
            DataFrame containing expression data for the specified cells.
            Columns include cell_id, gene_id, and expression values.

        Raises:
            ValueError: If any cell ID is not found in the dataset.
            RuntimeError: If the query execution fails.

        Examples:
            >>> # Get expression for a single cell
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> expression = slaf_array.get_cell_expression("cell_001")
            >>> print(f"Expression data shape: {expression.shape}")
            Expression data shape: (15000, 3)

            >>> # Get expression for multiple cells
            >>> expression = slaf_array.get_cell_expression(["cell_001", "cell_002", "cell_003"])
            >>> print(f"Expression data shape: {expression.shape}")
            Expression data shape: (45000, 3)

            >>> # Error handling for invalid cell ID
            >>> try:
            ...     expression = slaf_array.get_cell_expression("invalid_cell")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: cell ID 'invalid_cell' not found
        """
        # Convert to integer IDs
        integer_ids = self._normalize_entity_ids(cell_ids, "cell")

        # Use optimized query strategy
        sql = QueryOptimizer.build_optimized_query(integer_ids, "cell")

        return self.query(sql)

    def get_gene_expression(self, gene_ids: str | list[str]) -> pd.DataFrame:
        """
        Get expression data for specific genes using optimized query strategies.

        Retrieves expression data for specified genes using optimized SQL queries.
        The method automatically converts string gene IDs to integer IDs and uses
        query optimization for efficient data retrieval.

        Args:
            gene_ids: Single gene ID (string) or list of gene IDs to retrieve.
                     Can be string identifiers or integer IDs.

        Returns:
            DataFrame containing expression data for the specified genes.
            Columns include cell_id, gene_id, and expression values.

        Raises:
            ValueError: If any gene ID is not found in the dataset.
            RuntimeError: If the query execution fails.

        Examples:
            >>> # Get expression for a single gene
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> expression = slaf_array.get_gene_expression("GENE1")
            >>> print(f"Expression data shape: {expression.shape}")
            Expression data shape: (800, 3)

            >>> # Get expression for multiple genes
            >>> expression = slaf_array.get_gene_expression(["GENE1", "GENE2", "GENE3"])
            >>> print(f"Expression data shape: {expression.shape}")
            Expression data shape: (2400, 3)

            >>> # Error handling for invalid gene ID
            >>> try:
            ...     expression = slaf_array.get_gene_expression("invalid_gene")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: gene ID 'invalid_gene' not found
        """
        # Convert to integer IDs
        integer_ids = self._normalize_entity_ids(gene_ids, "gene")

        # Use optimized query strategy
        sql = QueryOptimizer.build_optimized_query(integer_ids, "gene")

        return self.query(sql)

    def get_submatrix(
        self, cell_selector: Any | None = None, gene_selector: Any | None = None
    ) -> pd.DataFrame:
        """
        Get expression data using cell/gene selectors that work with obs/var DataFrames.

        Retrieves a subset of expression data based on cell and gene selectors.
        The selectors can be slices, lists, boolean masks, or None for all cells/genes.
        This method provides a flexible interface for subsetting expression data.

        Args:
            cell_selector: Cell selector for subsetting. Can be:
                - None: Include all cells
                - slice: e.g., slice(0, 100) for first 100 cells
                - list: e.g., [0, 5, 10] for specific cell indices
                - boolean mask: e.g., [True, False, True, ...] for boolean selection
            gene_selector: Gene selector for subsetting. Can be:
                - None: Include all genes
                - slice: e.g., slice(0, 5000) for first 5000 genes
                - list: e.g., [0, 100, 200] for specific gene indices
                - boolean mask: e.g., [True, False, True, ...] for boolean selection

        Returns:
            DataFrame containing expression data for the selected subset.
            Columns include cell_id, gene_id, and expression values.

        Raises:
            ValueError: If selectors are invalid or out of bounds.
            RuntimeError: If the query execution fails.

        Examples:
            >>> # Get first 100 cells and first 5000 genes
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> submatrix = slaf_array.get_submatrix(
            ...     cell_selector=slice(0, 100),
            ...     gene_selector=slice(0, 5000)
            ... )
            >>> print(f"Submatrix shape: {submatrix.shape}")
            Submatrix shape: (500000, 3)

            >>> # Get specific cells and genes
            >>> submatrix = slaf_array.get_submatrix(
            ...     cell_selector=[0, 5, 10, 15],
            ...     gene_selector=[100, 200, 300]
            ... )
            >>> print(f"Submatrix shape: {submatrix.shape}")
            Submatrix shape: (12, 3)

            >>> # Get all cells for specific genes
            >>> submatrix = slaf_array.get_submatrix(
            ...     gene_selector=[0, 100, 200, 300, 400]
            ... )
            >>> print(f"Submatrix shape: {submatrix.shape}")
            Submatrix shape: (5000, 3)

            >>> # Error handling for invalid selector
            >>> try:
            ...     submatrix = slaf_array.get_submatrix(
            ...         cell_selector=slice(0, 1000000)  # Out of bounds
            ...     )
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Cell selector out of bounds
        """
        # Use optimized query builder from QueryOptimizer
        sql = QueryOptimizer.build_submatrix_query(
            cell_selector=cell_selector,
            gene_selector=gene_selector,
            cell_count=self.shape[0],
            gene_count=self.shape[1],
        )

        return self.query(sql)

    def info(self):
        """Print information about the SLAF dataset"""
        print("SLAF Dataset")
        print(f"  Shape: {self.shape[0]} cells × {self.shape[1]} genes")
        print(f"  Format version: {self.config.get('format_version', 'unknown')}")

        # Cell metadata columns
        cell_cols = list(self.obs.columns)
        print(f"  Cell metadata columns: {len(cell_cols)}")
        if cell_cols:
            print(
                f"    {', '.join(cell_cols[:5])}{'...' if len(cell_cols) > 5 else ''}"
            )

        # Gene metadata columns
        gene_cols = list(self.var.columns)
        print(f"  Gene metadata columns: {len(gene_cols)}")
        if gene_cols:
            print(
                f"    {', '.join(gene_cols[:5])}{'...' if len(gene_cols) > 5 else ''}"
            )

        # Record counts
        print("  Record counts:")
        print(f"    Cells: {len(self.obs):,}")
        print(f"    Genes: {len(self.var):,}")

        # Expression metadata - use pre-computed if available, otherwise query
        format_version = self.config.get("format_version", "0.1")
        if format_version >= "0.2" and "metadata" in self.config:
            # Use pre-computed metadata from config
            metadata = self.config["metadata"]
            expression_count = metadata["expression_count"]
            sparsity = metadata["sparsity"]
            density = metadata["density"]

            print(f"    Expression records: {expression_count:,}")
            print(f"    Sparsity: {sparsity:.1%}")
            print(f"    Density: {density:.1%}")

            # Show expression statistics if available
            if "expression_stats" in metadata:
                stats = metadata["expression_stats"]
                print("  Expression statistics:")
                print(f"    Min value: {stats['min_value']:.3f}")
                print(f"    Max value: {stats['max_value']:.3f}")
                print(f"    Mean value: {stats['mean_value']:.3f}")
                print(f"    Std value: {stats['std_value']:.3f}")
        else:
            # Backward compatibility: query expression count for older format versions
            print("    Expression records: computing...")
            expression_count = self.query("SELECT COUNT(*) as count FROM expression")
            print(f"    Expression records: {expression_count.iloc[0]['count']:,}")

        # Optimization info
        optimizations = self.config.get("optimizations", {})
        if optimizations:
            print("  Optimizations:")
            for opt, value in optimizations.items():
                print(f"    {opt}: {value}")
