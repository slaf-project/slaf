import json
import threading
from pathlib import Path
from typing import Any

import duckdb
import lance
import numpy as np
import polars as pl
from loguru import logger


def display_ascii_art():
    """Display SLAF ASCII art logo"""
    ascii_art = """
  z Z
 ( - . - )
 /  ^ \\ ⚡
(  (_)  (_) )
 \\_______/ """
    logger.info(ascii_art)


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

        # Validate dataset exists
        if not self.slaf_path.exists():
            raise FileNotFoundError(f"SLAF dataset not found: {self.slaf_path}")

        # Load configuration
        config_path = self.slaf_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"SLAF config not found at {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize shape
        self.shape = tuple(self.config["array_shape"])

        # Setup datasets
        self._setup_datasets()

        # Initialize metadata loading
        self._obs = None
        self._var = None
        self._metadata_loaded = False
        self._metadata_loading = False
        self._metadata_thread = None
        self._metadata_error = None

        # Initialize row mapper for cell-based queries
        from slaf.core.query_optimizer import RowIndexMapper

        self.row_mapper = RowIndexMapper(self)

        # Start async metadata loading in background
        self._start_async_metadata_loading()

        # Display helpful initialization message
        self._display_initialization_message()

    def _display_initialization_message(self):
        """Display helpful initialization message with basic metadata"""
        # Display ASCII art
        display_ascii_art()
        logger.info("")

        n_cells, n_genes = self.shape

        # Format large numbers with commas
        cells_str = f"{n_cells:,}" if n_cells >= 1000 else str(n_cells)
        genes_str = f"{n_genes:,}" if n_genes >= 1000 else str(n_genes)

        # Get format version
        format_version = self.config.get("format_version", "unknown")

        # Get dataset name from path
        dataset_name = self.slaf_path.name

        # Display message
        logger.info(f"📊 SLAF Dataset Loaded: {dataset_name}")
        logger.info(f"   • Shape: {cells_str} cells × {genes_str} genes")
        logger.info(f"   • Format: SLAF v{format_version}")

        # Add metadata info if available
        if "metadata" in self.config:
            metadata = self.config["metadata"]
            if "expression_count" in metadata:
                expr_count = metadata["expression_count"]
                expr_str = f"{expr_count:,}" if expr_count >= 1000 else str(expr_count)
                logger.info(f"   • Expression records: {expr_str}")

            if "sparsity" in metadata:
                sparsity = metadata["sparsity"]
                logger.info(f"   • Sparsity: {sparsity:.1%}")

        # Add optimization info if available
        optimizations = self.config.get("optimizations", {})
        if optimizations:
            opt_info = ", ".join([f"{k}: {v}" for k, v in optimizations.items()])
            logger.info(f"   • Optimizations: {opt_info}")

        # Status message
        logger.info("   • Status: Ready for queries (metadata loading in background)")
        logger.info("")

    def _start_async_metadata_loading(self):
        """Start async metadata loading in background thread"""
        if self._metadata_thread and self._metadata_thread.is_alive():
            return  # Already loading

        self._metadata_loading = True
        self._metadata_thread = threading.Thread(
            target=self._load_metadata_async, daemon=True
        )
        self._metadata_thread.start()

    def _load_metadata_async(self):
        """Load metadata in background thread"""
        try:
            self._load_metadata()
            self._metadata_loaded = True
        except Exception as e:
            self._metadata_error = e
            logger.error(f"Error loading metadata: {e}")
        finally:
            self._metadata_loading = False

    def _setup_datasets(self):
        """Setup Lance datasets for the new table structure"""
        # Load all Lance datasets
        self.expression = lance.dataset(
            str(self.slaf_path / self.config["tables"]["expression"])
        )
        self.cells = lance.dataset(str(self.slaf_path / self.config["tables"]["cells"]))
        self.genes = lance.dataset(str(self.slaf_path / self.config["tables"]["genes"]))

    def _ensure_metadata_loaded(self):
        """Ensure metadata is loaded (lazy loading with async support)"""
        if self._metadata_loaded:
            return

        if self._metadata_loading:
            # Wait for async loading to complete
            if self._metadata_thread and self._metadata_thread.is_alive():
                logger.info(
                    "Loading metadata in background... (this may take a few seconds)"
                )
                self._metadata_thread.join()

        if not self._metadata_loaded:
            # Fallback to synchronous loading if async failed
            self._load_metadata()
            self._metadata_loaded = True

    @property
    def obs(self):
        """Cell metadata (lazy loaded)"""
        self._ensure_metadata_loaded()
        return self._obs

    @property
    def var(self):
        """Gene metadata (lazy loaded)"""
        self._ensure_metadata_loaded()
        return self._var

    def is_metadata_ready(self) -> bool:
        """Check if metadata is ready for use"""
        return self._metadata_loaded

    def is_metadata_loading(self) -> bool:
        """Check if metadata is currently loading"""
        return self._metadata_loading

    def wait_for_metadata(self, timeout: float = None):
        """Wait for metadata to be loaded (with optional timeout)"""
        if self._metadata_loaded:
            return

        if self._metadata_loading and self._metadata_thread:
            if timeout:
                self._metadata_thread.join(timeout=timeout)
            else:
                self._metadata_thread.join()

        if not self._metadata_loaded:
            # Fallback to synchronous loading
            self._load_metadata()
            self._metadata_loaded = True

    def _load_metadata(self):
        """Load cell and gene metadata into polars DataFrames for fast operations"""
        # Load cell metadata using optimized Lance operations
        # Use more efficient loading strategy for large datasets
        # Load in chunks if dataset is very large
        if self.shape[0] > 1000000:  # 1M+ cells
            # Use scan() for large datasets to avoid loading everything at once
            cells_df = pl.from_arrow(self.cells.scanner().to_table())
        else:
            # Use direct to_table() for smaller datasets
            cells_df = pl.from_arrow(self.cells.to_table())

        self._obs = cells_df.sort("cell_integer_id")

        # Load gene metadata using optimized Lance operations
        # Genes are typically smaller, so use direct loading
        genes_df = pl.from_arrow(self.genes.to_table())
        self._var = genes_df.sort("gene_integer_id")

        # Cache column information for faster access
        self._obs_columns = list(self._obs.columns)
        self._var_columns = list(self._var.columns)

        # Build cell start indices for efficient row access (zero-copy Polars)
        if "n_genes" in self._obs.columns:
            # Use existing n_genes column
            cumsum = self._obs["n_genes"].cum_sum()
        elif "gene_count" in self._obs.columns:
            # Use existing gene_count column
            cumsum = self._obs["gene_count"].cum_sum()
        else:
            # Calculate n_genes per cell from expression data
            logger.info(
                "n_genes or gene_count column not found, calculating from expression data..."
            )

            # Query expression data to count genes per cell
            gene_counts_query = """
            SELECT cell_integer_id, COUNT(*) as n_genes
            FROM expression
            GROUP BY cell_integer_id
            ORDER BY cell_integer_id
            """
            gene_counts_df = self.query(gene_counts_query)

            # Convert to polars if needed and join with obs
            if hasattr(gene_counts_df, "to_pandas"):
                gene_counts_df = gene_counts_df.to_pandas()
            gene_counts_pl = pl.from_pandas(gene_counts_df)

            # Join with obs to get n_genes for each cell
            obs_with_counts = self._obs.join(
                gene_counts_pl.select(["cell_integer_id", "n_genes"]),
                on="cell_integer_id",
                how="left",
            )

            # Fill missing values with 0 (cells with no expression)
            obs_with_counts = obs_with_counts.with_columns(
                pl.col("n_genes").fill_null(0)
            )

            cumsum = obs_with_counts["n_genes"].cum_sum()

        self._cell_start_index = pl.concat([pl.Series([0]), cumsum])

        # Restore dtypes for obs using polars
        obs_dtypes = self.config.get("obs_dtypes", {})
        for col, dtype_info in obs_dtypes.items():
            if col in self._obs.columns:
                if dtype_info["dtype"] == "category":
                    # Convert to polars categorical
                    self._obs = self._obs.with_columns(
                        pl.col(col).cast(pl.Categorical).cast(pl.Utf8)
                    )
                else:
                    # Map pandas dtypes to polars dtypes
                    polars_dtype = self._map_pandas_to_polars_dtype(dtype_info["dtype"])
                    self._obs = self._obs.with_columns(pl.col(col).cast(polars_dtype))

        # Restore dtypes for var using polars
        var_dtypes = self.config.get("var_dtypes", {})
        for col, dtype_info in var_dtypes.items():
            if col in self._var.columns:
                if dtype_info["dtype"] == "category":
                    # Convert to polars categorical
                    self._var = self._var.with_columns(
                        pl.col(col).cast(pl.Categorical).cast(pl.Utf8)
                    )
                else:
                    # Map pandas dtypes to polars dtypes
                    polars_dtype = self._map_pandas_to_polars_dtype(dtype_info["dtype"])
                    self._var = self._var.with_columns(pl.col(col).cast(polars_dtype))

        # Infer categorical columns if not in config
        self._infer_categorical_columns()

    def _map_pandas_to_polars_dtype(self, pandas_dtype: str) -> type[pl.DataType]:
        """Map pandas dtype to polars dtype"""
        dtype_mapping = {
            "int64": pl.Int64,
            "int32": pl.Int32,
            "int16": pl.Int16,
            "int8": pl.Int8,
            "uint64": pl.UInt64,
            "uint32": pl.UInt32,
            "uint16": pl.UInt16,
            "uint8": pl.UInt8,
            "float64": pl.Float64,
            "float32": pl.Float32,
            "bool": pl.Boolean,
            "object": pl.Utf8,
            "string": pl.Utf8,
        }
        return dtype_mapping.get(pandas_dtype, pl.Utf8)

    def _infer_categorical_columns(self):
        """Infer categorical columns based on data characteristics using polars"""
        # For obs: infer categoricals for string columns with few unique values
        for col in self._obs.columns:
            if col not in self.config.get("obs_dtypes", {}):
                if self._obs[col].dtype == pl.Utf8:
                    unique_count = self._obs[col].n_unique()
                    total_count = len(self._obs)
                    unique_ratio = unique_count / total_count
                    # If less than 20% unique values, likely categorical
                    if unique_ratio < 0.2 and unique_count < 50:
                        self._obs = self._obs.with_columns(
                            pl.col(col).cast(pl.Categorical).cast(pl.Utf8)
                        )

        # For var: infer categoricals for string columns with few unique values
        for col in self._var.columns:
            if col not in self.config.get("var_dtypes", {}):
                if self._var[col].dtype == pl.Utf8:
                    unique_count = self._var[col].n_unique()
                    total_count = len(self._var)
                    unique_ratio = unique_count / total_count
                    # If less than 20% unique values, likely categorical
                    if unique_ratio < 0.2 and unique_count < 50:
                        self._var = self._var.with_columns(
                            pl.col(col).cast(pl.Categorical).cast(pl.Utf8)
                        )

    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute SQL query on the SLAF dataset.

        Executes SQL queries directly on the underlying Lance tables using DuckDB.
        The query can reference three tables: 'cells', 'genes', and 'expression'.
        This enables complex aggregations, joins, and filtering operations.

        Args:
            sql: SQL query string to execute. Can reference tables: cells, genes, expression.
                 Supports standard SQL operations including WHERE, GROUP BY, ORDER BY, etc.

        Returns:
            Polars DataFrame containing the query results.

        Raises:
            ValueError: If the SQL query is malformed or references non-existent tables.
            RuntimeError: If the query execution fails.

        Examples:
            >>> # Basic query to count cells
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> result = slaf_array.query("SELECT COUNT(*) as total_cells FROM cells")
            >>> print(f"Total cells: {result['total_cells'][0]}")
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
            shape: (3, 3)
            ┌────────────┬────────────┬────────────┐
            │ cell_type  ┆ cell_count ┆ avg_counts │
            │ ---        ┆ ---        ┆ ---        │
            │ str        ┆ i64        ┆ f64        │
            ╞════════════╪════════════╪════════════╡
            │ T cells    ┆ 250        ┆ 1250.5     │
            │ B cells    ┆ 200        ┆ 1100.2     │
            │ Monocytes  ┆ 150        ┆ 950.8      │
            └────────────┴────────────┴────────────┘

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
        # Create a new DuckDB connection for this query to avoid connection issues

        # Create a new connection
        con = duckdb.connect()

        # Reference Lance datasets in local scope so DuckDB can find them
        expression = self.expression  # noqa: F841
        cells = self.cells  # noqa: F841
        genes = self.genes  # noqa: F841

        # Execute the query
        result = con.execute(sql).fetchdf()

        # Close the connection
        con.close()

        # Convert pandas DataFrame to polars DataFrame
        return pl.from_pandas(result)

    def filter_cells(self, **filters: Any) -> pl.DataFrame:
        """
        Filter cells based on metadata columns.

        Provides a convenient interface for filtering cells using metadata columns.
        Supports exact matches, list values, and range queries with operators.
        Uses in-memory polars filtering when metadata is loaded, falls back to SQL otherwise.

        Args:
            **filters: Column name and filter value pairs. Supports:
                - Exact matches: cell_type="T cells"
                - List values: cell_type=["T cells", "B cells"]
                - Range queries: total_counts=">1000", total_counts="<=2000"
                - Multiple conditions: cell_type="T cells", total_counts=">500"

        Returns:
            Polars DataFrame containing filtered cell metadata.

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

    def filter_genes(self, **filters: Any) -> pl.DataFrame:
        """
        Filter genes based on metadata columns.

        Provides a convenient interface for filtering genes using metadata columns.
        Supports exact matches, list values, and range queries with operators.
        Uses in-memory polars filtering when metadata is loaded, falls back to SQL otherwise.

        Args:
            **filters: Column name and filter value pairs. Supports:
                - Exact matches: gene_type="protein_coding"
                - List values: gene_type=["protein_coding", "lncRNA"]
                - Range queries: expression_mean=">5.0", expression_mean="<=10.0"
                - Multiple conditions: gene_type="protein_coding", chromosome="chr1"

        Returns:
            Polars DataFrame containing filtered gene metadata.

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

    def _filter(self, table_name: str, **filters: Any) -> pl.DataFrame:
        """
        Generic filtering method that uses polars for metadata operations.

        Args:
            table_name: Either "cells" or "genes"
            **filters: Filter conditions

        Returns:
            Filtered polars DataFrame
        """
        if not filters:
            # Return all metadata if no filters
            if table_name == "cells":
                return (
                    self.obs
                    if self._metadata_loaded
                    else self.query("SELECT * FROM cells")
                )
            else:
                return (
                    self.var
                    if self._metadata_loaded
                    else self.query("SELECT * FROM genes")
                )

        # For filtering, we need metadata to be loaded to validate columns
        # If metadata is not loaded, load it first
        if not self._metadata_loaded:
            self._ensure_metadata_loaded()

        # Use polars filtering for metadata operations
        if table_name == "cells":
            return self._filter_with_polars(self.obs, **filters)
        elif table_name == "genes":
            return self._filter_with_polars(self.var, **filters)
        else:
            raise ValueError(f"Invalid table name: {table_name}")

    def _filter_with_polars(
        self, metadata_df: pl.DataFrame, **filters: Any
    ) -> pl.DataFrame:
        """
        Filter metadata using polars operations for high performance.

        Args:
            metadata_df: Polars DataFrame to filter (self.obs or self.var)
            **filters: Filter conditions

        Returns:
            Filtered polars DataFrame
        """
        # Start with all rows
        result = metadata_df

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

                    # Create boolean mask for the comparison using polars expressions
                    if operator == ">":
                        result = result.filter(pl.col(column) > numeric_filter_value)
                    elif operator == "<":
                        result = result.filter(pl.col(column) < numeric_filter_value)
                    elif operator == ">=":
                        result = result.filter(pl.col(column) >= numeric_filter_value)
                    elif operator == "<=":
                        result = result.filter(pl.col(column) <= numeric_filter_value)
                    else:
                        raise ValueError(f"Unsupported operator: {operator}")

                except Exception as err:
                    raise ValueError(
                        f"Cannot perform numeric comparison on non-numeric column '{column}'"
                    ) from err

            elif isinstance(value, list):
                # Handle list values using polars is_in
                result = result.filter(pl.col(column).is_in(value))
            else:
                # Handle exact matches
                result = result.filter(pl.col(column) == value)

        return result

    def _normalize_entity_ids(
        self, entity_ids: str | list[str], entity_type: str
    ) -> list[int]:
        """Convert string entity IDs to integer IDs using metadata mappings"""
        if entity_type == "cell":
            metadata = self.obs
            id_col = "cell_integer_id"
            index_col = "cell_id"
        else:
            metadata = self.var
            id_col = "gene_integer_id"
            index_col = "gene_id"

        # Convert to list if single ID
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]

        # Map string IDs to integer IDs using polars
        # Get the index values and corresponding integer IDs
        index_values = metadata[index_col].to_list()
        integer_id_values = metadata[id_col].to_list()

        # Create a mapping from index to integer ID
        id_mapping = dict(zip(index_values, integer_id_values, strict=False))

        # Map string IDs to integer IDs
        integer_ids = []
        for entity_id in entity_ids:
            if entity_id in id_mapping:
                integer_ids.append(id_mapping[entity_id])
            else:
                raise ValueError(f"{entity_type} ID '{entity_id}' not found")

        return integer_ids

    def get_cell_expression(self, cell_ids: str | list[str]) -> pl.DataFrame:
        """
        Get expression data for specific cells using Lance take() and Polars.

        Retrieves expression data for specified cells using efficient Lance row access
        and Polars for in-memory operations. This method provides significant
        performance improvements over SQL-based queries.

        Args:
            cell_ids: Single cell ID (string) or list of cell IDs to retrieve.
                     Can be string identifiers or integer IDs.

        Returns:
            Polars DataFrame containing expression data for the specified cells.
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

        if not integer_ids:
            return pl.DataFrame({"cell_id": [], "gene_id": [], "value": []})

        # Get row indices using RowIndexMapper
        row_indices = self.row_mapper.get_cell_row_ranges(integer_ids)

        # Load data with Lance take()
        expression_data = self.expression.take(row_indices)

        # Convert PyArrow Table to Polars DataFrame and join with metadata
        expression_df = pl.from_arrow(expression_data)
        return self._join_with_metadata(expression_df)

    def get_gene_expression(self, gene_ids: str | list[str]) -> pl.DataFrame:
        """
        Get gene expression data for specified genes.

        This method uses QueryOptimizer to generate optimal SQL queries based on the
        gene ID distribution (consecutive ranges use BETWEEN, scattered values use IN).

        Args:
            gene_ids: Gene ID(s) to query. Can be a single gene ID string or list of gene IDs.

        Returns:
            Polars DataFrame containing expression data for the specified genes.
            Columns include cell_id, gene_id, and expression values.

        Raises:
            ValueError: If any gene ID is not found in the dataset.
            RuntimeError: If the query execution fails.

        Examples:
            >>> # Get expression for a single gene
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> expression = slaf_array.get_gene_expression("GENE1")
        """

        # Convert to list if single gene ID
        if isinstance(gene_ids, str):
            gene_ids = [gene_ids]

        # Convert string gene IDs to integer IDs
        integer_ids = self._normalize_entity_ids(gene_ids, "gene")

        if not integer_ids:
            return pl.DataFrame({"cell_id": [], "gene_id": [], "value": []})

        # Use QueryOptimizer for optimal performance
        from slaf.core.query_optimizer import QueryOptimizer

        # Build optimized SQL query using QueryOptimizer
        sql_query = QueryOptimizer.build_optimized_query(
            entity_ids=integer_ids,
            entity_type="gene",
            use_adaptive_batching=True,
            max_batch_size=100,
        )

        # Fix the table name for Polars scan (use 'self' instead of 'expression')
        sql_query = sql_query.replace("expression", "self")

        # Execute the optimized query
        ldf = pl.scan_pyarrow_dataset(self.expression)
        expression_df = ldf.sql(sql_query).collect()

        # Join with metadata using Polars
        return self._join_with_metadata(expression_df)

    def get_submatrix(
        self, cell_selector: Any | None = None, gene_selector: Any | None = None
    ) -> pl.DataFrame:
        """
        Get expression data using cell/gene selectors with Lance take() and Polars.

        Retrieves a subset of expression data based on cell and gene selectors.
        The selectors can be slices, lists, boolean masks, or None for all cells/genes.
        This method provides a flexible interface for subsetting expression data with
        significant performance improvements over SQL-based queries.

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
            Polars DataFrame containing expression data for the selected subset.
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
        # Get row indices for cells using RowIndexMapper
        cell_indices = self.row_mapper.get_cell_row_ranges_by_selector(cell_selector)

        # Load data with Lance take()
        expression_data = self.expression.take(cell_indices)
        expression_df = pl.from_arrow(expression_data)

        # Filter by genes if specified
        if gene_selector is not None:
            # Convert gene selector to integer IDs
            if isinstance(gene_selector, int):
                if gene_selector < 0:
                    gene_selector = self.shape[1] + gene_selector
                if 0 <= gene_selector < self.shape[1]:
                    gene_integer_id = self.var["gene_integer_id"][gene_selector]
                    expression_df = expression_df.filter(
                        pl.col("gene_integer_id") == gene_integer_id  # type: ignore[arg-type]
                    )
                else:
                    raise ValueError(f"Gene index {gene_selector} out of bounds")
            elif isinstance(gene_selector, slice):
                start = gene_selector.start or 0
                stop = gene_selector.stop or self.shape[1]
                step = gene_selector.step or 1

                # Handle negative indices
                if start < 0:
                    start = self.shape[1] + start
                if stop < 0:
                    stop = self.shape[1] + stop

                # Clamp bounds
                start = max(0, min(start, self.shape[1]))
                stop = max(0, min(stop, self.shape[1]))

                gene_integer_ids = self.var["gene_integer_id"][
                    start:stop:step
                ].to_list()
                expression_df = expression_df.filter(
                    pl.col("gene_integer_id").is_in(gene_integer_ids)  # type: ignore[arg-type]
                )
            elif isinstance(gene_selector, list):
                gene_integer_ids = []
                for idx in gene_selector:
                    if isinstance(idx, int):
                        if idx < 0:
                            idx = self.shape[1] + idx
                        if 0 <= idx < self.shape[1]:
                            gene_integer_id = self.var["gene_integer_id"][idx]
                            gene_integer_ids.append(gene_integer_id)
                        else:
                            raise ValueError(f"Gene index {idx} out of bounds")
                    else:
                        raise ValueError(f"Invalid gene index type: {type(idx)}")
                expression_df = expression_df.filter(
                    pl.col("gene_integer_id").is_in(gene_integer_ids)  # type: ignore[arg-type]
                )
            elif isinstance(gene_selector, np.ndarray) and gene_selector.dtype == bool:
                if len(gene_selector) != self.shape[1]:
                    raise ValueError(
                        f"Boolean mask length {len(gene_selector)} doesn't match gene count {self.shape[1]}"
                    )
                gene_integer_ids = self.var["gene_integer_id"][gene_selector].to_list()
                expression_df = expression_df.filter(
                    pl.col("gene_integer_id").is_in(gene_integer_ids)  # type: ignore[arg-type]
                )
            else:
                raise ValueError(
                    f"Unsupported gene selector type: {type(gene_selector)}"
                )

        # Join with metadata using Polars
        return self._join_with_metadata(expression_df)

    def _join_with_metadata(
        self, expression_data: pl.DataFrame | pl.Series
    ) -> pl.DataFrame:
        """
        Join expression data with cell/gene metadata using Polars.

        Args:
            expression_data: Polars DataFrame or Series containing expression data

        Returns:
            Polars DataFrame with cell_id, gene_id, and value columns
        """
        # Convert Series to DataFrame if needed
        if isinstance(expression_data, pl.Series):
            expression_data = expression_data.to_frame()

        # Join with cell metadata
        result = expression_data.join(
            self.obs.select(["cell_integer_id", "cell_id"]),
            on="cell_integer_id",
            how="left",
        )

        # Join with gene metadata
        result = result.join(
            self.var.select(["gene_integer_id", "gene_id"]),
            on="gene_integer_id",
            how="left",
        )

        # Select final columns
        return result.select(["cell_id", "gene_id", "value"])

    def info(self):
        """Print information about the SLAF dataset"""
        # Build output string for both logging and printing
        output_lines = []

        output_lines.append("SLAF Dataset")
        output_lines.append(f"  Shape: {self.shape[0]} cells × {self.shape[1]} genes")
        output_lines.append(
            f"  Format version: {self.config.get('format_version', 'unknown')}"
        )

        # Cell metadata columns
        cell_cols = self.obs.columns
        output_lines.append(f"  Cell metadata columns: {len(cell_cols)}")
        if cell_cols:
            output_lines.append(
                f"    {', '.join(cell_cols[:5])}{'...' if len(cell_cols) > 5 else ''}"
            )

        # Gene metadata columns
        gene_cols = self.var.columns
        output_lines.append(f"  Gene metadata columns: {len(gene_cols)}")
        if gene_cols:
            output_lines.append(
                f"    {', '.join(gene_cols[:5])}{'...' if len(gene_cols) > 5 else ''}"
            )

        # Record counts
        output_lines.append("  Record counts:")
        output_lines.append(f"    Cells: {len(self.obs):,}")
        output_lines.append(f"    Genes: {len(self.var):,}")

        # Expression metadata - use pre-computed if available, otherwise query
        format_version = self.config.get("format_version", "0.1")
        if format_version >= "0.2" and "metadata" in self.config:
            # Use pre-computed metadata from config
            metadata = self.config["metadata"]
            expression_count = metadata["expression_count"]
            sparsity = metadata["sparsity"]
            density = metadata["density"]

            output_lines.append(f"    Expression records: {expression_count:,}")
            output_lines.append(f"    Sparsity: {sparsity:.1%}")
            output_lines.append(f"    Density: {density:.1%}")

            # Show expression statistics if available
            if "expression_stats" in metadata:
                stats = metadata["expression_stats"]
                output_lines.append("  Expression statistics:")
                output_lines.append(f"    Min value: {stats['min_value']:.3f}")
                output_lines.append(f"    Max value: {stats['max_value']:.3f}")
                output_lines.append(f"    Mean value: {stats['mean_value']:.3f}")
                output_lines.append(f"    Std value: {stats['std_value']:.3f}")
        else:
            # Backward compatibility: query expression count for older format versions
            output_lines.append("    Expression records: computing...")
            expression_count = self.query("SELECT COUNT(*) as count FROM expression")
            output_lines.append(
                f"    Expression records: {expression_count.item(0, 0):,}"
            )

        # Optimization info
        optimizations = self.config.get("optimizations", {})
        if optimizations:
            output_lines.append("  Optimizations:")
            for opt, value in optimizations.items():
                output_lines.append(f"    {opt}: {value}")

        # Print to stdout for backward compatibility
        output = "\n".join(output_lines)
        print(output)

        # Also log for consistency with other methods
        logger.info("SLAF Dataset info displayed")
