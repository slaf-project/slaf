import lance
import duckdb
import pandas as pd
from pathlib import Path
import json
from typing import List, Union
import numpy as np
from .query_optimizer import QueryOptimizer


class SLAFArray:
    """SLAF (Sparse Lazy Array Format) for efficient single-cell data storage and querying"""

    def __init__(self, slaf_path: Union[str, Path]):
        """
        Initialize SLAF array from path

        Args:
            slaf_path: Path to SLAF directory containing config.json and .lance files
        """
        self.slaf_path = Path(slaf_path)

        # Load configuration
        config_path = self.slaf_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"SLAF config not found at {config_path}")

        with open(config_path, "r") as f:
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
        """Execute SQL query on the SLAF dataset"""
        # Reference Lance datasets in local scope so DuckDB can find them
        expression = self.expression  # noqa: F841
        cells = self.cells  # noqa: F841
        genes = self.genes  # noqa: F841

        # Use global duckdb to query Lance datasets directly
        duckdb.query("SET enable_progress_bar = true;")
        return duckdb.query(sql).fetchdf()

    def filter_cells(self, **filters) -> pd.DataFrame:
        """
        Filter cells based on metadata columns

        Args:
            **filters: Column name and filter value pairs

        Returns:
            DataFrame of filtered cells
        """
        if not filters:
            return self.obs.copy()

        # Build filter conditions
        conditions = []
        for column, value in filters.items():
            if column not in self.obs.columns:
                raise ValueError(f"Column '{column}' not found in cell metadata")

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
        sql = f"SELECT * FROM cells WHERE {where_clause}"

        return self.query(sql)

    def filter_genes(self, **filters) -> pd.DataFrame:
        """
        Filter genes based on metadata columns

        Args:
            **filters: Column name and filter value pairs

        Returns:
            DataFrame of filtered genes
        """
        if not filters:
            return self.var.copy()

        # Build filter conditions
        conditions = []
        for column, value in filters.items():
            if column not in self.var.columns:
                raise ValueError(f"Column '{column}' not found in gene metadata")

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
        sql = f"SELECT * FROM genes WHERE {where_clause}"

        return self.query(sql)

    def _normalize_entity_ids(
        self, entity_ids: Union[str, List[str]], entity_type: str
    ) -> List[int]:
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

    def get_cell_expression(self, cell_ids: Union[str, List[str]]) -> pd.DataFrame:
        """Get expression data for specific cells using optimized query strategies"""
        # Convert to integer IDs
        integer_ids = self._normalize_entity_ids(cell_ids, "cell")

        # Use optimized query strategy
        sql = QueryOptimizer.build_optimized_query(integer_ids, "cell")

        return self.query(sql)

    def get_gene_expression(self, gene_ids: Union[str, List[str]]) -> pd.DataFrame:
        """Get expression data for specific genes using optimized query strategies"""
        # Convert to integer IDs
        integer_ids = self._normalize_entity_ids(gene_ids, "gene")

        # Use optimized query strategy
        sql = QueryOptimizer.build_optimized_query(integer_ids, "gene")

        return self.query(sql)

    def get_submatrix(self, cell_selector=None, gene_selector=None) -> pd.DataFrame:
        """
        Get expression data using cell/gene selectors that work with obs/var DataFrames

        Args:
            cell_selector: Cell selector (slice, list, boolean mask, or None for all)
            gene_selector: Gene selector (slice, list, boolean mask, or None for all)

        Returns:
            DataFrame with expression data
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

        # Only query expression count since that's not in memory
        expression_count = self.query("SELECT COUNT(*) as count FROM expression")
        print(f"    Expression records: {expression_count.iloc[0]['count']:,}")

        # Optimization info
        optimizations = self.config.get("optimizations", {})
        if optimizations:
            print("  Optimizations:")
            for opt, value in optimizations.items():
                print(f"    {opt}: {value}")
