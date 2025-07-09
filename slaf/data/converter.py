import json
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import scanpy as sc


class SLAFConverter:
    """
    Convert single-cell data formats to SLAF format with optimized storage.

    SLAFConverter provides efficient conversion from various single-cell data formats
    (primarily AnnData/h5ad) to the SLAF format. It optimizes storage by using
    integer keys, COO-style expression tables, and efficient metadata handling.

    Key Features:
        - AnnData/h5ad file conversion
        - Integer key optimization for memory efficiency
        - COO-style sparse matrix storage
        - Automatic metadata type inference
        - Lance format for high-performance storage

    Examples:
        >>> # Basic conversion from h5ad file
        >>> converter = SLAFConverter()
        >>> converter.convert("data.h5ad", "output.slaf")
        Converting data.h5ad to SLAF format...
        Optimizations: int_keys=True
        Loaded: 1000 cells × 20000 genes
        Conversion complete! Saved to output.slaf

        >>> # Conversion with custom optimization settings
        >>> converter = SLAFConverter(use_integer_keys=False)
        >>> converter.convert("data.h5ad", "output_string_keys.slaf")
        Converting data.h5ad to SLAF format...
        Optimizations: int_keys=False
        Loaded: 1000 cells × 20000 genes
        Conversion complete! Saved to output_string_keys.slaf

        >>> # Convert existing AnnData object
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad("data.h5ad")
        >>> converter = SLAFConverter()
        >>> converter.convert_anndata(adata, "output_from_object.slaf")
        Converting AnnData object to SLAF format...
        Optimizations: int_keys=True
        Loaded: 1000 cells × 20000 genes
        Conversion complete! Saved to output_from_object.slaf
    """

    def __init__(
        self,
        use_integer_keys: bool = True,
    ):
        """
        Initialize converter with optimization options.

        Args:
            use_integer_keys: Use integer keys instead of strings in sparse data.
                             This saves significant memory and improves query performance.
                             Set to False only if you need to preserve original string IDs.

        Examples:
            >>> # Default optimization (recommended)
            >>> converter = SLAFConverter()
            >>> print(f"Using integer keys: {converter.use_integer_keys}")
            Using integer keys: True

            >>> # Disable integer key optimization
            >>> converter = SLAFConverter(use_integer_keys=False)
            >>> print(f"Using integer keys: {converter.use_integer_keys}")
            Using integer keys: False
        """
        self.use_integer_keys = use_integer_keys

    def convert(self, h5ad_path: str, output_path: str):
        """
        Convert h5ad file to SLAF format with optimized storage.

        Loads an h5ad file and converts it to the SLAF format with COO-style
        expression tables and optimized metadata storage. This is the primary
        conversion method for AnnData files.

        Args:
            h5ad_path: Path to the input h5ad file to convert.
            output_path: Path where the SLAF dataset will be saved.
                        Should be a directory path, not a file path.

        Raises:
            FileNotFoundError: If the h5ad file doesn't exist.
            ValueError: If the h5ad file is corrupted or invalid.
            RuntimeError: If the conversion process fails.

        Examples:
            >>> # Convert a single h5ad file
            >>> converter = SLAFConverter()
            >>> converter.convert("pbmc3k.h5ad", "pbmc3k.slaf")
            Converting pbmc3k.h5ad to SLAF format...
            Optimizations: int_keys=True
            Loaded: 2700 cells × 32738 genes
            Conversion complete! Saved to pbmc3k.slaf

            >>> # Convert with custom output path
            >>> converter.convert("data.h5ad", "/path/to/output/dataset.slaf")
            Converting data.h5ad to SLAF format...
            Optimizations: int_keys=True
            Loaded: 1000 cells × 20000 genes
            Conversion complete! Saved to /path/to/output/dataset.slaf

            >>> # Error handling for missing file
            >>> try:
            ...     converter.convert("nonexistent.h5ad", "output.slaf")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
            Error: [Errno 2] No such file or directory: 'nonexistent.h5ad'
        """
        print(f"Converting {h5ad_path} to SLAF format...")
        print(f"Optimizations: int_keys={self.use_integer_keys}")

        # Load h5ad
        adata = sc.read_h5ad(h5ad_path)
        print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

        # Convert the loaded AnnData object
        self._convert_anndata(adata, output_path)

    def convert_anndata(self, adata, output_path: str):
        """Convert AnnData object to SLAF format with COO-style expression table"""
        print("Converting AnnData object to SLAF format...")
        print(f"Optimizations: int_keys={self.use_integer_keys}")
        print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

        # Convert the AnnData object
        self._convert_anndata(adata, output_path)

    def _convert_anndata(self, adata, output_path: str):
        """Internal method to convert AnnData object to SLAF format"""
        # Create output directory
        output_path_obj = Path(output_path)
        output_path_obj.mkdir(exist_ok=True)

        # Create integer key mappings if needed
        cell_id_mapping = None
        gene_id_mapping = None

        if self.use_integer_keys:
            print("Creating integer key mappings...")
            cell_id_mapping = self._create_id_mapping(adata.obs.index, "cell")
            gene_id_mapping = self._create_id_mapping(adata.var.index, "gene")

        # Convert expression data to COO format
        print("Converting expression data to COO format...")
        expression_table = self._sparse_to_coo_table(
            sparse_matrix=adata.X,
            cell_ids=adata.obs.index,
            gene_ids=adata.var.index,
        )

        # Convert metadata
        print("Converting metadata...")
        cell_metadata_table = self._create_metadata_table(
            df=adata.obs, entity_id_col="cell_id", integer_mapping=cell_id_mapping
        )
        gene_metadata_table = self._create_metadata_table(
            df=adata.var, entity_id_col="gene_id", integer_mapping=gene_id_mapping
        )

        # Write all Lance tables
        print("Writing Lance tables...")
        table_configs = [
            ("expression", expression_table),
            ("cells", cell_metadata_table),
            ("genes", gene_metadata_table),
        ]

        self._write_lance_tables(output_path_obj, table_configs)

        # Save config
        self._save_config(output_path_obj, adata.shape)
        print(f"Conversion complete! Saved to {output_path}")

    def _create_id_mapping(self, entity_ids, entity_type: str) -> list[dict[str, Any]]:
        """Create mapping from original entity IDs to integer indices"""
        # Direct assignment using pandas operations
        df = pd.DataFrame()
        df[f"{entity_type}_id"] = pd.Series(entity_ids).astype(str)
        df["integer_id"] = range(len(entity_ids))
        return df.to_dict(orient="records")

    def _sparse_to_coo_table(
        self,
        sparse_matrix,
        cell_ids,
        gene_ids,
    ):
        """Convert scipy sparse matrix to COO format PyArrow table with integer IDs"""
        coo_matrix = sparse_matrix.tocoo()
        print(f"Processing {len(coo_matrix.data):,} non-zero elements...")

        # Create string ID arrays
        cell_id_array = np.array(cell_ids)[coo_matrix.row].astype(str)
        gene_id_array = np.array(gene_ids)[coo_matrix.col].astype(str)

        # Create integer ID arrays for efficient range queries
        cell_integer_id_array = coo_matrix.row.astype(np.int32)
        gene_integer_id_array = coo_matrix.col.astype(np.int32)

        # Expression values
        value_array = coo_matrix.data

        # Check for nulls in string arrays
        if bool(np.any(pd.isnull(cell_id_array))) or bool(
            np.any(pd.isnull(gene_id_array))
        ):
            raise ValueError("Null values found in cell_id or gene_id arrays!")

        table = pa.table(
            {
                "cell_id": pa.array(cell_id_array, type=pa.string()),
                "gene_id": pa.array(gene_id_array, type=pa.string()),
                "cell_integer_id": pa.array(cell_integer_id_array, type=pa.int32()),
                "gene_integer_id": pa.array(gene_integer_id_array, type=pa.int32()),
                "value": pa.array(value_array, type=pa.float32()),
            }
        )

        # Note: Removed debug print for production

        # Validate schema
        expected_types = {
            "cell_id": pa.string(),
            "gene_id": pa.string(),
            "cell_integer_id": pa.int32(),
            "gene_integer_id": pa.int32(),
            "value": pa.float32(),
        }

        for col, expected_type in expected_types.items():
            assert table.schema.field(col).type == expected_type, (
                f"{col} is not {expected_type} type!"
            )
            assert table.column(col).null_count == 0, f"Nulls found in {col} column!"

        return table

    def _create_metadata_table(
        self,
        df: pd.DataFrame,
        entity_id_col: str,
        integer_mapping: list[dict[str, Any]] | None = None,
    ) -> pa.Table:
        result_df = df.copy()
        # Assign entity ID column using index directly to avoid misalignment
        result_df[entity_id_col] = df.index.astype(str)
        if integer_mapping and self.use_integer_keys:
            integer_id_col = f"{entity_id_col.replace('_id', '')}_integer_id"
            result_df[integer_id_col] = range(len(df))
        result_df = result_df.where(pd.notnull(result_df), None)
        # Convert all categorical/object columns to string for Arrow compatibility
        for col in result_df.columns:
            if (
                isinstance(result_df[col].dtype, pd.CategoricalDtype)
                or result_df[col].dtype == object
            ):
                result_df[col] = result_df[col].astype(str)
        # Ensure all ID columns are string and non-null
        result_df[entity_id_col] = result_df[entity_id_col].astype(str)
        if bool(result_df[entity_id_col].isnull().any()):
            raise ValueError(f"Null values found in {entity_id_col} column!")

        # Reset index to avoid __index_level_0__ column in Arrow table
        result_df = result_df.reset_index(drop=True)

        table = pa.table(result_df)

        # Note: Removed debug print for production

        return table

    def _write_lance_tables(
        self, output_path: Path, table_configs: list[tuple[str, pa.Table]]
    ):
        """Write multiple Lance tables with consistent naming"""
        for table_name, table in table_configs:
            table_path = output_path / f"{table_name}.lance"
            lance.write_dataset(table, str(table_path))

        # Create indices after all tables are written
        self._create_indices(output_path)

    def _create_indices(self, output_path: Path):
        """Create optimal indices for SLAF tables with column existence checks"""
        print("Creating indices for optimal query performance...")

        # Define desired indices for each table
        table_indices = {
            "cells": [
                "cell_id",
                "cell_integer_id",
                "cell_type",
                "batch",
                "total_counts",
                "n_genes_by_counts",
            ],
            "genes": ["gene_id", "gene_integer_id", "highly_variable"],
            "expression": [
                "cell_id",
                "gene_id",
                "cell_integer_id",
                "gene_integer_id",
            ],
        }

        # Create indices for each table
        for table_name, desired_columns in table_indices.items():
            table_path = output_path / f"{table_name}.lance"
            if table_path.exists():
                dataset = lance.dataset(str(table_path))
                schema = dataset.schema

                for column in desired_columns:
                    if column in schema.names:
                        print(f"  Creating index on {table_name}.{column}")
                        dataset.create_scalar_index(column, "BTREE")

        print("Index creation complete!")

    def _save_config(self, output_path_obj: Path, shape: tuple):
        """Save SLAF configuration"""
        config = {
            "format_version": "0.1",
            "array_shape": list(shape),
            "n_cells": shape[0],
            "n_genes": shape[1],
            "tables": {
                "expression": "expression.lance",
                "cells": "cells.lance",
                "genes": "genes.lance",
            },
            "optimizations": {
                "use_integer_keys": self.use_integer_keys,
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }

        config_path = output_path_obj / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
