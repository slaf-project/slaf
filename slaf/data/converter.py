import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import scanpy as sc
from scipy import sparse

from .chunked_reader import create_chunked_reader
from .utils import detect_format


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
        chunked: bool = False,
        chunk_size: int = 1000,
        sort_metadata: bool = False,
    ):
        """
        Initialize converter with optimization options.

        Args:
            use_integer_keys: Use integer keys instead of strings in sparse data.
                             This saves significant memory and improves query performance.
                             Set to False only if you need to preserve original string IDs.
            chunked: Use chunked processing for memory efficiency (no scanpy dependency).
            chunk_size: Size of each chunk when chunked=True.

        Examples:
            >>> # Default optimization (recommended)
            >>> converter = SLAFConverter()
            >>> print(f"Using integer keys: {converter.use_integer_keys}")
            Using integer keys: True

            >>> # Chunked processing for large datasets
            >>> converter = SLAFConverter(chunked=True, chunk_size=1000)
            >>> print(f"Chunked processing: {converter.chunked}")
            Chunked processing: True

            >>> # Disable integer key optimization
            >>> converter = SLAFConverter(use_integer_keys=False)
            >>> print(f"Using integer keys: {converter.use_integer_keys}")
            Using integer keys: False
        """
        self.use_integer_keys = use_integer_keys
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.sort_metadata = sort_metadata

    def convert(self, input_path: str, output_path: str, input_format: str = "auto"):
        """
        Convert single-cell data to SLAF format with optimized storage.

        SLAFConverter provides efficient conversion from various single-cell data formats
        to the SLAF format. It optimizes storage by using integer keys, COO-style
        expression tables, and efficient metadata handling.

        Supported Input Formats:
            - **h5ad**: AnnData files (.h5ad) - the standard single-cell format
            - **10x MTX**: 10x Genomics MTX directories containing matrix.mtx,
              barcodes.tsv, and genes.tsv files
            - **10x H5**: 10x Genomics H5 files (.h5) - Cell Ranger output format

        The converter automatically detects the input format based on file extension
        and directory structure. For optimal performance, you can also specify the
        format explicitly.

        Args:
            input_path: Path to the input file or directory to convert.
                       - For h5ad: path to .h5ad file
                       - For MTX: path to directory containing matrix.mtx, barcodes.tsv, genes.tsv
                       - For H5: path to .h5 file
            output_path: Path where the SLAF dataset will be saved.
                        Should be a directory path, not a file path.
            input_format: Format of input data. Options:
                         - "auto" (default): Auto-detect format
                         - "h5ad": AnnData format
                         - "10x_mtx": 10x MTX directory format
                         - "10x_h5": 10x H5 file format

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            ValueError: If the input file is corrupted, invalid, or format cannot be detected.
            RuntimeError: If the conversion process fails.

        Examples:
            >>> # Auto-detect format (recommended)
            >>> converter = SLAFConverter()
            >>> converter.convert("data.h5ad", "output.slaf")
            Converting data.h5ad to SLAF format...
            Optimizations: int_keys=True
            Loaded: 1000 cells × 20000 genes
            Conversion complete! Saved to output.slaf

            >>> # Convert 10x MTX directory
            >>> converter.convert("filtered_feature_bc_matrix/", "output.slaf")
            Converting 10x MTX directory filtered_feature_bc_matrix/ to SLAF format...
            Loaded: 2700 cells × 32738 genes
            Conversion complete! Saved to output.slaf

            >>> # Convert 10x H5 file
            >>> converter.convert("data.h5", "output.slaf")
            Converting 10x H5 file data.h5 to SLAF format...
            Loaded: 2700 cells × 32738 genes
            Conversion complete! Saved to output.slaf

            >>> # Explicit format specification
            >>> converter.convert("data.h5", "output.slaf", input_format="10x_h5")
            Converting 10x H5 file data.h5 to SLAF format...
            Loaded: 2700 cells × 32738 genes
            Conversion complete! Saved to output.slaf

            >>> # Convert with chunked processing for large datasets
            >>> converter = SLAFConverter(chunked=True, chunk_size=1000)
            >>> converter.convert("large_data.h5ad", "output.slaf")
            Converting large_data.h5ad to SLAF format...
            Optimizations: int_keys=True, chunked=True
            Processing in chunks of 1000 cells...
            Conversion complete! Saved to output.slaf

            >>> # Error handling for unsupported format
            >>> try:
            ...     converter.convert("unknown_file.txt", "output.slaf")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Cannot detect format for: unknown_file.txt
        """
        if input_format == "auto":
            input_format = detect_format(input_path)

        if input_format == "h5ad":
            self._convert_h5ad(input_path, output_path)
        elif input_format == "10x_mtx":
            self._convert_10x_mtx(input_path, output_path)
        elif input_format == "10x_h5":
            self._convert_10x_h5(input_path, output_path)
        else:
            raise ValueError(f"Unsupported format: {input_format}")

    def convert_anndata(self, adata, output_path: str):
        """Convert AnnData object to SLAF format with COO-style expression table"""
        if self.chunked:
            raise ValueError(
                "convert_anndata() not supported in chunked mode. "
                "Use convert() with file path instead."
            )

        print("Converting AnnData object to SLAF format...")
        print(f"Optimizations: int_keys={self.use_integer_keys}")
        print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

        # Convert the AnnData object
        self._convert_anndata(adata, output_path)

    def _convert_h5ad(self, h5ad_path: str, output_path: str):
        """Convert h5ad file to SLAF format (existing logic)"""
        print(f"Converting {h5ad_path} to SLAF format...")
        print(
            f"Optimizations: int_keys={self.use_integer_keys}, chunked={self.chunked}, sort_metadata={self.sort_metadata}"
        )

        if self.chunked:
            self._convert_chunked(h5ad_path, output_path)
        else:
            # Load h5ad using scanpy
            adata = sc.read_h5ad(h5ad_path)
            print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

            # Convert the loaded AnnData object
            self._convert_anndata(adata, output_path)

    def _convert_10x_mtx(self, mtx_dir: str, output_path: str):
        """Convert 10x MTX directory to SLAF format"""
        print(f"Converting 10x MTX directory {mtx_dir} to SLAF format...")

        if self.chunked:
            # Use native chunked reader for 10x MTX
            print("Using native chunked reader for 10x MTX...")
            self._convert_chunked(mtx_dir, output_path)
        else:
            # Use scanpy to read MTX files
            try:
                adata = sc.read_10x_mtx(mtx_dir)
            except Exception as e:
                print(f"Error reading 10x MTX files: {e}")
                print(
                    "Please ensure the directory contains matrix.mtx and either genes.tsv or features.tsv files"
                )
                raise ValueError(
                    f"Failed to read 10x MTX format from {mtx_dir}: {e}"
                ) from e

            print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

            # Convert using existing AnnData conversion logic
            self._convert_anndata(adata, output_path)

    def _convert_10x_h5(self, h5_path: str, output_path: str):
        """Convert 10x H5 file to SLAF format"""
        print(f"Converting 10x H5 file {h5_path} to SLAF format...")

        if self.chunked:
            # Use native chunked reader for 10x H5
            print("Using native chunked reader for 10x H5...")
            self._convert_chunked(h5_path, output_path)
        else:
            # Try to read as 10x H5 first, fall back to regular h5ad
            try:
                adata = sc.read_10x_h5(h5_path, genome="X")
            except Exception:
                # Fall back to reading as regular h5ad
                print("Reading as regular h5ad file...")
                adata = sc.read_h5ad(h5_path)

            print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

            # Convert using existing AnnData conversion logic
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

        # Sort metadata to match chunked converter behavior
        obs_df = adata.obs.copy()
        var_df = adata.var.copy()

        if self.sort_metadata:
            sort_columns = []
            if "cell_type" in obs_df.columns:
                sort_columns.append("cell_type")
            if "batch" in obs_df.columns:
                sort_columns.append("batch")

            if sort_columns:
                obs_df = obs_df.sort_values(sort_columns).reset_index(drop=True)
            else:
                obs_df = obs_df.reset_index(drop=True)

            sort_columns = []
            if "highly_variable" in var_df.columns:
                sort_columns.append("highly_variable")
            if "means" in var_df.columns:
                sort_columns.append("means")

            if sort_columns:
                var_df = var_df.sort_values(sort_columns).reset_index(drop=True)
            else:
                var_df = var_df.reset_index(drop=True)

        cell_metadata_table = self._create_metadata_table(
            df=obs_df, entity_id_col="cell_id", integer_mapping=cell_id_mapping
        )
        gene_metadata_table = self._create_metadata_table(
            df=var_df, entity_id_col="gene_id", integer_mapping=gene_id_mapping
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

    def _convert_chunked(self, h5ad_path: str, output_path: str):
        """Convert h5ad file using chunked processing with sorted-by-construction approach"""
        print(f"Processing in chunks of {self.chunk_size} cells...")

        with create_chunked_reader(h5ad_path) as reader:
            print(f"Loaded: {reader.n_obs} cells × {reader.n_vars} genes")

            # Create output directory
            output_path_obj = Path(output_path)
            output_path_obj.mkdir(exist_ok=True)

            # Read and sort metadata
            obs_df, var_df = self._sort_metadata_globally(reader)

            # Write sorted metadata tables
            self._write_sorted_metadata(obs_df, var_df, output_path_obj)

            # Stream expression data with sorted-by-construction
            expression_iterator = (
                self._expression_chunk_iterator_sorted_by_construction(reader)
            )

            lance.write_dataset(
                expression_iterator,
                str(output_path_obj / "expression.lance"),
                schema=self._get_expression_schema(),
                mode="overwrite",
            )

            # Create optimized indices
            self._create_optimized_indices(output_path_obj)

            # Save config
            self._save_config(output_path_obj, (reader.n_obs, reader.n_vars))
            print(f"Conversion complete! Saved to {output_path}")

    def _sort_metadata_globally(self, reader):
        """Sort metadata by common query patterns for optimal performance"""
        obs_df = reader.get_obs_metadata()
        var_df = reader.get_var_metadata()

        if self.sort_metadata:
            # Sort cells by common query patterns
            sort_columns = []
            if "cell_type" in obs_df.columns:
                sort_columns.append("cell_type")
            if "batch" in obs_df.columns:
                sort_columns.append("batch")

            if sort_columns:
                # Preserve original cell IDs before sorting
                obs_df["cell_id"] = obs_df.index.astype(str)
                obs_df = obs_df.sort_values(sort_columns).reset_index(drop=True)
            else:
                obs_df["cell_id"] = obs_df.index.astype(str)
                obs_df = obs_df.reset_index(drop=True)

            # Sort genes by expression patterns
            sort_columns = []
            if "highly_variable" in var_df.columns:
                sort_columns.append("highly_variable")
            if "means" in var_df.columns:
                sort_columns.append("means")

            if sort_columns:
                var_df = var_df.sort_values(sort_columns).reset_index(drop=True)
            else:
                var_df = var_df.reset_index(drop=True)

        else:
            # Preserve original order and IDs
            obs_df["cell_id"] = obs_df.index.astype(str)
            var_df["gene_id"] = var_df.index.astype(str)

        return obs_df, var_df

    def _write_sorted_metadata(
        self, obs_df: pd.DataFrame, var_df: pd.DataFrame, output_path: Path
    ):
        """Write sorted metadata tables"""
        # Create integer mappings to match traditional converter behavior
        cell_id_mapping = self._create_id_mapping(obs_df.index, "cell")
        gene_id_mapping = self._create_id_mapping(var_df.index, "gene")

        # Create simplified metadata tables matching traditional converter
        cell_metadata_table = self._create_metadata_table(
            obs_df, "cell_id", integer_mapping=cell_id_mapping
        )
        gene_metadata_table = self._create_metadata_table(
            var_df, "gene_id", integer_mapping=gene_id_mapping
        )

        lance.write_dataset(
            cell_metadata_table, str(output_path / "cells.lance"), mode="overwrite"
        )
        lance.write_dataset(
            gene_metadata_table, str(output_path / "genes.lance"), mode="overwrite"
        )

    def _expression_chunk_iterator_sorted_by_construction(
        self, reader
    ) -> Iterator[pa.RecordBatch]:
        """Iterator that produces naturally sorted chunks by construction"""
        total_chunks = (reader.n_obs + self.chunk_size - 1) // self.chunk_size

        for chunk_idx, (chunk, obs_slice) in enumerate(
            reader.iter_chunks(chunk_size=self.chunk_size)
        ):
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({obs_slice})")

            # Convert chunk to COO format with sequential integer IDs
            chunk_table = self._chunk_to_coo_table_sorted_by_construction(
                chunk, obs_slice, reader, chunk_idx * self.chunk_size
            )

            yield from chunk_table.to_batches()

    def _chunk_to_coo_table_sorted_by_construction(
        self, chunk, obs_slice, reader, global_cell_offset: int
    ):
        """Convert chunk to COO format with sequential integer IDs for natural sorting"""
        if sparse.issparse(chunk):
            coo_chunk = chunk.tocoo()
        else:
            coo_chunk = sparse.coo_matrix(chunk)

        # Get cell and gene names for this chunk
        chunk_cell_names = reader.obs_names[obs_slice]
        chunk_gene_names = reader.var_names

        # Debug: Check dimensions
        print(f"Debug: chunk shape: {chunk.shape}")
        print(f"Debug: coo_chunk shape: {coo_chunk.shape}")
        print(f"Debug: chunk_cell_names length: {len(chunk_cell_names)}")
        print(f"Debug: chunk_gene_names length: {len(chunk_gene_names)}")
        print(
            f"Debug: coo_chunk.col max: {coo_chunk.col.max() if len(coo_chunk.col) > 0 else 'empty'}"
        )
        print(
            f"Debug: coo_chunk.row max: {coo_chunk.row.max() if len(coo_chunk.row) > 0 else 'empty'}"
        )

        # Validate indices before accessing
        if len(coo_chunk.col) > 0 and coo_chunk.col.max() >= len(chunk_gene_names):
            raise ValueError(
                f"Column index {coo_chunk.col.max()} is out of bounds for gene names array of length {len(chunk_gene_names)}"
            )
        if len(coo_chunk.row) > 0 and coo_chunk.row.max() >= len(chunk_cell_names):
            raise ValueError(
                f"Row index {coo_chunk.row.max()} is out of bounds for cell names array of length {len(chunk_cell_names)}"
            )

        # Create arrays
        cell_ids = chunk_cell_names[coo_chunk.row].astype(str)
        gene_ids = chunk_gene_names[coo_chunk.col].astype(str)

        # Integer IDs are sequential within each chunk
        # This ensures natural sorting by (cell_integer_id, gene_integer_id)
        cell_integer_ids = (global_cell_offset + coo_chunk.row).astype(np.int32)
        gene_integer_ids = coo_chunk.col.astype(np.int32)

        # Create table - naturally sorted by construction
        table = pa.table(
            {
                "cell_id": pa.array(cell_ids),
                "gene_id": pa.array(gene_ids),
                "cell_integer_id": pa.array(cell_integer_ids),
                "gene_integer_id": pa.array(gene_integer_ids),
                "value": pa.array(coo_chunk.data.astype(np.float32)),
            }
        )

        return table

    def _get_expression_schema(self):
        """Get the schema for expression table"""
        return pa.schema(
            [
                ("cell_id", pa.string()),
                ("gene_id", pa.string()),
                ("cell_integer_id", pa.int32()),
                ("gene_integer_id", pa.int32()),
                ("value", pa.float32()),
            ]
        )

    def _create_optimized_indices(self, output_path: Path):
        """Create indices optimized for common query patterns"""
        print("Creating optimized indices for query performance...")

        # Expression table indices
        expression_path = output_path / "expression.lance"
        if expression_path.exists():
            dataset = lance.dataset(str(expression_path))

            # Primary indices for range queries
            dataset.create_scalar_index("cell_integer_id", "BTREE")
            dataset.create_scalar_index("gene_integer_id", "BTREE")

            # Composite index for cell-gene lookups
            # Note: Lance may not support composite indices yet, so we'll create individual indices
            dataset.create_scalar_index("cell_integer_id", "BTREE")
            dataset.create_scalar_index("gene_integer_id", "BTREE")

            # String indices for exact matches
            dataset.create_scalar_index("cell_id", "BTREE")
            dataset.create_scalar_index("gene_id", "BTREE")

        # Metadata table indices
        for table_name in ["cells", "genes"]:
            table_path = output_path / f"{table_name}.lance"
            if table_path.exists():
                dataset = lance.dataset(str(table_path))

                # Integer ID indices for joins
                integer_id_col = f"{table_name[:-1]}_integer_id"
                if integer_id_col in dataset.schema.names:
                    dataset.create_scalar_index(integer_id_col, "BTREE")

                # Common metadata columns
                common_columns = {
                    "cells": [
                        "cell_type",
                        "batch",
                        "total_counts",
                        "n_genes_by_counts",
                    ],
                    "genes": ["highly_variable", "means", "dispersions"],
                }

                for col in common_columns.get(table_name, []):
                    if col in dataset.schema.names:
                        dataset.create_scalar_index(col, "BTREE")

        print("Index creation complete!")

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

    def _compute_expression_statistics(self, expression_dataset) -> dict:
        """Compute basic statistics from expression dataset using SQL"""
        # Use DuckDB to compute statistics directly from Lance dataset
        import duckdb

        # Reference the Lance dataset in local scope for DuckDB
        expression = expression_dataset  # noqa: F841

        # Compute statistics using SQL
        stats_query = """
        SELECT
            MIN(value) as min_value,
            MAX(value) as max_value,
            AVG(value) as mean_value,
            STDDEV(value) as std_value
        FROM expression
        """

        result = duckdb.query(stats_query).fetchdf()

        # Convert to dictionary
        stats = {
            "min_value": float(result.iloc[0]["min_value"]),
            "max_value": float(result.iloc[0]["max_value"]),
            "mean_value": float(result.iloc[0]["mean_value"]),
            "std_value": float(result.iloc[0]["std_value"]),
        }

        return stats

    def _save_config(self, output_path_obj: Path, shape: tuple):
        """Save SLAF configuration with computed metadata"""
        n_cells = int(shape[0])
        n_genes = int(shape[1])

        # Compute additional metadata for faster info() method
        print("Computing dataset statistics...")

        # Get expression count and compute sparsity using SQL
        import duckdb

        # Reference Lance datasets in local scope for DuckDB
        expression = lance.dataset(str(output_path_obj / "expression.lance"))  # noqa: F841

        # Get expression count using SQL
        count_query = "SELECT COUNT(*) as count FROM expression"
        expression_count_result = duckdb.query(count_query).fetchdf()
        expression_count = expression_count_result.iloc[0]["count"]

        total_possible_elements = n_cells * n_genes
        sparsity = 1 - (expression_count / total_possible_elements)

        # Compute basic statistics from expression data using SQL
        expression_stats = self._compute_expression_statistics(expression)

        config = {
            "format_version": "0.2",
            "array_shape": [n_cells, n_genes],
            "n_cells": n_cells,
            "n_genes": n_genes,
            "tables": {
                "expression": "expression.lance",
                "cells": "cells.lance",
                "genes": "genes.lance",
            },
            "optimizations": {
                "use_integer_keys": self.use_integer_keys,
            },
            "metadata": {
                "expression_count": int(expression_count),
                "sparsity": float(sparsity),
                "density": float(1 - sparsity),
                "total_possible_elements": int(total_possible_elements),
                "expression_stats": expression_stats,
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }

        config_path = output_path_obj / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
