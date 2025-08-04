import json
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from scipy import sparse

# Optional imports for data conversion
try:
    import scanpy as sc

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logger.warning("Scanpy not available. Install with: pip install scanpy")

from .chunked_reader import create_chunked_reader
from .utils import detect_format


class SLAFConverter:
    """
    Convert single-cell data formats to SLAF format with optimized storage.

    SLAFConverter provides efficient conversion from various single-cell data formats
    (primarily AnnData/h5ad) to the SLAF format. It optimizes storage by using
    integer keys, COO-style expression tables, and efficient metadata handling.
    Chunked conversion is now the default for optimal memory efficiency.

    Key Features:
        - AnnData/h5ad file conversion
        - Integer key optimization for memory efficiency
        - COO-style sparse matrix storage
        - Automatic metadata type inference
        - Lance format for high-performance storage
        - Chunked processing by default for memory efficiency

    Examples:
        >>> # Basic conversion from h5ad file (chunked is now the default)
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

        >>> # Convert existing AnnData object (chunked is now the default)
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
        chunked: bool = True,  # Changed from False to True - make chunked the default
        chunk_size: int = 25000,  # Reduced from 50000 to prevent memory issues
        sort_metadata: bool = False,
        create_indices: bool = False,  # Disable indices by default for small datasets
        optimize_storage: bool = True,  # Only store integer IDs in expression table
        use_optimized_dtypes: bool = True,  # Use uint16/uint32 for better compression
        enable_v2_manifest: bool = True,  # Enable v2 manifest paths for better performance
        compact_after_write: bool = True,  # Compact dataset after writing for optimal storage
    ):
        """
        Initialize converter with optimization options.

        Args:
            use_integer_keys: Use integer keys instead of strings in sparse data.
                             This saves significant memory and improves query performance.
                             Set to False only if you need to preserve original string IDs.
            chunked: Use chunked processing for memory efficiency (default: True).
                    Chunked processing is now the default for optimal memory efficiency.
                    Set to False only for small datasets or debugging purposes.
            chunk_size: Size of each chunk when chunked=True (default: 25000).
            create_indices: Whether to create indices for query performance.
                          Default: False for small datasets to reduce storage overhead.
                          Set to True for large datasets where query performance is important.
            optimize_storage: Only store integer IDs in expression table to reduce storage size.
                           String IDs are available in metadata tables for mapping.
            use_optimized_dtypes: Use optimized data types (uint16/uint32) for better compression.
                                This can significantly reduce storage size for large datasets.
            enable_v2_manifest: Enable v2 manifest paths for better query performance.
                              This is recommended for large datasets.
            compact_after_write: Compact the dataset after writing to optimize storage.
                               This creates a new version but significantly reduces file size.

        Examples:
            >>> # Default optimization (recommended)
            >>> converter = SLAFConverter()
            >>> print(f"Using chunked processing: {converter.chunked}")
            Using chunked processing: True

            >>> # Non-chunked processing for small datasets
            >>> converter = SLAFConverter(chunked=False)
            >>> print(f"Using chunked processing: {converter.chunked}")
            Using chunked processing: False

            >>> # Custom chunk size for large datasets
            >>> converter = SLAFConverter(chunk_size=100000)
            >>> print(f"Chunk size: {converter.chunk_size}")
            Chunk size: 100000
        """
        self.use_integer_keys = use_integer_keys
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.sort_metadata = sort_metadata
        self.create_indices = create_indices
        self.optimize_storage = optimize_storage
        self.use_optimized_dtypes = use_optimized_dtypes
        self.enable_v2_manifest = enable_v2_manifest
        self.compact_after_write = compact_after_write

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
            if not SCANPY_AVAILABLE:
                raise ImportError(
                    "Scanpy is required for h5ad conversion. "
                    "Install with: pip install scanpy"
                )
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

        logger.info("Converting AnnData object to SLAF format...")
        logger.info(f"Optimizations: int_keys={self.use_integer_keys}")
        logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

        # Validate optimized data types
        if not self._validate_optimized_dtypes_anndata(adata):
            self.use_optimized_dtypes = False

        # Convert the AnnData object
        self._convert_anndata(adata, output_path)

    def _convert_h5ad(self, h5ad_path: str, output_path: str):
        """Convert h5ad file to SLAF format (existing logic)"""
        logger.info(f"Converting {h5ad_path} to SLAF format...")
        logger.info(
            f"Optimizations: int_keys={self.use_integer_keys}, chunked={self.chunked}, sort_metadata={self.sort_metadata}"
        )

        if self.chunked:
            self._convert_chunked(h5ad_path, output_path)
        else:
            # Load h5ad using scanpy
            adata = sc.read_h5ad(h5ad_path)
            logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

            # Convert the loaded AnnData object
            self._convert_anndata(adata, output_path)

    def _convert_10x_mtx(self, mtx_dir: str, output_path: str):
        """Convert 10x MTX directory to SLAF format"""
        logger.info(f"Converting 10x MTX directory {mtx_dir} to SLAF format...")

        if self.chunked:
            # Use native chunked reader for 10x MTX
            logger.info("Using native chunked reader for 10x MTX...")
            self._convert_chunked(mtx_dir, output_path)
        else:
            # Use scanpy to read MTX files
            try:
                adata = sc.read_10x_mtx(mtx_dir)
            except Exception as e:
                logger.error(f"Error reading 10x MTX files: {e}")
                logger.error(
                    "Please ensure the directory contains matrix.mtx and either genes.tsv or features.tsv files"
                )
                raise ValueError(
                    f"Failed to read 10x MTX format from {mtx_dir}: {e}"
                ) from e

            logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

            # Convert using existing AnnData conversion logic
            self._convert_anndata(adata, output_path)

    def _convert_10x_h5(self, h5_path: str, output_path: str):
        """Convert 10x H5 file to SLAF format"""
        logger.info(f"Converting 10x H5 file {h5_path} to SLAF format...")

        if self.chunked:
            # Use native chunked reader for 10x H5
            logger.info("Using native chunked reader for 10x H5...")
            self._convert_chunked(h5_path, output_path)
        else:
            # Try to read as 10x H5 first, fall back to regular h5ad
            try:
                adata = sc.read_10x_h5(h5_path, genome="X")
            except Exception:
                # Fall back to reading as regular h5ad
                logger.info("Reading as regular h5ad file...")
                adata = sc.read_h5ad(h5_path)

            logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

            # Convert using existing AnnData conversion logic
            self._convert_anndata(adata, output_path)

    def _convert_anndata(self, adata, output_path: str):
        """Internal method to convert AnnData object to SLAF format"""
        # Create output directory
        output_path_obj = Path(output_path)
        output_path_obj.mkdir(exist_ok=True)

        # Validate optimized data types and determine value type
        validation_result, value_type = self._validate_optimized_dtypes_anndata(adata)
        if not validation_result:
            self.use_optimized_dtypes = False

        # Create integer key mappings if needed
        cell_id_mapping = None
        gene_id_mapping = None

        if self.use_integer_keys:
            logger.info("Creating integer key mappings...")
            cell_id_mapping = self._create_id_mapping(adata.obs.index, "cell")
            gene_id_mapping = self._create_id_mapping(adata.var.index, "gene")

        # Convert expression data to COO format
        logger.info("Converting expression data to COO format...")
        expression_table = self._sparse_to_coo_table(
            sparse_matrix=adata.X,
            cell_ids=adata.obs.index,
            gene_ids=adata.var.index,
            value_type=value_type,
        )

        # Convert metadata
        logger.info("Converting metadata...")

        # Note: Sorting is disabled to maintain consistency between metadata and expression data ordering
        # TODO: Implement proper sorting that affects both metadata and expression data
        obs_df = adata.obs.copy()
        var_df = adata.var.copy()

        cell_metadata_table = self._create_metadata_table(
            df=obs_df, entity_id_col="cell_id", integer_mapping=cell_id_mapping
        )
        gene_metadata_table = self._create_metadata_table(
            df=var_df, entity_id_col="gene_id", integer_mapping=gene_id_mapping
        )

        # Write all Lance tables
        logger.info("Writing Lance tables...")
        table_configs = [
            ("expression", expression_table),
            ("cells", cell_metadata_table),
            ("genes", gene_metadata_table),
        ]

        self._write_lance_tables(output_path_obj, table_configs)

        # Compact dataset for optimal storage
        self._compact_dataset(output_path_obj)

        # Save config
        self._save_config(output_path_obj, adata.shape)
        logger.info(f"Conversion complete! Saved to {output_path}")

    def _convert_chunked(self, h5ad_path: str, output_path: str):
        """Convert h5ad file using chunked processing with sorted-by-construction approach"""
        logger.info(f"Processing in chunks of {self.chunk_size} cells...")

        with create_chunked_reader(h5ad_path) as reader:
            logger.info(f"Loaded: {reader.n_obs:,} cells × {reader.n_vars:,} genes")

            # Validate optimized data types and determine value type
            validation_result, value_type = self._validate_optimized_dtypes(reader)
            if not validation_result:
                self.use_optimized_dtypes = False

            # Create output directory
            output_path_obj = Path(output_path)
            output_path_obj.mkdir(exist_ok=True)

            # Write metadata tables efficiently (without loading everything into memory)
            self._write_metadata_efficiently(reader, output_path_obj)

            # Process expression data
            self._process_expression(reader, output_path_obj, value_type)

            # Create indices (if enabled)
            if self.create_indices:
                self._create_indices(output_path_obj)

            # Compact dataset for optimal storage
            self._compact_dataset(output_path_obj)

            # Save config
            self._save_config(output_path_obj, (reader.n_obs, reader.n_vars))
            logger.info(f"Conversion complete! Saved to {output_path}")

    def _write_metadata_efficiently(self, reader, output_path_obj: Path):
        """Write metadata tables efficiently while preserving all columns"""
        logger.info("Writing metadata tables...")

        # Get full metadata from reader (this loads all columns)
        obs_df = reader.get_obs_metadata()
        var_df = reader.get_var_metadata()

        # Ensure cell_id and gene_id columns exist with actual names
        if "cell_id" not in obs_df.columns:
            obs_df["cell_id"] = reader.obs_names
        if "gene_id" not in var_df.columns:
            var_df["gene_id"] = reader.var_names

        # Note: Sorting is disabled in chunked mode to maintain consistency
        # between metadata and expression data ordering
        # TODO: Implement proper sorting that affects both metadata and expression data

        # Add integer IDs if enabled
        if self.use_integer_keys:
            obs_df["cell_integer_id"] = range(len(obs_df))
            var_df["gene_integer_id"] = range(len(var_df))

        # Convert to Lance tables
        cell_metadata_table = self._create_metadata_table(
            obs_df,
            "cell_id",
            integer_mapping=None,  # Already added above
        )
        gene_metadata_table = self._create_metadata_table(
            var_df,
            "gene_id",
            integer_mapping=None,  # Already added above
        )

        # Get compression settings for metadata tables
        metadata_settings = self._get_compression_settings("metadata")

        # Write metadata tables
        lance.write_dataset(
            cell_metadata_table,
            str(output_path_obj / "cells.lance"),
            mode="overwrite",
            max_rows_per_group=metadata_settings["max_rows_per_group"],
            enable_v2_manifest_paths=self.enable_v2_manifest,
        )
        lance.write_dataset(
            gene_metadata_table,
            str(output_path_obj / "genes.lance"),
            mode="overwrite",
            max_rows_per_group=metadata_settings["max_rows_per_group"],
            enable_v2_manifest_paths=self.enable_v2_manifest,
        )

        logger.info("Metadata tables written!")

    def _process_expression(self, reader, output_path_obj: Path, value_type="uint16"):
        """Process expression data in single-threaded mode with large chunks"""
        logger.info("Processing expression data in single-threaded mode...")

        # Calculate total chunks
        total_chunks = (reader.n_obs + self.chunk_size - 1) // self.chunk_size
        logger.info(
            f"Processing {total_chunks} chunks with chunk size {self.chunk_size:,}..."
        )

        # Memory monitoring
        process = None
        initial_memory = None
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        except ImportError:
            logger.info("Install psutil for memory monitoring: pip install psutil")

        # Create Lance dataset with schema
        expression_path = output_path_obj / "expression.lance"
        schema = self._get_expression_schema(value_type)

        # Create empty dataset first
        logger.info("Creating initial Lance dataset...")
        schema = self._get_expression_schema(value_type)

        # Create empty table with correct schema based on settings
        if value_type == "uint16":
            value_pa_type = pa.uint16()
        elif value_type == "float32":
            value_pa_type = pa.float32()
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

        if self.optimize_storage:
            if self.use_optimized_dtypes:
                empty_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )
            else:
                empty_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.int32()),
                        "gene_integer_id": pa.array([], type=pa.int32()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )
        else:
            if self.use_optimized_dtypes:
                empty_table = pa.table(
                    {
                        "cell_id": pa.array([], type=pa.string()),
                        "gene_id": pa.array([], type=pa.string()),
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )
            else:
                empty_table = pa.table(
                    {
                        "cell_id": pa.array([], type=pa.string()),
                        "gene_id": pa.array([], type=pa.string()),
                        "cell_integer_id": pa.array([], type=pa.int32()),
                        "gene_integer_id": pa.array([], type=pa.int32()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )

        lance.write_dataset(
            empty_table,
            str(expression_path),
            mode="overwrite",
            schema=schema,
            max_rows_per_file=self._get_compression_settings("expression")[
                "max_rows_per_file"
            ],
            max_rows_per_group=self._get_compression_settings("expression")[
                "max_rows_per_group"
            ],
            max_bytes_per_file=self._get_compression_settings("expression")[
                "max_bytes_per_file"
            ],
            enable_v2_manifest_paths=self.enable_v2_manifest,
        )

        # Process chunks sequentially
        logger.info("Processing chunks sequentially...")
        for chunk_idx, (chunk_table, obs_slice) in enumerate(
            reader.iter_chunks(chunk_size=self.chunk_size)
        ):
            logger.info(
                f"Processing chunk {chunk_idx + 1}/{total_chunks} ({obs_slice})"
            )

            # Convert data types if needed
            if not self.use_optimized_dtypes:
                # Convert from optimized dtypes to standard dtypes
                cell_integer_ids = (
                    chunk_table.column("cell_integer_id").to_numpy().astype(np.int32)
                )
                gene_integer_ids = (
                    chunk_table.column("gene_integer_id").to_numpy().astype(np.int32)
                )
                values = chunk_table.column("value").to_numpy().astype(np.float32)

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values),
                    }
                )

            # Add string IDs if optimize_storage=False
            if not self.optimize_storage:
                # Get cell and gene names
                cell_names = reader.obs_names
                gene_names = reader.var_names

                # Create string ID arrays
                cell_integer_ids = chunk_table.column("cell_integer_id").to_numpy()
                gene_integer_ids = chunk_table.column("gene_integer_id").to_numpy()

                cell_ids = cell_names[cell_integer_ids].astype(str)
                gene_ids = gene_names[gene_integer_ids].astype(str)

                # Create new table with string IDs
                chunk_table = pa.table(
                    {
                        "cell_id": pa.array(cell_ids),
                        "gene_id": pa.array(gene_ids),
                        "cell_integer_id": chunk_table.column("cell_integer_id"),
                        "gene_integer_id": chunk_table.column("gene_integer_id"),
                        "value": chunk_table.column("value"),
                    }
                )

            # Write to Lance
            lance.write_dataset(
                chunk_table,
                str(expression_path),
                mode="append",
                max_rows_per_file=self._get_compression_settings("expression")[
                    "max_rows_per_file"
                ],
                max_rows_per_group=self._get_compression_settings("expression")[
                    "max_rows_per_group"
                ],
                max_bytes_per_file=self._get_compression_settings("expression")[
                    "max_bytes_per_file"
                ],
                enable_v2_manifest_paths=self.enable_v2_manifest,
            )

            logger.info(f"Completed chunk {chunk_idx + 1}/{total_chunks}")

        # Final memory report
        if process is not None and initial_memory is not None:
            try:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                logger.info(
                    f"Final memory usage: {final_memory:.1f} MB (change: {memory_increase:+.1f} MB)"
                )
            except Exception:
                pass

        logger.info("Expression data processing complete!")

    def _validate_optimized_dtypes(self, reader):
        """Validate that data fits in optimized data types and determine appropriate value type"""
        if not self.use_optimized_dtypes:
            return True, "float32"

        logger.info("Validating data fits in optimized data types...")

        # Check if gene count fits in uint16 (0-65535)
        if reader.n_vars > 65535:
            logger.info(
                f"Warning: {reader.n_vars:,} genes exceeds uint16 limit (65535)"
            )
            logger.info("Falling back to standard data types")
            return False, "float32"

        # Check if cell count fits in uint32 (0-4,294,967,295)
        if reader.n_obs > 4294967295:
            logger.info(
                f"Warning: {reader.n_obs:,} cells exceeds uint32 limit (4,294,967,295)"
            )
            logger.info("Falling back to standard data types")
            return False, "float32"

        # Sample original data from the file to determine data type
        logger.info("Sampling original expression values to determine data type...")

        # For chunked readers, we need to check the original data type from the file
        if hasattr(reader, "file") and reader.file is not None:
            # Check the original data type from the h5ad file
            X_group = reader.file["X"]
            if "data" in X_group:
                # Sample some original data
                data = X_group["data"]
                sample_size = min(10000, len(data))
                # Use sequential sampling instead of random to avoid indexing issues
                sample_data = data[:sample_size]

                # Check if original data is integer or float
                is_integer = np.issubdtype(sample_data.dtype, np.integer)
                max_value = np.max(sample_data)
                min_value = np.min(sample_data)

                if is_integer and max_value <= 65535 and min_value >= 0:
                    logger.info(
                        f"Original integer expression values fit in uint16 range: [{min_value}, {max_value}]"
                    )
                    logger.info("Using uint16 for integer count data")
                    return True, "uint16"
                elif not is_integer:
                    # Check if float data contains only integer values
                    # Round to nearest integer and check if it's the same
                    rounded_data = np.round(sample_data)
                    is_integer_values = np.allclose(
                        sample_data, rounded_data, rtol=1e-10
                    )

                    if is_integer_values and max_value <= 65535 and min_value >= 0:
                        logger.info(
                            f"Float data contains only integer values: [{min_value}, {max_value}]"
                        )
                        logger.info("Converting to uint16 for count data")
                        return True, "uint16"
                    else:
                        logger.info(
                            f"Float data contains fractional values: [{min_value}, {max_value}]"
                        )
                        logger.info("Keeping as float32 for normalized/float data")
                        return False, "float32"
                else:
                    logger.info(
                        f"Warning: Original integer values range [{min_value}, {max_value}] exceeds uint16 range [0, 65535]"
                    )
                    logger.info("Falling back to float32")
                    return False, "float32"
            else:
                # Fallback to checking processed data
                sample_size = min(100000, reader.n_obs)
                sample_chunks = list(reader.iter_chunks(chunk_size=sample_size))

                if sample_chunks:
                    sample_chunk = sample_chunks[0][0]

                    # Handle different data types from chunked reader
                    if hasattr(sample_chunk, "column"):  # PyArrow Table
                        # Get the value column from the PyArrow table
                        value_column = sample_chunk.column("value")
                        sample_data = value_column.to_numpy()
                    elif sparse.issparse(sample_chunk):
                        sample_data = sample_chunk.data
                    else:
                        sample_data = sample_chunk.flatten()

                    # Check if data is integer or float
                    is_integer = np.issubdtype(sample_data.dtype, np.integer)
                    max_value = np.max(sample_data)
                    min_value = np.min(sample_data)

                    if is_integer and max_value <= 65535 and min_value >= 0:
                        logger.info(
                            f"Integer expression values fit in uint16 range: [{min_value}, {max_value}]"
                        )
                        logger.info("Using uint16 for integer count data")
                        return True, "uint16"
                    elif not is_integer:
                        # Check if float data contains only integer values
                        # Round to nearest integer and check if it's the same
                        rounded_data = np.round(sample_data)
                        is_integer_values = np.allclose(
                            sample_data, rounded_data, rtol=1e-10
                        )

                        if is_integer_values and max_value <= 65535 and min_value >= 0:
                            logger.info(
                                f"Float data contains only integer values: [{min_value}, {max_value}]"
                            )
                            logger.info("Converting to uint16 for count data")
                            return True, "uint16"
                        else:
                            logger.info(
                                f"Float data contains fractional values: [{min_value}, {max_value}]"
                            )
                            logger.info("Keeping as float32 for normalized/float data")
                            return False, "float32"
                    else:
                        logger.info(
                            f"Warning: Integer values range [{min_value}, {max_value}] exceeds uint16 range [0, 65535]"
                        )
                        logger.info("Falling back to float32")
                        return False, "float32"

        logger.info("Data validation passed - using optimized data types")
        return True, "uint16"  # Default to uint16 for integer data

    def _validate_optimized_dtypes_anndata(self, adata):
        """Validate that AnnData object's expression data fits in optimized data types and determine appropriate value type"""
        if not self.use_optimized_dtypes:
            return True, "float32"

        logger.info(
            "Validating AnnData object's expression data fits in optimized data types..."
        )

        # Check if gene count fits in uint16 (0-65535)
        if adata.n_vars > 65535:
            logger.info(f"Warning: {adata.n_vars:,} genes exceeds uint16 limit (65535)")
            logger.info("Falling back to standard data types")
            return False, "float32"

        # Check if cell count fits in uint32 (0-4,294,967,295)
        if adata.n_obs > 4294967295:
            logger.warning(
                f"{adata.n_obs:,} cells exceeds uint32 limit (4,294,967,295)"
            )
            logger.info("Falling back to standard data types")
            return False, "float32"

        # Sample some values to check data type and range
        logger.info("Sampling expression values to determine data type...")
        sample_data = adata.X.data[:100000]

        # Check if data is integer or float
        is_integer = np.issubdtype(sample_data.dtype, np.integer)
        max_value = np.max(sample_data)
        min_value = np.min(sample_data)

        if is_integer and max_value <= 65535 and min_value >= 0:
            logger.info(
                f"Integer expression values fit in uint16 range: [{min_value}, {max_value}]"
            )
            logger.info("Using uint16 for integer count data")
            return True, "uint16"
        elif not is_integer:
            logger.info(f"Float expression values detected: [{min_value}, {max_value}]")
            logger.info("Using float32 for normalized/float data")
            return True, "float32"
        else:
            logger.info(
                f"Warning: Integer values range [{min_value}, {max_value}] exceeds uint16 range [0, 65535]"
            )
            logger.info("Falling back to float32")
            return False, "float32"

        logger.info(
            "AnnData object's expression data validation passed - using optimized data types"
        )
        return True, "uint16"  # Default to uint16 for integer data

    def _compact_dataset(self, output_path_obj: Path):
        """Compact the dataset to optimize storage after writing"""
        if not self.compact_after_write:
            return

        logger.info("Compacting dataset for optimal storage...")

        # Compact expression table
        expression_path = output_path_obj / "expression.lance"
        if expression_path.exists():
            logger.info("  Compacting expression table...")
            dataset = lance.dataset(str(expression_path))
            dataset.optimize.compact_files(
                target_rows_per_fragment=1024 * 1024
            )  # 1M rows per fragment
            logger.info("  Expression table compacted!")

        # Compact metadata tables
        for table_name in ["cells", "genes"]:
            table_path = output_path_obj / f"{table_name}.lance"
            if table_path.exists():
                logger.info(f"  Compacting {table_name} table...")
                dataset = lance.dataset(str(table_path))
                dataset.optimize.compact_files(
                    target_rows_per_fragment=100000
                )  # 100K rows per fragment for metadata
                logger.info(f"  {table_name} table compacted!")

        logger.info("Dataset compaction complete!")

    def _get_expression_schema(self, value_type="uint16"):
        """Get the schema for expression table"""
        # Determine the appropriate value type
        if value_type == "uint16":
            value_pa_type = pa.uint16()
        elif value_type == "float32":
            value_pa_type = pa.float32()
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

        if self.optimize_storage:
            # Only store integer IDs for maximum storage efficiency
            if self.use_optimized_dtypes:
                # Use optimized data types for better compression
                return pa.schema(
                    [
                        ("cell_integer_id", pa.uint32()),
                        ("gene_integer_id", pa.uint16()),
                        ("value", value_pa_type),
                    ]
                )
            else:
                # Use standard data types
                return pa.schema(
                    [
                        ("cell_integer_id", pa.int32()),
                        ("gene_integer_id", pa.int32()),
                        ("value", value_pa_type),
                    ]
                )
        else:
            # Store both string and integer IDs for compatibility
            if self.use_optimized_dtypes:
                return pa.schema(
                    [
                        ("cell_id", pa.string()),
                        ("gene_id", pa.string()),
                        ("cell_integer_id", pa.uint32()),
                        ("gene_integer_id", pa.uint16()),
                        ("value", value_pa_type),
                    ]
                )
            else:
                return pa.schema(
                    [
                        ("cell_id", pa.string()),
                        ("gene_id", pa.string()),
                        ("cell_integer_id", pa.int32()),
                        ("gene_integer_id", pa.int32()),
                        ("value", value_pa_type),
                    ]
                )

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
        value_type="uint16",
    ):
        """Convert scipy sparse matrix to COO format PyArrow table with integer IDs"""
        coo_matrix = sparse_matrix.tocoo()

        logger.info(f"Processing {len(coo_matrix.data):,} non-zero elements...")

        # Create integer ID arrays for efficient range queries
        if self.use_optimized_dtypes:
            cell_integer_id_array = coo_matrix.row.astype(np.uint32)
            gene_integer_id_array = coo_matrix.col.astype(np.uint16)
            # Convert values based on the determined type
            if value_type == "uint16":
                value_array = coo_matrix.data.astype(np.uint16)
                value_dtype = np.uint16
            elif value_type == "float32":
                value_array = coo_matrix.data.astype(np.float32)
                value_dtype = np.float32
            else:
                raise ValueError(f"Unsupported value type: {value_type}")
        else:
            cell_integer_id_array = coo_matrix.row.astype(np.int32)
            gene_integer_id_array = coo_matrix.col.astype(np.int32)
            # Expression values - use the determined type
            if value_type == "uint16":
                value_array = coo_matrix.data.astype(np.uint16)
                value_dtype = np.uint16
            elif value_type == "float32":
                value_array = coo_matrix.data.astype(np.float32)
                value_dtype = np.float32
            else:
                raise ValueError(f"Unsupported value type: {value_type}")

        # Create string ID arrays
        cell_id_array = np.array(cell_ids)[coo_matrix.row].astype(str)
        gene_id_array = np.array(gene_ids)[coo_matrix.col].astype(str)

        # Check for nulls in string arrays
        if bool(np.any(pd.isnull(cell_id_array))) or bool(
            np.any(pd.isnull(gene_id_array))
        ):
            raise ValueError("Null values found in cell_id or gene_id arrays!")

        # Create table based on storage optimization
        if self.optimize_storage:
            # Only store integer IDs for maximum storage efficiency
            if self.use_optimized_dtypes:
                table = pa.table(
                    {
                        "cell_integer_id": pa.array(
                            cell_integer_id_array, type=pa.uint32()
                        ),
                        "gene_integer_id": pa.array(
                            gene_integer_id_array, type=pa.uint16()
                        ),
                        "value": pa.array(
                            value_array.astype(value_dtype), type=value_type
                        ),
                    }
                )
            else:
                table = pa.table(
                    {
                        "cell_integer_id": pa.array(
                            cell_integer_id_array, type=pa.int32()
                        ),
                        "gene_integer_id": pa.array(
                            gene_integer_id_array, type=pa.int32()
                        ),
                        "value": pa.array(
                            value_array.astype(value_dtype), type=value_type
                        ),
                    }
                )
        else:
            # Store both string and integer IDs for compatibility
            if self.use_optimized_dtypes:
                table = pa.table(
                    {
                        "cell_id": pa.array(cell_id_array, type=pa.string()),
                        "gene_id": pa.array(gene_id_array, type=pa.string()),
                        "cell_integer_id": pa.array(
                            cell_integer_id_array, type=pa.uint32()
                        ),
                        "gene_integer_id": pa.array(
                            gene_integer_id_array, type=pa.uint16()
                        ),
                        "value": pa.array(
                            value_array.astype(value_dtype), type=value_type
                        ),
                    }
                )
            else:
                table = pa.table(
                    {
                        "cell_id": pa.array(cell_id_array, type=pa.string()),
                        "gene_id": pa.array(gene_id_array, type=pa.string()),
                        "cell_integer_id": pa.array(
                            cell_integer_id_array, type=pa.int32()
                        ),
                        "gene_integer_id": pa.array(
                            gene_integer_id_array, type=pa.int32()
                        ),
                        "value": pa.array(
                            value_array.astype(value_dtype), type=value_type
                        ),
                    }
                )

        # Validate schema
        if self.optimize_storage:
            if self.use_optimized_dtypes:
                expected_types = {
                    "cell_integer_id": pa.uint32(),
                    "gene_integer_id": pa.uint16(),
                    "value": value_type,
                }
            else:
                expected_types = {
                    "cell_integer_id": pa.int32(),
                    "gene_integer_id": pa.int32(),
                    "value": value_type,
                }
        else:
            if self.use_optimized_dtypes:
                expected_types = {
                    "cell_id": pa.string(),
                    "gene_id": pa.string(),
                    "cell_integer_id": pa.uint32(),
                    "gene_integer_id": pa.uint16(),
                    "value": value_type,
                }
            else:
                expected_types = {
                    "cell_id": pa.string(),
                    "gene_id": pa.string(),
                    "cell_integer_id": pa.int32(),
                    "gene_integer_id": pa.int32(),
                    "value": value_type,
                }

        # Validate schema
        for col, expected_type in expected_types.items():
            assert table.schema.field(col).type == expected_type, (
                f"{col} is not {expected_type} type!"
            )
            assert table.column(col).null_count == 0, f"Nulls found in {col} column!"

        return table

    def _get_compression_settings(self, table_type: str = "expression"):
        """Get optimal compression settings for high compression (write once, query infinitely)"""
        if table_type == "expression":
            # Expression tables benefit from very large groups due to sparsity
            # Use maximum compression settings for massive datasets
            return {
                "max_rows_per_file": 50000000,  # 50M rows per file (increased from 10M)
                "max_rows_per_group": 10000000,  # 10M rows per group (increased from 2M)
                "max_bytes_per_file": 100
                * 1024
                * 1024
                * 1024,  # 100GB limit (increased from 50GB)
            }
        else:
            # Metadata tables
            return {
                "max_rows_per_group": 500000,  # 500K rows per group (increased from 200K)
            }

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

        return table

    def _write_lance_tables(
        self, output_path: Path, table_configs: list[tuple[str, pa.Table]]
    ):
        """Write multiple Lance tables with consistent naming"""
        for table_name, table in table_configs:
            table_path = output_path / f"{table_name}.lance"

            # Use optimized compression settings based on table type
            if table_name == "expression":
                compression_settings = self._get_compression_settings("expression")
                lance.write_dataset(
                    table,
                    str(table_path),
                    max_rows_per_file=compression_settings["max_rows_per_file"],
                    max_rows_per_group=compression_settings["max_rows_per_group"],
                    max_bytes_per_file=compression_settings["max_bytes_per_file"],
                    enable_v2_manifest_paths=self.enable_v2_manifest,
                )
            else:
                # Metadata tables
                compression_settings = self._get_compression_settings("metadata")
                lance.write_dataset(
                    table,
                    str(table_path),
                    max_rows_per_group=compression_settings["max_rows_per_group"],
                    enable_v2_manifest_paths=self.enable_v2_manifest,
                )

        # Create indices after all tables are written (if enabled)
        if self.create_indices:
            self._create_indices(output_path)

    def _create_indices(self, output_path: Path):
        """Create optimal indices for SLAF tables with column existence checks"""
        logger.info("Creating indices for optimal query performance...")

        # Define desired indices for each table
        # For small datasets, create fewer indices to reduce overhead
        table_indices = {
            "cells": [
                "cell_id",
                "cell_integer_id",
                # Only create metadata indices for larger datasets
            ],
            "genes": ["gene_id", "gene_integer_id"],
            "expression": [
                "cell_integer_id",
                "gene_integer_id",
            ],  # Only integer indices for efficiency
        }

        # Create indices for each table
        for table_name, desired_columns in table_indices.items():
            table_path = output_path / f"{table_name}.lance"
            if table_path.exists():
                dataset = lance.dataset(str(table_path))
                schema = dataset.schema

                for column in desired_columns:
                    if column in schema.names:
                        logger.info(f"  Creating index on {table_name}.{column}")
                        dataset.create_scalar_index(column, "BTREE")

        logger.info("Index creation complete!")

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
        logger.info("Computing dataset statistics...")

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
                "optimize_storage": self.optimize_storage,
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
