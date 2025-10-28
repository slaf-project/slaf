import json
import os
from typing import Any

import lance
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from loguru import logger
from scipy import sparse

# Import smart-open for cloud storage compatibility
try:
    from smart_open import open as smart_open

    SMART_OPEN_AVAILABLE = True
except ImportError:
    SMART_OPEN_AVAILABLE = False
    logger.warning("smart-open not available. Install with: pip install smart-open[s3]")

# Optional imports for data conversion
try:
    import scanpy as sc

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logger.warning("Scanpy not available. Install with: pip install scanpy")

from .chunked_reader import create_chunked_reader
from .utils import discover_input_files, validate_input_files


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
        chunk_size: int = 5000,  # Smaller chunks to avoid memory alignment issues in Lance v2.1
        sort_metadata: bool = False,
        create_indices: bool = False,  # Disable indices by default for small datasets
        optimize_storage: bool = True,  # Only store integer IDs in expression table
        use_optimized_dtypes: bool = True,  # Use uint16/uint32 for better compression
        enable_v2_manifest: bool = True,  # Enable v2 manifest paths for better performance
        compact_after_write: bool = False,  # Compact dataset after writing for optimal storage (disabled by default to avoid manifest corruption)
        tiledb_collection_name: str = "RNA",  # Collection name for TileDB format
        enable_checkpointing: bool = True,  # Enable checkpointing for long-running conversions
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
            chunk_size: Size of each chunk when chunked=True (default: 5000).
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
            tiledb_collection_name: Name of the measurement collection for TileDB format.
                                  Default: "RNA". Only used when converting from TileDB format.
            enable_checkpointing: Enable checkpointing for long-running conversions.
                                This allows resuming from the last completed chunk if the
                                conversion is interrupted. Default: True.

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
            >>> converter = SLAFConverter(chunk_size=10000)
            >>> print(f"Chunk size: {converter.chunk_size}")
            Chunk size: 5000
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
        self.tiledb_collection_name = tiledb_collection_name
        self.enable_checkpointing = enable_checkpointing

    def _is_cloud_path(self, path: str) -> bool:
        """Check if path is a cloud storage path."""
        return path.startswith(("s3://", "gs://", "azure://", "r2://"))

    def _path_exists(self, path: str) -> bool:
        """Check if path exists, works with both local and cloud paths."""
        if self._is_cloud_path(path):
            if not SMART_OPEN_AVAILABLE:
                logger.warning(
                    "smart-open not available, cannot check cloud path existence"
                )
                return False
            try:
                # For Lance datasets, check if the directory exists by trying to list contents
                # Lance datasets are directories, not files
                if path.endswith(".lance"):
                    # Try to access the Lance dataset directory
                    import lance

                    lance.dataset(path)  # This will fail if the dataset doesn't exist
                    return True
                else:
                    # For regular files, try to access the file directly
                    with smart_open(path, "r") as f:
                        f.read(1)  # Try to read 1 byte
                    return True
            except Exception:
                return False
        else:
            return os.path.exists(path)

    def _ensure_directory_exists(self, path: str):
        """Ensure directory exists, works with both local and cloud paths."""
        if not self._is_cloud_path(path):
            os.makedirs(path, exist_ok=True)
        # For cloud paths, directories are created implicitly

    def _open_file(self, path: str, mode: str = "r"):
        """Open file with cloud storage compatibility."""
        if self._is_cloud_path(path):
            if not SMART_OPEN_AVAILABLE:
                raise ImportError("smart-open required for cloud storage operations")
            return smart_open(path, mode)
        else:
            return open(path, mode)

    def _save_checkpoint(self, output_path: str, checkpoint_data: dict):
        """Save checkpoint data to config.json"""
        if not self.enable_checkpointing:
            return

        config_path = f"{output_path}/config.json"

        # Load existing config if it exists, otherwise create minimal config
        if self._path_exists(config_path):
            with self._open_file(config_path) as f:
                config = json.load(f)
        else:
            # Create minimal config for checkpointing
            config = {
                "format_version": "0.3",
                "array_shape": [0, 0],  # Will be updated when conversion completes
                "n_cells": 0,
                "n_genes": 0,
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
                    "conversion_in_progress": True,
                    "sparsity": 0.0,
                    "density": 0.0,
                    "expression_count": 0,
                },
            }

        # Update checkpoint data
        config["checkpoint"] = checkpoint_data

        # Save updated config
        with self._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_data}")

    def _load_checkpoint(self, output_path: str) -> dict | None:
        """Load checkpoint data from config.json"""
        if not self.enable_checkpointing:
            return None

        config_path = f"{output_path}/config.json"

        if not self._path_exists(config_path):
            return None

        try:
            with self._open_file(config_path) as f:
                config = json.load(f)
                return config.get("checkpoint")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return None

    def _load_checkpoint_smart(self, output_path: str) -> dict | None:
        """Load checkpoint with fragment-based resume logic to avoid duplication"""
        checkpoint = self._load_checkpoint(output_path)
        if not checkpoint:
            return None

        # Get actual progress from Lance dataset using fragment count
        expression_path = f"{output_path}/expression.lance"
        logger.info(f"🔍 Checking fragment-based resume for: {expression_path}")

        # First, check if the path exists
        path_exists = self._path_exists(expression_path)
        logger.info(f"🔍 Path exists check result: {path_exists}")

        if path_exists:
            logger.info("✅ Expression dataset exists, loading...")
            try:
                expression_dataset = lance.dataset(expression_path)
                fragment_count = len(expression_dataset.get_fragments())
                logger.info(f"🔍 Found {fragment_count} fragments in dataset")
                logger.info(
                    f"🔍 Original checkpoint: last_completed_chunk={checkpoint.get('last_completed_chunk')}"
                )

                # Calculate exact resume chunk using fragment count
                last_checkpoint_boundary = checkpoint.get("last_completed_chunk", 0)
                checkpoint_interval = (
                    10  # Should match the interval used in checkpointing
                )
                chunks_since_checkpoint = fragment_count % checkpoint_interval

                # Resume from the next chunk after what was actually written
                # The fragment count represents the actual progress, so we don't need total_chunks
                # Just ensure we don't exceed the total chunks for the current file
                actual_resume_chunk = (
                    last_checkpoint_boundary + chunks_since_checkpoint + 1
                )

                # Update checkpoint with actual resume position
                checkpoint["last_completed_chunk"] = actual_resume_chunk - 1
                checkpoint["fragment_count"] = fragment_count
                checkpoint["chunks_since_checkpoint"] = chunks_since_checkpoint

                logger.info(
                    f"Fragment-based resume: {fragment_count} fragments written, "
                    f"chunks since checkpoint: {chunks_since_checkpoint}, "
                    f"resuming from chunk {actual_resume_chunk}"
                )
                logger.info(
                    f"🔧 Updated checkpoint: last_completed_chunk={checkpoint['last_completed_chunk']}"
                )

            except Exception as e:
                logger.warning(f"Could not determine fragment count: {e}")
                # Fall back to regular checkpoint logic
                pass

        return checkpoint

    def _clear_checkpoint(self, output_path: str):
        """Clear checkpoint data from config.json"""
        if not self.enable_checkpointing:
            return

        config_path = f"{output_path}/config.json"

        if not self._path_exists(config_path):
            return

        try:
            with self._open_file(config_path) as f:
                config = json.load(f)

            # Remove checkpoint data
            if "checkpoint" in config:
                del config["checkpoint"]

            # Save updated config
            with self._open_file(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info("Checkpoint cleared - conversion completed successfully")
        except Exception as e:
            logger.warning(f"Could not clear checkpoint: {e}")

    def convert(
        self,
        input_path: str | list[str],
        output_path: str,
        input_format: str = "auto",
        skip_validation: bool = False,
    ):
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
            - **tiledb**: TileDB SOMA format (.tiledb) - high-performance single-cell format

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
                         - "tiledb": TileDB SOMA format

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

            >>> # Convert TileDB SOMA file
            >>> converter.convert("data.tiledb", "output.slaf")
            Converting TileDB SOMA file data.tiledb to SLAF format...
            Loaded: 50000 cells × 20000 genes
            Conversion complete! Saved to output.slaf

            >>> # Explicit format specification
            >>> converter.convert("data.h5", "output.slaf", input_format="10x_h5")
            Converting 10x H5 file data.h5 to SLAF format...
            Loaded: 2700 cells × 32738 genes
            Conversion complete! Saved to output.slaf

            >>> # Convert with chunked processing for large datasets
            >>> converter = SLAFConverter(chunked=True, chunk_size=5000)
            >>> converter.convert("large_data.h5ad", "output.slaf")
            Converting large_data.h5ad to SLAF format...
            Optimizations: int_keys=True, chunked=True
            Processing in chunks of 5000 cells...
            Conversion complete! Saved to output.slaf

            >>> # Error handling for unsupported format
            >>> try:
            ...     converter.convert("unknown_file.txt", "output.slaf")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Cannot detect format for: unknown_file.txt
        """
        # Handle both single paths and lists of files
        if isinstance(input_path, list):
            # Direct list of files - use multi-file conversion
            input_files = input_path
            # Detect format from first file
            if input_format == "auto":
                from .utils import detect_format

                input_format = detect_format(input_files[0])
        else:
            # Single path - discover files
            input_files, detected_format = discover_input_files(input_path)
            # Use detected format if auto, otherwise use specified format
            if input_format == "auto":
                input_format = detected_format

        # Validate multi-file compatibility if multiple files (unless skipped)
        if not skip_validation:
            validate_input_files(input_files, input_format)

        # Handle multiple files vs single file
        if len(input_files) > 1:
            self._convert_multiple_files(input_files, output_path, input_format)
        else:
            # Single file - use existing logic
            single_file = input_files[0]
            if input_format == "h5ad":
                if not SCANPY_AVAILABLE:
                    raise ImportError(
                        "Scanpy is required for h5ad conversion. "
                        "Install with: pip install scanpy"
                    )
                self._convert_h5ad(single_file, output_path)
            elif input_format == "10x_mtx":
                self._convert_10x_mtx(single_file, output_path)
            elif input_format == "10x_h5":
                self._convert_10x_h5(single_file, output_path)
            elif input_format == "tiledb":
                self._convert_tiledb(single_file, output_path)
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
            # Load h5ad using scanpy backed mode, then convert to in-memory
            logger.info("Loading h5ad file in backed mode...")
            adata_backed = sc.read_h5ad(h5ad_path, backed="r")
            logger.info(
                f"Loaded: {adata_backed.n_obs} cells × {adata_backed.n_vars} genes"
            )

            # Convert backed data to in-memory AnnData to avoid CSRDataset issues
            logger.info("Converting backed data to in-memory format...")
            adata = sc.AnnData(
                X=adata_backed.X[:],  # Load the full matrix into memory
                obs=adata_backed.obs.copy(),
                var=adata_backed.var.copy(),
                uns=adata_backed.uns.copy() if hasattr(adata_backed, "uns") else {},
            )

            # Close the backed file
            adata_backed.file.close()

            logger.info("Successfully converted to in-memory AnnData")

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
                adata_backed = sc.read_10x_h5(h5_path, genome="X")
                logger.info("Successfully read as 10x H5 format")
            except Exception:
                # Fall back to reading as regular h5ad
                logger.info("Reading as regular h5ad file...")
                adata_backed = sc.read_h5ad(h5_path, backed="r")

            logger.info(
                f"Loaded: {adata_backed.n_obs} cells × {adata_backed.n_vars} genes"
            )

            # Convert backed data to in-memory AnnData to avoid CSRDataset issues
            logger.info("Converting backed data to in-memory format...")
            adata = sc.AnnData(
                X=adata_backed.X[:],  # Load the full matrix into memory
                obs=adata_backed.obs.copy(),
                var=adata_backed.var.copy(),
                uns=adata_backed.uns.copy() if hasattr(adata_backed, "uns") else {},
            )

            # Close the backed file if it exists
            if hasattr(adata_backed, "file") and adata_backed.file is not None:
                adata_backed.file.close()

            logger.info("Successfully converted to in-memory AnnData")

            # Convert using existing AnnData conversion logic
            self._convert_anndata(adata, output_path)

    def _convert_tiledb(self, tiledb_path: str, output_path: str):
        """Convert TileDB SOMA file to SLAF format"""
        logger.info(f"Converting TileDB SOMA file {tiledb_path} to SLAF format...")

        if self.chunked:
            # Use native chunked reader for TileDB
            logger.info("Using native chunked reader for TileDB...")
            self._convert_chunked(tiledb_path, output_path)
        else:
            # For non-chunked mode, we need to load the entire dataset
            # This is not recommended for large datasets
            logger.warning("Non-chunked mode not recommended for TileDB format")
            raise NotImplementedError(
                "Non-chunked conversion not supported for TileDB format. "
                "Use chunked=True (default) for TileDB conversion."
            )

    def _convert_multiple_files(
        self, input_files: list[str], output_path: str, input_format: str
    ):
        """
        Convert multiple files to a single SLAF dataset with auto-incrementing IDs and checkpointing.

        This method processes files sequentially and adds fragments to the same
        SLAF dataset, avoiding the overhead of creating separate SLAF files.

        Args:
            input_files: List of input file paths
            output_path: Output SLAF directory path
            input_format: Format of input files
        """
        logger.info(
            f"Converting {len(input_files)} {input_format} files to SLAF format..."
        )

        # Check for existing checkpoint with smart resume logic
        checkpoint = self._load_checkpoint_smart(output_path)
        if checkpoint:
            logger.info(f"Found checkpoint: {checkpoint}")
            if checkpoint.get("status") == "completed":
                logger.info(
                    "Multi-file conversion already completed according to checkpoint"
                )
                return
            elif checkpoint.get("status") == "in_progress":
                logger.info("Resuming multi-file conversion from checkpoint...")
            else:
                logger.info("Starting fresh multi-file conversion...")

        # Create output directory (only for local paths)
        self._ensure_directory_exists(output_path)

        # Track source file information
        source_file_info = []
        global_cell_offset = 0
        total_genes = 0
        total_cells = 0

        # Determine starting file and chunk from checkpoint
        start_file_idx = 0
        start_chunk_idx = 0
        if checkpoint and checkpoint.get("status") == "in_progress":
            # Handle both file-level and chunk-level checkpoints
            last_completed_chunk = checkpoint.get("last_completed_chunk", -1)
            if last_completed_chunk == -1:
                # File-level checkpoint: file was fully completed, start next file
                start_file_idx = checkpoint.get("last_completed_file", -1) + 1
                start_chunk_idx = 0
            else:
                # Chunk-level checkpoint: file is partially completed, resume same file
                start_file_idx = checkpoint.get("last_completed_file", -1)
                start_chunk_idx = last_completed_chunk + 1
            global_cell_offset = checkpoint.get("global_cell_offset", 0)

            # Enhanced logging for resume tracking
            logger.info("=" * 60)
            logger.info("🔄 CHECKPOINT RESUME DETECTED")
            logger.info("=" * 60)
            logger.info(
                f"📁 Resuming from FILE {start_file_idx + 1} of {len(input_files)}"
            )
            logger.info(f"📊 Resuming from CHUNK {start_chunk_idx} within that file")
            logger.info(
                f"📈 Last completed file: {checkpoint.get('last_completed_file', -1) + 1}"
            )
            logger.info(
                f"📈 Last completed chunk: {checkpoint.get('last_completed_chunk', -1)}"
            )
            logger.info(f"🔢 Global cell offset: {global_cell_offset:,}")
            if "fragment_count" in checkpoint:
                logger.info(
                    f"🧩 Fragment count: {checkpoint.get('fragment_count', 0):,}"
                )
            if "chunks_since_checkpoint" in checkpoint:
                logger.info(
                    f"⏭️  Chunks since checkpoint: {checkpoint.get('chunks_since_checkpoint', 0)}"
                )
            logger.info("=" * 60)

        # Process each file and add fragments to the same SLAF dataset
        for i, file_path in enumerate(input_files):
            # Skip files if resuming from checkpoint
            if i < start_file_idx:
                logger.info(
                    f"⏭️  Skipping file {i + 1}/{len(input_files)}: {os.path.basename(file_path)} (already processed)"
                )
                continue

            # Enhanced file processing logging
            if i == start_file_idx:
                logger.info("=" * 60)
                logger.info(
                    f"🔄 RESUMING FILE {i + 1}/{len(input_files)}: {os.path.basename(file_path)}"
                )
                logger.info(
                    f"📊 Starting from chunk {start_chunk_idx} within this file"
                )
                logger.info("=" * 60)
            else:
                logger.info(
                    f"📁 Processing file {i + 1}/{len(input_files)}: {os.path.basename(file_path)}"
                )

            try:
                # Use chunked reader to process file directly
                with create_chunked_reader(
                    file_path,
                    chunk_size=self.chunk_size,
                    collection_name=self.tiledb_collection_name,
                ) as reader:
                    logger.info(
                        f"Loaded: {reader.n_obs:,} cells × {reader.n_vars:,} genes"
                    )

                    # Get metadata from reader
                    obs_df = reader.get_obs_metadata()
                    var_df = reader.get_var_metadata()

                    # Ensure cell_id and gene_id columns exist
                    if "cell_id" not in obs_df.columns:
                        obs_df["cell_id"] = reader.obs_names
                    if "gene_id" not in var_df.columns:
                        var_df["gene_id"] = reader.var_names

                    # Add integer IDs with global offset
                    obs_df["cell_integer_id"] = range(
                        global_cell_offset, global_cell_offset + len(obs_df)
                    )
                    var_df["gene_integer_id"] = range(len(var_df))

                    # Add source file information to cell metadata
                    source_file = os.path.basename(file_path)
                    obs_df["source_file"] = source_file

                    # Precompute cell start indices
                    obs_df["cell_start_index"] = self._compute_cell_start_indices(
                        reader, obs_df
                    )

                    # Convert metadata to Lance tables
                    cell_metadata_table = self._create_metadata_table(
                        obs_df, "cell_id", integer_mapping=None
                    )
                    gene_metadata_table = self._create_metadata_table(
                        var_df, "gene_id", integer_mapping=None
                    )

                    # Process expression data in chunks and write directly
                    cells_path = f"{output_path}/cells.lance"
                    genes_path = f"{output_path}/genes.lance"

                    # Write metadata tables (overwrite for first file, append for subsequent)
                    # Skip metadata writing for files that were already completed
                    # Only write metadata for files that haven't been started yet, or for the current file if starting from beginning
                    should_write_metadata = (
                        i > start_file_idx  # Future files - always write metadata
                        or (
                            i == start_file_idx and start_chunk_idx == 0
                        )  # Current file only if starting from beginning
                    )

                    if i == 0:
                        # First file - create new datasets
                        lance.write_dataset(
                            cell_metadata_table,
                            cells_path,
                            mode="overwrite",
                            enable_v2_manifest_paths=self.enable_v2_manifest,
                            data_storage_version="2.1",
                        )
                        lance.write_dataset(
                            gene_metadata_table,
                            genes_path,
                            mode="overwrite",
                            enable_v2_manifest_paths=self.enable_v2_manifest,
                            data_storage_version="2.1",
                        )
                        total_genes = len(var_df)
                    elif should_write_metadata:
                        # Subsequent files - append to existing datasets (only if not resuming from middle)
                        lance.write_dataset(
                            cell_metadata_table,
                            cells_path,
                            mode="append",
                            enable_v2_manifest_paths=self.enable_v2_manifest,
                            data_storage_version="2.1",
                        )

                    # Process expression data in chunks with checkpointing support
                    # If we're resuming from this file, use the checkpoint chunk index
                    # Otherwise start from chunk 0
                    chunk_start_idx = start_chunk_idx if i == start_file_idx else 0
                    self._process_file_chunks_with_checkpoint(
                        reader, output_path, i, chunk_start_idx, global_cell_offset
                    )

                    # Track source file information
                    source_file_info.append(
                        {
                            "file_path": file_path,
                            "file_name": source_file,
                            "n_cells": len(obs_df),
                            "cell_offset": global_cell_offset,
                        }
                    )

                    # Update global cell offset
                    global_cell_offset += len(obs_df)
                    total_cells += len(obs_df)

                    # Save checkpoint after each file (chunk-level checkpointing is handled in _process_file_chunks_with_checkpoint)
                    if self.enable_checkpointing:
                        checkpoint_data = {
                            "status": "in_progress",
                            "last_completed_file": i,
                            "last_completed_chunk": -1,  # -1 indicates file is fully completed
                            "global_cell_offset": global_cell_offset,
                            "total_files": len(input_files),
                            "timestamp": pd.Timestamp.now().isoformat(),
                        }
                        self._save_checkpoint(output_path, checkpoint_data)

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                # Save checkpoint with error status
                if self.enable_checkpointing:
                    checkpoint_data = {
                        "status": "error",
                        "last_completed_file": i - 1,
                        "last_completed_chunk": -1,  # -1 indicates previous file was fully completed
                        "global_cell_offset": global_cell_offset,
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                    self._save_checkpoint(output_path, checkpoint_data)
                # Continue with other files
                continue

        if not source_file_info:
            raise RuntimeError("No files were successfully processed")

        # Create indices if enabled
        if self.create_indices:
            self._create_indices(output_path)

        # Compact dataset if enabled
        if self.compact_after_write:
            self._compact_dataset(output_path)

        # Save config with multi-file information
        self._save_multi_file_config(
            output_path,
            source_file_info,
            None,  # We don't need to pass tables since they're already written
            None,
            None,
        )

        # Clear checkpoint after successful completion
        self._clear_checkpoint(output_path)

        logger.info(
            f"Multi-file conversion complete! Processed {len(source_file_info)} files"
        )
        logger.info(f"Total cells: {total_cells}, Total genes: {total_genes}")

    def append(
        self, input_path: str, existing_slaf_path: str, input_format: str = "auto"
    ):
        """
        Append new data to an existing SLAF dataset.

        This method adds new data to an existing SLAF dataset by:
        1. Validating compatibility with existing dataset
        2. Reading existing dataset metadata
        3. Appending new data with auto-incrementing IDs
        4. Updating configuration with new source file info

        Args:
            input_path: Path to new input file or directory
            existing_slaf_path: Path to existing SLAF dataset
            input_format: Format of input data (auto-detected if not specified)
        """
        logger.info(
            f"Appending data from {input_path} to existing SLAF dataset {existing_slaf_path}"
        )

        # Check if existing SLAF dataset exists
        if not self._path_exists(existing_slaf_path):
            raise FileNotFoundError(
                f"Existing SLAF dataset not found: {existing_slaf_path}"
            )

        # Discover input files
        input_files, detected_format = discover_input_files(input_path)
        if input_format == "auto":
            input_format = detected_format

        # Validate compatibility with existing dataset
        self._validate_append_compatibility(
            input_files, input_format, existing_slaf_path
        )

        # Read existing dataset metadata
        existing_cells_dataset = lance.dataset(
            os.path.join(existing_slaf_path, "cells.lance")
        )
        existing_cells_table = existing_cells_dataset.to_table()
        current_cell_count = len(existing_cells_table)

        # Track source file information
        source_file_info = []
        global_cell_offset = current_cell_count
        total_new_cells = 0

        # Process each new file and append to existing dataset
        for i, file_path in enumerate(input_files):
            logger.info(
                f"Processing file {i + 1}/{len(input_files)}: {os.path.basename(file_path)}"
            )

            try:
                # Use chunked reader to process file directly
                with create_chunked_reader(
                    file_path,
                    chunk_size=self.chunk_size,
                    collection_name=self.tiledb_collection_name,
                ) as reader:
                    logger.info(
                        f"Loaded: {reader.n_obs:,} cells × {reader.n_vars:,} genes"
                    )

                    # Get metadata from reader
                    obs_df = reader.get_obs_metadata()
                    var_df = reader.get_var_metadata()

                    # Ensure cell_id and gene_id columns exist
                    if "cell_id" not in obs_df.columns:
                        obs_df["cell_id"] = reader.obs_names
                    if "gene_id" not in var_df.columns:
                        var_df["gene_id"] = reader.var_names

                    # Add integer IDs with global offset
                    obs_df["cell_integer_id"] = range(
                        global_cell_offset, global_cell_offset + len(obs_df)
                    )

                    # Add source file information to cell metadata
                    source_file = os.path.basename(file_path)
                    obs_df["source_file"] = source_file

                    # Check if existing dataset has source_file column
                    existing_cells_dataset = lance.dataset(
                        os.path.join(existing_slaf_path, "cells.lance")
                    )
                    existing_cells_table = existing_cells_dataset.to_table()
                    existing_columns = set(existing_cells_table.column_names)

                    # If existing dataset doesn't have source_file column, add it to existing data
                    if "source_file" not in existing_columns:
                        logger.info("Adding source_file column to existing dataset...")
                        # Read existing cells data
                        existing_cells_df = existing_cells_table.to_pandas()
                        existing_cells_df["source_file"] = (
                            "original_data"  # Default source for existing data
                        )

                        # Recreate the cells dataset with source_file column
                        updated_cells_table = pa.table(existing_cells_df)
                        lance.write_dataset(
                            updated_cells_table,
                            os.path.join(existing_slaf_path, "cells.lance"),
                            mode="overwrite",
                            enable_v2_manifest_paths=self.enable_v2_manifest,
                            data_storage_version="2.1",
                        )
                        logger.info("✓ Added source_file column to existing dataset")

                    # Precompute cell start indices
                    obs_df["cell_start_index"] = self._compute_cell_start_indices(
                        reader, obs_df
                    )

                    # Convert metadata to Lance tables
                    cell_metadata_table = self._create_metadata_table(
                        obs_df, "cell_id", integer_mapping=None
                    )

                    # Append to existing cells dataset
                    cells_path = os.path.join(existing_slaf_path, "cells.lance")
                    lance.write_dataset(
                        cell_metadata_table,
                        cells_path,
                        mode="append",
                        enable_v2_manifest_paths=self.enable_v2_manifest,
                        data_storage_version="2.1",
                    )

                    # Process expression data in chunks with checkpointing support
                    # Use the same checkpointing infrastructure as convert method
                    self._process_file_chunks_with_checkpoint(
                        reader, existing_slaf_path, i, 0, global_cell_offset
                    )

                    # Track source file information
                    source_file_info.append(
                        {
                            "file_path": file_path,
                            "file_name": source_file,
                            "n_cells": len(obs_df),
                            "cell_offset": global_cell_offset,
                        }
                    )

                    # Update global cell offset
                    global_cell_offset += len(obs_df)
                    total_new_cells += len(obs_df)

                    # Save checkpoint after each file (for append operations)
                    if self.enable_checkpointing:
                        checkpoint_data = {
                            "status": "in_progress",
                            "last_completed_file": i,
                            "last_completed_chunk": -1,  # -1 indicates file is fully completed
                            "global_cell_offset": global_cell_offset,
                            "operation": "append",  # Mark this as an append operation
                            "timestamp": pd.Timestamp.now().isoformat(),
                        }
                        self._save_checkpoint(existing_slaf_path, checkpoint_data)

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                # Save checkpoint with error status
                if self.enable_checkpointing:
                    checkpoint_data = {
                        "status": "error",
                        "last_completed_file": i - 1,
                        "last_completed_chunk": -1,  # -1 indicates previous file was fully completed
                        "global_cell_offset": global_cell_offset,
                        "operation": "append",  # Mark this as an append operation
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                    self._save_checkpoint(existing_slaf_path, checkpoint_data)
                # Continue with other files
                continue

        if not source_file_info:
            raise RuntimeError("No files were successfully processed")

        # Update configuration with new source file information
        self._update_config_with_append(
            existing_slaf_path, source_file_info, total_new_cells
        )

        # Clear checkpoint after successful completion
        self._clear_checkpoint(existing_slaf_path)

        logger.info(
            f"Append complete! Added {total_new_cells} cells from {len(source_file_info)} files"
        )
        logger.info(f"Total cells in dataset: {current_cell_count + total_new_cells}")

    def _validate_append_compatibility(
        self, input_files: list[str], input_format: str, existing_slaf_path: str
    ):
        """Validate that new files are compatible with existing SLAF dataset."""
        logger.info("Validating compatibility with existing SLAF dataset...")

        # Load existing dataset metadata
        existing_cells_dataset = lance.dataset(f"{existing_slaf_path}/cells.lance")
        existing_cells_table = existing_cells_dataset.to_table()
        existing_genes_dataset = lance.dataset(f"{existing_slaf_path}/genes.lance")
        existing_genes_table = existing_genes_dataset.to_table()

        # Get existing gene set and cell metadata schema
        existing_genes = set(existing_genes_table.column("gene_id").to_numpy())
        existing_cell_columns = set(existing_cells_table.column_names)

        # Validate new files against existing dataset
        for file_path in input_files:
            try:
                # Extract schema from new file
                genes, cells, value_type = self._extract_schema_info(
                    file_path, input_format
                )

                # Check gene compatibility
                if genes != existing_genes:
                    missing_genes = existing_genes - genes
                    extra_genes = genes - existing_genes
                    error_msg = f"File {os.path.basename(file_path)} is incompatible with existing dataset:"
                    if missing_genes:
                        error_msg += f"\n  Missing genes: {sorted(missing_genes)[:5]}{'...' if len(missing_genes) > 5 else ''}"
                    if extra_genes:
                        error_msg += f"\n  Extra genes: {sorted(extra_genes)[:5]}{'...' if len(extra_genes) > 5 else ''}"
                    raise ValueError(error_msg)

                # Check cell metadata schema compatibility
                # Exclude columns that are added during SLAF conversion
                slaF_added_columns = {
                    "cell_id",
                    "cell_integer_id",
                    "cell_start_index",
                    "source_file",
                }
                new_cell_columns = cells - slaF_added_columns
                existing_cell_columns_no_slaf = (
                    existing_cell_columns - slaF_added_columns
                )

                if new_cell_columns != existing_cell_columns_no_slaf:
                    missing_cols = existing_cell_columns_no_slaf - new_cell_columns
                    extra_cols = new_cell_columns - existing_cell_columns_no_slaf
                    error_msg = f"File {os.path.basename(file_path)} has incompatible cell metadata schema:"
                    if missing_cols:
                        error_msg += f"\n  Missing columns: {sorted(missing_cols)}"
                    if extra_cols:
                        error_msg += f"\n  Extra columns: {sorted(extra_cols)}"
                    raise ValueError(error_msg)

            except Exception as e:
                raise ValueError(
                    f"Validation failed for {os.path.basename(file_path)}: {e}"
                ) from e

        logger.info("✓ All files are compatible with existing dataset")

    def _load_existing_config(self, existing_slaf_path: str) -> dict:
        """Load existing SLAF configuration."""
        config_path = f"{existing_slaf_path}/config.json"
        if not self._path_exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with self._open_file(config_path) as f:
            return json.load(f)

    def _update_config_with_append(
        self, existing_slaf_path: str, source_file_info: list, total_new_cells: int
    ):
        """Update configuration with new source file information."""
        config_path = f"{existing_slaf_path}/config.json"
        config = self._load_existing_config(existing_slaf_path)

        # Update cell count
        config["n_cells"] += total_new_cells
        config["array_shape"][0] = config["n_cells"]

        # Update multi_file information
        if "multi_file" not in config:
            config["multi_file"] = {
                "source_files": [],
                "total_files": 0,
                "total_cells_from_files": 0,
            }

        # Add new source files
        config["multi_file"]["source_files"].extend(source_file_info)
        config["multi_file"]["total_files"] = len(config["multi_file"]["source_files"])
        config["multi_file"]["total_cells_from_files"] = sum(
            info["n_cells"] for info in config["multi_file"]["source_files"]
        )

        # Update created_at timestamp
        config["last_updated"] = pd.Timestamp.now().isoformat()

        # Save updated config
        with self._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(
            f"Updated configuration with {len(source_file_info)} new source files"
        )

    def _extract_schema_info(
        self, file_path: str, format_type: str
    ) -> tuple[set[str], set[str], str]:
        """Extract schema information from a file for compatibility checking."""
        if format_type == "h5ad":
            return self._extract_h5ad_schema(file_path)
        elif format_type == "10x_mtx":
            return self._extract_10x_mtx_schema(file_path)
        elif format_type == "10x_h5":
            return self._extract_10x_h5_schema(file_path)
        elif format_type == "tiledb":
            return self._extract_tiledb_schema(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _extract_h5ad_schema(self, file_path: str) -> tuple[set[str], set[str], str]:
        """Extract schema information from h5ad file."""
        import scanpy as sc

        # Read in backed mode for efficiency
        adata = sc.read_h5ad(file_path, backed="r")

        # Get gene IDs
        gene_ids = set(adata.var_names)

        # Get cell metadata columns
        cell_columns = set(adata.obs.columns)

        # Determine value type from expression data
        if hasattr(adata.X, "data"):
            sample_data = adata.X.data[:1000]  # Sample first 1000 values
        else:
            # For backed mode, try to get a sample
            try:
                sample_data = (
                    adata.X[:100, :100].data
                    if hasattr(adata.X, "data")
                    else adata.X[:100, :100].toarray().flatten()
                )
            except Exception:
                sample_data = np.array([0])  # Fallback

        # Determine value type
        if np.issubdtype(sample_data.dtype, np.integer):
            value_type = "uint16"
        else:
            value_type = "float32"

        adata.file.close()
        return gene_ids, cell_columns, value_type

    def _extract_10x_mtx_schema(self, file_path: str) -> tuple[set[str], set[str], str]:
        """Extract schema information from 10x MTX directory."""
        import scanpy as sc

        # Read MTX files
        adata = sc.read_10x_mtx(file_path)

        # Get gene IDs
        gene_ids = set(adata.var_names)

        # Get cell metadata columns
        cell_columns = set(adata.obs.columns)

        # 10x MTX typically has integer counts
        value_type = "uint16"

        return gene_ids, cell_columns, value_type

    def _extract_10x_h5_schema(self, file_path: str) -> tuple[set[str], set[str], str]:
        """Extract schema information from 10x H5 file."""
        import scanpy as sc

        try:
            # Try to read as 10x H5 first
            adata = sc.read_10x_h5(file_path, genome="X")
        except Exception:
            # Fall back to regular h5ad
            adata = sc.read_h5ad(file_path, backed="r")

        # Get gene IDs
        gene_ids = set(adata.var_names)

        # Get cell metadata columns
        cell_columns = set(adata.obs.columns)

        # 10x H5 typically has integer counts
        value_type = "uint16"

        if hasattr(adata, "file"):
            adata.file.close()

        return gene_ids, cell_columns, value_type

    def _extract_tiledb_schema(self, file_path: str) -> tuple[set[str], set[str], str]:
        """Extract schema information from TileDB SOMA file."""
        import tiledbsoma as soma

        # Open TileDB SOMA experiment
        with soma.open(file_path) as exp:
            # Get measurement collection
            ms = exp.ms[self.tiledb_collection_name]

            # Get gene IDs from var
            var_df = ms.var.read().concat().to_pandas()
            gene_ids = set(var_df.index)

            # Get cell metadata columns from obs
            obs_df = ms.obs.read().concat().to_pandas()
            cell_columns = set(obs_df.columns)

            # TileDB SOMA typically has float32 values
            value_type = "float32"

            return gene_ids, cell_columns, value_type

    def _convert_anndata(self, adata, output_path: str):
        """Internal method to convert AnnData object to SLAF format"""
        # Create output directory (only for local paths)
        self._ensure_directory_exists(output_path)

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

        # Precompute cell start indices for fast cell-based queries
        logger.info("Precomputing cell start indices...")
        obs_df["cell_start_index"] = self._compute_cell_start_indices_anndata(
            adata, obs_df
        )

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

        self._write_lance_tables(output_path, table_configs)

        # Compact dataset for optimal storage (only if enabled)
        if self.compact_after_write:
            self._compact_dataset(output_path)

        # Save config
        self._save_config(output_path, adata.shape)
        logger.info(f"Conversion complete! Saved to {output_path}")

    def _convert_chunked(self, h5ad_path: str, output_path: str):
        """Convert h5ad file using chunked processing with checkpointing support"""
        logger.info(f"Processing in chunks of {self.chunk_size} cells...")

        # Check for existing checkpoint with smart resume logic
        checkpoint = self._load_checkpoint_smart(output_path)
        if checkpoint:
            logger.info(f"Found checkpoint: {checkpoint}")
            if checkpoint.get("status") == "completed":
                logger.info("Conversion already completed according to checkpoint")
                return
            elif checkpoint.get("status") == "in_progress":
                logger.info("Resuming from checkpoint...")
                # Resume logic will be handled in _process_expression
            else:
                logger.info("Starting fresh conversion...")

        # First, create a temporary reader to determine the value type
        with create_chunked_reader(
            h5ad_path,
            chunk_size=self.chunk_size,
            collection_name=self.tiledb_collection_name,
        ) as temp_reader:
            # Validate optimized data types and determine value type
            validation_result, value_type = self._validate_optimized_dtypes(temp_reader)
            if not validation_result:
                self.use_optimized_dtypes = False

        # Now create the reader with the correct value type
        with create_chunked_reader(
            h5ad_path,
            chunk_size=self.chunk_size,
            value_type=value_type,
            collection_name=self.tiledb_collection_name,
        ) as reader:
            logger.info(f"Loaded: {reader.n_obs:,} cells × {reader.n_vars:,} genes")

            # Create output directory (only for local paths)
            self._ensure_directory_exists(output_path)

            # Write metadata tables efficiently (without loading everything into memory)
            # Only write metadata if not resuming from checkpoint
            if not checkpoint or checkpoint.get("status") != "in_progress":
                self._write_metadata_efficiently(reader, output_path)

            # Process expression data with checkpointing
            self._process_expression_with_checkpoint(
                reader, output_path, value_type, checkpoint
            )

            # Create indices (if enabled)
            if self.create_indices:
                self._create_indices(output_path)

            # Compact dataset for optimal storage (only if enabled)
            if self.compact_after_write:
                self._compact_dataset(output_path)

            # Save config and clear checkpoint
            self._save_config(output_path, (reader.n_obs, reader.n_vars))
            self._clear_checkpoint(output_path)
            logger.info(f"Conversion complete! Saved to {output_path}")

    def _write_metadata_efficiently(self, reader, output_path: str):
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

        # Precompute cell start indices for fast cell-based queries
        logger.info("Precomputing cell start indices...")
        obs_df["cell_start_index"] = self._compute_cell_start_indices(reader, obs_df)

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

        # Write metadata tables
        lance.write_dataset(
            cell_metadata_table,
            f"{output_path}/cells.lance",
            mode="overwrite",
            enable_v2_manifest_paths=self.enable_v2_manifest,
            data_storage_version="2.1",
        )
        lance.write_dataset(
            gene_metadata_table,
            f"{output_path}/genes.lance",
            mode="overwrite",
            enable_v2_manifest_paths=self.enable_v2_manifest,
            data_storage_version="2.1",
        )

        logger.info("Metadata tables written!")

    def _process_expression(self, reader, output_path: str, value_type="uint16"):
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
        expression_path = f"{output_path}/expression.lance"
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

        # Create initial Lance dataset (using max_rows_per_file for large fragments)
        lance.write_dataset(
            empty_table,
            expression_path,
            mode="overwrite",
            schema=schema,
            max_rows_per_file=10000000,  # 10M rows per file to avoid memory issues
            enable_v2_manifest_paths=self.enable_v2_manifest,
            data_storage_version="2.1",
        )

        # Process chunks sequentially
        logger.info("Processing chunks sequentially...")
        import time

        from tqdm import tqdm

        processing_start_time = time.time()

        for _chunk_idx, (chunk_table, _obs_slice) in enumerate(
            tqdm(
                reader.iter_chunks(chunk_size=self.chunk_size),
                total=total_chunks,
                desc="Processing chunks",
                unit="chunk",
            )
        ):
            # Process chunk (data type conversion and string ID addition if needed)
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

            # Write chunk to Lance dataset (using max_rows_per_file for large fragments)
            lance.write_dataset(
                chunk_table,
                expression_path,
                mode="append",
                max_rows_per_file=10000000,  # 10M rows per file to avoid memory issues
                enable_v2_manifest_paths=self.enable_v2_manifest,
                data_storage_version="2.1",
            )

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

        # Calculate and log overall statistics
        total_processing_time = time.time() - processing_start_time

        logger.info(
            f"Expression data processing complete! "
            f"Processed {total_chunks} chunks in {total_processing_time:.1f}s "
            f"({total_processing_time / total_chunks:.2f}s per chunk average)"
        )

    def _process_expression_with_checkpoint(
        self, reader, output_path: str, value_type="uint16", checkpoint=None
    ):
        """Process expression data with checkpointing support"""
        logger.info("Processing expression data with checkpointing...")

        # Calculate total chunks
        total_chunks = (reader.n_obs + self.chunk_size - 1) // self.chunk_size
        logger.info(
            f"Processing {total_chunks} chunks with chunk size {self.chunk_size:,}..."
        )

        # Determine starting chunk from checkpoint
        start_chunk = 0
        if checkpoint and checkpoint.get("status") == "in_progress":
            start_chunk = checkpoint.get("last_completed_chunk", 0) + 1
            logger.info(
                f"Resuming from chunk {start_chunk} (last completed: {checkpoint.get('last_completed_chunk', -1)})"
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
        expression_path = f"{output_path}/expression.lance"
        schema = self._get_expression_schema(value_type)

        # Create empty dataset first (only if not resuming)
        if start_chunk == 0:
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

            # Create initial Lance dataset (using max_rows_per_file for large fragments)
            lance.write_dataset(
                empty_table,
                expression_path,
                mode="overwrite",
                schema=schema,
                max_rows_per_file=10000000,  # 10M rows per file to avoid memory issues
                enable_v2_manifest_paths=self.enable_v2_manifest,
                data_storage_version="2.1",
            )

        # Process chunks sequentially with checkpointing
        logger.info("Processing chunks sequentially with checkpointing...")
        import time

        from tqdm import tqdm

        processing_start_time = time.time()

        # Create iterator for chunks
        chunk_iterator = reader.iter_chunks(chunk_size=self.chunk_size)

        # Skip chunks if resuming
        for _ in range(start_chunk):
            try:
                next(chunk_iterator)
            except StopIteration:
                logger.warning(
                    f"Tried to skip {start_chunk} chunks but iterator ended early"
                )
                break

        # Process remaining chunks
        for chunk_idx in tqdm(
            range(start_chunk, total_chunks),
            desc="Processing chunks",
            unit="chunk",
            initial=start_chunk,
            total=total_chunks,
        ):
            try:
                # Get next chunk
                chunk_table, _obs_slice = next(chunk_iterator)

                # Process chunk (data type conversion and string ID addition if needed)
                if not self.use_optimized_dtypes:
                    # Convert from optimized dtypes to standard dtypes
                    cell_integer_ids = (
                        chunk_table.column("cell_integer_id")
                        .to_numpy()
                        .astype(np.int32)
                    )
                    gene_integer_ids = (
                        chunk_table.column("gene_integer_id")
                        .to_numpy()
                        .astype(np.int32)
                    )
                    values = chunk_table.column("value").to_numpy().astype(np.float32)

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": pa.array(gene_integer_ids),
                            "value": pa.array(values),
                        }
                    )

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

                # Save checkpoint BEFORE writing chunk to avoid duplication
                if self.enable_checkpointing:
                    should_save_checkpoint = (
                        chunk_idx % 10 == 0  # Every 10 chunks
                        or chunk_idx == total_chunks - 1  # Last chunk
                    )

                    if should_save_checkpoint:
                        checkpoint_data = {
                            "status": "in_progress",
                            "last_completed_chunk": chunk_idx,
                            "total_chunks": total_chunks,
                            "chunk_size": self.chunk_size,
                            "timestamp": pd.Timestamp.now().isoformat(),
                        }
                        self._save_checkpoint(output_path, checkpoint_data)

                # Write chunk to Lance dataset (using max_rows_per_file for large fragments)
                lance.write_dataset(
                    chunk_table,
                    expression_path,
                    mode="append",
                    max_rows_per_file=10000000,  # 10M rows per file to avoid memory issues
                    enable_v2_manifest_paths=self.enable_v2_manifest,
                    data_storage_version="2.1",
                )

            except StopIteration:
                logger.warning(f"Chunk iterator ended at chunk {chunk_idx}")
                break
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                # Save checkpoint with error status
                if self.enable_checkpointing:
                    checkpoint_data = {
                        "status": "error",
                        "last_completed_chunk": chunk_idx - 1,
                        "total_chunks": total_chunks,
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                    self._save_checkpoint(output_path, checkpoint_data)
                raise

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

        # Calculate and log overall statistics
        total_processing_time = time.time() - processing_start_time

        logger.info(
            f"Expression data processing complete! "
            f"Processed {total_chunks} chunks in {total_processing_time:.1f}s "
            f"({total_processing_time / total_chunks:.2f}s per chunk average)"
        )

    def _process_file_chunks_with_checkpoint(
        self,
        reader,
        output_path: str,
        file_idx: int,
        start_chunk_idx: int,
        global_cell_offset: int,
    ):
        """Process chunks for a single file with checkpointing support"""
        expression_path = f"{output_path}/expression.lance"

        # Calculate total chunks for this file
        total_chunks = (reader.n_obs + self.chunk_size - 1) // self.chunk_size

        logger.info(
            f"📊 Processing file {file_idx + 1} chunks: {start_chunk_idx}/{total_chunks}"
        )
        if start_chunk_idx > 0:
            logger.info(
                f"🔄 Resuming from chunk {start_chunk_idx} (chunks 0-{start_chunk_idx - 1} already processed)"
            )

        # Create iterator for chunks
        chunk_iterator = reader.iter_chunks(chunk_size=self.chunk_size)

        # Skip chunks if resuming from checkpoint
        if start_chunk_idx > 0:
            logger.info(f"⏭️  Skipping {start_chunk_idx} already-processed chunks...")
            for _ in range(start_chunk_idx):
                try:
                    next(chunk_iterator)
                except StopIteration:
                    logger.warning(
                        f"Tried to skip {start_chunk_idx} chunks but iterator ended early"
                    )
                    break

        # Process remaining chunks
        for chunk_idx in range(start_chunk_idx, total_chunks):
            try:
                # Get next chunk
                chunk_table, _obs_slice = next(chunk_iterator)

                # Log progress every 10 chunks or on first/last chunk
                if (
                    chunk_idx % 10 == 0
                    or chunk_idx == start_chunk_idx
                    or chunk_idx == total_chunks - 1
                ):
                    logger.info(
                        f"📈 Processing chunk {chunk_idx + 1}/{total_chunks} (file {file_idx + 1})"
                    )

                # Adjust cell integer IDs with global offset
                cell_integer_ids = (
                    chunk_table.column("cell_integer_id").to_numpy()
                    + global_cell_offset
                )

                # Create adjusted chunk table
                if self.optimize_storage:
                    # Only store integer IDs for maximum storage efficiency
                    adjusted_chunk = pa.table(
                        {
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": chunk_table.column("gene_integer_id"),
                            "value": chunk_table.column("value"),
                        }
                    )
                else:
                    # Store both string and integer IDs for compatibility
                    adjusted_chunk = pa.table(
                        {
                            "cell_id": chunk_table.column("cell_id"),
                            "gene_id": chunk_table.column("gene_id"),
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": chunk_table.column("gene_integer_id"),
                            "value": chunk_table.column("value"),
                        }
                    )

                # Save checkpoint BEFORE writing chunk to avoid duplication
                # This ensures we know exactly what was written if a failure occurs
                if self.enable_checkpointing:
                    should_save_checkpoint = (
                        chunk_idx % 10 == 0  # Every 10 chunks
                        or chunk_idx == total_chunks - 1  # Last chunk of file
                    )

                    if should_save_checkpoint:
                        checkpoint_data = {
                            "status": "in_progress",
                            "last_completed_file": file_idx,  # Current file being processed
                            "last_completed_chunk": chunk_idx,  # Current chunk being processed
                            "global_cell_offset": global_cell_offset,
                            "timestamp": pd.Timestamp.now().isoformat(),
                        }
                        self._save_checkpoint(output_path, checkpoint_data)

                # Write chunk directly to expression dataset
                # Check if this is the first chunk of the first file in a new dataset
                is_first_chunk_of_new_dataset = file_idx == 0 and chunk_idx == 0

                # Check if the expression dataset already exists (for append operations)
                expression_dataset_exists = self._path_exists(expression_path)
                if is_first_chunk_of_new_dataset and not expression_dataset_exists:
                    # First chunk of first file in a new dataset - create new dataset
                    lance.write_dataset(
                        adjusted_chunk,
                        expression_path,
                        mode="overwrite",
                        max_rows_per_file=10000000,
                        enable_v2_manifest_paths=self.enable_v2_manifest,
                        data_storage_version="2.1",
                    )
                else:
                    # Append to existing dataset (either subsequent chunks or append operations)
                    lance.write_dataset(
                        adjusted_chunk,
                        expression_path,
                        mode="append",
                        max_rows_per_file=10000000,
                        enable_v2_manifest_paths=self.enable_v2_manifest,
                        data_storage_version="2.1",
                    )

            except StopIteration:
                logger.warning(f"Chunk iterator ended at chunk {chunk_idx}")
                break
            except Exception as e:
                logger.error(
                    f"Error processing chunk {chunk_idx} in file {file_idx + 1}: {e}"
                )
                # Save checkpoint with error status
                if self.enable_checkpointing:
                    checkpoint_data = {
                        "status": "error",
                        "last_completed_file": file_idx,
                        "last_completed_chunk": chunk_idx - 1,
                        "global_cell_offset": global_cell_offset,
                        "error": str(e),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                    self._save_checkpoint(output_path, checkpoint_data)
                raise

    def _validate_optimized_dtypes(self, reader):
        """Validate that data fits in optimized data types and determine appropriate value type"""
        if not self.use_optimized_dtypes:
            return True, "float32"

        logger.info("Validating data fits in optimized data types...")

        # Check if gene count fits in uint16
        if reader.n_vars > 65535:
            logger.info(f"Warning: {reader.n_vars:,} genes exceeds uint16 limit")
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
        # Handle TileDB readers specially
        if hasattr(reader, "_experiment") and reader._experiment is not None:
            # TileDB reader - check original data directly
            try:
                X = reader._experiment.ms[reader.collection_name].X["data"]
                # Sample some original data
                sample_size = min(10000, X.shape[0])
                sample_data = X.read((slice(0, sample_size),)).tables().concat()
                sample_values = sample_data.column("soma_data").to_numpy()

                # Check if original data is integer or float
                is_integer = np.issubdtype(sample_values.dtype, np.integer)
                max_value = np.max(sample_values)
                min_value = np.min(sample_values)

                if is_integer and max_value <= 65535 and min_value >= 0:
                    logger.info(
                        f"Original integer expression values fit in uint16 range: [{min_value}, {max_value}]"
                    )
                    logger.info("Using uint16 for integer count data")
                    return True, "uint16"
                elif not is_integer:
                    # Check if float data contains only integer values
                    rounded_data = np.round(sample_values)
                    is_integer_values = np.allclose(
                        sample_values, rounded_data, rtol=1e-10
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
            except Exception as e:
                logger.warning(f"Error checking TileDB original data: {e}")
                # Fall through to fallback logic
        elif hasattr(reader, "file") and reader.file is not None:
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
            logger.info(f"Warning: {adata.n_vars:,} genes exceeds uint16 limit")
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
        # Handle backed mode where X might not have .data attribute
        if hasattr(adata.X, "data"):
            sample_data = adata.X.data[:100000]
        else:
            # For backed mode, convert to COO and get data
            if hasattr(adata.X, "tocoo"):
                coo = adata.X.tocoo()
            else:
                try:
                    coo = sparse.coo_matrix(adata.X)
                except ValueError:
                    # If dtype is not supported, try to convert to a supported type
                    logger.warning(
                        "adata.X has unsupported dtype, attempting conversion"
                    )
                    try:
                        coo = sparse.coo_matrix(adata.X.astype(np.float32))
                    except Exception:
                        logger.warning(
                            "Could not convert adata.X, using fallback method"
                        )
                        # For backed mode, we might need to load a small sample
                        sample_size = min(1000, adata.n_obs)
                        sample_adata = adata[:sample_size, :]
                        if sparse.issparse(sample_adata.X):
                            coo = sample_adata.X.tocoo()
                        else:
                            coo = sparse.coo_matrix(sample_adata.X)
            sample_data = coo.data[:100000]

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
                f"Warning: Integer values range [{min_value}, {max_value}] exceeds uint16 range"
            )
            logger.info("Falling back to float32")
            return False, "float32"

        logger.info(
            "AnnData object's expression data validation passed - using optimized data types"
        )
        return True, "uint16"  # Default to uint16 for integer data

    def _compact_dataset(self, output_path: str):
        """Compact the dataset to optimize storage after writing"""
        logger.info("Compacting dataset for optimal storage...")

        try:
            # Compact expression table
            expression_path = f"{output_path}/expression.lance"
            if self._path_exists(expression_path):
                logger.info("  Compacting expression table...")
                dataset = lance.dataset(expression_path)
                dataset.optimize.compact_files(
                    target_rows_per_fragment=1024 * 1024
                )  # 1M rows per fragment
                logger.info("  Expression table compacted!")
            else:
                logger.warning("  Expression table not found, skipping compaction")

            # Compact metadata tables
            for table_name in ["cells", "genes"]:
                table_path = f"{output_path}/{table_name}.lance"
                if self._path_exists(table_path):
                    logger.info(f"  Compacting {table_name} table...")
                    dataset = lance.dataset(table_path)
                    dataset.optimize.compact_files(
                        target_rows_per_fragment=100000
                    )  # 100K rows per fragment for metadata
                    logger.info(f"  {table_name} table compacted!")
                else:
                    logger.warning(
                        f"  {table_name} table not found, skipping compaction"
                    )

            logger.info("Dataset compaction complete!")
        except Exception as e:
            logger.error(f"Error during dataset compaction: {e}")
            logger.error(
                "Dataset may be in an inconsistent state. Consider recreating without compaction."
            )
            raise

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
        # Handle backed mode where tocoo() might not be available
        if hasattr(sparse_matrix, "tocoo"):
            coo_matrix = sparse_matrix.tocoo()
        else:
            # For backed mode, convert to COO using scipy
            try:
                coo_matrix = sparse.coo_matrix(sparse_matrix)
            except (ValueError, AttributeError):
                # If dtype is not supported or object doesn't have expected methods
                logger.warning(
                    "sparse_matrix has unsupported dtype or missing methods, using fallback"
                )
                # For CSRDataset objects, we need to handle them differently
                # Let's try to get the data in a way that works for backed mode
                try:
                    # Try to get the data directly from the backed object
                    if hasattr(sparse_matrix, "data"):
                        data = sparse_matrix.data
                        if hasattr(sparse_matrix, "indices") and hasattr(
                            sparse_matrix, "indptr"
                        ):
                            # It's a CSR matrix
                            indices = sparse_matrix.indices
                            indptr = sparse_matrix.indptr
                            # Convert to COO
                            row_indices = []
                            col_indices = []
                            for i in range(len(indptr) - 1):
                                start = indptr[i]
                                end = indptr[i + 1]
                                row_indices.extend([i] * (end - start))
                                col_indices.extend(indices[start:end])
                            coo_matrix = sparse.coo_matrix(
                                (data, (row_indices, col_indices)),
                                shape=sparse_matrix.shape,
                            )
                        else:
                            # Fallback: try to convert to numpy array first
                            try:
                                dense_array = np.array(sparse_matrix)
                                coo_matrix = sparse.coo_matrix(dense_array)
                            except Exception:
                                # Last resort: create empty COO matrix
                                logger.warning(
                                    "Could not convert sparse_matrix, creating empty COO matrix"
                                )
                                coo_matrix = sparse.coo_matrix(sparse_matrix.shape)
                    else:
                        # Try to convert to numpy array first
                        try:
                            dense_array = np.array(sparse_matrix)
                            coo_matrix = sparse.coo_matrix(dense_array)
                        except Exception:
                            # Last resort: create empty COO matrix
                            logger.warning(
                                "Could not convert sparse_matrix, creating empty COO matrix"
                            )
                            coo_matrix = sparse.coo_matrix(sparse_matrix.shape)
                except Exception as e:
                    logger.warning(f"All conversion methods failed: {e}")
                    # Last resort: create empty COO matrix
                    coo_matrix = sparse.coo_matrix(sparse_matrix.shape)

        logger.info(f"Processing {len(coo_matrix.data):,} non-zero elements...")

        # Create integer ID arrays for efficient range queries
        if self.use_optimized_dtypes:
            cell_integer_id_array = coo_matrix.row.astype(np.uint32)
            gene_integer_id_array = coo_matrix.col.astype(np.uint16)
            # Convert values based on the determined type
            if value_type == "uint16":
                value_array = coo_matrix.data.astype(np.uint16)
                value_dtype = np.uint16
                value_pa_type = pa.uint16()
            elif value_type == "float32":
                value_array = coo_matrix.data.astype(np.float32)
                value_dtype = np.float32
                value_pa_type = pa.float32()
            else:
                raise ValueError(f"Unsupported value type: {value_type}")
        else:
            cell_integer_id_array = coo_matrix.row.astype(np.int32)
            gene_integer_id_array = coo_matrix.col.astype(np.int32)
            # Expression values - use the determined type
            if value_type == "uint16":
                value_array = coo_matrix.data.astype(np.uint16)
                value_dtype = np.uint16
                value_pa_type = pa.uint16()
            elif value_type == "float32":
                value_array = coo_matrix.data.astype(np.float32)
                value_dtype = np.float32
                value_pa_type = pa.float32()
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
                            value_array.astype(value_dtype), type=value_pa_type
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
                            value_array.astype(value_dtype), type=value_pa_type
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
                            value_array.astype(value_dtype), type=value_pa_type
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
                            value_array.astype(value_dtype), type=value_pa_type
                        ),
                    }
                )

        # Validate schema
        if self.optimize_storage:
            if self.use_optimized_dtypes:
                expected_types = {
                    "cell_integer_id": pa.uint32(),
                    "gene_integer_id": pa.uint16(),
                    "value": value_pa_type,
                }
            else:
                expected_types = {
                    "cell_integer_id": pa.int32(),
                    "gene_integer_id": pa.int32(),
                    "value": value_pa_type,
                }
        else:
            if self.use_optimized_dtypes:
                expected_types = {
                    "cell_id": pa.string(),
                    "gene_id": pa.string(),
                    "cell_integer_id": pa.uint32(),
                    "gene_integer_id": pa.uint16(),
                    "value": value_pa_type,
                }
            else:
                expected_types = {
                    "cell_id": pa.string(),
                    "gene_id": pa.string(),
                    "cell_integer_id": pa.int32(),
                    "gene_integer_id": pa.int32(),
                    "value": value_pa_type,
                }

        # Validate schema
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

        return table

    def _write_lance_tables(
        self, output_path: str, table_configs: list[tuple[str, pa.Table]]
    ):
        """Write multiple Lance tables with consistent naming"""
        for table_name, table in table_configs:
            table_path = f"{output_path}/{table_name}.lance"

            # Write table with basic settings (using max_rows_per_file for large fragments)
            if table_name == "expression":
                lance.write_dataset(
                    table,
                    table_path,
                    max_rows_per_file=10000000,  # 10M rows per file to avoid memory issues
                    enable_v2_manifest_paths=self.enable_v2_manifest,
                    data_storage_version="2.1",
                )
            else:
                lance.write_dataset(
                    table,
                    table_path,
                    enable_v2_manifest_paths=self.enable_v2_manifest,
                    data_storage_version="2.1",
                )

        # Create indices after all tables are written (if enabled)
        if self.create_indices:
            self._create_indices(output_path)

    def _create_indices(self, output_path: str):
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
            table_path = f"{output_path}/{table_name}.lance"
            if self._path_exists(table_path):
                dataset = lance.dataset(table_path)
                schema = dataset.schema

                for column in desired_columns:
                    if column in schema.names:
                        logger.info(f"  Creating index on {table_name}.{column}")
                        dataset.create_scalar_index(column, "BTREE")

        logger.info("Index creation complete!")

    def _compute_cell_start_indices(self, reader, obs_df: pd.DataFrame) -> list[int]:
        """Compute cell start indices during metadata creation"""
        # Check for existing n_genes or gene_count column
        if "n_genes" in obs_df.columns:
            logger.info("Using existing n_genes column for cell start indices")
            gene_counts = obs_df["n_genes"].to_numpy()
        elif "gene_count" in obs_df.columns:
            logger.info("Using existing gene_count column for cell start indices")
            gene_counts = obs_df["gene_count"].to_numpy()
        elif "n_genes_by_counts" in obs_df.columns:
            logger.info(
                "Using existing n_genes_by_counts column for cell start indices"
            )
            gene_counts = obs_df["n_genes_by_counts"].to_numpy()
        else:
            # Calculate from expression data
            logger.info("Calculating gene counts from expression data...")

            # Collect all chunk counts
            all_chunk_gene_counts = []
            for chunk_table, _obs_slice in reader.iter_chunks(
                chunk_size=self.chunk_size
            ):
                # Count genes per cell in this chunk using Polars groupby
                chunk_df = pl.from_arrow(chunk_table)
                assert isinstance(chunk_df, pl.DataFrame)
                if len(chunk_df) > 0:
                    chunk_gene_counts = chunk_df.group_by("cell_integer_id").agg(
                        pl.len().alias("count")
                    )
                    all_chunk_gene_counts.append(chunk_gene_counts)

            # Concatenate and aggregate all chunk counts
            if all_chunk_gene_counts:
                combined_gene_counts = pl.concat(all_chunk_gene_counts)
                final_gene_counts = combined_gene_counts.group_by(
                    "cell_integer_id"
                ).agg(pl.sum("count").alias("count"))

                # Create a complete gene counts array for all cells
                # Initialize with zeros for all cells
                gene_counts = np.zeros(len(obs_df), dtype=np.int64)

                # Fill in the counts for cells that have expression data
                cell_ids = final_gene_counts["cell_integer_id"].to_numpy()
                counts = final_gene_counts["count"].to_numpy()
                gene_counts[cell_ids] = counts

                logger.info(f"Gene counts: {gene_counts}")
            else:
                gene_counts = np.zeros(len(obs_df), dtype=np.int64)

        # Compute cumulative sum with first value as 0
        return np.insert(np.cumsum(gene_counts)[:-1], 0, 0).tolist()

    def _compute_cell_start_indices_anndata(
        self, adata, obs_df: pd.DataFrame
    ) -> list[int]:
        """Compute cell start indices for AnnData object"""
        # Check for existing n_genes or gene_count column
        if "n_genes" in obs_df.columns:
            logger.info("Using existing n_genes column for cell start indices")
            gene_counts = obs_df["n_genes"].to_numpy()
        elif "gene_count" in obs_df.columns:
            logger.info("Using existing gene_count column for cell start indices")
            gene_counts = obs_df["gene_count"].to_numpy()
        elif "n_genes_by_counts" in obs_df.columns:
            logger.info(
                "Using existing n_genes_by_counts column for cell start indices"
            )
            gene_counts = obs_df["n_genes_by_counts"].to_numpy()
        else:
            # Calculate from expression data
            logger.info("Calculating gene counts from expression data...")
            # Convert sparse matrix to COO to count genes per cell
            if sparse.issparse(adata.X):
                if hasattr(adata.X, "tocoo"):
                    coo = adata.X.tocoo()
                else:
                    # Handle backed mode where tocoo() might not be available
                    try:
                        coo = sparse.coo_matrix(adata.X)
                    except ValueError:
                        # If dtype is not supported, try to convert to a supported type
                        logger.warning(
                            "adata.X has unsupported dtype, attempting conversion"
                        )
                        # Try to convert to float32 first
                        try:
                            coo = sparse.coo_matrix(adata.X.astype(np.float32))
                        except Exception:
                            # If that fails, try to get the data in a different way
                            logger.warning(
                                "Could not convert adata.X, using fallback method"
                            )
                            # For backed mode, we might need to load a small sample
                            sample_size = min(1000, adata.n_obs)
                            sample_adata = adata[:sample_size, :]
                            if sparse.issparse(sample_adata.X):
                                coo = sample_adata.X.tocoo()
                            else:
                                coo = sparse.coo_matrix(sample_adata.X)
            else:
                try:
                    coo = sparse.coo_matrix(adata.X)
                except ValueError:
                    # If dtype is not supported, try to convert to a supported type
                    logger.warning(
                        "adata.X has unsupported dtype, attempting conversion"
                    )
                    try:
                        coo = sparse.coo_matrix(adata.X.astype(np.float32))
                    except Exception:
                        logger.warning(
                            "Could not convert adata.X, using fallback method"
                        )
                        # For backed mode, we might need to load a small sample
                        sample_size = min(1000, adata.n_obs)
                        sample_adata = adata[:sample_size, :]
                        if sparse.issparse(sample_adata.X):
                            coo = sample_adata.X.tocoo()
                        else:
                            coo = sparse.coo_matrix(sample_adata.X)

            # Count genes per cell using numpy bincount
            n_cells = adata.n_obs
            gene_counts = np.bincount(coo.row, minlength=n_cells)

        # Compute cumulative sum with first value as 0
        return np.insert(np.cumsum(gene_counts)[:-1], 0, 0).tolist()

    def _compute_expression_statistics(
        self, expression_dataset
    ) -> tuple[dict[str, float], int]:
        """Compute basic statistics from expression dataset using SQL"""
        # Use Polars to compute statistics directly from Lance dataset

        logger.info(
            "Computing expression statistics using fragment-by-fragment processing..."
        )

        # Initialize running statistics
        running_stats = {
            "min_value": float("inf"),
            "max_value": float("-inf"),
            "sum_value": 0.0,
            "sum_squared": 0.0,
            "count": 0,
        }

        # Process each fragment individually to avoid memory issues
        fragments = expression_dataset.get_fragments()
        total_fragments = len(fragments)

        logger.info(
            f"Processing {total_fragments} fragments for statistics computation..."
        )

        from tqdm import tqdm

        for i, fragment in enumerate(tqdm(fragments, desc="Computing statistics")):
            try:
                # Create Polars LazyFrame from this fragment
                ldf = pl.scan_pyarrow_dataset(fragment)

                # Compute fragment-level statistics
                fragment_stats = ldf.select(
                    [
                        pl.col("value").min().alias("min_value"),
                        pl.col("value").max().alias("max_value"),
                        pl.col("value").sum().alias("sum_value"),
                        (pl.col("value") ** 2).sum().alias("sum_squared"),
                        pl.col("value").count().alias("count"),
                    ]
                ).collect()

                # Extract values from the result
                row = fragment_stats.row(0)
                frag_min, frag_max, frag_sum, frag_sum_squared, frag_count = row

                # Update running statistics
                running_stats["min_value"] = min(running_stats["min_value"], frag_min)
                running_stats["max_value"] = max(running_stats["max_value"], frag_max)
                running_stats["sum_value"] += frag_sum
                running_stats["sum_squared"] += frag_sum_squared
                running_stats["count"] += frag_count

            except Exception as e:
                logger.warning(f"Error processing fragment {i}: {e}")
                logger.warning("Continuing with remaining fragments...")
                continue

        # Compute final statistics
        if running_stats["count"] == 0:
            logger.warning("No valid data found for statistics computation")
            return {
                "min_value": 0.0,
                "max_value": 0.0,
                "mean_value": 0.0,
                "std_value": 0.0,
            }, 0

        # Calculate mean
        mean_value = running_stats["sum_value"] / running_stats["count"]

        # Calculate standard deviation using the formula: sqrt((sum(x²) - n*mean²) / (n-1))
        variance = (
            running_stats["sum_squared"] - running_stats["count"] * mean_value**2
        ) / (running_stats["count"] - 1)
        std_value = variance**0.5 if variance > 0 else 0.0

        stats = {
            "min_value": float(running_stats["min_value"]),
            "max_value": float(running_stats["max_value"]),
            "mean_value": float(mean_value),
            "std_value": float(std_value),
        }

        logger.info(
            f"Statistics computed: min={stats['min_value']:.2f}, max={stats['max_value']:.2f}, mean={stats['mean_value']:.2f}, std={stats['std_value']:.2f}"
        )

        return stats, int(running_stats["count"])

    def _save_config(self, output_path: str, shape: tuple):
        """Save SLAF configuration with computed metadata"""
        n_cells = int(shape[0])
        n_genes = int(shape[1])

        # Compute additional metadata for faster info() method
        logger.info("Computing dataset statistics...")

        # Reference Lance dataset
        expression = lance.dataset(f"{output_path}/expression.lance")

        # Compute basic statistics and count from expression data
        expression_stats, expression_count = self._compute_expression_statistics(
            expression
        )

        total_possible_elements = n_cells * n_genes
        sparsity = 1 - (expression_count / total_possible_elements)

        # Load existing config to preserve checkpoint data
        config_path = f"{output_path}/config.json"
        existing_config = {}
        if self._path_exists(config_path):
            try:
                with self._open_file(config_path) as f:
                    existing_config = json.load(f)
            except Exception:
                pass  # If we can't load existing config, start fresh

        config = {
            "format_version": "0.3",
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

        # Preserve checkpoint data if it exists
        if "checkpoint" in existing_config:
            config["checkpoint"] = existing_config["checkpoint"]

        config_path = f"{output_path}/config.json"
        with self._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _save_multi_file_config(
        self,
        output_path: str,
        source_file_info: list,
        combined_expression: pa.Table | None = None,
        combined_cells: pa.Table | None = None,
        combined_genes: pa.Table | None = None,
    ):
        """Save SLAF configuration for multi-file conversion with source file tracking"""

        # If tables are not provided, load them from the Lance datasets
        if combined_cells is None or combined_genes is None:
            cells_dataset = lance.dataset(f"{output_path}/cells.lance")
            genes_dataset = lance.dataset(f"{output_path}/genes.lance")
            n_cells = len(cells_dataset.to_table())
            n_genes = len(genes_dataset.to_table())
        else:
            n_cells = len(combined_cells) if combined_cells is not None else 0
            n_genes = len(combined_genes) if combined_genes is not None else 0

        # Compute additional metadata for faster info() method
        logger.info("Computing multi-file dataset statistics...")

        # Reference Lance dataset
        expression = lance.dataset(f"{output_path}/expression.lance")

        # Compute basic statistics and count from expression data
        expression_stats, expression_count = self._compute_expression_statistics(
            expression
        )

        total_possible_elements = n_cells * n_genes
        sparsity = (
            1 - (expression_count / total_possible_elements)
            if total_possible_elements > 0
            else 1.0
        )

        # Calculate total cells and genes from source files
        total_cells_from_files = sum(info["n_cells"] for info in source_file_info)

        config = {
            "format_version": "0.3",
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
            "multi_file": {
                "source_files": source_file_info,
                "total_files": len(source_file_info),
                "total_cells_from_files": total_cells_from_files,
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }

        config_path = f"{output_path}/config.json"
        with self._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)
