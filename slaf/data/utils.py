import os

from loguru import logger


def detect_format(input_path: str) -> str:
    """
    Auto-detect input format based on file structure.

    Parameters:
    -----------
    input_path : str
        Path to the input file or directory

    Returns:
    --------
    str
        Detected format: "h5ad", "10x_mtx", "10x_h5", or "tiledb"

    Raises:
    -------
    ValueError
        If the format cannot be detected
    """
    if input_path.endswith(".h5ad"):
        return "h5ad"
    elif input_path.endswith(".h5"):
        return "10x_h5"
    elif os.path.isdir(input_path) and (
        # Check for TileDB SOMA format indicators
        any(
            f.endswith(".tdb")
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
        )
        or os.path.exists(os.path.join(input_path, "ms"))
        and os.path.exists(os.path.join(input_path, "obs"))
    ):
        # Check for TileDB SOMA format
        # TileDB SOMA experiments have .tdb files and ms/obs directories
        return "tiledb"
    elif os.path.isdir(input_path):
        # Check for 10x MTX files (both old and new formats)
        if os.path.exists(os.path.join(input_path, "matrix.mtx")) or os.path.exists(
            os.path.join(input_path, "matrix.mtx.gz")
        ):
            # Check for either genes.tsv or features.tsv (old vs new 10x format)
            if (
                os.path.exists(os.path.join(input_path, "genes.tsv"))
                or os.path.exists(os.path.join(input_path, "genes.tsv.gz"))
                or os.path.exists(os.path.join(input_path, "features.tsv"))
                or os.path.exists(os.path.join(input_path, "features.tsv.gz"))
            ):
                return "10x_mtx"

    raise ValueError(f"Cannot detect format for: {input_path}")


def discover_input_files(input_path: str, max_depth: int = 2) -> tuple[list[str], str]:
    """
    Discover input files from a path (file or directory).

    Parameters:
    -----------
    input_path : str
        Path to input file or directory
    max_depth : int
        Maximum depth to search for files in directories (default: 2)

    Returns:
    --------
    Tuple[List[str], str]
        Tuple of (list of file paths, detected format)

    Raises:
    -------
    ValueError
        If no valid files are found or formats are inconsistent
    FileNotFoundError
        If input path doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if os.path.isfile(input_path):
        # Single file - use existing format detection
        format_type = detect_format(input_path)
        return [input_path], format_type

    elif os.path.isdir(input_path):
        # Directory - discover files
        discovered_files = []
        detected_formats = set()

        # For directories, check if it's a known format directory structure
        # rather than looking for specific file extensions
        try:
            # Try to detect the directory format first
            directory_format = detect_format(input_path)
            # If it's a known format directory, treat it as a single "file"
            discovered_files = [input_path]
            detected_formats = {directory_format}
        except ValueError:
            # Fall back to searching for files with supported extensions
            supported_extensions = {".h5ad", ".h5", ".tiledb"}

            for root, _dirs, files in os.walk(input_path):
                for file in files:
                    if any(file.endswith(ext) for ext in supported_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            file_format = detect_format(file_path)
                            discovered_files.append(file_path)
                            detected_formats.add(file_format)
                        except ValueError:
                            # Skip files that can't be detected
                            continue

        if not discovered_files:
            raise ValueError(f"No supported files found in directory: {input_path}")

        if len(detected_formats) > 1:
            raise ValueError(
                f"Multiple formats detected in directory\n"
                f"  Found: {', '.join(sorted(detected_formats))}\n"
                f"  All files must use the same format"
            )

        # Sort files for consistent processing order
        discovered_files.sort()

        return discovered_files, list(detected_formats)[0]

    else:
        raise ValueError(f"Input path is neither a file nor directory: {input_path}")


def validate_input_files(file_paths: list[str], format_type: str) -> None:
    """
    Validate that input files are compatible for multi-file conversion.

    Performs comprehensive validation including:
    - File existence and readability
    - Format consistency
    - Schema compatibility (genes, cells, value types)
    - File size validation (non-empty files)

    Parameters:
    -----------
    file_paths : list[str]
        List of file paths to validate
    format_type : str
        Format type of the files

    Raises:
    -------
    ValueError
        If files are not compatible for merging
    FileNotFoundError
        If any file doesn't exist
    """
    if len(file_paths) < 2:
        return  # Single file, no validation needed

    logger.info(f"Validating compatibility of {len(file_paths)} {format_type} files...")

    # Collect all errors before reporting
    errors = []

    # 1. Basic file checks
    for file_path in file_paths:
        if not os.path.exists(file_path):
            errors.append(f"File does not exist: {file_path}")
            continue

        # Check file size (non-empty)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            errors.append(f"File is empty: {file_path}")
        elif file_size < 1024:  # Less than 1KB might be suspicious
            logger.warning(f"File is very small ({file_size} bytes): {file_path}")

    # 2. Schema compatibility validation
    try:
        _validate_schema_compatibility(file_paths, format_type)
    except ValueError as e:
        errors.append(str(e))

    # Report all errors together
    if errors:
        error_msg = "Validation failed with the following errors:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    logger.info("✓ All files are compatible for conversion")


def _validate_schema_compatibility(file_paths: list[str], format_type: str) -> None:
    """
    Validate schema compatibility between files.

    Checks for:
    - Identical gene sets
    - Identical cell metadata schemas
    - Same value types
    - Same format consistency
    """
    if len(file_paths) < 2:
        return

    # Use first file as ground truth
    reference_file = file_paths[0]
    reference_genes, reference_cells, reference_value_type = _extract_schema_info(
        reference_file, format_type
    )

    errors = []

    for i, file_path in enumerate(file_paths[1:], 1):
        try:
            genes, cells, value_type = _extract_schema_info(file_path, format_type)

            # Check gene set compatibility
            if genes != reference_genes:
                missing_genes = reference_genes - genes
                extra_genes = genes - reference_genes
                if missing_genes or extra_genes:
                    error_msg = f"File {os.path.basename(file_path)} is incompatible with existing dataset:"
                    if missing_genes:
                        error_msg += f"\n  Missing genes: {sorted(missing_genes)[:5]}{'...' if len(missing_genes) > 5 else ''}"
                    if extra_genes:
                        error_msg += f"\n  Extra genes: {sorted(extra_genes)[:5]}{'...' if len(extra_genes) > 5 else ''}"
                    errors.append(error_msg)

            # Check cell metadata schema compatibility
            if cells != reference_cells:
                missing_cols = reference_cells - cells
                extra_cols = cells - reference_cells
                if missing_cols or extra_cols:
                    error_msg = f"File {os.path.basename(file_path)} has incompatible cell metadata schema:"
                    if missing_cols:
                        error_msg += f"\n  Missing columns: {sorted(missing_cols)}"
                    if extra_cols:
                        error_msg += f"\n  Extra columns: {sorted(extra_cols)}"
                    errors.append(error_msg)

            # Check value type compatibility
            if value_type != reference_value_type:
                errors.append(
                    f"File {os.path.basename(file_path)} has value type {value_type}, expected {reference_value_type}"
                )

        except Exception as e:
            errors.append(
                f"File {i + 1} ({os.path.basename(file_path)}) could not be read: {e}"
            )

    if errors:
        raise ValueError(
            "Schema compatibility validation failed:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


def _extract_schema_info(
    file_path: str, format_type: str
) -> tuple[set[str], set[str], str]:
    """
    Extract schema information from a single file.

    Returns:
    --------
    tuple[set[str], set[str], str]
        (gene_ids, cell_metadata_columns, value_type)
    """
    try:
        if format_type == "h5ad":
            return _extract_h5ad_schema(file_path)
        elif format_type == "10x_mtx":
            return _extract_10x_mtx_schema(file_path)
        elif format_type == "10x_h5":
            return _extract_10x_h5_schema(file_path)
        elif format_type == "tiledb":
            return _extract_tiledb_schema(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    except Exception as e:
        raise ValueError(f"Could not extract schema from {file_path}: {e}") from e


def _extract_h5ad_schema(file_path: str) -> tuple[set[str], set[str], str]:
    """Extract schema from h5ad file."""
    try:
        import numpy as np
        import scanpy as sc

        # Read in backed mode to avoid loading full data
        adata = sc.read_h5ad(file_path, backed="r")

        # Extract gene IDs
        gene_ids = set(adata.var_names.astype(str))

        # Extract cell metadata columns
        cell_columns = set(adata.obs.columns.astype(str))

        # Determine value type from a small sample
        sample_size = min(1000, adata.n_obs)
        sample_data = (
            adata.X[:sample_size].data
            if hasattr(adata.X, "data")
            else adata.X[:sample_size].toarray().flatten()
        )

        if np.issubdtype(sample_data.dtype, np.integer):
            max_val = np.max(sample_data)
            if max_val <= 65535:
                value_type = "uint16"
            else:
                value_type = "int32"
        else:
            value_type = "float32"

        adata.file.close()

        return gene_ids, cell_columns, value_type

    except ImportError:
        raise ImportError(
            "Scanpy is required for h5ad validation. Install with: pip install scanpy"
        ) from None


def _extract_10x_mtx_schema(file_path: str) -> tuple[set[str], set[str], str]:
    """Extract schema from 10x MTX directory."""
    try:
        import scanpy as sc

        # Read a small sample to get schema
        adata = sc.read_10x_mtx(file_path)

        gene_ids = set(adata.var_names.astype(str))
        cell_columns = set(adata.obs.columns.astype(str))

        # 10x data is typically integer counts
        value_type = "uint16"

        return gene_ids, cell_columns, value_type

    except ImportError:
        raise ImportError(
            "Scanpy is required for 10x MTX validation. Install with: pip install scanpy"
        ) from None


def _extract_10x_h5_schema(file_path: str) -> tuple[set[str], set[str], str]:
    """Extract schema from 10x H5 file."""
    try:
        import scanpy as sc

        # Try 10x H5 first, fall back to regular h5ad
        try:
            adata = sc.read_10x_h5(file_path, genome="X")
        except Exception:
            adata = sc.read_h5ad(file_path, backed="r")

        gene_ids = set(adata.var_names.astype(str))
        cell_columns = set(adata.obs.columns.astype(str))

        # 10x data is typically integer counts
        value_type = "uint16"

        if hasattr(adata, "file"):
            adata.file.close()

        return gene_ids, cell_columns, value_type

    except ImportError:
        raise ImportError(
            "Scanpy is required for 10x H5 validation. Install with: pip install scanpy"
        ) from None


def _extract_tiledb_schema(file_path: str) -> tuple[set[str], set[str], str]:
    """Extract schema from TileDB SOMA file."""
    try:
        import tiledbsoma as soma

        with soma.open(file_path) as exp:
            # Get gene IDs from var table
            var_df = exp.ms["RNA"].var.read().concat().to_pandas()
            gene_ids = set(var_df.index.astype(str))

            # Get cell metadata columns from obs table
            obs_df = exp.obs.read().concat().to_pandas()
            cell_columns = set(obs_df.columns.astype(str))

            # TileDB SOMA typically uses float32 for expression data
            value_type = "float32"

            return gene_ids, cell_columns, value_type

    except ImportError:
        raise ImportError(
            "tiledbsoma is required for TileDB validation. Install with: pip install tiledbsoma"
        ) from None


# Backward compatibility alias
validate_multi_file_compatibility = validate_input_files
