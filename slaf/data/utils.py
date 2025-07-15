from pathlib import Path


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
        Detected format: "h5ad", "10x_mtx", or "10x_h5"

    Raises:
    -------
    ValueError
        If the format cannot be detected
    """
    path = Path(input_path)

    if path.suffix == ".h5ad":
        return "h5ad"
    elif path.suffix == ".h5":
        return "10x_h5"
    elif path.is_dir():
        # Check for 10x MTX files (both old and new formats)
        if (path / "matrix.mtx").exists() or (path / "matrix.mtx.gz").exists():
            # Check for either genes.tsv or features.tsv (old vs new 10x format)
            if (
                (path / "genes.tsv").exists()
                or (path / "genes.tsv.gz").exists()
                or (path / "features.tsv").exists()
                or (path / "features.tsv.gz").exists()
            ):
                return "10x_mtx"

    raise ValueError(f"Cannot detect format for: {input_path}")
