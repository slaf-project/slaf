import gzip
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from scipy import sparse


class BaseChunkedReader(ABC):
    """
    Base class for chunked readers of single-cell data formats.

    This provides a unified interface for reading different single-cell data formats
    in a memory-efficient, chunked manner. Subclasses implement format-specific
    reading logic while maintaining a consistent API.
    """

    def __init__(self, file_path: str):
        """
        Initialize the chunked reader.

        Args:
            file_path: Path to the data file or directory. Can be a single file
                      (h5ad, h5) or directory (10x MTX format).

        Examples:
            >>> # Initialize with h5ad file
            >>> reader = BaseChunkedReader("data.h5ad")
            >>> print(f"File path: {reader.file_path}")
            File path: data.h5ad

            >>> # Initialize with 10x MTX directory
            >>> reader = BaseChunkedReader("10x_data/")
            >>> print(f"File path: {reader.file_path}")
            File path: 10x_data/
        """
        self.file_path = file_path
        self._n_obs: int | None = None
        self._n_vars: int | None = None
        self._obs_names: np.ndarray | None = None
        self._var_names: np.ndarray | None = None

    def __enter__(self):
        """Context manager entry"""
        self._open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._close_file()

    @abstractmethod
    def _open_file(self) -> None:
        """Open the data file/directory. Called by __enter__."""
        pass

    @abstractmethod
    def _close_file(self) -> None:
        """Close the data file/directory. Called by __exit__."""
        pass

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """
        Number of observations (cells) in the dataset.

        Returns:
            Total number of cells in the dataset.
        """
        pass

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """
        Number of variables (genes) in the dataset.

        Returns:
            Total number of genes in the dataset.
        """
        pass

    @property
    @abstractmethod
    def obs_names(self) -> np.ndarray:
        """
        Get observation names (cell barcodes).

        Returns:
            Array of cell barcode strings.
        """
        pass

    @property
    @abstractmethod
    def var_names(self) -> np.ndarray:
        """
        Get variable names (gene names).

        Returns:
            Array of gene name strings.
        """
        pass

    @abstractmethod
    def get_obs_metadata(self) -> pd.DataFrame:
        """
        Get observation metadata as pandas DataFrame.

        Returns:
            DataFrame containing cell metadata with cell barcodes as index.
        """
        pass

    @abstractmethod
    def get_var_metadata(self) -> pd.DataFrame:
        """
        Get variable metadata as pandas DataFrame.

        Returns:
            DataFrame containing gene metadata with gene names as index.
        """
        pass

    @abstractmethod
    def iter_chunks(
        self, chunk_size: int = 1000, obs_chunk: bool = True
    ) -> Iterator[tuple[pa.Table, slice]]:
        """
        Iterate over data in chunks, returning Arrow tables directly.

        Args:
            chunk_size: Number of observations per chunk. Default: 1000.
            obs_chunk: If True, chunk by observations (cells). If False, chunk by
                      variables (genes). Default: True.

        Yields:
            Tuple of (chunk_table, slice) where:
                - chunk_table: Arrow table with columns:
                    - cell_integer_id: uint32 array
                    - gene_integer_id: uint16 array
                    - value: uint16 array
                - slice: Slice object indicating the chunk boundaries

        Examples:
            >>> # Iterate over cells in chunks
            >>> reader = ChunkedH5ADReader("data.h5ad")
            >>> for chunk, slice_obj in reader.iter_chunks(chunk_size=500):
            ...     print(f"Chunk shape: {chunk.shape}")
            ...     print(f"Slice: {slice_obj}")
            Chunk shape: (500, 3)
            Slice: slice(0, 500, None)
        """
        pass

    def get_chunk(
        self, obs_slice: slice | None = None, var_slice: slice | None = None
    ) -> pa.Table:
        """
        Get a specific chunk of data as an Arrow table.

        Args:
            obs_slice: Slice for observations (cells). If None, returns all cells.
            var_slice: Slice for variables (genes). If None, returns all genes.

        Returns:
            Arrow table with the requested data chunk containing columns:
                - cell_integer_id: uint32 array
                - gene_integer_id: uint16 array
                - value: uint16 array

        Examples:
            >>> # Get first 100 cells, all genes
            >>> reader = ChunkedH5ADReader("data.h5ad")
            >>> chunk = reader.get_chunk(obs_slice=slice(0, 100))
            >>> print(f"Chunk shape: {chunk.shape}")
            Chunk shape: (100, 3)

            >>> # Get specific cell and gene ranges
            >>> chunk = reader.get_chunk(
            ...     obs_slice=slice(50, 150),
            ...     var_slice=slice(1000, 2000)
            ... )
            >>> print(f"Chunk shape: {chunk.shape}")
            Chunk shape: (100, 3)
        """
        if obs_slice is None:
            obs_slice = slice(0, self.n_obs)
        if var_slice is None:
            var_slice = slice(0, self.n_vars)

        return self._get_chunk_impl(obs_slice, var_slice)

    @abstractmethod
    def _get_chunk_impl(self, obs_slice: slice, var_slice: slice) -> pa.Table:
        """Implementation of chunk retrieval. Called by get_chunk."""
        pass


class ChunkedH5ADReader(BaseChunkedReader):
    """
    A chunked reader for h5ad files using scanpy's backed mode for memory-efficient processing.

    This reader provides memory-efficient access to h5ad files by leveraging scanpy's
    backed reading capabilities. It supports both sparse and dense expression matrices
    and provides a consistent interface for chunked processing.

    Key Features:
        - Memory-efficient chunked reading using scanpy backed mode
        - Support for sparse and dense matrices
        - Automatic metadata extraction
        - Context manager support
        - Arrow table output format

    Examples:
        >>> # Basic usage with context manager
        >>> with ChunkedH5ADReader("data.h5ad") as reader:
        ...     print(f"Dataset shape: {reader.n_obs} cells × {reader.n_vars} genes")
        ...     for chunk, slice_obj in reader.iter_chunks(chunk_size=500):
        ...         print(f"Processing chunk: {slice_obj}")
        Dataset shape: 2700 cells × 32738 genes
        Processing chunk: slice(0, 500, None)

        >>> # Access metadata
        >>> reader = ChunkedH5ADReader("data.h5ad")
        >>> obs_meta = reader.get_obs_metadata()
        >>> print(f"Cell metadata columns: {list(obs_meta.columns)}")
        Cell metadata columns: ['cell_type', 'total_counts', 'batch']
    """

    def __init__(
        self, filename: str, chunk_size: int = 25000, value_type: str = "uint16"
    ):
        """
        Initialize the chunked h5ad reader.

        Args:
            filename: Path to the h5ad file. Must be a valid h5ad file with
                     proper AnnData structure.
            chunk_size: Size of chunks for processing. Default: 25,000 elements.

        Raises:
            FileNotFoundError: If the h5ad file doesn't exist.
            ValueError: If the file is not a valid h5ad format.

        Examples:
            >>> # Initialize with existing file
            >>> reader = ChunkedH5ADReader("pbmc3k.h5ad")
            >>> print(f"File path: {reader.file_path}")
            File path: pbmc3k.h5ad

            >>> # Initialize with custom chunk size
            >>> reader = ChunkedH5ADReader("large_dataset.h5ad", chunk_size=50000)
            >>> print(f"Chunk size: {reader.chunk_size}")
            Chunk size: 50000

            >>> # Error handling for missing file
            >>> try:
            ...     reader = ChunkedH5ADReader("nonexistent.h5ad")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
            Error: [Errno 2] No such file or directory: 'nonexistent.h5ad'
        """
        super().__init__(filename)
        self.adata = None
        self.chunk_size = chunk_size
        self.value_type = value_type

    def _open_file(self) -> None:
        """Open the h5ad file using scanpy backed mode"""
        import scanpy as sc

        self.adata = sc.read_h5ad(self.file_path, backed="r")
        assert self.adata is not None
        self.file = self.adata.file

    def _close_file(self) -> None:
        """Close the h5ad file"""
        if self.adata is not None:
            self.adata.file.close()
            self.adata = None
            self.file = None

    @property
    def n_obs(self) -> int:
        """
        Number of observations (cells) in the h5ad dataset.

        Returns:
            Total number of cells in the dataset.

        Raises:
            RuntimeError: If the file is not opened (use context manager).
        """
        if self._n_obs is None:
            if self.adata is None:
                raise RuntimeError("File not opened. Use context manager.")
            self._n_obs = self.adata.n_obs
        return self._n_obs

    @property
    def n_vars(self) -> int:
        """
        Number of variables (genes) in the h5ad dataset.

        Returns:
            Total number of genes in the dataset.

        Raises:
            RuntimeError: If the file is not opened (use context manager).
        """
        if self._n_vars is None:
            if self.adata is None:
                raise RuntimeError("File not opened. Use context manager.")
            self._n_vars = self.adata.n_vars
        return self._n_vars

    @property
    def obs_names(self) -> np.ndarray:
        """Get observation names (cell barcodes)"""
        if self._obs_names is None:
            if self.adata is None:
                raise RuntimeError("File not opened. Use context manager.")
            self._obs_names = self.adata.obs_names.values
        return self._obs_names

    @property
    def var_names(self) -> np.ndarray:
        """Get variable names (gene names)"""
        if self._var_names is None:
            if self.adata is None:
                raise RuntimeError("File not opened. Use context manager.")
            self._var_names = self.adata.var_names.values
        return self._var_names

    def get_obs_metadata(self) -> pd.DataFrame:
        """Get observation metadata as pandas DataFrame"""
        if self.adata is None:
            raise RuntimeError("File not opened. Use context manager.")
        return self.adata.obs.copy()

    def get_var_metadata(self) -> pd.DataFrame:
        """Get variable metadata as pandas DataFrame"""
        if self.adata is None:
            raise RuntimeError("File not opened. Use context manager.")
        return self.adata.var.copy()

    def iter_chunks(
        self, chunk_size: int = 1000, obs_chunk: bool = True
    ) -> Iterator[tuple[pa.Table, slice]]:
        """
        Iterate over data in chunks, returning Arrow tables directly.

        Parameters:
        -----------
        chunk_size : int
            Number of observations per chunk
        obs_chunk : bool
            If True, chunk by observations (cells). If False, chunk by variables (genes).

        Yields:
        -------
        tuple
            (chunk_table, slice) where chunk_table is an Arrow table with columns:
            - cell_integer_id: uint32 array
            - gene_integer_id: uint16 array
            - value: uint16 array
            and slice is the slice object indicating the chunk boundaries
        """
        if obs_chunk:
            total_obs = self.n_obs
            for start in range(0, total_obs, chunk_size):
                end = min(start + chunk_size, total_obs)
                chunk_table = self._read_chunk_as_arrow(start, end)
                yield chunk_table, slice(start, end)
        else:
            # Variable chunking not implemented for h5ad
            raise NotImplementedError("Variable chunking not supported for h5ad files")

    def _read_chunk_as_arrow(self, start_row: int, end_row: int) -> pa.Table:
        """Read a chunk and return as Arrow table directly"""
        if self.adata is None:
            raise RuntimeError("File not opened. Use context manager.")

        # Use scanpy's backed slicing to get the chunk
        chunk_adata = self.adata[start_row:end_row, :]

        # Convert sparse matrix to Arrow arrays using optimized approach
        if sparse.issparse(chunk_adata.X):
            # Convert to COO format
            coo = chunk_adata.X.tocoo()

            if coo.nnz > 0:
                # Create Arrow arrays directly
                cell_integer_ids = coo.row.astype(np.uint32) + start_row
                gene_integer_ids = coo.col.astype(np.uint16)
                # Use the specified value type
                if self.value_type == "uint16":
                    values = coo.data.astype(np.uint16)
                elif self.value_type == "float32":
                    values = coo.data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                result = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values),
                    }
                )
            else:
                # Empty chunk
                value_pa_type = (
                    pa.uint16() if self.value_type == "uint16" else pa.float32()
                )
                result = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )
        else:
            # Dense matrix - convert to COO (this should be rare)
            coo = sparse.coo_matrix(chunk_adata.X)

            if coo.nnz > 0:
                cell_integer_ids = coo.row.astype(np.uint32) + start_row
                gene_integer_ids = coo.col.astype(np.uint16)
                # Use the specified value type
                if self.value_type == "uint16":
                    values = coo.data.astype(np.uint16)
                elif self.value_type == "float32":
                    values = coo.data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                result = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values),
                    }
                )
            else:
                value_pa_type = (
                    pa.uint16() if self.value_type == "uint16" else pa.float32()
                )
                result = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )

        return result

    def _get_chunk_impl(self, obs_slice: slice, var_slice: slice) -> pa.Table:
        """Implementation of chunk retrieval for h5ad files"""
        if self.adata is None:
            raise RuntimeError("File not opened. Use context manager.")

        start_row = obs_slice.start or 0
        end_row = obs_slice.stop or self.n_obs
        start_col = var_slice.start or 0
        end_col = var_slice.stop or self.n_vars

        # Use scanpy's backed slicing to get the chunk
        chunk_adata = self.adata[start_row:end_row, start_col:end_col]

        # Convert to COO format
        if sparse.issparse(chunk_adata.X):
            coo = chunk_adata.X.tocoo()
        else:
            coo = sparse.coo_matrix(chunk_adata.X)

        if coo.nnz > 0:
            # Create Arrow arrays directly
            cell_integer_ids = coo.row.astype(np.uint32) + start_row
            gene_integer_ids = coo.col.astype(np.uint16) + start_col
            values = coo.data.astype(np.uint16)

            return pa.table(
                {
                    "cell_integer_id": pa.array(cell_integer_ids),
                    "gene_integer_id": pa.array(gene_integer_ids),
                    "value": pa.array(values),
                }
            )
        else:
            # Empty chunk
            return pa.table(
                {
                    "cell_integer_id": pa.array([], type=pa.uint32()),
                    "gene_integer_id": pa.array([], type=pa.uint16()),
                    "value": pa.array([], type=pa.uint16()),
                }
            )

    def get_gene_expression(
        self, gene_names: list, chunk_size: int = 1000
    ) -> Iterator[pd.DataFrame]:
        """
        Get gene expression data for specific genes in chunks.

        Parameters:
        -----------
        gene_names : list
            List of gene names to extract
        chunk_size : int
            Number of cells per chunk

        Yields:
        -------
        pd.DataFrame
            DataFrame with gene expression data for the chunk
        """
        if self.adata is None:
            raise RuntimeError("File not opened. Use context manager.")

        # Find gene indices using scanpy's var_names
        var_names = self.adata.var_names
        gene_indices = []
        for gene in gene_names:
            if gene in var_names:
                gene_indices.append(var_names.get_loc(gene))
            else:
                warnings.warn(f"Gene {gene} not found in dataset", stacklevel=2)

        if not gene_indices:
            return

        # Iterate over chunks and extract gene expression
        total_obs = self.n_obs
        for start in range(0, total_obs, chunk_size):
            end = min(start + chunk_size, total_obs)

            # Use scanpy's backed slicing to get the chunk
            chunk_adata = self.adata[start:end, gene_indices]

            # Convert to DataFrame
            chunk_df = chunk_adata.to_df()

            # Rename columns to gene names
            chunk_df.columns = [gene_names[i] for i in gene_indices]

            yield chunk_df


class Chunked10xMTXReader(BaseChunkedReader):
    """
    Native chunked reader for 10x MTX directory format.
    Reads chunks directly from matrix.mtx without loading the entire matrix into memory.
    """

    def __init__(self, mtx_dir: str, value_type: str = "uint16"):
        super().__init__(mtx_dir)
        self.mtx_dir = Path(mtx_dir)
        self.value_type = value_type
        self._barcodes = None
        self._genes = None
        self._matrix_file = None
        self._matrix_header = None
        self._n_obs = None
        self._n_vars = None
        self._nnz = None

    def _open_file(self):
        # Read barcodes
        barcodes_path = self.mtx_dir / "barcodes.tsv"
        if not barcodes_path.exists():
            barcodes_path = self.mtx_dir / "barcodes.tsv.gz"
        self._barcodes = (
            pd.read_csv(barcodes_path, header=None, sep="\t")
            .iloc[:, 0]
            .astype(str)
            .values
        )

        # Read genes/features
        genes_path = self.mtx_dir / "genes.tsv"
        if not genes_path.exists():
            genes_path = self.mtx_dir / "genes.tsv.gz"
        if not genes_path.exists():
            genes_path = self.mtx_dir / "features.tsv"
        if not genes_path.exists():
            genes_path = self.mtx_dir / "features.tsv.gz"

        # Read genes file and match scanpy behavior
        genes_df = pd.read_csv(genes_path, header=None, sep="\t")
        if genes_df.shape[1] >= 2:
            # Use second column (gene_symbol) like scanpy
            self._genes = genes_df.iloc[:, 1].astype(str).values
        else:
            # Use first column if only one column
            self._genes = genes_df.iloc[:, 0].astype(str).values

        # Open matrix file for streaming
        mtx_path = self.mtx_dir / "matrix.mtx"
        if not mtx_path.exists():
            mtx_path = self.mtx_dir / "matrix.mtx.gz"

        if str(mtx_path).endswith(".gz"):
            self._matrix_file = gzip.open(mtx_path, "rt")
        else:
            self._matrix_file = open(mtx_path)

        # Read header
        self._read_matrix_header()

        self._n_obs = len(self._barcodes)
        self._n_vars = len(self._genes)

    def _read_matrix_header(self):
        """Read MTX file header to get dimensions"""
        # Skip comments
        line = self._matrix_file.readline()
        while line.startswith("%"):
            line = self._matrix_file.readline()

        # Read dimensions
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError("Invalid MTX header format")

        self._n_vars, self._n_obs, self._nnz = map(int, parts)

    def _close_file(self):
        if self._matrix_file:
            self._matrix_file.close()
            self._matrix_file = None
        self._barcodes = None
        self._genes = None

    @property
    def n_obs(self) -> int:
        if self._n_obs is None:
            raise RuntimeError("n_obs not initialized")
        return self._n_obs

    @property
    def n_vars(self) -> int:
        if self._n_vars is None:
            raise RuntimeError("n_vars not initialized")
        return self._n_vars

    @property
    def obs_names(self) -> np.ndarray:
        if self._barcodes is None:
            raise RuntimeError("Barcodes not initialized")
        return self._barcodes

    @property
    def var_names(self) -> np.ndarray:
        if self._genes is None:
            raise RuntimeError("Genes not initialized")
        return self._genes

    def get_obs_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(index=self._barcodes)

    def get_var_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(index=self._genes)

    def iter_chunks(self, chunk_size: int = 1000, obs_chunk: bool = True):
        """Iterate over chunks by reading MTX file directly and returning Arrow tables"""
        if not obs_chunk:
            raise NotImplementedError("Variable chunking not supported for MTX files")

        # Reset file position to start of data
        if self._matrix_file is None:
            raise RuntimeError("Matrix file not opened")
        self._matrix_file.seek(0)
        self._read_matrix_header()

        # Read all non-zero entries
        entries = []
        for line in self._matrix_file:
            row, col, val = map(float, line.strip().split())
            # MTX is 1-indexed, convert to 0-indexed
            # MTX stores (gene, cell, value) but matrix is transposed
            # So we need to swap row and col to get (cell, gene, value)
            entries.append((int(col) - 1, int(row) - 1, val))

        # Sort by column (cell) for efficient chunking
        entries.sort(key=lambda x: x[0])

        # Yield chunks
        n_obs = self._n_obs
        if n_obs is None:
            raise RuntimeError("n_obs not initialized")

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)

            # Filter entries for this chunk
            chunk_entries = [
                (row - start, col, val)
                for row, col, val in entries
                if start <= row < end
            ]

            # Convert to Arrow table directly
            if chunk_entries:
                rows, cols, vals = zip(*chunk_entries, strict=False)
                # Create Arrow arrays directly
                cell_integer_ids = np.array(rows, dtype=np.uint32) + start
                gene_integer_ids = np.array(cols, dtype=np.uint16)
                # Use the specified value type
                if self.value_type == "uint16":
                    values = np.array(vals, dtype=np.uint16)
                elif self.value_type == "float32":
                    values = np.array(vals, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                # Use the specified value type for Arrow array
                if self.value_type == "uint16":
                    value_pa_type = pa.uint16()
                elif self.value_type == "float32":
                    value_pa_type = pa.float32()
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values, type=value_pa_type),
                    }
                )
            else:
                # Empty chunk
                # Use the specified value type for Arrow array
                if self.value_type == "uint16":
                    value_pa_type = pa.uint16()
                elif self.value_type == "float32":
                    value_pa_type = pa.float32()
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )

            yield chunk_table, slice(start, end)

    def _get_chunk_impl(self, obs_slice: slice, var_slice: slice):
        """Get a specific chunk by reading the entire file and filtering"""
        # For specific chunks, we need to read the file and filter
        # This is not ideal but necessary for the interface
        start_row = obs_slice.start or 0
        end_row = obs_slice.stop or self._n_obs

        # Reset file position
        if self._matrix_file is None:
            raise RuntimeError("Matrix file not opened")
        self._matrix_file.seek(0)
        self._read_matrix_header()

        # Read and filter entries
        entries = []
        for line in self._matrix_file:
            row, col, val = map(float, line.strip().split())
            # MTX stores (gene, cell, value) but matrix is transposed
            # So we need to swap row and col to get (cell, gene, value)
            cell_idx, gene_idx = int(col) - 1, int(row) - 1
            if start_row <= cell_idx < end_row:
                entries.append((cell_idx - start_row, gene_idx, val))

        # Convert to Arrow table directly
        n_vars = self._n_vars
        if n_vars is None:
            raise RuntimeError("n_vars not initialized")

        if entries:
            rows, cols, vals = zip(*entries, strict=False)
            # Create Arrow arrays directly
            cell_integer_ids = np.array(rows, dtype=np.uint32) + start_row
            gene_integer_ids = np.array(cols, dtype=np.uint16)
            # Use the specified value type
            if self.value_type == "uint16":
                values = np.array(vals, dtype=np.uint16)
            elif self.value_type == "float32":
                values = np.array(vals, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported value type: {self.value_type}")

            # Use the specified value type for Arrow array
            if self.value_type == "uint16":
                value_pa_type = pa.uint16()
            elif self.value_type == "float32":
                value_pa_type = pa.float32()
            else:
                raise ValueError(f"Unsupported value type: {self.value_type}")

            chunk_table = pa.table(
                {
                    "cell_integer_id": pa.array(cell_integer_ids),
                    "gene_integer_id": pa.array(gene_integer_ids),
                    "value": pa.array(values, type=value_pa_type),
                }
            )
        else:
            # Empty chunk
            # Use the specified value type for Arrow array
            if self.value_type == "uint16":
                value_pa_type = pa.uint16()
            elif self.value_type == "float32":
                value_pa_type = pa.float32()
            else:
                raise ValueError(f"Unsupported value type: {self.value_type}")

            chunk_table = pa.table(
                {
                    "cell_integer_id": pa.array([], type=pa.uint32()),
                    "gene_integer_id": pa.array([], type=pa.uint16()),
                    "value": pa.array([], type=value_pa_type),
                }
            )

        # Apply variable slicing if needed
        n_vars = self._n_vars
        if n_vars is None:
            raise RuntimeError("n_vars not initialized")
        if var_slice.start != 0 or var_slice.stop != n_vars:
            # Filter by gene_integer_id
            gene_integer_ids = chunk_table.column("gene_integer_id").to_numpy()
            mask = (gene_integer_ids >= var_slice.start) & (
                gene_integer_ids < var_slice.stop
            )
            chunk_table = chunk_table.filter(pa.array(mask))

            # Adjust gene_integer_id to be relative to the slice
            if var_slice.start > 0:
                gene_integer_ids = (
                    chunk_table.column("gene_integer_id").to_numpy() - var_slice.start
                )
                chunk_table = chunk_table.set_column(
                    1, "gene_integer_id", pa.array(gene_integer_ids)
                )

        return chunk_table


class ChunkedTileDBReader(BaseChunkedReader):
    """Chunked reader for TileDB SOMA format files."""

    def __init__(
        self, tiledb_path: str, collection_name: str = "RNA", value_type: str = "auto"
    ):
        """
        Initialize TileDB chunked reader.

        Args:
            tiledb_path: Path to TileDB SOMA experiment directory
            collection_name: Name of the measurement collection (default: "RNA")
            value_type: Data type for expression values (default: "uint16")
        """
        super().__init__(tiledb_path)
        self.collection_name = collection_name
        self.value_type = value_type
        self._experiment: Any = None
        self._X: Any = None
        self._obs_df: pd.DataFrame | None = None
        self._var_df: pd.DataFrame | None = None

    def _open_file(self) -> None:
        """Open TileDB SOMA experiment."""
        try:
            import tiledbsoma
        except ImportError as e:
            raise ImportError(
                "TileDB SOMA is required for TileDB format support. "
                "Install with: pip install tiledbsoma"
            ) from e

        self._experiment = tiledbsoma.Experiment.open(self.file_path)
        assert self._experiment is not None, "Failed to open TileDB experiment"

        # Get the measurement collection
        if self.collection_name not in self._experiment.ms:
            available_collections = list(self._experiment.ms.keys())
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                f"Available collections: {available_collections}"
            )

        self._X = self._experiment.ms[self.collection_name].X["data"]

    def _close_file(self) -> None:
        """Close TileDB SOMA experiment."""
        if self._experiment is not None:
            self._experiment.close()
            self._experiment = None
            self._X = None

    @property
    def n_obs(self) -> int:
        """Number of observations (cells)."""
        if self._X is None:
            raise RuntimeError("File not opened")
        return self._X.shape[0]

    @property
    def n_vars(self) -> int:
        """Number of variables (genes)."""
        if self._X is None:
            raise RuntimeError("File not opened")
        return self._X.shape[1]

    @property
    def obs_names(self) -> np.ndarray:
        """Observation names (cell IDs)."""
        if self._obs_df is None:
            self._obs_df = self.get_obs_metadata()
        assert self._obs_df is not None, "Failed to load observation metadata"
        return self._obs_df.index.values

    @property
    def var_names(self) -> np.ndarray:
        """Variable names (gene IDs)."""
        if self._var_df is None:
            self._var_df = self.get_var_metadata()
        assert self._var_df is not None, "Failed to load variable metadata"
        return self._var_df.index.values

    def get_obs_metadata(self) -> pd.DataFrame:
        """Get observation metadata."""
        if self._experiment is None:
            raise RuntimeError("File not opened")

        # Get obs dataframe from the experiment level
        obs_df = pl.from_arrow(self._experiment.obs.read().concat()).to_pandas()
        return obs_df

    def get_var_metadata(self) -> pd.DataFrame:
        """Get variable metadata."""
        if self._experiment is None:
            raise RuntimeError("File not opened")

        # Get var dataframe from the measurement collection
        var_df = pl.from_arrow(
            self._experiment.ms[self.collection_name].var.read().concat()
        ).to_pandas()
        return var_df

    def iter_chunks(self, chunk_size: int = 1000, obs_chunk: bool = True):
        """Iterate over chunks by reading from TileDB and returning Arrow tables."""
        if not obs_chunk:
            raise NotImplementedError(
                "Variable chunking not supported for TileDB files"
            )

        if self._X is None:
            raise RuntimeError("File not opened")

        total_obs = self.n_obs
        for start in range(0, total_obs, chunk_size):
            end = min(start + chunk_size, total_obs)

            # Read slice from TileDB as Arrow table
            arrow_data = self._X.read((slice(start, end),)).tables().concat()

            # Convert Arrow table to Polars DataFrame for processing
            df = pl.from_arrow(arrow_data)
            assert isinstance(df, pl.DataFrame), "Expected DataFrame from Arrow table"

            # Rename SOMA columns to expected names
            df = df.rename(
                {
                    "soma_dim_0": "cell_integer_id",
                    "soma_dim_1": "gene_integer_id",
                    "soma_data": "value",
                }
            )

            # Convert data types to match expected schema
            # Always convert cell and gene IDs to the correct types
            df = df.with_columns(
                [
                    pl.col("cell_integer_id").cast(pl.UInt32),
                    pl.col("gene_integer_id").cast(pl.UInt16),
                ]
            )

            # Only convert value column if value_type is specified and different from original
            if self.value_type == "uint16":
                df = df.with_columns([pl.col("value").cast(pl.UInt16)])
            elif self.value_type == "float32":
                df = df.with_columns([pl.col("value").cast(pl.Float32)])
            # If value_type is None or "auto", keep original data type for validation

            # Convert back to Arrow table
            chunk_table = df.to_arrow()

            # Create slice for this chunk
            obs_slice = slice(start, end)

            yield chunk_table, obs_slice

    def _get_chunk_impl(self, obs_slice: slice, var_slice: slice):
        """Get a specific chunk of data."""
        if self._X is None:
            raise RuntimeError("File not opened")

        # Read slice from TileDB as Arrow table
        arrow_data = (
            self._X.read(
                (
                    obs_slice,
                    var_slice,
                )
            )
            .tables()
            .concat()
        )

        # Convert Arrow table to Polars DataFrame for processing
        df = pl.from_arrow(arrow_data)
        assert isinstance(df, pl.DataFrame), "Expected DataFrame from Arrow table"

        # Rename SOMA columns to expected names
        df = df.rename(
            {
                "soma_dim_0": "cell_integer_id",
                "soma_dim_1": "gene_integer_id",
                "soma_data": "value",
            }
        )

        # Convert data types to match expected schema
        # Always convert cell and gene IDs to the correct types
        df = df.with_columns(
            [
                pl.col("cell_integer_id").cast(pl.UInt32),
                pl.col("gene_integer_id").cast(pl.UInt16),
            ]
        )

        # Only convert value column if value_type is specified and different from original
        if self.value_type == "uint16":
            df = df.with_columns([pl.col("value").cast(pl.UInt16)])
        elif self.value_type == "float32":
            df = df.with_columns([pl.col("value").cast(pl.Float32)])
        # If value_type is None or "auto", keep original data type for validation

        # Convert back to Arrow table
        return df.to_arrow()


class Chunked10xH5Reader(BaseChunkedReader):
    """
    Native chunked reader for 10x H5 file format.
    Uses h5py to read chunks directly from the H5 file without loading everything into memory.
    """

    def __init__(self, h5_path: str, value_type: str = "uint16"):
        super().__init__(h5_path)
        self.h5_path = h5_path
        self.value_type = value_type
        self._file = None
        self._matrix_dataset = None
        self._obs_names = None
        self._var_names = None

    def _open_file(self):
        """Open H5 file and set up for chunked reading"""
        self._file = h5py.File(self.h5_path, "r")

        # Try to find the matrix dataset - handle both 10x H5 and regular h5ad formats
        if "X" in self._file:
            # Regular h5ad format
            self._matrix_dataset = self._file["X"]
        elif "matrix" in self._file:
            # 10x H5 format
            matrix_group = self._file["matrix"]
            if "data" in matrix_group:
                self._matrix_dataset = matrix_group["data"]
            else:
                # Try to find the matrix in other locations
                for key in matrix_group.keys():
                    if isinstance(matrix_group[key], h5py.Dataset):
                        self._matrix_dataset = matrix_group[key]
                        break
        else:
            # Try to find matrix in root
            for key in self._file.keys():
                if isinstance(self._file[key], h5py.Dataset):
                    self._matrix_dataset = self._file[key]
                    break

        if self._matrix_dataset is None:
            raise ValueError("Could not find matrix dataset in H5 file")

    def _close_file(self):
        if self._file:
            self._file.close()
            self._file = None
        self._matrix_dataset = None
        self._obs_names = None
        self._var_names = None

    @property
    def n_obs(self) -> int:
        if self._n_obs is None:
            if self._matrix_dataset is None:
                raise RuntimeError("File not opened")
            # Handle dense or sparse
            if hasattr(self._matrix_dataset, "shape"):
                # Dense
                self._n_obs = self._matrix_dataset.shape[0]
            elif isinstance(self._matrix_dataset, h5py.Group):
                # Sparse CSR: shape is (n_obs, n_vars)
                indptr = self._matrix_dataset["indptr"][:]
                self._n_obs = len(indptr) - 1
            else:
                raise ValueError("Unknown matrix dataset type for n_obs")
        return self._n_obs

    @property
    def n_vars(self) -> int:
        if self._n_vars is None:
            if self._matrix_dataset is None:
                raise RuntimeError("File not opened")
            if hasattr(self._matrix_dataset, "shape"):
                # Dense
                self._n_vars = self._matrix_dataset.shape[1]
            elif isinstance(self._matrix_dataset, h5py.Group):
                # Sparse CSR: infer from indices
                indices = self._matrix_dataset["indices"][:]
                self._n_vars = int(indices.max()) + 1 if len(indices) > 0 else 0
            else:
                raise ValueError("Unknown matrix dataset type for n_vars")
        return self._n_vars

    @property
    def obs_names(self) -> np.ndarray:
        if self._obs_names is None:
            if self._file is None:
                raise RuntimeError("File not opened")

            # Try to find cell barcodes - handle both formats
            if "obs" in self._file and "_index" in self._file["obs"]:
                # Regular h5ad format
                self._obs_names = self._file["obs"]["_index"][:]
            elif "matrix" in self._file and "barcodes" in self._file["matrix"]:
                # 10x H5 format
                self._obs_names = self._file["matrix"]["barcodes"][:]
            elif "cell_names" in self._file:
                self._obs_names = self._file["cell_names"][:]
            else:
                # Generate default names
                self._obs_names = np.array([f"cell_{i}" for i in range(self.n_obs)])

            # Convert to string if needed
            if self._obs_names.dtype.kind in ("S", "O", "U"):
                self._obs_names = self._obs_names.astype(str)

        return self._obs_names

    @property
    def var_names(self) -> np.ndarray:
        if self._var_names is None:
            if self._file is None:
                raise RuntimeError("File not opened")

            # Try to find gene names - handle both formats
            if "var" in self._file and "_index" in self._file["var"]:
                # Regular h5ad format
                self._var_names = self._file["var"]["_index"][:]
            elif "matrix" in self._file and "features" in self._file["matrix"]:
                # 10x H5 format
                features = self._file["matrix"]["features"]
                if "name" in features:
                    self._var_names = features["name"][:]
                elif "id" in features:
                    self._var_names = features["id"][:]
                else:
                    # Use first available field
                    for key in features.keys():
                        if isinstance(features[key], h5py.Dataset):
                            self._var_names = features[key][:]
                            break
            elif "gene_names" in self._file:
                self._var_names = self._file["gene_names"][:]
            else:
                # Generate default names
                self._var_names = np.array([f"gene_{i}" for i in range(self.n_vars)])

            # Convert to string if needed
            if self._var_names.dtype.kind in ("S", "O", "U"):
                self._var_names = self._var_names.astype(str)

        return self._var_names

    def get_obs_metadata(self) -> pd.DataFrame:
        """Get observation metadata"""
        if self._file is None:
            raise RuntimeError("File not opened")

        # Try to find cell metadata
        metadata = {}
        if "matrix" in self._file and "barcodes" in self._file["matrix"]:
            # 10x H5 format
            pass  # No additional metadata in standard 10x H5
        else:
            # Look for other metadata
            for key in self._file.keys():
                if key not in ["matrix", "cell_names", "gene_names"]:
                    dataset = self._file[key]
                    if (
                        isinstance(dataset, h5py.Dataset)
                        and dataset.shape[0] == self.n_obs
                    ):
                        metadata[key] = dataset[:]

        return pd.DataFrame(metadata, index=self.obs_names)

    def get_var_metadata(self) -> pd.DataFrame:
        """Get variable metadata"""
        if self._file is None:
            raise RuntimeError("File not opened")

        # Try to find gene metadata
        metadata = {}
        if "matrix" in self._file and "features" in self._file["matrix"]:
            features = self._file["matrix"]["features"]
            for key in features.keys():
                if (
                    isinstance(features[key], h5py.Dataset)
                    and features[key].shape[0] == self.n_vars
                ):
                    metadata[key] = features[key][:]
        else:
            # Look for other metadata
            for key in self._file.keys():
                if key not in ["matrix", "cell_names", "gene_names"]:
                    dataset = self._file[key]
                    if (
                        isinstance(dataset, h5py.Dataset)
                        and dataset.shape[0] == self.n_vars
                    ):
                        metadata[key] = dataset[:]

        return pd.DataFrame(metadata, index=self.var_names)

    def iter_chunks(self, chunk_size: int = 1000, obs_chunk: bool = True):
        """Iterate over chunks by reading directly from H5 dataset and returning Arrow tables"""
        if not obs_chunk:
            raise NotImplementedError("Variable chunking not supported for H5 files")

        if self._matrix_dataset is None:
            raise RuntimeError("File not opened")

        total_obs = self.n_obs
        for start in range(0, total_obs, chunk_size):
            end = min(start + chunk_size, total_obs)

            if hasattr(self._matrix_dataset, "shape"):
                # Dense
                chunk_data = self._matrix_dataset[start:end, :]
                # Convert dense matrix to COO format
                coo = sparse.coo_matrix(chunk_data)

                if coo.nnz > 0:
                    # Create Arrow arrays directly
                    cell_integer_ids = coo.row.astype(np.uint32) + start
                    gene_integer_ids = coo.col.astype(np.uint16)
                    # Use the specified value type
                    if self.value_type == "uint16":
                        values = coo.data.astype(np.uint16)
                    elif self.value_type == "float32":
                        values = coo.data.astype(np.float32)
                    else:
                        raise ValueError(f"Unsupported value type: {self.value_type}")

                    # Use the specified value type for Arrow array
                    if self.value_type == "uint16":
                        value_pa_type = pa.uint16()
                    elif self.value_type == "float32":
                        value_pa_type = pa.float32()
                    else:
                        raise ValueError(f"Unsupported value type: {self.value_type}")

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": pa.array(gene_integer_ids),
                            "value": pa.array(values, type=value_pa_type),
                        }
                    )
                else:
                    # Empty chunk
                    # Use the specified value type for Arrow array
                    if self.value_type == "uint16":
                        value_pa_type = pa.uint16()
                    elif self.value_type == "float32":
                        value_pa_type = pa.float32()
                    else:
                        raise ValueError(f"Unsupported value type: {self.value_type}")

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array([], type=pa.uint32()),
                            "gene_integer_id": pa.array([], type=pa.uint16()),
                            "value": pa.array([], type=value_pa_type),
                        }
                    )
            elif isinstance(self._matrix_dataset, h5py.Group):
                # Sparse CSR
                indptr = self._matrix_dataset["indptr"][start : end + 1]
                start_idx = indptr[0]
                end_idx = indptr[-1]
                data = self._matrix_dataset["data"][start_idx:end_idx]
                indices = self._matrix_dataset["indices"][start_idx:end_idx]
                indptr = indptr - start_idx

                # Convert to COO format for Arrow table
                # Create row indices from indptr
                row_indices = []
                for i in range(len(indptr) - 1):
                    row_indices.extend([i] * (indptr[i + 1] - indptr[i]))

                if row_indices:
                    # Create Arrow arrays directly
                    cell_integer_ids = np.array(row_indices, dtype=np.uint32) + start
                    gene_integer_ids = indices.astype(np.uint16)
                    # Use the specified value type
                    if self.value_type == "uint16":
                        values = data.astype(np.uint16)
                    elif self.value_type == "float32":
                        values = data.astype(np.float32)
                    else:
                        raise ValueError(f"Unsupported value type: {self.value_type}")

                    # Use the specified value type for Arrow array
                    if self.value_type == "uint16":
                        value_pa_type = pa.uint16()
                    elif self.value_type == "float32":
                        value_pa_type = pa.float32()
                    else:
                        raise ValueError(f"Unsupported value type: {self.value_type}")

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": pa.array(gene_integer_ids),
                            "value": pa.array(values, type=value_pa_type),
                        }
                    )
                else:
                    # Empty chunk
                    # Use the specified value type for Arrow array
                    if self.value_type == "uint16":
                        value_pa_type = pa.uint16()
                    elif self.value_type == "float32":
                        value_pa_type = pa.float32()
                    else:
                        raise ValueError(f"Unsupported value type: {self.value_type}")

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array([], type=pa.uint32()),
                            "gene_integer_id": pa.array([], type=pa.uint16()),
                            "value": pa.array([], type=value_pa_type),
                        }
                    )
            else:
                raise ValueError("Unknown matrix dataset type for chunking")

            yield chunk_table, slice(start, end)

    def _get_chunk_impl(self, obs_slice: slice, var_slice: slice):
        """Get a specific chunk by reading directly from H5 dataset"""
        if self._matrix_dataset is None:
            raise RuntimeError("File not opened")
        start_row = obs_slice.start or 0
        end_row = obs_slice.stop or self.n_obs
        start_col = var_slice.start or 0
        end_col = var_slice.stop or self.n_vars

        if hasattr(self._matrix_dataset, "shape"):
            # Dense
            chunk_data = self._matrix_dataset[start_row:end_row, start_col:end_col]
            # Convert dense matrix to COO format
            coo = sparse.coo_matrix(chunk_data)

            if coo.nnz > 0:
                # Create Arrow arrays directly
                cell_integer_ids = coo.row.astype(np.uint32) + start_row
                gene_integer_ids = coo.col.astype(np.uint16) + start_col
                # Use the specified value type
                if self.value_type == "uint16":
                    values = coo.data.astype(np.uint16)
                elif self.value_type == "float32":
                    values = coo.data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                # Use the specified value type for Arrow array
                if self.value_type == "uint16":
                    value_pa_type = pa.uint16()
                elif self.value_type == "float32":
                    value_pa_type = pa.float32()
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values, type=value_pa_type),
                    }
                )
            else:
                # Empty chunk
                # Use the specified value type for Arrow array
                if self.value_type == "uint16":
                    value_pa_type = pa.uint16()
                elif self.value_type == "float32":
                    value_pa_type = pa.float32()
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )
        elif isinstance(self._matrix_dataset, h5py.Group):
            # Sparse CSR
            indptr = self._matrix_dataset["indptr"][start_row : end_row + 1]
            start_idx = indptr[0]
            end_idx = indptr[-1]
            data = self._matrix_dataset["data"][start_idx:end_idx]
            indices = self._matrix_dataset["indices"][start_idx:end_idx]
            indptr = indptr - start_idx

            # Convert to COO format for Arrow table
            # Create row indices from indptr
            row_indices = []
            for i in range(len(indptr) - 1):
                row_indices.extend([i] * (indptr[i + 1] - indptr[i]))

            if row_indices:
                # Create Arrow arrays directly
                cell_integer_ids = np.array(row_indices, dtype=np.uint32) + start_row
                gene_integer_ids = indices.astype(np.uint16)
                # Use the specified value type
                if self.value_type == "uint16":
                    values = data.astype(np.uint16)
                elif self.value_type == "float32":
                    values = data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                # Use the specified value type for Arrow array
                if self.value_type == "uint16":
                    value_pa_type = pa.uint16()
                elif self.value_type == "float32":
                    value_pa_type = pa.float32()
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values, type=value_pa_type),
                    }
                )

                # Apply variable slicing if needed
                if start_col != 0 or end_col != self.n_vars:
                    # Filter by gene_integer_id
                    gene_integer_ids = chunk_table.column("gene_integer_id").to_numpy()
                    mask = (gene_integer_ids >= start_col) & (
                        gene_integer_ids < end_col
                    )
                    chunk_table = chunk_table.filter(pa.array(mask))

                    # Adjust gene_integer_id to be relative to the slice
                    if start_col > 0:
                        gene_integer_ids = (
                            chunk_table.column("gene_integer_id").to_numpy() - start_col
                        )
                        chunk_table = chunk_table.set_column(
                            1, "gene_integer_id", pa.array(gene_integer_ids)
                        )
            else:
                # Empty chunk
                # Use the specified value type for Arrow array
                if self.value_type == "uint16":
                    value_pa_type = pa.uint16()
                elif self.value_type == "float32":
                    value_pa_type = pa.float32()
                else:
                    raise ValueError(f"Unsupported value type: {self.value_type}")

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=value_pa_type),
                    }
                )
        else:
            raise ValueError("Unknown matrix dataset type for chunking")

        return chunk_table


def create_chunked_reader(
    file_path: str,
    chunk_size: int = 25000,
    value_type: str = "uint16",
    collection_name: str = "RNA",
) -> BaseChunkedReader:
    """
    Factory function to create the appropriate chunked reader based on file format.

    Parameters:
    -----------
    file_path : str
        Path to the data file or directory
    chunk_size : int
        Size of chunks for processing. Default: 25,000 elements.
    value_type : str
        Data type for expression values. Default: "uint16".
    collection_name : str
        Name of the measurement collection for TileDB format. Default: "RNA".

    Returns:
    --------
    BaseChunkedReader
        Appropriate chunked reader for the file format

    Raises:
    -------
    ValueError
        If the file format is not supported
    """
    # Import here to avoid circular imports
    from .utils import detect_format

    try:
        format_type = detect_format(file_path)
    except ValueError as e:
        raise ValueError(f"Cannot create chunked reader: {e}") from e

    if format_type == "h5ad":
        return ChunkedH5ADReader(
            file_path, chunk_size=chunk_size, value_type=value_type
        )
    elif format_type == "10x_mtx":
        return Chunked10xMTXReader(file_path, value_type=value_type)
    elif format_type == "10x_h5":
        return Chunked10xH5Reader(file_path, value_type=value_type)
    elif format_type == "tiledb":
        # For TileDB, use "auto" as default to preserve original data types for validation
        tiledb_value_type = value_type if value_type != "uint16" else "auto"
        return ChunkedTileDBReader(
            file_path, collection_name=collection_name, value_type=tiledb_value_type
        )
    else:
        raise ValueError(f"Unsupported format for chunked reading: {format_type}")
