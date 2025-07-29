import gzip
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
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
    A chunked reader for h5ad files using h5py for memory-efficient processing.

    This reader provides memory-efficient access to h5ad files by reading data
    in chunks rather than loading the entire dataset into memory. It supports
    both sparse and dense expression matrices.

    Key Features:
        - Memory-efficient chunked reading
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

    def __init__(self, filename: str):
        """
        Initialize the chunked h5ad reader.

        Args:
            filename: Path to the h5ad file. Must be a valid h5ad file with
                     proper AnnData structure.

        Raises:
            FileNotFoundError: If the h5ad file doesn't exist.
            ValueError: If the file is not a valid h5ad format.

        Examples:
            >>> # Initialize with existing file
            >>> reader = ChunkedH5ADReader("pbmc3k.h5ad")
            >>> print(f"File path: {reader.file_path}")
            File path: pbmc3k.h5ad

            >>> # Error handling for missing file
            >>> try:
            ...     reader = ChunkedH5ADReader("nonexistent.h5ad")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
            Error: [Errno 2] No such file or directory: 'nonexistent.h5ad'
        """
        super().__init__(filename)
        self.file: h5py.File | None = None

    def _open_file(self) -> None:
        """Open the h5ad file"""
        self.file = h5py.File(self.file_path, "r")

    def _close_file(self) -> None:
        """Close the h5ad file"""
        if self.file:
            self.file.close()
            self.file = None

    @property
    def n_obs(self) -> int:
        """
        Number of observations (cells) in the h5ad dataset.

        Returns:
            Total number of cells in the dataset.

        Raises:
            RuntimeError: If the file is not opened (use context manager).
            ValueError: If the dataset structure is invalid.
        """
        if self._n_obs is None:
            if self.file is None:
                raise RuntimeError("File not opened. Use context manager.")
            # For sparse matrices, we need to infer shape from indptr
            if self._is_sparse():
                indptr = self.file["X"]["indptr"][:]
                # Type assertion for h5py dataset
                if isinstance(indptr, np.ndarray):
                    self._n_obs = int(len(indptr)) - 1
                else:
                    raise ValueError("Expected numpy array for indptr")
            else:
                # For dense matrices, use shape directly
                X_dataset = self.file["X"]
                if isinstance(X_dataset, h5py.Dataset):
                    self._n_obs = int(X_dataset.shape[0])
                else:
                    raise ValueError("Expected h5py.Dataset for X")
        return self._n_obs

    @property
    def n_vars(self) -> int:
        """
        Number of variables (genes) in the h5ad dataset.

        Returns:
            Total number of genes in the dataset.

        Raises:
            RuntimeError: If the file is not opened (use context manager).
            ValueError: If the dataset structure is invalid.
        """
        if self._n_vars is None:
            if self.file is None:
                raise RuntimeError("File not opened. Use context manager.")
            # For sparse matrices, we need to infer shape from indices
            if self._is_sparse():
                indices = self.file["X"]["indices"][:]
                # Type assertion for h5py dataset
                if isinstance(indices, np.ndarray):
                    max_idx = int(indices.max()) if len(indices) > 0 else -1
                    self._n_vars = max_idx + 1 if max_idx >= 0 else 0
                else:
                    raise ValueError("Expected numpy array for indices")
            else:
                # For dense matrices, use shape directly
                X_dataset = self.file["X"]
                if isinstance(X_dataset, h5py.Dataset):
                    self._n_vars = int(X_dataset.shape[1])
                else:
                    raise ValueError("Expected h5py.Dataset for X")
        return self._n_vars

    @property
    def obs_names(self) -> np.ndarray:
        """Get observation names (cell barcodes)"""
        if self._obs_names is None:
            if self.file is None:
                raise RuntimeError("File not opened. Use context manager.")
            obs_group = self.file["obs"]
            if isinstance(obs_group, h5py.Group) and "_index" in obs_group:
                index_dataset = obs_group["_index"]
                if isinstance(index_dataset, h5py.Dataset):
                    self._obs_names = index_dataset[:]
                    # Handle bytes to string conversion if needed
                    if (
                        hasattr(self._obs_names, "dtype")
                        and self._obs_names.dtype.kind == "S"
                    ):
                        self._obs_names = self._obs_names.astype(str)
                else:
                    raise ValueError("Expected h5py.Dataset for _index")
            else:
                self._obs_names = np.array([f"cell_{i}" for i in range(self.n_obs)])
        assert self._obs_names is not None
        return self._obs_names

    @property
    def var_names(self) -> np.ndarray:
        """Get variable names (gene names)"""
        if self._var_names is None:
            if self.file is None:
                raise RuntimeError("File not opened. Use context manager.")
            var_group = self.file["var"]
            if isinstance(var_group, h5py.Group) and "_index" in var_group:
                index_dataset = var_group["_index"]
                if isinstance(index_dataset, h5py.Dataset):
                    self._var_names = index_dataset[:]
                    # Handle bytes to string conversion if needed
                    if (
                        hasattr(self._var_names, "dtype")
                        and self._var_names.dtype.kind == "S"
                    ):
                        self._var_names = self._var_names.astype(str)
                else:
                    raise ValueError("Expected h5py.Dataset for _index")
            elif isinstance(var_group, h5py.Group) and "gene_id" in var_group:
                # Handle case where gene names are stored in gene_id column
                gene_id_dataset = var_group["gene_id"]
                if isinstance(gene_id_dataset, h5py.Dataset):
                    self._var_names = gene_id_dataset[:]
                    # Handle bytes to string conversion if needed
                    if (
                        hasattr(self._var_names, "dtype")
                        and self._var_names.dtype.kind == "S"
                    ):
                        self._var_names = self._var_names.astype(str)
                else:
                    raise ValueError("Expected h5py.Dataset for gene_id")
            else:
                self._var_names = np.array([f"gene_{i}" for i in range(self.n_vars)])
        assert self._var_names is not None
        return self._var_names

    def get_obs_metadata(self) -> pd.DataFrame:
        """Get observation metadata as pandas DataFrame"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        obs = self.file["obs"]

        if isinstance(obs, h5py.Dataset):
            arr = obs[:]
            if arr.dtype.fields is None or len(arr.dtype.fields) == 0:
                obs = self.file["obs"]
            else:
                df = pd.DataFrame.from_records(arr)
                if "_index" in df.columns:
                    df.index = df["_index"].astype(str)
                    df = df.drop(columns=["_index"])
                return df
        # Group-of-datasets logic
        obs_data = {}
        for key in obs.keys():  # type: ignore
            item = obs[key]  # type: ignore
            if isinstance(item, h5py.Dataset) and key != "_index":
                data = item[:]
                if data.dtype.kind in ("S", "O", "U"):
                    data = data.astype(str)
                obs_data[key] = data
            elif isinstance(item, h5py.Group):
                # AnnData categorical encoding: group with 'categories' and 'codes'
                if "categories" in item and "codes" in item:
                    categories = item["categories"][:]  # type: ignore
                    if categories.dtype.kind in ("S", "O", "U"):
                        categories = categories.astype(str)
                    codes = item["codes"][:]  # type: ignore
                    # Map codes to categories, handling -1 as missing
                    col = np.array(
                        [
                            categories[c] if c >= 0 and c < len(categories) else None
                            for c in codes
                        ],
                        dtype=object,
                    )
                    obs_data[key] = col
                # else: ignore other group formats for now
        # Handle index
        if "_index" in obs:
            index = obs["_index"][:]
            if hasattr(index, "dtype") and index.dtype.kind in ("S", "O", "U"):
                index = index.astype(str)
            else:
                index = index.astype(str)
            df = pd.DataFrame(obs_data, index=index)
            df.index.name = "index"  # Match scanpy behavior
            return df
        elif "index" in obs_data:
            # Handle case where 'index' is a regular column (like in this dataset)
            index_values = obs_data.pop("index")
            df = pd.DataFrame(obs_data, index=index_values)
            df.index.name = "index"  # Match scanpy behavior
            return df
        else:
            df = pd.DataFrame(obs_data)
            # Set the index to the actual cell names
            df.index = self.obs_names
            return df

    def get_var_metadata(self) -> pd.DataFrame:
        """Get variable metadata as pandas DataFrame"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        var = self.file["var"]
        if isinstance(var, h5py.Dataset):
            arr = var[:]
            if arr.dtype.fields is None or len(arr.dtype.fields) == 0:
                var = self.file["var"]
            else:
                df = pd.DataFrame.from_records(arr)
                if "_index" in df.columns:
                    df.index = df["_index"].astype(str)
                    df = df.drop(columns=["_index"])
                return df
        var_data = {}
        for key in var.keys():
            item = var[key]
            if isinstance(item, h5py.Dataset) and key != "_index":
                data = item[:]
                if data.dtype.kind in ("S", "O", "U"):
                    data = data.astype(str)
                var_data[key] = data
        if "_index" in var:
            index = var["_index"][:]
            if index.dtype.kind in ("S", "O", "U"):
                index = index.astype(str)
            df = pd.DataFrame(var_data, index=index)
            df.index.name = "index"  # Match scanpy behavior
            return df
        elif "index" in var_data:
            # Handle case where 'index' is a regular column (like in this dataset)
            index_values = var_data.pop("index")
            df = pd.DataFrame(var_data, index=index_values)
            df.index.name = "index"  # Match scanpy behavior
            return df
        else:
            df = pd.DataFrame(var_data)
            # Set the index to the actual gene names
            df.index = self.var_names
            return df

    def _is_sparse(self) -> bool:
        """Check if the data is stored in sparse format"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        X_group = self.file["X"]
        return isinstance(X_group, h5py.Group) and "data" in X_group

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
        if self._is_sparse():
            return self._read_sparse_chunk_as_arrow(start_row, end_row)
        else:
            return self._read_dense_chunk_as_arrow(start_row, end_row)

    def _read_sparse_chunk_as_arrow(self, start_row: int, end_row: int) -> pa.Table:
        """Read a sparse chunk and return as Arrow table directly"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        X_group = self.file["X"]

        if "data" in X_group and "indices" in X_group and "indptr" in X_group:
            # CSR format
            indptr = X_group["indptr"][start_row : end_row + 1]
            start_idx = indptr[0]
            end_idx = indptr[-1]

            data = X_group["data"][start_idx:end_idx]
            indices = X_group["indices"][start_idx:end_idx]

            # Adjust indptr to start from 0
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
                values = data.astype(np.uint16)

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
        else:
            raise ValueError("Unsupported sparse matrix format")

    def _read_dense_chunk_as_arrow(self, start_row: int, end_row: int) -> pa.Table:
        """Read a dense chunk and return as Arrow table directly"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        X_dataset = self.file["X"]
        chunk_data = X_dataset[start_row:end_row, :]

        # Convert dense matrix to COO format
        coo = sparse.coo_matrix(chunk_data)

        if coo.nnz > 0:
            # Create Arrow arrays directly
            cell_integer_ids = coo.row.astype(np.uint32) + start_row
            gene_integer_ids = coo.col.astype(np.uint16)
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

    def _get_chunk_impl(self, obs_slice: slice, var_slice: slice) -> pa.Table:
        """Implementation of chunk retrieval for h5ad files"""
        start_row = obs_slice.start or 0
        end_row = obs_slice.stop or self.n_obs

        # Get the full chunk first
        chunk_table = self._read_chunk_as_arrow(start_row, end_row)

        # Apply variable slicing if needed
        if var_slice.start != 0 or var_slice.stop != self.n_vars:
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
        # Find gene indices
        var_names = self.var_names
        gene_indices = []
        for gene in gene_names:
            # Handle byte string to string conversion if needed
            if isinstance(gene, bytes):
                gene_str = gene.decode("utf-8")
            else:
                gene_str = gene

            # Handle case where var_names contains byte strings
            if var_names.dtype.kind == "S" or (
                hasattr(var_names, "dtype")
                and var_names.dtype.kind == "O"
                and any(isinstance(x, bytes) for x in var_names)
            ):
                # Convert var_names to strings for comparison
                var_names_str = np.array(
                    [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in var_names
                    ]
                )
                if gene_str in var_names_str:
                    gene_indices.append(np.where(var_names_str == gene_str)[0][0])
                else:
                    warnings.warn(f"Gene {gene} not found in dataset", stacklevel=2)
            else:
                if gene_str in var_names:
                    gene_indices.append(np.where(var_names == gene_str)[0][0])
                else:
                    warnings.warn(f"Gene {gene} not found in dataset", stacklevel=2)

        if not gene_indices:
            return

        # Iterate over chunks and extract gene expression
        for chunk_table, _obs_slice in self.iter_chunks(chunk_size=chunk_size):
            # Convert Arrow table to DataFrame for gene extraction
            chunk_df = chunk_table.to_pandas()

            # Filter for specific genes
            gene_mask = chunk_df["gene_integer_id"].isin(gene_indices)
            chunk_genes = chunk_df[gene_mask]

            # Always yield a DataFrame for this chunk, even if empty
            if not chunk_genes.empty:
                # Pivot to get genes as columns
                chunk_genes_pivot = chunk_genes.pivot(
                    index="cell_integer_id", columns="gene_integer_id", values="value"
                ).fillna(0)

                # Create DataFrame with gene names as columns
                chunk_df_final = pd.DataFrame(
                    chunk_genes_pivot.values,
                    index=[f"cell_{i}" for i in chunk_genes_pivot.index],
                    columns=[
                        gene_names[i]
                        for i in gene_indices
                        if i in chunk_genes_pivot.columns
                    ],
                )

                # Ensure all requested genes are present (fill with zeros if missing)
                for i, _gene_idx in enumerate(gene_indices):
                    if gene_names[i] not in chunk_df_final.columns:
                        chunk_df_final[gene_names[i]] = 0

                # Reorder columns to match the requested gene order
                chunk_df_final = chunk_df_final[[gene_names[i] for i in gene_indices]]
            else:
                # Create empty DataFrame with correct structure
                chunk_df_final = pd.DataFrame(
                    columns=[gene_names[i] for i in gene_indices]
                )

            yield chunk_df_final


class Chunked10xMTXReader(BaseChunkedReader):
    """
    Native chunked reader for 10x MTX directory format.
    Reads chunks directly from matrix.mtx without loading the entire matrix into memory.
    """

    def __init__(self, mtx_dir: str):
        super().__init__(mtx_dir)
        self.mtx_dir = Path(mtx_dir)
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
                values = np.array(vals, dtype=np.uint16)

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values),
                    }
                )
            else:
                # Empty chunk
                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=pa.uint16()),
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
            values = np.array(vals, dtype=np.uint16)

            chunk_table = pa.table(
                {
                    "cell_integer_id": pa.array(cell_integer_ids),
                    "gene_integer_id": pa.array(gene_integer_ids),
                    "value": pa.array(values),
                }
            )
        else:
            # Empty chunk
            chunk_table = pa.table(
                {
                    "cell_integer_id": pa.array([], type=pa.uint32()),
                    "gene_integer_id": pa.array([], type=pa.uint16()),
                    "value": pa.array([], type=pa.uint16()),
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


class Chunked10xH5Reader(BaseChunkedReader):
    """
    Native chunked reader for 10x H5 file format.
    Uses h5py to read chunks directly from the H5 file without loading everything into memory.
    """

    def __init__(self, h5_path: str):
        super().__init__(h5_path)
        self.h5_path = h5_path
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
                    values = coo.data.astype(np.uint16)

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": pa.array(gene_integer_ids),
                            "value": pa.array(values),
                        }
                    )
                else:
                    # Empty chunk
                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array([], type=pa.uint32()),
                            "gene_integer_id": pa.array([], type=pa.uint16()),
                            "value": pa.array([], type=pa.uint16()),
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
                    values = data.astype(np.uint16)

                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array(cell_integer_ids),
                            "gene_integer_id": pa.array(gene_integer_ids),
                            "value": pa.array(values),
                        }
                    )
                else:
                    # Empty chunk
                    chunk_table = pa.table(
                        {
                            "cell_integer_id": pa.array([], type=pa.uint32()),
                            "gene_integer_id": pa.array([], type=pa.uint16()),
                            "value": pa.array([], type=pa.uint16()),
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
                values = coo.data.astype(np.uint16)

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values),
                    }
                )
            else:
                # Empty chunk
                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=pa.uint16()),
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
                values = data.astype(np.uint16)

                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array(cell_integer_ids),
                        "gene_integer_id": pa.array(gene_integer_ids),
                        "value": pa.array(values),
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
                chunk_table = pa.table(
                    {
                        "cell_integer_id": pa.array([], type=pa.uint32()),
                        "gene_integer_id": pa.array([], type=pa.uint16()),
                        "value": pa.array([], type=pa.uint16()),
                    }
                )
        else:
            raise ValueError("Unknown matrix dataset type for chunking")

        return chunk_table


def create_chunked_reader(file_path: str) -> BaseChunkedReader:
    """
    Factory function to create the appropriate chunked reader based on file format.

    Parameters:
    -----------
    file_path : str
        Path to the data file or directory

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
        return ChunkedH5ADReader(file_path)
    elif format_type == "10x_mtx":
        return Chunked10xMTXReader(file_path)
    elif format_type == "10x_h5":
        return Chunked10xH5Reader(file_path)
    else:
        raise ValueError(f"Unsupported format for chunked reading: {format_type}")
