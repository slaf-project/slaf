import gzip
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
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

        Parameters:
        -----------
        file_path : str
            Path to the data file or directory
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
        """Number of observations (cells)"""
        pass

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Number of variables (genes)"""
        pass

    @property
    @abstractmethod
    def obs_names(self) -> np.ndarray:
        """Get observation names (cell barcodes)"""
        pass

    @property
    @abstractmethod
    def var_names(self) -> np.ndarray:
        """Get variable names (gene names)"""
        pass

    @abstractmethod
    def get_obs_metadata(self) -> pd.DataFrame:
        """Get observation metadata as pandas DataFrame"""
        pass

    @abstractmethod
    def get_var_metadata(self) -> pd.DataFrame:
        """Get variable metadata as pandas DataFrame"""
        pass

    @abstractmethod
    def iter_chunks(
        self, chunk_size: int = 1000, obs_chunk: bool = True
    ) -> Iterator[tuple[np.ndarray | sparse.csr_matrix, slice]]:
        """
        Iterate over data in chunks.

        Parameters:
        -----------
        chunk_size : int
            Number of observations per chunk
        obs_chunk : bool
            If True, chunk by observations (cells). If False, chunk by variables (genes).

        Yields:
        -------
        tuple
            (chunk_data, slice) where chunk_data is the data matrix and slice is the
            slice object indicating the chunk boundaries
        """
        pass

    def get_chunk(
        self, obs_slice: slice | None = None, var_slice: slice | None = None
    ) -> np.ndarray | sparse.csr_matrix:
        """
        Get a specific chunk of data.

        Parameters:
        -----------
        obs_slice : slice, optional
            Slice for observations (cells)
        var_slice : slice, optional
            Slice for variables (genes)

        Returns:
        --------
        np.ndarray or sparse.csr_matrix
            The requested data chunk
        """
        if obs_slice is None:
            obs_slice = slice(0, self.n_obs)
        if var_slice is None:
            var_slice = slice(0, self.n_vars)

        return self._get_chunk_impl(obs_slice, var_slice)

    @abstractmethod
    def _get_chunk_impl(
        self, obs_slice: slice, var_slice: slice
    ) -> np.ndarray | sparse.csr_matrix:
        """Implementation of chunk retrieval. Called by get_chunk."""
        pass


class ChunkedH5ADReader(BaseChunkedReader):
    """
    A chunked reader for h5ad files using h5py for memory-efficient processing.
    """

    def __init__(self, filename: str):
        """
        Initialize the chunked reader.

        Parameters:
        -----------
        filename : str
            Path to the h5ad file
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
        """Number of observations (cells)"""
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
        """Number of variables (genes)"""
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
            return df

    def _read_sparse_chunk(self, start_row: int, end_row: int) -> sparse.csr_matrix:
        """Read a chunk from sparse matrix format"""
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

            chunk_shape = (end_row - start_row, self.n_vars)
            return sparse.csr_matrix((data, indices, indptr), shape=chunk_shape)
        else:
            raise ValueError("Unsupported sparse matrix format")

    def _read_dense_chunk(self, start_row: int, end_row: int) -> np.ndarray:
        """Read a chunk from dense matrix format"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        X_dataset = self.file["X"]
        return X_dataset[start_row:end_row, :]

    def iter_chunks(
        self, chunk_size: int = 1000, obs_chunk: bool = True
    ) -> Iterator[tuple[np.ndarray | sparse.csr_matrix, slice]]:
        """
        Iterate over data in chunks.

        Parameters:
        -----------
        chunk_size : int
            Number of observations per chunk
        obs_chunk : bool
            If True, chunk by observations (cells). If False, chunk by variables (genes).

        Yields:
        -------
        tuple
            (chunk_data, slice) where chunk_data is the data matrix and slice is the
            slice object indicating the chunk boundaries
        """
        if obs_chunk:
            total_obs = self.n_obs
            for start in range(0, total_obs, chunk_size):
                end = min(start + chunk_size, total_obs)
                chunk = (
                    self._read_sparse_chunk(start, end)
                    if self._is_sparse()
                    else self._read_dense_chunk(start, end)
                )
                yield chunk, slice(start, end)
        else:
            # Variable chunking not implemented for h5ad
            raise NotImplementedError("Variable chunking not supported for h5ad files")

    def _get_chunk_impl(
        self, obs_slice: slice, var_slice: slice
    ) -> np.ndarray | sparse.csr_matrix:
        """Implementation of chunk retrieval for h5ad files"""
        if self._is_sparse():
            # For sparse matrices, we need to handle slicing differently
            # This is a simplified implementation - in practice, you might want
            # to implement proper sparse matrix slicing
            start_row = obs_slice.start or 0
            end_row = obs_slice.stop or self.n_obs
            chunk = self._read_sparse_chunk(start_row, end_row)

            # Apply variable slicing if needed
            if var_slice.start != 0 or var_slice.stop != self.n_vars:
                chunk = chunk[:, var_slice]

            return chunk
        else:
            # For dense matrices, use direct slicing
            start_row = obs_slice.start or 0
            end_row = obs_slice.stop or self.n_obs
            start_col = var_slice.start or 0
            end_col = var_slice.stop or self.n_vars

            if self.file is None:
                raise RuntimeError("File not opened")
            return self.file["X"][start_row:end_row, start_col:end_col]

    def _is_sparse(self) -> bool:
        """Check if the data is stored in sparse format"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        X_group = self.file["X"]
        return isinstance(X_group, h5py.Group) and "data" in X_group

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
            if gene in var_names:
                gene_indices.append(np.where(var_names == gene)[0][0])
            else:
                warnings.warn(f"Gene {gene} not found in dataset", stacklevel=2)

        if not gene_indices:
            return

        # Iterate over chunks and extract gene expression
        for chunk, obs_slice in self.iter_chunks(chunk_size=chunk_size):
            if self._is_sparse():
                # For sparse matrices, extract the specific genes
                chunk_genes = chunk[:, gene_indices].toarray()
            else:
                chunk_genes = chunk[:, gene_indices]

            # Create DataFrame
            chunk_df = pd.DataFrame(
                chunk_genes,
                index=self.obs_names[obs_slice],
                columns=[gene_names[i] for i in range(len(gene_indices))],
            )
            yield chunk_df


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
        return self._barcodes

    @property
    def var_names(self) -> np.ndarray:
        return self._genes

    def get_obs_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(index=self._barcodes)

    def get_var_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(index=self._genes)

    def iter_chunks(self, chunk_size: int = 1000, obs_chunk: bool = True):
        """Iterate over chunks by reading MTX file directly"""
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
            entries.append((int(row) - 1, int(col) - 1, val))

        # Sort by column (cell) for efficient chunking
        entries.sort(key=lambda x: x[1])

        # Yield chunks
        n_obs = self._n_obs
        if n_obs is None:
            raise RuntimeError("n_obs not initialized")

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)

            # Filter entries for this chunk
            chunk_entries = [
                (row, col - start, val)
                for row, col, val in entries
                if start <= col < end
            ]

            # Convert to sparse matrix
            if chunk_entries:
                rows, cols, vals = zip(*chunk_entries, strict=False)
                chunk = sparse.csr_matrix(
                    (vals, (rows, cols)), shape=(self._n_vars, end - start)
                ).T  # Transpose to get (cells, genes)
            else:
                chunk = sparse.csr_matrix((end - start, self._n_vars))

            yield chunk, slice(start, end)

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
            row_idx, col_idx = int(row) - 1, int(col) - 1
            if start_row <= col_idx < end_row:
                entries.append((row_idx, col_idx - start_row, val))

        # Convert to sparse matrix
        n_vars = self._n_vars
        if n_vars is None:
            raise RuntimeError("n_vars not initialized")

        if entries:
            rows, cols, vals = zip(*entries, strict=False)
            chunk = sparse.csr_matrix(
                (vals, (rows, cols)), shape=(n_vars, end_row - start_row)
            ).T
        else:
            chunk = sparse.csr_matrix((end_row - start_row, n_vars))

        # Apply variable slicing if needed
        n_vars = self._n_vars
        if n_vars is None:
            raise RuntimeError("n_vars not initialized")
        if var_slice.start != 0 or var_slice.stop != n_vars:
            chunk = chunk[:, var_slice]

        return chunk


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
        """Iterate over chunks by reading directly from H5 dataset"""
        if not obs_chunk:
            raise NotImplementedError("Variable chunking not supported for H5 files")

        if self._matrix_dataset is None:
            raise RuntimeError("File not opened")

        total_obs = self.n_obs
        for start in range(0, total_obs, chunk_size):
            end = min(start + chunk_size, total_obs)
            if hasattr(self._matrix_dataset, "shape"):
                # Dense
                chunk = self._matrix_dataset[start:end, :]
            elif isinstance(self._matrix_dataset, h5py.Group):
                # Sparse CSR
                indptr = self._matrix_dataset["indptr"][start : end + 1]
                start_idx = indptr[0]
                end_idx = indptr[-1]
                data = self._matrix_dataset["data"][start_idx:end_idx]
                indices = self._matrix_dataset["indices"][start_idx:end_idx]
                indptr = indptr - start_idx
                chunk_shape = (end - start, self.n_vars)
                from scipy import sparse

                chunk = sparse.csr_matrix((data, indices, indptr), shape=chunk_shape)
            else:
                raise ValueError("Unknown matrix dataset type for chunking")
            yield chunk, slice(start, end)

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
            chunk = self._matrix_dataset[start_row:end_row, start_col:end_col]
        elif isinstance(self._matrix_dataset, h5py.Group):
            # Sparse CSR
            indptr = self._matrix_dataset["indptr"][start_row : end_row + 1]
            start_idx = indptr[0]
            end_idx = indptr[-1]
            data = self._matrix_dataset["data"][start_idx:end_idx]
            indices = self._matrix_dataset["indices"][start_idx:end_idx]
            indptr = indptr - start_idx
            chunk_shape = (end_row - start_row, self.n_vars)
            from scipy import sparse

            chunk = sparse.csr_matrix((data, indices, indptr), shape=chunk_shape)
            if start_col != 0 or end_col != self.n_vars:
                chunk = chunk[:, start_col:end_col]
        else:
            raise ValueError("Unknown matrix dataset type for chunking")
        return chunk


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
