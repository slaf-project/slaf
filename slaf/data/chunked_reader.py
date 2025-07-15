import warnings
from collections.abc import Iterator

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


class ChunkedH5ADReader:
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
        self.filename = filename
        self.file: h5py.File | None = None
        self._n_obs: int | None = None
        self._n_vars: int | None = None
        self._obs_names: np.ndarray | None = None
        self._var_names: np.ndarray | None = None

    def __enter__(self):
        self.file = h5py.File(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

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
                    if self._obs_names.dtype.kind == "S":
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
                    if self._var_names.dtype.kind == "S":
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
        for key in obs.keys():
            item = obs[key]
            if isinstance(item, h5py.Dataset) and key != "_index":
                data = item[:]
                if data.dtype.kind in ("S", "O", "U"):
                    data = data.astype(str)
                obs_data[key] = data
            elif isinstance(item, h5py.Group):
                # AnnData categorical encoding: group with 'categories' and 'codes'
                if "categories" in item and "codes" in item:
                    categories = item["categories"][:]
                    if categories.dtype.kind in ("S", "O", "U"):
                        categories = categories.astype(str)
                    codes = item["codes"][:]
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
            raise ValueError("Sparse matrix format not recognized")

    def _read_dense_chunk(self, start_row: int, end_row: int) -> np.ndarray:
        """Read a chunk from dense matrix format"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        return self.file["X"][start_row:end_row, :]

    def iter_chunks(
        self, chunk_size: int = 1000, obs_chunk: bool = True
    ) -> Iterator[tuple[np.ndarray | sparse.csr_matrix, slice]]:
        """
        Iterate over chunks of the expression matrix.

        Parameters:
        -----------
        chunk_size : int
            Size of each chunk
        obs_chunk : bool
            If True, chunk by observations (rows). If False, chunk by variables (columns).

        Yields:
        -------
        chunk : np.ndarray or sparse.csr_matrix
            The data chunk
        slice_obj : slice
            The slice object indicating which rows/cols this chunk represents
        """
        if obs_chunk:
            total_size = self.n_obs
        else:
            total_size = self.n_vars

        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size, total_size)
            slice_obj = slice(start, end)

            if obs_chunk:
                # Check if data is sparse
                if self._is_sparse():
                    chunk = self._read_sparse_chunk(start, end)
                else:
                    chunk = self._read_dense_chunk(start, end)
            else:
                # Variable chunking (columns)
                if self._is_sparse():
                    # For sparse matrices, we need to convert to CSC first or handle differently
                    warnings.warn(
                        "Variable chunking with sparse matrices is not optimized",
                        stacklevel=2,
                    )
                    chunk = self._read_sparse_chunk(0, self.n_obs)[:, start:end]
                else:
                    if self.file is None:
                        raise RuntimeError("File not opened. Use context manager.")
                    chunk = self.file["X"][:, start:end]

            yield chunk, slice_obj

    def _is_sparse(self) -> bool:
        """Check if the matrix is stored in sparse format"""
        if self.file is None:
            raise RuntimeError("File not opened. Use context manager.")
        X_group = self.file["X"]
        return "data" in X_group and "indices" in X_group and "indptr" in X_group

    def get_chunk(
        self, obs_slice: slice | None = None, var_slice: slice | None = None
    ) -> np.ndarray | sparse.csr_matrix:
        """
        Get a specific chunk of data.

        Parameters:
        -----------
        obs_slice : slice, optional
            Slice for observations (rows)
        var_slice : slice, optional
            Slice for variables (columns)

        Returns:
        --------
        chunk : np.ndarray or sparse.csr_matrix
            The requested data chunk
        """
        if obs_slice is None:
            obs_slice = slice(None)
        if var_slice is None:
            var_slice = slice(None)

        if self._is_sparse():
            # For sparse matrices, we need to be more careful
            start_row = obs_slice.start or 0
            stop_row = obs_slice.stop or self.n_obs
            chunk = self._read_sparse_chunk(start_row, stop_row)

            if var_slice != slice(None):
                chunk = chunk[:, var_slice]
        else:
            if self.file is None:
                raise RuntimeError("File not opened. Use context manager.")
            chunk = self.file["X"][obs_slice, var_slice]

        return chunk

    def get_gene_expression(
        self, gene_names: list, chunk_size: int = 1000
    ) -> Iterator[pd.DataFrame]:
        """
        Get expression data for specific genes in chunks.

        Parameters:
        -----------
        gene_names : list
            List of gene names to extract
        chunk_size : int
            Size of each chunk

        Yields:
        -------
        chunk_df : pd.DataFrame
            DataFrame with cells as rows and specified genes as columns
        """
        # Find gene indices
        gene_indices = []
        var_names = self.var_names

        for gene in gene_names:
            if gene in var_names:
                gene_indices.append(np.where(var_names == gene)[0][0])
            else:
                warnings.warn(f"Gene '{gene}' not found in dataset", stacklevel=2)

        if not gene_indices:
            return

        # Get gene subset
        for chunk, obs_slice in self.iter_chunks(chunk_size=chunk_size):
            if sparse.issparse(chunk):
                gene_data_sparse = chunk[:, gene_indices]
                # Only call .toarray() if it's a sparse matrix
                gene_data = gene_data_sparse.toarray()  # type: ignore
            else:
                gene_data = np.asarray(chunk[:, gene_indices])

            chunk_obs_names = self.obs_names[obs_slice]
            selected_gene_names = [
                gene_names[i]
                for i in range(len(gene_names))
                if gene_names[i] in var_names
            ]

            yield pd.DataFrame(
                gene_data,
                index=chunk_obs_names,
                columns=selected_gene_names,  # type: ignore
            )
