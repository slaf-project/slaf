import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from slaf.core.slaf import SLAFArray
from slaf.data import SLAFConverter


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create sparse matrix
    n_cells, n_genes = 100, 50
    density = 0.1  # 10% sparsity
    n_nonzero = int(n_cells * n_genes * density)

    # Generate random sparse data
    data = np.random.lognormal(0, 1, n_nonzero).astype(np.float32)
    row_indices = np.random.randint(0, n_cells, n_nonzero)
    col_indices = np.random.randint(0, n_genes, n_nonzero)
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_cells, n_genes)
    )

    # Ensure each row and column has at least one non-zero value
    for i in range(n_cells):
        if sparse_matrix[i, :].nnz == 0:
            # Add a random non-zero value to this row
            j = np.random.randint(0, n_genes)
            sparse_matrix[i, j] = np.random.uniform(1.0, 5.0)

    for j in range(n_genes):
        if sparse_matrix[:, j].nnz == 0:
            # Add a random non-zero value to this column
            i = np.random.randint(0, n_cells)
            sparse_matrix[i, j] = np.random.uniform(1.0, 5.0)

    # Create cell metadata using polars
    obs = (
        pl.DataFrame(
            {
                "cell_type": np.random.choice(["T-cell", "B-cell", "NK-cell"], n_cells),
                "batch": np.random.choice(["batch_1", "batch_2"], n_cells),
                "total_counts": sparse_matrix.sum(
                    axis=1
                ).A1,  # Use actual counts from matrix
                "n_genes_by_counts": (sparse_matrix > 0)
                .sum(axis=1)
                .A1,  # Use actual gene counts
                "high_mito": np.random.choice([True, False], n_cells),
            }
        )
        .with_row_index("cell_id", offset=0)
        .with_columns(
            pl.col("cell_id").map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
        )
    )

    # Create gene metadata using polars
    var = (
        pl.DataFrame(
            {
                "gene_type": np.random.choice(["protein_coding", "lncRNA"], n_genes),
                "highly_variable": np.random.choice([True, False], n_genes),
                "total_counts": sparse_matrix.sum(
                    axis=0
                ).A1,  # Use actual counts from matrix
                "n_cells_by_counts": (sparse_matrix > 0)
                .sum(axis=0)
                .A1,  # Use actual cell counts
            }
        )
        .with_row_index("gene_id", offset=0)
        .with_columns(
            pl.col("gene_id").map_elements(lambda x: f"gene_{x}", return_dtype=pl.Utf8)
        )
    )

    # Convert to pandas for AnnData compatibility
    obs_pd = obs.to_pandas().set_index("cell_id")
    var_pd = var.to_pandas().set_index("gene_id")

    # Create AnnData object
    adata = sc.AnnData(X=sparse_matrix, obs=obs_pd, var=var_pd)
    return adata


@pytest.fixture
def small_sample_adata():
    """Create a small sample AnnData object for testing (from test_converter.py)"""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create sparse matrix
    n_cells, n_genes = 5, 3
    data = np.array([1.0, 2.0, 3.0, 4.0])
    row_indices = np.array([0, 0, 1, 2])
    col_indices = np.array([0, 1, 1, 2])
    X = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_genes))

    # Create obs (cell metadata) using polars
    obs = (
        pl.DataFrame(
            {
                "cell_type": ["T_cell", "B_cell", "NK_cell", "T_cell", "B_cell"],
                "batch": ["batch1", "batch1", "batch1", "batch2", "batch2"],
            }
        )
        .with_row_index("cell_id", offset=0)
        .with_columns(
            pl.col("cell_id").map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
        )
    )

    # Create var (gene metadata) using polars
    var = (
        pl.DataFrame(
            {
                "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
                "highly_variable": [True, True, False],
            }
        )
        .with_row_index("gene_id", offset=0)
        .with_columns(
            pl.col("gene_id").map_elements(
                lambda x: f"ENSG_{x:08d}", return_dtype=pl.Utf8
            )
        )
    )

    # Convert to pandas for AnnData compatibility
    obs_pd = obs.to_pandas().set_index("cell_id")
    var_pd = var.to_pandas().set_index("gene_id")

    return sc.AnnData(X=X, obs=obs_pd, var=var_pd)


@pytest.fixture
def large_sample_adata():
    """Create a larger sample AnnData object for optimization testing"""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create sparse matrix with realistic data
    n_cells, n_genes = 1000, 500
    density = 0.1  # 10% sparsity
    n_nonzero = int(n_cells * n_genes * density)

    # Generate random sparse data
    data = np.random.lognormal(0, 1, n_nonzero).astype(np.float32)
    row_indices = np.random.randint(0, n_cells, n_nonzero)
    col_indices = np.random.randint(0, n_genes, n_nonzero)
    X = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_genes))

    # Create obs (cell metadata) using polars
    obs = (
        pl.DataFrame(
            {
                "cell_type": np.random.choice(["T_cell", "B_cell", "NK_cell"], n_cells),
                "batch": np.random.choice(["batch1", "batch2"], n_cells),
                "n_genes_by_counts": np.random.poisson(100, n_cells),
                "total_counts": np.random.lognormal(8, 1, n_cells),
            }
        )
        .with_row_index("cell_id", offset=0)
        .with_columns(
            pl.col("cell_id").map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
        )
    )

    # Create var (gene metadata) using polars
    var = (
        pl.DataFrame(
            {
                "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
                "highly_variable": np.random.choice([True, False], n_genes),
                "n_cells_by_counts": np.random.poisson(50, n_genes),
                "total_counts": np.random.lognormal(6, 1, n_genes),
            }
        )
        .with_row_index("gene_id", offset=0)
        .with_columns(
            pl.col("gene_id").map_elements(
                lambda x: f"ENSG_{x:08d}", return_dtype=pl.Utf8
            )
        )
    )

    # Convert to pandas for AnnData compatibility
    obs_pd = obs.to_pandas().set_index("cell_id")
    var_pd = var.to_pandas().set_index("gene_id")

    return sc.AnnData(X=X, obs=obs_pd, var=var_pd)


@pytest.fixture
def tiny_adata():
    """Create a tiny AnnData object for quick testing."""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create sparse matrix
    n_cells, n_genes = 100, 50
    density = 0.3  # 30% sparsity for more data
    n_nonzero = int(n_cells * n_genes * density)

    # Generate random sparse data
    data = np.random.uniform(1.0, 5.0, n_nonzero).astype(np.float32)
    row_indices = np.random.randint(0, n_cells, n_nonzero)
    col_indices = np.random.randint(0, n_genes, n_nonzero)
    X = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_genes))

    # Ensure each row and column has at least one non-zero value
    for i in range(n_cells):
        if X[i, :].nnz == 0:
            j = np.random.randint(0, n_genes)
            X[i, j] = np.random.uniform(1.0, 5.0)

    for j in range(n_genes):
        if X[:, j].nnz == 0:
            i = np.random.randint(0, n_cells)
            X[i, j] = np.random.uniform(1.0, 5.0)

    # Create obs (cell metadata) using polars
    obs = (
        pl.DataFrame(
            {
                "cell_type": np.random.choice(["A", "B"], n_cells),
                "total_counts": X.sum(axis=1).A1,
            }
        )
        .with_row_index("cell_id", offset=0)
        .with_columns(
            pl.col("cell_id").map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
        )
    )

    # Create var (gene metadata) using polars
    var = (
        pl.DataFrame(
            {
                "gene_type": np.random.choice(["protein_coding", "lncRNA"], n_genes),
                "highly_variable": np.random.choice([True, False], n_genes),
                "expression_mean": np.random.uniform(0.1, 2.0, n_genes),
            }
        )
        .with_row_index("gene_id", offset=0)
        .with_columns(
            pl.col("gene_id").map_elements(lambda x: f"gene_{x}", return_dtype=pl.Utf8)
        )
    )

    # Convert to pandas for AnnData compatibility
    obs_pd = obs.to_pandas().set_index("cell_id")
    var_pd = var.to_pandas().set_index("gene_id")

    return sc.AnnData(X=X, obs=obs_pd, var=var_pd)


@pytest.fixture
def small_adata():
    """Create a small AnnData object for SLAFArray testing."""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create sparse matrix
    n_cells, n_genes = 10, 5
    density = 0.3  # 30% sparsity for more data
    n_nonzero = int(n_cells * n_genes * density)

    # Generate random sparse data
    data = np.random.uniform(1.0, 5.0, n_nonzero).astype(np.float32)
    row_indices = np.random.randint(0, n_cells, n_nonzero)
    col_indices = np.random.randint(0, n_genes, n_nonzero)
    X = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_genes))

    # Ensure each row and column has at least one non-zero value
    for i in range(n_cells):
        if X[i, :].nnz == 0:
            j = np.random.randint(0, n_genes)
            X[i, j] = np.random.uniform(1.0, 5.0)

    for j in range(n_genes):
        if X[:, j].nnz == 0:
            i = np.random.randint(0, n_genes)
            X[i, j] = np.random.uniform(1.0, 5.0)

    # Create obs (cell metadata) using polars
    obs = (
        pl.DataFrame(
            {
                "cell_type": np.random.choice(["A", "B"], n_cells),
                "total_counts": X.sum(axis=1).A1,
            }
        )
        .with_row_index("cell_id", offset=0)
        .with_columns(
            pl.col("cell_id").map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
        )
    )

    # Create var (gene metadata) using polars
    var = (
        pl.DataFrame(
            {
                "gene_type": np.random.choice(["protein_coding", "lncRNA"], n_genes),
                "highly_variable": np.random.choice([True, False], n_genes),
                "expression_mean": np.random.uniform(0.1, 2.0, n_genes),
            }
        )
        .with_row_index("gene_id", offset=0)
        .with_columns(
            pl.col("gene_id").map_elements(lambda x: f"gene_{x}", return_dtype=pl.Utf8)
        )
    )

    # Convert to pandas for AnnData compatibility
    obs_pd = obs.to_pandas().set_index("cell_id")
    var_pd = var.to_pandas().set_index("gene_id")

    return sc.AnnData(X=X, obs=obs_pd, var=var_pd)


@pytest.fixture
def tiny_slaf(temp_dir, tiny_adata):
    """Create a tiny SLAFArray for testing."""
    # Convert to SLAF format
    converter = SLAFConverter(use_optimized_dtypes=False, compact_after_write=False)
    slaf_path = Path(temp_dir) / "tiny_test_dataset.slaf"
    converter.convert_anndata(tiny_adata, str(slaf_path))

    return SLAFArray(str(slaf_path))


@pytest.fixture
def small_slaf(temp_dir, small_adata):
    """Create a small SLAFArray for SLAFArray testing."""
    # Convert to SLAF format
    converter = SLAFConverter(use_optimized_dtypes=False, compact_after_write=False)
    slaf_path = Path(temp_dir) / "small_test_dataset.slaf"
    converter.convert_anndata(small_adata, str(slaf_path))

    return SLAFArray(str(slaf_path))


@pytest.fixture
def tiny_slaf_path(temp_dir):
    """Create a tiny sample SLAF dataset and return its path"""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create a small test dataset
    n_cells, n_genes = 100, 50

    # Create sparse matrix with controlled sparsity
    density = 0.3
    n_nonzero = int(n_cells * n_genes * density)

    # Generate random sparse data
    data = np.random.uniform(1.0, 10.0, n_nonzero).astype(float)
    row_indices = np.random.randint(0, n_cells, n_nonzero)
    col_indices = np.random.randint(0, n_genes, n_nonzero)

    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_cells, n_genes)
    )

    # Ensure we have at least some non-zero values in each row and column
    for i in range(n_cells):
        if sparse_matrix[i, :].nnz == 0:
            j = np.random.randint(0, n_genes)
            sparse_matrix[i, j] = np.random.uniform(1.0, 5.0)

    for j in range(n_genes):
        if sparse_matrix[:, j].nnz == 0:
            i = np.random.randint(0, n_cells)
            sparse_matrix[i, j] = np.random.uniform(1.0, 5.0)

    # Create cell metadata
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(["T-cell", "B-cell", "NK-cell"], n_cells),
            "batch": np.random.choice(["batch_1", "batch_2"], n_cells),
            "total_counts": sparse_matrix.sum(axis=1).A1,
            "n_genes_by_counts": (sparse_matrix > 0).sum(axis=1).A1,
            "high_mito": np.random.choice([True, False], n_cells),
        },
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )

    # Create gene metadata
    var = pd.DataFrame(
        {
            "gene_type": np.random.choice(["protein_coding", "lncRNA"], n_genes),
            "highly_variable": np.random.choice([True, False], n_genes),
            "total_counts": sparse_matrix.sum(axis=0).A1,
            "n_cells_by_counts": (sparse_matrix > 0).sum(axis=0).A1,
        },
        index=pd.Index([f"gene_{i}" for i in range(n_genes)]),
    )

    # Create AnnData object
    adata = sc.AnnData(X=sparse_matrix, obs=obs, var=var)

    # Convert to SLAF format
    converter = SLAFConverter(use_optimized_dtypes=False, compact_after_write=False)
    slaf_path = Path(temp_dir) / "tiny_test_dataset.slaf"
    converter.convert_anndata(adata, str(slaf_path))

    return str(slaf_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    import shutil

    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_h5ad_file():
    """Create a sample h5ad file for chunked reader testing"""
    import tempfile

    import h5py

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            # Create a simple sparse matrix (2x2)
            data = np.array([1.0, 2.0, 3.0, 4.0])
            indices = np.array([0, 1, 0, 1])
            indptr = np.array([0, 2, 4])

            # Create X group with sparse format
            x_group = f.create_group("X")
            x_group.create_dataset("data", data=data)
            x_group.create_dataset("indices", data=indices)
            x_group.create_dataset("indptr", data=indptr)

            # Create obs group
            obs_group = f.create_group("obs")
            obs_group.create_dataset("_index", data=np.array([b"cell_0", b"cell_1"]))
            obs_group.create_dataset("cell_type", data=np.array([b"type_A", b"type_B"]))

            # Create var group
            var_group = f.create_group("var")
            var_group.create_dataset("_index", data=np.array([b"gene_0", b"gene_1"]))
            var_group.create_dataset("highly_variable", data=np.array([True, False]))

        yield tmp.name
        # Cleanup
        Path(tmp.name).unlink()


@pytest.fixture
def chunked_converter():
    """Create a chunked converter instance for testing"""
    from slaf.data.converter import SLAFConverter

    return SLAFConverter(chunked=True, chunk_size=1000)
