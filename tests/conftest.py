import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from slaf.core.slaf import SLAFArray

# Set random seed for reproducible tests
np.random.seed(42)


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing"""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create a small test dataset
    n_cells, n_genes = 100, 50

    # Create sparse matrix with controlled sparsity
    # Use a more reliable approach than random poisson
    density = 0.3  # 30% sparsity - ensure we have enough non-zero values
    n_nonzero = int(n_cells * n_genes * density)

    # Generate random sparse data with controlled values
    data = np.random.uniform(1.0, 10.0, n_nonzero).astype(float)
    row_indices = np.random.randint(0, n_cells, n_nonzero)
    col_indices = np.random.randint(0, n_genes, n_nonzero)

    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_cells, n_genes)
    )

    # Ensure we have at least some non-zero values in each row and column
    # This prevents completely empty rows/columns that could cause issues
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

    # Create cell metadata
    obs = pd.DataFrame(
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
        },
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )

    # Create gene metadata
    var = pd.DataFrame(
        {
            "gene_type": np.random.choice(["protein_coding", "lncRNA"], n_genes),
            "highly_variable": np.random.choice([True, False], n_genes),
            "total_counts": sparse_matrix.sum(
                axis=0
            ).A1,  # Use actual counts from matrix
            "n_cells_by_counts": (sparse_matrix > 0)
            .sum(axis=0)
            .A1,  # Use actual cell counts
        },
        index=pd.Index([f"gene_{i}" for i in range(n_genes)]),
    )

    # Create AnnData object
    adata = sc.AnnData(X=sparse_matrix, obs=obs, var=var)
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

    # Create obs (cell metadata)
    obs = pd.DataFrame(
        {
            "cell_type": ["T_cell", "B_cell", "NK_cell", "T_cell", "B_cell"],
            "batch": ["batch1", "batch1", "batch1", "batch2", "batch2"],
        },
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )

    # Create var (gene metadata)
    var = pd.DataFrame(
        {
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
            "highly_variable": [True, True, False],
        },
        index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
    )

    return sc.AnnData(X=X, obs=obs, var=var)


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

    # Create obs (cell metadata)
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(["T_cell", "B_cell", "NK_cell"], n_cells),
            "batch": np.random.choice(["batch1", "batch2"], n_cells),
            "n_genes_by_counts": np.random.poisson(100, n_cells),
            "total_counts": np.random.lognormal(8, 1, n_cells),
        },
        index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
    )

    # Create var (gene metadata)
    var = pd.DataFrame(
        {
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
            "highly_variable": np.random.choice([True, False], n_genes, p=[0.2, 0.8]),
        },
        index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
    )

    return sc.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def tiny_adata():
    """Create a tiny sample AnnData object for testing"""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create a small test dataset (100 cells, 50 genes)
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

    return sc.AnnData(X=sparse_matrix, obs=obs, var=var)


@pytest.fixture
def tiny_slaf(temp_dir):
    """Create a tiny sample SLAF dataset for testing"""
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
    from slaf.data import SLAFConverter

    converter = SLAFConverter()
    slaf_path = Path(temp_dir) / "tiny_test_dataset.slaf"
    converter.convert_anndata(adata, str(slaf_path))

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
    from slaf.data import SLAFConverter

    converter = SLAFConverter()
    slaf_path = Path(temp_dir) / "tiny_test_dataset.slaf"
    converter.convert_anndata(adata, str(slaf_path))

    return str(slaf_path)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
