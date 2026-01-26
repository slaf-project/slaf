"""
End-to-end tests for metadata mutations (obs/var/obsm/varm/uns writes).

Tests obs and var mutation functionality:
- Creating new columns via assignment (immediate write)
- Verifying columns are saved to cells.lance/genes.lance immediately
- Verifying config.json is updated correctly
- Testing overwrite of mutable columns
- Testing immutability protection
- Round-trip: create -> reload -> verify
- Selector support
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from slaf.core.slaf import SLAFArray
from slaf.data.converter import SLAFConverter
from slaf.integrations.anndata import LazyAnnData

# Mark all tests in this file as using SLAFArray instances
pytestmark = pytest.mark.slaf_array


def test_create_new_obs_column(anndata_without_layers):
    """Test creating a new obs column (eager write - immediate)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create a new obs column
        n_obs = adata.n_obs
        cluster_labels = np.array([f"cluster_{i % 3}" for i in range(n_obs)])
        adata.obs["cluster"] = cluster_labels

        # Column should be available immediately (eager write)
        assert "cluster" in adata.obs
        assert len(adata.obs) >= 1

        # Verify column data (AnnData-compatible: returns Series)
        retrieved = adata.obs["cluster"]
        assert isinstance(retrieved, pd.Series)
        assert np.array_equal(retrieved.values, cluster_labels)

        # Reload dataset and verify column persists
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        assert "cluster" in adata2.obs
        retrieved2 = adata2.obs["cluster"]
        assert isinstance(retrieved2, pd.Series)
        assert np.array_equal(retrieved2.values, cluster_labels)

        # Verify config.json was updated
        assert "cluster" in slaf2.config.get("obs", {}).get("available", [])
        assert "cluster" in slaf2.config.get("obs", {}).get("mutable", [])


def test_create_new_var_column(anndata_without_layers):
    """Test creating a new var column (eager write - immediate)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create a new var column
        n_vars = adata.n_vars
        hvg_flags = np.array([i % 2 == 0 for i in range(n_vars)], dtype=bool)
        adata.var["highly_variable"] = hvg_flags.astype(np.float32)

        # Column should be available immediately
        assert "highly_variable" in adata.var

        # Verify column data (AnnData-compatible: returns Series)
        retrieved = adata.var["highly_variable"]
        assert isinstance(retrieved, pd.Series)
        assert len(retrieved) == n_vars

        # Reload dataset and verify column persists
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        assert "highly_variable" in adata2.var
        retrieved2 = adata2.var["highly_variable"]
        assert isinstance(retrieved2, pd.Series)
        assert len(retrieved2) == n_vars

        # Verify config.json was updated
        assert "highly_variable" in slaf2.config.get("var", {}).get("available", [])
        assert "highly_variable" in slaf2.config.get("var", {}).get("mutable", [])


def test_update_mutable_obs_column(anndata_without_layers):
    """Test updating a mutable obs column"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create column
        n_obs = adata.n_obs
        original = np.array([i % 3 for i in range(n_obs)], dtype=np.float32)
        adata.obs["mutable_col"] = original

        # Update column
        updated = np.array([i % 5 for i in range(n_obs)], dtype=np.float32)
        adata.obs["mutable_col"] = updated

        # Verify update (AnnData-compatible: returns Series)
        retrieved = adata.obs["mutable_col"]
        assert isinstance(retrieved, pd.Series)
        assert np.array_equal(retrieved.values, updated)

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        retrieved2 = adata2.obs["mutable_col"]
        assert isinstance(retrieved2, pd.Series)
        assert np.array_equal(retrieved2.values, updated)


def test_delete_mutable_obs_column(anndata_without_layers):
    """Test deleting a mutable obs column"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create column
        n_obs = adata.n_obs
        adata.obs["temp_col"] = np.array([1.0] * n_obs)
        assert "temp_col" in adata.obs

        # Delete column
        del adata.obs["temp_col"]
        assert "temp_col" not in adata.obs

        # Reload and verify deletion
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        assert "temp_col" not in adata2.obs
        assert "temp_col" not in slaf2.config.get("obs", {}).get("available", [])


def test_obs_selector_support(anndata_without_layers):
    """Test that obs respects selectors from parent LazyAnnData"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create column on full dataset
        n_obs = adata.n_obs
        adata.obs["cluster"] = np.array([i % 3 for i in range(n_obs)])

        # Subset the adata
        adata_subset = adata[:3]

        # Access column on subset
        cluster_subset = adata_subset.obs["cluster"]

        # Verify correct shape
        assert len(cluster_subset) == 3


def test_var_selector_support(anndata_without_layers):
    """Test that var respects selectors from parent LazyAnnData"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create column on full dataset
        n_vars = adata.n_vars
        adata.var["hvg"] = np.array([i % 2 for i in range(n_vars)], dtype=np.float32)

        # Subset the adata
        adata_subset = adata[:, :3]

        # Access column on subset
        hvg_subset = adata_subset.var["hvg"]

        # Verify correct shape
        assert len(hvg_subset) == 3


def test_backward_compatibility_obs_var(anndata_without_layers):
    """Test that obs and var properties still work (backward compatibility)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # obs and var should behave like DataFrames (but are view objects)
        import pandas as pd

        assert hasattr(adata.obs, "columns")
        assert hasattr(adata.var, "columns")
        # obs and var are now view objects, but behave like DataFrames
        # Check that they have DataFrame-like attributes
        assert hasattr(adata.obs, "_get_dataframe")
        assert hasattr(adata.var, "_get_dataframe")
        # Verify they return DataFrames when accessed as DataFrames
        obs_df = adata.obs._get_dataframe()
        var_df = adata.var._get_dataframe()
        assert isinstance(obs_df, pd.DataFrame)
        assert isinstance(var_df, pd.DataFrame)

        # n_obs and n_vars should still work
        assert isinstance(adata.n_obs, int)
        assert isinstance(adata.n_vars, int)
        assert adata.n_obs > 0
        assert adata.n_vars > 0


def test_create_multiple_obs_columns(anndata_without_layers):
    """Test creating multiple obs columns"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        n_obs = adata.n_obs

        # Create multiple columns
        adata.obs["cluster"] = np.array([i % 3 for i in range(n_obs)])
        adata.obs["batch"] = np.array([i % 2 for i in range(n_obs)])
        adata.obs["quality"] = np.array([float(i) for i in range(n_obs)])

        # Verify all columns exist
        assert "cluster" in adata.obs
        assert "batch" in adata.obs
        assert "quality" in adata.obs

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        assert "cluster" in adata2.obs
        assert "batch" in adata2.obs
        assert "quality" in adata2.obs


def test_dtype_optimization(anndata_without_layers):
    """Test that dtype optimization works correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        n_obs = adata.n_obs

        # Create column with integers (should be optimized to uint16)
        int_values = np.array([i % 100 for i in range(n_obs)], dtype=np.int32)
        adata.obs["int_col"] = int_values

        # Create column with floats (should stay float32)
        float_values = np.array(
            [float(i) / 10.0 for i in range(n_obs)], dtype=np.float32
        )
        adata.obs["float_col"] = float_values

        # Verify columns exist
        assert "int_col" in adata.obs
        assert "float_col" in adata.obs

        # Reload and verify data integrity
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        int_retrieved = adata2.obs["int_col"]
        float_retrieved = adata2.obs["float_col"]

        # Values should match (within tolerance for floats)
        assert np.allclose(int_retrieved, int_values, rtol=1e-5)
        assert np.allclose(float_retrieved, float_values, rtol=1e-5)


def test_create_new_obsm_embedding(anndata_without_layers):
    """Test creating a new obsm embedding (eager write - immediate)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create a new obsm embedding (UMAP-like)
        n_obs = adata.n_obs
        umap_coords = np.random.rand(n_obs, 2).astype(np.float32)
        adata.obsm["X_umap"] = umap_coords

        # Embedding should be available immediately (eager write)
        assert "X_umap" in adata.obsm
        assert len(adata.obsm) >= 1

        # Verify embedding data
        retrieved = adata.obsm["X_umap"]
        assert retrieved.shape == (n_obs, 2)
        assert np.allclose(retrieved, umap_coords, rtol=1e-5)

        # Reload dataset and verify embedding persists
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        assert "X_umap" in adata2.obsm
        retrieved2 = adata2.obsm["X_umap"]
        assert retrieved2.shape == (n_obs, 2)
        assert np.allclose(retrieved2, umap_coords, rtol=1e-5)

        # Verify config.json was updated
        assert "X_umap" in slaf2.config.get("obsm", {}).get("available", [])
        assert "X_umap" in slaf2.config.get("obsm", {}).get("mutable", [])
        assert slaf2.config.get("obsm", {}).get("dimensions", {}).get("X_umap") == 2


def test_create_new_varm_embedding(anndata_without_layers):
    """Test creating a new varm embedding (eager write - immediate)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create a new varm embedding (PCA loadings-like)
        n_vars = adata.n_vars
        pcs = np.random.rand(n_vars, 50).astype(np.float32)
        adata.varm["PCs"] = pcs

        # Embedding should be available immediately
        assert "PCs" in adata.varm

        # Verify embedding data
        retrieved = adata.varm["PCs"]
        assert retrieved.shape == (n_vars, 50)
        assert np.allclose(retrieved, pcs, rtol=1e-5)

        # Reload dataset and verify embedding persists
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        assert "PCs" in adata2.varm
        retrieved2 = adata2.varm["PCs"]
        assert retrieved2.shape == (n_vars, 50)
        assert np.allclose(retrieved2, pcs, rtol=1e-5)

        # Verify config.json was updated
        assert "PCs" in slaf2.config.get("varm", {}).get("available", [])
        assert "PCs" in slaf2.config.get("varm", {}).get("mutable", [])
        assert slaf2.config.get("varm", {}).get("dimensions", {}).get("PCs") == 50


def test_update_mutable_obsm_embedding(anndata_without_layers):
    """Test updating a mutable obsm embedding"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        n_obs = adata.n_obs

        # Create embedding
        original = np.random.rand(n_obs, 2).astype(np.float32)
        adata.obsm["X_umap"] = original

        # Update embedding
        updated = np.random.rand(n_obs, 2).astype(np.float32)
        adata.obsm["X_umap"] = updated

        # Verify update
        retrieved = adata.obsm["X_umap"]
        assert np.allclose(retrieved, updated, rtol=1e-5)

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        retrieved2 = adata2.obsm["X_umap"]
        assert np.allclose(retrieved2, updated, rtol=1e-5)


def test_delete_mutable_obsm_embedding(anndata_without_layers):
    """Test deleting a mutable obsm embedding"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        n_obs = adata.n_obs

        # Create embedding
        adata.obsm["X_umap"] = np.random.rand(n_obs, 2).astype(np.float32)
        assert "X_umap" in adata.obsm

        # Delete embedding
        del adata.obsm["X_umap"]
        assert "X_umap" not in adata.obsm

        # Reload and verify deletion
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        assert "X_umap" not in adata2.obsm
        assert "X_umap" not in slaf2.config.get("obsm", {}).get("available", [])


def test_obsm_selector_support(anndata_without_layers):
    """Test that obsm respects selectors from parent LazyAnnData"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create embedding on full dataset
        n_obs = adata.n_obs
        adata.obsm["X_umap"] = np.random.rand(n_obs, 2).astype(np.float32)

        # Subset the adata
        adata_subset = adata[:3]

        # Access embedding on subset
        umap_subset = adata_subset.obsm["X_umap"]

        # Verify correct shape
        assert umap_subset.shape == (3, 2)


def test_varm_selector_support(anndata_without_layers):
    """Test that varm respects selectors from parent LazyAnnData"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create embedding on full dataset
        n_vars = adata.n_vars
        adata.varm["PCs"] = np.random.rand(n_vars, 50).astype(np.float32)

        # Subset the adata
        adata_subset = adata[:, :3]

        # Access embedding on subset
        pcs_subset = adata_subset.varm["PCs"]

        # Verify correct shape
        assert pcs_subset.shape == (3, 50)


def test_create_multiple_obsm_embeddings(anndata_without_layers):
    """Test creating multiple obsm embeddings"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        n_obs = adata.n_obs

        # Create multiple embeddings
        adata.obsm["X_umap"] = np.random.rand(n_obs, 2).astype(np.float32)
        adata.obsm["X_pca"] = np.random.rand(n_obs, 50).astype(np.float32)
        adata.obsm["X_tsne"] = np.random.rand(n_obs, 2).astype(np.float32)

        # Verify all embeddings exist
        assert "X_umap" in adata.obsm
        assert "X_pca" in adata.obsm
        assert "X_tsne" in adata.obsm

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        assert "X_umap" in adata2.obsm
        assert "X_pca" in adata2.obsm
        assert "X_tsne" in adata2.obsm

        # Verify dimensions in config
        assert slaf2.config.get("obsm", {}).get("dimensions", {}).get("X_umap") == 2
        assert slaf2.config.get("obsm", {}).get("dimensions", {}).get("X_pca") == 50
        assert slaf2.config.get("obsm", {}).get("dimensions", {}).get("X_tsne") == 2


def test_create_new_uns_key(anndata_without_layers, temp_dir):
    """Test creating a new uns key (eager write - immediate)"""
    tmpdir = temp_dir
    # Convert AnnData to SLAF
    converter = SLAFConverter(
        use_optimized_dtypes=False,
        compact_after_write=False,
        chunked=False,
    )
    converter.convert_anndata(anndata_without_layers, tmpdir)

    # Load SLAF dataset
    slaf = SLAFArray(tmpdir)
    adata = LazyAnnData(slaf)

    # Create a new uns key
    adata.uns["neighbors"] = {"params": {"n_neighbors": 15, "metric": "euclidean"}}

    # Key should be available immediately (eager write)
    assert "neighbors" in adata.uns
    assert len(adata.uns) >= 1

    # Verify data
    retrieved = adata.uns["neighbors"]
    assert retrieved == {"params": {"n_neighbors": 15, "metric": "euclidean"}}

    # Reload dataset and verify key persists
    slaf2 = SLAFArray(tmpdir)
    adata2 = LazyAnnData(slaf2)

    assert "neighbors" in adata2.uns
    retrieved2 = adata2.uns["neighbors"]
    assert retrieved2 == {"params": {"n_neighbors": 15, "metric": "euclidean"}}


def test_update_uns_key(anndata_without_layers):
    """Test updating an existing uns key"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create key
        adata.uns["neighbors"] = {"params": {"n_neighbors": 15}}

        # Update key
        adata.uns["neighbors"] = {"params": {"n_neighbors": 20, "metric": "cosine"}}

        # Verify update
        retrieved = adata.uns["neighbors"]
        assert retrieved == {"params": {"n_neighbors": 20, "metric": "cosine"}}

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        retrieved2 = adata2.uns["neighbors"]
        assert retrieved2 == {"params": {"n_neighbors": 20, "metric": "cosine"}}


def test_delete_uns_key(anndata_without_layers, temp_dir):
    """Test deleting an uns key"""
    tmpdir = temp_dir
    # Convert AnnData to SLAF
    converter = SLAFConverter(
        use_optimized_dtypes=False,
        compact_after_write=False,
        chunked=False,
    )
    converter.convert_anndata(anndata_without_layers, tmpdir)

    # Load SLAF dataset
    slaf = SLAFArray(tmpdir)
    adata = LazyAnnData(slaf)

    # Create key
    adata.uns["temp_key"] = {"value": 1}
    assert "temp_key" in adata.uns

    # Delete key
    del adata.uns["temp_key"]
    assert "temp_key" not in adata.uns

    # Reload and verify deletion
    slaf2 = SLAFArray(tmpdir)
    adata2 = LazyAnnData(slaf2)
    assert "temp_key" not in adata2.uns


def test_uns_json_serialization(anndata_without_layers, temp_dir):
    """Test that uns properly serializes numpy arrays and pandas objects"""
    tmpdir = temp_dir
    # Convert AnnData to SLAF
    converter = SLAFConverter(
        use_optimized_dtypes=False,
        compact_after_write=False,
        chunked=False,
    )
    converter.convert_anndata(anndata_without_layers, tmpdir)

    # Load SLAF dataset
    slaf = SLAFArray(tmpdir)
    adata = LazyAnnData(slaf)

    # Store various types that need serialization
    adata.uns["pca"] = {
        "variance_ratio": np.array([0.1, 0.05, 0.03, 0.02]),
        "variance": np.array([100.0, 50.0, 30.0, 20.0]),
        "n_components": np.int32(50),
    }

    import pandas as pd

    adata.uns["series_data"] = pd.Series([1, 2, 3, 4, 5])

    # Reload and verify serialization
    slaf2 = SLAFArray(tmpdir)
    adata2 = LazyAnnData(slaf2)

    pca_data = adata2.uns["pca"]
    assert isinstance(pca_data["variance_ratio"], list)
    assert pca_data["variance_ratio"] == [0.1, 0.05, 0.03, 0.02]
    assert isinstance(pca_data["n_components"], int)
    assert pca_data["n_components"] == 50

    series_data = adata2.uns["series_data"]
    assert isinstance(series_data, list)
    assert series_data == [1, 2, 3, 4, 5]


def test_create_multiple_uns_keys(anndata_without_layers):
    """Test creating multiple uns keys"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Create multiple keys
        adata.uns["neighbors"] = {"params": {"n_neighbors": 15}}
        adata.uns["pca"] = {"variance_ratio": [0.1, 0.05, 0.03]}
        adata.uns["leiden"] = {"params": {"resolution": 0.5}}

        # Verify all keys exist
        assert "neighbors" in adata.uns
        assert "pca" in adata.uns
        assert "leiden" in adata.uns
        assert len(adata.uns) == 3

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)
        assert "neighbors" in adata2.uns
        assert "pca" in adata2.uns
        assert "leiden" in adata2.uns
        assert len(adata2.uns) == 3


def test_uns_nested_structure(anndata_without_layers, temp_dir):
    """Test storing and retrieving nested structures in uns"""
    tmpdir = temp_dir
    # Convert AnnData to SLAF
    converter = SLAFConverter(
        use_optimized_dtypes=False,
        compact_after_write=False,
        chunked=False,
    )
    converter.convert_anndata(anndata_without_layers, tmpdir)

    # Load SLAF dataset
    slaf = SLAFArray(tmpdir)
    adata = LazyAnnData(slaf)

    # Store deeply nested structure
    nested = {
        "analysis": {
            "pca": {
                "variance_ratio": np.array([0.1, 0.05]),
                "params": {"n_components": 50},
            },
            "umap": {
                "params": {"n_neighbors": 15, "min_dist": 0.1},
            },
        },
        "clustering": {
            "leiden": {"params": {"resolution": 0.5}},
        },
    }
    adata.uns["results"] = nested

    # Reload and verify nested structure
    slaf2 = SLAFArray(tmpdir)
    adata2 = LazyAnnData(slaf2)

    results = adata2.uns["results"]
    assert isinstance(results["analysis"]["pca"]["variance_ratio"], list)
    assert results["analysis"]["pca"]["variance_ratio"] == [0.1, 0.05]
    assert results["analysis"]["pca"]["params"]["n_components"] == 50
    assert results["clustering"]["leiden"]["params"]["resolution"] == 0.5


def test_obs_cache_invalidation_on_column_add(anndata_without_layers):
    """Test that obs DataFrame cache is invalidated when adding columns via obs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access obs to populate cache
        original_columns = set(adata.obs.columns)
        assert "cluster" not in original_columns

        # Add new column via obs
        n_obs = adata.n_obs
        adata.obs["cluster"] = np.array([f"cluster_{i % 3}" for i in range(n_obs)])

        # Access obs again - should include the new column (cache invalidated)
        updated_columns = set(adata.obs.columns)
        assert "cluster" in updated_columns
        assert len(updated_columns) == len(original_columns) + 1

        # Verify the new column data is accessible
        cluster_values = adata.obs["cluster"].values
        assert len(cluster_values) == n_obs
        assert cluster_values[0] in ["cluster_0", "cluster_1", "cluster_2"]


def test_var_cache_invalidation_on_column_add(anndata_without_layers):
    """Test that var DataFrame cache is invalidated when adding columns via var"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access var to populate cache
        original_columns = set(adata.var.columns)
        assert "highly_variable" not in original_columns

        # Add new column via var
        n_vars = adata.n_vars
        adata.var["highly_variable"] = np.array(
            [i % 2 == 0 for i in range(n_vars)], dtype=np.float32
        )

        # Access var again - should include the new column (cache invalidated)
        updated_columns = set(adata.var.columns)
        assert "highly_variable" in updated_columns
        assert len(updated_columns) == len(original_columns) + 1

        # Verify the new column data is accessible
        hvg_values = adata.var["highly_variable"].values
        assert len(hvg_values) == n_vars


def test_obs_cache_invalidation_on_column_delete(anndata_without_layers):
    """Test that obs DataFrame cache is invalidated when deleting columns via obs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Add a column via obs
        n_obs = adata.n_obs
        adata.obs["temp_col"] = np.array([1.0] * n_obs)

        # Access obs to populate cache
        columns_before = set(adata.obs.columns)
        assert "temp_col" in columns_before

        # Delete the column via obs
        del adata.obs["temp_col"]

        # Access obs again - should not include the deleted column (cache invalidated)
        columns_after = set(adata.obs.columns)
        assert "temp_col" not in columns_after
        assert len(columns_after) == len(columns_before) - 1


def test_obs_cache_invalidation_on_obsm_add(anndata_without_layers):
    """Test that obs DataFrame cache is invalidated when adding obsm embeddings"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access obs to populate cache
        original_columns = set(adata.obs.columns)

        # Add obsm embedding (this modifies cells.lance)
        n_obs = adata.n_obs
        adata.obsm["X_umap"] = np.random.rand(n_obs, 2).astype(np.float32)

        # Access obs again - cache should be invalidated (even though obs columns didn't change)
        # This ensures the cache is cleared when the underlying table is modified
        updated_columns = set(adata.obs.columns)
        # Columns should be the same, but cache was invalidated and reloaded
        assert updated_columns == original_columns

        # Verify obs still works correctly after cache invalidation
        assert len(adata.obs) == n_obs


def test_var_cache_invalidation_on_varm_add(anndata_without_layers):
    """Test that var DataFrame cache is invalidated when adding varm embeddings"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access var to populate cache
        original_columns = set(adata.var.columns)

        # Add varm embedding (this modifies genes.lance)
        n_vars = adata.n_vars
        adata.varm["PCs"] = np.random.rand(n_vars, 50).astype(np.float32)

        # Access var again - cache should be invalidated (even though var columns didn't change)
        # This ensures the cache is cleared when the underlying table is modified
        updated_columns = set(adata.var.columns)
        # Columns should be the same, but cache was invalidated and reloaded
        assert updated_columns == original_columns

        # Verify var still works correctly after cache invalidation
        assert len(adata.var) == n_vars


def test_obs_names_cache_invalidation(anndata_without_layers):
    """Test that obs_names cache is invalidated when obs cache is invalidated"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access obs_names to populate cache
        original_obs_names = adata.obs_names.copy()

        # Add column via obs (this invalidates obs cache)
        n_obs = adata.n_obs
        adata.obs["cluster"] = np.array([f"cluster_{i % 3}" for i in range(n_obs)])

        # Access obs_names again - should still work (cache was invalidated and reloaded)
        updated_obs_names = adata.obs_names
        assert len(updated_obs_names) == len(original_obs_names)
        # Names should be the same (we didn't change cell IDs)
        assert list(updated_obs_names) == list(original_obs_names)


def test_var_names_cache_invalidation(anndata_without_layers):
    """Test that var_names cache is invalidated when var cache is invalidated"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access var_names to populate cache
        original_var_names = adata.var_names.copy()

        # Add column via var (this invalidates var cache)
        n_vars = adata.n_vars
        adata.var["highly_variable"] = np.array(
            [i % 2 == 0 for i in range(n_vars)], dtype=np.float32
        )

        # Access var_names again - should still work (cache was invalidated and reloaded)
        updated_var_names = adata.var_names
        assert len(updated_var_names) == len(original_var_names)
        # Names should be the same (we didn't change gene IDs)
        assert list(updated_var_names) == list(original_var_names)
