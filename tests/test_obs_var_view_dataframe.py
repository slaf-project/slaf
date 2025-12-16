"""
Tests for DataFrame-like behavior of obs_view and var_view.

Tests that obs_view and var_view can be used both as DataFrames and as dictionaries.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from slaf.core.slaf import SLAFArray
from slaf.data.converter import SLAFConverter
from slaf.integrations.anndata import LazyAnnData


@pytest.fixture
def anndata_with_metadata():
    """Create an AnnData object with obs/var columns and obsm/varm for testing"""
    import scanpy as sc

    np.random.seed(42)
    n_cells, n_genes = 10, 5

    X = csr_matrix(np.random.rand(n_cells, n_genes), dtype=np.float32)
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Add obs columns
    adata.obs["cluster"] = [f"cluster_{i % 3}" for i in range(n_cells)]
    adata.obs["total_counts"] = np.random.rand(n_cells).astype(np.float32)

    # Add var columns
    adata.var["highly_variable"] = [True, False, True, False, True]

    # Add obsm (vector columns)
    adata.obsm["X_umap"] = np.random.rand(n_cells, 2).astype(np.float32)

    # Add varm (vector columns)
    adata.varm["PCs"] = np.random.rand(n_genes, 50).astype(np.float32)

    return adata


def test_obs_view_dataframe_access(anndata_with_metadata):
    """Test that obs_view behaves like a DataFrame when accessed directly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access as DataFrame - obs_view should behave like DataFrame
        df = adata.obs_view._get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10  # n_cells

        # DataFrame attributes should work
        assert hasattr(df, "columns")
        assert hasattr(df, "head")
        assert hasattr(df, "shape")
        assert hasattr(df, "index")

        # Check columns (should NOT include vector columns like X_umap)
        columns = list(df.columns)
        assert "cluster" in columns
        assert "total_counts" in columns
        assert "X_umap" not in columns  # Vector column should be excluded

        # DataFrame methods should work
        head_df = df.head(5)
        assert isinstance(head_df, pd.DataFrame)
        assert len(head_df) == 5

        # DataFrame indexing should work
        subset = df[["cluster", "total_counts"]]
        assert isinstance(subset, pd.DataFrame)
        assert list(subset.columns) == ["cluster", "total_counts"]


def test_obs_view_dict_access(anndata_with_metadata):
    """Test that obs_view still supports dictionary-like access"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Dictionary-like access should return numpy array
        cluster = adata.obs_view["cluster"]
        assert isinstance(cluster, np.ndarray)
        assert len(cluster) == 10  # n_cells

        # Dictionary methods should work
        assert "cluster" in adata.obs_view
        assert len(adata.obs_view) > 0
        assert "cluster" in list(adata.obs_view.keys())


def test_obs_view_dual_interface(anndata_with_metadata):
    """Test that obs_view supports both DataFrame and dict interfaces simultaneously"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # DataFrame interface
        df = adata.obs_view._get_dataframe()
        assert isinstance(df, pd.DataFrame)

        # Dict interface (string key returns array)
        cluster_array = adata.obs_view["cluster"]
        assert isinstance(cluster_array, np.ndarray)

        # DataFrame indexing (list of columns returns DataFrame)
        subset_df = adata.obs_view[["cluster", "total_counts"]]
        assert isinstance(subset_df, pd.DataFrame)

        # Both should give same data
        assert np.array_equal(df["cluster"].values, cluster_array)


def test_var_view_dataframe_access(anndata_with_metadata):
    """Test that var_view behaves like a DataFrame when accessed directly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access as DataFrame - var_view should behave like DataFrame
        df = adata.var_view._get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # n_genes

        # Check columns (should NOT include vector columns like PCs)
        columns = list(df.columns)
        assert "highly_variable" in columns
        assert "PCs" not in columns  # Vector column should be excluded

        # DataFrame methods should work
        head_df = df.head(3)
        assert isinstance(head_df, pd.DataFrame)
        assert len(head_df) == 3


def test_var_view_dict_access(anndata_with_metadata):
    """Test that var_view still supports dictionary-like access"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Dictionary-like access should return numpy array
        hvg = adata.var_view["highly_variable"]
        assert isinstance(hvg, np.ndarray)
        assert len(hvg) == 5  # n_genes


def test_obs_view_vector_columns_excluded(anndata_with_metadata):
    """Test that vector columns (obsm) are excluded from obs_view DataFrame"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Add a new obsm embedding
        adata.obsm["X_pca"] = np.random.rand(10, 50).astype(np.float32)  # n_cells

        # obs_view should NOT include vector columns
        df = adata.obs_view._get_dataframe()
        columns = list(df.columns)
        assert "X_umap" not in columns
        assert "X_pca" not in columns

        # But obsm should have them
        assert "X_umap" in adata.obsm
        assert "X_pca" in adata.obsm


def test_var_view_vector_columns_excluded(anndata_with_metadata):
    """Test that vector columns (varm) are excluded from var_view DataFrame"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Add a new varm embedding
        adata.varm["loadings"] = np.random.rand(5, 30).astype(np.float32)  # n_genes

        # var_view should NOT include vector columns
        df = adata.var_view._get_dataframe()
        columns = list(df.columns)
        assert "PCs" not in columns
        assert "loadings" not in columns

        # But varm should have them
        assert "PCs" in adata.varm
        assert "loadings" in adata.varm


def test_obs_view_cache_invalidation(anndata_with_metadata):
    """Test that obs_view DataFrame cache is invalidated when columns change"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access DataFrame (creates cache)
        df1 = adata.obs_view._get_dataframe()
        initial_columns = set(df1.columns)

        # Add a new column
        adata.obs_view["new_col"] = np.random.rand(10)  # n_cells

        # Access DataFrame again (should be refreshed)
        df2 = adata.obs_view._get_dataframe()
        new_columns = set(df2.columns)

        # New column should be present
        assert "new_col" in new_columns
        assert "new_col" not in initial_columns


def test_var_view_cache_invalidation(anndata_with_metadata):
    """Test that var_view DataFrame cache is invalidated when columns change"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Access DataFrame (creates cache)
        df1 = adata.var_view._get_dataframe()
        initial_columns = set(df1.columns)

        # Add a new column
        adata.var_view["new_col"] = np.random.rand(5)  # n_genes

        # Access DataFrame again (should be refreshed)
        df2 = adata.var_view._get_dataframe()
        new_columns = set(df2.columns)

        # New column should be present
        assert "new_col" in new_columns
        assert "new_col" not in initial_columns


def test_obs_view_selector_support(anndata_with_metadata):
    """Test that obs_view DataFrame respects selectors from parent LazyAnnData"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Subset adata
        subset_adata = adata[:5]

        # obs_view should respect selector
        df = subset_adata.obs_view._get_dataframe()
        assert len(df) == 5

        # Dict access should also respect selector
        cluster = subset_adata.obs_view["cluster"]
        assert len(cluster) == 5


def test_var_view_selector_support(anndata_with_metadata):
    """Test that var_view DataFrame respects selectors from parent LazyAnnData"""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Subset adata
        subset_adata = adata[:, :3]

        # var_view should respect selector
        df = subset_adata.var_view._get_dataframe()
        assert len(df) == 3

        # Dict access should also respect selector
        hvg = subset_adata.var_view["highly_variable"]
        assert len(hvg) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
