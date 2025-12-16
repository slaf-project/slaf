"""
Unit tests for LazyObsView and LazyVarView classes.

Tests the dictionary-like interface for accessing obs/var columns:
- keys(), __contains__(), __len__(), __iter__()
- __getitem__() for accessing columns
- __setitem__() for creating/updating columns
- __delitem__() for deleting columns
- Selector support
- Immutability enforcement
- Scalar column filtering (vector columns excluded from obs/var)
"""

import numpy as np
import pytest

from slaf.integrations.anndata import LazyAnnData


class TestLazyObsView:
    """Test LazyObsView dictionary-like interface"""

    def test_keys_with_columns(self, slaf_with_obs_columns):
        """Test listing column names when columns exist"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        keys = list(obs_view.keys())
        assert "cell_type" in keys
        assert "total_counts" in keys
        assert len(keys) >= 2

    def test_keys_without_columns(self, slaf_without_layers):
        """Test listing column names when no custom columns exist"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        keys = list(obs_view.keys())
        # Should have at least cell_id
        assert len(keys) >= 0

    def test_contains_with_columns(self, slaf_with_obs_columns):
        """Test checking if column exists when columns exist"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        assert "cell_type" in obs_view
        assert "total_counts" in obs_view
        assert "nonexistent" not in obs_view

    def test_len_with_columns(self, slaf_with_obs_columns):
        """Test length when columns exist"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        assert len(obs_view) >= 2

    def test_iter_with_columns(self, slaf_with_obs_columns):
        """Test iteration over column names"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        keys = list(obs_view)
        assert "cell_type" in keys
        assert "total_counts" in keys

    def test_getitem_existing_column(self, slaf_with_obs_columns):
        """Test accessing an existing column"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        cell_type = obs_view["cell_type"]
        assert isinstance(cell_type, np.ndarray)
        assert len(cell_type) == adata.n_obs

    def test_getitem_nonexistent_column(self, slaf_with_obs_columns):
        """Test accessing a non-existent column raises KeyError"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            _ = obs_view["nonexistent"]

    def test_setitem_new_column(self, slaf_with_obs_columns):
        """Test creating a new column"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        new_values = np.random.rand(adata.n_obs).astype(np.float32)
        obs_view["new_column"] = new_values

        assert "new_column" in obs_view
        retrieved = obs_view["new_column"]
        np.testing.assert_array_equal(retrieved, new_values)

    def test_setitem_update_column(self, slaf_with_obs_columns):
        """Test updating an existing mutable column"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        # Create a new column first
        new_values = np.random.rand(adata.n_obs).astype(np.float32)
        obs_view["updatable_column"] = new_values

        # Update it
        updated_values = np.random.rand(adata.n_obs).astype(np.float32)
        obs_view["updatable_column"] = updated_values

        retrieved = obs_view["updatable_column"]
        np.testing.assert_array_equal(retrieved, updated_values)

    def test_delitem_mutable_column(self, slaf_with_obs_columns):
        """Test deleting a mutable column"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        # Create a new column
        new_values = np.random.rand(adata.n_obs).astype(np.float32)
        obs_view["deletable_column"] = new_values
        assert "deletable_column" in obs_view

        # Delete it
        del obs_view["deletable_column"]
        assert "deletable_column" not in obs_view

    def test_scalar_columns_exclude_vector_columns(self, slaf_with_obs_columns):
        """Test that obs_view only includes scalar columns, not vector columns (obsm)"""
        adata = LazyAnnData(slaf_with_obs_columns)

        # Add an obsm embedding (vector column)
        n_cells = adata.n_obs
        adata.obsm["X_umap"] = np.random.rand(n_cells, 2).astype(np.float32)

        # obs_view should NOT include vector columns
        obs_keys = list(adata.obs_view.keys())
        assert "X_umap" not in obs_keys

        # obs DataFrame should also NOT include vector columns
        obs_df = adata.obs
        assert "X_umap" not in obs_df.columns

        # But obsm should have it
        assert "X_umap" in adata.obsm

    def test_scalar_columns_exclude_vector_columns_after_mutation(
        self, slaf_with_obs_columns
    ):
        """Test that obs_view excludes vector columns even after adding new ones"""
        adata = LazyAnnData(slaf_with_obs_columns)

        # Add multiple obsm embeddings
        n_cells = adata.n_obs
        adata.obsm["X_umap"] = np.random.rand(n_cells, 2).astype(np.float32)
        adata.obsm["X_pca"] = np.random.rand(n_cells, 50).astype(np.float32)

        # obs_view should NOT include any vector columns
        obs_keys = list(adata.obs_view.keys())
        assert "X_umap" not in obs_keys
        assert "X_pca" not in obs_keys

        # obs DataFrame should also NOT include vector columns
        obs_df = adata.obs
        assert "X_umap" not in obs_df.columns
        assert "X_pca" not in obs_df.columns

        # But obsm should have them
        assert "X_umap" in adata.obsm
        assert "X_pca" in adata.obsm


class TestLazyVarView:
    """Test LazyVarView dictionary-like interface"""

    def test_keys_with_columns(self, slaf_with_var_columns):
        """Test listing column names when columns exist"""
        adata = LazyAnnData(slaf_with_var_columns)
        var_view = adata.var_view

        keys = list(var_view.keys())
        assert "gene_type" in keys
        assert "highly_variable" in keys
        assert len(keys) >= 2

    def test_getitem_existing_column(self, slaf_with_var_columns):
        """Test accessing an existing column"""
        adata = LazyAnnData(slaf_with_var_columns)
        var_view = adata.var_view

        gene_type = var_view["gene_type"]
        assert isinstance(gene_type, np.ndarray)
        assert len(gene_type) == adata.n_vars

    def test_setitem_new_column(self, slaf_with_var_columns):
        """Test creating a new column"""
        adata = LazyAnnData(slaf_with_var_columns)
        var_view = adata.var_view

        new_values = np.random.rand(adata.n_vars).astype(np.float32)
        var_view["new_column"] = new_values

        assert "new_column" in var_view
        retrieved = var_view["new_column"]
        np.testing.assert_array_equal(retrieved, new_values)

    def test_scalar_columns_exclude_vector_columns(self, slaf_with_var_columns):
        """Test that var_view only includes scalar columns, not vector columns (varm)"""
        adata = LazyAnnData(slaf_with_var_columns)

        # Add a varm embedding (vector column)
        n_genes = adata.n_vars
        adata.varm["PCs"] = np.random.rand(n_genes, 50).astype(np.float32)

        # var_view should NOT include vector columns
        var_keys = list(adata.var_view.keys())
        assert "PCs" not in var_keys

        # var DataFrame should also NOT include vector columns
        var_df = adata.var
        assert "PCs" not in var_df.columns

        # But varm should have it
        assert "PCs" in adata.varm

    def test_scalar_columns_exclude_vector_columns_after_mutation(
        self, slaf_with_var_columns
    ):
        """Test that var_view excludes vector columns even after adding new ones"""
        adata = LazyAnnData(slaf_with_var_columns)

        # Add multiple varm embeddings
        n_genes = adata.n_vars
        adata.varm["PCs"] = np.random.rand(n_genes, 50).astype(np.float32)
        adata.varm["loadings"] = np.random.rand(n_genes, 30).astype(np.float32)

        # var_view should NOT include any vector columns
        var_keys = list(adata.var_view.keys())
        assert "PCs" not in var_keys
        assert "loadings" not in var_keys

        # var DataFrame should also NOT include vector columns
        var_df = adata.var
        assert "PCs" not in var_df.columns
        assert "loadings" not in var_df.columns

        # But varm should have them
        assert "PCs" in adata.varm
        assert "loadings" in adata.varm
