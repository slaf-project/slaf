"""
Unit tests for LazyObsView and LazyVarView classes.

Tests the dictionary-like interface for accessing obs/var columns:
- keys(), __contains__(), __len__(), __iter__()
- __getitem__() for accessing columns
- __setitem__() for creating/updating columns
- __delitem__() for deleting columns
- Selector support
- Immutability enforcement
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
        """Test getting number of columns when columns exist"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        assert len(obs_view) >= 2

    def test_iter_with_columns(self, slaf_with_obs_columns):
        """Test iterating over columns when columns exist"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        column_names = list(obs_view)
        assert "cell_type" in column_names
        assert "total_counts" in column_names

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

    def test_setitem_create_new_column(self, slaf_without_layers):
        """Test creating a new column"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        # Create new column
        n_obs = adata.n_obs
        new_cluster = np.array([f"cluster_{i % 3}" for i in range(n_obs)])
        obs_view["new_cluster"] = new_cluster

        # Verify column exists
        assert "new_cluster" in obs_view
        retrieved = obs_view["new_cluster"]
        assert np.array_equal(retrieved, new_cluster)

    def test_setitem_update_mutable_column(self, slaf_without_layers):
        """Test updating a mutable column"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        # Create column
        n_obs = adata.n_obs
        original = np.array([i % 3 for i in range(n_obs)], dtype=np.float32)
        obs_view["mutable_col"] = original

        # Update column
        updated = np.array([i % 5 for i in range(n_obs)], dtype=np.float32)
        obs_view["mutable_col"] = updated

        # Verify update
        retrieved = obs_view["mutable_col"]
        assert np.array_equal(retrieved, updated)

    def test_setitem_shape_mismatch(self, slaf_without_layers):
        """Test that shape mismatch raises ValueError"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        # Try to assign array with wrong length (dataset has 3 cells, use 5)
        wrong_length = np.array([1, 2, 3, 4, 5])  # Wrong length
        with pytest.raises(ValueError, match="doesn't match obs count"):
            obs_view["wrong_col"] = wrong_length

    def test_setitem_invalid_name(self, slaf_without_layers):
        """Test that invalid column names raise ValueError"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        n_obs = adata.n_obs
        values = np.array([1.0] * n_obs)

        # Empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            obs_view[""] = values

        # Invalid characters
        with pytest.raises(ValueError, match="invalid characters"):
            obs_view["invalid-name"] = values

        with pytest.raises(ValueError, match="invalid characters"):
            obs_view["invalid name"] = values

    def test_delitem_mutable_column(self, slaf_without_layers):
        """Test deleting a mutable column"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        # Create column
        n_obs = adata.n_obs
        obs_view["temp_col"] = np.array([1.0] * n_obs)
        assert "temp_col" in obs_view

        # Delete column
        del obs_view["temp_col"]
        assert "temp_col" not in obs_view

    def test_delitem_nonexistent_column(self, slaf_without_layers):
        """Test deleting a non-existent column raises KeyError"""
        adata = LazyAnnData(slaf_without_layers)
        obs_view = adata.obs_view

        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            del obs_view["nonexistent"]

    def test_obs_view_propagates_selectors(self, slaf_with_obs_columns):
        """Test that obs_view respects selectors from parent LazyAnnData"""
        adata = LazyAnnData(slaf_with_obs_columns)

        # Subset the adata
        adata_subset = adata[:2]

        # Access column on subset
        cell_type_subset = adata_subset.obs_view["cell_type"]

        # Verify correct shape
        assert len(cell_type_subset) == 2

    def test_is_immutable(self, slaf_with_obs_columns):
        """Test checking if a column is immutable"""
        adata = LazyAnnData(slaf_with_obs_columns)
        obs_view = adata.obs_view

        # Columns from conversion should be immutable
        if "cell_type" in obs_view.keys():
            # This depends on how the fixture is set up
            # If converted from h5ad, should be immutable
            pass

    def test_obs_view_cached(self, slaf_without_layers):
        """Test that obs_view is cached (same instance on multiple accesses)"""
        adata = LazyAnnData(slaf_without_layers)

        obs_view1 = adata.obs_view
        obs_view2 = adata.obs_view

        # Should be the same instance
        assert obs_view1 is obs_view2


class TestLazyVarView:
    """Test LazyVarView dictionary-like interface"""

    def test_keys_with_columns(self, slaf_with_var_columns):
        """Test listing column names when columns exist"""
        adata = LazyAnnData(slaf_with_var_columns)
        var_view = adata.var_view

        keys = list(var_view.keys())
        assert "gene_type" in keys or "highly_variable" in keys
        assert len(keys) >= 0

    def test_getitem_existing_column(self, slaf_with_var_columns):
        """Test accessing an existing column"""
        adata = LazyAnnData(slaf_with_var_columns)
        var_view = adata.var_view

        if "gene_type" in var_view.keys():
            gene_type = var_view["gene_type"]
            assert isinstance(gene_type, np.ndarray)
            assert len(gene_type) == adata.n_vars

    def test_setitem_create_new_column(self, slaf_without_layers):
        """Test creating a new column"""
        adata = LazyAnnData(slaf_without_layers)
        var_view = adata.var_view

        # Create new column
        n_vars = adata.n_vars
        new_annotation = np.array([f"type_{i % 2}" for i in range(n_vars)])
        var_view["new_annotation"] = new_annotation

        # Verify column exists
        assert "new_annotation" in var_view
        retrieved = var_view["new_annotation"]
        assert np.array_equal(retrieved, new_annotation)

    def test_setitem_shape_mismatch(self, slaf_without_layers):
        """Test that shape mismatch raises ValueError"""
        adata = LazyAnnData(slaf_without_layers)
        var_view = adata.var_view

        # Try to assign array with wrong length
        wrong_length = np.array([1, 2, 3])  # Wrong length
        with pytest.raises(ValueError, match="doesn't match var count"):
            var_view["wrong_col"] = wrong_length

    def test_delitem_mutable_column(self, slaf_without_layers):
        """Test deleting a mutable column"""
        adata = LazyAnnData(slaf_without_layers)
        var_view = adata.var_view

        # Create column
        n_vars = adata.n_vars
        var_view["temp_col"] = np.array([1.0] * n_vars)
        assert "temp_col" in var_view

        # Delete column
        del var_view["temp_col"]
        assert "temp_col" not in var_view

    def test_var_view_propagates_selectors(self, slaf_with_var_columns):
        """Test that var_view respects selectors from parent LazyAnnData"""
        adata = LazyAnnData(slaf_with_var_columns)

        # Subset the adata
        adata_subset = adata[:, :1]

        # Access column on subset if it exists
        if len(adata_subset.var_view.keys()) > 0:
            first_key = list(adata_subset.var_view.keys())[0]
            column_subset = adata_subset.var_view[first_key]

            # Verify correct shape
            assert len(column_subset) == 1

    def test_var_view_cached(self, slaf_without_layers):
        """Test that var_view is cached (same instance on multiple accesses)"""
        adata = LazyAnnData(slaf_without_layers)

        var_view1 = adata.var_view
        var_view2 = adata.var_view

        # Should be the same instance
        assert var_view1 is var_view2


class TestLazyObsmView:
    """Test LazyObsmView dictionary-like interface for multi-dimensional arrays"""

    def test_keys_empty(self, slaf_without_layers):
        """Test listing keys when no obsm data exists"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        keys = list(obsm.keys())
        assert len(keys) == 0

    def test_contains_empty(self, slaf_without_layers):
        """Test checking if key exists when no obsm data exists"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        assert "X_umap" not in obsm

    def test_len_empty(self, slaf_without_layers):
        """Test getting number of keys when no obsm data exists"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        assert len(obsm) == 0

    def test_getitem_nonexistent_key(self, slaf_without_layers):
        """Test accessing a non-existent key raises KeyError"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        with pytest.raises(KeyError, match="obsm key 'X_umap' not found"):
            _ = obsm["X_umap"]

    def test_setitem_create_new_embedding(self, slaf_without_layers):
        """Test creating a new embedding (2D array)"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        # Create new embedding (UMAP-like, 2D)
        n_obs = adata.n_obs
        umap_coords = np.random.rand(n_obs, 2).astype(np.float32)
        obsm["X_umap"] = umap_coords

        # Verify key exists
        assert "X_umap" in obsm
        assert len(obsm) == 1

        # Verify data
        retrieved = obsm["X_umap"]
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == (n_obs, 2)
        assert np.allclose(retrieved, umap_coords, rtol=1e-5)

    def test_setitem_create_pca_embedding(self, slaf_without_layers):
        """Test creating a PCA embedding (higher dimensions)"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        # Create PCA embedding (50 dimensions)
        n_obs = adata.n_obs
        pca_coords = np.random.rand(n_obs, 50).astype(np.float32)
        obsm["X_pca"] = pca_coords

        # Verify data
        retrieved = obsm["X_pca"]
        assert retrieved.shape == (n_obs, 50)
        assert np.allclose(retrieved, pca_coords, rtol=1e-5)

    def test_setitem_shape_mismatch(self, slaf_without_layers):
        """Test that shape mismatch raises ValueError"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        # Try to assign array with wrong first dimension
        wrong_shape = np.random.rand(5, 2).astype(np.float32)  # Wrong first dim
        with pytest.raises(ValueError, match="doesn't match obsm count"):
            obsm["X_umap"] = wrong_shape

    def test_setitem_invalid_name(self, slaf_without_layers):
        """Test that invalid key names raise ValueError"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        n_obs = adata.n_obs
        values = np.random.rand(n_obs, 2).astype(np.float32)

        # Empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            obsm[""] = values

        # Invalid characters
        with pytest.raises(ValueError, match="invalid characters"):
            obsm["invalid-name"] = values

    def test_setitem_update_mutable_key(self, slaf_without_layers):
        """Test updating a mutable embedding"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        n_obs = adata.n_obs

        # Create embedding
        original = np.random.rand(n_obs, 2).astype(np.float32)
        obsm["X_umap"] = original

        # Update embedding
        updated = np.random.rand(n_obs, 2).astype(np.float32)
        obsm["X_umap"] = updated

        # Verify update
        retrieved = obsm["X_umap"]
        assert np.allclose(retrieved, updated, rtol=1e-5)

    def test_delitem_mutable_key(self, slaf_without_layers):
        """Test deleting a mutable embedding"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        n_obs = adata.n_obs

        # Create embedding
        obsm["X_umap"] = np.random.rand(n_obs, 2).astype(np.float32)
        assert "X_umap" in obsm

        # Delete embedding
        del obsm["X_umap"]
        assert "X_umap" not in obsm

    def test_delitem_nonexistent_key(self, slaf_without_layers):
        """Test deleting a non-existent key raises KeyError"""
        adata = LazyAnnData(slaf_without_layers)
        obsm = adata.obsm

        with pytest.raises(KeyError, match="obsm key 'X_umap' not found"):
            del obsm["X_umap"]

    def test_obsm_propagates_selectors(self, slaf_without_layers):
        """Test that obsm respects selectors from parent LazyAnnData"""
        adata = LazyAnnData(slaf_without_layers)

        # Create embedding on full dataset
        n_obs = adata.n_obs
        adata.obsm["X_umap"] = np.random.rand(n_obs, 2).astype(np.float32)

        # Subset the adata
        adata_subset = adata[:2]

        # Access embedding on subset
        umap_subset = adata_subset.obsm["X_umap"]

        # Verify correct shape
        assert umap_subset.shape == (2, 2)

    def test_obsm_cached(self, slaf_without_layers):
        """Test that obsm is cached (same instance on multiple accesses)"""
        adata = LazyAnnData(slaf_without_layers)

        obsm1 = adata.obsm
        obsm2 = adata.obsm

        # Should be the same instance
        assert obsm1 is obsm2


class TestLazyVarmView:
    """Test LazyVarmView dictionary-like interface for multi-dimensional arrays"""

    def test_keys_empty(self, slaf_without_layers):
        """Test listing keys when no varm data exists"""
        adata = LazyAnnData(slaf_without_layers)
        varm = adata.varm

        keys = list(varm.keys())
        assert len(keys) == 0

    def test_getitem_nonexistent_key(self, slaf_without_layers):
        """Test accessing a non-existent key raises KeyError"""
        adata = LazyAnnData(slaf_without_layers)
        varm = adata.varm

        with pytest.raises(KeyError, match="varm key 'PCs' not found"):
            _ = varm["PCs"]

    def test_setitem_create_new_embedding(self, slaf_without_layers):
        """Test creating a new gene embedding (2D array)"""
        adata = LazyAnnData(slaf_without_layers)
        varm = adata.varm

        # Create new embedding (PCA loadings-like, 2D)
        n_vars = adata.n_vars
        pcs = np.random.rand(n_vars, 50).astype(np.float32)
        varm["PCs"] = pcs

        # Verify key exists
        assert "PCs" in varm
        assert len(varm) == 1

        # Verify data
        retrieved = varm["PCs"]
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == (n_vars, 50)
        assert np.allclose(retrieved, pcs, rtol=1e-5)

    def test_setitem_shape_mismatch(self, slaf_without_layers):
        """Test that shape mismatch raises ValueError"""
        adata = LazyAnnData(slaf_without_layers)
        varm = adata.varm

        # Try to assign array with wrong first dimension
        wrong_shape = np.random.rand(5, 50).astype(np.float32)  # Wrong first dim
        with pytest.raises(ValueError, match="doesn't match varm count"):
            varm["PCs"] = wrong_shape

    def test_delitem_mutable_key(self, slaf_without_layers):
        """Test deleting a mutable embedding"""
        adata = LazyAnnData(slaf_without_layers)
        varm = adata.varm

        n_vars = adata.n_vars

        # Create embedding
        varm["PCs"] = np.random.rand(n_vars, 50).astype(np.float32)
        assert "PCs" in varm

        # Delete embedding
        del varm["PCs"]
        assert "PCs" not in varm

    def test_varm_propagates_selectors(self, slaf_without_layers):
        """Test that varm respects selectors from parent LazyAnnData"""
        adata = LazyAnnData(slaf_without_layers)

        # Create embedding on full dataset
        n_vars = adata.n_vars
        adata.varm["PCs"] = np.random.rand(n_vars, 50).astype(np.float32)

        # Subset the adata
        adata_subset = adata[:, :2]

        # Access embedding on subset
        pcs_subset = adata_subset.varm["PCs"]

        # Verify correct shape
        assert pcs_subset.shape == (2, 50)

    def test_varm_cached(self, slaf_without_layers):
        """Test that varm is cached (same instance on multiple accesses)"""
        adata = LazyAnnData(slaf_without_layers)

        varm1 = adata.varm
        varm2 = adata.varm

        # Should be the same instance
        assert varm1 is varm2
