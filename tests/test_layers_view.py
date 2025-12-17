"""
Unit tests for LazyLayersView class.

Tests the dictionary-like interface for accessing layers:
- keys(), __contains__(), __len__(), __iter__()
- __getitem__() for accessing layers
- Empty layers for old datasets
- Config.json consistency handling
"""

import numpy as np
import pytest

from slaf.integrations.anndata import LazyAnnData


class TestLazyLayersView:
    """Test LazyLayersView dictionary-like interface"""

    def test_keys_with_layers(self, slaf_with_layers):
        """Test listing layer names when layers exist"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        keys = list(layers.keys())
        assert set(keys) == {"spliced", "unspliced"}
        assert len(keys) == 2

    def test_keys_without_layers(self, slaf_without_layers):
        """Test listing layer names when no layers exist (backward compatibility)"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        keys = list(layers.keys())
        assert keys == []
        assert len(keys) == 0

    def test_contains_with_layers(self, slaf_with_layers):
        """Test checking if layer exists when layers exist"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        assert "spliced" in layers
        assert "unspliced" in layers
        assert "nonexistent" not in layers

    def test_contains_without_layers(self, slaf_without_layers):
        """Test checking if layer exists when no layers exist"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        assert "spliced" not in layers
        assert "nonexistent" not in layers

    def test_len_with_layers(self, slaf_with_layers):
        """Test getting number of layers when layers exist"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        assert len(layers) == 2

    def test_len_without_layers(self, slaf_without_layers):
        """Test getting number of layers when no layers exist"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        assert len(layers) == 0

    def test_iter_with_layers(self, slaf_with_layers):
        """Test iterating over layers when layers exist"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        layer_names = list(layers)
        assert set(layer_names) == {"spliced", "unspliced"}

    def test_iter_without_layers(self, slaf_without_layers):
        """Test iterating over layers when no layers exist"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        layer_names = list(layers)
        assert layer_names == []

    def test_getitem_existing_layer(self, slaf_with_layers):
        """Test accessing an existing layer"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        spliced = layers["spliced"]
        assert spliced.shape == adata.X.shape
        assert spliced.shape == (3, 2)

        # Verify it's a LazyExpressionMatrix
        from slaf.integrations.anndata import LazyExpressionMatrix

        assert isinstance(spliced, LazyExpressionMatrix)
        assert spliced.table_name == "layers"
        assert spliced.layer_name == "spliced"

    def test_getitem_nonexistent_layer(self, slaf_with_layers):
        """Test accessing a non-existent layer raises KeyError"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        with pytest.raises(KeyError, match="Layer 'nonexistent' not found"):
            _ = layers["nonexistent"]

    def test_getitem_all_layers(self, slaf_with_layers):
        """Test accessing all available layers"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        spliced = layers["spliced"]
        unspliced = layers["unspliced"]

        # Both should have the same shape as X
        assert spliced.shape == adata.X.shape
        assert unspliced.shape == adata.X.shape

        # Both should be LazyExpressionMatrix instances
        from slaf.integrations.anndata import LazyExpressionMatrix

        assert isinstance(spliced, LazyExpressionMatrix)
        assert isinstance(unspliced, LazyExpressionMatrix)

    def test_layers_view_propagates_selectors(self, slaf_with_layers):
        """Test that layer matrices inherit selectors from parent LazyAnnData"""
        adata = LazyAnnData(slaf_with_layers)

        # Subset the adata
        adata_subset = adata[:2, :1]

        # Access layer on subset
        spliced_subset = adata_subset.layers["spliced"]

        # Verify selectors are propagated
        assert spliced_subset._cell_selector is not None
        assert spliced_subset._gene_selector is not None
        assert spliced_subset.shape == (2, 1)

    def test_is_immutable(self, slaf_with_layers):
        """Test checking if a layer is immutable"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        # Both layers should be immutable (converted from h5ad)
        assert layers._is_immutable("spliced")
        assert layers._is_immutable("unspliced")

    def test_layers_view_cached(self, slaf_with_layers):
        """Test that layers view is cached (same instance on multiple accesses)"""
        adata = LazyAnnData(slaf_with_layers)

        layers1 = adata.layers
        layers2 = adata.layers

        # Should be the same instance
        assert layers1 is layers2

    def test_layers_view_empty_dataset(self, slaf_without_layers):
        """Test layers view on dataset without layers (backward compatibility)"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Should work without errors
        assert len(layers) == 0
        assert list(layers.keys()) == []
        assert "spliced" not in layers

        # Accessing a layer should raise KeyError
        with pytest.raises(KeyError):
            _ = layers["spliced"]

    def test_setitem_eager_write(self, slaf_without_layers):
        """Test creating a new layer with eager write (immediate)"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create a new layer (eager write - immediate)
        new_layer = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        layers["normalized"] = new_layer

        # Layer should be available immediately
        assert "normalized" in layers.keys()
        assert "normalized" in layers

        # Verify layer data
        normalized = layers["normalized"]
        assert normalized.shape == (3, 2)

    def test_setitem_shape_mismatch(self, slaf_without_layers):
        """Test that setting layer with wrong shape raises ValueError"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Try to create layer with wrong shape
        wrong_shape_layer = scipy.sparse.csr_matrix(
            [[1, 2, 3], [4, 5, 6]], dtype=np.float32
        )  # (2, 3) instead of (3, 2)

        with pytest.raises(ValueError, match="Layer shape.*doesn't match X shape"):
            layers["wrong"] = wrong_shape_layer

    def test_setitem_invalid_layer_name(self, slaf_without_layers):
        """Test that invalid layer names raise ValueError"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        new_layer = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        # Empty name
        with pytest.raises(ValueError, match="Name cannot be empty"):
            layers[""] = new_layer

        # Invalid characters
        with pytest.raises(ValueError, match="invalid characters"):
            layers["layer-name"] = new_layer  # hyphen not allowed

        with pytest.raises(ValueError, match="invalid characters"):
            layers["layer name"] = new_layer  # space not allowed

    def test_setitem_overwrite_immutable(self, slaf_with_layers):
        """Test that overwriting immutable layer raises ValueError"""
        import scipy.sparse

        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        # Try to overwrite immutable layer
        new_layer = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        with pytest.raises(ValueError, match="is immutable.*cannot be overwritten"):
            layers["spliced"] = new_layer

    def test_setitem_overwrite_mutable(self, slaf_without_layers):
        """Test that overwriting mutable layer works"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create a new layer (immediate write)
        layer1 = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        layers["mutable_layer"] = layer1

        # Verify it exists immediately
        assert "mutable_layer" in layers

        # Overwrite it (immediate write)
        layer2 = scipy.sparse.csr_matrix(
            [[10, 20], [30, 40], [50, 60]], dtype=np.float32
        )
        layers["mutable_layer"] = layer2

        # Verify it was updated immediately
        assert "mutable_layer" in layers
        updated = layers["mutable_layer"]
        assert updated.shape == (3, 2)

    def test_create_multiple_layers(self, slaf_without_layers):
        """Test creating multiple layers (each write is immediate)"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create multiple layers (each assignment writes immediately)
        layer1 = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        layer2 = scipy.sparse.csr_matrix(
            [[10, 20], [30, 40], [50, 60]], dtype=np.float32
        )

        layers["layer1"] = layer1
        # First layer should be available immediately
        assert "layer1" in layers
        assert len(layers) == 1

        layers["layer2"] = layer2
        # Both should be available immediately
        assert "layer1" in layers
        assert "layer2" in layers
        assert len(layers) == 2

    def test_setitem_lazy_expression_matrix(self, slaf_without_layers):
        """Test setting layer from LazyExpressionMatrix"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create layer from X (LazyExpressionMatrix) - immediate write
        layers["copy_of_x"] = adata.X

        # Verify it was created immediately
        assert "copy_of_x" in layers
        copy_layer = layers["copy_of_x"]
        assert copy_layer.shape == adata.X.shape

    def test_setitem_dense_matrix(self, slaf_without_layers):
        """Test setting layer from dense numpy array (converted to sparse)"""
        import numpy as np

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create dense matrix
        dense_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        layers["dense_layer"] = dense_matrix

        # Verify it was created and converted to sparse (immediately available)
        assert "dense_layer" in layers
        dense_layer = layers["dense_layer"]
        assert dense_layer.shape == (3, 2)

    def test_delitem_mutable_layer(self, slaf_without_layers):
        """Test deleting a mutable layer"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create a new layer (mutable)
        layer = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        layers["mutable_layer"] = layer

        # Verify it exists
        assert "mutable_layer" in layers
        assert len(layers) == 1

        # Delete it
        del layers["mutable_layer"]

        # Verify it's gone
        assert "mutable_layer" not in layers
        assert len(layers) == 0

        # Verify it raises KeyError when accessing
        with pytest.raises(KeyError, match="Layer 'mutable_layer' not found"):
            _ = layers["mutable_layer"]

    def test_delitem_immutable_layer(self, slaf_with_layers):
        """Test that deleting immutable layer raises ValueError"""
        adata = LazyAnnData(slaf_with_layers)
        layers = adata.layers

        # Verify layer exists and is immutable
        assert "spliced" in layers
        assert layers._is_immutable("spliced")

        # Try to delete immutable layer
        with pytest.raises(ValueError, match="is immutable.*cannot be deleted"):
            del layers["spliced"]

        # Verify it still exists
        assert "spliced" in layers

    def test_delitem_nonexistent_layer(self, slaf_without_layers):
        """Test that deleting non-existent layer raises KeyError"""
        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Try to delete non-existent layer
        with pytest.raises(KeyError, match="Layer 'nonexistent' not found"):
            del layers["nonexistent"]

    def test_delitem_multiple_layers(self, slaf_without_layers):
        """Test deleting multiple mutable layers"""
        import scipy.sparse

        adata = LazyAnnData(slaf_without_layers)
        layers = adata.layers

        # Create multiple layers
        layer1 = scipy.sparse.csr_matrix([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        layer2 = scipy.sparse.csr_matrix(
            [[10, 20], [30, 40], [50, 60]], dtype=np.float32
        )
        layer3 = scipy.sparse.csr_matrix(
            [[100, 200], [300, 400], [500, 600]], dtype=np.float32
        )

        layers["layer1"] = layer1
        layers["layer2"] = layer2
        layers["layer3"] = layer3

        assert len(layers) == 3

        # Delete one layer
        del layers["layer2"]
        assert "layer2" not in layers
        assert len(layers) == 2
        assert "layer1" in layers
        assert "layer3" in layers

        # Delete another layer
        del layers["layer1"]
        assert "layer1" not in layers
        assert len(layers) == 1
        assert "layer3" in layers

        # Delete last layer
        del layers["layer3"]
        assert "layer3" not in layers
        assert len(layers) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
