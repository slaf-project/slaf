"""
Unit tests for LazyLayersView class.

Tests the dictionary-like interface for accessing layers:
- keys(), __contains__(), __len__(), __iter__()
- __getitem__() for accessing layers
- Empty layers for old datasets
- Config.json consistency handling
"""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
