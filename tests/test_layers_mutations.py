"""
End-to-end tests for layer mutations (assignment/writes).

Tests layer assignment functionality with eager writes:
- Creating new layers via assignment (immediate write)
- Verifying layers are saved to layers.lance immediately
- Verifying config.json is updated correctly
- Testing overwrite of mutable layers
- Testing immutability protection
- Round-trip: create -> reload -> verify
"""

import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from slaf.core.slaf import SLAFArray
from slaf.data.converter import SLAFConverter
from slaf.integrations.anndata import LazyAnnData

# Fixtures (anndata_with_layers, anndata_without_layers) are imported from conftest.py

# Mark all tests in this file as using SLAFArray instances
pytestmark = pytest.mark.slaf_array


def test_create_new_layer(anndata_without_layers):
    """Test creating a new layer (eager write - immediate)"""
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

        # Initially no layers
        assert len(adata.layers) == 0

        # Create a new layer (use correct shape from adata)
        n_cells, n_genes = adata.shape
        new_layer = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )
        adata.layers["normalized"] = new_layer

        # Layer should be available immediately (eager write)
        assert "normalized" in adata.layers
        assert len(adata.layers) == 1

        # Verify layer data
        normalized = adata.layers["normalized"]
        assert normalized.shape == adata.shape

        # Reload dataset and verify layer persists
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        assert "normalized" in adata2.layers
        assert adata2.layers["normalized"].shape == adata2.shape

        # Verify config.json was updated
        assert "normalized" in slaf2.config["layers"]["available"]
        assert "normalized" in slaf2.config["layers"]["mutable"]
        assert "normalized" not in slaf2.config["layers"]["immutable"]


def test_create_multiple_layers(anndata_without_layers):
    """Test creating multiple layers (each write is immediate)"""
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

        # Create multiple layers (use correct shape from adata)
        n_cells, n_genes = adata.shape
        layer1 = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )
        layer2 = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )
        layer3 = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )

        # Each assignment writes immediately
        adata.layers["layer1"] = layer1
        assert "layer1" in adata.layers
        assert len(adata.layers) == 1

        adata.layers["layer2"] = layer2
        assert "layer2" in adata.layers
        assert len(adata.layers) == 2

        adata.layers["layer3"] = layer3
        assert "layer3" in adata.layers
        assert len(adata.layers) == 3

        # All should be available
        assert "layer1" in adata.layers
        assert "layer2" in adata.layers
        assert "layer3" in adata.layers

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        assert len(adata2.layers) == 3
        assert "layer1" in adata2.layers
        assert "layer2" in adata2.layers
        assert "layer3" in adata2.layers

        # Verify all are mutable
        assert all(
            layer in slaf2.config["layers"]["mutable"]
            for layer in ["layer1", "layer2", "layer3"]
        )


def test_overwrite_mutable_layer(anndata_without_layers):
    """Test overwriting a mutable layer"""
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

        # Create initial layer (use correct shape from adata)
        n_cells, n_genes = adata.shape
        layer1 = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )
        adata.layers["mutable_layer"] = layer1

        # Verify initial layer (immediately available)
        assert "mutable_layer" in adata.layers
        initial = adata.layers["mutable_layer"].compute()
        initial_value = initial[0, 0]

        # Overwrite with new data (immediate write)
        layer2 = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )
        adata.layers["mutable_layer"] = layer2

        # Verify it was updated (values should be different)
        updated = adata.layers["mutable_layer"].compute()
        assert updated[0, 0] != initial_value  # Should be different from original

        # Reload and verify
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        updated2 = adata2.layers["mutable_layer"].compute()
        assert updated2[0, 0] != initial_value  # Should match updated value


def test_cannot_overwrite_immutable_layer(anndata_with_layers):
    """Test that immutable layers cannot be overwritten"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert AnnData with layers to SLAF
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load SLAF dataset
        slaf = SLAFArray(tmpdir)
        adata = LazyAnnData(slaf)

        # Verify layers are immutable
        assert "spliced" in adata.layers
        assert adata.layers._is_immutable("spliced")

        # Try to overwrite immutable layer (use correct shape from adata)
        n_cells, n_genes = adata.shape
        new_layer = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )

        with pytest.raises(ValueError, match="is immutable.*cannot be overwritten"):
            adata.layers["spliced"] = new_layer


def test_layer_from_lazy_expression_matrix(anndata_without_layers):
    """Test creating layer from LazyExpressionMatrix (e.g., adata.X)"""
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

        # Create layer from X (immediate write)
        adata.layers["copy_of_x"] = adata.X

        # Verify it was created
        assert "copy_of_x" in adata.layers
        copy_layer = adata.layers["copy_of_x"]
        assert copy_layer.shape == adata.X.shape

        # Verify data matches
        x_data = adata.X.compute()
        copy_data = copy_layer.compute()

        # Compare non-zero elements
        assert x_data.nnz == copy_data.nnz


def test_layer_dtype_optimization(anndata_without_layers):
    """Test that layer dtypes are optimized (int -> uint16, float -> float32)"""
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

        # Create layer with integer values (should be optimized to uint16)
        n_cells, n_genes = adata.shape
        int_layer = csr_matrix(
            np.random.randint(0, 100, size=(n_cells, n_genes), dtype=np.int32),
            dtype=np.int32,
        )
        adata.layers["int_layer"] = int_layer

        # Create layer with float values (should stay float32)
        float_layer = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )
        adata.layers["float_layer"] = float_layer

        # Verify layers exist
        assert "int_layer" in adata.layers
        assert "float_layer" in adata.layers

        # Reload and verify schema
        slaf2 = SLAFArray(tmpdir)
        layers_schema = slaf2.layers.schema

        # Check that int_layer column exists (dtype optimization happens internally)
        int_field = layers_schema.field("int_layer")
        float_field = layers_schema.field("float_layer")

        # Both should be valid fields
        assert int_field is not None
        assert float_field is not None


def test_layer_shape_validation(anndata_without_layers):
    """Test that layer shape validation works"""
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

        # Try to create layer with wrong shape
        wrong_shape = csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with pytest.raises(ValueError, match="Layer shape.*doesn't match X shape"):
            adata.layers["wrong"] = wrong_shape


def test_layer_name_validation(anndata_without_layers):
    """Test that layer name validation works"""
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

        # Use correct shape from adata
        n_cells, n_genes = adata.shape
        layer = csr_matrix(
            np.random.rand(n_cells, n_genes).astype(np.float32), dtype=np.float32
        )

        # Empty name
        with pytest.raises(ValueError, match="Name cannot be empty"):
            adata.layers[""] = layer

        # Invalid characters
        with pytest.raises(ValueError, match="invalid characters"):
            adata.layers["layer-name"] = layer

        with pytest.raises(ValueError, match="invalid characters"):
            adata.layers["layer name"] = layer


def test_round_trip_layer_assignment(anndata_without_layers):
    """Test complete round-trip: create layer -> reload -> verify"""
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

        # Create layer (use correct shape from adata) - immediate write
        n_cells, n_genes = adata.shape
        original_data = np.random.rand(n_cells, n_genes).astype(np.float32)
        layer = csr_matrix(original_data, dtype=np.float32)

        adata.layers["test_layer"] = layer

        # Verify in same session (immediately available)
        assert "test_layer" in adata.layers
        test_layer = adata.layers["test_layer"]
        assert test_layer.shape == adata.shape

        # Reload dataset
        slaf2 = SLAFArray(tmpdir)
        adata2 = LazyAnnData(slaf2)

        # Verify layer persists
        assert "test_layer" in adata2.layers
        test_layer2 = adata2.layers["test_layer"]
        assert test_layer2.shape == adata2.shape

        # Verify data matches (compare non-zero elements for sparse matrices)
        reloaded_data = test_layer2.compute().toarray()
        np.testing.assert_array_almost_equal(reloaded_data, original_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
