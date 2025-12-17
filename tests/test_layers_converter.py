"""
Unit Tests for Layer Conversion Functionality

These tests focus on verifying specific aspects of layer conversion:
- Schema validation (wide format)
- Config metadata (immutability, layer lists)
- Converter behavior with layers

These are unit tests that don't require full end-to-end conversion workflows.
"""

import tempfile

import pytest

from slaf.core.slaf import SLAFArray
from slaf.data.converter import SLAFConverter

# Mark all tests in this file as using SLAFArray instances
pytestmark = pytest.mark.slaf_array


def test_layers_wide_format(anndata_with_layers):
    """Test that layers are stored in wide format (one column per layer)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and check schema
        slaf = SLAFArray(tmpdir, load_metadata=False)
        schema = slaf.layers.schema

        # Verify wide format: one column per layer
        column_names = [field.name for field in schema]
        assert "cell_integer_id" in column_names
        assert "gene_integer_id" in column_names
        assert "spliced" in column_names
        assert "unspliced" in column_names

        # Verify it's NOT long format (no layer_name column)
        assert "layer_name" not in column_names


def test_layers_immutable_after_conversion(anndata_with_layers):
    """Test that converted layers are marked as immutable"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and check immutability
        slaf = SLAFArray(tmpdir, load_metadata=False)

        # Verify layers are in immutable list
        assert "spliced" in slaf.config["layers"]["immutable"]
        assert "unspliced" in slaf.config["layers"]["immutable"]

        # Verify layers are not in mutable list
        assert "spliced" not in slaf.config["layers"]["mutable"]
        assert "unspliced" not in slaf.config["layers"]["mutable"]


def test_layers_config_metadata(anndata_with_layers):
    """Test that config.json has correct layers metadata after conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and check config
        slaf = SLAFArray(tmpdir, load_metadata=False)

        # Verify layers metadata structure
        assert "layers" in slaf.config
        assert "available" in slaf.config["layers"]
        assert "immutable" in slaf.config["layers"]
        assert "mutable" in slaf.config["layers"]

        # Verify layer names
        assert set(slaf.config["layers"]["available"]) == {"spliced", "unspliced"}
        assert set(slaf.config["layers"]["immutable"]) == {"spliced", "unspliced"}
        assert slaf.config["layers"]["mutable"] == []

        # Verify layers table is in config
        assert "layers" in slaf.config["tables"]
        assert slaf.config["tables"]["layers"] == "layers.lance"


def test_layers_config_without_layers(anndata_without_layers):
    """Test that config.json does not have layers metadata when no layers exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load and check config
        slaf = SLAFArray(tmpdir, load_metadata=False)

        # Verify layers metadata does not exist
        assert "layers" not in slaf.config
        assert "layers" not in slaf.config["tables"]


def test_layers_schema_types(anndata_with_layers):
    """Test that layers schema has correct data types"""
    import pyarrow as pa

    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and check schema
        slaf = SLAFArray(tmpdir, load_metadata=False)
        schema = slaf.layers.schema

        # Verify data types
        # When use_optimized_dtypes=False, integer IDs use int32 instead of uint32/uint16
        assert schema.field("cell_integer_id").type == pa.int32()
        assert schema.field("gene_integer_id").type == pa.int32()
        # Layers should be float (float64) when use_optimized_dtypes=False
        # (float32 is used in chunked conversion, but non-chunked uses float64)
        spliced_type = schema.field("spliced").type
        unspliced_type = schema.field("unspliced").type
        # Accept float (float64) or float32
        assert spliced_type in [
            pa.float64(),
            pa.float32(),
        ], f"Expected float64 or float32, got {spliced_type}"
        assert unspliced_type in [
            pa.float64(),
            pa.float32(),
        ], f"Expected float64 or float32, got {unspliced_type}"
