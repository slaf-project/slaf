"""
End-to-end tests for layer conversion functionality.

Tests layer conversion functionality:
- Converting h5ad files with layers to SLAF format
- Verifying layers are correctly converted and accessible
- Verifying config.json has correct layers metadata
- Verifying layers are marked as immutable (converted from h5ad)
- Testing round-trip: convert -> load -> access
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


def test_convert_anndata_with_layers(anndata_with_layers):
    """Test converting AnnData with layers to SLAF format"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert slaf.shape == anndata_with_layers.shape
        assert slaf.config["format_version"] == "0.4"

        # Verify layers table exists
        assert slaf.layers is not None

        # Verify config has layers metadata
        assert "layers" in slaf.config
        assert "available" in slaf.config["layers"]
        assert set(slaf.config["layers"]["available"]) == {"spliced", "unspliced"}
        assert set(slaf.config["layers"]["immutable"]) == {"spliced", "unspliced"}
        assert slaf.config["layers"]["mutable"] == []

        # Verify layers table is in config
        assert "layers" in slaf.config["tables"]
        assert slaf.config["tables"]["layers"] == "layers.lance"


def test_convert_anndata_without_layers(anndata_without_layers):
    """Test converting AnnData without layers (backward compatibility)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert slaf.shape == anndata_without_layers.shape
        assert slaf.config["format_version"] == "0.4"

        # Verify layers table does not exist
        assert slaf.layers is None

        # Verify config does not have layers metadata
        assert "layers" not in slaf.config

        # Verify layers table is not in config
        assert "layers" not in slaf.config["tables"]


def test_layers_accessible_after_conversion(anndata_with_layers):
    """Test that layers are accessible after conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and access layers
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Verify layers are accessible
        assert "spliced" in adata.layers
        assert "unspliced" in adata.layers
        assert len(adata.layers) == 2

        # Access layers
        spliced = adata.layers["spliced"]
        unspliced = adata.layers["unspliced"]

        # Verify shapes
        assert spliced.shape == adata.X.shape
        assert unspliced.shape == adata.X.shape
        assert spliced.shape == anndata_with_layers.shape


def test_layers_data_preserved(anndata_with_layers):
    """Test that layer data is preserved during conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get original layer data
        original_spliced = anndata_with_layers.layers["spliced"].toarray()

        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and access layers
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Get converted layer data
        converted_spliced = adata.layers["spliced"].compute().toarray()

        # Verify data matches (within floating point precision)
        np.testing.assert_array_almost_equal(
            original_spliced, converted_spliced, decimal=5
        )


def test_layers_query_after_conversion(anndata_with_layers):
    """Test querying layers table after conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and query
        slaf = SLAFArray(tmpdir, load_metadata=False)

        # Query layers table
        result = slaf.query(
            """
            SELECT cell_integer_id, gene_integer_id, spliced
            FROM layers
            WHERE spliced IS NOT NULL
            LIMIT 10
            """
        )

        # Verify results
        assert len(result) > 0
        assert "spliced" in result.columns
        assert "cell_integer_id" in result.columns
        assert "gene_integer_id" in result.columns


def test_convert_multiple_layers(anndata_with_layers):
    """Test converting AnnData with multiple layers"""
    # Add more layers
    anndata_with_layers.layers["counts"] = csr_matrix(
        np.random.rand(anndata_with_layers.n_obs, anndata_with_layers.n_vars),
        dtype=np.float32,
    )
    anndata_with_layers.layers["velocity"] = csr_matrix(
        np.random.rand(anndata_with_layers.n_obs, anndata_with_layers.n_vars),
        dtype=np.float32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Verify all layers are present
        assert len(adata.layers) == 4
        assert set(adata.layers.keys()) == {
            "spliced",
            "unspliced",
            "counts",
            "velocity",
        }

        # Verify all are in config
        assert set(slaf.config["layers"]["available"]) == {
            "spliced",
            "unspliced",
            "counts",
            "velocity",
        }
        assert set(slaf.config["layers"]["immutable"]) == {
            "spliced",
            "unspliced",
            "counts",
            "velocity",
        }


def test_round_trip_conversion(anndata_with_layers):
    """Test complete round-trip: convert -> load -> access -> verify"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_layers, tmpdir)

        # Step 2: Load
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Step 3: Access and verify
        assert len(adata.layers) == 2
        assert "spliced" in adata.layers
        assert "unspliced" in adata.layers

        # Step 4: Compute and verify data
        spliced_matrix = adata.layers["spliced"].compute()
        unspliced_matrix = adata.layers["unspliced"].compute()

        assert spliced_matrix.shape == anndata_with_layers.shape
        assert unspliced_matrix.shape == anndata_with_layers.shape

        # Step 5: Verify config
        assert slaf.config["format_version"] == "0.4"
        assert "layers" in slaf.config["tables"]
        assert "layers" in slaf.config


def test_layers_empty_after_conversion(anndata_without_layers):
    """Test that datasets without layers have empty layers view"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_without_layers, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Layers should be empty
        assert len(adata.layers) == 0
        assert list(adata.layers.keys()) == []
        assert "spliced" not in adata.layers


def test_convert_anndata_chunked_with_layers(anndata_with_layers):
    """Test chunked conversion with layers"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to h5ad first, then convert with chunked processing
        h5ad_path = f"{tmpdir}/test.h5ad"
        anndata_with_layers.write(h5ad_path)

        # Convert with chunked processing
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=True,
            chunk_size=3,  # Small chunk size for testing
        )
        converter.convert(h5ad_path, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert slaf.shape == anndata_with_layers.shape
        assert slaf.config["format_version"] == "0.4"

        # Verify layers table exists
        assert slaf.layers is not None

        # Verify config has layers metadata
        assert "layers" in slaf.config
        assert set(slaf.config["layers"]["available"]) == {"spliced", "unspliced"}

        # Access layers and verify they work
        adata = LazyAnnData(slaf)
        spliced = adata.layers["spliced"].compute()
        assert spliced.shape == anndata_with_layers.shape


def test_convert_multiple_files_consistent_layers():
    """Test multi-file conversion with consistent layers"""
    import scanpy as sc

    # Create multiple files with consistent layers
    n_cells_per_file, n_genes = 10, 5
    files = []
    original_layers_data = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(2):
            X = csr_matrix(np.random.rand(n_cells_per_file, n_genes), dtype=np.float32)
            adata = sc.AnnData(X=X)
            adata.obs_names = [f"cell_{i}_{j}" for j in range(n_cells_per_file)]
            adata.var_names = [f"gene_{j}" for j in range(n_genes)]

            # Add consistent layers
            spliced = csr_matrix(
                np.random.rand(n_cells_per_file, n_genes), dtype=np.float32
            )
            unspliced = csr_matrix(
                np.random.rand(n_cells_per_file, n_genes), dtype=np.float32
            )
            adata.layers["spliced"] = spliced
            adata.layers["unspliced"] = unspliced

            # Store original data for verification
            original_layers_data.append(
                {
                    "spliced": spliced.toarray(),
                    "unspliced": unspliced.toarray(),
                }
            )

            h5ad_path = f"{tmpdir}/file_{i}.h5ad"
            adata.write(h5ad_path)
            files.append(h5ad_path)

        # Convert multiple files
        output_path = f"{tmpdir}/output.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=5,
            use_optimized_dtypes=False,
            compact_after_write=False,
        )
        converter.convert(files, output_path)

        # Load and verify
        slaf = SLAFArray(output_path, load_metadata=False)
        assert slaf.config["format_version"] == "0.4"

        # Verify layers table exists
        assert slaf.layers is not None

        # Verify config has layers metadata
        assert "layers" in slaf.config
        assert set(slaf.config["layers"]["available"]) == {"spliced", "unspliced"}

        # Access layers and verify data
        adata = LazyAnnData(slaf)
        spliced = adata.layers["spliced"].compute().toarray()
        unspliced = adata.layers["unspliced"].compute().toarray()

        # Verify shapes
        assert spliced.shape == (n_cells_per_file * 2, n_genes)
        assert unspliced.shape == (n_cells_per_file * 2, n_genes)

        # Verify data matches (first file's data should be in first n_cells_per_file rows)
        np.testing.assert_array_almost_equal(
            spliced[:n_cells_per_file],
            original_layers_data[0]["spliced"],
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            unspliced[:n_cells_per_file],
            original_layers_data[0]["unspliced"],
            decimal=5,
        )


def test_convert_multiple_files_inconsistent_layers():
    """Test multi-file conversion with inconsistent layers (should warn and skip)"""
    import scanpy as sc

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with inconsistent layers
        n_cells_per_file, n_genes = 10, 5
        files = []

        for i in range(2):
            X = csr_matrix(np.random.rand(n_cells_per_file, n_genes), dtype=np.float32)
            adata = sc.AnnData(X=X)
            adata.obs_names = [f"cell_{i}_{j}" for j in range(n_cells_per_file)]
            adata.var_names = [f"gene_{j}" for j in range(n_genes)]

            # Add different layers for each file
            if i == 0:
                adata.layers["spliced"] = csr_matrix(
                    np.random.rand(n_cells_per_file, n_genes), dtype=np.float32
                )
                adata.layers["unspliced"] = csr_matrix(
                    np.random.rand(n_cells_per_file, n_genes), dtype=np.float32
                )
            else:
                # Different layer name
                adata.layers["counts"] = csr_matrix(
                    np.random.rand(n_cells_per_file, n_genes), dtype=np.float32
                )

            h5ad_path = f"{tmpdir}/file_{i}.h5ad"
            adata.write(h5ad_path)
            files.append(h5ad_path)

        # Convert multiple files
        output_path = f"{tmpdir}/output.slaf"
        converter = SLAFConverter(
            chunked=True,
            chunk_size=5,
            use_optimized_dtypes=False,
            compact_after_write=False,
        )
        converter.convert(files, output_path)

        # Load and verify
        slaf = SLAFArray(output_path, load_metadata=False)
        assert slaf.config["format_version"] == "0.4"

        # Verify layers table does NOT exist
        assert slaf.layers is None

        # Verify config does NOT have layers metadata
        assert "layers" not in slaf.config
        assert "layers" not in slaf.config["tables"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
