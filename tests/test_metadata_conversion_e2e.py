"""
End-to-end tests for metadata conversion (obsm/varm/uns).

Tests metadata conversion functionality:
- Converting h5ad files with obsm/varm/uns to SLAF format
- Verifying obsm/varm/uns are correctly converted and accessible
- Verifying config.json has correct metadata (obsm/varm/uns)
- Verifying obsm/varm/uns are marked as immutable (converted from h5ad)
- Verifying dimensions are correctly tracked for obsm/varm
- Testing round-trip: convert -> load -> access -> verify data matches
"""

import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from slaf.core.slaf import SLAFArray
from slaf.data.converter import SLAFConverter
from slaf.integrations.anndata import LazyAnnData

# Mark all tests in this file as using SLAFArray instances
pytestmark = pytest.mark.slaf_array


@pytest.fixture
def anndata_with_metadata():
    """Create an AnnData object with obsm, varm, and uns for testing"""
    import scanpy as sc

    # Set random seed for reproducible tests
    np.random.seed(42)

    n_cells, n_genes = 10, 5

    # Create expression matrix
    X = csr_matrix(np.random.rand(n_cells, n_genes), dtype=np.float32)
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Add obsm (multi-dimensional obs annotations)
    adata.obsm["X_umap"] = np.random.rand(n_cells, 2).astype(np.float32)
    adata.obsm["X_pca"] = np.random.rand(n_cells, 50).astype(np.float32)

    # Add varm (multi-dimensional var annotations)
    adata.varm["PCs"] = np.random.rand(n_genes, 50).astype(np.float32)

    # Add uns (unstructured metadata)
    adata.uns["neighbors"] = {
        "params": {"n_neighbors": 15, "metric": "euclidean"},
        "connectivities_key": "connectivities",
    }
    adata.uns["pca"] = {
        "variance_ratio": np.array([0.1, 0.05, 0.03, 0.02, 0.015]),
        "variance": np.array([100.0, 50.0, 30.0, 20.0, 15.0]),
    }
    adata.uns["leiden"] = {"params": {"resolution": 0.5, "random_state": 0}}

    return adata


def test_convert_anndata_with_obsm(anndata_with_metadata):
    """Test converting AnnData with obsm to SLAF format"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert slaf.shape == anndata_with_metadata.shape
        assert slaf.config["format_version"] == "0.4"

        # Verify config has obsm metadata
        assert "obsm" in slaf.config
        assert "available" in slaf.config["obsm"]
        assert set(slaf.config["obsm"]["available"]) == {"X_umap", "X_pca"}
        assert set(slaf.config["obsm"]["immutable"]) == {"X_umap", "X_pca"}
        assert slaf.config["obsm"]["mutable"] == []

        # Verify dimensions are tracked
        assert "dimensions" in slaf.config["obsm"]
        assert slaf.config["obsm"]["dimensions"]["X_umap"] == 2
        assert slaf.config["obsm"]["dimensions"]["X_pca"] == 50


def test_convert_anndata_with_varm(anndata_with_metadata):
    """Test converting AnnData with varm to SLAF format"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert slaf.shape == anndata_with_metadata.shape

        # Verify config has varm metadata
        assert "varm" in slaf.config
        assert "available" in slaf.config["varm"]
        assert set(slaf.config["varm"]["available"]) == {"PCs"}
        assert set(slaf.config["varm"]["immutable"]) == {"PCs"}
        assert slaf.config["varm"]["mutable"] == []

        # Verify dimensions are tracked
        assert "dimensions" in slaf.config["varm"]
        assert slaf.config["varm"]["dimensions"]["PCs"] == 50


def test_convert_anndata_with_uns(anndata_with_metadata):
    """Test converting AnnData with uns to SLAF format"""
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Verify uns.json exists
        uns_path = Path(tmpdir) / "uns.json"
        assert uns_path.exists()

        # Load and verify uns.json content
        with open(uns_path) as f:
            uns_data = json.load(f)

        assert "neighbors" in uns_data
        assert "pca" in uns_data
        assert "leiden" in uns_data

        # Verify structure
        assert uns_data["neighbors"]["params"]["n_neighbors"] == 15
        assert uns_data["neighbors"]["params"]["metric"] == "euclidean"
        assert isinstance(uns_data["pca"]["variance_ratio"], list)
        assert len(uns_data["pca"]["variance_ratio"]) == 5
        assert uns_data["leiden"]["params"]["resolution"] == 0.5


def test_obsm_accessible_after_conversion(anndata_with_metadata):
    """Test that obsm embeddings are accessible after conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load and access obsm
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Verify obsm keys are accessible
        assert "X_umap" in adata.obsm
        assert "X_pca" in adata.obsm
        assert len(adata.obsm) == 2

        # Verify shapes match original
        umap_original = anndata_with_metadata.obsm["X_umap"]
        umap_converted = adata.obsm["X_umap"]
        assert umap_converted.shape == umap_original.shape
        assert umap_converted.shape == (10, 2)

        pca_original = anndata_with_metadata.obsm["X_pca"]
        pca_converted = adata.obsm["X_pca"]
        assert pca_converted.shape == pca_original.shape
        assert pca_converted.shape == (10, 50)

        # Verify data matches (within floating point tolerance)
        np.testing.assert_array_almost_equal(umap_converted, umap_original, decimal=5)
        np.testing.assert_array_almost_equal(pca_converted, pca_original, decimal=5)


def test_varm_accessible_after_conversion(anndata_with_metadata):
    """Test that varm embeddings are accessible after conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load and access varm
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Verify varm keys are accessible
        assert "PCs" in adata.varm
        assert len(adata.varm) == 1

        # Verify shapes match original
        pcs_original = anndata_with_metadata.varm["PCs"]
        pcs_converted = adata.varm["PCs"]
        assert pcs_converted.shape == pcs_original.shape
        assert pcs_converted.shape == (5, 50)

        # Verify data matches (within floating point tolerance)
        np.testing.assert_array_almost_equal(pcs_converted, pcs_original, decimal=5)


def test_uns_accessible_after_conversion(anndata_with_metadata):
    """Test that uns metadata is accessible after conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load and access uns
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Verify uns keys are accessible
        assert "neighbors" in adata.uns
        assert "pca" in adata.uns
        assert "leiden" in adata.uns
        assert len(adata.uns) == 3

        # Verify structure matches original
        assert adata.uns["neighbors"]["params"]["n_neighbors"] == 15
        assert adata.uns["neighbors"]["params"]["metric"] == "euclidean"
        assert isinstance(adata.uns["pca"]["variance_ratio"], list)
        assert len(adata.uns["pca"]["variance_ratio"]) == 5
        assert adata.uns["pca"]["variance_ratio"] == pytest.approx(
            [0.1, 0.05, 0.03, 0.02, 0.015]
        )
        assert adata.uns["leiden"]["params"]["resolution"] == 0.5


def test_obsm_immutable_after_conversion(anndata_with_metadata):
    """Test that converted obsm embeddings are immutable"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Try to delete immutable obsm key (should fail)
        with pytest.raises(ValueError, match="immutable"):
            del adata.obsm["X_umap"]

        # Try to overwrite immutable obsm key (should fail)
        with pytest.raises(ValueError, match="immutable"):
            adata.obsm["X_umap"] = np.random.rand(10, 2).astype(np.float32)

        # Verify original data still exists
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (10, 2)


def test_varm_immutable_after_conversion(anndata_with_metadata):
    """Test that converted varm embeddings are immutable"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Try to delete immutable varm key (should fail)
        with pytest.raises(ValueError, match="immutable"):
            del adata.varm["PCs"]

        # Try to overwrite immutable varm key (should fail)
        with pytest.raises(ValueError, match="immutable"):
            adata.varm["PCs"] = np.random.rand(5, 50).astype(np.float32)

        # Verify original data still exists
        assert "PCs" in adata.varm
        assert adata.varm["PCs"].shape == (5, 50)


def test_convert_anndata_without_metadata():
    """Test converting AnnData without obsm/varm/uns (backward compatibility)"""
    import scanpy as sc

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create AnnData without metadata
        n_cells, n_genes = 10, 5
        X = csr_matrix(np.random.rand(n_cells, n_genes), dtype=np.float32)
        adata = sc.AnnData(X=X)
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(adata, tmpdir)

        # Load and verify
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert slaf.shape == adata.shape

        # Verify config does not have obsm/varm metadata (or they are empty)
        # uns.json might not exist or be empty
        from pathlib import Path

        uns_path = Path(tmpdir) / "uns.json"
        if uns_path.exists():
            import json

            with open(uns_path) as f:
                uns_data = json.load(f)
            assert len(uns_data) == 0


def test_convert_anndata_with_all_metadata(anndata_with_metadata):
    """Test converting AnnData with obsm, varm, and uns together"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Load and verify all metadata
        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        # Verify obsm
        assert "X_umap" in adata.obsm
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_umap"].shape == (10, 2)
        assert adata.obsm["X_pca"].shape == (10, 50)

        # Verify varm
        assert "PCs" in adata.varm
        assert adata.varm["PCs"].shape == (5, 50)

        # Verify uns
        assert "neighbors" in adata.uns
        assert "pca" in adata.uns
        assert "leiden" in adata.uns

        # Verify config has all metadata
        assert "obsm" in slaf.config
        assert "varm" in slaf.config
        assert len(slaf.config["obsm"]["available"]) == 2
        assert len(slaf.config["varm"]["available"]) == 1


def test_round_trip_metadata_conversion(anndata_with_metadata):
    """Test round-trip: convert -> reload -> verify all metadata matches"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_metadata, tmpdir)

        # Reload
        slaf2 = SLAFArray(tmpdir, load_metadata=False)
        adata2 = LazyAnnData(slaf2)

        # Verify obsm data matches
        for key in anndata_with_metadata.obsm.keys():
            original = anndata_with_metadata.obsm[key]
            converted = adata2.obsm[key]
            assert converted.shape == original.shape
            np.testing.assert_array_almost_equal(converted, original, decimal=5)

        # Verify varm data matches
        for key in anndata_with_metadata.varm.keys():
            original = anndata_with_metadata.varm[key]
            converted = adata2.varm[key]
            assert converted.shape == original.shape
            np.testing.assert_array_almost_equal(converted, original, decimal=5)

        # Verify uns structure matches (JSON serialization may change types)
        assert set(adata2.uns.keys()) == set(anndata_with_metadata.uns.keys())
        assert adata2.uns["neighbors"]["params"]["n_neighbors"] == 15
        assert isinstance(adata2.uns["pca"]["variance_ratio"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
