"""
Tests for obsp and varp support (pairwise matrices in COO storage).

- Conversion from h5ad (obsp/varp to cellsxcells.lance / genesxgenes.lance)
- LazyObspView / LazyVarpView access and mutations
- Selector support and immutability
"""

import os
import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from slaf.core.slaf import SLAFArray
from slaf.data.converter import SLAFConverter
from slaf.integrations.anndata import LazyAnnData

pytestmark = pytest.mark.slaf_array


@pytest.fixture
def anndata_with_obsp_varp():
    """AnnData with obsp and varp for testing."""
    import scanpy as sc

    np.random.seed(42)
    n_cells, n_genes = 10, 5
    X = csr_matrix(np.random.rand(n_cells, n_genes), dtype=np.float32)
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # obsp: square (n_cells, n_cells), sparse-ish
    conn = np.zeros((n_cells, n_cells), dtype=np.float32)
    conn[0, 1] = conn[1, 0] = 0.5
    conn[1, 2] = conn[2, 1] = 0.3
    conn[2, 0] = conn[0, 2] = 0.2
    adata.obsp["connectivities"] = conn
    adata.obsp["distances"] = conn * 2.0

    # varp: square (n_genes, n_genes)
    varp_mat = np.eye(n_genes, dtype=np.float32) * 0.5
    varp_mat[0, 1] = varp_mat[1, 0] = 0.1
    adata.varp["correlation"] = varp_mat

    return adata


def test_convert_anndata_with_obsp(anndata_with_obsp_varp):
    """Convert h5ad with obsp; cellsxcells.lance and config.obsp exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        assert os.path.isdir(os.path.join(tmpdir, "cellsxcells.lance"))
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert "obsp" in slaf.config
        assert set(slaf.config["obsp"]["available"]) == {"connectivities", "distances"}
        assert set(slaf.config["obsp"]["immutable"]) == {"connectivities", "distances"}
        assert slaf.config["obsp"]["dimensions"]["connectivities"] == 10
        assert slaf.config["tables"].get("cellsxcells") == "cellsxcells.lance"


def test_convert_anndata_with_varp(anndata_with_obsp_varp):
    """Convert h5ad with varp; genesxgenes.lance and config.varp exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        assert os.path.isdir(os.path.join(tmpdir, "genesxgenes.lance"))
        slaf = SLAFArray(tmpdir, load_metadata=False)
        assert "varp" in slaf.config
        assert "correlation" in slaf.config["varp"]["available"]
        assert slaf.config["varp"]["dimensions"]["correlation"] == 5
        assert slaf.config["tables"].get("genesxgenes") == "genesxgenes.lance"


def test_obsp_accessible_after_conversion(anndata_with_obsp_varp):
    """After conversion, adata.obsp[key] matches original and has correct shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp
        assert len(adata.obsp) == 2

        orig_conn = anndata_with_obsp_varp.obsp["connectivities"]
        conv_conn = adata.obsp["connectivities"]
        assert conv_conn.shape == (10, 10)
        np.testing.assert_array_almost_equal(conv_conn, orig_conn, decimal=5)

        orig_dist = anndata_with_obsp_varp.obsp["distances"]
        conv_dist = adata.obsp["distances"]
        np.testing.assert_array_almost_equal(conv_dist, orig_dist, decimal=5)


def test_varp_accessible_after_conversion(anndata_with_obsp_varp):
    """After conversion, adata.varp[key] matches original."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        assert "correlation" in adata.varp
        orig = anndata_with_obsp_varp.varp["correlation"]
        conv = adata.varp["correlation"]
        assert conv.shape == (5, 5)
        np.testing.assert_array_almost_equal(conv, orig, decimal=5)


def test_obsp_immutable_after_conversion(anndata_with_obsp_varp):
    """Converted obsp keys are immutable (delete/overwrite raise)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        with pytest.raises(ValueError, match="immutable"):
            del adata.obsp["connectivities"]

        with pytest.raises(ValueError, match="immutable"):
            adata.obsp["connectivities"] = np.zeros((10, 10), dtype=np.float32)

        assert "connectivities" in adata.obsp


def test_create_new_obsp_key(anndata_with_obsp_varp):
    """Create a new obsp key (mutable); round-trip and config updated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        new_mat = np.eye(10, dtype=np.float32) * 0.7
        new_mat[0, 1] = 0.2
        adata.obsp["custom"] = new_mat

        assert "custom" in adata.obsp
        np.testing.assert_array_almost_equal(adata.obsp["custom"], new_mat, decimal=5)

        # Reload and check config + data
        slaf2 = SLAFArray(tmpdir, load_metadata=False)
        assert "custom" in slaf2.config["obsp"]["available"]
        assert "custom" in slaf2.config["obsp"]["mutable"]
        adata2 = LazyAnnData(slaf2)
        np.testing.assert_array_almost_equal(adata2.obsp["custom"], new_mat, decimal=5)


def test_create_new_varp_key(anndata_with_obsp_varp):
    """Create a new varp key; round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        slaf = SLAFArray(tmpdir, load_metadata=False)
        adata = LazyAnnData(slaf)

        new_mat = np.eye(5, dtype=np.float32) * 0.3
        adata.varp["new_key"] = new_mat

        assert "new_key" in adata.varp
        np.testing.assert_array_almost_equal(adata.varp["new_key"], new_mat, decimal=5)

        slaf2 = SLAFArray(tmpdir, load_metadata=False)
        adata2 = LazyAnnData(slaf2)
        np.testing.assert_array_almost_equal(adata2.varp["new_key"], new_mat, decimal=5)


def test_empty_obsp_varp_when_absent():
    """When converting adata without obsp/varp, keys are empty and no extra tables."""
    import scanpy as sc

    np.random.seed(42)
    n_cells, n_genes = 5, 3
    X = csr_matrix(np.random.rand(n_cells, n_genes), dtype=np.float32)
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"c_{i}" for i in range(n_cells)]
    adata.var_names = [f"g_{i}" for i in range(n_genes)]
    # no adata.obsp, no adata.varp

    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
        )
        converter.convert_anndata(adata, tmpdir)

        slaf = SLAFArray(tmpdir, load_metadata=False)
        lazy = LazyAnnData(slaf)

        assert len(lazy.obsp) == 0
        assert len(lazy.varp) == 0
        assert "connectivities" not in lazy.obsp
        with pytest.raises(KeyError, match="obsp key"):
            _ = lazy.obsp["connectivities"]

        assert not os.path.isdir(os.path.join(tmpdir, "cellsxcells.lance"))
        assert not os.path.isdir(os.path.join(tmpdir, "genesxgenes.lance"))
