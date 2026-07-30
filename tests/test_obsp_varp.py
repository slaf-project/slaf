"""
Tests for obsp and varp support (pairwise matrices in COO storage).

- Conversion from h5ad (obsp/varp to cellsxcells.lance / genesxgenes.lance)
- LazyObspView / LazyVarpView access and mutations
- Selector support and immutability
"""

import os
import tempfile

import lance
import numpy as np
import pyarrow as pa
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csr

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


@pytest.fixture
def converted_obsp_varp_slaf(tmp_path, anndata_with_obsp_varp):
    converter = SLAFConverter(
        use_optimized_dtypes=False,
        compact_after_write=False,
        chunked=False,
    )
    slaf_path = tmp_path / "dataset.slaf"
    converter.convert_anndata(anndata_with_obsp_varp, str(slaf_path))
    return SLAFArray(slaf_path, load_metadata=False)


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


def test_convert_anndata_with_obsp_creates_source_cell_index(
    anndata_with_obsp_varp,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SLAFConverter(
            use_optimized_dtypes=False,
            compact_after_write=False,
            chunked=False,
            create_indices=True,
        )
        converter.convert_anndata(anndata_with_obsp_varp, tmpdir)

        cellsxcells = lance.dataset(os.path.join(tmpdir, "cellsxcells.lance"))
        indexed_fields = {
            field
            for index in cellsxcells.list_indices()
            for field in index.get("fields", [])
        }
        assert "cell_integer_id_i" in indexed_fields


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
        assert isspmatrix_csr(conv_conn)
        assert conv_conn.shape == (10, 10)
        np.testing.assert_array_almost_equal(conv_conn.toarray(), orig_conn, decimal=5)

        orig_dist = anndata_with_obsp_varp.obsp["distances"]
        conv_dist = adata.obsp["distances"]
        assert isspmatrix_csr(conv_dist)
        np.testing.assert_array_almost_equal(conv_dist.toarray(), orig_dist, decimal=5)


def test_get_obsp_entries_filters_source_rows(converted_obsp_varp_slaf):
    entries = converted_obsp_varp_slaf.get_obsp_entries(
        "connectivities", [2, 0, 2]
    ).sort(["cell_integer_id_i", "cell_integer_id_j"])

    assert entries.columns == [
        "cell_integer_id_i",
        "cell_integer_id_j",
        "connectivities",
    ]
    assert entries.select("cell_integer_id_i", "cell_integer_id_j").rows() == [
        (0, 1),
        (0, 2),
        (2, 0),
        (2, 1),
    ]
    np.testing.assert_allclose(
        entries["connectivities"].to_numpy(),
        [0.5, 0.2, 0.2, 0.3],
    )


def test_get_obsp_entries_supports_contiguous_and_empty_selectors(
    converted_obsp_varp_slaf,
):
    explicit = converted_obsp_varp_slaf.get_obsp_entries("distances", [0, 1]).sort(
        ["cell_integer_id_i", "cell_integer_id_j"]
    )
    contiguous = converted_obsp_varp_slaf.get_obsp_entries(
        "distances", slice(0, 2)
    ).sort(["cell_integer_id_i", "cell_integer_id_j"])
    empty = converted_obsp_varp_slaf.get_obsp_entries("distances", [])

    assert explicit.equals(contiguous)
    assert empty.is_empty()
    assert empty.columns == [
        "cell_integer_id_i",
        "cell_integer_id_j",
        "distances",
    ]


def test_get_obsp_entries_validates_key_schema_and_row_ids(
    converted_obsp_varp_slaf,
):
    with pytest.raises(KeyError, match="missing"):
        converted_obsp_varp_slaf.get_obsp_entries("missing", [0])
    with pytest.raises(ValueError, match="outside"):
        converted_obsp_varp_slaf.get_obsp_entries("connectivities", [10])
    with pytest.raises(TypeError, match="integers"):
        converted_obsp_varp_slaf.get_obsp_entries("connectivities", [0.5])

    converted_obsp_varp_slaf.cellsxcells = None
    with pytest.raises(ValueError, match="does not contain obsp"):
        converted_obsp_varp_slaf.get_obsp_entries("connectivities", [0])

    converted_obsp_varp_slaf.cellsxcells = converted_obsp_varp_slaf.cells
    with pytest.raises(ValueError, match="required columns"):
        converted_obsp_varp_slaf.get_obsp_entries("connectivities", [0])


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
        assert isspmatrix_csr(conv)
        assert conv.shape == (5, 5)
        np.testing.assert_array_almost_equal(conv.toarray(), orig, decimal=5)


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
        custom = adata.obsp["custom"]
        assert isspmatrix_csr(custom)
        np.testing.assert_array_almost_equal(custom.toarray(), new_mat, decimal=5)

        # Reload and check config + data
        slaf2 = SLAFArray(tmpdir, load_metadata=False)
        assert "custom" in slaf2.config["obsp"]["available"]
        assert "custom" in slaf2.config["obsp"]["mutable"]
        adata2 = LazyAnnData(slaf2)
        np.testing.assert_array_almost_equal(
            adata2.obsp["custom"].toarray(), new_mat, decimal=5
        )


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
        np.testing.assert_array_almost_equal(
            adata.varp["new_key"].toarray(), new_mat, decimal=5
        )

        slaf2 = SLAFArray(tmpdir, load_metadata=False)
        adata2 = LazyAnnData(slaf2)
        np.testing.assert_array_almost_equal(
            adata2.varp["new_key"].toarray(), new_mat, decimal=5
        )


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


def _write_obsp_h5ad(path, cell_prefix, adjacency, extra_obsp=None, obsm=None):
    import scanpy as sc

    n_cells = adjacency.shape[0]
    x = csr_matrix(np.eye(n_cells, 3, dtype=np.float32))
    adata = sc.AnnData(X=x)
    adata.obs_names = [f"{cell_prefix}_{idx}" for idx in range(n_cells)]
    adata.var_names = [f"gene_{idx}" for idx in range(3)]
    adata.obsp["adjacency_matrix"] = csr_matrix(adjacency, dtype=np.float32)
    for key, matrix in (extra_obsp or {}).items():
        adata.obsp[key] = csr_matrix(matrix, dtype=np.float32)
    for key, values in (obsm or {}).items():
        adata.obsm[key] = np.asarray(values, dtype=np.float32)
    adata.write_h5ad(path)


def _convert_h5ad_directory(input_dir, output_dir):
    converter = SLAFConverter(
        use_optimized_dtypes=False,
        compact_after_write=False,
        chunked=False,
        enable_checkpointing=False,
    )
    converter.convert(input_dir, output_dir)


def test_multi_file_h5ad_conversion_preserves_obsp_block_diagonal():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "out.slaf")
        os.makedirs(input_dir)
        first = np.zeros((2, 2), dtype=np.float32)
        first[0, 1] = first[1, 0] = 1.0
        second = np.zeros((3, 3), dtype=np.float32)
        second[0, 2] = 0.7
        second[2, 0] = 0.4
        _write_obsp_h5ad(os.path.join(input_dir, "a.h5ad"), "a", first)
        _write_obsp_h5ad(os.path.join(input_dir, "b.h5ad"), "b", second)

        _convert_h5ad_directory(input_dir, output_dir)

        slaf = SLAFArray(output_dir, load_metadata=False)
        assert slaf.config["tables"]["cellsxcells"] == "cellsxcells.lance"
        assert slaf.config["obsp"]["available"] == ["adjacency_matrix"]
        assert slaf.config["obsp"]["dimensions"]["adjacency_matrix"] == 5

        matrix = LazyAnnData(slaf).obsp["adjacency_matrix"]
        assert isspmatrix_csr(matrix)
        expected = np.zeros((5, 5), dtype=np.float32)
        expected[:2, :2] = first
        expected[2:, 2:] = second
        np.testing.assert_array_almost_equal(matrix.toarray(), expected, decimal=5)


def test_multi_file_h5ad_conversion_preserves_union_obsp_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "out.slaf")
        os.makedirs(input_dir)
        adjacency = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        distances = adjacency * 2.0
        _write_obsp_h5ad(
            os.path.join(input_dir, "a.h5ad"),
            "a",
            adjacency,
            extra_obsp={"distances": distances},
        )
        _write_obsp_h5ad(os.path.join(input_dir, "b.h5ad"), "b", adjacency)

        _convert_h5ad_directory(input_dir, output_dir)

        slaf = SLAFArray(output_dir, load_metadata=False)
        assert set(slaf.config["obsp"]["available"]) == {
            "adjacency_matrix",
            "distances",
        }
        lazy = LazyAnnData(slaf)
        expected_distances = np.zeros((4, 4), dtype=np.float32)
        expected_distances[:2, :2] = distances
        np.testing.assert_array_almost_equal(
            lazy.obsp["distances"].toarray(),
            expected_distances,
            decimal=5,
        )


def test_multi_file_h5ad_conversion_preserves_obsm_spatial():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "out.slaf")
        os.makedirs(input_dir)
        adjacency_a = np.zeros((2, 2), dtype=np.float32)
        adjacency_b = np.zeros((3, 3), dtype=np.float32)
        spatial_a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        spatial_b = np.asarray(
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            dtype=np.float32,
        )
        _write_obsp_h5ad(
            os.path.join(input_dir, "a.h5ad"),
            "a",
            adjacency_a,
            obsm={"spatial": spatial_a},
        )
        _write_obsp_h5ad(
            os.path.join(input_dir, "b.h5ad"),
            "b",
            adjacency_b,
            obsm={"spatial": spatial_b},
        )

        _convert_h5ad_directory(input_dir, output_dir)

        slaf = SLAFArray(output_dir, load_metadata=False)
        assert slaf.config["obsm"]["available"] == ["spatial"]
        assert slaf.config["obsm"]["dimensions"]["spatial"] == 2
        spatial_field = slaf.cells.schema.field("spatial")
        assert isinstance(spatial_field.type, pa.FixedSizeListType)
        assert spatial_field.type.list_size == 2
        np.testing.assert_array_almost_equal(
            LazyAnnData(slaf).obsm["spatial"],
            np.vstack([spatial_a, spatial_b]),
            decimal=5,
        )


def test_multi_file_h5ad_conversion_fills_missing_obsm_key_with_nan():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "out.slaf")
        os.makedirs(input_dir)
        adjacency = np.zeros((2, 2), dtype=np.float32)
        spatial_a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        umap_a = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        spatial_b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        _write_obsp_h5ad(
            os.path.join(input_dir, "a.h5ad"),
            "a",
            adjacency,
            obsm={"spatial": spatial_a, "X_umap": umap_a},
        )
        _write_obsp_h5ad(
            os.path.join(input_dir, "b.h5ad"),
            "b",
            adjacency,
            obsm={"spatial": spatial_b},
        )

        _convert_h5ad_directory(input_dir, output_dir)

        lazy = LazyAnnData(SLAFArray(output_dir, load_metadata=False))
        assert set(lazy.obsm.keys()) == {"spatial", "X_umap"}
        np.testing.assert_array_almost_equal(
            lazy.obsm["spatial"],
            np.vstack([spatial_a, spatial_b]),
            decimal=5,
        )
        converted_umap = lazy.obsm["X_umap"]
        np.testing.assert_array_almost_equal(converted_umap[:2], umap_a, decimal=5)
        assert np.isnan(converted_umap[2:]).all()


def test_multi_file_h5ad_conversion_skips_inconsistent_obsm_dimensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "out.slaf")
        os.makedirs(input_dir)
        adjacency = np.zeros((2, 2), dtype=np.float32)
        _write_obsp_h5ad(
            os.path.join(input_dir, "a.h5ad"),
            "a",
            adjacency,
            obsm={"X_pca": np.ones((2, 3), dtype=np.float32)},
        )
        _write_obsp_h5ad(
            os.path.join(input_dir, "b.h5ad"),
            "b",
            adjacency,
            obsm={"X_pca": np.ones((2, 4), dtype=np.float32)},
        )

        _convert_h5ad_directory(input_dir, output_dir)

        slaf = SLAFArray(output_dir, load_metadata=False)
        assert "obsm" not in slaf.config
        assert "X_pca" not in LazyAnnData(slaf).obsm
