"""Tests for configurable TileDB SOMA X layer selection (ChunkedTileDBReader / PR #45)."""

from __future__ import annotations

import pytest

pytest.importorskip("tiledbsoma")

from slaf.data.chunked_reader import ChunkedTileDBReader, create_chunked_reader


def _min_nonzero_value(reader: ChunkedTileDBReader) -> float:
    for chunk_table, _ in reader.iter_chunks(chunk_size=1000):
        vals = (
            chunk_table.column("value").combine_chunks().to_numpy(zero_copy_only=False)
        )
        nz = vals[vals != 0]
        if nz.size:
            return float(nz.min())
    pytest.fail("expected at least one non-zero matrix entry")


def test_chunked_tiledb_reader_default_layer_matches_data(
    tiledb_soma_two_x_layers_path,
):
    with ChunkedTileDBReader(tiledb_soma_two_x_layers_path) as r_default:
        with ChunkedTileDBReader(
            tiledb_soma_two_x_layers_path, layer_name="data"
        ) as r_data:
            assert r_default.n_obs == r_data.n_obs == 5
            assert r_default.n_vars == r_data.n_vars == 3
            assert _min_nonzero_value(r_default) == pytest.approx(
                _min_nonzero_value(r_data)
            )


def test_chunked_tiledb_reader_norm_layer_scaled(tiledb_soma_two_x_layers_path):
    with ChunkedTileDBReader(
        tiledb_soma_two_x_layers_path, layer_name="data"
    ) as r_data:
        with ChunkedTileDBReader(
            tiledb_soma_two_x_layers_path, layer_name="norm"
        ) as r_norm:
            v_data = _min_nonzero_value(r_data)
            v_norm = _min_nonzero_value(r_norm)
            assert v_data > 0
            assert v_norm == pytest.approx(v_data * 10.0)


def test_create_chunked_reader_passes_layer_name(tiledb_soma_two_x_layers_path):
    with create_chunked_reader(
        tiledb_soma_two_x_layers_path, layer_name="norm"
    ) as reader:
        assert isinstance(reader, ChunkedTileDBReader)
        assert reader.layer_name == "norm"
    with create_chunked_reader(
        tiledb_soma_two_x_layers_path, layer_name="data"
    ) as r_data:
        with create_chunked_reader(
            tiledb_soma_two_x_layers_path, layer_name="norm"
        ) as r_norm:
            assert _min_nonzero_value(r_norm) == pytest.approx(
                10.0 * _min_nonzero_value(r_data)
            )


def test_chunked_tiledb_reader_missing_layer_error_includes_available(
    tiledb_soma_two_x_layers_path,
):
    with pytest.raises(ValueError, match="Available layers"):
        with ChunkedTileDBReader(
            tiledb_soma_two_x_layers_path, layer_name="nonexistent"
        ):
            pass
