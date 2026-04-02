"""Tests for MoS / fragment cell-boundary reassembly correctness (synthetic fixture).

The `slaf_mos_boundary_reassembly` fixture provides real multi-fragment Lance data
with at least one cell whose expression rows span a fragment boundary.
"""

from __future__ import annotations

from pathlib import Path

import lance
import polars as pl
import pytest

pytestmark = [pytest.mark.slaf_array]


def test_fixture_has_at_least_two_lance_fragments(slaf_mos_boundary_reassembly):
    ds = lance.dataset(
        str(Path(slaf_mos_boundary_reassembly.slaf_path) / "expression.lance")
    )
    fragments = list(ds.get_fragments())
    assert len(fragments) >= 2, (
        "MoS uses one generator per fragment; need ≥2 fragments to exercise "
        "cross-fragment cell stitching."
    )


def test_fixture_has_cell_spanning_fragment_boundary(slaf_mos_boundary_reassembly):
    """Cell 1 is authored with rows in both the first and second Lance fragments."""
    ds = lance.dataset(
        str(Path(slaf_mos_boundary_reassembly.slaf_path) / "expression.lance")
    )
    frag_cell_ids: list[set[int]] = []
    for frag in ds.get_fragments():
        df = pl.from_arrow(frag.to_table())
        frag_cell_ids.append(set(df["cell_integer_id"].to_list()))

    spanning = [i for i in range(len(frag_cell_ids)) if 1 in frag_cell_ids[i]]
    assert len(spanning) >= 2, (
        "expected cell_integer_id 1 in multiple fragments; "
        f"per-fragment cell id sets: {frag_cell_ids!r}"
    )


def test_fixture_cell_start_index_matches_expression_counts(
    slaf_mos_boundary_reassembly,
):
    slaf = slaf_mos_boundary_reassembly
    slaf.wait_for_metadata()
    csi = slaf._cell_start_index
    n_cells = len(csi) - 1

    expr = pl.from_arrow(slaf.expression.to_table())
    observed = (
        expr.group_by("cell_integer_id")
        .agg(pl.len().alias("n"))
        .sort("cell_integer_id")
    )

    for row in observed.iter_rows(named=True):
        c = row["cell_integer_id"]
        n = row["n"]
        exp = int(csi[c + 1]) - int(csi[c])
        assert n == exp, f"cell {c}: expression rows {n}, _cell_start_index delta {exp}"

    assert observed.height == n_cells


def test_fixture_expected_row_and_gene_totals(slaf_mos_boundary_reassembly):
    slaf = slaf_mos_boundary_reassembly
    slaf.wait_for_metadata()
    assert slaf.config.get("n_cells") == 6
    assert slaf.config.get("n_genes") == 10
    assert slaf.config.get("metadata", {}).get("expression_count") == 16
