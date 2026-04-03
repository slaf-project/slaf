"""Tests for MoS / fragment cell-boundary reassembly correctness (synthetic fixture).

The `slaf_mos_boundary_reassembly` fixture provides real multi-fragment Lance data
with at least one cell whose expression rows span a fragment boundary.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import lance
import polars as pl
import pytest

from slaf.ml.aggregators import ScGPTWindow
from slaf.ml.datasets import PrefetchBatchProcessor, RawPrefetchBatch
from slaf.ml.samplers import RandomShuffle

pytestmark = [pytest.mark.slaf_array]


def _collect_raw_prefetch_epoch(
    processor: PrefetchBatchProcessor,
) -> list[RawPrefetchBatch]:
    batches: list[RawPrefetchBatch] = []
    while True:
        try:
            batches.append(processor.load_prefetch_batch())
        except StopIteration:
            return batches


def _assert_raw_epoch_matches_cell_start_index(
    slaf, batches: list[RawPrefetchBatch]
) -> None:
    slaf.wait_for_metadata()
    csi = slaf._cell_start_index
    n_cells = len(csi) - 1

    all_dfs: list[pl.DataFrame] = []
    for batch in batches:
        all_dfs.extend(batch.batch_dfs)
    combined = pl.concat(all_dfs, how="vertical")

    mismatches: list[str] = []
    for row in (
        combined.group_by("cell_integer_id")
        .agg(pl.len().alias("gene_count"))
        .sort("cell_integer_id")
        .iter_rows(named=True)
    ):
        c = int(row["cell_integer_id"])
        obs = int(row["gene_count"])
        exp = int(csi[c + 1]) - int(csi[c])
        if obs != exp:
            mismatches.append(f"cell {c}: observed {obs} genes, expected {exp}")

    assert not mismatches, "\n".join(mismatches[:20])

    pairs = combined.select("cell_integer_id", "gene_integer_id")
    assert len(pairs) == len(pairs.unique()), (
        f"duplicate (cell, gene) rows: {len(pairs) - len(pairs.unique())} duplicates"
    )

    all_cell_ids: list[int] = []
    for batch in batches:
        all_cell_ids.extend(batch.cell_integer_ids)
    counts = Counter(all_cell_ids)
    assert set(counts.keys()) == set(range(n_cells)), (
        f"expected cells 0..{n_cells - 1}, got {sorted(counts.keys())}"
    )
    dup = {c: n for c, n in counts.items() if n != 1}
    assert not dup, f"cells not emitted exactly once: {dup}"


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


def test_mos_exhaustion_flushes_remaining_partial_cells(slaf_mos_boundary_reassembly):
    """When every fragment generator is exhausted, remaining partial rows must still be emitted.

    Repro: previously `combined_df` was built from `partial_cell_data` but execution fell
    through to the sampling path with no active generators, hit `continue`, and dropped
    the flushed rows (then raised StopIteration).
    """
    slaf = slaf_mos_boundary_reassembly
    processor = PrefetchBatchProcessor(
        slaf_array=slaf,
        window=ScGPTWindow(),
        shuffle=RandomShuffle(),
        tokenizer=None,
        raw_mode=True,
        use_mixture_of_scanners=True,
        n_scanners=2,
        prefetch_batch_size=1000,
        batch_size=8,
        seed=42,
        n_epochs=1,
        verbose=False,
    )

    # Full cell 1 (6 genes) so MoS CSI check treats rows as complete and emits them
    partial = pl.DataFrame(
        {
            "cell_integer_id": [1] * 6,
            "gene_integer_id": [0, 1, 2, 3, 4, 5],
            "value": [1.0] * 6,
        }
    )
    for i in range(len(processor.generator_active)):
        processor.generator_active[i] = False
    processor.partial_cell_data = {1: partial}

    batch = processor.load_prefetch_batch()
    all_dfs = []
    for df in batch.batch_dfs:
        all_dfs.append(df)
    out = pl.concat(all_dfs, how="vertical")
    assert out.height == 6
    assert set(out["cell_integer_id"].to_list()) == {1}
    assert set(out["gene_integer_id"].to_list()) == set(range(6))


@pytest.mark.parametrize("by_fragment", [True, False], ids=["fragment", "batch"])
def test_tiny_slaf_one_epoch_non_mos_raw_matches_cell_start_index(
    tiny_slaf, by_fragment: bool
):
    """Single-fragment SLAF: sequential fragment/batch loaders must match CSI for one epoch."""
    processor = PrefetchBatchProcessor(
        slaf_array=tiny_slaf,
        window=ScGPTWindow(),
        shuffle=RandomShuffle(),
        tokenizer=None,
        raw_mode=True,
        use_mixture_of_scanners=False,
        by_fragment=by_fragment,
        prefetch_batch_size=1000,
        batch_size=32,
        seed=42,
        n_epochs=1,
        verbose=False,
    )
    batches = _collect_raw_prefetch_epoch(processor)
    _assert_raw_epoch_matches_cell_start_index(tiny_slaf, batches)


@pytest.mark.parametrize("by_fragment", [True, False], ids=["fragment", "batch"])
def test_cross_fragment_slaf_one_epoch_non_mos_raw_matches_cell_start_index(
    slaf_mos_boundary_reassembly, by_fragment: bool
):
    """Multi-fragment SLAF with a cell spanning fragments: non-MoS must still match CSI."""
    processor = PrefetchBatchProcessor(
        slaf_array=slaf_mos_boundary_reassembly,
        window=ScGPTWindow(),
        shuffle=RandomShuffle(),
        tokenizer=None,
        raw_mode=True,
        use_mixture_of_scanners=False,
        by_fragment=by_fragment,
        prefetch_batch_size=1000,
        batch_size=8,
        seed=42,
        n_epochs=1,
        verbose=False,
    )
    batches = _collect_raw_prefetch_epoch(processor)
    _assert_raw_epoch_matches_cell_start_index(slaf_mos_boundary_reassembly, batches)


def test_tiny_slaf_one_epoch_mos_raw_matches_cell_start_index(tiny_slaf):
    """Single-fragment AnnData SLAF: one MoS raw epoch matches _cell_start_index."""
    processor = PrefetchBatchProcessor(
        slaf_array=tiny_slaf,
        window=ScGPTWindow(),
        shuffle=RandomShuffle(),
        tokenizer=None,
        raw_mode=True,
        use_mixture_of_scanners=True,
        by_fragment=True,
        n_scanners=4,
        prefetch_batch_size=1000,
        batch_size=32,
        seed=42,
        n_epochs=1,
        verbose=False,
    )
    batches = _collect_raw_prefetch_epoch(processor)
    _assert_raw_epoch_matches_cell_start_index(tiny_slaf, batches)


def test_cross_fragment_slaf_one_epoch_raw_matches_cell_start_index_mos(
    slaf_mos_boundary_reassembly,
):
    """Multi-fragment fixture with a cell spanning fragments: only MoS uses CSI reassembly.

    Sequential fragment/batch modes buffer the last cell per chunk and can drop rows when
    the same logical cell continues on a later Lance fragment.
    """
    processor = PrefetchBatchProcessor(
        slaf_array=slaf_mos_boundary_reassembly,
        window=ScGPTWindow(),
        shuffle=RandomShuffle(),
        tokenizer=None,
        raw_mode=True,
        use_mixture_of_scanners=True,
        by_fragment=True,
        n_scanners=4,
        prefetch_batch_size=1000,
        batch_size=8,
        seed=42,
        n_epochs=1,
        verbose=False,
    )
    batches = _collect_raw_prefetch_epoch(processor)
    _assert_raw_epoch_matches_cell_start_index(slaf_mos_boundary_reassembly, batches)
