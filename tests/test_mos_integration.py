"""Optional integration tests against a large real dataset (heart_10k).

CI coverage for cell-boundary / MoS correctness lives in
``tests/test_cell_boundary_reassembly.py`` (synthetic multi-fragment fixture).

These tests are skipped unless ``SLAF_HEART_10K_PATH`` points at a checkout of
``heart_10k.slaf``; they stress many intra-fragment reads with a small
``prefetch_batch_size``.
"""

import os
from collections import Counter

import polars as pl
import pytest

from slaf.core.slaf import SLAFArray
from slaf.ml.aggregators import ScGPTWindow
from slaf.ml.datasets import PrefetchBatchProcessor, RawPrefetchBatch
from slaf.ml.samplers import RandomShuffle

HEART_10K_PATH = os.environ.get(
    "SLAF_HEART_10K_PATH", os.path.expanduser("~/slaf-datasets/heart_10k.slaf")
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slaf_array,
    pytest.mark.skipif(
        not os.path.isdir(HEART_10K_PATH),
        reason=f"Real dataset not found at {HEART_10K_PATH}",
    ),
]


@pytest.fixture(scope="session")
def heart_slaf():
    """Open the heart_10k SLAF array and ensure metadata is populated."""
    arr = SLAFArray(HEART_10K_PATH)
    arr.wait_for_metadata()
    return arr


@pytest.fixture(scope="session")
def ground_truth_gene_counts(heart_slaf):
    """Build {cell_id: gene_count} from _cell_start_index."""
    csi = heart_slaf._cell_start_index
    counts = {}
    for i in range(len(csi) - 1):
        counts[i] = int(csi[i + 1]) - int(csi[i])
    return counts


@pytest.fixture(scope="session")
def all_mos_batches(heart_slaf):
    """Iterate one full epoch with MoS and small prefetch_batch_size, collecting all raw batches."""
    processor = PrefetchBatchProcessor(
        slaf_array=heart_slaf,
        window=ScGPTWindow(),
        shuffle=RandomShuffle(),
        tokenizer=None,
        raw_mode=True,
        use_mixture_of_scanners=True,
        n_scanners=4,
        prefetch_batch_size=100_000,
        batch_size=64,
        seed=42,
        n_epochs=1,
        verbose=False,
    )

    batches: list[RawPrefetchBatch] = []
    while True:
        try:
            batch = processor.load_prefetch_batch()
            batches.append(batch)
        except StopIteration:
            break

    return batches, processor


class TestMoSCellBoundaryReassembly:
    """Verify MoS boundary reassembly produces correct data against ground truth."""

    def test_cell_gene_counts_match_ground_truth(
        self, all_mos_batches, ground_truth_gene_counts
    ):
        """Every cell's observed gene count must match the ground truth from _cell_start_index."""
        batches, _ = all_mos_batches

        # Concat all batch DataFrames into one
        all_dfs = []
        for batch in batches:
            all_dfs.extend(batch.batch_dfs)
        combined = pl.concat(all_dfs, how="vertical")

        # Count genes per cell
        observed = (
            combined.group_by("cell_integer_id")
            .agg(pl.col("gene_integer_id").count().alias("gene_count"))
            .sort("cell_integer_id")
        )

        mismatches = []
        for row in observed.iter_rows(named=True):
            cell_id = row["cell_integer_id"]
            obs_count = row["gene_count"]
            expected = ground_truth_gene_counts.get(cell_id)
            if expected is None:
                mismatches.append(
                    f"cell {cell_id}: unexpected cell (not in ground truth)"
                )
            elif obs_count != expected:
                mismatches.append(
                    f"cell {cell_id}: observed {obs_count} genes, expected {expected}"
                )

        assert not mismatches, (
            f"{len(mismatches)} cells with wrong gene counts:\n"
            + "\n".join(mismatches[:20])
        )

    def test_no_duplicate_expression_rows(self, all_mos_batches):
        """No (cell_integer_id, gene_integer_id) pair should appear more than once."""
        batches, _ = all_mos_batches

        all_dfs = []
        for batch in batches:
            all_dfs.extend(batch.batch_dfs)
        combined = pl.concat(all_dfs, how="vertical")

        pairs = combined.select("cell_integer_id", "gene_integer_id")
        total = len(pairs)
        unique = len(pairs.unique())

        assert unique == total, (
            f"Found {total - unique} duplicate (cell, gene) rows out of {total} total"
        )

    def test_all_cells_emitted_exactly_once(self, all_mos_batches, heart_slaf):
        """Every cell index must appear exactly once across all batches (size from CSI)."""
        batches, _ = all_mos_batches
        heart_slaf.wait_for_metadata()
        n_cells = len(heart_slaf._cell_start_index) - 1

        # Collect cell IDs from each batch's metadata
        all_cell_ids = []
        for batch in batches:
            all_cell_ids.extend(batch.cell_integer_ids)

        cell_counts = Counter(all_cell_ids)
        expected_cells = set(range(n_cells))
        observed_cells = set(cell_counts.keys())

        missing = expected_cells - observed_cells
        extra = observed_cells - expected_cells
        duplicated = {c: n for c, n in cell_counts.items() if n > 1}

        errors = []
        if missing:
            errors.append(f"Missing {len(missing)} cells: {sorted(missing)[:20]}")
        if extra:
            errors.append(f"Extra {len(extra)} unexpected cells: {sorted(extra)[:20]}")
        if duplicated:
            dup_sample = dict(sorted(duplicated.items())[:20])
            errors.append(f"Duplicated {len(duplicated)} cells: {dup_sample}")

        assert not errors, "\n".join(errors)

    def test_partial_cell_buffer_was_exercised(self, all_mos_batches):
        """With 100K prefetch on ~500K-row fragments, we expect many reads proving boundary logic ran."""
        batches, _ = all_mos_batches

        # 39 fragments with ~500K rows each at 100K prefetch = ~195 reads minimum
        # We should see substantially more than 39 batches
        assert len(batches) > 39, (
            f"Only {len(batches)} batches produced — expected >39 to confirm "
            f"multiple reads per fragment exercised the boundary logic"
        )
