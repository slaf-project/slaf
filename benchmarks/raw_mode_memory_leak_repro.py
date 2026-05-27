"""SLAFDataLoader(raw_mode=True) host-memory leak reproducer.

The bug
=======

Before the fix, ``PrefetchBatchProcessor.__init__`` pre-created one
persistent ``fragment.to_batches(...)`` iterator per Lance fragment and held
the list alive on ``self.fragment_generators``. Each iterator allocates
~70 MiB of Lance-side scanner state on its first ``next()`` and keeps it for
its lifetime. Mixture-of-Scanners samples fresh fragments per load, so the
touched-iterator set grows unboundedly — for a 2000-fragment dataset the
worst case is ~140 GiB.

What this script does
=====================

Constructs a ``SLAFDataLoader(raw_mode=True)`` against a synthesised small
SLAF (or a user-provided dataset via ``--slaf-path``), paces the consumer
with ``--consumer-sleep-ms`` to mimic a GPU step, and samples RSS and queue
depth every two seconds. Before the fix this OOMs a 24 GiB cgroup in ~8 s on
Parse-10M scale; after the fix RSS plateaus.

Usage
=====

Quick local run (synthesises ~150 MiB SLAF in /tmp on first run)::

    uv run python benchmarks/raw_mode_memory_leak_repro.py

Against a real dataset, under a cgroup cap so a regression cannot crash the
host::

    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- \\
        uv run python benchmarks/raw_mode_memory_leak_repro.py \\
        --slaf-path /path/to/your.slaf
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import scanpy as sc
from loguru import logger
from scipy.sparse import random as sparse_random

from slaf import SLAFArray
from slaf.data import SLAFConverter
from slaf.ml.dataloaders import SLAFDataLoader

# Quiet the loguru-driven banner SLAFArray prints on every open.
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "WARNING"))


# Sized so expression.lance produces multiple Lance fragments (so n_scanners=16
# has parallel work) without taking too long to build. ~150 MB on disk.
SYNTH_N_CELLS = 500_000
SYNTH_N_GENES = 2000
SYNTH_DENSITY = 0.05  # ~50M nonzeros


def build_synthetic_slaf(out_path: Path) -> None:
    print(f"Building synthetic SLAF at {out_path} ({SYNTH_N_CELLS:,} cells)...")
    rng = np.random.default_rng(0)
    X = sparse_random(
        SYNTH_N_CELLS,
        SYNTH_N_GENES,
        density=SYNTH_DENSITY,
        format="csr",
        dtype=np.float32,
        random_state=rng,
        data_rvs=lambda n: rng.integers(1, 50, size=n).astype(np.float32),
    )
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(SYNTH_N_CELLS)]
    adata.var_names = [f"gene_{i}" for i in range(SYNTH_N_GENES)]
    h5ad = out_path.with_suffix(".h5ad")
    adata.write_h5ad(h5ad)
    SLAFConverter().convert(str(h5ad), str(out_path))
    h5ad.unlink(missing_ok=True)


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def measure(
    slaf_path: str,
    raw_mode: bool,
    measure_sec: float,
    consumer_sleep_ms: float,
    n_scanners: int = 16,
    prefetch_batch_size: int = 1_048_576,
) -> dict:
    label = "raw_mode=True" if raw_mode else "raw_mode=False (tokenized)"
    print(f"\n=== {label} (consumer pacing: {consumer_sleep_ms} ms/batch) ===")

    gc.collect()
    base_rss = rss_mib()
    print(f"baseline: RSS={base_rss:7.0f} MiB", flush=True)

    slaf = SLAFArray(slaf_path)
    _ = len(slaf.obs)

    # Same settings as benchmarks/benchmark_dataloaders_internal.py raw branch.
    dataloader = SLAFDataLoader(
        slaf_array=slaf,
        tokenizer_type="raw" if raw_mode else "geneformer",
        batch_size=32,
        max_genes=2048,
        vocab_size=50000,
        n_expression_bins=10,
        n_epochs=1000,
        raw_mode=raw_mode,
        verbose=False,
        use_mixture_of_scanners=True,
        by_fragment=False,
        batches_per_chunk=1,
        n_scanners=n_scanners,
        prefetch_batch_size=prefetch_batch_size,
        # Let SLAFDataLoader pick the per-mode default (1 for raw, 5000 for
        # tokenized). The reproducer used to force 5000 here to exhibit the
        # original leak — that's no longer the load-bearing signal.
    )

    # Reach into the prefetcher to observe queue depth — the queue depth is
    # the load-bearing signal here: it tells us how much Arrow data is pinned.
    prefetcher = dataloader._dataset.prefetcher
    queue = prefetcher.queue
    queue_max = prefetcher.max_queue_size

    # Brief warmup so the prefetcher has filled its initial reservoir.
    for i, _ in enumerate(dataloader):
        if i >= 5:
            break

    samples = []
    t0 = time.perf_counter()
    last_log = t0
    batch_count = 0
    sleep_sec = consumer_sleep_ms / 1000.0
    for _batch in dataloader:
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        batch_count += 1
        now = time.perf_counter()
        if now - last_log >= 2.0:
            samples.append((now - t0, rss_mib(), queue.qsize(), batch_count))
            print(
                f"  t={now - t0:5.1f}s  batches={batch_count:5d}  "
                f"queue={queue.qsize():5d}/{queue_max}  RSS={rss_mib():7.0f} MiB",
                flush=True,
            )
            last_log = now
        if now - t0 >= measure_sec:
            break

    end_rss = rss_mib()
    end_queue = queue.qsize()

    # SLAFDataLoader.__del__ does not stop the prefetcher (slaf/ml/dataloaders.py
    # line ~668 — just `pass`). Stop it explicitly so the queue's references
    # to RawPrefetchBatch payloads are dropped, then measure release.
    prefetcher.stop()
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break
    del dataloader
    del slaf
    del prefetcher
    del queue
    gc.collect()
    post_rss = rss_mib()
    print(
        f"after stop+drain+gc: RSS={post_rss:7.0f} MiB "
        f"(reclaimed {end_rss - post_rss:.0f} MiB of {end_rss - base_rss:.0f} MiB held)",
        flush=True,
    )

    return {
        "label": label,
        "samples": samples,
        "rss_growth_mib": end_rss - base_rss,
        "rss_after_cleanup_mib": post_rss - base_rss,
        "end_queue_depth": end_queue,
        "queue_max": queue_max,
        "duration_sec": measure_sec,
        "batches": batch_count,
    }


def summarise(result: dict) -> None:
    print(f"\n--- summary: {result['label']} ---")
    print(f"  duration:                 {result['duration_sec']:.1f}s")
    print(f"  batches iterated:         {result['batches']}")
    print(
        f"  end queue depth:          {result['end_queue_depth']}/{result['queue_max']}"
    )
    rss_growth = result["rss_growth_mib"]
    print(
        f"  RSS growth (in-flight):   {rss_growth:+.0f} MiB "
        f"({rss_growth / result['duration_sec']:+.1f} MiB/s)"
    )
    reclaimed = result["rss_growth_mib"] - result["rss_after_cleanup_mib"]
    print(
        f"  RSS reclaimed by del+gc:  {reclaimed:.0f} MiB "
        f"(net after cleanup: {result['rss_after_cleanup_mib']:+.0f} MiB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slaf-path",
        type=str,
        default=None,
        help="Existing SLAF dataset. If omitted, a small synthetic one is built.",
    )
    parser.add_argument(
        "--measure-sec",
        type=float,
        default=20.0,
        help="Per-mode iteration duration in seconds (default: 20).",
    )
    parser.add_argument(
        "--consumer-sleep-ms",
        type=float,
        default=50.0,
        help=(
            "Sleep added after each consumed batch, in ms, to mimic a GPU "
            "training step. The leak only manifests when the consumer is "
            "slower than the producer (default: 50ms; real training runs are "
            "around 175ms/step)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "tokenized", "both"),
        default="both",
        help="Which mode(s) to measure (default: both — contrast is the demo).",
    )
    parser.add_argument(
        "--n-scanners",
        type=int,
        default=16,
        help=(
            "Lance Mixture-of-Scanners parallelism (default: 16, matches "
            "benchmark_dataloaders_internal.py). Lower values reduce Lance's "
            "per-scanner readahead footprint independently of the queue-boundary "
            "fix; useful for isolating where memory is being held."
        ),
    )
    parser.add_argument(
        "--prefetch-batch-size",
        type=int,
        default=1_048_576,
        help="Lance per-scanner batch size in rows (default: 1,048,576).",
    )
    args = parser.parse_args()

    if args.slaf_path is None:
        path = Path("/tmp/slaf_leak_repro.slaf")
        if not path.exists():
            build_synthetic_slaf(path)
        slaf_path = str(path)
    else:
        slaf_path = args.slaf_path

    results = []
    if args.mode in ("raw", "both"):
        results.append(
            measure(
                slaf_path,
                raw_mode=True,
                measure_sec=args.measure_sec,
                consumer_sleep_ms=args.consumer_sleep_ms,
                n_scanners=args.n_scanners,
                prefetch_batch_size=args.prefetch_batch_size,
            )
        )
    if args.mode in ("tokenized", "both"):
        results.append(
            measure(
                slaf_path,
                raw_mode=False,
                measure_sec=args.measure_sec,
                consumer_sleep_ms=args.consumer_sleep_ms,
                n_scanners=args.n_scanners,
                prefetch_batch_size=args.prefetch_batch_size,
            )
        )

    print("\n" + "=" * 60)
    for r in results:
        summarise(r)


if __name__ == "__main__":
    main()
