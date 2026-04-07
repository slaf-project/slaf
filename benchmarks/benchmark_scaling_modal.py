#!/usr/bin/env python3
"""
Modal scaling benchmarks for SLAF cloud dataloaders (Tahoe train split on S3).

Two experiments (same split pattern: Modal **producer** workers + fixed **consumer** driver):
  - **Vertical**: always **1** producer; sweep **producer vCPUs** (`--vertical-cpus`). Driver uses a fixed
    8 vCPU / 32 GiB container (same as horizontal) for apples-to-apples queue throughput vs **n_workers=1** horizontal.
  - **Horizontal**: sweep **producer count** (`--horizontal-nodes`) at fixed `--cpu-per-worker`.

Both use the same `DistributedSLAFDataLoader` producer + `DistributedDataLoader` consumer. Use one
`--n-scanners-per-worker` for both (default **32**) so MoS width matches.

Dataset default: s3://slaf-datasets/Tahoe100M_train_SLAF

Prerequisites:
  - Modal secrets: `s3-credentials` with read access to the bucket.
  - Deploy producers before each sweep point with matching CPU/RAM, e.g.:
        python -c "from slaf.ml.distributed import deploy_dataloader_app; deploy_dataloader_app(cpu=8, memory=32768)"
    The script calls `deploy_dataloader_app` with the same `--cpu-per-worker` / `--memory-per-worker-mib` you pass.

Examples:
  modal run benchmarks/benchmark_scaling_modal.py --experiment both \\
      --vertical-cpus 4,8,16,32 --horizontal-nodes 1,4,16,32 \\
      --cpu-per-worker 8 --n-scanners-per-worker 32

  Default timing: 10s warmup + 60s measurement (override with ``--warmup-duration`` /
  ``--measurement-duration``).

Horizontal path matches `DistributedSLAFDataLoader` + queue consumer usage in fast-scgpt
`train_ddp.py` (spawn producers, wait for queue, consume via `DistributedDataLoader` + decompressing
wrapper — see https://github.com/slaf-project/fast-scgpt/blob/main/fast_scgpt/train_ddp.py ).
If your Modal app uses a named environment for queues (e.g. ``environment_name=\"main\"``), set
``SLAF_MODAL_QUEUE_ENVIRONMENT`` or ``--modal-queue-environment`` so workers and consumer attach to
the same queue. Redeploy ``slaf/ml/distributed.py`` after pulling queue-environment support.

The horizontal driver sets ``SLAF_MODAL_DEFER_DEFAULT_APP=1`` so importing ``slaf.ml.distributed`` inside
the Modal run does not register a second App (otherwise Modal retries ``distributed_prefetch_worker``
and fails to start). Training apps that construct ``DistributedSLAFDataLoader`` inside a Modal
function should set the same env before that import.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

from slaf.core.slaf import SLAFArray

# Do NOT import slaf.ml.distributed at module load: that module registers a second Modal App
# (distributed_prefetch_worker). `modal run` this file would then try to provision it even for
# --experiment vertical only. Lazy-import distributed pieces inside horizontal helpers.

# ---------------------------------------------------------------------------
# Modal image (align with slaf/ml/distributed.py: slafdb[ml] + build stamp)
# ---------------------------------------------------------------------------
_BUILD_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "python3-dev", "git")
    .pip_install("uv")
    .uv_pip_install("slafdb[ml]", "matplotlib", "psutil>=6.0.0")
    .run_commands(f"echo 'Image built at {_BUILD_TS}'")
)

APP_NAME = "slaf-scaling-benchmark"

app = modal.App(APP_NAME)

DEFAULT_S3_PATH = "s3://slaf-datasets/Tahoe100M_train_SLAF"
# Consumer Modal function (`run_vertical` / `run_horizontal`): fixed so vertical vs horizontal compare fairly.
DRIVER_CPU = 8
DRIVER_MEMORY_MIB = 32768
DEFAULT_N_SCANNERS_PER_WORKER = 32


@dataclass
class ScalingRow:
    """One measurement row for CSV/JSON and plotting."""

    experiment: str
    s3_path: str
    vcpus: float | None
    n_nodes: int | None
    cpu_per_node: float | None
    memory_per_node_mib: int | None
    throughput_cells_per_sec: float
    throughput_cells_per_sec_per_node: float | None
    total_cells: int
    measurement_time_s: float
    warmup_duration_s: float
    batch_size: int
    timestamp_utc: str


def _measure_throughput(
    dataloader,
    *,
    warmup_duration: float,
    measurement_duration: float,
    batch_size: int,
    label: str = "measure",
    progress_interval_s: float = 15.0,
) -> tuple[int, float, float]:
    """Return (cells in measurement window, measurement_time_s, elapsed_total_s).

    Counts cells only after ``warmup_duration``; throughput = total_cells / measurement_time.
    When ``progress_interval_s > 0``, prints a short progress line to stdout (Modal logs) on that interval
    during the measurement window.
    """
    start_time = time.time()
    total_cells = 0
    measurement_started = False
    measurement_start_wall: float | None = None
    last_progress_log = 0.0

    total_duration = warmup_duration + measurement_duration

    for batch in dataloader:
        batch_start = time.time()

        if isinstance(batch, dict):
            if "cell_ids" in batch:
                actual_batch_size = len(batch["cell_ids"])
            elif "X" in batch:
                actual_batch_size = batch["X"].shape[0]
            elif "batch_size" in batch:
                actual_batch_size = batch["batch_size"]
            else:
                actual_batch_size = batch_size
        elif hasattr(batch, "X"):
            actual_batch_size = batch.X.shape[0]
        else:
            actual_batch_size = batch_size

        _ = batch_start  # batch timing optional

        elapsed = time.time() - start_time

        if elapsed >= warmup_duration and not measurement_started:
            measurement_started = True
            measurement_start_wall = time.time()
            print(
                f"[benchmark-scaling] {label}: measurement window started "
                f"(warmup {warmup_duration}s done; timing {measurement_duration}s)",
                flush=True,
            )

        if measurement_started:
            total_cells += actual_batch_size
            if (
                progress_interval_s > 0
                and measurement_start_wall is not None
                and total_cells > 0
            ):
                now = time.time()
                if now - last_progress_log >= progress_interval_s:
                    mw = now - measurement_start_wall
                    rate = total_cells / mw if mw > 0 else 0.0
                    print(
                        f"[benchmark-scaling] {label}: {total_cells:,} cells in "
                        f"{mw:.1f}s → {rate:,.0f} cells/s (provisional)",
                        flush=True,
                    )
                    last_progress_log = now

        if elapsed >= total_duration:
            break

    elapsed_time = time.time() - start_time
    measurement_time = elapsed_time - warmup_duration
    if measurement_time <= 0:
        return 0, 0.0, elapsed_time
    return total_cells, measurement_time, elapsed_time


def _run_split_queue_benchmark(
    n_workers: int,
    cpu_per_worker: float,
    memory_per_worker_mib: int,
    s3_path: str,
    warmup_duration: float,
    measurement_duration: float,
    batch_size: int,
    n_scanners_per_worker: int,
    modal_queue_environment: str | None,
    *,
    mode: str,
) -> ScalingRow:
    """Producer workers (Modal) + queue + consumer in this process (train_ddp-style split)."""
    if mode not in ("vertical", "horizontal"):
        raise ValueError(f"mode must be vertical|horizontal, got {mode!r}")
    if mode == "vertical" and n_workers != 1:
        raise ValueError(
            "vertical mode is 1 producer only; use horizontal to sweep n_workers"
        )

    from slaf.distributed.dataloader import (
        DecompressingQueueWrapper,
        DistributedDataLoader,
    )
    from slaf.ml.distributed import DistributedSLAFDataLoader

    env = (
        modal_queue_environment
        or os.environ.get("SLAF_MODAL_QUEUE_ENVIRONMENT")
        or None
    )
    queue_name = f"slaf-scaling-{uuid.uuid4().hex}"

    slaf_array = SLAFArray(s3_path)
    producer = DistributedSLAFDataLoader(
        slaf_array=slaf_array,
        tokenizer_type="raw",
        batch_size=batch_size,
        max_genes=2000,
        vocab_size=65000,
        n_expression_bins=10,
        n_epochs=1000,
        raw_mode=True,
        queue_name=queue_name,
        modal_queue_environment=env,
        n_workers=n_workers,
        n_scanners=n_scanners_per_worker,
        prefetch_batch_size=262144,
        prefetch_batch_count=32,
        cpu=cpu_per_worker,
        memory=memory_per_worker_mib,
        return_tensors=False,
        prefetch_factor=2,
        queue_timeout=30.0,
    )

    if mode == "vertical":
        print(
            f"[benchmark-scaling] vertical: waiting for queue (1×{cpu_per_worker:g} vCPU producer, "
            f"n_scanners={n_scanners_per_worker}, queue={queue_name})",
            flush=True,
        )
    else:
        print(
            f"[benchmark-scaling] horizontal: waiting for queue "
            f"(n_workers={n_workers}, queue={queue_name})",
            flush=True,
        )
    producer.wait_for_queue(min_batches=50, timeout_seconds=300.0)

    q_kw: dict[str, Any] = {"create_if_missing": False}
    if env:
        q_kw["environment_name"] = env
    modal_queue = modal.Queue.from_name(queue_name, **q_kw)
    consumer_queue = DecompressingQueueWrapper(modal_queue)
    consumer = DistributedDataLoader(
        consumer_queue,
        batch_size=batch_size,
        return_tensors=False,
        prefetch_factor=16,
        queue_timeout=30.0,
        enable_diagnostics=False,
    )

    measure_label = (
        f"vertical producer_vcpus={cpu_per_worker}"
        if mode == "vertical"
        else f"horizontal n_workers={n_workers}"
    )
    print(
        f"[benchmark-scaling] {mode}: consuming from queue "
        f"(warmup {warmup_duration}s + measure {measurement_duration}s)",
        flush=True,
    )
    it = iter(consumer)
    try:
        total_cells, measurement_time, _ = _measure_throughput(
            it,
            warmup_duration=warmup_duration,
            measurement_duration=measurement_duration,
            batch_size=batch_size,
            label=measure_label,
        )
    finally:
        closer = getattr(it, "close", None)
        if callable(closer):
            closer()

    thr = total_cells / measurement_time if measurement_time > 0 else 0.0

    try:
        producer.stop_prefetch_workers()
    except Exception:
        pass

    del producer, consumer, slaf_array
    gc.collect()

    ts = datetime.utcnow().isoformat() + "Z"

    if mode == "vertical":
        print(
            f"[benchmark-scaling] vertical DONE producer={cpu_per_worker:g} vCPU: "
            f"{thr:,.0f} cells/s | {total_cells:,} cells in {measurement_time:.2f}s",
            flush=True,
        )
        return ScalingRow(
            experiment="vertical",
            s3_path=s3_path,
            vcpus=float(cpu_per_worker),
            n_nodes=None,
            cpu_per_node=None,
            memory_per_node_mib=None,
            throughput_cells_per_sec=thr,
            throughput_cells_per_sec_per_node=None,
            total_cells=total_cells,
            measurement_time_s=measurement_time,
            warmup_duration_s=warmup_duration,
            batch_size=batch_size,
            timestamp_utc=ts,
        )

    per_node = thr / n_workers if n_workers > 0 else None
    per_node_s = f"{per_node:,.0f}" if per_node is not None else "n/a"
    print(
        f"[benchmark-scaling] horizontal DONE n_workers={n_workers}: "
        f"{thr:,.0f} cells/s aggregate | {per_node_s} cells/s per worker | "
        f"{total_cells:,} cells in {measurement_time:.2f}s",
        flush=True,
    )
    return ScalingRow(
        experiment="horizontal",
        s3_path=s3_path,
        vcpus=None,
        n_nodes=n_workers,
        cpu_per_node=cpu_per_worker,
        memory_per_node_mib=memory_per_worker_mib,
        throughput_cells_per_sec=thr,
        throughput_cells_per_sec_per_node=per_node,
        total_cells=total_cells,
        measurement_time_s=measurement_time,
        warmup_duration_s=warmup_duration,
        batch_size=batch_size,
        timestamp_utc=ts,
    )


@app.function(
    image=image,
    cpu=DRIVER_CPU,
    memory=DRIVER_MEMORY_MIB,
    timeout=7200,
    secrets=[modal.Secret.from_name("s3-credentials")],
    region="us-east-1",
    name="vertical_sweep",
)
def run_vertical(
    producer_vcpus: float,
    memory_per_worker_mib: int = 32768,
    s3_path: str = DEFAULT_S3_PATH,
    warmup_duration: float = 10.0,
    measurement_duration: float = 60.0,
    batch_size: int = 64,
    n_scanners_per_worker: int = DEFAULT_N_SCANNERS_PER_WORKER,
    modal_queue_environment: str | None = None,
) -> dict:
    os.environ["SLAF_MODAL_DEFER_DEFAULT_APP"] = "1"
    if os.environ.get("AWS_ENDPOINT_URL"):
        os.environ.setdefault(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )

    row = _run_split_queue_benchmark(
        n_workers=1,
        cpu_per_worker=producer_vcpus,
        memory_per_worker_mib=memory_per_worker_mib,
        s3_path=s3_path,
        warmup_duration=warmup_duration,
        measurement_duration=measurement_duration,
        batch_size=batch_size,
        n_scanners_per_worker=n_scanners_per_worker,
        modal_queue_environment=modal_queue_environment,
        mode="vertical",
    )
    return asdict(row)


@app.function(
    image=image,
    cpu=DRIVER_CPU,
    memory=DRIVER_MEMORY_MIB,
    timeout=7200,
    secrets=[modal.Secret.from_name("s3-credentials")],
    region="us-east-1",
    name="horizontal_sweep",
)
def run_horizontal(
    n_workers: int,
    cpu_per_worker: float = 8.0,
    memory_per_worker_mib: int = 32768,
    s3_path: str = DEFAULT_S3_PATH,
    warmup_duration: float = 10.0,
    measurement_duration: float = 60.0,
    batch_size: int = 64,
    n_scanners_per_worker: int = DEFAULT_N_SCANNERS_PER_WORKER,
    modal_queue_environment: str | None = None,
) -> dict:
    # Must be set before slaf.ml.distributed is imported (pulls in a second Modal App).
    os.environ["SLAF_MODAL_DEFER_DEFAULT_APP"] = "1"
    if os.environ.get("AWS_ENDPOINT_URL"):
        os.environ.setdefault(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )

    row = _run_split_queue_benchmark(
        n_workers=n_workers,
        cpu_per_worker=cpu_per_worker,
        memory_per_worker_mib=memory_per_worker_mib,
        s3_path=s3_path,
        warmup_duration=warmup_duration,
        measurement_duration=measurement_duration,
        batch_size=batch_size,
        n_scanners_per_worker=n_scanners_per_worker,
        modal_queue_environment=modal_queue_environment,
        mode="horizontal",
    )
    return asdict(row)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _write_csv(rows: list[ScalingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_json(rows: list[ScalingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)


def _plot_vertical(rows: list[ScalingRow], out: Path) -> None:
    import matplotlib.pyplot as plt

    xs = [r.vcpus for r in rows if r.vcpus is not None]
    ys = [r.throughput_cells_per_sec for r in rows if r.vcpus is not None]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Producer vCPUs (1 worker; split consumer driver)")
    plt.ylabel("Throughput (cells / sec)")
    plt.title("Vertical scaling — Tahoe (S3), queue-split producers + consumer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_horizontal(rows: list[ScalingRow], out: Path) -> None:
    import matplotlib.pyplot as plt

    xs = [r.n_nodes for r in rows if r.n_nodes is not None]
    ys = [r.throughput_cells_per_sec for r in rows if r.n_nodes is not None]
    ys_per = [
        r.throughput_cells_per_sec_per_node for r in rows if r.n_nodes is not None
    ]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, ys, marker="o", color="C0", label="Aggregate throughput")
    ax1.set_xlabel("Number of producer nodes (n_workers)")
    ax1.set_ylabel("Aggregate throughput (cells / sec)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        xs, ys_per, marker="s", color="C1", linestyle="--", label="Per-node throughput"
    )
    ax2.set_ylabel("Per-node throughput (cells / sec)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    fig.suptitle("Horizontal scaling — DistributedSLAFDataLoader (S3)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_vertical_sweep_local(
    cpus: list[int],
    s3_path: str,
    warmup_duration: float,
    measurement_duration: float,
    batch_size: int,
    out_dir: Path,
    memory_per_worker_mib: int,
    n_scanners_per_worker: int,
    skip_plots: bool = False,
    modal_queue_environment: str | None = None,
) -> list[ScalingRow]:
    from slaf.ml.distributed import deploy_dataloader_app

    if cpus:
        print(
            f"[benchmark-scaling] Vertical: {len(cpus)} producer sizes up to {max(cpus)} vCPU; "
            f"redeploy workers before each tier; consumer driver {DRIVER_CPU} vCPU. "
            f"n_scanners={n_scanners_per_worker}",
            flush=True,
        )
    rows: list[ScalingRow] = []
    for c in cpus:
        deploy_dataloader_app(
            cpu=float(c),
            memory=memory_per_worker_mib,
            show_logs=False,
        )
        d = run_vertical.remote(
            producer_vcpus=float(c),
            memory_per_worker_mib=memory_per_worker_mib,
            s3_path=s3_path,
            warmup_duration=warmup_duration,
            measurement_duration=measurement_duration,
            batch_size=batch_size,
            n_scanners_per_worker=n_scanners_per_worker,
            modal_queue_environment=modal_queue_environment,
        )
        rows.append(ScalingRow(**d))
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    _write_csv(rows, out_dir / f"vertical_{stamp}.csv")
    _write_json(rows, out_dir / f"vertical_{stamp}.json")
    if not skip_plots:
        _plot_vertical(rows, out_dir / f"vertical_{stamp}.png")
    return rows


def run_horizontal_sweep_local(
    nodes: list[int],
    cpu_per_worker: float,
    memory_per_worker_mib: int,
    s3_path: str,
    warmup_duration: float,
    measurement_duration: float,
    batch_size: int,
    n_scanners_per_worker: int,
    out_dir: Path,
    skip_plots: bool = False,
    modal_queue_environment: str | None = None,
) -> list[ScalingRow]:
    from slaf.ml.distributed import deploy_dataloader_app

    deploy_dataloader_app(
        cpu=cpu_per_worker,
        memory=memory_per_worker_mib,
        show_logs=False,
    )

    max_procs = max(nodes)
    print(
        f"[benchmark-scaling] Horizontal: largest run uses {max_procs} producer workers "
        f"× {cpu_per_worker:g} vCPU each (= {max_procs * cpu_per_worker:g} vCPU) plus one "
        f"{DRIVER_CPU:g} vCPU driver — ensure Modal workspace quota.",
        flush=True,
    )

    rows: list[ScalingRow] = []
    for n in nodes:
        d = run_horizontal.remote(
            n_workers=n,
            cpu_per_worker=cpu_per_worker,
            memory_per_worker_mib=memory_per_worker_mib,
            s3_path=s3_path,
            warmup_duration=warmup_duration,
            measurement_duration=measurement_duration,
            batch_size=batch_size,
            n_scanners_per_worker=n_scanners_per_worker,
            modal_queue_environment=modal_queue_environment,
        )
        rows.append(ScalingRow(**d))
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    _write_csv(rows, out_dir / f"horizontal_{stamp}.csv")
    _write_json(rows, out_dir / f"horizontal_{stamp}.json")
    if not skip_plots:
        _plot_horizontal(rows, out_dir / f"horizontal_{stamp}.png")
    return rows


@app.local_entrypoint()
def main(
    experiment: str = "both",
    s3_path: str = DEFAULT_S3_PATH,
    vertical_cpus: str = "4,8,16,32",
    horizontal_nodes: str = "1,2,4,8,16,32,64",
    cpu_per_worker: float = 8.0,
    memory_per_worker_mib: int = 32768,
    warmup_duration: float = 10.0,
    measurement_duration: float = 60.0,
    batch_size: int = 64,
    n_scanners_per_worker: int = DEFAULT_N_SCANNERS_PER_WORKER,
    out_dir: str = "benchmarks/scaling_results",
    skip_plots: bool = False,
    modal_queue_environment: str | None = None,
):
    """
    Run scaling benchmarks on Modal and write CSV/JSON/PNGs under out_dir.

    experiment: vertical | horizontal | both
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    v_cpus = _parse_int_list(vertical_cpus)
    h_nodes = _parse_int_list(horizontal_nodes)

    if experiment in ("vertical", "both"):
        run_vertical_sweep_local(
            cpus=v_cpus,
            s3_path=s3_path,
            warmup_duration=warmup_duration,
            measurement_duration=measurement_duration,
            batch_size=batch_size,
            out_dir=out,
            memory_per_worker_mib=memory_per_worker_mib,
            n_scanners_per_worker=n_scanners_per_worker,
            skip_plots=skip_plots,
            modal_queue_environment=modal_queue_environment,
        )

    if experiment in ("horizontal", "both"):
        run_horizontal_sweep_local(
            nodes=h_nodes,
            cpu_per_worker=cpu_per_worker,
            memory_per_worker_mib=memory_per_worker_mib,
            s3_path=s3_path,
            warmup_duration=warmup_duration,
            measurement_duration=measurement_duration,
            batch_size=batch_size,
            n_scanners_per_worker=n_scanners_per_worker,
            out_dir=out,
            skip_plots=skip_plots,
            modal_queue_environment=modal_queue_environment,
        )

    print(f"Wrote results under {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAF Modal scaling benchmarks")
    parser.add_argument(
        "--experiment",
        choices=("vertical", "horizontal", "both"),
        default="both",
    )
    parser.add_argument("--s3-path", type=str, default=DEFAULT_S3_PATH)
    parser.add_argument("--vertical-cpus", type=str, default="4,8,16,32")
    parser.add_argument("--horizontal-nodes", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--cpu-per-worker", type=float, default=8.0)
    parser.add_argument("--memory-per-worker-mib", type=int, default=32768)
    parser.add_argument("--warmup-duration", type=float, default=10.0)
    parser.add_argument(
        "--measurement-duration",
        type=float,
        default=60.0,
        help="Timed window for throughput (after warmup); cells/sec = cells / this",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--n-scanners-per-worker",
        type=int,
        default=DEFAULT_N_SCANNERS_PER_WORKER,
        help=f"MoS scanners per producer worker (vertical + horizontal); default {DEFAULT_N_SCANNERS_PER_WORKER}",
    )
    parser.add_argument("--out-dir", type=str, default="benchmarks/scaling_results")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--modal-queue-environment",
        type=str,
        default="",
        help="Modal Queue environment name (e.g. main). Same as SLAF_MODAL_QUEUE_ENVIRONMENT.",
    )
    args = parser.parse_args()

    with app.run():
        main(
            experiment=args.experiment,
            s3_path=args.s3_path,
            vertical_cpus=args.vertical_cpus,
            horizontal_nodes=args.horizontal_nodes,
            cpu_per_worker=args.cpu_per_worker,
            memory_per_worker_mib=args.memory_per_worker_mib,
            warmup_duration=args.warmup_duration,
            measurement_duration=args.measurement_duration,
            batch_size=args.batch_size,
            n_scanners_per_worker=args.n_scanners_per_worker,
            out_dir=args.out_dir,
            skip_plots=args.skip_plots,
            modal_queue_environment=args.modal_queue_environment or None,
        )
