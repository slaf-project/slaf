# Pretraining scGPT on Tahoe-100M for ~$10 over a lunch break

## How distributed dataloading changes the economics of single-cell foundation models

*This is the third post in the Blazing Fast Dataloaders series.
[Post 1](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders/)
covered throughput: thread-based prefetch, vectorized outer loop, and contiguous
reads — achieving 28k cells/sec on local disk.
[Post 2](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders-2/)
covered randomness: the Mixture of Scanners architecture for near-perfect
shuffling without drop in throughput. This post covers scale: what changes when training
data lives in object storage, why vertical scaling hits a wall, and how
separating ingestion from training makes the economics of pretraining look very
different.*

---

## Dataloaders: the unglamorous work that makes GPUs useful

Training a foundation model spends most of its wall-clock time on three things:
matrix operations that compute gradients, parameter updates that apply them, and
everything that has to happen before any of that can start. The third category is
the dataloader's job.

A dataloader coordinates the pipeline from raw storage to training-ready tensors.
It fetches data asynchronously so the GPU never sits idle waiting for the next
batch. It handles filtering, normalization, and tokenization. It shuffles samples
so the model doesn't develop spurious biases from the order data was
collected or stored. It packs individual samples into the batch shapes the training regime
expects. None of this is computationally interesting in the way that attention
mechanisms or optimizer algorithms are — but leave it unoptimized, and you end
up with expensive GPU hardware waiting on a CPU bottleneck for most of its
working life.

A few design patterns have emerged over the years for keeping GPUs fed.

**Async prefetch with threads.** The canonical PyTorch approach spawns one worker
process per prefetch thread. Each process loads and preprocesses samples
independently, shipping results back to the main process for batching. This works
but carries costs: spawning processes duplicates memory, pickles tensors across
process boundaries, and imposes coordination overhead that scales poorly. A
thread-based alternative, explored in recent work by
[NVIDIA](https://developer.nvidia.com/blog/improved-data-loading-with-threads/),
[Meta](https://ai.meta.com/blog/spdl-faster-ai-model-training-with-thread-based-data-loading-reality-labs/),
and [Ray](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html),
uses concurrent threads instead of processes. The GIL is not an obstacle for
I/O-bound and numerical work because Lance, Arrow, NumPy, and Polars all release
it during their hot paths. Threads share memory with the training process, avoid
pickling, and start in milliseconds rather than seconds. SLAF adopted this design
from the beginning. ([Post 1](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders/))

**Moving preprocessing into the outer loop.** Dataloaders typically hand raw
samples to the training loop, which applies tokenization and preprocessing
per-step. There's no technical reason for this division — it's mostly convention,
since ML researchers want control over what happens to their data before it
reaches the model. But preprocessing is embarrassingly parallel, and doing it on
large prefetch batches rather than per-sample exploits the sub-linear scaling of
vectorized operations: the per-cell cost of gene ranking and tokenization in
Polars drops by roughly 3x when processing 1024 cells at once versus 32. Moving
these operations into the outer loop, at prefetch batch granularity, is a
straightforward win. ([Post 1](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders/))

**Randomization without pre-shuffling.** Most teams avoid the complexity of
on-the-fly shuffling by creating pre-shuffled, training-specific copies of their
datasets and staging them to cluster-attached storage. This is expensive in
storage terms and inflexible: each new training configuration requires a new
copy. The alternative — randomizing during streaming without a throughput penalty
— lacks a general-purpose solution. SLAF's Mixture of Scanners solves this for
Lance-backed data by keeping one sequential `to_batches` iterator per fragment,
randomly sampling which iterators to advance each step, then applying a
block-level shuffle within the prefetch batch.
The pipeline reaches 88–90% of theoretical maximum entropy at 97% of sequential
throughput. The dataset stays untouched in object storage; randomization happens
in the delivery mechanism. ([Post 2](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders-2/))

**Distributing ingestion independently of training.** With the first three
patterns in place, SLAF's dataloader reaches 28,000 cells/sec from local disk.
What happens when the data lives in object storage? How do we build an ingestion
architecture that scales horizontally without changing the training code? That is
the subject of this post.

---

## The case for object-storage-native training

Keeping training data in object storage and streaming it to ephemeral compute is
becoming the default for serious ML infrastructure, not a workaround. Two recent
benchmarks make the argument concretely.

[In December 2025](https://aws.amazon.com/blogs/machine-learning/applying-data-loading-best-practices-for-ml-training-with-amazon-s3-clients/),
AWS published results from benchmarking ML training throughput directly from S3
using the S3 Connector for PyTorch and Mountpoint for S3. Their core finding:
the bottleneck in object storage workloads is per-request latency,
not bandwidth. Each S3 GET request incurs a time-to-first-byte overhead that is
largely independent of object size — connection setup, round trips, and
service-side processing all accumulate before any bytes transfer. Datasets stored
as many small objects (one file per sample) are latency-bound: workers spend most
of their time blocked on that overhead rather than transferring data. Their
practical recommendation is to consolidate data into larger shards in the 100 MB
to 1 GB range, read them sequentially, and use high-performance clients built on
the AWS CRT. With that stack, a single GPU can be kept fully saturated from S3.

Tigris extended the same benchmark against S3-compatible storage
[in March 2026](https://www.tigrisdata.com/blog/training-object-storage/),
with one addition: the Tigris Acceleration Gateway (TAG), a local NVMe-backed
caching layer running on the training instance. Their entitlement measurement —
throughput with the GPU replaced by a no-op to measure the raw pipeline ceiling —
showed the data pipeline capable of delivering samples at 46x the rate a GPU can
consume them when correctly configured, rising to ~200x with TAG's warm cache.
Their multi-epoch results showed warm-cache epochs completing 5.7x faster,
cutting the number of dataloader workers needed to saturate a GPU from 16 to 4.

SLAF's data loader is different from loading JPEG images in one important way:
a single cell is not a single record-linked file. It is many sparse `(cell_id, gene_id, value)`
triples stored as consecutive rows in a Lance table. Those rows can straddle
fragment boundaries. Distributed ingestion requires handling that case explicitly.
The access pattern principles, though — sequential reads inside shards, enough
read parallelism, amortized per-request overhead — apply directly.

---

## From object storage to GPUs via horizontally scaled dataloaders

Running the Mixture of Scanners pipeline on Tahoe-100M (train split, raw mode,
batch size 64) stored in a Tigris bucket on a single Modal node shows a familiar pattern.

```
Throughput (cells/sec), single Modal node
─────────────────────────────────────────────────────────────────
4 vCPUs                  ████████████████████ 2,169
8 vCPUs                  ██████████████████████████████████ 3,572
16 vCPUs                 ██████████████████████████████ 3,126
32 vCPUs                 █████████████████████████████████ 3,459
─────────────────────────────────────────────────────────────────
                         0      900   1,800  2,700  3,600
```

Throughput roughly doubles from 4 to 8 vCPUs and then levels off near 3,500
cells/sec. 16 and 32 vCPUs perform no better than 8. This is not a compute
ceiling: 32 vCPUs have plenty of capacity to run the Polars windowing and
tokenization pipeline faster. The constraint is the network path from the node
to the bucket: one machine, one connection pool, one point through which all
S3 reads flow regardless of how many threads are issuing them.

This is where the traditional GPU training cluster model creates an awkward
constraint. Training hardware is sold as bundled nodes: a certain number of GPUs
paired with a fixed allocation of CPUs, RAM, and network capacity. That bundle
was sized for local-storage training, where the CPU had to keep pace with fast
NVMe. In cloud-native training from object storage, the CPU ceiling that actually
matters is the aggregate read parallelism across the network to the bucket — and
that has nothing to do with how many vCPUs a GPU node happens to include.

Horizontal scaling addresses this differently. Each additional worker is its own
client issuing its own S3 requests from its own container. Adding workers doesn't
add more threads to the same network path; it adds independent paths. The
aggregate throughput ceiling is set by what the object store can serve, which is
far higher than any single node can generate.

```
Aggregate throughput (cells/sec), producer fleet
─────────────────────────────────────────────────────────────────
1 producer node          █ 1,012
4 producer nodes         █████████ 8,575
16 producer nodes        ███████████████████████████████ 29,526
32 producer nodes        ████████████████████████████████ 30,581
─────────────────────────────────────────────────────────────────
                         0     8k    16k    24k    32k
```

At 16 horizontally scaled workers, aggregate throughput reaches roughly 8x what a single optimized
node delivers from cloud storage.

Horizontal scaling for data loading is not without precedent. Alibaba's
[GoldMiner, SIGMOD 2023](https://dl.acm.org/doi/10.1145/3589773)
identifies the point at which faster GPUs push preprocessing into
the critical path, and demonstrates that elastically scaling the preprocessing
fleet independently of the training fleet is the correct structural response.
[TensorSocket, ArXiv 2025](https://arxiv.org/abs/2409.18749) makes a related
point for multi-experiment workloads: a single producer serving batches over a socket
to multiple consumers eliminates the redundant reads that would otherwise occur
when each training process replays the same I/O independently. Ray Data's
[Streaming Batch Model, ArXiv 2025](https://arxiv.org/abs/2501.12407) arrives at
the same CPU/GPU decoupling from the angle of heterogeneous cluster scheduling.
Each of these systems arrives at the same structural conclusion: the preprocessing
fleet and the training fleet should scale on separate axes, coordinated by a queue.

---

## `DistributedSLAFDataLoader`

`DistributedSLAFDataLoader` is the realization of these ideas in SLAF.
The key design insight is that the outer loop and the inner loop have entirely
different resource profiles, and forcing them onto the same machine makes both
worse.

The outer loop — reading from object storage, ranking genes, tokenizing,
shuffling — is CPU-bound, memory-bandwidth-bound, and network I/O-bound. It
benefits from cheap horizontal scale and has no use for a GPU.

The inner loop — forward pass, backward pass, optimizer step, gradient
communication — is GPU-bound. It benefits from a steady supply of ready batches
and should spend as little time as possible waiting.

Separating them behind a queue allows each to scale independently. CPU workers
produce tokenized samples into a distributed FIFO; GPU workers consume and train.
Adding CPU workers doesn't touch the training code. Changing batch size doesn't
touch the ingestion workers. The two fleets evolve independently.

`DistributedSLAFDataLoader` implements this with three components.

### CPU worker fleet

Each worker is a stateless function running the complete Mixture of Scanners
pipeline against object storage: select a random Lance fragment, read a
contiguous block, apply Polars window functions for gene ranking, block-shuffle
within the prefetch batch, and tokenize. Workers have no shared state and no
awareness of each other. Any worker can contribute samples to any GPU rank.
Scaling the fleet means spawning more of the same function; the coordination
logic doesn't change.

### `modal.Queue` and `modal.Dict`

A named distributed FIFO queue sits between producers and consumers. Each queue
item is one fully processed, compressed sample. The GPU consumer reads
`batch_size` items per step and assembles the batch locally.

Compression is necessary: a tokenized cell at `max_genes=2048` occupies tens to
low hundreds of kilobytes uncompressed, and the queue caps individual items at
1 MiB. Compressing before enqueue and decompressing after keeps items well within
that limit.

A shared dictionary handles a problem that arises specifically from SLAF's
columnar layout. A single cell spans many rows — its expression values are stored
as `(cell_id, gene_id, value)` triples across consecutive Lance records. Those
records can straddle fragment boundaries, meaning two different workers may each
hold part of the same cell. Neither can produce a complete training sample alone.

When a worker finishes reading a fragment and finds an incomplete cell at the
boundary, it writes the partial record to the dictionary keyed by cell ID. The
next worker, reading the adjacent fragment, checks for pending partials,
completes the cell, and enqueues it. The dictionary entry is deleted. Every cell
is enqueued exactly once. This coordination problem is invisible in pipelines
where one file equals one sample, but it's structural in sparse columnar formats
where a sample spans many rows.

The queue and dictionary together solve the two hard problems of distributed
ingestion for SLAF data: transport and boundary assembly.

### GPU consumer

The training process sees a thin iterator: pull `batch_size` items from the
queue, decompress, stack, move to device. The training container has no
dependency on Lance, Polars, or SLAF's dataframe stack. It knows how to pop
items from a queue.

---

## Deployment

The full pipeline runs as a single Modal app.

CPU workers are `@modal.function(cpu=N)` with configurable concurrency. Because
they're stateless, scaling from a few workers to dozens is a parameter change.
Modal handles container restarts and retries, so worker-level fault tolerance is
delegated to the orchestrator rather than implemented inside the dataloader code.

The queue and dictionary are named app objects visible to every function in the
app — same credentials, same region. They're scoped to the run and cleaned up
when it completes.

GPU training workers are `@modal.function(gpu="H100")` running standard PyTorch
DDP. They contain training code only.

The cost structure follows from the separation. CPU workers run on commodity
compute at a fraction of H100 rates. H100 time is spent on training steps, not
on data movement. If the queue depth runs low, you add CPU workers. If you need
more gradient throughput, you add GPU workers. Each fleet has its own cost
knob.

---

## Results: `fast-scgpt` on Tahoe-100M

[`fast-scgpt`](https://github.com/slaf-project/fast-scgpt) is a reference
implementation in Modal and trains a scGPT-style transformer on Tahoe-100M,
streamed from object storage or [Hugging Face](https://huggingface.co/datasets/slaf-project/Tahoe-100M).
Model weights and downstream evaluation are not part of this release, stay tuned.

| Config | Median step latency | Global cells/sec | Dataloader wait | MFU |
|---|---|---|---|---|
| 1x H100 (`modal_train.py`, `scgpt` ~51M params) | ~285 ms | ~840 | ~0 ms | ~37% |
| 8x H100 DDP (`modal_train_distributed.py`, same model) | ~60 ms | ~31.9k | ~0 ms | ~32% |

Numbers follow the current [`fast-scgpt` README](https://github.com/slaf-project/fast-scgpt):
Flash Attention 4, **no** `torch.compile` (`--no-use-compile`), sparse gene head,
**1024** max genes, **240** cells per step on 1× GPU and **240 per GPU** on 8×
(effective global batch **1920**), Tahoe-100M from **S3** through the distributed
dataloader with **two** CPU prefetch workers on the 8-GPU run. Median step time
on 8 GPUs is a **max over ranks** per step (slowest GPU), then median across
steps—see the README for MFU / `nvidia-smi` definitions.

In single-GPU setting, we use `SLAFDataLoader`.
Both data loading and training share the same node. Median step latency is about **285 ms**, and time
blocked on `next(batch)` is approximately **0 ms** in steady state (the README’s
profiled breakdown labels that phase **`dl`**). The
dataloader has been removed from the critical path even when CPU and GPU share
the same node. That's a consequence of posts 1 and 2: thread-based prefetch with
vectorized outer loop tokenization and Mixture of Scanners keeps the queue full
faster than the GPU can drain it.

The 8x H100 result, based on `DistributedSLAFDataLoader`, tells a different story.
Median step latency drops to about **60 ms** per step at the same per-GPU batch (**worst rank** per step, then median across steps)
with no change to training code. At **240** cells per GPU the README still reports **sub-100 ms** steps.

The more important difference between these two configurations is what the CPU
is doing during the training step. On the single H100, the CPU is running the
full SLAF ingestion pipeline on the same host as the GPU, so **CPU cores, host
DRAM bandwidth, and the PCIe link to the device** are shared between
preprocessing and staging batches for the accelerator. On the 8x H100
configuration, each GPU container's CPU has one job: pop items from the queue, decompress
and call `.to(device)`. The ingestion work happens entirely on the CPU worker
fleet, on separate machines. That single-responsibility separation is why the
per-GPU step latency is lower and it's the concrete payoff of the architectural decoupling.

At a global batch size of **1,920** cells (240 × 8 GPUs) and **~60 ms** per step,
*training* throughput—not just dataloading throughput—is
roughly **31.9k cells/sec** including forward pass, backward pass, optimizer step,
and **NCCL gradient synchronization** (in standard DDP this is mostly an
**all-reduce** of gradients across ranks).

---

## The $10 pretraining run

To put these numbers in context, consider the reference point from the original
scGPT pretraining discussion. In a [2023 GitHub thread](https://github.com/bowang-lab/scGPT/issues/5)
on pretraining cost, the authors described training on roughly 10.3M cells across six epochs — about 62M
cell presentations through the training loop — taking approximately three to four
days on four A100s, with data staged to cluster-attached storage.

Running 62M cell presentations through the 8x H100 configuration at **~31.9k**
cells/sec (README steady-state global throughput):

62,000,000 / 31,900 ≈ 1,940 seconds — about **32 minutes** of GPU wall time in
steady state.

Billed GPU time: 8 GPUs × (1,940s / 3,600 s/hr) ≈ **4.3 GPU-hours**. At $2.50 per
H100 per hour, that's roughly **$11 in GPU line items**. CPU prefetch workers
round to noise next to eight H100s. Separately, the object store bill for
repeated streaming matters: [Tigris does not charge egress](https://www.tigrisdata.com/pricing)
(data transfer out is free on Standard and several other tiers at time of writing),
so iterative training and many short experiments do not accumulate a large
egress line item next to GPU time the way they often do on hyperscaler object
storage.

The comparison isn't exact — different silicon, different year, different
dollars-per-GPU-hour. The A100s that took four days in 2023 are not H100s in
2026. But the order-of-magnitude
shift in both wall-clock time and cost is real, and it comes from three sources:
better hardware, a storage format that's streamable from object store,
and a dataloader that doesn't make the hardware wait.

Of the three, the hardware contribution is well-understood. The dataloader
contribution is less appreciated. An H100 sitting idle at 0% GPU utilization
while waiting for the next batch costs exactly the same per hour as one running
at 100% GPU utilization. The goal of everything in this series has been to make the latter
the steady state.

What changes when a training run costs on the order of **$10** in GPU time instead of several thousand: teams
can treat pretraining as iterative rather than monolithic. The question of
whether six epochs generalizes better than three becomes an afternoon experiment.
Fine-tuning feels like working with scikit-learn in a notebook.
A new architecture variant doesn't require committing to a multi-day cluster
reservation. When a new atlas drop arrives, the data source pointer changes;
nothing is staged or copied. The Atlas stays in object storage; the ingestion
fleet meets it there.

---

## Getting started

If you want to **run the numbers yourself**, start from the reference Modal
harness [`fast-scgpt`](https://github.com/slaf-project/fast-scgpt): clone the
repo, follow the README for environment and `modal` setup, and launch training
against [Tahoe-100M on Hugging Face](https://huggingface.co/datasets/slaf-project/Tahoe-100M)
using the `hf://datasets/slaf-project/Tahoe-100M` URI so cells stream with no
local staging step.

If you are **building your own model**, install SLAF with ML extras
(`pip install 'slafdb[ml]'`), open the dataset with `SLAFArray` and the same
`hf://` path, and wire in `SLAFDataLoader` or `DistributedSLAFDataLoader` as in
the harness. The [SLAF documentation](https://slaf-project.github.io/slaf/) has
the API surface; reach out on
GitHub issues for [SLAF](https://github.com/slaf-project/slaf/issues) or
[`fast-scgpt`](https://github.com/slaf-project/fast-scgpt/issues) or [Discord](https://discord.com/invite/7Q95RVhURe).
