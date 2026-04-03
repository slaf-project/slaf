# Blazing Fast Dataloaders #3: Object storage and distributed ingestion

## Distributed dataloading for single-cell foundation models

_This is the third post in the Blazing Fast Dataloaders series. [Post 1](../blazing-fast-dataloaders.md) covered throughput: thread-based prefetch, vectorized outer loop, and contiguous reads. [Post 2](../blazing-fast-dataloaders-2.md) covered randomness: the Mixture of Scanners architecture for near-perfect shuffling at streaming speed. This post covers scale: training from object storage, when the bottleneck shifts to the network, and how we split CPU-side ingestion from GPU training with a queue-backed distributed dataloader._

---

The first two posts describe a single-node pipeline that reaches roughly **28,207 cells per second** from **local disk**—enough to keep one H100 fed at scGPT-scale batch sizes. The stack is still useful as-is for workstations and staged data.

Atlas-scale training in the cloud is different: the canonical copy of the data sits in **object storage**, and compute is **ephemeral**. Pointing the same Lance-backed pipeline at **S3** (or any similar endpoint) does not “break” the dataloader; throughput remains **respectable in absolute terms**, but the **bottleneck moves**. On a single node, adding cores and prefetch threads eventually stops helping because **egress bandwidth and request economics cap how much data can reach that box**, regardless of how efficiently the CPU would transform it afterward. In other words, once the link to storage is the limiter, vertical scaling on the training host is the wrong knob.

This post describes how we **separate concerns**—CPU workers that stream and preprocess in parallel, a narrow-waist queue, and GPU workers that mostly train—implemented today as **`DistributedSLAFDataLoader`** on Modal.

---

## The story so far

Posts 1 and 2 built a single-node dataloader that squeezes everything it can from one machine:

- **Thread-based prefetch** uses GIL-releasing work in Lance, Arrow, and NumPy so prefetch threads overlap I/O and preprocessing without paying for multiprocessing copies.
- The **vectorized outer loop** moves tokenization out of the per-step hot path and applies it across large prefetch batches, riding sub-linear Polars window-function costs (roughly **3× better per-cell efficiency** at 1024 cells versus 32 in our scaling tables).
- **Mixture of Scanners (MoS)** keeps contiguous reads while approximating a global shuffle: random subsets of scanners read sequential runs from different offsets, then an in-memory block shuffle mixes deliveries—**88–90%** of theoretical maximum entropy at **97%** of sequential throughput.

Those ideas share a hidden assumption: **the CPU that prefetches and the GPU that trains share one box**. They compete for the same cores, memory bandwidth, and NUMA domains. On a workstation with fast local disk, that competition is usually secondary to “can we read and transform fast enough?” In the cloud, with **Tahoe-100M on S3** and multi-GPU jobs, **network delivery** often joins (or replaces) that story as the hard ceiling on a single ingest host.

> 💡 **Figure (placeholder).** Single-node architecture: prefetch threads and training loop on one host. Annotate shared CPU, memory bandwidth, and the moment the object-storage link becomes the bottleneck.

---

## The cloud storage wall

The local-disk headline—**28k cells/sec**—is real, but it is not how atlas-scale training runs in production. Data lives in **object storage**; compute is **ephemeral**. Moving the Lance dataset behind S3 (or any similar endpoint) keeps the pipeline recognizable: you still stream, shuffle, and tokenize—but **peak single-node throughput drops** because bytes and metadata round-trips now cross the **network path to the bucket**, not a local NVMe link.

On our hardware, even with aggressive thread counts and core use, cloud-streamed throughput on **one** node tended to **plateau around ~2–3k cells/sec**. That is still a non-trivial rate, but it sits **below** the roughly **5–6k cells/sec** we target to saturate one H100 at scGPT-style settings. Pushing further on **that same node**—more prefetch threads, more parallelism in the transform stack—yielded **diminishing returns**: the pipe from object storage was doing what a pipe does. Whether the last mile on the box was dominated by Lance read threads or Polars time was secondary; **the link budget had become the gating factor**. At that point the interesting question is not “optimize another 10% on CPU” but **add another path for data to enter the cluster**—i.e. scale ingestion **horizontally**.

The scaling story splits into two regimes:

| Scaling strategy | Throughput | Ceiling |
| ---------------- | ---------- | ------- |
| Vertical (more threads, single node) | ~[X] cells/sec | Hard ceiling: network / request asymptote |
| Horizontal (more nodes, distributed) | ~[Y] cells/sec per node | Stays linear up to [Z] nodes in our tests |

_Fill in X, Y, Z when benchmarks are finalized. Scaffold note: up to 64 CPU worker nodes tested; on the order of ~64k cells/sec aggregate observed at 64 nodes._

The practical lesson: **vertical scaling on the training host has an asymptote once storage I/O is the cap; horizontal scaling on the ingestion side does not** (within the limits of your object store and wallet). If GPUs keep getting faster and atlases keep growing, the durable fix is to **add dataloading capacity** the same way you add training capacity: as its **own** fleet, not only as a side effect of buying a bigger GPU box.

> 💡 **Figure (placeholder).** Two curves: vertical throughput flattening versus near-linear horizontal scaling. Mark the crossover where distributed ingestion wins.

---

## The new default: training streams from object storage

None of this is unique to genomics. Over the last few years, “**stream training data from object storage**” crossed from brave to **boring-in-a-good-way** for ML training in the cloud. Once you accept that default, two engineering questions move from the appendix to chapter one: **what pattern of reads** you impose on storage, and **which client** you use to issue those reads.

Amazon’s machine-learning team made the read-pattern argument crisply in [_Applying data loading best practices for ML training with Amazon S3 clients_](https://aws.amazon.com/blogs/machine-learning/applying-data-loading-best-practices-for-ml-training-with-amazon-s3-clients/) (December 2025). The core observation is familiar to anyone who has profiled S3: each `GET` pays a largely **size-independent time-to-first-byte (TTFB)**—connection setup, round trips, and service-side work add overhead before bytes flow. If **every training sample is its own object**, you spend your life in TTFB; workers wait on latency, not on bandwidth. That is a **latency-bound** regime. Their remedy is structural: **fewer, larger shards** (they highlight roughly **100 MB–1 GB** as a happy band in their CV benchmark), **sequential reads inside a shard**, and enough **parallelism and prefetch** to keep the pipe full. They also emphasize **high-quality clients** built on the AWS CRT—**Mountpoint for S3** and the **S3 Connector for PyTorch**—plus **caching** when epochs repeat (Mountpoint’s cache, long metadata TTL) so you stop re-paying network tolls for the same bytes.

Important caveat they echo: **sharding alone does not guarantee sequential access**. If your format encourages random jumps inside a large object—think Parquet-style seeks or HDF5-style hyperslabs—you can erase the benefit of a big shard. The dataloader and the on-disk layout have to tell the **same story**.

Tigris ran the same **shape** of benchmark against **S3-compatible** storage and wrote it up in [_Benchmarking ML Training Throughput on Tigris Object Storage_](https://www.tigrisdata.com/blog/training-object-storage/) (March 2026). Qualitatively, the pictures match AWS: **random, one-object-per-image** workloads need **many** dataloader workers before the GPU saturates; **tar-sharded sequential** workloads saturate with **fewer** workers because TTFB is amortized across many samples inside one stream. Their extension is **Tigris Acceleration Gateway (TAG)**: an **S3-compatible cache** on the training instance. Epoch one warms NVMe; later epochs look like local reads to the trainer. For **multi-epoch** jobs, that is a big deal—you stop treating “epoch two” as a full replay of the same remote read pattern.

The punch line in both posts is not the ViT numbers themselves; it is the **entitlement** exercise. Strip the model down to a no-op forward pass and measure how fast the pipeline can feed samples. The answer is **orders of magnitude** higher than a GPU-bound training step when the stack is configured sanely. That headroom is “useless” only if you insist on **colocating** preprocessing with the GPU. The moment you **decouple** producers and consumers, that slack becomes **breathing room**—for jitter, for stragglers, for bigger batches later.

!!! info "Same physics, different samples"

    Single-cell transcriptomics is not JPEG classification, but the object-storage physics is the same: TTFB, sequential versus random, shard sizing, worker counts, and whether repeated epochs re-hit the network. SLAF’s twist is semantic: a **cell** is not one file. It is **many sparse `(cell, gene, value)` rows** in Lance, and those rows can **span fragment boundaries**. Distributed producers therefore need a coordination story: otherwise two honest workers each hold half a cell and neither can ship a training example.

> 💡 **Figure (placeholder).** Side-by-side: “many tiny objects + random GETs” versus “larger shards + sequential scan + optional local cache.” Label latency-bound versus bandwidth-bound regions conceptually.

---

## Queues, pub/sub, and splitting CPU from GPU

So why not just turn `num_workers` to eleven and buy a bigger NIC?

Because the **outer loop** (read, rank genes, tokenize, shuffle) and the **inner loop** (forward, backward, optimizer) are not the same job. The outer loop is **CPU-, bandwidth-, and network-bound**; it likes **cheap horizontal scale** and does not need a GPU. The inner loop is **GPU-bound**; it likes **quiet CPUs**, steady batches, and minimal stalls. When both run on one host, they **steal** from each other in ways that profiling summaries understate: you pay context switches, cache pollution, and the opportunity cost of cores that could either prefetch or drive CUDA host-side work.

The standard distributed-systems move is a **queue** as the narrow waist—pub/sub intuition without requiring you to name a particular broker. **Producers** push prepared samples; **consumers** pop and train. Decoupling **throughput** from **latency**: if the network hiccups, depth in the queue buys time; if the GPU experiment steps up batch size, you drain faster without rewriting preprocessing code for efficiency.

Two papers anchor that story without turning this post into a literature review.

**GoldMiner** (Alibaba and collaborators, **SIGMOD 2023**) attacks the moment when **faster GPUs make CPU preprocessing the bottleneck**. Training is inherently multi-stage: features are transformed, embeddings are gathered, bytes are decoded—**before** the kernel launch you care about. GoldMiner **splits** the preprocessing graph from GPU training and **elastically scales** the preprocessing side (including automatic identification of stateless operators and scheduler-aware scaling). The useful mental model for us: **the long pole is not always `loss.backward()`**; sometimes it is everything that has to happen **before** the tensor hits the device—and that stage deserves its **own** fleet.

**TensorSocket** ([arXiv:2409.18749](https://arxiv.org/abs/2409.18749), _TensorSocket: Shared Data Loading for Deep Learning Training_) is a complementary snapshot. A **TensorProducer** wraps a dataloader and serves batches over **ZeroMQ** to multiple **TensorConsumers**. Collocated experiments such as hyperparameter search, architecture search, and anything that launches **many** training processes against the **same** dataset stop **replaying** identical I/O and CPU transforms in every process. It is a concrete example of the pattern we care about: **socket-mediated handoff**, explicit producer/consumer roles, and **amortized** data work across consumers.

Both papers are about **separating concerns** so you can scale the side that hurts.

### Who waits? Three regimes

The queue helps in every regime, but the **knob you turn** changes:

| Regime | Examples | Who waits? | What the queue buys |
| ------ | ---------- | ---------- | ------------------- |
| **Long GPU step, short data step** | Large transformer at modest batch with heavy matmuls; optimized attention; simple decode | GPU rarely starves; pipeline has **slack** | Hides **jitter** from remote storage; safe to run **fewer** CPU workers per GPU; matches the “entitlement” headroom AWS and Tigris measure |
| **Short GPU step, long data step** | Tiny models, large microbatches of cheap ops, or **expensive** per-sample CPU (video, audio, multimodal fusion); **single-cell** with ranking + tokenization + sparse assembly from S3 | GPU **starves** without help | Scale **producers** (more CPU nodes, optional local cache) until the queue stays ahead; GoldMiner’s outer-loop story |
| **Both long** | Huge model **and** brutal preprocessing | Everything hurts | Still worth splitting: **independent failure domains**, no GIL/NUMA fights on the GPU box, elastic CPU pool |

In [post 1](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders/) and [post 2](https://slaf-project.github.io/slaf/blog/blazing-fast-dataloaders-2/), I talked through the optimization of the **outer loop** for our domain. Mixture of Scanners (MoS) for shuffle quality at scan speed, vectorized tokenization for per-cell cost. **Cloud** pushes that optimized outer loop into a **network ceiling** on one machine. The fix we ship is to **replicate** the outer loop: **stateless CPU workers** running the same MoS pipeline against S3-backed Lance tables, a **shared queue** in front of training, and a **thin GPU consumer** that only knows how to batch and `to(device)`.

> 💡 **Figure (placeholder).** CPU fleet → queue (+ dict) → GPUs. Label independent scaling knobs; annotate producer-bound versus consumer-bound with arrows.

---

## SLAF’s shape of the problem: `DistributedSLAFDataLoader`

**`DistributedSLAFDataLoader`** ([`slaf/ml/distributed.py`](https://github.com/slaf-project/slaf/blob/main/slaf/ml/distributed.py)) is that idea with SLAF-specific bones:

1. **CPU worker fleet** — Modal containers, each stateless, each running the **full** MoS pipeline: contiguous Lance reads from object storage, Polars windowing, block shuffle, vectorized tokenization.
2. **`modal.Queue` + `modal.Dict`** — the narrow waist. The queue is a distributed FIFO of **ready samples**. The dict exists for a genomics-specific reason: **cells that straddle Lance fragments** need cross-worker assembly (next section).
3. **GPU consumer** — a small PyTorch-facing iterator: pull `batch_size` samples, stack, move to device. **No** Lance, **no** Polars, **no** SQL in the training image.

The **object storage** discussion above explains **why** sequential, well-sharded reads matter; **queue decoupling** here explains **why** the GPU container stays dumb; the implementation details below explain **how** we survive sparse COO reality on a distributed fleet.

---

## Implementation: `modal.Queue`, `modal.Dict`, and the boundary problem

### One compressed sample per queue item

Modal queues cap individual items at **1 MiB**. A tokenized cell (`max_genes=2048`, gene ids + values + mask at int64 widths) is on the order of **tens to low hundreds of KB** uncompressed—fine, but not spacious. We **compress** each sample before `put` and **decompress** on the consumer.

**How this differs from a typical PyTorch `DataLoader`:** In the common pattern, **worker processes** pull **raw** examples (or indices), apply `collate_fn`, and hand the training loop **one batch tensor dict per step**—the batch is the natural unit between workers and the main process. Here, **remote CPU workers** finish the **full** SLAF path (read, rank, tokenize, shuffle participation) and enqueue **one fully processed sample per queue item**. The process on the GPU host **collects `batch_size` queue entries** and stacks them into a batch itself. So the handoff granularity is **sample-at-a-time on the queue**, not **batch-at-a-time from a worker**. That costs a bit more **host-side assembly** work per step, but buys **finer queue depth control**, clearer **backpressure** (you reason about how many cells are buffered, not how many batches), and **per-sample** fault isolation—a single bad cell does not force you to discard an entire prefetched batch.

### `modal.Dict` for cross-partition boundary cells

SLAF stores expression as **COO** triples: `(cell_integer_id, gene_integer_id, value)`. Lance stores them in **fragments**—contiguous row runs on disk. A **single cell** can spill across **two fragments**: some of its nonzero rows land at the tail of fragment *K*, the rest at the head of fragment *K + 1*.

On one node, the prefetcher simply reads across the seam. With **distributed** workers, fragment *K* and fragment *K + 1* might be read by **different** containers. Neither container alone sees a complete cell.

We use **`modal.Dict`** as a small, explicit coordination store. When a worker finishes a fragment but a cell’s records **continue** in the next fragment, it stores the **partial cell** keyed by `cell_integer_id`. When the worker that owns the next fragment starts, it **looks up** partials, completes them, enqueues finished samples, and **deletes** the dict entry.

This problem barely exists in “one file = one image” CV pipelines, or indeed other datasets where one record = one training sample. It is **intrinsic** to sparse single-cell layouts if you want contiguous columnar scans without materializing per-cell files.

### Fault tolerance via Modal

CPU workers crash. Networks flap. We deliberately **delegate** worker restarts and retries to Modal’s container model. The **queue** and **dict** are platform-managed; they survive individual container failures within a run, betting that **operational** fault tolerance belongs in the orchestrator, not in bespoke dataloader recovery code inside every training script.

> 💡 **Figure (placeholder).** Sequence diagram: read fragment → dict check / merge partial cell → compress → `Queue.put` → consumer `get` × batch_size → decompress → batch → `to(device)` → step.

---

## Deploying on Modal

The whole apparatus is **three Modal concepts** in one app—no separate Redis cluster, no hand-rolled broker.

**CPU workers** are `@modal.function(cpu=N)` with tunable concurrency. Stateless workers mean **any** rank can be fed by **any** container; scaling from a handful to **dozens** of workers is mostly a parameter change.

**`modal.Queue` and `modal.Dict`** are named app objects visible to both CPU and GPU functions. Same credentials, same region, no connection string sprawl.

**GPU workers** are `@modal.function(gpu="H100")` (or whatever you choose) running ordinary PyTorch **DDP**. The training code sees an iterator; it does not import SLAF’s dataframe stack.

Economically, this is the shape you want: **CPU hours are cheap relative to H100 hours**. If prefetch ever threatens to starve the GPU, you add CPU containers until the queue depth stabilizes—**long before** you consider paying for idle silicon.

> 💡 **Figure (placeholder).** Modal diagram: CPU pool ($/hr) → queue + dict → H100 container ($/hr). Show two independent scaling sliders.

---

## `fast-scgpt`: the proof (infrastructure only)

[`fast-scgpt`](https://github.com/slaf-project/fast-scgpt) is a reference harness that wires **all four** dataloader layers together: thread prefetch, vectorized transforms, Mixture of Scanners, and **distributed** Modal ingestion. It trains a **scGPT-style** single-cell transformer on **Tahoe-100M** streamed from cloud storage, in three layouts:

- **Single GPU** — sanity-check that the dataloader is not the bottleneck at small scale.
- **Multi-GPU DDP** — eight H100s on one node; the classical sampler gives way to a **queue consumer**.
- **Multi-node DDP** — **[N]** nodes × eight H100s; every rank drains the **same** logical queue while CPU workers scale **orthogonally**.

We are releasing it for free (under **MIT License**) as a reference **implementation**. On Tahoe-100M, this stack achieves the following step-time behavior (numbers to be filled after the final benchmark pass):

| Config | Step latency (ms) | Cells/sec | Dataloader wait (ms) | GPU utilization (%) |
| ------ | ----------------- | --------- | -------------------- | ------------------- |
| Single GPU | [TBD] | [TBD] | [TBD] | [TBD] |
| Multi-GPU (8× H100) | [TBD] | [TBD] | [TBD] | [TBD] |
| Multi-node ([N]× H100) | [TBD] | [TBD] | [TBD] | [TBD] |

_Targets from internal scaffolding: ~300–400 ms per step, **> 2500** cells/sec aggregate where applicable, **sub-millisecond** dataloader wait, GPU utilization pinned to “as high as the model allows.”_

At multi-node scale, variance across CPU workers should disappear into **queue depth** instead of showing up as idle GPU time.

Beyond dataloading, `fast-scgpt` enables **Flash Attention 3** and several deviations from the original scGPT stack. Treat those as **starting points**, not as peer-reviewed improvements to the architecture.

> 💡 **Figure (placeholder).** Profiler strip for multi-node: dataloader wait hugging zero.

---

## What this unlocks: economics of scale

Engineering tables turn into biology timelines quickly once you multiply by **Chinchilla-style** token budgets—on the order of **~100-1000 tokens per parameter** for compute-optimal training based on a rough empirical survey of single-cell foundation models. For a scGPT-scale model (~**[M]** parameters), that implies ~**[C]** billion tokens. If one cell corresponds to roughly **[K]** tokens at your gene sequence settings, a Chinchilla-shaped run needs on the order of **[C/K]** million cells—**[fraction]** of a full Tahoe-100M epoch. Plug your realized **[X]** cells/sec and **[N]** GPUs into that arithmetic and you get a wall-clock **[T]** hours.

| Config | Cells/sec | Wall clock (Chinchilla-shaped run) |
| ------ | --------- | ----------------------------------- |
| Single GPU, local disk | 28,207 | [T1] hrs |
| Single GPU, cloud (vertical limit) | ~2–3k | [T2] hrs |
| Multi-node, distributed ([N] nodes) | [X]k | [T3] hrs |

_Fill T1–T3 once X and N are measured._

### Dollars per run

At Modal’s **~$[P]/hr** ballpark per H100, a **[T3]**-hour job on **[N]** GPUs is **~$[GPU_cost]** before discounts. CPU workers add **~$[CPU_cost]**: typically small because CPU containers sit at a **[ratio]×** cheaper price band than the GPU tier.

Total back-of-envelope for a Chinchilla-optimal pretraining-style run on the largest open atlas we ship against: **~$[total]** when you plug real pricing and duration.

### What becomes possible

Those numbers matter because they change **who** can run the experiment. A lean biotech team, a university lab, or an individual maintainer can aim at **atlas-scale pretraining** without standing up a data platform team, without petabyte-scale **local** staging, and without treating the dataset as anything other than **the SLAF tables already in object storage**.

When the next hundred-million- or billion-cell atlas lands, the response is not “rewrite the format.” It is **turn up more CPU workers** until the queue depth says stop. The data stays in SLAF on S3; the training fleet stays **ephemeral** and **proportional** to the question you are asking.

> 💡 **Figure (placeholder).** Economics summary table: config, cells/sec, wall clock, GPU $, CPU $, total.

---

## Appendix: why not Ray Data?

Ray Data is a serious system; its [streaming batch paper (arXiv:2501.12407)](https://arxiv.org/abs/2501.12407) is a good read if you want another angle on **heterogeneous** CPU/GPU pipelines. We evaluated it carefully. Three **specific** mismatches pushed us toward a custom path:

### 1. Training-time shuffle is local, not global

For streaming training, the practical knob is often `local_shuffle_buffer_size`—a **bounded** window shuffle. Ray’s own docs are clear that this is **not** a global shuffle. MoS delivers **88–90%** of theoretical maximum entropy on our distance tests at **97%** of sequential throughput; achieving that requires **sub-file** control MoS relies on that we did not find exposed through `read_lance()` in Ray Data.

### 2. No fine-grained partition control for MoS

Ray Data’s Lance integration schedules partition ranges centrally. MoS needs to say: “scanner *i* starts at row **1,247,822** and reads **819,200** contiguous rows **this** iteration.” Without that API, you get file-level randomness (fast, low entropy) or a full `random_shuffle` (high entropy, heavy materialization, poor pipelining).

### 3. The cross-partition boundary problem

Ray Data does not give a first-class primitive for **logical records split across physical fragments**. SLAF’s COO layout needs **stateful** coordination between workers—exactly the niche `modal.Dict` fills. Alternatives imply **pre-localizing** cells to fragments (violating store-once, query-in-place) or **custom actors** bolted beside Ray Data.

None of that is a knock on Ray Data **in general**. It is the intersection of **our** sparse row model and **our** shuffle requirements with **their** abstractions.

---

The distributed stack lives in [`slaf/ml/distributed.py`](https://github.com/slaf-project/slaf/blob/main/slaf/ml/distributed.py) (Modal app + `DistributedSLAFDataLoader`) and the framework-agnostic pieces under [`slaf/distributed/`](https://github.com/slaf-project/slaf/tree/main/slaf/distributed). The runnable harness is [`fast-scgpt`](https://github.com/slaf-project/fast-scgpt).
