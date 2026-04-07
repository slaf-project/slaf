# Blazing Fast Dataloaders #3: From object storage to distributed training

## Train single-cell foundation models on a lunch break. Spend less than a doordash meal.

_This is the third post in the Blazing Fast Dataloaders series. [Post 1](../blazing-fast-dataloaders.md) covered throughput: thread-based prefetch, vectorized outer loop, and contiguous reads. [Post 2](../blazing-fast-dataloaders-2.md) covered randomness: the Mixture of Scanners architecture for near-perfect shuffling at streaming speed. This post covers scale: training from object storage, when the bottleneck shifts to the network, and how we split CPU-side ingestion from GPU training with a queue-backed distributed dataloader._

---

The first two posts describe a single-node pipeline that reaches roughly **28,207 cells per second** from **local disk**, more than enough to keep one H100 fed at single-cell GPT (scGPT)-scale batch sizes. The stack is still useful as-is for workstations and staged data.

Atlas-scale training in the cloud is different: the canonical copy of the data sits in **object storage**, and compute is **ephemeral**. Pointing the same pipeline at S3 (or any similar cloud storage backend) does not achieve the same throughput. On a single node, adding cores and prefetch threads eventually stops helping because egress bandwidth and request economics cap how much data can reach that box, regardless of how efficiently the CPU would transform it afterward. In other words, once the link to storage is the limiter, vertical scaling on the training host is the wrong knob.

This post describes how we **separate concerns**—CPU workers that stream and preprocess in parallel, a narrow-waist queue, and GPU workers that mostly train—implemented today as **`DistributedSLAFDataLoader`** on Modal.

---

## The story so far

In posts 1 and 2, we described the optimizations that led to a single-node dataloader that squeezes everything it can from one machine:

- **Thread-based prefetch** uses GIL-releasing work in Lance, Arrow, and NumPy so prefetch threads overlap I/O and preprocessing without paying for multiprocessing copies.
- The **vectorized outer loop** moves tokenization out of the per-step hot path and applies it across large prefetch batches, riding sub-linear scaling of Polars window-function costs (roughly **3× better per-cell efficiency** at 1024 cells versus 32 in our scaling tables).
- **Mixture of Scanners (MoS)** streams contiguous reads while approximating a global shuffle: random subsets of scanners read sequential runs from different offsets, then an in-memory block shuffle mixes deliveries, reaching **88–90%** of theoretical maximum entropy at **97%** of sequential throughput.

Those ideas share a hidden assumption: **the CPU that prefetches and the GPU that trains share one box**. They compete for the same cores, memory bandwidth, and NUMA domains. On a workstation with fast local disk, that competition is usually secondary to “can we read and transform fast enough?” In the cloud, with a dataset as large as **Tahoe-100M on S3** and multi-GPU jobs, **network delivery** often joins (or replaces) that story as the hard ceiling on a single ingest host.

---

## The cloud storage wall

When data lives in object storage and compute is ephemeral, we still stream, shuffle, and tokenize online, but peak single-node throughput drops because round-trips now cross the network path to the bucket.

The deeper constraint is **I/O concurrency into S3**: one ingest host, however many vCPUs, ultimately funnels several **logical read streams** through **one** place—NIC, connection pool, and implicit per-prefix request budgeting all pile up on that path. **Horizontal** ingestion is different: each CPU worker is its **own** client talking to object storage from its **own** container. You are not only adding cores; you are adding **independent read parallelism** against the bucket. That raises the **aggregate** ceiling far beyond what “more threads on the same box” can unlock.

On **Tahoe-100M (train split) streamed from S3**, we measured the same MoS pipeline on Modal (**raw mode**, batch size 64). **Vertical** scaling—larger single-node ingest—lifts throughput from about **2.2k** to **~3.5k cells/sec** and then **levels off**; 16 and 32 vCPUs stay in the same band as 8, consistent with a **single-path** limit rather than “keep adding CPU forever.”

| vCPUs | Throughput (cells/sec) |
| ----- | ---------------------: |
| 4 | 2,169 |
| 8 | 3,572 |
| 16 | 3,126 |
| 32 | 3,459 |

**Horizontal** scaling—**distributed** producers on the same dataset—shows why that ceiling is not the end of the story. Aggregate cells/sec reaches the **tens of thousands** while the vertical curve stays stuck near **~3.5k** on one host.

| Producer nodes | Aggregate throughput (cells/sec) |
| --- | ---: |
| 1 | 1,012 |
| 4 | 8,575 |
| 16 | 29,526 |
| 32 | 30,581 |

The benchmark harnesses are **good enough** for the argument we care about here: the **shape**—flattening on one machine versus a **much higher** aggregate when many clients read in parallel from object storage—not publication-grade coupling of every tuning knob.

The practical lesson: **scale ingest the same way the cloud scales storage: as many readers as you need**, not only a bigger single-node prefetcher. If GPUs keep getting faster and atlases keep growing, the durable move is **distributed dataloading** with its own fleet, so **distributed I/O concurrency** matches the **ceiling object storage can offer**, which is far above what one node’s worth of S3 client parallelism tends to realize.

> 💡 **Figure (placeholder).** Two curves: vertical throughput flattening (low thousands) versus horizontal **aggregate** throughput climbing toward **~30k cells/sec**. Mark where multi-reader ingestion crosses the single-node plateau.

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

Two papers anchor that story.

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
- **Multi-node DDP** — Modal clustered runs (e.g. 16× GPU across two nodes); every rank drains the **same** logical queue while CPU workers scale **orthogonally** (see repo for `--multinode`; treat as beta per Modal).

The repo’s **Benchmarks** section reports end-to-end Modal runs; the rows below are copied from that README for the **`scgpt`** preset (**~51.1M trainable parameters**), **Tahoe-100M**, **`max_genes=512`**, **128 cells per GPU per step** (effective global batch **1024** on 8 GPUs), **`--data-source s3`**, **Flash Attention 4**, **`--sparse-gene-head`**, **`--no-use-compile`**, and **two CPU prefetch workers** on the distributed run. Step time on 8 GPUs is the **median** of per-step **max over ranks** (slowest rank sets the barrier); global **cells/s** is **global batch ÷ that step time**.

| Config | Median step latency (ms) | Global cells/sec | Dataloader wait (ms) | Training efficiency |
| ------ | ----------------- | ---------------- | -------------------- | ------------------- |
| Single GPU (1× **H200**, `modal_train.py`) | **323** | **~396** | **~0** (`train_step` **`dl`** chunk in README profile) | MFU **~23.7%**; peak VRAM **~68 GB** |
| Multi-GPU (8× **H100**, `modal_train_distributed.py`) | **56** | **~18.3k** | not split out on this run† | MFU **~17.3%**; peak VRAM **~71.7 GB / GPU** |

†On the single-GPU trace, time blocked on `next(batch)` is reported as **~0 ms** once warm. The 8-GPU README table does not break out the same **`dl`** field; steady-state **56 ms** steps are the joint result of compute, collectives, and feeding the global batch.

**Multi-node:** the README documents the `modal run … --multinode` command line but does not publish a second throughput table yet—when it does, this post can add a third row.

At multi-node scale, variance across CPU workers should disappear into **queue depth** instead of showing up as idle GPU time.

Beyond dataloading, `fast-scgpt` defaults to **Flash Attention 4** in the documented benchmarks (FA3 and compile toggles exist in the scripts). Other choices differ from published scGPT—treat them as **starting points**, not as peer-reviewed improvements to the architecture.

> 💡 **Figure (placeholder).** Profiler strip for multi-node: dataloader wait hugging zero.

---

## What this unlocks: economics of scale

**Apples to apples:** in [this reply on scGPT pretraining cost](https://github.com/bowang-lab/scGPT/issues/5#issuecomment-1551327337), the authors describe a reference setup on the order of **10.3 M cells × six epochs**—about **62 M cell presentations** through the training loop (same cell revisited across epochs, but **62 M optimizer steps worth of “a batch was just drawn from the atlas”** at that scale). That is the regime people meant when they asked how long pretraining takes.

They also point to **roughly three to four days on four A100s** for that job—a wall-clock world that assumes **staged data**, **cluster babysitting**, and **multi-day** GPU reservations.

**Same cell budget, 2026 stack:** the [`fast-scgpt` README](https://github.com/slaf-project/fast-scgpt) benchmark on **8× H100** reports **~18.3k cells/s** **global** training throughput in steady state (forward + backward + optimizer + NCCL—not dataloader-only). If you push **62 million** such presentations through at that rate, **62 × 10⁶ / 18 300 ≈ 3 390 s**—about **56 minutes** of **GPU-wall** time in the idealized steady state (Modal startup, queue ramp, compile warmup, checkpointing, and eval still add real minutes).

**Sticker math:** **8 GPUs** × (**~3 390 s** / **3 600 s/h**) ≈ **7.5** billed GPU-hours if every device is busy for that wall. At **\$2.50 / H100 / hour**, that is **~\$19** in GPU line items—**under twenty bucks**, with **CPU prefetch workers and egress** still rounding noise next to eight H100s. And you never **mirrored the whole atlas onto cluster storage**: Tahoe-100M stays in **object storage**; the dataloader meets you there.

So the headline is not “cheaper than one burrito” at this exact cell count—it is **under an hour and under twenty dollars on paper** for the **same order of magnitude of training traffic** the original issue discussed, versus **half a week on four A100s** in 2023. Different silicon, different stack, different dollars-per-hour—but the **magnitude** of the lifestyle change is the point.

### What becomes possible

The shift is **access**: a lean team can iterate on **pretraining-scale** runs without a data platform org, without **cloning the atlas to every cluster’s local filesystem**, and without the **multi-day** turnaround that made “try six epochs on 10 M cells” a calendar event. When the next hundred-million-cell drop lands, you dial **CPU producers** and **GPU trainers** independently; SLAF stays **authoritative in object storage**; the training fleet stays **ephemeral**.

> 💡 **Figure (placeholder).** Side-by-side: 2023 timeline (4× A100, **~3–4 days**, [per comment](https://github.com/bowang-lab/scGPT/issues/5#issuecomment-1551327337)) vs. **8× H100** stack (**~1 h** wall, **~\$19** GPU at \$2.50/h in the napkin above, stream from bucket).

---

## Appendix: why not Ray Data?

Ray Data is a serious system; its [streaming batch paper (arXiv:2501.12407)](https://arxiv.org/abs/2501.12407) is a good read if you want another angle on **heterogeneous** CPU/GPU pipelines. We evaluated it carefully. Three **specific** mismatches pushed us toward a custom path:

### 1. Training-time shuffle is local, not global

For streaming training, the practical knob is often `local_shuffle_buffer_size`—a **bounded** window shuffle. Ray’s own docs are clear that this is **not** a global shuffle. MoS delivers **88–90%** of theoretical maximum entropy on our distance tests at **97%** of sequential throughput; achieving that requires **sub-file** control MoS relies on that we did not find exposed through `read_lance()` in Ray Data.

### 2. No fine-grained partition control for MoS

Ray Data’s Lance integration schedules partition ranges centrally. MoS needs to say: “scanner *i* starts at row **1,247,822** and reads **819,200** contiguous rows **this** iteration.” Without that API, you get file-level randomness (fast, low entropy) or a full `random_shuffle` (high entropy, heavy materialization, poor pipelining).

### 3. The cross-partition boundary problem

Ray Data does not give a first-class primitive for **logical records split across physical fragments**. SLAF’s COO layout needs **stateful** coordination between workers—exactly the niche `modal.Dict` fills. Alternatives imply **pre-localizing** cells to fragments (violating store-once, query-in-place) or **custom actors** bolted beside Ray Data.

---

The distributed stack lives in [`slaf/ml/distributed.py`](https://github.com/slaf-project/slaf/blob/main/slaf/ml/distributed.py) (Modal app + `DistributedSLAFDataLoader`) and the framework-agnostic pieces under [`slaf/distributed/`](https://github.com/slaf-project/slaf/tree/main/slaf/distributed). The runnable harness is [`fast-scgpt`](https://github.com/slaf-project/fast-scgpt).
