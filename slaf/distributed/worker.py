"""
Generic worker implementation for distributed dataloading.

"""

import asyncio
import importlib
import pickle
import queue as queue_module
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl

# Modal queue item size limit (1 MiB); we compress samples to stay under this
# See https://modal.com/docs/guide/queues and https://modal.com/docs/reference/modal.Queue
MODAL_QUEUE_ITEM_SIZE_LIMIT_BYTES = 1024 * 1024


def prefetch_worker(
    worker_id: str,
    partition_indices: list[int],
    data_source_config: dict[str, Any],
    processor_config: dict[str, Any],
    queue: Any,  # Modal Queue or any queue-like object
    n_scanners: int = 8,
    prefetch_batch_count: int = 32,
    prefetch_batch_size: int = 262144,
    max_batches: int | None = None,
    partial_groups_kv: Any | None = None,  # Modal Dict or any KV store-like object
    queue_name: str | None = None,  # Optional: for logging (e.g. Modal queue name)
) -> dict[str, Any]:
    """
    Worker function - imports processors dynamically to avoid circular deps.

    Args:
        worker_id: Unique identifier for this worker
        partition_indices: List of partition indices assigned to this worker
        data_source_config: Configuration for data source (type, path, etc.)
        processor_config: Configuration for batch processor (schema, window, shuffle, etc.)
        queue: Queue-like object to write samples to (producer-side)
        n_scanners: [PRODUCER] Number of generators to sample simultaneously per worker
        prefetch_batch_count: [PRODUCER] Number of generator reads per partition before processing.
                             Controls chunk size: each chunk contains prefetch_batch_size * prefetch_batch_count rows.
                             Higher values reduce processing overhead but increase memory per chunk.
        prefetch_batch_size: [PRODUCER] Number of rows per generator read (batch size for reading from partitions)
        max_batches: Maximum number of batches to produce (for testing)
        partial_groups_kv: KV store-like object for cross-worker partial group merging

    Returns:
        Dictionary with worker metrics
    """
    # Import data source (generic)
    from slaf.distributed.data_source import LanceDataSource

    if data_source_config["type"] == "lance":
        print(
            f"[{worker_id}] Creating LanceDataSource for: {data_source_config['path']}"
        )
        data_source = LanceDataSource(data_source_config["path"])
        print(
            f"[{worker_id}] DataSource created, partition count: {data_source.get_partition_count()}"
        )
    else:
        raise ValueError(f"Unknown data source type: {data_source_config['type']}")

    # Import processor (generic)
    from slaf.distributed.processor import BatchProcessor, DataSchema
    from slaf.distributed.shuffle import Shuffle
    from slaf.distributed.window import Window

    # Import Window/Shuffle based on config
    #
    # EXTRACTABILITY: The worker doesn't hardcode imports from slaf.ml.
    # Instead, it imports based on module paths provided in processor_config.
    # Falls back to generic implementations if no factory config provided.
    #
    # This allows slaf.distributed to be extractable:
    # - When in slaf repo: config provides "slaf.ml.aggregators" module path
    # - When extracted: users provide their own module paths (e.g., "my_package.windows")
    # - Out of the box: uses Window/Shuffle from slaf.distributed if no factory config
    # - No hardcoded dependencies on slaf.ml in slaf.distributed code

    # Window: use factory if provided, otherwise use default
    if processor_config.get("window_factory"):
        # Dynamic import based on config - module path comes from config, not hardcoded
        factory_config = processor_config["window_factory"]
        window_module = __import__(
            factory_config["module"], fromlist=[factory_config["function"]]
        )
        window_factory = getattr(window_module, factory_config["function"])
        window = window_factory(
            factory_config["type"], **factory_config.get("kwargs", {})
        )
    else:
        # Use generic implementation (works out of the box)
        window = Window()

    # Shuffle: use factory if provided, otherwise use default
    if processor_config.get("shuffle_factory"):
        # Dynamic import based on config - module path comes from config, not hardcoded
        factory_config = processor_config["shuffle_factory"]
        shuffle_module = __import__(
            factory_config["module"], fromlist=[factory_config["function"]]
        )
        shuffle_factory = getattr(shuffle_module, factory_config["function"])
        shuffle = shuffle_factory(
            factory_config["type"], **factory_config.get("kwargs", {})
        )
    else:
        # Use generic implementation (works out of the box)
        shuffle = Shuffle()

    # Tokenizer is passed as a factory function name (will be created in ml/distributed.py)
    tokenizer_fn = None
    if processor_config.get("tokenizer_factory"):
        # Dynamic import and factory call
        tokenizer_config = processor_config["tokenizer_factory"]
        tokenizer_module = importlib.import_module(tokenizer_config["module"])
        tokenizer_class = getattr(tokenizer_module, tokenizer_config["class"])

        # SLAFTokenizer needs a slaf_array, so we need to recreate it from the data source path
        # Extract slaf_path from data_source_config (assumes Lance path is under slaf_path/expression.lance)
        if data_source_config["type"] == "lance":
            lance_path = data_source_config["path"]
            # Assume lance_path is like "path/to/slaf/expression.lance"
            slaf_path = lance_path.replace("/expression.lance", "")

            # Recreate SLAFArray in worker
            from slaf.core.slaf import SLAFArray

            slaf_array = SLAFArray(slaf_path, load_metadata=False)

            # Create tokenizer instance
            tokenizer_instance = tokenizer_class(
                slaf_array=slaf_array, **tokenizer_config["kwargs"]
            )

            # Create tokenizer function that works with grouped DataFrame
            # The grouped DataFrame has gene_sequence and optionally expr_sequence columns
            def tokenize_grouped(
                grouped_df: pl.DataFrame, schema: DataSchema
            ) -> dict[str, Any]:
                """Tokenize grouped DataFrame with gene sequences."""
                # Extract gene sequences and expression sequences
                gene_sequences = grouped_df[schema.item_list_key].to_list()

                # Check if we have expression sequences (for scGPT)
                if (
                    schema.value_list_key
                    and schema.value_list_key in grouped_df.columns
                ):
                    expr_sequences = grouped_df[schema.value_list_key].to_list()
                    input_ids, attention_mask = tokenizer_instance.tokenize(
                        gene_sequences, expr_sequences
                    )
                else:
                    input_ids, attention_mask = tokenizer_instance.tokenize(
                        gene_sequences
                    )

                # Return as dict (format expected by processor)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

            tokenizer_fn = tokenize_grouped

    # Create processor with data schema
    schema = DataSchema(**processor_config["schema"])

    # Create boundary handler with KV store support
    # partial_groups_kv is passed as a parameter (created by caller)
    from slaf.distributed.boundary import GroupBoundaryHandler

    boundary_handler = GroupBoundaryHandler(
        schema=schema,
        continuity_check=processor_config.get("continuity_check", "sequential"),
        partial_groups_kv=partial_groups_kv,
    )

    processor = BatchProcessor(
        schema=schema,
        window=window,
        shuffle=shuffle,
        tokenizer=tokenizer_fn,
        boundary_handler=boundary_handler,
        max_items=processor_config.get("max_items", 1024),
        seed=processor_config.get("seed", 42),
        continuity_check=processor_config.get("continuity_check", "sequential"),
        **processor_config.get("window_kwargs", {}),
    )

    # Queue is passed as a parameter (created by caller)

    # Initialize partition readers (lazy - created on-demand)
    partition_readers: dict[int, Any] = {}
    reader_active: dict[int, bool] = dict.fromkeys(partition_indices, True)

    def get_or_create_reader(partition_idx: int):
        """Get or create a reader for a partition (lazy initialization)."""
        if partition_idx not in partition_readers:
            try:
                reader = data_source.create_reader(partition_idx, prefetch_batch_size)
                partition_readers[partition_idx] = reader
                return reader
            except Exception as e:
                print(
                    f"[{worker_id}] Error creating reader for partition {partition_idx}: {e}"
                )
                reader_active[partition_idx] = False
                return None
        return partition_readers[partition_idx]

    _first_read_done: dict[str, bool] = {}  # worker_id -> whether we logged first read

    def read_from_partition(partition_idx: int, batches_to_read: int):
        """Read multiple batches from a partition.

        Batch size flow (for memory tuning):
        - batches_to_read = prefetch_batch_count (e.g. 32)
        - reader yields DataFrames of up to prefetch_batch_size rows each
          (LanceDataSource.create_reader(partition_idx, prefetch_batch_size)
           → fragment.to_batches(batch_size=prefetch_batch_size))
        - So we accumulate up to prefetch_batch_count DataFrames, total rows
          ≤ prefetch_batch_count * prefetch_batch_size per partition.
        - With n_scanners partitions in parallel, peak raw rows in memory
          ≈ n_scanners * prefetch_batch_count * prefetch_batch_size (then
          process_batch runs and we release batch_dfs after processing).
        """
        try:
            reader = get_or_create_reader(partition_idx)
            if reader is None:
                return partition_idx, [], True, 0

            batch_dfs = []
            total_rows = 0
            is_exhausted = False

            for _ in range(batches_to_read):
                try:
                    batch_df = next(reader)  # size ≤ prefetch_batch_size rows
                    batch_dfs.append(batch_df)
                    total_rows += len(batch_df)
                    # Log once per worker when first read from any partition completes (diagnostics)
                    if worker_id not in _first_read_done:
                        _first_read_done[worker_id] = True
                        print(
                            f"[{worker_id}] First read completed: partition={partition_idx}, "
                            f"batch_rows={len(batch_df)}, total_chunk_rows={total_rows}"
                        )
                except StopIteration:
                    is_exhausted = True
                    reader_active[partition_idx] = False
                    break

            return partition_idx, batch_dfs, is_exhausted, total_rows
        except Exception as e:
            print(f"[{worker_id}] Error reading from partition {partition_idx}: {e}")
            reader_active[partition_idx] = False
            return partition_idx, [], True, 0

    # Process partitions using Mixture of Scanners (MoS) approach
    total_batches = 0
    total_rows = 0
    epochs = processor_config.get("n_epochs", 1)

    # Async writer: use Modal's .aio so we don't block on network I/O (see modal.com/docs/guide/queues)
    writer_queue: queue_module.Queue[list[dict[str, Any]] | None] = queue_module.Queue(
        maxsize=64
    )
    writer_total_put = {"n": 0}

    async def _writer_async() -> None:
        put_count = 0
        loop = asyncio.get_event_loop()
        while True:
            batch = await loop.run_in_executor(None, writer_queue.get)
            if batch is None:
                writer_queue.task_done()
                break
            try:
                # Compress each sample so each queue item stays under Modal's 1 MiB limit
                items_to_put: list[bytes | dict[str, Any]] = []
                for sample in batch:
                    if isinstance(sample, dict) and "end_of_epoch" in sample:
                        items_to_put.append(sample)
                    else:
                        items_to_put.append(
                            zlib.compress(pickle.dumps(sample), level=6)
                        )
                max_compressed = max(
                    (len(x) for x in items_to_put if isinstance(x, bytes)),
                    default=0,
                )
                total_compressed = sum(
                    len(x) for x in items_to_put if isinstance(x, bytes)
                )
                print(
                    f"[{worker_id}] put.aio chunk: {len(items_to_put)} items, "
                    f"compressed max_per_item={max_compressed:,} bytes, total={total_compressed:,}"
                )
                if max_compressed > MODAL_QUEUE_ITEM_SIZE_LIMIT_BYTES:
                    print(
                        f"[{worker_id}] WARNING: max item {max_compressed:,} > "
                        f"Modal limit {MODAL_QUEUE_ITEM_SIZE_LIMIT_BYTES:,}"
                    )
                for item in items_to_put:
                    await queue.put.aio(item)
                put_count += len(batch)
                writer_total_put["n"] = put_count
                if put_count == len(batch):
                    print(
                        f"[{worker_id}] First batches put to queue "
                        f"(chunk={len(batch)}, async writer)"
                    )
            except Exception as e:
                print(f"[{worker_id}] Async writer Error putting to queue: {e}")
            finally:
                writer_queue.task_done()

    def _run_async_writer() -> None:
        asyncio.run(_writer_async())

    writer_thread = threading.Thread(target=_run_async_writer, daemon=False)
    writer_thread.start()

    if queue_name:
        print(f"[{worker_id}] Queue name: {queue_name}")
    print(
        f"[{worker_id}] Starting processing: {epochs} epochs, {len(partition_indices)} partitions"
    )
    print(
        f"[{worker_id}] Prefetch: batch_size={prefetch_batch_size}, batch_count={prefetch_batch_count}, "
        f"n_scanners={n_scanners} → peak raw rows/round ≤ {n_scanners * prefetch_batch_count * prefetch_batch_size:,}"
    )

    _epoch_first_round: dict[int, bool] = {}  # epoch -> have we logged first round

    for epoch in range(epochs):
        print(f"[{worker_id}] Starting epoch {epoch + 1}/{epochs}")
        # Reset reader active status for new epoch
        reader_active = dict.fromkeys(partition_indices, True)
        partition_readers = {}  # Reset readers for new epoch

        while any(reader_active.values()):
            # Sample n_scanners active partitions
            active_partitions = [idx for idx in partition_indices if reader_active[idx]]
            if not active_partitions:
                break

            # Sample up to n_scanners partitions
            n_to_sample = min(n_scanners, len(active_partitions))
            sampled_partitions = random.sample(active_partitions, n_to_sample)
            if epoch not in _epoch_first_round:
                _epoch_first_round[epoch] = True
                print(
                    f"[{worker_id}] First read round: sampling {len(sampled_partitions)} "
                    f"partitions (of {len(active_partitions)} active)"
                )

            # Read from sampled partitions in parallel
            with ThreadPoolExecutor(max_workers=n_scanners) as executor:
                futures = {
                    executor.submit(
                        read_from_partition, part_idx, prefetch_batch_count
                    ): part_idx
                    for part_idx in sampled_partitions
                }

                # Process each partition result as it completes (maintains sequential processing per partition)
                # Collect samples from all partitions, then batch put_many for efficiency
                all_samples_batch = []
                completed_futures = 0

                for future in as_completed(futures):
                    partition_idx, batch_dfs, is_exhausted, rows = future.result()
                    completed_futures += 1

                    # Mark partition as inactive if exhausted
                    if is_exhausted:
                        reader_active[partition_idx] = False

                    if batch_dfs:
                        # Process batch through pipeline immediately (maintains partition state)
                        # Returns list of samples (one per group)
                        # Pass is_exhausted so boundary handler knows if partition is done
                        # rows here = total rows in this chunk (sum of all batch_dfs); "First read completed" logged only the first batch
                        if total_batches == 0:
                            print(
                                f"[{worker_id}] First process_batch: partition={partition_idx}, "
                                f"rows={rows}, dfs={len(batch_dfs)}"
                            )
                        t0 = time.perf_counter()
                        samples = processor.process_batch(
                            batch_dfs,
                            epoch=epoch,
                            partition_id=partition_idx,
                            is_partition_exhausted=is_exhausted,
                        )
                        elapsed = time.perf_counter() - t0
                        n_samples = len(samples)
                        all_samples_batch.extend(samples)
                        total_rows += rows
                        # Log process_batch timing and sample count (first 4 per worker, or if 0/slow)
                        if total_batches == 0 and (
                            completed_futures <= 4 or n_samples == 0 or elapsed > 5.0
                        ):
                            print(
                                f"[{worker_id}] process_batch done: partition={partition_idx}, "
                                f"samples={n_samples}, elapsed={elapsed:.2f}s"
                            )

                    # Batch put_many when we have enough samples or all futures complete
                    # Use low threshold (10) so queue populates quickly; flush remainder when round completes
                    all_futures_complete = completed_futures >= len(sampled_partitions)
                    if len(all_samples_batch) >= 10 or (
                        all_futures_complete and len(all_samples_batch) > 0
                    ):
                        n_put = len(all_samples_batch)
                        # Hand off to writer thread in chunks (main loop doesn't block on Modal queue)
                        put_chunk_size = 50
                        total_put = 0
                        for i in range(0, n_put, put_chunk_size):
                            chunk = all_samples_batch[i : i + put_chunk_size]
                            writer_queue.put(
                                chunk
                            )  # may block if writer is slow (backpressure)
                            total_put += len(chunk)
                        total_batches += total_put
                        all_samples_batch.clear()

                        # Check max_batches limit
                        if max_batches is not None and total_batches >= max_batches:
                            writer_queue.put(None)
                            writer_thread.join()
                            return {
                                "worker_id": worker_id,
                                "batches_produced": total_batches,
                                "rows_processed": total_rows,
                                "status": "completed",
                            }

                    # If round finished with no samples put, log once (helps debug queue size 0)
                    elif all_futures_complete and total_batches == 0:
                        print(
                            f"[{worker_id}] Round complete but 0 samples put so far "
                            f"(boundary may be holding partial groups; next round may flush)"
                        )

                # Put any remaining samples to writer queue
                if all_samples_batch:
                    writer_queue.put(list(all_samples_batch))
                    total_batches += len(all_samples_batch)
                    all_samples_batch.clear()

    # After all epochs: flush writer thread and wait for it to finish
    writer_queue.put(None)
    writer_thread.join()

    # Send end-of-epoch marker
    queue.put({"end_of_epoch": True})

    return {
        "worker_id": worker_id,
        "batches_produced": total_batches,
        "rows_processed": total_rows,
        "status": "completed",
    }
