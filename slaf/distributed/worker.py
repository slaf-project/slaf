"""
Generic worker implementation for distributed dataloading.

"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl


def prefetch_worker(
    worker_id: str,
    partition_indices: list[int],
    data_source_config: dict[str, Any],
    processor_config: dict[str, Any],
    queue: Any,  # Modal Queue or any queue-like object
    n_scanners: int = 8,
    batches_per_partition: int = 32,
    prefetch_batch_size: int = 262144,
    max_batches: int | None = None,
    partial_groups_kv: Any | None = None,  # Modal Dict or any KV store-like object
) -> dict[str, Any]:
    """
    Worker function - imports processors dynamically to avoid circular deps.

    Args:
        worker_id: Unique identifier for this worker
        partition_indices: List of partition indices assigned to this worker
        data_source_config: Configuration for data source (type, path, etc.)
        processor_config: Configuration for batch processor (schema, window, shuffle, etc.)
        queue_name: Name of Modal Queue to write batches to
        n_scanners: Number of generators to sample simultaneously per worker
        batches_per_partition: Number of batches to read per partition before processing
        prefetch_batch_size: Batch size for reading from partitions
        max_batches: Maximum number of batches to produce (for testing)

    Returns:
        Dictionary with worker metrics
    """
    # Import data source (generic)
    from slaf.distributed.data_source import LanceDataSource

    if data_source_config["type"] == "lance":
        data_source = LanceDataSource(data_source_config["path"])
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
        tokenizer_module = __import__(tokenizer_config["module"])
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

    def read_from_partition(partition_idx: int, batches_to_read: int):
        """Read multiple batches from a partition."""
        try:
            reader = get_or_create_reader(partition_idx)
            if reader is None:
                return partition_idx, [], True, 0

            batch_dfs = []
            total_rows = 0
            is_exhausted = False

            for _ in range(batches_to_read):
                try:
                    batch_df = next(reader)
                    batch_dfs.append(batch_df)
                    total_rows += len(batch_df)
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

    for epoch in range(epochs):
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

            # Read from sampled partitions in parallel
            with ThreadPoolExecutor(max_workers=n_scanners) as executor:
                futures = {
                    executor.submit(
                        read_from_partition, part_idx, batches_per_partition
                    ): part_idx
                    for part_idx in sampled_partitions
                }

                for future in as_completed(futures):
                    partition_idx, batch_dfs, is_exhausted, rows = future.result()

                    if batch_dfs:
                        # Process batch through pipeline
                        result = processor.process_batch(
                            batch_dfs,
                            epoch=epoch,
                            partition_id=partition_idx,
                        )

                        # Skip empty batches
                        if result.get("empty", False):
                            continue

                        # Send to queue
                        queue.put(result)
                        total_batches += 1
                        total_rows += rows

                        # Check max_batches limit
                        if max_batches is not None and total_batches >= max_batches:
                            return {
                                "worker_id": worker_id,
                                "batches_produced": total_batches,
                                "rows_processed": total_rows,
                                "status": "completed",
                            }

    # Send end-of-epoch marker
    queue.put({"end_of_epoch": True})

    return {
        "worker_id": worker_id,
        "batches_produced": total_batches,
        "rows_processed": total_rows,
        "status": "completed",
    }
