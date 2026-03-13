"""
SLAF-specific distributed dataloader.

Composes generic distributed components with SLAF-specific logic.
"""

from datetime import datetime
from typing import Any

import modal
from loguru import logger

from slaf.core.slaf import SLAFArray
from slaf.distributed.coordinator import Coordinator

# Import generic distributed components
from slaf.distributed.data_source import LanceDataSource
from slaf.distributed.dataloader import DecompressingQueueWrapper, DistributedDataLoader
from slaf.distributed.processor import DataSchema

# Import SLAF-specific components (for type hints and adapters)
from slaf.ml.aggregators import Window
from slaf.ml.samplers import Shuffle
from slaf.ml.tokenizers import SLAFTokenizer

# Configure Modal image for SLAF workers
# Cache bust: the git install runs inside run_commands with a client-side timestamp
# so this layer rebuilds every deploy and picks up the latest branch commit.
_BUILD_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "python3-dev", "git")
    .pip_install("uv")
    .uv_pip_install(
        "slafdb[ml]",
        force_build=True,
    )
    .run_commands(f"echo 'Image built at {_BUILD_TS}'")
)

# Create SLAF-specific Modal app
# IMPORTANT: This app must be deployed before use. Options (install-path agnostic):
#   Python:  from slaf.ml.distributed import deploy_dataloader_app; deploy_dataloader_app()
#   CLI:     slaf deploy
# See: https://modal.com/docs/guide/apps#deployed-apps
app = modal.App("slaf-distributed-dataloader")


def deploy_dataloader_app(*, show_logs: bool = False) -> None:
    """Deploy the SLAF distributed dataloader Modal app so it is available for training.

    Call this from a training script (or anywhere slaf is installed) to deploy the
    app before using DistributedSLAFDataLoader. Works regardless of install method
    (uv, pip, venv, conda, containers).

    Args:
        show_logs: If True, stream Modal build/deploy logs to stdout. Default False
            so deploy is quiet when called from a training script that logs its own
            progress (avoids interleaving with trainer logs).
    """
    if show_logs:
        with modal.enable_output():
            app.deploy()
    else:
        app.deploy()


@app.function(
    image=image,
    cpu=8,
    memory=32768,  # 32 GB per worker
    timeout=3600,
    secrets=[modal.Secret.from_name("s3-credentials")],
)
def distributed_prefetch_worker(
    worker_id: str,
    partition_indices: list[int],
    data_source_config: dict[str, Any],
    processor_config: dict[str, Any],
    queue_name: str,
    n_scanners: int = 8,
    prefetch_batch_count: int = 32,
    prefetch_batch_size: int = 16384,
    max_batches: int | None = None,
    partial_groups_kv_name: str | None = None,
) -> dict[str, Any]:
    """
    SLAF-specific Modal worker function with SLAF image.

    This function wraps the framework-agnostic worker implementation and
    handles all Modal-specific setup (Queue, KV store, etc.).
    """
    from slaf.distributed.worker import prefetch_worker

    # Create Modal Queue
    # Use create_if_missing=True to ensure queue exists (it should be created in __init__)
    queue = modal.Queue.from_name(queue_name, create_if_missing=True)

    # Create Modal Dict for cross-worker boundary merging if enabled
    partial_groups_kv = None
    if (
        processor_config.get("enable_cross_worker_boundary_merging", False)
        and partial_groups_kv_name
    ):
        partial_groups_kv = modal.Dict.from_name(
            partial_groups_kv_name, create_if_missing=True
        )

    # Call the framework-agnostic implementation
    return prefetch_worker(
        worker_id=worker_id,
        partition_indices=partition_indices,
        data_source_config=data_source_config,
        processor_config=processor_config,
        queue=queue,
        n_scanners=n_scanners,
        prefetch_batch_count=prefetch_batch_count,
        prefetch_batch_size=prefetch_batch_size,
        max_batches=max_batches,
        partial_groups_kv=partial_groups_kv,
        queue_name=queue_name,
    )


class DistributedSLAFDataLoader:
    """
    SLAF-specific distributed dataloader.

    Composes generic distributed components with SLAF-specific configuration.

    Queue warmup: Workers start asynchronously. Before iterating, wait for the
    queue to fill (e.g. call wait_for_queue()) or you may see timeouts.
    See benchmarks/benchmark_cloud_dataloaders_modal.py for a full pattern
    (raw_mode=True, wait for queue size >= 50, then iterate).
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        tokenizer_type: str = "geneformer",
        window: Window | None = None,
        shuffle: Shuffle | None = None,
        tokenizer: SLAFTokenizer | None = None,
        n_workers: int = 64,
        n_scanners: int = 16,
        prefetch_batch_size: int = 16384,  # 16K rows per Lance batch (small default for testing; raise e.g. 262144 for prod)
        prefetch_batch_count: int = 32,
        batch_size: int = 32,
        max_genes: int = 1024,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        n_epochs: int = 1,
        raw_mode: bool = False,
        return_tensors: bool = True,
        prefetch_factor: int = 4,
        queue_timeout: float = 1.0,
        seed: int = 42,
        queue_name: str | None = None,
        **window_kwargs: Any,
    ):
        """
        Initialize distributed SLAF dataloader.

        Args:
            slaf_array: SLAFArray instance containing the data
            tokenizer_type [PRODUCER]: Tokenization strategy ("geneformer", "scgpt", or "raw")
                           If "raw", raw_mode is automatically enabled
            window [PRODUCER]: Window function instance (optional, created automatically if None)
            shuffle [PRODUCER]: Shuffle function instance (optional, created automatically if None)
            tokenizer [PRODUCER]: SLAFTokenizer instance (optional, created automatically from tokenizer_type if None)
            n_workers: [PRODUCER] Number of Modal workers (producer-side parallelism)
            n_scanners: [PRODUCER] Number of scanners per worker (for Mixture of Scanners)
            prefetch_batch_size: [PRODUCER] Max rows per Lance batch (passed to fragment.to_batches).
                                 Lower = less worker memory; peak ≈ n_scanners * prefetch_batch_count * this.
            prefetch_batch_count: [PRODUCER] Number of Lance batches read per partition before processing.
                                 Chunk size ≤ prefetch_batch_size * prefetch_batch_count rows per partition.
                                 Lower = less memory; higher = fewer process_batch calls.
            batch_size: [CONSUMER] Training batch size (number of samples per batch)
            max_genes [PRODUCER]: Maximum genes per cell after window function
            vocab_size [PRODUCER]: Vocabulary size for tokenizer
            n_expression_bins: Number of expression bins for scGPT
            n_epochs [PRODUCER]: Number of epochs to process
            raw_mode [PRODUCER]: If True, return raw data without tokenization
            return_tensors: [CONSUMER] If True, return torch.Tensor objects (matches SLAFDataLoader).
                          If False, return Python lists/objects (matches Hugging Face).
                          Default: True.
            prefetch_factor: [CONSUMER] Number of concurrent threads for queue.get_many() calls.
                           Each thread makes independent get_many() calls, allowing true parallelism.
                           Higher values use more memory but improve throughput when network I/O is the bottleneck.
                           Default: 4 (4 concurrent threads).
            queue_timeout: [CONSUMER] Timeout in seconds for queue.get_many() calls.
                          Higher values allow longer waits when queue is empty (useful for slow producers).
                          Lower values fail faster (useful for detecting end-of-data quickly).
                          Default: 1.0 seconds.
            seed: Random seed for reproducibility
            queue_name: Name of Modal Queue (auto-generated if None)
            **window_kwargs: Additional window function parameters
        """
        # Handle raw_mode
        if tokenizer_type == "raw":
            raw_mode = True
            tokenizer = None
        elif raw_mode:
            tokenizer = None

        # Create tokenizer if not provided and not in raw mode
        if tokenizer is None and not raw_mode:
            tokenizer = SLAFTokenizer(
                slaf_array=slaf_array,
                tokenizer_type=tokenizer_type,
                vocab_size=vocab_size,
                n_expression_bins=n_expression_bins,
            )
        # Create data source
        lance_path = f"{slaf_array.slaf_path}/expression.lance"
        data_source = LanceDataSource(lance_path)

        # Create coordinator
        coordinator = Coordinator(data_source, n_workers)
        assignments = coordinator.assign_partitions(seed=seed)

        # Create queue and KV store (same name used for consumer and workers — see queue flow below)
        if queue_name is None:
            queue_name = f"slaf-dataloader-{id(slaf_array)}"
        modal.Queue.from_name(queue_name, create_if_missing=True)

        # Create KV store for cross-worker boundary merging
        # We'll create it after processor_config is defined, but the name is deterministic
        partial_groups_kv_name = f"{queue_name}-partial-groups"

        # Prepare configs for workers
        data_source_config = {
            "type": "lance",
            "path": lance_path,
        }

        # Data schema for SLAF (maps generic schema to SLAF column names)
        schema = DataSchema(
            group_key="cell_integer_id",  # Group by cell
            item_key="gene_integer_id",  # Items are genes
            value_key="value",  # Values are expression
            group_key_out="cell_integer_id",  # Output keeps same group key
            item_list_key="gene_sequence",  # Aggregated gene list
            value_list_key="expr_sequence",  # Aggregated expression list (for scGPT)
        )

        processor_config = {
            "schema": {
                "group_key": schema.group_key,
                "item_key": schema.item_key,
                "value_key": schema.value_key,
                "group_key_out": schema.group_key_out,
                "item_list_key": schema.item_list_key,
                "value_list_key": schema.value_list_key,
            },
            # Window factory config (module path in config, not hardcoded)
            # For now, we'll use None and let the worker use the default Window
            # TODO: Add factory functions to slaf.ml.aggregators and slaf.ml.samplers
            "window_factory": None,  # Will use default Window from slaf.distributed
            "shuffle_factory": None,  # Will use default Shuffle from slaf.distributed
            "max_items": max_genes,  # Generic name
            "seed": seed,
            "n_epochs": n_epochs,
            "window_kwargs": window_kwargs,
            "continuity_check": "sequential",  # How to detect continuity between partitions
            "enable_cross_worker_boundary_merging": True,  # Enable cross-worker merging via KV store
        }

        # Create the KV dict to ensure it exists before workers try to access it
        if processor_config.get("enable_cross_worker_boundary_merging", True):
            modal.Dict.from_name(partial_groups_kv_name, create_if_missing=True)

        # Tokenizer factory config (for dynamic import in worker)
        # Only include tokenizer if not in raw mode
        if tokenizer and not raw_mode:
            processor_config["tokenizer_factory"] = {
                "module": "slaf.ml.tokenizers",
                "class": "SLAFTokenizer",
                "kwargs": {
                    # Serialize tokenizer config (vocab size, etc.)
                    # Tokenizer instance itself can't be serialized
                    "tokenizer_type": tokenizer.tokenizer_type.value,
                    "vocab_size": tokenizer.vocab_size,
                    "n_expression_bins": tokenizer.n_expression_bins,
                },
            }
        else:
            # Raw mode - no tokenizer
            processor_config["tokenizer_factory"] = None

        # Spawn workers
        # NOTE: The app must be deployed before spawning workers:
        #   modal deploy slaf/ml/distributed.py
        # Or ensure the app is running when called from another Modal function.
        # We cannot use app.run() here because we may be inside another Modal function.
        # Reference the deployed function by name to avoid hydration issues
        # when called from another Modal app
        # The app must be deployed first: modal deploy slaf/ml/distributed.py
        worker_function = modal.Function.from_name(
            "slaf-distributed-dataloader", "distributed_prefetch_worker"
        )
        # Explicitly hydrate the function to ensure it's synchronized with the Modal server
        # This is needed when referencing a function from a deployed app
        worker_function.hydrate()

        worker_handles = []
        logger.info("Spawning {n} workers...", n=len(assignments))
        for worker_id, assignment in assignments.items():
            logger.info(
                "  Spawning worker {worker_id} with {n_partitions} partitions",
                worker_id=worker_id,
                n_partitions=len(assignment.partition_indices),
            )
            try:
                handle = worker_function.spawn(
                    worker_id=worker_id,
                    partition_indices=assignment.partition_indices,
                    data_source_config=data_source_config,
                    processor_config=processor_config,
                    queue_name=queue_name,
                    n_scanners=n_scanners,
                    prefetch_batch_count=prefetch_batch_count,
                    prefetch_batch_size=prefetch_batch_size,
                    partial_groups_kv_name=partial_groups_kv_name,
                )
                worker_handles.append(handle)
                logger.info(
                    "  ✅ Worker {worker_id} spawned successfully (handle: {handle})",
                    worker_id=worker_id,
                    handle=handle,
                )
            except Exception as e:
                logger.error(
                    "  ❌ Error spawning worker {worker_id}: {e}",
                    worker_id=worker_id,
                    e=e,
                )
                import traceback

                traceback.print_exc()
                raise

        # Queue flow (consumer and workers use the SAME queue):
        # - queue_name is set above (or passed in); workers are spawned with queue_name=queue_name.
        # - Each worker does modal.Queue.from_name(queue_name, ...) and calls queue.put_many(...).
        # - We get the same queue by name here; DistributedDataLoader iterates via queue.get_many().
        # So the queue we pass into DistributedDataLoader is the same one workers write to.
        modal_queue = modal.Queue.from_name(queue_name, create_if_missing=True)
        # Workers compress items (zlib) to stay under Modal's 1 MiB/item limit; decompress on read
        consumer_queue = DecompressingQueueWrapper(modal_queue)

        def _shutdown_modal_workers() -> None:
            for handle in worker_handles:
                try:
                    handle.cancel()
                except Exception as e:
                    logger.warning(
                        "Failed to cancel worker {handle}: {e}",
                        handle=handle,
                        e=e,
                    )
            logger.info("Stopped {n} prefetch workers", n=len(worker_handles))

        # Create dataloader with queue object and batch_size (framework-agnostic)
        # Enable concurrent prefetching with multiple threads making concurrent get_many() calls
        # This is like having multiple consumers in the same process, allowing true parallelism
        # prefetch_factor controls number of threads, each making concurrent queue.get_many() calls
        # Enable diagnostics for bottleneck analysis
        self.dataloader = DistributedDataLoader(
            consumer_queue,
            batch_size=batch_size,
            return_tensors=return_tensors,
            prefetch_factor=prefetch_factor,  # Number of concurrent threads for queue.get_many() calls
            enable_diagnostics=True,  # Enable diagnostics for bottleneck analysis
            queue_timeout=queue_timeout,  # Timeout for queue operations
            shutdown_workers=_shutdown_modal_workers,
        )
        self.worker_handles = worker_handles
        self.queue_name = queue_name  # Store queue name for external access
        self.partial_groups_kv_name = (
            partial_groups_kv_name  # Store KV store name for external access
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)

    def wait_for_queue(
        self,
        min_batches: int = 50,
        timeout_seconds: float = 300,
        poll_interval: float = 1.0,
    ) -> int:
        """Wait for workers to fill the queue before starting consumption.

        Call this after creating the dataloader and before iterating, so the
        queue has enough batches to avoid consumer timeouts.

        Returns:
            Queue size when target was reached (or final size if timeout).
        """
        import time

        queue = modal.Queue.from_name(self.queue_name, create_if_missing=True)
        for _ in range(int(timeout_seconds)):
            try:
                size = queue.len()
                if size >= min_batches:
                    return size
            except Exception:
                pass
            time.sleep(poll_interval)
        try:
            return queue.len()
        except Exception:
            return 0

    def stop_prefetch_workers(self) -> None:
        """Cancel all prefetch workers so they exit and release Modal resources.

        Call this after training (e.g. when the dataloader is exhausted or you
        break out of the training loop) so workers do not keep running on Modal.
        Delegates to the generic dataloader's shutdown callback.
        """
        self.dataloader.stop_prefetch_workers()
