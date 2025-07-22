import queue
import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue

import polars as pl

# Try to import torch, but make it optional
try:
    import torch
    from torch.utils.data import IterableDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Tensor operations will be disabled.")

# Try to import Lance, but make it optional
try:
    import lance

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False
    print("Warning: Lance not available. Fragment loading will be disabled.")

# Try to import Polars, but make it optional
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Warning: Polars not available. Fragment loading will be disabled.")

from slaf.core.slaf import SLAFArray
from slaf.ml.tokenizers import SLAFTokenizer


@dataclass
class RawFragment:
    """Container for a fragment with raw gene sequences (not tokenized)"""

    fragment_id: int
    gene_sequences: list[list[int]]  # List of gene ID sequences, one per cell
    cell_integer_ids: list[int]  # Corresponding cell integer IDs


class FragmentLoader:
    """Loads and processes Lance fragments using Polars"""

    def __init__(self, slaf_array: SLAFArray, seed: int = 42, max_genes: int = 1024):
        self.slaf_array = slaf_array
        self.seed = seed
        self.max_genes = max_genes  # Configurable like Phase0

        # Create fragments once in __init__ to save cycles
        self.fragments = lance.dataset(
            f"{self.slaf_array.slaf_path}/expression.lance"
        ).get_fragments()

    def load_fragment(self, fragment_id: int) -> RawFragment:
        """Load a fragment and apply window functions with Polars (simplified like Phase0)"""
        start_time = time.time()

        # Load fragment using Lance dataset (exactly like Phase0)
        fragment = self.fragments[fragment_id]
        fragment_df = pl.from_arrow(fragment.to_table())
        assert isinstance(fragment_df, pl.DataFrame), (
            f"Expected DataFrame, got {type(fragment_df)}"
        )
        load_time = time.time() - start_time

        # Apply window functions (exactly like Phase0)
        window_start = time.time()
        result = fragment_df.with_columns(
            [
                pl.col("value")
                .rank(method="dense", descending=True)
                .over("cell_integer_id")
                .alias("gene_rank")
            ]
        ).filter(pl.col("gene_rank") <= self.max_genes)

        # Group by cell and create gene sequences (exactly like Phase0)
        grouped = result.group_by("cell_integer_id").agg(
            [
                pl.col("gene_integer_id").alias("gene_sequence"),
                pl.col("value").alias("expr_sequence"),
            ]
        )
        window_time = time.time() - window_start

        # Shuffle cells within fragment (exactly like Phase0)
        shuffle_start = time.time()
        random.seed(self.seed + fragment_id)
        cell_shuffle_df = pl.DataFrame(
            {
                "cell_integer_id": grouped["cell_integer_id"].unique(),
                "shuffle_key": [
                    random.random()
                    for _ in range(len(grouped["cell_integer_id"].unique()))
                ],
            }
        )
        shuffled_grouped = grouped.join(
            cell_shuffle_df, on="cell_integer_id", how="left"
        )
        shuffled_grouped = shuffled_grouped.sort("shuffle_key")
        shuffled_cell_integer_ids = shuffled_grouped["cell_integer_id"].to_list()
        shuffled_gene_sequences = shuffled_grouped["gene_sequence"].to_list()
        shuffle_end = time.time()
        shuffle_time = shuffle_end - shuffle_start
        total_time = time.time() - start_time

        print(f"Fragment {fragment_id} timing:")
        print(f"  Load: {load_time:.3f}s")
        print(f"  Window functions: {window_time:.3f}s")
        print(f"  Shuffle: {shuffle_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")
        print(f"  Cells: {len(shuffled_cell_integer_ids)}")

        return RawFragment(
            fragment_id=fragment_id,
            gene_sequences=shuffled_gene_sequences,
            cell_integer_ids=shuffled_cell_integer_ids,
        )


class AsyncFragmentPrefetcher:
    """Async prefetcher for Lance fragments"""

    def __init__(
        self,
        fragment_processor,  # Can be FragmentLoader or tokenizer-specific processor
        max_queue_size: int = 10,
        sampler=None,  # Not used - sequential loading only
    ):
        self.fragment_processor = fragment_processor
        self.max_queue_size = max_queue_size
        self.sampler = sampler
        self.queue: Queue[RawFragment] = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.should_stop = False

    def start(self):
        """Start the prefetching worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.should_stop = False
            self.worker_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self.worker_thread.start()

    def stop(self):
        """Stop the prefetching worker thread"""
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def _prefetch_worker(self):
        """Worker thread that loads fragments in background (simplified like Phase0)"""
        fragment_id = 0
        while not self.should_stop:
            try:
                # Load fragment sequentially (like Phase0)
                fragment = self.fragment_processor.load_fragment(fragment_id)

                # Put in queue (like Phase0)
                try:
                    self.queue.put_nowait(fragment)
                    fragment_id += 1  # Sequential like Phase0
                except queue.Full:
                    # Queue is full, wait a bit
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error loading fragment {fragment_id}: {e}")
                break

    def get_fragment(self) -> RawFragment | None:
        """Get next fragment from queue"""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def has_fragment(self) -> bool:
        """Check if fragment is available"""
        return not self.queue.empty()


class SLAFIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming SLAF data with async prefetching.

    This dataset provides efficient streaming of tokenized single-cell data
    with background fragment loading to minimize GPU idle time.
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        tokenizer: SLAFTokenizer,
        batch_size: int = 32,
        seed: int = 42,
        max_queue_size: int = 10,
        device: str | None = None,
        pin_memory: bool = False,
        sampler_strategy: str = "sequential",
        tokenizer_type: str = "geneformer",  # Add tokenizer type parameter
    ):
        super().__init__()
        self.slaf_array = slaf_array
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seed = seed
        self.max_queue_size = max_queue_size
        self.device = device
        self.tokenizer_type = tokenizer_type
        self.max_genes = (
            1024 if tokenizer_type == "scgpt" else 2048
        )  # Default max_genes
        self.pin_memory = pin_memory

        # Set device
        if device is not None and TORCH_AVAILABLE:
            self.device = torch.device(device)
        elif TORCH_AVAILABLE:
            # Auto-detect optimal device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = None

        # Initialize fragment processor with tokenizer-specific max_genes
        max_genes = 1024 if tokenizer_type == "scgpt" else 2048
        self.fragment_loader = FragmentLoader(
            slaf_array, seed=seed, max_genes=max_genes
        )

        # Initialize async prefetcher (sequential like Phase0)
        self.prefetcher = AsyncFragmentPrefetcher(
            fragment_processor=self.fragment_loader,
            max_queue_size=max_queue_size,
            sampler=None,  # No complex sampling - sequential like Phase0
        )

        # Start async prefetching
        self.prefetcher.start()

        # Wait for prefetcher to initialize and load first fragment
        self._wait_for_prefetcher_ready()

    def _wait_for_prefetcher_ready(self, timeout: float = 10.0):
        """Wait for prefetcher to be ready with fragments"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.prefetcher.has_fragment():
                print(
                    f"✓ Prefetcher ready with fragments after {time.time() - start_time:.2f}s"
                )
                return
            time.sleep(0.1)

        print(f"⚠️ Prefetcher not ready after {timeout}s, proceeding anyway...")

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through batches of tokenized data.

        Yields:
            dict: Batch containing:
                - input_ids: Tokenized sequences (torch.Tensor)
                - attention_mask: Boolean mask for valid tokens (torch.Tensor)
                - cell_ids: Cell integer IDs (torch.Tensor)
        """
        start_time = time.time()
        batches_yielded = 0
        last_rate_time = start_time
        last_rate_batches = 0

        while True:
            # Get fragment from prefetcher
            fragment = self.prefetcher.get_fragment()
            if fragment is None:
                print("No more fragments available")
                break

            num_cells = len(fragment.cell_integer_ids)

            # Process all cells in this fragment - SEQUENTIAL FROM PRE-SHUFFLED DATA
            for batch_start in range(0, num_cells, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_cells)

                # Extract batch data
                batch_gene_sequences = fragment.gene_sequences[batch_start:batch_end]
                batch_cell_ids = fragment.cell_integer_ids[batch_start:batch_end]

                # Tokenize the batch using tokenizer-specific processors
                batch_tokens = []

                for _i, (gene_sequence, _cell_id) in enumerate(
                    zip(batch_gene_sequences, batch_cell_ids, strict=False)
                ):
                    # Use tokenizer-specific conversion methods
                    if self.tokenizer_type == "scgpt":
                        # scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
                        # Use simple conversion since we don't have expression values in fragment
                        tokens = self.tokenizer._convert_gene_sequence_to_scgpt_tokens_simple(
                            gene_sequence, max_genes=1024
                        )
                    else:  # geneformer
                        # Geneformer format: ranked gene tokens
                        tokens = (
                            self.tokenizer._convert_gene_sequence_to_geneformer_tokens(
                                gene_sequence, max_genes=2048
                            )
                        )

                    batch_tokens.append(tokens)

                # Convert to tensors
                batch_tensors = torch.tensor(batch_tokens, dtype=torch.long)
                attention_mask = batch_tensors != self.tokenizer.special_tokens["PAD"]
                cell_ids_tensor = torch.tensor(batch_cell_ids, dtype=torch.long)

                # Transfer to device if specified
                device_start = time.time()
                if self.device is not None:
                    batch_tensors = batch_tensors.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    cell_ids_tensor = cell_ids_tensor.to(self.device)
                device_time = time.time() - device_start

                # Print device transfer overhead every 100 batches
                if batches_yielded % 100 == 0:
                    print(f"Device transfer overhead: {device_time:.3f}s")

                batches_yielded += 1

                # Logging
                if batches_yielded % 100 == 0:
                    current_time = time.time()
                    time_since_last_rate = current_time - last_rate_time
                    batches_since_last_rate = batches_yielded - last_rate_batches
                    if time_since_last_rate > 0:
                        instantaneous_rate = (
                            batches_since_last_rate / time_since_last_rate
                        )
                        overall_rate = batches_yielded / (current_time - start_time)
                        print(
                            f"📊 Batch {batches_yielded}: {instantaneous_rate:.1f} batches/sec (instantaneous, overall: {overall_rate:.1f})"
                        )
                    last_rate_time = current_time
                    last_rate_batches = batches_yielded

                yield {
                    "input_ids": batch_tensors,
                    "attention_mask": attention_mask,
                    "cell_ids": cell_ids_tensor,
                }

    def __del__(self):
        """Cleanup when dataset is destroyed"""
        self.prefetcher.stop()
