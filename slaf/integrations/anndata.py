from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse

from slaf.core.slaf import SLAFArray
from slaf.core.sparse_ops import LazySparseMixin

if TYPE_CHECKING:
    from typing import Any

    import scanpy as sc

# Import FragmentProcessor for runtime use
try:
    from slaf.core.fragment_processor import FragmentProcessor
except ImportError:
    FragmentProcessor = None  # type: ignore


class LazyExpressionMatrix(LazySparseMixin):
    """
    Lazy expression matrix backed by SLAF with scipy.sparse interface.

    LazyExpressionMatrix provides a scipy.sparse-compatible interface for accessing
    single-cell expression data stored in SLAF format. It implements lazy evaluation
    to avoid loading all data into memory, making it suitable for large datasets.

    Key Features:
        - scipy.sparse-compatible interface
        - Lazy evaluation for memory efficiency
        - Caching for repeated queries
        - Support for cell and gene subsetting
        - Integration with AnnData objects

    Examples:
        >>> # Basic usage
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> lazy_matrix = LazyExpressionMatrix(slaf_array)
        >>> print(f"Matrix shape: {lazy_matrix.shape}")
        Matrix shape: (1000, 20000)

        >>> # With AnnData integration
        >>> adata = LazyAnnData(slaf_array)
        >>> matrix = adata.X
        >>> print(f"Expression matrix shape: {matrix.shape}")
        Expression matrix shape: (1000, 20000)

        >>> # Subsetting operations
        >>> subset_matrix = matrix[:100, :5000]  # First 100 cells, first 5000 genes
        >>> print(f"Subset shape: {subset_matrix.shape}")
        Subset shape: (100, 5000)
    """

    def __init__(self, slaf_array: SLAFArray):
        """
        Initialize lazy expression matrix with SLAF array.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
                       Used for database queries and metadata access.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> matrix = LazyExpressionMatrix(slaf_array)
            >>> print(f"Initialized with shape: {matrix.shape}")
            Initialized with shape: (1000, 20000)

            >>> # Check parent reference
            >>> print(f"Parent adata: {matrix.parent_adata}")
            Parent adata: None
        """
        super().__init__()
        self.slaf_array = slaf_array
        self.parent_adata: LazyAnnData | None = None
        # Store slicing selectors
        self._cell_selector: Any = None
        self._gene_selector: Any = None
        # Initialize shape attribute (required by LazySparseMixin)
        self._shape = self.slaf_array.shape
        self._cache: dict[str, Any] = {}  # Simple caching for repeated queries

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the expression matrix.

        Returns:
            Tuple of (n_cells, n_genes) representing the matrix dimensions.

        Examples:
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> matrix = LazyExpressionMatrix(slaf_array)
            >>> print(f"Matrix shape: {matrix.shape}")
            Matrix shape: (1000, 20000)
        """
        return self._shape

    @property
    def obs_names(self) -> pd.Index | None:
        """
        Cell names (observations).

        Returns:
            pandas.Index of cell names if parent AnnData is available, None otherwise.

        Examples:
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> matrix = adata.X
            >>> print(f"Cell names: {len(matrix.obs_names)}")
            Cell names: 1000
        """
        if hasattr(self, "parent_adata") and self.parent_adata is not None:
            return self.parent_adata.obs_names
        return None

    @property
    def var_names(self) -> pd.Index | None:
        """
        Gene names (variables).

        Returns:
            pandas.Index of gene names if parent AnnData is available, None otherwise.

        Examples:
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> matrix = adata.X
            >>> print(f"Gene names: {len(matrix.var_names)}")
            Gene names: 20000
        """
        if hasattr(self, "parent_adata") and self.parent_adata is not None:
            return self.parent_adata.var_names
        return None

    def _update_shape(self):
        """Update the shape attribute based on current selectors"""
        if self._cell_selector is not None or self._gene_selector is not None:
            # Calculate the shape based on selectors
            cell_selector = (
                self._cell_selector if self._cell_selector is not None else slice(None)
            )
            gene_selector = (
                self._gene_selector if self._gene_selector is not None else slice(None)
            )

            # Use the same logic as _get_result_shape in LazySparseMixin
            cell_selector = self._compose_selectors(cell_selector, None, axis=0)
            gene_selector = self._compose_selectors(gene_selector, None, axis=1)

            n_cells = self._calculate_selected_count(cell_selector, axis=0)
            n_genes = self._calculate_selected_count(gene_selector, axis=1)

            self._shape = (n_cells, n_genes)
        else:
            # No selectors applied, return original shape
            self._shape = self.slaf_array.shape

    def _calculate_selected_count(self, selector, axis: int) -> int:
        """Calculate the number of selected entities for a given selector"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return self.slaf_array.shape[axis]

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or self.slaf_array.shape[axis]
            step = selector.step or 1

            # Clamp bounds to actual data size
            start = max(0, min(start, self.slaf_array.shape[axis]))
            stop = max(0, min(stop, self.slaf_array.shape[axis]))

            return len(range(start, stop, step))
        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                return np.sum(selector)
            return len(selector)
        elif isinstance(selector, int | np.integer):
            return 1
        else:
            return self.slaf_array.shape[axis]

    def __getitem__(self, key) -> "LazyExpressionMatrix":
        """
        Lazy slicing - returns a new LazyExpressionMatrix with composed selectors
        No computation happens until .compute() is called
        """
        cell_selector, gene_selector = self._parse_key(key)

        # Handle selectors directly as integer indices
        # For slices, lists, ints - these are already relative to the current view
        # We need to compose them with existing selectors

        # Create a new LazyExpressionMatrix with composed selectors
        new_matrix = LazyExpressionMatrix(self.slaf_array)
        new_matrix.parent_adata = self.parent_adata

        # Compose selectors (these are relative to current view)
        new_matrix._cell_selector = self._compose_selectors(
            self._cell_selector, cell_selector, axis=0
        )
        new_matrix._gene_selector = self._compose_selectors(
            self._gene_selector, gene_selector, axis=1
        )

        # Update shape based on new selectors
        new_matrix._update_shape()

        return new_matrix

    def _compose_selectors(self, old, new, axis):
        """Compose two selectors (helper method)"""
        if old is None:
            return new
        if new is None or (isinstance(new, slice) and new == slice(None)):
            return old

        # If old is a slice, we need to apply new to the range defined by old
        if isinstance(old, slice):
            # Get the range from the old slice
            old_start = old.start or 0
            old_stop = old.stop or self.slaf_array.shape[axis]
            old_step = old.step or 1

            # Handle negative indices in old slice
            if old_start < 0:
                old_start = self.slaf_array.shape[axis] + old_start
            if old_stop < 0:
                old_stop = self.slaf_array.shape[axis] + old_stop

            # Clamp old slice bounds
            old_start = max(0, min(old_start, self.slaf_array.shape[axis]))
            old_stop = max(0, min(old_stop, self.slaf_array.shape[axis]))

            # Create the range from the old slice
            old_range = list(range(old_start, old_stop, old_step))

            # Now apply the new selector to this range
            if isinstance(new, slice):
                # Apply new slice to the old range
                new_start = new.start or 0
                new_stop = new.stop or len(old_range)
                new_step = new.step or 1

                # Handle negative indices in new slice
                if new_start < 0:
                    new_start = len(old_range) + new_start
                if new_stop < 0:
                    new_stop = len(old_range) + new_stop

                # Clamp new slice bounds
                new_start = max(0, min(new_start, len(old_range)))
                new_stop = max(0, min(new_stop, len(old_range)))

                # Apply the new slice to the old range
                result_indices = old_range[new_start:new_stop:new_step]
                return result_indices
            elif isinstance(new, int | np.integer):
                # Single index into the old range
                if 0 <= new < len(old_range):
                    return [old_range[new]]
                else:
                    return []
            elif isinstance(new, list | np.ndarray):
                # List of indices into the old range
                result = []
                for idx in new:
                    if 0 <= idx < len(old_range):
                        result.append(old_range[idx])
                return result
            else:
                return new

        # If old is a list of indices, apply new to those indices
        elif isinstance(old, list | np.ndarray):
            if isinstance(new, slice):
                # Apply slice to the old list
                new_start = new.start or 0
                new_stop = new.stop or len(old)
                new_step = new.step or 1

                # Handle negative indices
                if new_start < 0:
                    new_start = len(old) + new_start
                if new_stop < 0:
                    new_stop = len(old) + new_stop

                # Clamp bounds
                new_start = max(0, min(new_start, len(old)))
                new_stop = max(0, min(new_stop, len(old)))

                return old[new_start:new_stop:new_step]
            elif isinstance(new, int | np.integer):
                # Single index into the old list
                if 0 <= new < len(old):
                    return [old[new]]
                else:
                    return []
            elif isinstance(new, list | np.ndarray):
                # List of indices into the old list
                result = []
                for idx in new:
                    if 0 <= idx < len(old):
                        result.append(old[idx])
                return result
            else:
                return new

        # For other cases, return the new selector as fallback
        return new

    def _estimate_slice_size(self, cell_selector, gene_selector) -> int:
        """Estimate the size of the slice for strategy selection"""
        cell_count = self._estimate_selected_count(cell_selector, axis=0)
        gene_count = self._estimate_selected_count(gene_selector, axis=1)
        return cell_count * gene_count

    def _apply_transformations(
        self,
        matrix: scipy.sparse.csr_matrix,
        cell_selector,
        gene_selector,
    ) -> scipy.sparse.csr_matrix:
        """Apply any stored transformations to the matrix"""
        # Get transformations from the parent LazyAnnData if available
        if self.parent_adata is not None and hasattr(
            self.parent_adata, "_transformations"
        ):
            transformations = self.parent_adata._transformations
        else:
            return matrix

        # Avoid copying if no transformations
        if not transformations:
            return matrix

        # Try to apply transformations at SQL level first
        sql_transformed = self._apply_sql_transformations(
            cell_selector, gene_selector, transformations
        )

        if sql_transformed is not None:
            # SQL transformations were applied, reconstruct matrix
            return self._reconstruct_sparse_matrix(
                sql_transformed, cell_selector, gene_selector
            )

        # Fall back to numpy transformations
        return self._apply_numpy_transformations(
            matrix, cell_selector, gene_selector, transformations
        )

    def _apply_sql_transformations(
        self, cell_selector, gene_selector, transformations
    ) -> pd.DataFrame | None:
        """Apply transformations at SQL level when possible"""
        # Check if we can apply all transformations in SQL
        sql_applicable = []
        numpy_needed = []

        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                # normalize_total can be applied in SQL
                sql_applicable.append((transform_name, transform_data))
            elif transform_name == "log1p":
                # log1p can be applied in SQL
                sql_applicable.append((transform_name, transform_data))
            else:
                # Unknown transformation, needs numpy
                numpy_needed.append((transform_name, transform_data))

        # If we have numpy-only transformations, don't use SQL
        if numpy_needed:
            return None

        # Build SQL query with transformations
        # Convert list of tuples to dictionary for _build_transformed_query
        sql_transformations = dict(sql_applicable)
        query = self._build_transformed_query(
            cell_selector, gene_selector, sql_transformations
        )

        try:
            # Execute the transformed query
            return self.slaf_array.query(query)
        except Exception:
            # If SQL transformation fails, fall back to numpy
            return None

    def _build_transformed_query(
        self, cell_selector, gene_selector, transformations
    ) -> str:
        """Build SQL query with transformations applied"""
        # Build base SQL query string
        base_query = self._build_submatrix_sql(cell_selector, gene_selector)

        # Apply transformations in order
        transformed_query = base_query
        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                transformed_query = self._apply_sql_normalize_total(
                    transformed_query, transform_data
                )
            elif transform_name == "log1p":
                transformed_query = self._apply_sql_log1p(transformed_query)

        return transformed_query

    def _build_submatrix_sql(self, cell_selector, gene_selector) -> str:
        """Build SQL query string for submatrix selection"""
        # Use the QueryOptimizer to build the SQL string
        from slaf.core.query_optimizer import QueryOptimizer

        return QueryOptimizer.build_submatrix_query(
            cell_selector=cell_selector,
            gene_selector=gene_selector,
            cell_count=self.slaf_array.shape[0],  # Use original dataset dimensions
            gene_count=self.slaf_array.shape[1],  # Use original dataset dimensions
        )

    def _apply_sql_normalize_total(self, query: str, transform_data: dict) -> str:
        """Apply normalize_total transformation in SQL"""
        cell_factors = transform_data["cell_factors"]
        # target_sum = transform_data["target_sum"]  # Removed unused variable

        # Create a CASE statement for cell factors
        case_statements = []
        for cell_id, factor in cell_factors.items():
            # Convert scientific notation to regular decimal format for SQL compatibility
            factor_str = (
                f"{factor:.10f}".rstrip("0").rstrip(".") if factor != 0 else "0"
            )

            # Handle both integer and string cell IDs
            if isinstance(cell_id, str) and cell_id.startswith("cell_"):
                # Extract integer from string like "cell_0" -> 0
                try:
                    cell_integer_id_int = int(cell_id.split("_")[1])
                except (ValueError, IndexError):
                    # Fallback: skip this cell if we can't parse it
                    continue
            elif isinstance(cell_id, int | np.integer):
                cell_integer_id_int = int(cell_id)
            else:
                # Skip if we can't handle this cell ID type
                continue

            case_statements.append(
                f"WHEN e.cell_integer_id = {cell_integer_id_int} THEN {factor_str}"
            )

        factor_case = "CASE " + " ".join(case_statements) + " ELSE 1.0 END"

        # Wrap the query to apply normalization
        return f"""
        SELECT
            e.cell_integer_id,
            e.gene_integer_id,
            e.value * {factor_case} as value
        FROM ({query}) as base_data e
        """

    def _apply_sql_log1p(self, query: str) -> str:
        """Apply log1p transformation in SQL, only to nonzero values (sparse semantics)"""
        return f"""
        SELECT
            e.cell_integer_id,
            e.gene_integer_id,
            CASE WHEN e.value != 0 THEN LN(1 + e.value) ELSE 0 END as value
        FROM ({query}) as base_data e
        """

    def _apply_numpy_transformations(
        self,
        matrix: scipy.sparse.csr_matrix,
        cell_selector,
        gene_selector,
        transformations,
    ) -> scipy.sparse.csr_matrix:
        """Apply transformations using numpy operations"""
        # Apply transformations in the order they were added (preserves order)
        result = matrix
        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                result = self._apply_normalize_total(
                    result, cell_selector, transform_data
                )
            elif transform_name == "log1p":
                result = self._apply_log1p(result)

        return result

    def _apply_normalize_total(
        self,
        matrix: scipy.sparse.csr_matrix,
        cell_selector,
        transform_data,
    ) -> scipy.sparse.csr_matrix:
        """Apply normalize_total transformation using vectorized operations"""
        cell_factors = transform_data.get("cell_factors", {})
        obs_names_local = []  # Always a list
        obs_names = None
        if hasattr(self, "parent_adata") and self.parent_adata is not None:
            try:
                obs_names = self.parent_adata.obs_names
            except (AttributeError, TypeError):
                obs_names = None
        if obs_names is None or not isinstance(obs_names, list | np.ndarray | pd.Index):
            if (
                matrix is not None
                and hasattr(matrix, "shape")
                and matrix.shape is not None
            ):
                obs_names_local = [f"cell_{i}" for i in range(matrix.shape[0])]
            else:
                obs_names_local = []
        else:
            obs_names_local = list(obs_names)
        # Now obs_names_local is always a list
        # Determine selected_cell_names based on cell_selector
        if cell_selector is None or (
            isinstance(cell_selector, slice) and cell_selector == slice(None)
        ):
            selected_cell_names = obs_names_local
        elif isinstance(cell_selector, slice):
            start = cell_selector.start or 0
            stop = cell_selector.stop or len(obs_names_local)
            step = cell_selector.step or 1
            # Clamp bounds
            start = max(0, min(start, len(obs_names_local)))
            stop = max(0, min(stop, len(obs_names_local)))
            selected_cell_names = obs_names_local[start:stop:step]
        elif isinstance(cell_selector, list | np.ndarray):
            if len(obs_names_local) > 0:
                if (
                    isinstance(cell_selector, np.ndarray)
                    and cell_selector.dtype == bool
                ):
                    selected_cell_names = [
                        obs_names_local[i]
                        for i, keep in enumerate(cell_selector)
                        if keep and 0 <= i < len(obs_names_local)
                    ]
                else:
                    selected_cell_names = [
                        obs_names_local[i]
                        for i in cell_selector
                        if isinstance(i, int | np.integer)
                        and 0 <= i < len(obs_names_local)
                    ]
            else:
                selected_cell_names = []
        elif isinstance(cell_selector, int | np.integer):
            if 0 <= cell_selector < len(obs_names_local):
                selected_cell_names = [obs_names_local[cell_selector]]
            else:
                selected_cell_names = []
        else:
            selected_cell_names = obs_names_local
        # Create a vector of factors for all cells at once
        cell_factors_vector = np.array(
            [cell_factors.get(name, 1.0) for name in selected_cell_names]
        )
        # Apply vectorized scaling using CSR matrix properties
        # Create a copy only if we need to modify the data
        if not np.allclose(cell_factors_vector, 1.0):
            result = matrix.copy()
            # Apply scaling to each row using vectorized operations
            for i in range(result.shape[0]):
                if i < len(cell_factors_vector):
                    factor = cell_factors_vector[i]
                    if factor != 1.0:
                        # Scale the non-zero elements in this row
                        start_idx = result.indptr[i]
                        end_idx = result.indptr[i + 1]
                        result.data[start_idx:end_idx] *= factor
            return result
        else:
            # No scaling needed, return original matrix
            return matrix

    def _apply_log1p(self, matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """Apply log1p transformation using vectorized operations"""
        # Create a copy only if we need to modify the data
        result = matrix.copy()
        # Apply log1p to all non-zero values using vectorized operation
        result.data = np.log1p(result.data)
        return result

    def __array_function__(self, func, types, args, kwargs):
        """Intercept numpy functions for lazy evaluation"""
        if func == np.mean:
            axis = kwargs.get("axis", None)
            return self.mean(axis=axis)
        elif func == np.sum:
            axis = kwargs.get("axis", None)
            return self.sum(axis=axis)
        elif func == np.var:
            axis = kwargs.get("axis", None)
            return self.var(axis=axis)
        elif func == np.std:
            axis = kwargs.get("axis", None)
            return self.std(axis=axis)
        else:
            # Fall back to materializing the matrix
            matrix = self.compute()
            return func(matrix, *args[1:], **kwargs)

    def mean(
        self, axis: int | None = None, fragments: bool | None = None
    ) -> float | np.ndarray:
        """Compute mean along axis via SQL aggregation"""
        return self._aggregation_with_fragments("mean", fragments, axis=axis)

    def sum(
        self, axis: int | None = None, fragments: bool | None = None
    ) -> float | np.ndarray:
        """Compute sum along axis via SQL aggregation"""
        return self._aggregation_with_fragments("sum", fragments, axis=axis)

    def var(self, axis: int | None = None) -> float | np.ndarray:
        """Compute variance along axis via SQL aggregation"""
        return self._sql_aggregation("variance", axis)

    def std(self, axis: int | None = None) -> float | np.ndarray:
        """Compute standard deviation along axis"""
        return self._sql_aggregation("stddev", axis)

    def toarray(self) -> np.ndarray:
        """Convert to dense numpy array"""
        matrix = self.compute()
        return matrix.toarray()

    def compute(self, fragments: bool | None = None) -> scipy.sparse.csr_matrix:
        """
        Explicitly compute the matrix with all transformations applied.
        This is where the actual SQL query and materialization happens.

        Args:
            fragments: Whether to use fragment processing (None for automatic)

        Returns:
            Sparse matrix with transformations applied
        """
        # Determine processing strategy
        if fragments is not None:
            use_fragments = fragments
        else:
            # Check if dataset has multiple fragments
            try:
                fragments_list = self.slaf_array.expression.get_fragments()
                use_fragments = len(fragments_list) > 1
            except Exception:
                use_fragments = False

        if use_fragments:
            processor = FragmentProcessor(
                self.slaf_array,
                cell_selector=self._cell_selector,
                gene_selector=self._gene_selector,
                max_workers=4,
                enable_caching=True,
            )
            # Use smart strategy selection for optimal performance
            lazy_pipeline = processor.build_lazy_pipeline_smart("compute_matrix")
            result = processor.compute(lazy_pipeline)
            return self._convert_to_sparse_matrix(result)
        else:
            return self._compute_global()

    def _compute_global(self) -> scipy.sparse.csr_matrix:
        """Compute the matrix globally (original implementation)"""
        # Build the SQL query with transformations
        if self.parent_adata is not None and hasattr(
            self.parent_adata, "_transformations"
        ):
            transformations = self.parent_adata._transformations
        else:
            transformations = {}

        # Try SQL-level transformations first
        if transformations:
            sql_result = self._apply_sql_transformations(
                self._cell_selector, self._gene_selector, transformations
            )
            if sql_result is not None:
                return self._reconstruct_sparse_matrix(
                    sql_result, self._cell_selector, self._gene_selector
                )

        # Fall back to base query + numpy transformations
        base_query = self._build_submatrix_sql(self._cell_selector, self._gene_selector)
        base_result = self.slaf_array.query(base_query)

        # Reconstruct base matrix
        base_matrix = self._reconstruct_sparse_matrix(
            base_result, self._cell_selector, self._gene_selector
        )

        # Apply transformations in numpy if needed
        if transformations:
            return self._apply_numpy_transformations(
                base_matrix,
                self._cell_selector,
                self._gene_selector,
                transformations,
            )

        return base_matrix

    def _convert_to_sparse_matrix(
        self, result_df: pl.DataFrame
    ) -> scipy.sparse.csr_matrix:
        """
        Convert fragment processing result to sparse matrix.

        Args:
            result_df: Polars DataFrame from fragment processing

        Returns:
            Sparse matrix representation
        """
        if len(result_df) == 0:
            # Return empty matrix with appropriate shape
            return scipy.sparse.csr_matrix(self.shape)

        # Convert to COO format for efficient sparse matrix construction
        rows = result_df["cell_integer_id"].to_numpy()
        cols = result_df["gene_integer_id"].to_numpy()
        data = result_df["value"].to_numpy()

        # Create sparse matrix
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=self.shape).tocsr()


class LazyAnnData(LazySparseMixin):
    """
    AnnData-compatible interface for SLAF data with lazy evaluation.

    LazyAnnData provides a drop-in replacement for AnnData objects that works with
    SLAF datasets. It implements lazy evaluation to avoid loading all data into memory,
    making it suitable for large single-cell datasets.

    Key Features:
        - AnnData-compatible interface
        - Lazy evaluation for memory efficiency
        - Support for cell and gene subsetting
        - Integration with scanpy workflows
        - Automatic metadata loading
        - Transformation caching

    Examples:
        >>> # Basic usage
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> print(f"AnnData shape: {adata.shape}")
        AnnData shape: (1000, 20000)

        >>> # Access expression data
        >>> print(f"Expression matrix shape: {adata.X.shape}")
        Expression matrix shape: (1000, 20000)

        >>> # Access metadata
        >>> print(f"Cell metadata columns: {list(adata.obs.columns)}")
        Cell metadata columns: ['cell_type', 'total_counts', 'batch']
        >>> print(f"Gene metadata columns: {list(adata.var.columns)}")
        Gene metadata columns: ['gene_type', 'chromosome']

        >>> # Subsetting operations
        >>> subset = adata[:100, :5000]  # First 100 cells, first 5000 genes
        >>> print(f"Subset shape: {subset.shape}")
        Subset shape: (100, 5000)
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        backend: str = "auto",
    ):
        """
        Initialize LazyAnnData with SLAF array.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
                       Used for database queries and metadata access.
            backend: Backend for expression matrix. Currently supports "scipy" and "auto".
                    "auto" defaults to "scipy" for sparse matrix operations.

        Raises:
            ValueError: If the backend is not supported.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> print(f"Backend: {adata.backend}")
            Backend: auto

            >>> # With explicit backend
            >>> adata = LazyAnnData(slaf_array, backend="scipy")
            >>> print(f"Backend: {adata.backend}")
            Backend: scipy

            >>> # Error handling for unsupported backend
            >>> try:
            ...     adata = LazyAnnData(slaf_array, backend="unsupported")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Unknown backend: unsupported
        """
        super().__init__()
        self.slaf = slaf_array
        self.backend = backend

        # Initialize expression matrix
        if backend == "scipy" or backend == "auto":
            self._X = LazyExpressionMatrix(slaf_array)
            self._X.parent_adata = self  # Set parent reference for transformations
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Lazy-loaded metadata
        self._obs = None
        self._var = None
        self._cached_obs_names: pd.Index | None = None
        self._cached_var_names: pd.Index | None = None

        # Filter selectors for subsetting
        self._cell_selector: Any = None
        self._gene_selector: Any = None
        self._filtered_obs: Callable[[], pd.DataFrame] | None = None
        self._filtered_var: Callable[[], pd.DataFrame] | None = None

        # Transformations for lazy evaluation
        self._transformations: dict[str, Any] = {}

    @property
    def X(self) -> LazyExpressionMatrix:
        """
        Access to expression data.

        Returns the lazy expression matrix that provides scipy.sparse-compatible
        access to the single-cell expression data. The matrix is lazily evaluated
        to avoid loading all data into memory.

        Returns:
            LazyExpressionMatrix providing scipy.sparse-compatible interface.

        Examples:
            >>> # Access expression data
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> print(f"Expression matrix shape: {adata.X.shape}")
            Expression matrix shape: (1000, 20000)

            >>> # Subsetting expression data
            >>> subset_X = adata.X[:100, :5000]
            >>> print(f"Subset expression shape: {subset_X.shape}")
            Subset expression shape: (100, 5000)

            >>> # Check matrix type
            >>> print(f"Matrix type: {type(adata.X)}")
            Matrix type: <class 'slaf.integrations.anndata.LazyExpressionMatrix'>
        """
        return self._X

    @property
    def obs(self) -> pd.DataFrame:
        """
        Cell metadata (observations).

        This property triggers computation of metadata when accessed.
        For lazy access to metadata structure only, use obs.columns, obs.index, etc.
        """
        if self._filtered_obs is not None:
            result = self._filtered_obs()
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame()
        if self._obs is None:
            # Use the obs from SLAFArray (now polars DataFrame)
            obs_df = getattr(self.slaf, "obs", None)
            if obs_df is not None:
                # Work with polars DataFrame internally
                obs_pl = obs_df
                # Drop cell_integer_id column if present to match AnnData expectations
                if "cell_integer_id" in obs_pl.columns:
                    obs_pl = obs_pl.drop("cell_integer_id")
                # Convert to pandas DataFrame for AnnData compatibility at API boundary
                obs_copy = obs_pl.to_pandas()
                # Set cell_id as index if present, otherwise use default index
                if "cell_id" in obs_copy.columns:
                    obs_copy = obs_copy.set_index("cell_id")
                # Set index name to match AnnData format
                if hasattr(obs_copy, "index"):
                    obs_copy.index.name = "cell_id"
                self._obs = obs_copy
            else:
                self._obs = pd.DataFrame()
        return self._obs

    @property
    def var(self) -> pd.DataFrame:
        """
        Gene metadata (variables).

        This property triggers computation of metadata when accessed.
        For lazy access to metadata structure only, use var.columns, var.index, etc.
        """
        if self._filtered_var is not None:
            result = self._filtered_var()
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame()
        if self._var is None:
            var_df = getattr(self.slaf, "var", None)
            if var_df is not None:
                # Work with polars DataFrame internally
                var_pl = var_df
                # Drop gene_integer_id column if present to match AnnData expectations
                if "gene_integer_id" in var_pl.columns:
                    var_pl = var_pl.drop("gene_integer_id")
                # Convert to pandas DataFrame for AnnData compatibility at API boundary
                var_copy = var_pl.to_pandas()
                # Set gene_id as index if present, otherwise use default index
                if "gene_id" in var_copy.columns:
                    var_copy = var_copy.set_index("gene_id")
                # Set index name to match AnnData format
                if hasattr(var_copy, "index"):
                    var_copy.index.name = "gene_id"
                self._var = var_copy
            else:
                self._var = pd.DataFrame()
        return self._var

    @property
    def obs_names(self) -> pd.Index:
        """Cell names"""
        if self._cached_obs_names is None:
            self._cached_obs_names = self.obs.index
        return self._cached_obs_names

    @property
    def var_names(self) -> pd.Index:
        """Gene names"""
        if self._cached_var_names is None:
            self._cached_var_names = self.var.index
        return self._cached_var_names

    @property
    def n_obs(self) -> int:
        """Number of observations (cells)"""
        return self.shape[0]

    @property
    def n_vars(self) -> int:
        """Number of variables (genes)"""
        return self.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the data, accounting for any applied filters"""
        if (
            getattr(self, "_cell_selector", None) is not None
            or getattr(self, "_gene_selector", None) is not None
        ):
            # Calculate the shape based on selectors
            cell_selector = (
                self._cell_selector
                if getattr(self, "_cell_selector", None) is not None
                else slice(None)
            )
            gene_selector = (
                self._gene_selector
                if getattr(self, "_gene_selector", None) is not None
                else slice(None)
            )

            # Use the same logic as _get_result_shape in LazySparseMixin
            n_cells = self._calculate_selected_count(cell_selector, axis=0)
            n_genes = self._calculate_selected_count(gene_selector, axis=1)

            return (n_cells, n_genes)
        else:
            # No filters applied, return original shape
            return self.slaf.shape

    def _calculate_selected_count(self, selector, axis: int) -> int:
        """Calculate the number of selected entities for a given selector"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return self.slaf.shape[axis]

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or self.slaf.shape[axis]
            step = selector.step or 1

            # Handle negative indices
            if start < 0:
                start = self.slaf.shape[axis] + start
            if stop < 0:
                stop = self.slaf.shape[axis] + stop

            # Clamp bounds to actual data size
            start = max(0, min(start, self.slaf.shape[axis]))
            stop = max(0, min(stop, self.slaf.shape[axis]))

            return len(range(start, stop, step))
        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                return np.sum(selector)
            return len(selector)
        elif isinstance(selector, int | np.integer):
            return 1
        else:
            return self.slaf.shape[axis]

    def __getitem__(self, key) -> "LazyAnnData":
        """Subset the data, composing selectors if already sliced"""
        # Disallow chained slicing on LazyAnnData objects
        if self._cell_selector is not None or self._gene_selector is not None:
            raise NotImplementedError(
                "Chained slicing on LazyAnnData objects is not supported. "
                "Please use adata[rows, cols] for single-step slicing instead of "
                "adata[rows][:, cols]. For chained slicing on the expression matrix, "
                "use adata.X[rows][:, cols]."
            )

        # Parse indexing key using the same logic as LazyExpressionMatrix
        cell_selector, gene_selector = self._parse_key(key)

        # Create a new LazyAnnData with the same backend
        new_adata = LazyAnnData(self.slaf, backend=self.backend)

        # Store the selectors for lazy filtering
        new_adata._cell_selector = cell_selector
        new_adata._gene_selector = gene_selector

        # Update the LazyExpressionMatrix to know about the slicing
        if isinstance(new_adata._X, LazyExpressionMatrix):
            new_adata._X.parent_adata = new_adata
            # Store the selectors in the LazyExpressionMatrix so it can use them
            new_adata._X._cell_selector = cell_selector
            new_adata._X._gene_selector = gene_selector
            # Update the shape to reflect the slicing
            new_adata._X._update_shape()

        # Override obs and var properties to apply filtering
        def filtered_obs() -> pd.DataFrame:
            # Always apply the composed selector to the original obs
            obs_df = self.obs
            if cell_selector is None or (
                isinstance(cell_selector, slice) and cell_selector == slice(None)
            ):
                pass
            else:
                obs_df = obs_df.copy()
                if isinstance(cell_selector, slice):
                    start = cell_selector.start or 0
                    stop = cell_selector.stop or len(obs_df)
                    step = cell_selector.step or 1

                    # Handle negative indices
                    if start < 0:
                        start = len(obs_df) + start
                    if stop < 0:
                        stop = len(obs_df) + stop

                    # Clamp bounds to valid range
                    start = max(0, min(start, len(obs_df)))
                    stop = max(0, min(stop, len(obs_df)))
                    obs_df = obs_df.iloc[start:stop:step]
                elif (
                    isinstance(cell_selector, np.ndarray)
                    and cell_selector.dtype == bool
                ):
                    # Boolean mask - ensure it matches the length
                    if len(cell_selector) == len(obs_df):
                        obs_df = obs_df[cell_selector]
                    else:
                        # Pad or truncate the mask to match
                        if len(cell_selector) < len(obs_df):
                            mask: np.ndarray = np.zeros(len(obs_df), dtype=bool)
                            mask[: len(cell_selector)] = cell_selector
                        else:
                            mask = cell_selector[: len(obs_df)]
                        obs_df = obs_df[mask]
                elif isinstance(cell_selector, list | np.ndarray):
                    # Integer indices
                    obs_df = obs_df.iloc[cell_selector]
                elif isinstance(cell_selector, int | np.integer):
                    obs_df = obs_df.iloc[[int(cell_selector)]]
                # else: leave as is
            # Remove unused categories for all categorical columns if not empty
            if obs_df is not None and not obs_df.empty:
                for col in obs_df.select_dtypes(include="category").columns:
                    col_data = obs_df[col]
                    # Type-safe check for pandas categorical data
                    if isinstance(col_data, pd.Series) and hasattr(col_data, "cat"):
                        obs_df[col] = col_data.cat.remove_unused_categories()
            if isinstance(obs_df, pd.DataFrame):
                return obs_df
            return pd.DataFrame()

        def filtered_var() -> pd.DataFrame:
            # Always apply the composed selector to the original var
            var_df = self.var
            if gene_selector is None or (
                isinstance(gene_selector, slice) and gene_selector == slice(None)
            ):
                pass
            else:
                var_df = var_df.copy()
                if isinstance(gene_selector, slice):
                    start = gene_selector.start or 0
                    stop = gene_selector.stop or len(var_df)
                    step = gene_selector.step or 1

                    # Handle negative indices
                    if start < 0:
                        start = len(var_df) + start
                    if stop < 0:
                        stop = len(var_df) + stop

                    # Clamp bounds to valid range
                    start = max(0, min(start, len(var_df)))
                    stop = max(0, min(stop, len(var_df)))
                    var_df = var_df.iloc[start:stop:step]
                elif (
                    isinstance(gene_selector, np.ndarray)
                    and gene_selector.dtype == bool
                ):
                    # Boolean mask - ensure it matches the length
                    if len(gene_selector) == len(var_df):
                        var_df = var_df[gene_selector]
                    else:
                        # Pad or truncate the mask to match
                        if len(gene_selector) < len(var_df):
                            mask: np.ndarray = np.zeros(len(var_df), dtype=bool)
                            mask[: len(gene_selector)] = gene_selector
                        else:
                            mask = gene_selector[: len(var_df)]
                        var_df = var_df[mask]
                elif isinstance(gene_selector, list | np.ndarray):
                    # Integer indices
                    var_df = var_df.iloc[gene_selector]
                elif isinstance(gene_selector, int | np.integer):
                    var_df = var_df.iloc[[int(gene_selector)]]
                # else: leave as is
            # Remove unused categories for all categorical columns if not empty
            if var_df is not None and not var_df.empty:
                for col in var_df.select_dtypes(include="category").columns:
                    col_data = var_df[col]
                    # Type-safe check for pandas categorical data
                    if isinstance(col_data, pd.Series) and hasattr(col_data, "cat"):
                        var_df[col] = col_data.cat.remove_unused_categories()
            if isinstance(var_df, pd.DataFrame):
                return var_df
            return pd.DataFrame()

        # Store the filter functions
        new_adata._filtered_obs = filtered_obs
        new_adata._filtered_var = filtered_var

        # Ensure obs_names and var_names match the filtered DataFrames
        new_adata._cached_obs_names = new_adata._filtered_obs().index
        new_adata._cached_var_names = new_adata._filtered_var().index

        # Copy transformations
        new_adata._transformations = self._transformations.copy()

        return new_adata

    def copy(self) -> "LazyAnnData":
        """Create a copy"""
        new_adata = LazyAnnData(self.slaf, backend=self.backend)

        # Copy selectors
        new_adata._cell_selector = self._cell_selector
        new_adata._gene_selector = self._gene_selector

        # Copy filtered metadata functions
        new_adata._filtered_obs = self._filtered_obs
        new_adata._filtered_var = self._filtered_var

        # Copy transformations
        new_adata._transformations = self._transformations.copy()

        # Set up parent reference for the expression matrix
        if isinstance(new_adata._X, LazyExpressionMatrix):
            new_adata._X.parent_adata = new_adata

        return new_adata

    def to_memory(self) -> scipy.sparse.spmatrix:
        """Load entire matrix into memory"""
        return self.get_expression_data()

    def write(self, filename: str):
        """Write to h5ad format (would need implementation)"""
        raise NotImplementedError("Writing LazyAnnData not yet implemented")

    def get_expression_data(self) -> scipy.sparse.csr_matrix:
        """Get expression data with any applied filters"""
        if isinstance(self._X, LazyExpressionMatrix):
            # If the LazyExpressionMatrix already has selectors stored, just get the data
            # without passing additional selectors to avoid double-composition
            if self._X._cell_selector is not None or self._X._gene_selector is not None:
                return self._X[:, :].compute()
            elif self._cell_selector is not None or self._gene_selector is not None:
                cell_sel = (
                    self._cell_selector
                    if self._cell_selector is not None
                    else slice(None)
                )
                gene_sel = (
                    self._gene_selector
                    if self._gene_selector is not None
                    else slice(None)
                )
                return self._X[cell_sel, gene_sel].compute()
            else:
                return self._X[:, :].compute()
        else:
            raise NotImplementedError(
                "Dask backend not yet implemented for get_expression_data"
            )

    def compute(self) -> "sc.AnnData":
        """Explicitly compute and return a native AnnData object"""
        import scanpy as sc

        # Create native AnnData object
        adata = sc.AnnData(X=self._X.compute(), obs=self.obs, var=self.var)

        return adata

    def _update_with_normalized_data(
        self, result_df: pl.DataFrame, target_sum: float, inplace: bool
    ) -> "LazyAnnData | None":
        """
        Update AnnData with normalized data from fragment processing.

        Args:
            result_df: Polars DataFrame with normalized values
            target_sum: Target sum used for normalization
            inplace: Whether to modify in place

        Returns:
            Updated LazyAnnData or None if inplace=True
        """
        if inplace:
            # Store normalization transformation
            if not hasattr(self, "_transformations"):
                self._transformations = {}

            self._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": target_sum,
                "fragment_processed": True,
                "result_df": result_df,
            }

            print(
                f"Applied normalize_total with target_sum={target_sum} (fragment processing)"
            )
            return None
        else:
            # Create a copy with the transformation
            new_adata = self.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": target_sum,
                "fragment_processed": True,
                "result_df": result_df,
            }

            return new_adata

    def _update_with_log1p_data(
        self, result_df: pl.DataFrame, inplace: bool
    ) -> "LazyAnnData | None":
        """
        Update AnnData with log1p data from fragment processing.

        Args:
            result_df: Polars DataFrame with log1p values
            inplace: Whether to modify in place

        Returns:
            Updated LazyAnnData or None if inplace=True
        """
        if inplace:
            # Store log1p transformation
            if not hasattr(self, "_transformations"):
                self._transformations = {}

            self._transformations["log1p"] = {
                "type": "log1p",
                "fragment_processed": True,
                "result_df": result_df,
            }

            print("Applied log1p transformation (fragment processing)")
            return None
        else:
            # Create a copy with the transformation
            new_adata = self.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["log1p"] = {
                "type": "log1p",
                "fragment_processed": True,
                "result_df": result_df,
            }

            return new_adata

    def _get_processing_strategy(self, fragments: bool | None = None) -> bool:
        """Determine the processing strategy based on fragments and dataset size"""
        if fragments is not None:
            return fragments
        try:
            fragments_list = self.slaf.expression.get_fragments()
            return len(fragments_list) > 1
        except Exception:
            return False


def read_slaf(filename: str, backend: str = "auto") -> LazyAnnData:
    """Read SLAF file as LazyAnnData object"""
    slaf_array = SLAFArray(filename)
    return LazyAnnData(slaf_array, backend=backend)
