# Changelog

## [0.3.2] - 2026-01-26

### Added
- f78b969 allow maximum rows per fragment to be controllable
- f3f9d00 change coverage parsing to work with pytest.ini change
- 8e18498 type and tests refactoring
- 21c6557 mypy error on ci
- 66e2482 unit tests for hf remote dataset
- 36b9e0e add support for slaf on remote huggingface datasets via hf:// protocol
- a1704cb mypy errors in ci fixed
- e59bc65 bump polars version
- 781dd5c adds pushdown filtering for layers
- 812245c end to end tests for layers and metadata
- 52c3d36 resolve mypy errors
- 5a70606 docs and examples for layers and metadata
- 92e3836 obs_view -> obs; var_view -> var; obs -> obs_deprecated; var -> var_deprecated
- a79b705 return dataframe with obs_view or var_view are accessed
- 229f769 remove spurious tests
- b6c1380 conversion support for obsm, varm, uns
- d511ab6 tests for cache invalidation lifecycle
- 7df6792 cache invalidation on mutation + tests for uns
- 0b80122 mixin refactor for DRY + obs, var, obsm, varm views with tests
- 980c138 implement layer deletion and tests
- cca2c91 unit tests for layer assignment
- cb93491 implement layer assignment to lazy anndata
- 09868c0 unit tests for conversion from h5ad with layers
- 7605312 handle chunked writes and multi file conversion scenario for layers
- 93cab14 adds basic layer conversion support for anndata
- c672402 test: Phase 2 - Add unit tests for layers access
- 6830c6b feat: Phase 2 - Add layers access support
- 806eae2 feat: Add layers infrastructure (Phase 1, format v0.4)
- 8ac5a9a CSC -> CSR typo fix
- 00478a5 update ml benchmarks docs and add charts
- d3492b6 update benchmarks with annbatch
- 4a5a911 handle dot in fieldname responsibly: lance doesn't like this
- 0e10d93 mypy errors
- 4074a54 optionally don't load metadata in SLAFArray
- b5b17cb parallelize fragment reads for cloud data loader
- 098bef1 update badge
