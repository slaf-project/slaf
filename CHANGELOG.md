# Changelog

## [0.3.1] - 2025-11-06

### Added
- 335c18b pyproject.toml version accidental upgrade fix
- 37e8f08 removes outdated benchmark tests
- 5731c0c minor maintenance
- 95c94cd increase timeout tolerance for cloud data loaders
- d8b9531 expose queue size as a parameter
- 218d6ab bug fixes in reading and writing config.json
- 5ae1f6e existence check bug fix
- fadd2df SLAFArray should be compatible with cloud paths
- 98c4229 add n_genes_by_counts as a pre-computed candidate for gene counts
- f148ada better path checks and more logging
- 26eecac corner case for metadata with checkpointing + logging
- 7cd9141 handle list of str in convert
- f06c23c bug fix and performance refactor for converter
- 63a11f1 advanced checkpointing for long running conversions
- c8c049d use smart open instead of open to handle cloud io
- d686293 update mocks in tests
- 15c9f6e removes pathlib because it's not reliable with cloud URIs
- 400365f fix bug in s3 path handling
- c35da9a prefix check // -> /
- b9ade88 pass Path instead of str to lance.write_dataset
- 2375ec6 existence check for remote path before creation
- 99f2f2e sort glob order
- 6ce721f fix mtx file format detection
- cab6388 update backward compatibility tests issues
- 2bf3c0f swap input output order in docs for append
- bbc3188 tests for multi file conversion and append functionality
- 33607e9 update conversion docs
- e595d5b multifile conversion and append functionality
- c799ffa remove fluff in docs
- 2e5cda5 add bionemo scdl to external dataloader benchmarks
- d0db0ee updated benchmarks
- 8d7c651 better memory usage
- 9116114 warmup and pushdown filtering for tiledb benchmarks
- 3e82da4 blog index and typos
- 4bf96a7 typo fix
- 48e5fd7 typo fix
- 5c1882f wordsmithing
- 7c8ffe7 mixture of scanners blog post
- 93f1e45 docstrings for tiledb dataloaders
- 83d6606 pypi badge
