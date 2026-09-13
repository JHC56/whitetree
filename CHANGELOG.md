# Changelog

## 0.1.0

First release.

- `MahalanobisIndex`: covariance estimated at `fit` (relative ridge, Ledoit-Wolf shrinkage
  when n < 5d), Cholesky whitening, exact kNN.
- `DynamicKDTree`: insert and delete without rebuild (geometric run stack over
  `scipy.spatial.cKDTree`, distance-bound pruning, tombstones, `delete_many()`,
  `consolidate()`).
- Drift tracking: `drift()`, `refit()`, optional `refit_threshold`.
- One writer, any number of concurrent readers.
- Single-point and small-batch queries use the largest run's k-th distance as a bound for
  the other runs; one-run indexes with nothing deleted call `cKDTree.query` directly.
- Tombstoned points are dropped whenever their run is rebuilt by a later insert.
