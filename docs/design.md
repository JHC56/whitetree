# Design notes

How the index works and the measurements behind the defaults. Numbers are single-thread,
dim = 3 unless stated. Scripts are in `benchmarks/`.

## Whitening

If `C = L Lᵀ` is the Cholesky factorization of the covariance, the Mahalanobis distance
between `a` and `b` equals the Euclidean distance between `L⁻¹a` and `L⁻¹b`. So each point
is transformed once at `fit` and the index is an ordinary Euclidean KD-tree.

Covariance estimation:

- Ridge is relative to scale: `reg * trace(C)/d` is added to the diagonal. With an absolute
  ridge, changing the units of the input changed the answers (recall 0.76 to 0.18 at unit
  scale 1e-3). The default `1e-9` still moves about 0.4% of neighbours at condition
  number 1e8.
- Ledoit-Wolf shrinkage is applied only when `n < 5d`. Below that it helps (recall 0.004
  to 0.538 at `n = d`); above it hurts (0.912 to 0.787 at `n = 10d`). The rule is based
  on sample size, not condition number. On data with condition number 1e10, shrinkage was
  worse at every `n` we tried.

FAISS has the same transform (`PCAMatrix(d, d, -0.5)`) but estimates the covariance from
a `1000·d` subsample in float32. On data with a DC offset of 1e4 it produces negative
eigenvalues and NaN. At condition number 1e8 it gets 0.0 to 0.6 recall even with all
points. If you give FAISS the already whitened points, its search is exact.

## Dynamic updates

`scipy.spatial.cKDTree` cannot be updated after construction. The usual fix is the
logarithmic method (Bentley and Saxe, 1980): keep several trees, rebuild small ones often
and large ones rarely.

The binary version (runs of size `2^i`) did not work well here. It keeps `popcount(n)`
trees, 6 at n = 50,000, and every query has to visit all of them. Query throughput was 20
to 30% of a single static tree. The merge loop was only 25% of that cost, so optimizing
it could not get to 50%. The reason is that `cKDTree.query` has a fixed per-call cost that
barely depends on tree size (1.6 µs on 16 points vs 3.2 µs on 50,000 points, batched), so
each extra tree adds about 50% no matter how small it is. What matters is the number of
trees and how skewed their sizes are.

What we use instead is a geometric run stack. Runs are kept largest-first with

```
size(run i) >= size_ratio * size(run i + 1)
```

The newest `micro_size` points sit in a buffer with no tree and are scanned directly. When
the buffer is full it becomes a chunk. Any run that would break the invariant against the
chunk is popped and concatenated into it, and the result is rebuilt as one tree. The number
of runs is `log_T(n / micro_size) + 1`, so 3 or 4 for any practical `n`, and the second
largest run is at most `n / size_ratio`. We first tried LSM-style leveling with fixed level
capacities. At some values of `n` a middle level grew to 33% of the data and retention
dropped to 45%, so that was replaced.

Pruning: the largest run is searched first and its k-th distance is used as
`distance_upper_bound` for the other runs. For batches of 64 or more the bound is
different per row, so the batch is sorted by k-th distance and split into chunks, each
with its own bound; one global bound (the batch maximum) did not prune anything there.
Smaller batches use the global bound, which for a single query is its own exact bound.

Single-point queries lose more to extra runs than batches do, because each `cKDTree.query`
call costs about 30 µs regardless of tree size and a single query pays that per run. On
the UCI road network (clustered, many near-duplicates) a single query runs at about a
third of static speed after half the data was inserted, versus about half for batches.
Runs of 3,000 points or fewer are scanned with `cdist` instead, which is cheaper than a
tree call at that size.

Defaults `size_ratio=32, micro_size=8` came from a sweep over n = 5e3 to 1e6, interleaved
with the static baseline so CPU throttling affects both equally:

| configuration | insert n=160k | insert slope | worst query retention |
|---|---|---|---|
| binary Bentley-Saxe (T=2, b=1) | 5.76 s | 1.02 | 20% |
| T=32, b=8 (default) | 4.40 s | 1.23 | 50% |
| T=64, b=8 | 6.28 s | 1.21 | 45% |
| T=64, b=32 | 4.99 s | 1.26 | 51% |

Retention sits around 50% because of the structure: two components keep 60 to 80%, three
keep 50 to 65%, and past n ≈ 1e5 there are always three. On real sensor logs it is 80 to
93%, because duplicates and clusters slow the static baseline more than they slow us.
`consolidate()` merges everything into one tree and returns retention to 100%.

Deletes are tombstones. If a component's candidates were tombstoned, the query doubles `k`
for that component until it has enough live points or the component is exhausted. The
tombstone check is a binary search into a sorted id array; `np.isin` sorted the whole
array on every query and cost milliseconds once tens of thousands of points were deleted.
Tombstoned points are physically dropped whenever their run is rebuilt by a later insert,
which keeps the tombstone set small in a sliding-window stream, and a full compaction
runs once tombstones exceed `compact_ratio` of what is stored.

## Threading

One writer, any number of readers. The writer changes its own state under a lock and then
publishes one tuple `(components, dead_ids, meta)`. Readers read that tuple once and never
take the lock. Rules:

1. a published tuple is never modified;
2. flushing the micro buffer allocates new arrays, so a reader holding the old view still
   sees the points it had;
3. `insert` writes slot `j` before publishing a view whose count is `j + 1`.

Rule 2 came from a real failure. `cKDTree` construction releases the GIL, so reader
threads ran during the window when flushed points belonged to no component. In a 3-reader
test, 99% of queries came back from an index that looked empty.

`MahalanobisIndex` keeps its whitening matrix in `meta`, so `refit()` publishes the new
matrix and the re-whitened points in the same assignment. A reader that got one without
the other would return a wrong answer rather than a stale one.

## Drift

The whitening is estimated once. On four years of a household power log, fitting once and
then streaming the rest dropped recall (against a freshly fitted metric) from 0.99 to 0.76.

`drift()` is `‖W C_recent Wᵀ − I‖_F / √d` over a ring buffer of the last `cov_window`
rows. It is zero when the current whitening still whitens the current data. On that stream
it correlates with true recall at r = 0.90. Re-fitting when it exceeds 0.35 raised mean
recall from 0.85 to 0.95 for +2.4% insert time. Re-fitting on a fixed timer was worse at
the tail than not re-fitting (0.77 vs 0.80), because it fires during calm periods and
misses the turbulent ones. Automatic re-fit is off by default: a re-fit rebuilds the whole
index and pauses the stream (1.8 s at n = 2M).

## Prior art

Whitening by Cholesky is standard. Log-structured KD-trees exist in parallel C++ (BDL-tree,
arXiv:2112.06188). The run-stack layout is LSM-tree leveling with RocksDB-style relative
level sizing; the read/write amplification trade-off is worked out in "Competitive
Data-Structure Dynamization" (arXiv:2011.02615) and "Towards Systematic Index Dynamization"
(PVLDB 2024). This package puts those pieces together for low-dimensional Mahalanobis
search with only numpy and scipy as dependencies.
