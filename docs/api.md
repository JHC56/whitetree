# API

## `MahalanobisIndex`

```python
MahalanobisIndex(cov=None, reg=1e-9, dynamic=True, size_ratio=32, micro_size=8,
                 cov_window=100_000, refit_threshold=None, check_every=10_000,
                 shrinkage="auto")
```

| parameter | default | meaning |
|---|---|---|
| `cov` | `None` | covariance to whiten with; estimated from the `fit` data when `None` |
| `reg` | `1e-9` | ridge added to the diagonal, relative to `trace(C)/d` |
| `dynamic` | `True` | `True`: `DynamicKDTree` backend with insert/delete. `False`: static `cKDTree` |
| `size_ratio` | `32` | backend: ratio between consecutive runs |
| `micro_size` | `8` | backend: newest points held outside any tree |
| `cov_window` | `100_000` | recent rows kept for `drift()` / `refit()`; `None` disables tracking |
| `refit_threshold` | `None` | auto `refit()` once `drift()` exceeds it; `0.35` recommended if used |
| `check_every` | `10_000` | inserts between drift checks |
| `shrinkage` | `"auto"` | Ledoit-Wolf shrinkage: `"auto"` only when `n < 5d`, a float forces the intensity, `None` disables |

| method | returns | notes |
|---|---|---|
| `fit(X)` | `self` | `X` is `(n, d)`. Raises on NaN/inf, constant columns, `n < 2`; warns above condition number 1e12 |
| `kneighbors(q, k=5, workers=-1)` | `(distances, ids)`, each `(nq, k)` | `q` is `(d,)` or `(nq, d)`. Rows shorter than `k` pad with `inf` / `-1`. Thread-safe |
| `insert(vec, payload=None)` | `int` id | dynamic only |
| `insert_many(vecs, payloads=None)` | `(n,)` ids | dynamic only |
| `delete(point_id)` | `bool` | `False` if the id is unknown or already deleted |
| `delete_many(point_ids)` | `int` | number actually removed; one lock and one publish for the whole batch |
| `consolidate()` | `self` | merges every run into one tree |
| `drift()` | `float` | `‖W C_recent Wᵀ − I‖_F / √d`; `0.0` while fewer than `10d` rows are tracked |
| `refit(cov=None)` | `self` | re-estimates the whitening (from `cov_window` rows, or `cov`) and rebuilds |
| `recent_covariance()` | `(d, d)` or `None` | covariance of the tracked rows |
| `len(idx)` | `int` | live points |

Attributes after `fit`: `L` (Cholesky factor), `L_inv` (whitening matrix), `dim`,
`shrinkage_` (intensity actually applied), `backend`.

## `DynamicKDTree`

Euclidean backend; usable on its own.

```python
DynamicKDTree(dim, size_ratio=32, micro_size=8, compact_ratio=0.5, fast_build=False)
```

| parameter | default | meaning |
|---|---|---|
| `compact_ratio` | `0.5` | compact automatically once tombstones exceed this fraction of stored points |
| `fast_build` | `False` | midpoint splits: inserts 1.4 to 1.6x cheaper, queries 1.02 to 1.09x slower |

| method | returns |
|---|---|
| `insert(vec, payload=None)` | `int` id |
| `insert_many(vecs, payloads=None)` | `(n,)` ids |
| `bulk_load(pts, emulate_inserts=False)` | `self`; replaces the contents |
| `query(x, k=5, workers=-1, prune=True, prune_chunks=16)` | `(distances, ids)`, each `(nq, k)` |
| `delete(point_id)` | `bool` |
| `delete_many(point_ids)` | `int` removed |
| `consolidate()` | `self` |
| `compact()` | `None`; drops tombstoned points |
| `payload(pid)` | stored payload or `None` |
| `n_components` | number of components a query visits |
| `run_sizes()` | list of run sizes, largest first |
| `len(t)` | live points |
