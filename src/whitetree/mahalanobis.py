# -*- coding: utf-8 -*-
"""Mahalanobis kNN: estimate the covariance, whiten once (Cholesky), run a Euclidean KD-tree.

    idx = MahalanobisIndex().fit(X)
    d, i = idx.kneighbors(q, k=5)
    idx.insert(new_vec)          # dynamic=True only

One writer, any number of readers. The whitening matrix is published with the index
contents in one snapshot, so a concurrent `refit()` can make a query stale, not wrong.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree

from .dynamic_kdtree import DynamicKDTree

__all__ = ["MahalanobisIndex"]


class MahalanobisIndex:
    """Parameters
    ----------
    cov : (d, d) array or None
        Covariance to whiten with. Estimated in `fit` when None.
    reg : float
        Ridge relative to the data scale (`reg * trace(C)/d`). Moves ~0.4% of neighbours
        at condition number 1e8; use 1e-12 if that matters.
    shrinkage : "auto" | float | None
        Ledoit-Wolf shrinkage. "auto" applies it only when `n < 5*d`. A float forces the
        intensity; None disables it. The intensity used is stored in `shrinkage_`.
    dynamic : bool
        True for an insertable `DynamicKDTree`, False for a static `cKDTree`.
    size_ratio, micro_size : int
        Backend parameters (dynamic=True).
    cov_window : int or None
        Recent points kept for `drift()` / `refit()`. None disables tracking.
    refit_threshold : float or None
        Auto `refit()` once `drift()` exceeds this. Off by default: a refit rebuilds the
        index (1.8 s at n = 2M). 0.35 is the recommended value.
    check_every : int
        Inserts between drift checks.
    """

    def __init__(self, cov=None, reg=1e-9, dynamic=True, size_ratio=32, micro_size=8,
                 cov_window=100_000, refit_threshold=None, check_every=10_000,
                 shrinkage="auto"):
        self.cov = cov
        self.reg = reg
        self.shrinkage = shrinkage
        self.shrinkage_ = 0.0
        self.dynamic = dynamic
        self.size_ratio = size_ratio
        self.micro_size = micro_size
        self.cov_window = cov_window
        self.refit_threshold = refit_threshold
        self.check_every = check_every
        self.backend = None
        self.L = None
        self.L_inv = None
        self.dim = None
        self._recent = None
        self._recent_n = 0
        self._recent_at = 0
        self._since_check = 0
        self._refits = 0

    # ------------------------------------------------------ covariance estimation
    @staticmethod
    def _ledoit_wolf(X):
        """Ledoit & Wolf (2004). Returns (covariance, intensity in [0, 1])."""
        X = X - X.mean(0)
        n, d = X.shape
        S = X.T @ X / n
        m = np.trace(S) / d
        d2 = np.sum((S - m * np.eye(d)) ** 2) / d
        if d2 <= 0:
            return S, 0.0
        # sum_k ||x_k x_k^T - S||^2, expanded so it does not build n outer products
        sq = np.einsum("ij,ij->i", X, X)
        b2 = (np.sum(sq ** 2) / d - 2 * np.einsum("ij,jk,ik->", X, S, X) / d
              + n * np.sum(S ** 2) / d) / n ** 2
        a = min(max(b2, 0.0) / d2, 1.0)
        return (1 - a) * S + a * m * np.eye(d), float(a)

    def _estimate_cov(self, X):
        """Sample covariance; Ledoit-Wolf shrinkage only when n < 5*d."""
        n, d = X.shape
        S = np.atleast_2d(np.cov(X, rowvar=False))
        mode = self.shrinkage
        if mode in (None, 0, 0.0, False):
            self.shrinkage_ = 0.0
            return S
        if mode == "auto":
            if n >= 5 * d:
                self.shrinkage_ = 0.0
                return S
            C, a = self._ledoit_wolf(X)
            self.shrinkage_ = a
            return C
        a = float(mode)
        if not 0.0 <= a <= 1.0:
            raise ValueError("shrinkage must be 'auto', None, or a number in [0, 1]")
        self.shrinkage_ = a
        m = np.trace(S) / d
        return (1 - a) * S + a * m * np.eye(d)

    @staticmethod
    def _validate(X, what="X"):
        if X.ndim != 2:
            raise ValueError(f"{what} must be 2-D, got shape {X.shape}")
        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            raise ValueError(
                f"{what} contains {len(bad)} non-finite value(s), first at row {bad[0][0]}, "
                f"column {bad[0][1]}. Distances would come back NaN; drop or impute them."
            )

    # ---------------------------------------------------------------- whitening
    def _set_whitening(self, cov, d):
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if cov.shape != (d, d):
            raise ValueError(f"covariance must be {d}x{d}, got {cov.shape}")
        scale = max(np.trace(cov) / d, np.finfo(float).tiny)
        raw_ev = np.linalg.eigvalsh(cov)
        cond = raw_ev[-1] / raw_ev[0] if raw_ev[0] > 0 else np.inf
        cov = cov + np.eye(d) * self.reg * scale
        ev = np.linalg.eigvalsh(cov)
        if ev[0] <= 0:
            raise np.linalg.LinAlgError(
                "covariance is not positive definite even after regularization; "
                "check for duplicated or constant channels"
            )
        if cond > 1e12:
            warnings.warn(
                f"covariance condition number is {cond:.2e}: the whitening is dominated by "
                f"a direction with almost no variance, so distances along it are mostly "
                f"numerical noise. Drop the redundant channel, or pass shrinkage=<float>.",
                RuntimeWarning, stacklevel=3,
            )
        self.L = np.linalg.cholesky(cov)
        self.L_inv = np.linalg.inv(self.L)
        self._ref_cov = cov

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._validate(X)
        n, d = X.shape
        if n < 2:
            raise ValueError(f"need at least 2 rows to estimate a covariance, got {n}")
        sd = X.std(0)
        const = np.flatnonzero(sd == 0)
        if len(const) and self.cov is None:
            raise ValueError(
                f"column(s) {const.tolist()} are constant, so they have no scale to whiten "
                f"by; any later variation in them would dominate every distance. Drop them, "
                f"or pass an explicit cov=."
            )
        self.dim = d
        if self.cov is None:
            cov = self._estimate_cov(X)
        else:
            cov = np.asarray(self.cov, float)
            self.shrinkage_ = 0.0
        self._set_whitening(cov, d)
        Xw = X @ self.L_inv.T
        if self.dynamic:
            self.backend = DynamicKDTree(
                d, size_ratio=self.size_ratio, micro_size=self.micro_size
            ).bulk_load(Xw).set_meta(self.L_inv)
        else:
            self.backend = cKDTree(Xw)
        if self.cov_window:
            self._recent = np.empty((self.cov_window, d))
            self._recent_n = 0
            self._recent_at = 0
            self._observe(X)
            self._since_check = 0
        return self

    # -------------------------------------------------------------------- drift
    def _observe(self, X):
        """Append raw points to the ring buffer (single writer)."""
        if not self.cov_window:
            return
        X = np.atleast_2d(X)
        m = len(X)
        if m >= self.cov_window:
            self._recent[:] = X[-self.cov_window:]
            self._recent_n = self.cov_window
            self._recent_at = 0
            return
        end = self._recent_at + m
        if end <= self.cov_window:
            self._recent[self._recent_at:end] = X
        else:
            cut = self.cov_window - self._recent_at
            self._recent[self._recent_at:] = X[:cut]
            self._recent[:m - cut] = X[cut:]
        self._recent_at = end % self.cov_window
        self._recent_n = min(self._recent_n + m, self.cov_window)

    def recent_covariance(self):
        """Covariance of the ring buffer; `shrinkage_` is left untouched."""
        if not self.cov_window or self._recent_n < self.dim * 10:
            return None
        keep = self.shrinkage_
        C = self._estimate_cov(self._recent[:self._recent_n])
        self.shrinkage_ = keep
        return C

    def drift(self):
        """||W C_recent W^T - I||_F / sqrt(d); 0.0 means the whitening still fits."""
        C = self.recent_covariance()
        if C is None:
            return 0.0
        M = self.L_inv @ C @ self.L_inv.T
        return float(np.linalg.norm(M - np.eye(self.dim)) / np.sqrt(self.dim))

    def refit(self, cov=None):
        """Re-estimate the whitening from recent data and rebuild the index. O(n log n)."""
        if not self.dynamic:
            raise RuntimeError("this index was built with dynamic=False; it cannot refit")
        C = np.asarray(cov, float) if cov is not None else self.recent_covariance()
        if C is None:
            return self
        b = self.backend
        with b._wlock:
            pts = [p for p, _, _ in b.runs]
            ids = [g for _, g, _ in b.runs]
            if b._micro_n:
                pts.append(b._mpts.copy())
                ids.append(b._mids.copy())
            # fresh arrays: a reader holding the previous view still points into the old ones
            b._micro_pts = np.empty((b.micro_size, b.dim))
            b._micro_ids = np.empty(b.micro_size, dtype=np.int64)
            b._micro_n = 0
            if not pts:
                self._set_whitening(C, self.dim)
                b._meta = self.L_inv
                b._publish()
                return self
            Pw = np.vstack(pts)
            G = np.concatenate(ids)
            raw = Pw @ self.L.T                      # back to the original coordinates
            self._set_whitening(C, self.dim)
            Pn = raw @ self.L_inv.T
            b._sync_dead()
            if b._dead_arr.size:
                keep = ~np.isin(G, b._dead_arr)
                Pn, G = Pn[keep], G[keep]
                b._dead = set()
                b._dead_dirty = True
                b._sync_dead()
            b.runs = [(Pn, G, cKDTree(Pn, **b._tree_opts))] if len(Pn) else []
            b._meta = self.L_inv                     # published together with the points
            b._publish()
        self._refits += 1
        return self

    def _maybe_refit(self, m):
        if not self.refit_threshold or not self.cov_window:
            return
        self._since_check += m
        if self._since_check < self.check_every:
            return
        self._since_check = 0
        if self.drift() > self.refit_threshold:
            self.refit()

    # ---------------------------------------------------------------------- API
    def _whiten(self, q):
        return np.atleast_2d(np.asarray(q, dtype=float)) @ self.L_inv.T

    def kneighbors(self, q, k=5, workers=-1):
        """Exact Mahalanobis kNN. Returns (distances, ids), each (nq, k). Thread-safe."""
        Q = np.atleast_2d(np.asarray(q, dtype=float))
        self._validate(Q, "the query")
        if not self.dynamic:
            d, i = self.backend.query(Q @ self.L_inv.T, k=k, workers=workers)
            return d.reshape(len(Q), k), i.reshape(len(Q), k)   # cKDTree flattens k == 1
        view = self.backend._view                # one snapshot: (components, dead, whitening)
        L_inv = view[2]
        return self.backend.query(Q @ L_inv.T, k=k, workers=workers, view=view)

    def insert(self, vec, payload=None):
        if not self.dynamic:
            raise RuntimeError("this index was built with dynamic=False; it cannot insert")
        v = np.asarray(vec, dtype=float).reshape(1, -1)
        self._validate(v, "the inserted point")   # a NaN here would poison the index
        pid = self.backend.insert((v @ self.L_inv.T)[0], payload)
        self._observe(v)
        self._maybe_refit(1)
        return pid

    def insert_many(self, vecs, payloads=None):
        if not self.dynamic:
            raise RuntimeError("this index was built with dynamic=False; it cannot insert")
        V = np.atleast_2d(np.asarray(vecs, dtype=float))
        self._validate(V, "the inserted points")
        ids = self.backend.insert_many(V @ self.L_inv.T, payloads)
        self._observe(V)
        self._maybe_refit(len(V))
        return ids

    def delete(self, point_id):
        if not self.dynamic:
            raise RuntimeError("this index was built with dynamic=False; it cannot delete")
        return self.backend.delete(point_id)

    def delete_many(self, point_ids):
        if not self.dynamic:
            raise RuntimeError("this index was built with dynamic=False; it cannot delete")
        return self.backend.delete_many(point_ids)

    def consolidate(self):
        """Merge the backend's runs into one tree."""
        if self.dynamic:
            self.backend.consolidate()
        return self

    def __len__(self):
        return len(self.backend) if self.dynamic else self.backend.n
