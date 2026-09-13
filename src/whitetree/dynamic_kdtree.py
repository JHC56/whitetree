# -*- coding: utf-8 -*-
"""Exact kNN index with insert and delete: a geometric stack of cKDTrees.

Runs are kept largest-first under `size(run i) >= size_ratio * size(run i+1)`. A micro
buffer holds the newest points and is scanned directly. Queries take the k-th distance
from the largest run and use it as an upper bound for the rest. Results are identical to
one static `scipy.spatial.cKDTree` over the same points.
"""
from __future__ import annotations

import threading

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

__all__ = ["DynamicKDTree"]

# Midpoint splits build ~2x faster and query 1.02-1.09x slower. Results are identical.
_FAST_BUILD = dict(balanced_tree=False, compact_nodes=False)
_EXACT_BUILD = dict(balanced_tree=True, compact_nodes=True)


def _in_sorted(values, table):
    """Elementwise membership of `values` in the sorted int array `table`."""
    if table.size == 0:
        return np.zeros(np.shape(values), dtype=bool)
    pos = np.searchsorted(table, values)
    pos_c = np.minimum(pos, table.size - 1)
    return table[pos_c] == values


def _select_k(d, gid, k):
    """k nearest out of (nq, c) candidates, padded with inf / -1 when short."""
    nq, c = d.shape
    take = min(k, c)
    if nq == 1:                                   # single query: skip the batched path
        row = d[0]
        sel = np.argsort(row, kind="stable")[:take]
        out_d = row[sel].reshape(1, take)
        out_i = np.asarray(gid[0])[sel].reshape(1, take)
        if take < k:
            out_d = np.hstack([out_d, np.full((1, k - take), np.inf)])
            out_i = np.hstack([out_i, np.full((1, k - take), -1, dtype=np.int64)])
        return out_d, out_i
    if take < c:
        part = np.argpartition(d, take - 1, axis=1)[:, :take]
        pd = np.take_along_axis(d, part, axis=1)
        sel = np.take_along_axis(part, np.argsort(pd, axis=1, kind="stable"), axis=1)
    else:
        sel = np.argsort(d, axis=1, kind="stable")
    out_d = np.take_along_axis(d, sel, axis=1)
    out_i = np.take_along_axis(gid, sel, axis=1)
    if take < k:
        out_d = np.hstack([out_d, np.full((nq, k - take), np.inf)])
        out_i = np.hstack([out_i, np.full((nq, k - take), -1, dtype=np.int64)])
    return out_d, out_i


class DynamicKDTree:
    """Parameters
    ----------
    dim : int
    size_ratio : int
        Ratio between consecutive runs. Larger: fewer runs, faster queries, costlier inserts.
    micro_size : int
        Newest points held outside any tree. Larger: fewer tree builds, longer scan.
    compact_ratio : float
        Compact automatically once tombstones exceed this fraction of stored points.
    fast_build : bool
        Midpoint splits: inserts 1.4-1.6x cheaper, queries 1.02-1.09x slower.

    Thread safety: one writer, any number of readers. Readers work off one published
    snapshot (`_view`) and never block; a published tuple is never mutated afterwards.
    """

    def __init__(self, dim, size_ratio=32, micro_size=8, compact_ratio=0.5, fast_build=False):
        if size_ratio < 2:
            raise ValueError("size_ratio must be >= 2")
        if micro_size < 1:
            raise ValueError("micro_size must be >= 1")
        self.dim = int(dim)
        self.size_ratio = int(size_ratio)
        self.micro_size = int(micro_size)
        self.compact_ratio = float(compact_ratio)
        self.fast_build = bool(fast_build)
        self._tree_opts = _FAST_BUILD if fast_build else _EXACT_BUILD

        self.runs = []              # (pts, ids, cKDTree), largest first
        self._micro_pts = np.empty((self.micro_size, self.dim))
        self._micro_ids = np.empty(self.micro_size, dtype=np.int64)
        self._micro_n = 0
        self._next_id = 0
        self._dead = set()
        self._dead_arr = np.empty(0, dtype=np.int64)
        self._dead_dirty = False
        self._payload = {}
        self._meta = None
        self._wlock = threading.Lock()
        self._view = ((), self._dead_arr, None)
        self._publish()

    # ------------------------------------------------------------------ internals
    @property
    def _mpts(self):
        return self._micro_pts[:self._micro_n]

    @property
    def _mids(self):
        return self._micro_ids[:self._micro_n]

    def _publish(self):
        """Swap in a new (components, dead_ids, meta) snapshot, largest run first."""
        self._sync_dead()
        comps = list(self.runs)
        if self._micro_n:
            comps.append((self._mpts, self._mids, None))
        self._view = (tuple(comps), self._dead_arr, self._meta)

    def set_meta(self, meta):
        """Publish caller state atomically together with the current index contents."""
        with self._wlock:
            self._meta = meta
            self._publish()
        return self

    def _absorb(self, pts, ids):
        """Add a chunk, restoring size(run i) >= T * size(run i+1)."""
        T = self.size_ratio
        while self.runs and len(self.runs[-1][1]) < T * len(ids):
            p, g, _ = self.runs.pop()
            pts = np.vstack([p, pts])
            ids = np.concatenate([g, ids])
        if self._dead:
            # the run is being rebuilt anyway, so drop its tombstoned points for free
            self._sync_dead()
            gone = _in_sorted(ids, self._dead_arr)
            if gone.any():
                self._dead.difference_update(ids[gone].tolist())
                self._dead_dirty = True
                pts, ids = pts[~gone], ids[~gone]
        if len(ids):
            self.runs.append((pts, ids, cKDTree(pts, **self._tree_opts)))

    def _flush_micro(self):
        """Move the micro buffer into the runs. Caller publishes."""
        if self._micro_n == 0:
            return
        pts = self._mpts
        ids = self._mids
        # fresh arrays: a reader holding the previous view still points into the old ones
        self._micro_pts = np.empty((self.micro_size, self.dim))
        self._micro_ids = np.empty(self.micro_size, dtype=np.int64)
        self._micro_n = 0
        self._absorb(pts, ids)

    def _sync_dead(self):
        if self._dead_dirty:
            self._dead_arr = np.fromiter(self._dead, dtype=np.int64, count=len(self._dead))
            self._dead_arr.sort()
            self._dead_dirty = False

    def _simulate_runs(self, n):
        """Run sizes after n sequential inserts, from counters alone."""
        T, b = self.size_ratio, self.micro_size
        runs = []
        for _ in range(n // b):
            m = b
            while runs and runs[-1] < T * m:
                m += runs.pop()
            runs.append(m)
        return runs

    # ------------------------------------------------------------------ writes
    def insert(self, vec, payload=None):
        """Insert one point and return its id."""
        with self._wlock:
            pid = self._next_id
            if payload is not None:
                self._payload[pid] = payload
            j = self._micro_n
            self._micro_pts[j] = vec            # written before any view exposes slot j
            self._micro_ids[j] = pid
            self._micro_n = j + 1
            self._next_id = pid + 1
            if self._micro_n >= self.micro_size:
                self._flush_micro()
            self._publish()
        return pid

    def insert_many(self, vecs, payloads=None):
        """Batch insert; produces the same run layout as repeated insert()."""
        vecs = np.asarray(vecs, dtype=float).reshape(-1, self.dim)
        n = len(vecs)
        with self._wlock:
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            if payloads is not None:
                for pid, pl in zip(ids, payloads):
                    if pl is not None:
                        self._payload[int(pid)] = pl
            off = 0
            while off < n:
                take = min(self.micro_size - self._micro_n, n - off)
                j = self._micro_n
                self._micro_pts[j:j + take] = vecs[off:off + take]
                self._micro_ids[j:j + take] = ids[off:off + take]
                self._micro_n = j + take
                off += take
                self._next_id += take
                if self._micro_n >= self.micro_size:
                    self._flush_micro()
            self._publish()
        return ids

    def bulk_load(self, pts, emulate_inserts=False):
        """Replace the contents with one run (or, with emulate_inserts, the layout that
        inserting one at a time would give). Ids keep counting up."""
        pts = np.asarray(pts, dtype=float).reshape(-1, self.dim)
        n = len(pts)
        with self._wlock:
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            self._next_id += n
            self.runs = []
            self._micro_n = 0
            self._dead = set()
            self._dead_dirty = True
            self._payload = {}
            if n:
                sizes = self._simulate_runs(n) if emulate_inserts else [n]
                off = 0
                for cnt in sizes:
                    chunk = pts[off:off + cnt]
                    self.runs.append((chunk, ids[off:off + cnt],
                                      cKDTree(chunk, **self._tree_opts)))
                    off += cnt
                rest = n - off
                if rest:
                    self._micro_pts[:rest] = pts[off:]
                    self._micro_ids[:rest] = ids[off:]
                    self._micro_n = rest
            self._publish()
        return self

    def _is_present(self, pid):
        """Binary search; every run's id array is ascending by construction."""
        for _, g, _ in self.runs:
            if g.size and g[0] <= pid <= g[-1]:
                j = int(np.searchsorted(g, pid))
                if j < g.size and g[j] == pid:
                    return True
        m = self._mids
        return bool(m.size and (m == pid).any())

    def delete(self, point_id):
        """Tombstone a point. Returns False if it is not there."""
        pid = int(point_id)
        with self._wlock:
            if pid < 0 or pid >= self._next_id or pid in self._dead:
                return False
            if not self._is_present(pid):       # already dropped by a compaction
                return False
            self._dead.add(pid)
            self._dead_dirty = True
            self._payload.pop(pid, None)
            if len(self._dead) > self.compact_ratio * self._stored():
                self._compact_locked()
            self._publish()
        return True

    def delete_many(self, point_ids):
        """Tombstone several points under one lock and one publish. Returns how many were
        actually removed."""
        n = 0
        with self._wlock:
            for pid in np.asarray(point_ids, dtype=np.int64).tolist():
                if pid < 0 or pid >= self._next_id or pid in self._dead:
                    continue
                if not self._is_present(pid):
                    continue
                self._dead.add(pid)
                self._payload.pop(pid, None)
                n += 1
            if n:
                self._dead_dirty = True
                if len(self._dead) > self.compact_ratio * self._stored():
                    self._compact_locked()
                self._publish()
        return n

    def consolidate(self):
        """Merge every run into one. Call before a query-heavy phase."""
        with self._wlock:
            if self.n_components <= 1 and not self._dead:
                return self
            self._sync_dead()
            pts = [p for p, _, _ in self.runs]
            ids = [g for _, g, _ in self.runs]
            if self._micro_n:
                pts.append(self._mpts.copy())
                ids.append(self._mids.copy())
            self._micro_n = 0
            P = np.vstack(pts)
            G = np.concatenate(ids)
            if self._dead_arr.size:
                keep = ~np.isin(G, self._dead_arr)
                P, G = P[keep], G[keep]
                self._dead = set()
                self._dead_dirty = True
                self._sync_dead()
            self.runs = [(P, G, cKDTree(P, **self._tree_opts))] if len(P) else []
            self._publish()
        return self

    def compact(self):
        """Physically drop tombstoned points and rebuild into a single run."""
        with self._wlock:
            self._compact_locked()
            self._publish()

    def _compact_locked(self):
        if not self._dead:
            return
        self._sync_dead()
        pts, ids = [], []
        for p, g, _ in self.runs:
            keep = ~np.isin(g, self._dead_arr)
            if keep.any():
                pts.append(p[keep])
                ids.append(g[keep])
        if self._micro_n:
            keep = ~np.isin(self._mids, self._dead_arr)
            if keep.any():
                pts.append(self._mpts[keep])
                ids.append(self._mids[keep])
        self.runs = []
        self._micro_n = 0
        self._dead = set()
        self._dead_dirty = True
        self._sync_dead()
        if pts:
            P = np.vstack(pts)
            G = np.concatenate(ids)
            self.runs.append((P, G, cKDTree(P, **self._tree_opts)))

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _comp_size(comp):
        return len(comp[1])

    def _brute(self, pts, gids, x, kk, bound):
        """Scan a component with cdist. Keep cdist: the expanded (x^2 - 2xp + p^2) form
        is off by ~1e-14 and breaks bit-for-bit agreement with cKDTree."""
        m = len(gids)
        kq = min(kk, m)
        d = cdist(x, pts)
        if np.isfinite(bound):
            d[d > bound] = np.inf
        if len(x) == 1:
            if kq < m:
                row = d[0]
                sel = np.argpartition(row, kq - 1)[:kq]
                return row[sel].reshape(1, kq), gids[sel].reshape(1, kq), kq
            return d, np.asarray(gids).reshape(1, m), m
        if kq < m:
            sel = np.argpartition(d, kq - 1, axis=1)[:, :kq]
            return np.take_along_axis(d, sel, axis=1), gids[sel], kq
        return d, np.broadcast_to(gids, d.shape), m

    def _raw(self, comp, x, kk, workers, bound=np.inf):
        """Nearest kk within `bound` from one component: (distances, ids, kk used)."""
        p, g, tree = comp
        # cKDTree.query has a fixed per-call cost; tiny runs and small batches are
        # cheaper to scan directly
        if tree is None or tree.n <= 40 or (len(x) * tree.n <= 100_000 and tree.n <= 3000):
            return self._brute(p, g, x, kk, bound)
        kq = min(kk, tree.n)
        d, idx = tree.query(x, k=kq, workers=workers, distance_upper_bound=bound)
        d = d.reshape(len(x), kq)
        idx = idx.reshape(len(x), kq)
        miss = idx >= tree.n                      # outside the bound: scipy pads with n
        if miss.any():
            gid = np.where(miss, -1, g[np.where(miss, 0, idx)])
            d = np.where(miss, np.inf, d)
        else:
            gid = g[idx]
        return d, gid, kq

    def _topk(self, comp, x, k, workers, dead, bound=np.inf):
        """k nearest live points of one component as (nq, k), padded with inf / -1."""
        has_dead = dead.size > 0
        size = self._comp_size(comp)
        kk = k
        while True:
            d, gid, kq = self._raw(comp, x, kk, workers, bound)
            if not has_dead:
                break
            raw_fin = np.isfinite(d).sum(1)
            bad = _in_sorted(gid, dead)
            if bad.any():
                d = np.where(bad, np.inf, d)
                gid = np.where(bad, -1, gid)
            if kq >= size:
                break
            # only rows truncated by kk (not by the bound) can still be missing live points
            need = (raw_fin >= kq) & (np.isfinite(d).sum(1) < k)
            if not need.any():
                break
            kk = min(kk * 2, size)
        return _select_k(d, gid, k)

    def _bounded_topk(self, comp, x, k, workers, dead, r, chunks):
        """Sort the batch by k-th distance and query each chunk under its own bound."""
        nq = len(x)
        out_d = np.full((nq, k), np.inf)
        out_i = np.full((nq, k), -1, dtype=np.int64)
        order = np.argsort(r, kind="stable")
        for ch in np.array_split(order, chunks):
            if ch.size == 0:
                continue
            d, gid = self._topk(comp, x[ch], k, workers, dead, bound=float(r[ch[-1]]))
            out_d[ch] = d
            out_i[ch] = gid
        return out_d, out_i

    def query(self, x, k=5, workers=-1, prune=True, prune_chunks=16, view=None):
        """Exact kNN. Returns (distances, ids), each (nq, k); short rows pad with inf / -1.

        `prune=False` visits every component in full (same answer; for testing). `view`
        lets a caller reuse a snapshot it already read.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        nq = len(x)
        comps, dead, _ = view if view is not None else self._view
        if not comps:
            return np.full((nq, k), np.inf), np.full((nq, k), -1, dtype=np.int64)

        if len(comps) == 1 and dead.size == 0 and comps[0][2] is not None \
                and k <= comps[0][2].n:
            # one run, nothing deleted: the state after fit() or consolidate(). Go
            # straight to cKDTree; the generic path costs ~30 us of dispatch per call.
            _, g, tree = comps[0]
            d, idx = tree.query(x, k=k, workers=workers)
            return d.reshape(nq, k), g[idx.reshape(nq, k)]

        best_d, best_i = self._topk(comps[0], x, k, workers, dead)
        if len(comps) == 1:
            return best_d, best_i

        r = best_d[:, k - 1]                      # bound from the largest run
        can_prune = prune and np.isfinite(r).any()
        chunked = nq >= 4 * prune_chunks
        # small batches get one shared bound, the largest k-th distance; for a single
        # query that is its own exact bound
        r_all = float(r.max()) if can_prune and not chunked else np.inf
        ds, ids = [], []
        for comp in comps[1:]:
            if can_prune and comp[2] is not None and chunked:
                d, gid = self._bounded_topk(comp, x, k, workers, dead, r, prune_chunks)
            else:
                d, gid = self._topk(comp, x, k, workers, dead, bound=r_all)
            ds.append(d)
            ids.append(gid)

        cd = np.hstack(ds) if len(ds) > 1 else ds[0]
        ci = np.hstack(ids) if len(ids) > 1 else ids[0]
        if nq == 1:
            # min over the whole row: cd is several components side by side
            if cd[0].min() <= r[0]:
                return _select_k(np.hstack([best_d, cd]), np.hstack([best_i, ci]), k)
            return best_d, best_i
        hit = cd[:, 0] <= r if len(ds) == 1 else cd.min(axis=1) <= r
        if hit.any():
            md, mi = _select_k(np.hstack([best_d[hit], cd[hit]]),
                               np.hstack([best_i[hit], ci[hit]]), k)
            best_d[hit] = md
            best_i[hit] = mi
        return best_d, best_i

    # ------------------------------------------------------------------ misc
    @property
    def n_components(self):
        """How many components a query visits."""
        return len(self.runs) + (1 if self._micro_n else 0)

    def run_sizes(self):
        return [len(g) for _, g, _ in self.runs]

    def payload(self, pid):
        return self._payload.get(int(pid))

    def _stored(self):
        """Points physically held, tombstoned or not."""
        return sum(len(g) for _, g, _ in self.runs) + self._micro_n

    def __len__(self):
        return self._stored() - len(self._dead)

    def __repr__(self):
        return (f"DynamicKDTree(n={len(self)}, dim={self.dim}, T={self.size_ratio}, "
                f"b={self.micro_size}, components={self.n_components}, "
                f"runs={self.run_sizes()}, micro={self._micro_n})")
