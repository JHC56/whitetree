# -*- coding: utf-8 -*-
"""Streaming benchmark, big-ann-benchmarks (NeurIPS'23 streaming track) style runbook.

Sliding-window runbook: insert the first n0 points, then R rounds of
    insert B new points -> delete the B oldest -> search (Q single-point queries, then batch)
Recall is measured against exact ground truth over the live set at every search step.
Every method answers the same Mahalanobis question with the same fixed covariance.

    python bench_streaming.py [dataset ...]

Writes CSV, a markdown summary and plots to results/ next to this script:
    streaming_<dataset>.csv, streaming_summary.md, streaming_time.png
"""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
from whitetree import MahalanobisIndex  # noqa: E402

try:
    import faiss
    faiss.omp_set_num_threads(1)
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
try:
    from sklearn.neighbors import BallTree
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

K = 10
N0, B, R = 200_000, 20_000, 10          # initial size, batch, rounds
NQ_SINGLE, NQ_BATCH = 2_000, 2_000
DATASETS = sys.argv[1:] or ["road3d", "household"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def recall(ids, Xw, Qw, r):
    ok = 0
    for i in range(len(Qw)):
        g = ids[i][ids[i] >= 0]
        if g.size:
            d = np.sqrt(((Xw[g] - Qw[i]) ** 2).sum(1))
            ok += int((d <= r[i] * (1 + 1e-9)).sum())
    return ok / (len(Qw) * K)


# --------------------------------------------------------------------------- methods
# interface: insert(rows, ids) / delete(ids) / search_single(q) -> ids / search_batch(Q) -> ids
# Every method receives raw rows and the shared whitening W; whitening cost is charged to it.

class WhiteTree:
    name = "whitetree"

    def __init__(self, C, W, d):
        self.idx = MahalanobisIndex(cov=C)
        self.first = True

    def insert(self, rows, ids):
        if self.first:
            self.idx.fit(rows)
            self.first = False
        else:
            self.idx.insert_many(rows)

    def delete(self, ids):
        self.idx.delete_many(ids)

    def search_single(self, q):
        return self.idx.kneighbors(q, k=K, workers=1)[1][0]

    def search_batch(self, Q):
        return self.idx.kneighbors(Q, k=K, workers=1)[1]


class FaissFlat:
    name = "faiss IndexFlatL2 + IDMap2"

    def __init__(self, C, W, d):
        self.W = W
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))

    def insert(self, rows, ids):
        self.index.add_with_ids(np.ascontiguousarray(rows @ self.W.T, dtype=np.float32),
                                np.asarray(ids, dtype=np.int64))

    def delete(self, ids):
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search_single(self, q):
        q32 = np.ascontiguousarray((q @ self.W.T)[None], dtype=np.float32)
        return self.index.search(q32, K)[1][0]

    def search_batch(self, Q):
        return self.index.search(np.ascontiguousarray(Q @ self.W.T, dtype=np.float32), K)[1]


class RebuildCKD:
    """What people do today: keep the rows, rebuild scipy's tree before each search."""
    name = "scipy cKDTree, rebuild per search"

    def __init__(self, C, W, d):
        self.W = W
        self.pts, self.ids, self.dead = [], [], set()
        self.tree = None

    def insert(self, rows, ids):
        self.pts.append(rows @ self.W.T)
        self.ids.append(np.asarray(ids))

    def delete(self, ids):
        self.dead.update(int(i) for i in ids)

    def _rebuild(self):
        P = np.vstack(self.pts)
        G = np.concatenate(self.ids)
        keep = ~np.isin(G, np.fromiter(self.dead, dtype=np.int64, count=len(self.dead)))
        self.pts, self.ids, self.dead = [P[keep]], [G[keep]], set()
        self.tree = cKDTree(self.pts[0])

    def search_single(self, q):
        if self.tree is None:
            self._rebuild()
        return self.ids[0][self.tree.query(q @ self.W.T, k=K, workers=1)[1]]

    def search_batch(self, Q):
        if self.tree is None:
            self._rebuild()
        return self.ids[0][self.tree.query(Q @ self.W.T, k=K, workers=1)[1]]

    def round_start(self):
        self.tree = None


class RebuildBallTree(RebuildCKD):
    """sklearn's Mahalanobis path: BallTree(metric='mahalanobis', VI=...), rebuilt per search."""
    name = "sklearn BallTree mahalanobis, rebuild per search"

    def __init__(self, C, W, d):
        super().__init__(C, W, d)
        self.VI = np.linalg.inv(C)

    def insert(self, rows, ids):
        self.pts.append(np.asarray(rows))
        self.ids.append(np.asarray(ids))

    def _rebuild(self):
        P = np.vstack(self.pts)
        G = np.concatenate(self.ids)
        keep = ~np.isin(G, np.fromiter(self.dead, dtype=np.int64, count=len(self.dead)))
        self.pts, self.ids, self.dead = [P[keep]], [G[keep]], set()
        self.tree = BallTree(self.pts[0], metric="mahalanobis", VI=self.VI)

    def search_single(self, q):
        if self.tree is None:
            self._rebuild()
        return self.ids[0][self.tree.query(q[None], k=K)[1][0]]

    def search_batch(self, Q):
        if self.tree is None:
            self._rebuild()
        return self.ids[0][self.tree.query(Q, k=K)[1]]


class NumpyBrute(RebuildCKD):
    name = "numpy brute force"

    def _rebuild(self):
        P = np.vstack(self.pts)
        G = np.concatenate(self.ids)
        keep = ~np.isin(G, np.fromiter(self.dead, dtype=np.int64, count=len(self.dead)))
        self.pts, self.ids, self.dead = [P[keep]], [G[keep]], set()
        self.tree = True

    def search_single(self, q):
        if self.tree is None:
            self._rebuild()
        d = ((self.pts[0] - q @ self.W.T) ** 2).sum(1)
        return self.ids[0][np.argpartition(d, K - 1)[:K]]

    def search_batch(self, Q):
        if self.tree is None:
            self._rebuild()
        from scipy.spatial.distance import cdist
        out = np.empty((len(Q), K), dtype=np.int64)
        Qw = Q @ self.W.T
        for s in range(0, len(Q), 50):
            d = cdist(Qw[s:s + 50], self.pts[0])
            out[s:s + 50] = self.ids[0][np.argpartition(d, K - 1, axis=1)[:, :K]]
        return out


METHODS = [WhiteTree]
if HAVE_FAISS:
    METHODS.append(FaissFlat)
METHODS.append(RebuildCKD)
if HAVE_SK:
    METHODS.append(RebuildBallTree)
METHODS.append(NumpyBrute)


def run_dataset(name):
    X, desc = datasets.get(name, n_max=N0 + B * R + NQ_BATCH)
    need = N0 + B * R + NQ_BATCH
    if len(X) < need:
        raise SystemExit(f"{name}: need {need:,} rows, have {len(X):,}")
    Q, X = X[:NQ_BATCH], X[NQ_BATCH:]
    C = np.cov(X, rowvar=False)
    W = np.linalg.inv(np.linalg.cholesky(C))
    Xw, Qw = X @ W.T, Q @ W.T
    d = X.shape[1]
    print(f"\n{desc}: n0={N0:,}, {R} rounds of insert {B:,} / delete {B:,} / "
          f"search {NQ_SINGLE:,} single + {NQ_BATCH:,} batch")

    # ground truth per round over the live window
    truths = []
    for rnd in range(R + 1):
        lo, hi = rnd * B, N0 + rnd * B
        live = np.arange(lo, hi)
        tree = cKDTree(Xw[live])
        dist, _ = tree.query(Qw, k=K, workers=-1)
        truths.append(dist[:, K - 1])

    rows = []
    for cls in METHODS:
        m = cls(C, W, d)
        t_ins = t_del = t_single = t_batch = 0.0
        recs_s, recs_b = [], []

        t0 = time.perf_counter()
        m.insert(X[:N0], np.arange(N0))
        t_ins += time.perf_counter() - t0
        for rnd in range(R):
            lo, hi = N0 + rnd * B, N0 + (rnd + 1) * B
            t0 = time.perf_counter()
            m.insert(X[lo:hi], np.arange(lo, hi))
            t_ins += time.perf_counter() - t0
            t0 = time.perf_counter()
            m.delete(np.arange(rnd * B, (rnd + 1) * B))
            t_del += time.perf_counter() - t0
            if hasattr(m, "round_start"):
                m.round_start()
            r = truths[rnd + 1]
            out = np.empty((NQ_SINGLE, K), dtype=np.int64)
            t0 = time.perf_counter()
            for i in range(NQ_SINGLE):
                out[i] = m.search_single(Q[i])
            t_single += time.perf_counter() - t0
            recs_s.append(recall(out, Xw, Qw[:NQ_SINGLE], r[:NQ_SINGLE]))
            t0 = time.perf_counter()
            outb = np.asarray(m.search_batch(Q))
            t_batch += time.perf_counter() - t0
            recs_b.append(recall(outb, Xw, Qw, r))
        total = t_ins + t_del + t_single
        row = dict(dataset=name, method=m.name, insert_s=t_ins, delete_s=t_del,
                   search_single_s=t_single, search_batch_s=t_batch, total_s=total,
                   recall_single=float(np.mean(recs_s)), recall_batch=float(np.mean(recs_b)),
                   recall_min=float(min(recs_s + recs_b)))
        rows.append(row)
        print(f"  {m.name:<48} insert {t_ins:6.2f}s  delete {t_del:6.2f}s  "
              f"search(single) {t_single:6.2f}s  search(batch) {t_batch:6.2f}s  "
              f"total {total:6.2f}s  recall {row['recall_single']:.4f}/{row['recall_batch']:.4f}")
    return rows


def main():
    all_rows = []
    for name in DATASETS:
        rows = run_dataset(name)
        all_rows += rows
        with open(os.path.join(OUT, f"streaming_{name}.csv"), "w", encoding="utf-8") as f:
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                                 for k in keys) + "\n")

    lines = [f"Machine: {platform.processor() or platform.machine()}, {platform.system()} "
             f"{platform.release()}, Python {platform.python_version()}, single thread",
             f"Runbook (big-ann-benchmarks streaming style): insert {N0:,}, then {R} rounds of "
             f"insert {B:,} / delete oldest {B:,} / search ({NQ_SINGLE:,} single-point queries, "
             f"then {NQ_BATCH:,} as one batch). k={K}. Recall vs exact ground truth over the "
             "live window at each step. 'total' = insert + delete + single-point search.", ""]
    for name in DATASETS:
        rs = [r for r in all_rows if r["dataset"] == name]
        lines += [f"### {name}", "",
                  "| method | insert (s) | delete (s) | search single (s) | search batch (s) | "
                  "total (s) | recall single | recall batch |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            lines.append(f"| {r['method']} | {r['insert_s']:.2f} | {r['delete_s']:.2f} | "
                         f"{r['search_single_s']:.2f} | {r['search_batch_s']:.2f} | "
                         f"{r['total_s']:.2f} | {r['recall_single']:.4f} | "
                         f"{r['recall_batch']:.4f} |")
        lines.append("")
    with open(os.path.join(OUT, "streaming_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [c.name for c in METHODS]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, DATASETS):
        rs = {r["method"]: r for r in all_rows if r["dataset"] == name}
        y = np.arange(len(names))
        ins = [rs[m]["insert_s"] for m in names]
        dele = [rs[m]["delete_s"] for m in names]
        srch = [rs[m]["search_single_s"] for m in names]
        ax.barh(y, ins, color="#1f77b4", label="insert")
        ax.barh(y, dele, left=ins, color="#ff7f0e", label="delete")
        ax.barh(y, srch, left=np.add(ins, dele), color="#2ca02c", label="search (single)")
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("seconds, whole runbook (log)", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Sliding-window runbook: n0={N0:,}, {R} x (insert {B:,}, delete {B:,}, "
                 f"{NQ_SINGLE:,} queries)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "streaming_time.png"), dpi=130)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
