# -*- coding: utf-8 -*-
"""Static benchmark, ann-benchmarks protocol: held-out queries, k=10, single thread,
recall@10 against float64 brute-force ground truth, throughput as queries per second.

Every method answers the same Mahalanobis question. Methods that need whitened input get
the same float64-whitened points; that preprocessing time is included in their build time.

    python bench_static.py [dataset ...]

Writes CSV, a markdown summary and plots to results/ next to this script:
    static_<dataset>.csv, static_summary.md, static_qps.png, static_recall_qps.png
"""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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
    from sklearn.neighbors import BallTree, KDTree
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

K = 10
N_TEST = 10_000
N_SINGLE = 2_000            # single-query loop length
N_MAX = 500_000
DATASETS = sys.argv[1:] or ["road3d", "household", "airquality", "gaussian-3", "gaussian-8"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def ground_truth(Xw, Qw):
    r = np.empty(len(Qw))
    for s in range(0, len(Qw), 100):
        d = cdist(Qw[s:s + 100], Xw)
        r[s:s + 100] = np.partition(d, K - 1, axis=1)[:, K - 1]
    return r


def recall(ids, Xw, Qw, r):
    ok = 0
    for i in range(len(Qw)):
        g = ids[i][ids[i] >= 0]
        if g.size:
            d = np.sqrt(((Xw[g] - Qw[i]) ** 2).sum(1))
            ok += int((d <= r[i] * (1 + 1e-9)).sum())
    return ok / (len(Qw) * K)


def whiten_params(X):
    C = np.cov(X, rowvar=False)
    return C, np.linalg.inv(np.linalg.cholesky(C))


# --------------------------------------------------------------------------- methods
# each returns (build_seconds, batch_query_fn, single_query_fn); query fns return ids

def m_whitetree(X, Q):
    t0 = time.perf_counter()
    idx = MahalanobisIndex().fit(X)
    b = time.perf_counter() - t0
    return b, lambda: idx.kneighbors(Q, k=K, workers=1)[1], \
        lambda q: idx.kneighbors(q, k=K, workers=1)[1][0]


def m_whitetree_streamed(X, Q):
    C, _ = whiten_params(X)
    h = len(X) // 2
    t0 = time.perf_counter()
    idx = MahalanobisIndex(cov=C).fit(X[:h])
    idx.insert_many(X[h:])
    b = time.perf_counter() - t0
    return b, lambda: idx.kneighbors(Q, k=K, workers=1)[1], \
        lambda q: idx.kneighbors(q, k=K, workers=1)[1][0]


def m_ckdtree(X, Q):
    t0 = time.perf_counter()
    _, W = whiten_params(X)
    Xw = X @ W.T
    tree = cKDTree(Xw)
    b = time.perf_counter() - t0
    Qw = Q @ W.T
    return b, lambda: tree.query(Qw, k=K, workers=1)[1], \
        lambda q: tree.query(q @ W.T, k=K, workers=1)[1]


def m_sk_kdtree(X, Q):
    t0 = time.perf_counter()
    _, W = whiten_params(X)
    tree = KDTree(X @ W.T)
    b = time.perf_counter() - t0
    Qw = Q @ W.T
    return b, lambda: tree.query(Qw, k=K)[1], \
        lambda q: tree.query((q @ W.T)[None], k=K)[1][0]


def m_sk_balltree_maha(X, Q):
    t0 = time.perf_counter()
    C = np.cov(X, rowvar=False)
    tree = BallTree(X, metric="mahalanobis", VI=np.linalg.inv(C))
    b = time.perf_counter() - t0
    return b, lambda: tree.query(Q, k=K)[1], \
        lambda q: tree.query(q[None], k=K)[1][0]


def m_faiss_flat(X, Q):
    t0 = time.perf_counter()
    _, W = whiten_params(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(np.ascontiguousarray(X @ W.T, dtype=np.float32))
    b = time.perf_counter() - t0
    Q32 = np.ascontiguousarray(Q @ W.T, dtype=np.float32)
    return b, lambda: index.search(Q32, K)[1], \
        lambda q: index.search(np.ascontiguousarray((q @ W.T)[None], dtype=np.float32), K)[1][0]


def m_faiss_pca(X, Q):
    d = X.shape[1]
    t0 = time.perf_counter()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    pca = faiss.PCAMatrix(d, d, -0.5)
    pca.train(X32)
    index = faiss.IndexPreTransform(pca, faiss.IndexFlatL2(d))
    index.add(X32)
    b = time.perf_counter() - t0
    Q32 = np.ascontiguousarray(Q, dtype=np.float32)
    return b, lambda: index.search(Q32, K)[1], \
        lambda q: index.search(np.ascontiguousarray(q[None], dtype=np.float32), K)[1][0]


def m_numpy_brute(X, Q):
    t0 = time.perf_counter()
    _, W = whiten_params(X)
    Xw = X @ W.T
    b = time.perf_counter() - t0
    Qw = Q @ W.T

    def batch():
        out = np.empty((len(Qw), K), dtype=np.int64)
        for s in range(0, len(Qw), 50):
            d = cdist(Qw[s:s + 50], Xw)
            out[s:s + 50] = np.argpartition(d, K - 1, axis=1)[:, :K]
        return out

    def single(q):
        d = np.sqrt(((Xw - q @ W.T) ** 2).sum(1))
        return np.argpartition(d, K - 1)[:K]
    return b, batch, single


METHODS = [
    ("whitetree", m_whitetree),
    ("whitetree, after 50% inserted", m_whitetree_streamed),
    ("scipy cKDTree (pre-whitened)", m_ckdtree),
]
if HAVE_SK:
    METHODS += [("sklearn KDTree (pre-whitened)", m_sk_kdtree),
                ("sklearn BallTree mahalanobis", m_sk_balltree_maha)]
if HAVE_FAISS:
    METHODS += [("faiss IndexFlatL2 (pre-whitened)", m_faiss_flat),
                ("faiss PCAMatrix + IndexFlatL2", m_faiss_pca)]
METHODS += [("numpy brute force", m_numpy_brute)]


def run_dataset(name):
    X, desc = datasets.get(name, n_max=N_MAX + N_TEST)
    n_test = min(N_TEST, len(X) // 10)
    Q, X = X[:n_test], X[n_test:]
    C, W = whiten_params(X)
    Xw, Qw = X @ W.T, Q @ W.T
    print(f"\n{desc}: train {len(X):,}, test {len(Q):,}, k={K}")
    r = ground_truth(Xw, Qw)
    rows = []
    for mname, fn in METHODS:
        if mname == "numpy brute force" and len(X) > 200_000:
            n_single = 200
        else:
            n_single = min(N_SINGLE, len(Q))
        build, batch, single = fn(X, Q)
        best = np.inf
        ids = None
        for _ in range(1 if mname == "numpy brute force" else 3):
            t0 = time.perf_counter()
            ids = batch()
            best = min(best, time.perf_counter() - t0)
        rec = recall(np.asarray(ids), Xw, Qw, r)
        t0 = time.perf_counter()
        for i in range(n_single):
            single(Q[i])
        t_single = time.perf_counter() - t0
        row = dict(dataset=name, method=mname, n=len(X), d=X.shape[1], build_s=build,
                   batch_qps=len(Q) / best, single_qps=n_single / t_single, recall=rec)
        rows.append(row)
        print(f"  {mname:<34} build {build:7.2f}s  batch {row['batch_qps']:>9,.0f} q/s  "
              f"single {row['single_qps']:>8,.0f} q/s  recall {rec:.4f}")
    return rows


def main():
    all_rows = []
    for name in DATASETS:
        rows = run_dataset(name)
        all_rows += rows
        with open(os.path.join(OUT, f"static_{name}.csv"), "w", encoding="utf-8") as f:
            f.write("dataset,method,n,d,build_s,batch_qps,single_qps,recall\n")
            for r in rows:
                f.write(f"{r['dataset']},{r['method']},{r['n']},{r['d']},{r['build_s']:.4f},"
                        f"{r['batch_qps']:.1f},{r['single_qps']:.1f},{r['recall']:.4f}\n")

    # summary table
    lines = [f"Machine: {platform.processor() or platform.machine()}, {platform.system()} "
             f"{platform.release()}, Python {platform.python_version()}, single thread",
             f"Protocol: ann-benchmarks style. Held-out queries, k={K}, recall@{K} vs float64 "
             "brute force, batch QPS = best of 3, single QPS = one query at a time.", ""]
    for name in DATASETS:
        rs = [r for r in all_rows if r["dataset"] == name]
        lines += [f"### {name} (n={rs[0]['n']:,}, d={rs[0]['d']})", "",
                  "| method | build (s) | batch q/s | single q/s | recall@10 |",
                  "|---|---|---|---|---|"]
        for r in rs:
            lines.append(f"| {r['method']} | {r['build_s']:.2f} | {r['batch_qps']:,.0f} | "
                         f"{r['single_qps']:,.0f} | {r['recall']:.4f} |")
        lines.append("")
    with open(os.path.join(OUT, "static_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [m for m, _ in METHODS]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, DATASETS):
        rs = {r["method"]: r for r in all_rows if r["dataset"] == name}
        vals = [rs[m]["single_qps"] for m in methods]
        colors = ["#d62728" if m.startswith("whitetree") else "#7f7f7f" for m in methods]
        ax.barh(range(len(methods)), vals, color=colors)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods if ax is axes[0] else [""] * len(methods), fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"{name}  n={rs[methods[0]]['n']:,}  d={rs[methods[0]]['d']}", fontsize=9)
        ax.set_xlabel("single queries / s (log)", fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Exact Mahalanobis k={K}, single thread, one query at a time", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "static_qps.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = "osD^v<>Px"
    for mi, m in enumerate(methods):
        rs = [r for r in all_rows if r["method"] == m]
        ax.scatter([r["batch_qps"] for r in rs], [r["recall"] for r in rs],
                   marker=markers[mi % len(markers)], s=60,
                   color="#d62728" if m.startswith("whitetree") else None, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("batch queries / s (log)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("recall vs throughput, all datasets (ann-benchmarks style)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "static_recall_qps.png"), dpi=130)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
