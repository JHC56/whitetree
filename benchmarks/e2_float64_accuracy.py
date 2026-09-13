# -*- coding: utf-8 -*-
"""Recall@10 against a float64 brute-force reference: ours vs FAISS (float32).

Ties within the k-th distance count as correct. Columns:
    ours            MahalanobisIndex, half bulk-loaded, half inserted
    faiss-pre       IndexFlatL2 on the same float64-whitened points, cast to float32
    faiss-pre-ctr   same, centred before the cast
    faiss-pca       IndexPreTransform(PCAMatrix(d, d, -0.5)), FAISS defaults
    pca-full        same, covariance from all n points (max_points_per_d lifted)
    pca-full-ctr    same, data centred in float64 first

    pip install faiss-cpu
    python e2_float64_accuracy.py
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.distance import cdist

from whitetree import MahalanobisIndex

try:
    import faiss
    faiss.omp_set_num_threads(1)
except ImportError:
    raise SystemExit("pip install faiss-cpu")

K = 10
NQ = 2_000
rng = np.random.RandomState(2)


def ground_truth(Xw, Qw, k):
    """Float64 brute force; returns the k-th distance per query (the recall cut)."""
    r = np.empty(len(Qw))
    for s in range(0, len(Qw), 200):
        d = cdist(Qw[s:s + 200], Xw)
        r[s:s + 200] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return r


def recall(ids, Xw, Qw, r):
    """Fraction of returned ids whose true float64 distance is within the k-th cut."""
    ok = 0
    for i in range(len(Qw)):
        g = ids[i]
        g = g[g >= 0]
        if g.size == 0:
            continue
        d = np.sqrt(((Xw[g] - Qw[i]) ** 2).sum(1))
        ok += int((d <= r[i] * (1 + 1e-9)).sum())
    return ok / (len(Qw) * K)


def run_ours(X, Q, cov):
    """Half bulk-loaded, half inserted. `cov` is passed in so the metric matches the reference."""
    n = len(X)
    idx = MahalanobisIndex(cov=cov, dynamic=True, cov_window=None).fit(X[: n // 2])
    idx.insert_many(X[n // 2:])
    assert idx.backend.n_components >= 2, "must test the multi-run state"
    _, ids = idx.kneighbors(Q, k=K, workers=1)
    return ids, idx.L_inv


def run_faiss_pre(Xw, Qw, centre):
    mu = Xw.mean(0) if centre else 0.0
    index = faiss.IndexFlatL2(Xw.shape[1])
    index.add(np.ascontiguousarray(Xw - mu, dtype=np.float32))
    _, ids = index.search(np.ascontiguousarray(Qw - mu, dtype=np.float32), K)
    return ids


def run_faiss_pca(X, Q, full=False, centre=False):
    """FAISS's own whitening. Its covariance comes from a 1000*d subsample in float32 by
    default; `full` lifts the subsample, `centre` subtracts the mean in float64 first."""
    d = X.shape[1]
    mu = X.mean(0) if centre else 0.0
    X32 = np.ascontiguousarray(X - mu, dtype=np.float32)
    Q = Q - mu
    pca = faiss.PCAMatrix(d, d, -0.5)
    if full:
        pca.max_points_per_d = len(X)
    pca.train(X32)
    index = faiss.IndexPreTransform(pca, faiss.IndexFlatL2(d))
    index.add(X32)
    _, ids = index.search(np.ascontiguousarray(Q, dtype=np.float32), K)
    return ids


def evaluate(name, X, Q):
    n, d = X.shape
    t0 = time.perf_counter()
    # reference metric independent of the library: float64 np.cov, Cholesky, no ridge
    C = np.cov(X, rowvar=False)
    L_inv = np.linalg.inv(np.linalg.cholesky(C))
    Xw, Qw = X @ L_inv.T, Q @ L_inv.T
    r = ground_truth(Xw, Qw, K)
    ids_ours, _ = run_ours(X, Q, C)
    res = {
        "ours": recall(ids_ours, Xw, Qw, r),
        "faiss-pre": recall(run_faiss_pre(Xw, Qw, False), Xw, Qw, r),
        "faiss-pre-ctr": recall(run_faiss_pre(Xw, Qw, True), Xw, Qw, r),
        "faiss-pca": recall(run_faiss_pca(X, Q), Xw, Qw, r),
        "pca-full": recall(run_faiss_pca(X, Q, full=True), Xw, Qw, r),
        "pca-full-ctr": recall(run_faiss_pca(X, Q, full=True, centre=True), Xw, Qw, r),
    }
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    print(f"{name:<34}{n:>9,}{d:>4}{ev[-1] / ev[0]:>9.1e}"
          + "".join(f"{res[k]:>14.4f}" for k in res)
          + f"{time.perf_counter() - t0:>6.0f}s")
    return res


def synthetic(n, d, cond, offset=0.0, scale=1.0):
    """Gaussian with a covariance of the given condition number, then shifted/scaled."""
    A = rng.randn(d, d)
    U, _ = np.linalg.qr(A)
    ev = np.logspace(0, np.log10(cond), d)
    C = (U * ev) @ U.T
    L = np.linalg.cholesky(C)
    Z = rng.randn(n + NQ, d) @ L.T
    Z = Z * scale + offset
    return Z[:n], Z[n:]


print(f"recall@{K} against float64 brute force, {NQ:,} queries, k-th distance as the cut\n")
print(f"{'dataset':<34}{'n':>9}{'d':>4}{'cond':>9}"
      f"{'ours':>14}{'faiss-pre':>14}{'faiss-pre-ctr':>14}{'faiss-pca':>14}"
      f"{'pca-full':>14}{'pca-full-ctr':>14}{'time':>7}")

N = 100_000
evaluate("gaussian d=3 cond=10", *synthetic(N, 3, 1e1))
evaluate("gaussian d=8 cond=1e4", *synthetic(N, 8, 1e4))
evaluate("gaussian d=8 cond=1e8", *synthetic(N, 8, 1e8))
evaluate("gaussian d=3 offset=1e4 (DC bias)", *synthetic(N, 3, 1e1, offset=1e4))
evaluate("gaussian d=3 offset=1e6", *synthetic(N, 3, 1e1, offset=1e6))
evaluate("gaussian d=3 scale=1e-4", *synthetic(N, 3, 1e1, scale=1e-4))

try:
    from uci_data import load_household, load_airquality
    Xh, _ = load_household(max_rows=N + NQ + 50_000)
    sel = rng.choice(len(Xh), N + NQ, replace=False)
    evaluate("UCI household power (7 ch)", Xh[sel[:N]], Xh[sel[N:]])
    Xa, _ = load_airquality()
    qa = rng.choice(len(Xa), NQ, replace=False)
    mask = np.ones(len(Xa), bool)
    mask[qa] = False
    evaluate("UCI air quality (12 ch)", Xa[mask], Xa[qa])
except FileNotFoundError as e:
    print("real data skipped:", e)

print("""
faiss-pre: same whitened points, float32 storage.  faiss-pca: FAISS's own whitening.
'ours' below 1.0 at high condition number comes from the ridge reg=1e-9 (reg=1e-12 gives 1.0).""")
