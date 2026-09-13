# -*- coding: utf-8 -*-
"""Insert n points one at a time, then 20,000 single-point queries: ours vs faiss.IndexFlatL2.

    pip install faiss-cpu
    python bench_vs_faiss_scale.py
"""
from __future__ import annotations

import time

import numpy as np

from whitetree import DynamicKDTree

try:
    import faiss
    faiss.omp_set_num_threads(1)
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

DIM, K = 3, 5
N_QUERIES = 20_000
NS = [20_000, 50_000, 100_000, 200_000, 500_000]

rng = np.random.RandomState(5)
BIG = rng.randn(max(NS), DIM)
QS = rng.randn(N_QUERIES, DIM)


def time_ours(n):
    """Insert n points one at a time, then run N_QUERIES single-point queries."""
    t = DynamicKDTree(DIM)
    t0 = time.perf_counter()
    for p in BIG[:n]:
        t.insert(p)
    t_ins = time.perf_counter() - t0
    out = np.empty((N_QUERIES, K), dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(N_QUERIES):
        _, ix = t.query(QS[i], k=K, workers=1)
        out[i] = ix[0]
    return t_ins, time.perf_counter() - t0, out


def time_faiss(n):
    index = faiss.IndexFlatL2(DIM)
    pts32 = np.ascontiguousarray(BIG[:n], dtype=np.float32)
    qs32 = np.ascontiguousarray(QS, dtype=np.float32)
    t0 = time.perf_counter()
    for s in range(0, n, 32):                     # batched add, as anyone would write it
        index.add(pts32[s:s + 32])
    t_ins = time.perf_counter() - t0
    out = np.empty((N_QUERIES, K), dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(N_QUERIES):
        _, ix = index.search(qs32[i:i + 1], K)
        out[i] = ix[0]
    return t_ins, time.perf_counter() - t0, out


print(f"{N_QUERIES:,} single-point queries after n inserts,  dim={DIM}, k={K}, 1 thread")
print("FAISS available:", HAVE_FAISS)
if not HAVE_FAISS:
    raise SystemExit("pip install faiss-cpu to run this comparison")
print()
print(f"{'n':>9}" + f"{'ours ins':>10}{'ours qry':>10}{'ours tot':>10}"
      + f"{'faiss ins':>11}{'faiss qry':>11}{'faiss tot':>11}"
      + f"{'per-query':>11}{'winner':>10}{'mismatch':>10}")
for n in NS:
    oi, oq, o_out = time_ours(n)
    fi, fq, f_out = time_faiss(n)
    ot, ft = oi + oq, fi + fq
    mism = int((np.sort(o_out, axis=1) != np.sort(f_out, axis=1)).any(axis=1).sum())
    print(f"{n:>9,}{oi:>10.2f}{oq:>10.2f}{ot:>10.2f}"
          f"{fi:>11.2f}{fq:>11.2f}{ft:>11.2f}"
          f"{fq/oq:>10.2f}x{('ours' if ot < ft else 'faiss'):>10}{mism:>10}")

print("\n'per-query': one FAISS query / one of ours. 'mismatch': queries whose neighbour set")
print("differs (FAISS stores float32, so near-ties can resolve differently).")
