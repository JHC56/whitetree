# -*- coding: utf-8 -*-
"""One writer thread, several readers querying the whole time.

Checked: no torn reads, and no point going missing. A query issued when m points had
finished inserting must return a distance no larger than the true nearest over those m.
"""
from __future__ import annotations

import threading
import traceback

import numpy as np
from scipy.spatial import cKDTree

from whitetree import DynamicKDTree

FAIL = []


def check(name, cond, extra=""):
    print(f'  {"PASS" if cond else "FAIL"}  {name}{("  " + extra) if extra else ""}')
    if not cond:
        FAIL.append(name)


DIM, K = 3, 5
N = 60_000
N_PROBES = 64
N_READERS = 3

rng = np.random.RandomState(4)
PTS = rng.randn(N, DIM)
PROBES = rng.randn(N_PROBES, DIM)

# true_nn[p, m-1] = distance from probe p to the nearest of PTS[:m]
_D = np.sqrt(((PTS[None, :, :] - PROBES[:, None, :]) ** 2).sum(-1))
TRUE_NN = np.minimum.accumulate(_D, axis=1)
del _D

t = DynamicKDTree(DIM)
errors, torn, missing = [], [], []
stop = threading.Event()
counts = [0] * N_READERS


def writer():
    try:
        for p in PTS:
            t.insert(p)
    except Exception:
        errors.append("writer: " + traceback.format_exc())
    finally:
        stop.set()


def reader(slot, allow_missing=False):
    try:
        qi = 0
        while not stop.is_set():
            p = qi % N_PROBES
            q = PROBES[p]
            # Every id below m-1 belongs to an insert() call that already returned, so those
            # points must be visible no matter what the writer is doing right now.
            m = t._next_id - 1
            d, i = t.query(q, k=K, workers=1)
            qi += 1
            live = i[0] >= 0
            ids = i[0][live]
            if ids.size:
                if ids.max() >= N:
                    torn.append(f"id out of range: {ids.max()}")
                    continue
                real = np.sqrt(((PTS[ids] - q) ** 2).sum(1))
                if not np.array_equal(real, d[0][live]):
                    torn.append(f"distance/id mismatch, max diff "
                                f"{np.abs(real - d[0][live]).max():.3e}")
                dd = d[0][live]
                if dd.size > 1 and not np.all(dd[:-1] <= dd[1:]):
                    torn.append("distances not sorted")
            if not allow_missing and m >= 1:
                got = d[0][0] if live.any() else np.inf
                if got > TRUE_NN[p, m - 1] + 1e-12:
                    missing.append(f"m={m}: got {got:.6f}, true {TRUE_NN[p, m-1]:.6f}")
        counts[slot] = qi
    except Exception:
        errors.append(f"reader {slot}: " + traceback.format_exc())


print(f"1. {N_READERS} readers querying while one writer inserts {N:,} points")
threads = [threading.Thread(target=reader, args=(s,)) for s in range(N_READERS)]
w = threading.Thread(target=writer)
for th in threads:
    th.start()
w.start()
w.join()
for th in threads:
    th.join()

print(f"     queries completed during the run: {sum(counts):,}")
check("no exceptions in any thread", not errors, errors[0][:400] if errors else "")
check("no torn reads", not torn, f"{len(torn)} incidents: {torn[:2]}")
check("no point went missing mid-merge", not missing,
      f"{len(missing)} incidents, e.g. {missing[:3]}")
check("readers actually ran", sum(counts) > 1000, str(counts))

print("\n2. After the writer finishes, the index is exactly correct")
QS = rng.randn(500, DIM)
d_ref, i_ref = cKDTree(PTS).query(QS, k=K)
d, i = t.query(QS, k=K, workers=1)
ok = (np.abs(d - d_ref).max() == 0.0
      and all(set(a) == set(b) for a, b in zip(i.tolist(), i_ref.tolist())))
check("matches a static cKDTree over all points", ok,
      f"max dist err {np.abs(d - d_ref).max():.2e}")
check("every point is present", len(t) == N, f"len={len(t)}")

print("\n3. Deleting while readers query")
stop.clear()
torn.clear()
del_errors = []


def deleter():
    try:
        for pid in range(0, N, 3):
            t.delete(pid)
    except Exception:
        del_errors.append("deleter: " + traceback.format_exc())
    finally:
        stop.set()


# Deletions legitimately move neighbours further away, so the missing-point invariant does
# not apply here; torn reads still must not happen.
threads = [threading.Thread(target=reader, args=(s, True)) for s in range(N_READERS)]
dth = threading.Thread(target=deleter)
for th in threads:
    th.start()
dth.start()
dth.join()
for th in threads:
    th.join()
check("no exceptions while deleting", not del_errors, del_errors[0][:400] if del_errors else "")
check("no torn reads while deleting", not torn, f"{len(torn)} incidents: {torn[:2]}")

alive = np.array([p for p in range(N) if p % 3 != 0])
dr, ir = cKDTree(PTS[alive]).query(QS[:300], k=K)
d2, i2 = t.query(QS[:300], k=K, workers=1)
ok = (np.abs(d2 - dr).max() == 0.0
      and all(set(a) == set(b) for a, b in zip(alive[ir].tolist(), i2.tolist())))
check("correct over the survivors afterwards", ok)
check("no deleted id leaked", not (set(i2.ravel().tolist()) & set(range(0, N, 3))))

print("\n4. Re-fitting the whitening while readers query")
# A re-fit moves every point to new coordinates. A reader that whitened its query under the
# old matrix and then searched an index already rebuilt under the new one gets a *wrong*
# answer, not a stale one -- the two coordinate systems simply do not correspond. So alternate
# between two genuinely different covariances and require every answer to be the true kNN
# under one of them. Anything else means the two halves of a query came from different worlds.
from whitetree import MahalanobisIndex  # noqa: E402

rng2 = np.random.RandomState(21)
covA = np.array([[6.0, 4.0, 0.0], [4.0, 6.0, 0.0], [0.0, 0.0, 1.0]])
covB = np.array([[6.0, -4.5, 1.0], [-4.5, 6.0, 0.0], [1.0, 0.0, 4.0]])
PTS2 = rng2.multivariate_normal(np.zeros(3), covA, size=30_000)
PROBES2 = rng2.multivariate_normal(np.zeros(3), covA, size=32)

truth = {}
for tag, cv in (("A", covA), ("B", covB)):
    W = np.linalg.inv(np.linalg.cholesky(cv))
    ids = cKDTree(PTS2 @ W.T).query(PROBES2 @ W.T, k=K, workers=-1)[1]
    truth[tag] = [frozenset(row) for row in ids.tolist()]

mi = MahalanobisIndex(cov=covA).fit(PTS2)
mismatched, n_reads = [], [0] * N_READERS
stop.clear()


def refitter():
    try:
        for i in range(40):
            mi.refit(cov=covB if i % 2 == 0 else covA)
    except Exception:
        errors.append("refitter: " + traceback.format_exc())
    finally:
        stop.set()


def probe_reader(slot):
    try:
        c = 0
        while not stop.is_set():
            p = c % len(PROBES2)
            c += 1
            got = frozenset(mi.kneighbors(PROBES2[p], k=K, workers=1)[1][0].tolist())
            if got != truth["A"][p] and got != truth["B"][p]:
                mismatched.append(p)
        n_reads[slot] = c
    except Exception:
        errors.append(f"probe reader {slot}: " + traceback.format_exc())


errors.clear()
threads = [threading.Thread(target=probe_reader, args=(s,)) for s in range(N_READERS)]
rt = threading.Thread(target=refitter)
for th in threads:
    th.start()
rt.start()
rt.join()
for th in threads:
    th.join()

print(f"     queries during 40 re-fits: {sum(n_reads):,}")
check("no exceptions while re-fitting", not errors, errors[0][:400] if errors else "")
check("every answer matches one whitening or the other", not mismatched,
      f"{len(mismatched)} answers matched neither")
check("readers actually ran", sum(n_reads) > 100, str(n_reads))

print("\n" + "=" * 60)
print("all checks passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
raise SystemExit(1 if FAIL else 0)
