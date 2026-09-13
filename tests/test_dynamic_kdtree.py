# -*- coding: utf-8 -*-
"""Correctness tests. These matter more than any benchmark: the whole selling point is
that results are bit-for-bit identical to a single static cKDTree."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from whitetree import DynamicKDTree
from whitetree import MahalanobisIndex

FAIL = []


def check(name, cond, extra=""):
    print(f'  {"PASS" if cond else "FAIL"}  {name}{("  " + extra) if extra else ""}')
    if not cond:
        FAIL.append(name)


def same_knn(a_idx, b_idx, a_d, b_d, tol=1e-9):
    """Distances must match exactly; ids compared as sets because ties are arbitrary."""
    if not np.allclose(a_d, b_d, atol=tol):
        return False
    return all(set(x) == set(y) for x, y in zip(a_idx.tolist(), b_idx.tolist()))


def invariant(t):
    """Runs sorted largest-first with size(run i) >= T * size(run i+1)."""
    s = t.run_sizes()
    return all(s[i] >= t.size_ratio * s[i + 1] for i in range(len(s) - 1))


print("1. Matches a static cKDTree after insertion (sweeping T and micro_size)")
rng = np.random.RandomState(0)
X = rng.randn(3000, 4)
Q = rng.randn(300, 4)
d_ref, i_ref = cKDTree(X).query(Q, k=5)
for T in (2, 4, 8, 32, 128):
    for b in (1, 32, 256):
        t = DynamicKDTree(4, size_ratio=T, micro_size=b)
        t.insert_many(X)
        d, i = t.query(Q, k=5, workers=1)
        ok = same_knn(i, i_ref, d, d_ref) and invariant(t)
        check(f"insert_many  T={T:<4} micro={b:<4}", ok,
              f"components={t.n_components} runs={t.run_sizes()} micro={t._micro_n}")

print("\n2. Same result when points go in one at a time")
t = DynamicKDTree(4, size_ratio=8, micro_size=16)
for p in X:
    t.insert(p)
d, i = t.query(Q, k=5, workers=1)
check("insert() x3000", same_knn(i, i_ref, d, d_ref), repr(t))
check("run invariant holds", invariant(t), str(t.run_sizes()))
check("no points lost", sum(t.run_sizes()) + t._micro_n == 3000)

print("\n3. Deletion: nothing leaks, and the answer matches the surviving points")
t2 = DynamicKDTree(4, size_ratio=8, micro_size=16, compact_ratio=0.9)
t2.insert_many(X[:1000])
dead = list(range(0, 400))
for pid in dead:
    t2.delete(pid)
d2, i2 = t2.query(Q, k=5, workers=1)
leaked = set(i2.ravel().tolist()) & set(dead)
check("no deleted id in results", not leaked, f"{len(leaked)} leaked")
check("all k slots filled (no -1)", (i2 >= 0).all())
alive = np.arange(400, 1000)
dr, ir = cKDTree(X[alive]).query(Q, k=5)
check("matches ground truth over survivors", same_knn(alive[ir], i2, dr, d2))

t2m = DynamicKDTree(4, size_ratio=8, micro_size=16, compact_ratio=0.9)
t2m.insert_many(X[:1000])
n_removed = t2m.delete_many(dead + [5, 5000, -1])      # duplicates and unknown ids ignored
check("delete_many returns the number removed", n_removed == 400, f"got {n_removed}")
d2m, i2m = t2m.query(Q, k=5, workers=1)
check("delete_many == repeated delete", same_knn(i2m, i2, d2m, d2))
check("delete_many again removes nothing", t2m.delete_many(dead) == 0)
check("len after delete_many", len(t2m) == 600, f"len={len(t2m)}")
t2c = DynamicKDTree(4, size_ratio=8, micro_size=16, compact_ratio=0.3)
t2c.insert_many(X[:1000])
t2c.delete_many(dead)                                   # crosses compact_ratio -> compacts
check("delete_many triggers compaction", len(t2c._dead) == 0 and len(t2c) == 600)
d2c, i2c = t2c.query(Q, k=5, workers=1)
check("correct after compaction via delete_many", same_knn(alive[ir], i2c, dr, d2c))

# tombstoned points are dropped whenever their run gets rebuilt by a later insert
t2r = DynamicKDTree(4, size_ratio=4, micro_size=8, compact_ratio=0.99)
t2r.insert_many(X[:1000])
t2r.delete_many(range(0, 1000, 2))
n_dead_before = len(t2r._dead)
t2r.insert_many(X[1000:3000])                           # forces merges of the old runs
check("rebuilt runs shed their tombstones", len(t2r._dead) < n_dead_before,
      f"{n_dead_before} -> {len(t2r._dead)}")
check("len() consistent after shedding", len(t2r) == 500 + 2000, f"len={len(t2r)}")
check("delete of a shed id is refused", t2r.delete(0) is False)
al2 = np.concatenate([np.arange(1, 1000, 2), np.arange(1000, 3000)])
dr3, ir3 = cKDTree(X[al2]).query(Q, k=5)
d3r, i3r = t2r.query(Q, k=5, workers=1)
check("correct after shedding", same_knn(al2[ir3], i3r, dr3, d3r))
check("no shed id leaked", not (set(i3r.ravel().tolist()) & set(range(0, 1000, 2))))

print("\n4. compact() keeps both the invariant and the answers")
t2.compact()
check("no tombstones left", len(t2._dead) == 0)
check("rebuilt into a single run", len(t2.runs) == 1, str(t2.run_sizes()))
# Counting live points as `_next_id - len(dead)` breaks here: compaction drops the dead
# points and clears the tombstones while `_next_id` keeps climbing. It reported 1000.
check("len() is correct after compaction", len(t2) == 600,
      f"len={len(t2)}, stored={t2._stored()}")
d3, i3 = t2.query(Q, k=5, workers=1)
check("answers unchanged by compaction", same_knn(i3, i2, d3, d2))
t2.insert_many(rng.randn(500, 4))
check("inserting after compaction keeps the invariant", invariant(t2), str(t2.run_sizes()))

print("\n5. bulk_load(emulate_inserts=True) reproduces the sequential-insert layout")
a = DynamicKDTree(4, size_ratio=8, micro_size=16)
a.insert_many(X)
b_ = DynamicKDTree(4, size_ratio=8, micro_size=16).bulk_load(X, emulate_inserts=True)
check("identical run layout", a.run_sizes() == b_.run_sizes(),
      f"{a.run_sizes()} vs {b_.run_sizes()}")
db, ib = b_.query(Q, k=5, workers=1)
check("and identical answers", same_knn(ib, i_ref, db, d_ref))
c = DynamicKDTree(4).bulk_load(X)
check("plain bulk_load gives one run", c.n_components == 1, repr(c))
dc, ic = c.query(Q, k=5, workers=1)
check("one-run state is correct too", same_knn(ic, i_ref, dc, d_ref))

print("\n6. consolidate() folds every run into one")
cc = DynamicKDTree(4, size_ratio=8, micro_size=16)
cc.insert_many(X)
before = cc.n_components
dcb, icb = cc.query(Q, k=5, workers=1)
cc.consolidate()
check("single component afterwards", cc.n_components == 1, f"{before} -> {cc.n_components}")
dca, ica = cc.query(Q, k=5, workers=1)
check("still correct", same_knn(ica, i_ref, dca, d_ref))
check("answers unchanged", same_knn(ica, icb, dca, dcb))
cc.insert_many(rng.randn(400, 4))
check("inserts still work afterwards", invariant(cc), str(cc.run_sizes()))
cd_ = DynamicKDTree(4, size_ratio=8, micro_size=16)
cd_.insert_many(X[:500])
for pid in range(100):
    cd_.delete(pid)
cd_.consolidate()
dcc, icc = cd_.query(Q, k=5, workers=1)
al = np.arange(100, 500)
dr2, ir2 = cKDTree(X[al]).query(Q, k=5)
check("correct when consolidating with tombstones", same_knn(al[ir2], icc, dr2, dcc))
check("consolidate also clears tombstones", len(cd_._dead) == 0)

print("\n7. Edge cases")
e = DynamicKDTree(4)
de, ie = e.query(Q[:3], k=5)
check("empty index", np.isinf(de).all() and (ie == -1).all())
s = DynamicKDTree(4, micro_size=2)
s.insert_many(X[:3])
ds, is_ = s.query(Q[:3], k=5)
dref3, iref3 = cKDTree(X[:3]).query(Q[:3], k=3)
check("k > n pads with inf / -1", np.isfinite(ds[:, :3]).all() and np.isinf(ds[:, 3:]).all())
check("k > n returns the right points", same_knn(is_[:, :3], iref3, ds[:, :3], dref3))
check("a single query vector works",
      DynamicKDTree(4).bulk_load(X).query(Q[0], k=5)[0].shape == (1, 5))
try:
    DynamicKDTree(4, size_ratio=1)
    bad_ok = False
except ValueError:
    bad_ok = True
check("size_ratio < 2 is rejected", bad_ok)

print("\n8. Distance-bound pruning agrees with exhaustive search")
r2 = np.random.RandomState(7)
Xb = r2.randn(20000, 3)
Qb = r2.randn(2000, 3)
db_ref, ib_ref = cKDTree(Xb).query(Qb, k=5)
for T, b in [(64, 32), (32, 32), (8, 32), (2, 1)]:
    tt = DynamicKDTree(3, size_ratio=T, micro_size=b)
    tt.insert_many(Xb)
    d_on, i_on = tt.query(Qb, k=5, workers=1, prune=True)
    d_off, i_off = tt.query(Qb, k=5, workers=1, prune=False)
    check(f"prune on == prune off  (T={T}, {tt.n_components} components)",
          same_knn(i_on, i_off, d_on, d_off))
    check(f"matches static cKDTree  (T={T})", same_knn(i_on, ib_ref, d_on, db_ref))

for kk in (1, 20):
    tt = DynamicKDTree(3, size_ratio=64, micro_size=32)
    tt.insert_many(Xb)
    dk, ik = tt.query(Qb, k=kk, workers=1, prune=True)
    dk2, ik2 = tt.query(Qb, k=kk, workers=1, prune=False)
    dr_, ir_ = cKDTree(Xb).query(Qb, k=kk)
    check(f"k={kk} pruned result is exact",
          same_knn(ik.reshape(len(Qb), kk), ir_.reshape(len(Qb), kk),
                   dk.reshape(len(Qb), kk), dr_.reshape(len(Qb), kk)))
    check(f"k={kk} prune on == prune off", same_knn(ik, ik2, dk, dk2))

print("\n9. Deletion and pruning together")
tt = DynamicKDTree(3, size_ratio=64, micro_size=32, compact_ratio=0.99)
tt.insert_many(Xb)
dead = r2.choice(20000, 6000, replace=False)
for pid in dead:
    tt.delete(int(pid))
alive = np.setdiff1d(np.arange(20000), dead)
dra, ira = cKDTree(Xb[alive]).query(Qb, k=5)
d1, i1 = tt.query(Qb, k=5, workers=1, prune=True)
d0, i0 = tt.query(Qb, k=5, workers=1, prune=False)
check("30% deleted: pruned == exhaustive", same_knn(i1, i0, d1, d0))
check("30% deleted: matches ground truth", same_knn(alive[ira], i1, dra, d1))
check("30% deleted: no leaked ids", not (set(i1.ravel().tolist()) & set(dead.tolist())))

# single-point and small-batch queries take a different pruning path (one shared bound)
for nq in (1, 2, 7, 63):
    ok_p = ok_t = True
    for s in range(0, 400, nq):
        qs = Qb[s:s + nq]
        dp, ip = tt.query(qs, k=5, workers=1, prune=True)
        dn, in_ = tt.query(qs, k=5, workers=1, prune=False)
        ok_p &= same_knn(ip, in_, dp, dn)
        ok_t &= same_knn(alive[ira[s:s + nq]], ip, dra[s:s + nq], dp)
    check(f"nq={nq} with deletes: pruned == exhaustive", ok_p)
    check(f"nq={nq} with deletes: matches ground truth", ok_t)
# clustered data with duplicates, the case where the bound matters most
r3 = np.random.RandomState(11)
Xc = np.repeat(r3.randn(2000, 3) * [1, 50, 0.01], 10, axis=0) + 1e-3 * r3.randn(20000, 3)
Qc = Xc[r3.choice(20000, 300, replace=False)] + 1e-3 * r3.randn(300, 3)
tc = DynamicKDTree(3)
tc.bulk_load(Xc[:10000])
tc.insert_many(Xc[10000:])
dcr, icr = cKDTree(Xc).query(Qc, k=10)
ok_c = True
for i in range(300):
    dq, iq = tc.query(Qc[i], k=10, workers=1)
    ok_c &= same_knn(iq, icr[i:i + 1], dq, dcr[i:i + 1])
check("clustered data, single queries, multi-run: exact", ok_c, repr(tc))

print("\n10. MahalanobisIndex agrees with sklearn BallTree(metric='mahalanobis')")
try:
    from sklearn.neighbors import BallTree  # noqa: E402
    HAVE_SKLEARN = True
except ImportError:                      # not a runtime dependency; cross-check only
    HAVE_SKLEARN = False
    print("  (scikit-learn not installed; cross-checks against it skipped)")
cov = np.array([[10., 8., 0.], [8., 10., 0.], [0., 0., 0.5]])
P = rng.multivariate_normal(np.zeros(3), cov, size=5000)
Qm = rng.multivariate_normal(np.zeros(3), cov, size=200)
mi = MahalanobisIndex(cov=cov, dynamic=True).fit(P)
dm, im = mi.kneighbors(Qm, k=5, workers=1)
if HAVE_SKLEARN:
    bt = BallTree(P, metric="mahalanobis", VI=np.linalg.inv(cov))
    db_, ib_ = bt.query(Qm, k=5)
    check("same neighbours as BallTree",
          all(set(a) == set(b) for a, b in zip(im.tolist(), ib_.tolist())))
    check("same distances", np.allclose(dm, db_, atol=1e-8),
          f"max err {np.abs(dm - db_).max():.2e}")
mi.insert(P[0])
check("still works after an insert", len(mi) == 5001)
mi2 = MahalanobisIndex(dynamic=False).fit(P)          # estimated-covariance path
d_s, i_s = mi2.kneighbors(Qm, k=5, workers=1)
mi3 = MahalanobisIndex(dynamic=True).fit(P)
d_d, i_d = mi3.kneighbors(Qm, k=5, workers=1)
check("dynamic=True and dynamic=False agree", same_knn(i_d, i_s, d_d, d_s))

print("\n11. Single-point queries after every single insert")
# Regression test. The batch path and the nq==1 path are different code, and the suite used
# to insert one point at a time but always query in batches -- so a bug that only affected
# single-point queries with three or more components survived every check here. It dropped
# the micro buffer whenever an older run answered first, costing ~0.2% of queries their
# true nearest neighbour. Checking the invariant after *every* insert is what caught it.
r3 = np.random.RandomState(4)
Xs = r3.randn(3000, 3)
Ps = r3.randn(8, 3)
true_nn = np.minimum.accumulate(
    np.sqrt(((Xs[None, :, :] - Ps[:, None, :]) ** 2).sum(-1)), axis=1)
ts = DynamicKDTree(3)
violations = 0
for m in range(1, len(Xs) + 1):
    ts.insert(Xs[m - 1])
    p = m % len(Ps)
    dq, _ = ts.query(Ps[p], k=5, workers=1)
    if not dq[0, 0] <= true_nn[p, m - 1] + 1e-12:
        violations += 1
check("nearest neighbour correct after every insert", violations == 0,
      f"{violations} violations out of {len(Xs)}")

seen_components = set()
ts2 = DynamicKDTree(3, size_ratio=4, micro_size=4)
worst = 0
for m in range(1, 2001):
    ts2.insert(Xs[m - 1])
    seen_components.add(ts2.n_components)
    p = m % len(Ps)
    dq, _ = ts2.query(Ps[p], k=5, workers=1)
    worst = max(worst, dq[0, 0] - true_nn[p, m - 1])
check("same with many components (T=4, b=4)", worst <= 1e-12,
      f"worst excess {worst:.3e}, component counts seen {sorted(seen_components)}")

print("\n12. Covariance drift tracking and re-fit (D4)")
cov3 = np.array([[10., 8., 0.], [8., 10., 0.], [0., 0., 0.5]])
r4 = np.random.RandomState(11)
A = r4.multivariate_normal(np.zeros(3), cov3, size=20000)
Qd = r4.multivariate_normal(np.zeros(3), cov3, size=300)
mi = MahalanobisIndex().fit(A)
d0, i0 = mi.kneighbors(Qd, k=5, workers=1)
check("drift is ~0 right after fit", mi.drift() < 1e-6, f"{mi.drift():.2e}")

mi.refit(cov=np.cov(A, rowvar=False))
d1, i1 = mi.kneighbors(Qd, k=5, workers=1)
check("re-fit to the same covariance changes nothing", same_knn(i1, i0, d1, d0, tol=1e-12))
for _ in range(30):
    mi.refit(cov=np.cov(A, rowvar=False))
d2, _ = mi.kneighbors(Qd, k=5, workers=1)
check("31 re-fits stay numerically stable", np.abs(d2 - d0).max() < 1e-12,
      f"max drift of distances {np.abs(d2 - d0).max():.2e}")

# a stream from a different covariance must register as drift, and auto-refit must clear it
cov4 = np.array([[10., -8., 0.], [-8., 10., 0.], [0., 0., 3.]])
B = r4.multivariate_normal(np.zeros(3), cov4, size=60000)
m_off = MahalanobisIndex().fit(A)
m_off.insert_many(B)
check("drift is detected on a shifted stream", m_off.drift() > 1.0, f"{m_off.drift():.2f}")
check("no re-fit happens without a threshold", m_off._refits == 0)

m_on = MahalanobisIndex(refit_threshold=0.25, check_every=5000).fit(A)
m_on.insert_many(B)
check("auto re-fit fires and clears the drift", m_on._refits >= 1 and m_on.drift() < 0.25,
      f"refits={m_on._refits} drift={m_on.drift():.3f}")
check("auto re-fit keeps every point", len(m_on) == len(A) + len(B), f"len={len(m_on)}")
ref_tree = cKDTree(np.vstack([A, B]) @ m_on.L_inv.T)
dref, iref = ref_tree.query(Qd @ m_on.L_inv.T, k=5, workers=1)
dgot, igot = m_on.kneighbors(Qd, k=5, workers=1)
check("index after auto re-fit is still exact", same_knn(igot, iref, dgot, dref, tol=1e-12),
      f"max dist err {np.abs(dgot - dref).max():.2e}")

# ids must survive a re-fit, and tombstones must not
m_id = MahalanobisIndex().fit(A[:5000])
pid = m_id.insert(A[0])
m_id.delete(3)
m_id.refit(cov=np.cov(A, rowvar=False))
check("ids survive a re-fit", m_id.kneighbors(A[0], k=1, workers=1)[1][0][0] in (0, pid))
check("re-fit clears tombstones", len(m_id.backend._dead) == 0 and len(m_id) == 5000)

m_none = MahalanobisIndex(cov_window=None).fit(A)
check("cov_window=None disables tracking", m_none.drift() == 0.0)
try:
    MahalanobisIndex(dynamic=False).fit(A).refit()
    ok_static = False
except RuntimeError:
    ok_static = True
check("re-fit on a static index is refused", ok_static)

print("\n13. Covariance estimation: scale, shrinkage, degenerate input (D3)")
import warnings  # noqa: E402

r5 = np.random.RandomState(3)
if HAVE_SKLEARN:
    from sklearn.covariance import ledoit_wolf as _sk_lw  # noqa: E402
    ok_lw = True
    for nn, dd in [(20, 5), (60, 8), (200, 4), (15, 16)]:
        Z = r5.randn(nn, dd) @ r5.randn(dd, dd)
        C1, a1 = MahalanobisIndex._ledoit_wolf(Z)
        C2, a2 = _sk_lw(Z)
        ok_lw &= abs(a1 - a2) < 1e-12 and np.abs(C1 - C2).max() < 1e-10
    check("Ledoit-Wolf matches sklearn", ok_lw)

# The ridge must not depend on what units the data happens to be in.
Xs = r5.randn(4000, 4) @ np.diag([1.0, 1.0, 1.0, 1e-5]) @ r5.randn(4, 4)
Qs = Xs[:200]
d_a, i_a = MahalanobisIndex().fit(Xs).kneighbors(Qs, k=5, workers=1)
for mult in (1e-4, 1e4):
    d_b, i_b = MahalanobisIndex().fit(Xs * mult).kneighbors(Qs * mult, k=5, workers=1)
    same_ids = all(set(a) == set(b) for a, b in zip(i_a.tolist(), i_b.tolist()))
    check(f"whitening is scale-free (x{mult:g})", same_ids,
          "" if same_ids else "rescaling the inputs changed the neighbours")

# auto shrinkage: on below 5d, off above, and it must rescue n <= d
m_small = MahalanobisIndex().fit(r5.randn(12, 6))
check("auto shrinkage engages when n < 5d", m_small.shrinkage_ > 0,
      f"intensity {m_small.shrinkage_:.3f}")
m_big = MahalanobisIndex().fit(r5.randn(600, 6))
check("auto shrinkage stays off when n >= 5d", m_big.shrinkage_ == 0.0)
m_tiny = MahalanobisIndex().fit(r5.randn(5, 6))       # n < d: sample cov is singular
dt, it = m_tiny.kneighbors(r5.randn(3, 6), k=2, workers=1)
check("n < d still produces a usable index", np.isfinite(dt).all(),
      f"intensity {m_tiny.shrinkage_:.3f}")

m_fix = MahalanobisIndex(shrinkage=0.3).fit(r5.randn(600, 6))
check("explicit shrinkage is honoured", abs(m_fix.shrinkage_ - 0.3) < 1e-12)
m_off = MahalanobisIndex(shrinkage=None).fit(r5.randn(12, 6))
check("shrinkage=None disables it", m_off.shrinkage_ == 0.0)
try:
    MahalanobisIndex(shrinkage=1.7).fit(r5.randn(12, 6))
    bad_shrink = False
except ValueError:
    bad_shrink = True
check("out-of-range shrinkage is rejected", bad_shrink)


# degenerate inputs must fail loudly rather than silently
def raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True


Xc = r5.randn(100, 3)
Xc[:, 1] = 4.0
check("a constant column is rejected", raises(lambda: MahalanobisIndex().fit(Xc)))
Xn = r5.randn(100, 3)
Xn[7, 2] = np.nan
check("NaN input is rejected", raises(lambda: MahalanobisIndex().fit(Xn)))
Xi_ = r5.randn(100, 3)
Xi_[9, 0] = np.inf
check("infinite input is rejected", raises(lambda: MahalanobisIndex().fit(Xi_)))
check("a single row is rejected", raises(lambda: MahalanobisIndex().fit(r5.randn(1, 3))))
check("wrong-shaped cov is rejected",
      raises(lambda: MahalanobisIndex(cov=np.eye(2)).fit(r5.randn(50, 3))))

# near-duplicate channels: usable, but the user has to be told
Xd = r5.randn(5000, 4)
Xd[:, 1] = Xd[:, 0] * 2.0 + r5.randn(5000) * 1e-9
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    MahalanobisIndex().fit(Xd)
    warned = any(issubclass(x.category, RuntimeWarning) for x in w)
check("ill-conditioned covariance warns", warned)

print("\n14. Audit findings -- regressions for things that were wrong")
r6 = np.random.RandomState(5)
Xa = r6.randn(4000, 3)

# Every run's id array has to stay ascending: `_is_present` binary-searches them.
ta = DynamicKDTree(3, size_ratio=4, micro_size=4)
ta.insert_many(Xa)
sorted_ok = all(np.all(np.diff(g) > 0) for _, g, _ in ta.runs)
for pid in range(0, 1200):
    ta.delete(pid)                                    # forces a compaction on the way
ta.insert_many(r6.randn(500, 3))
sorted_ok &= all(np.all(np.diff(g) > 0) for _, g, _ in ta.runs)
check("run id arrays stay ascending", sorted_ok, str(ta.run_sizes()))

# Deleting an id that a compaction already purged must not decrement the live count.
tb = DynamicKDTree(3, compact_ratio=0.4)
tb.insert_many(Xa[:1000])
for pid in range(500):
    tb.delete(pid)
tb.compact()
live_before = len(tb)
again = [tb.delete(pid) for pid in range(0, 20)]
check("re-deleting purged ids is refused", not any(again))
check("live count survives it", len(tb) == live_before == 500,
      f"len={len(tb)} before={live_before}")
check("deleting a never-issued id is refused", tb.delete(10_000) is False)

# refit must hand readers fresh micro storage, or a stale view reads overwritten slots.
mi_r = MahalanobisIndex().fit(Xa[:2000])
mi_r.insert(Xa[2000])
buf_before = mi_r.backend._micro_pts
mi_r.refit(cov=np.cov(Xa, rowvar=False))
mi_r.insert(Xa[2001])
check("refit allocates a fresh micro buffer", mi_r.backend._micro_pts is not buf_before)

# drift() must not quietly rewrite what fit recorded
mi_s = MahalanobisIndex().fit(r6.randn(30, 6))        # small n, so shrinkage engages
rec = mi_s.shrinkage_
mi_s.insert_many(r6.randn(500, 6))
mi_s.drift()
check("drift() leaves shrinkage_ alone", mi_s.shrinkage_ == rec)

# a NaN reaching the index would poison it permanently
bad_pt = np.array([1.0, np.nan, 2.0])
mi_n = MahalanobisIndex().fit(r6.randn(200, 3))
check("inserting a NaN is refused", raises(lambda: mi_n.insert(bad_pt)))
check("inserting NaNs in a batch is refused",
      raises(lambda: mi_n.insert_many(np.vstack([r6.randn(3, 3), bad_pt]))))
check("the index is untouched after a refused insert", len(mi_n) == 200, f"len={len(mi_n)}")

print("\n" + "=" * 60)
print("all checks passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
raise SystemExit(1 if FAIL else 0)
