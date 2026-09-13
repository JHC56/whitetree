# Benchmarks

All runs: one thread, k = 10, recall@10 against float64 brute force. Scripts: `benchmarks/bench_static.py`, `bench_streaming.py`, `bench_interleaved.py`. Datasets: UCI 3D Road Network (434,874 x 3), UCI Household Power (500,000 x 7 subsample), UCI Air Quality (6,941 x 12), synthetic Gaussians (500,000 x 3 and x 8).

## Static search (ann-benchmarks protocol)

Machine: Intel i5-7300HQ, Windows 10, Python 3.14.5, single thread
Protocol: ann-benchmarks style. Held-out queries, k=10, recall@10 vs float64 brute force, batch QPS = best of 3, single QPS = one query at a time.

### road3d (n=424,874, d=3)

| method | build (s) | batch q/s | single q/s | recall@10 |
|---|---|---|---|---|
| whitetree | 0.30 | 176,332 | 14,319 | 1.0000 |
| whitetree, after 50% inserted | 7.17 | 82,562 | 2,929 | 1.0000 |
| scipy cKDTree (pre-whitened) | 0.28 | 170,413 | 18,655 | 1.0000 |
| sklearn KDTree (pre-whitened) | 0.94 | 82,098 | 6,314 | 1.0000 |
| sklearn BallTree mahalanobis | 0.73 | 256 | 236 | 1.0000 |
| faiss IndexFlatL2 (pre-whitened) | 0.04 | 366 | 359 | 0.9997 |
| faiss PCAMatrix + IndexFlatL2 | 0.01 | 403 | 350 | 0.9717 |
| numpy brute force | 0.03 | 114 | 31 | 1.0000 |

### household (n=500,000, d=7)

| method | build (s) | batch q/s | single q/s | recall@10 |
|---|---|---|---|---|
| whitetree | 1.11 | 16,220 | 7,856 | 1.0000 |
| whitetree, after 50% inserted | 14.50 | 15,384 | 4,414 | 1.0000 |
| scipy cKDTree (pre-whitened) | 0.57 | 16,330 | 8,126 | 1.0000 |
| sklearn KDTree (pre-whitened) | 3.02 | 6,908 | 2,983 | 1.0000 |
| sklearn BallTree mahalanobis | 1.86 | 25 | 25 | 1.0000 |
| faiss IndexFlatL2 (pre-whitened) | 0.09 | 234 | 230 | 1.0000 |
| faiss PCAMatrix + IndexFlatL2 | 0.06 | 232 | 213 | 0.9681 |
| numpy brute force | 0.05 | 108 | 22 | 1.0000 |

### gaussian-3 (n=500,000, d=3)

| method | build (s) | batch q/s | single q/s | recall@10 |
|---|---|---|---|---|
| whitetree | 0.38 | 135,540 | 14,302 | 1.0000 |
| whitetree, after 50% inserted | 9.39 | 114,469 | 6,796 | 1.0000 |
| scipy cKDTree (pre-whitened) | 0.34 | 140,636 | 21,088 | 1.0000 |
| sklearn KDTree (pre-whitened) | 1.14 | 56,834 | 5,920 | 1.0000 |
| sklearn BallTree mahalanobis | 0.89 | 383 | 330 | 1.0000 |
| faiss IndexFlatL2 (pre-whitened) | 0.03 | 307 | 226 | 1.0000 |
| faiss PCAMatrix + IndexFlatL2 | 0.04 | 301 | 266 | 0.9829 |
| numpy brute force | 0.06 | 114 | 34 | 1.0000 |

### gaussian-8 (n=500,000, d=8)

| method | build (s) | batch q/s | single q/s | recall@10 |
|---|---|---|---|---|
| whitetree | 0.64 | 2,855 | 2,302 | 1.0000 |
| whitetree, after 50% inserted | 13.11 | 2,760 | 1,879 | 1.0000 |
| scipy cKDTree (pre-whitened) | 0.56 | 2,664 | 1,873 | 1.0000 |
| sklearn KDTree (pre-whitened) | 3.11 | 735 | 705 | 1.0000 |
| sklearn BallTree mahalanobis | 3.02 | 13 | 14 | 1.0000 |
| faiss IndexFlatL2 (pre-whitened) | 0.14 | 343 | 317 | 1.0000 |
| faiss PCAMatrix + IndexFlatL2 | 0.09 | 285 | 278 | 0.9624 |
| numpy brute force | 0.05 | 91 | 17 | 1.0000 |

### airquality (n=6,247, d=12)

| method | build (s) | batch q/s | single q/s | recall@10 |
|---|---|---|---|---|
| whitetree | 0.01 | 18,375 | 7,888 | 0.9993 |
| whitetree, after 50% inserted | 0.07 | 14,314 | 2,002 | 0.9993 |
| scipy cKDTree (pre-whitened) | 0.00 | 16,407 | 5,653 | 1.0000 |
| sklearn KDTree (pre-whitened) | 0.03 | 9,063 | 1,756 | 1.0000 |
| sklearn BallTree mahalanobis | 0.02 | 1,155 | 871 | 1.0000 |
| faiss IndexFlatL2 (pre-whitened) | 0.00 | 18,243 | 14,204 | 1.0000 |
| faiss PCAMatrix + IndexFlatL2 | 0.00 | 19,670 | 15,555 | 1.0000 |
| numpy brute force | 0.00 | 8,062 | 2,945 | 1.0000 |

## Streaming, batched runbook (big-ann-benchmarks style)

Machine: Intel i5-7300HQ, Windows 10, Python 3.14.5, single thread
Runbook (big-ann-benchmarks streaming style): insert 200,000, then 10 rounds of insert 20,000 / delete oldest 20,000 / search (2,000 single-point queries, then 2,000 as one batch). k=10. Recall vs exact ground truth over the live window at each step. 'total' = insert + delete + single-point search.

### road3d

| method | insert (s) | delete (s) | search single (s) | search batch (s) | total (s) | recall single | recall batch |
|---|---|---|---|---|---|---|---|
| whitetree | 6.30 | 0.72 | 7.86 | 0.45 | 14.88 | 1.0000 | 1.0000 |
| faiss IndexFlatL2 + IDMap2 | 0.10 | 0.54 | 25.92 | 25.45 | 26.55 | 0.9998 | 0.9998 |
| scipy cKDTree, rebuild per search | 0.00 | 0.04 | 2.16 | 0.11 | 2.20 | 1.0000 | 1.0000 |
| sklearn BallTree mahalanobis, rebuild per search | 0.00 | 0.03 | 44.44 | 32.35 | 44.48 | 1.0000 | 1.0000 |
| numpy brute force | 0.00 | 0.04 | 211.43 | 59.31 | 211.47 | 1.0000 | 1.0000 |

### household

| method | insert (s) | delete (s) | search single (s) | search batch (s) | total (s) | recall single | recall batch |
|---|---|---|---|---|---|---|---|
| whitetree | 8.50 | 0.72 | 9.86 | 2.48 | 19.07 | 1.0000 | 1.0000 |
| faiss IndexFlatL2 + IDMap2 | 0.11 | 0.52 | 35.01 | 34.40 | 35.64 | 1.0000 | 1.0000 |
| scipy cKDTree, rebuild per search | 0.01 | 0.04 | 3.91 | 0.90 | 3.96 | 1.0000 | 1.0000 |
| sklearn BallTree mahalanobis, rebuild per search | 0.00 | 0.03 | 310.12 | 316.74 | 310.15 | 1.0000 | 1.0000 |
| numpy brute force | 0.01 | 0.03 | 388.13 | 75.81 | 388.17 | 1.0000 | 1.0000 |

## Streaming, interleaved runbook

Machine: Intel i5-7300HQ, Windows 10, Python 3.14.5, single thread
Runbook: window of 200,000 points; each step = insert 1, delete the oldest, search 1 (k=10); every search must reflect every insert so far. Rebuild-per-query baselines run 200 steps and are extrapolated.

### road3d

| method | ms / step | steps / s | time for 20,000 steps | recall@10 |
|---|---|---|---|---|
| whitetree | 0.894 | 1,119 | 17.9 s | 1.0000 |
| faiss IndexFlatL2 + IDMap2 | 53.696 | 19 | 1,073.9 s | 1.0000 |
| numpy brute force | 23.454 | 43 | 469.1 s | 1.0000 |
| scipy cKDTree, rebuild per search | 123.424 | 8 | 2,468.5 s (extrapolated) | 1.0000 |
| sklearn BallTree mahalanobis, rebuild per search | 275.250 | 4 | 5,505.0 s (extrapolated) | 1.0000 |

### household

| method | ms / step | steps / s | time for 20,000 steps | recall@10 |
|---|---|---|---|---|
| whitetree | 0.952 | 1,050 | 19.0 s | 1.0000 |
| faiss IndexFlatL2 + IDMap2 | 48.358 | 21 | 967.2 s | 1.0000 |
| numpy brute force | 34.750 | 29 | 695.0 s | 1.0000 |
| scipy cKDTree, rebuild per search | 183.220 | 5 | 3,664.4 s (extrapolated) | 1.0000 |
| sklearn BallTree mahalanobis, rebuild per search | 626.563 | 2 | 12,531.3 s (extrapolated) | 1.0000 |
