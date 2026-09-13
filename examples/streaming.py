# -*- coding: utf-8 -*-
"""A sensor stream: fit on the first hour, then keep inserting and querying.

    python examples/streaming.py
"""
import numpy as np

from whitetree import MahalanobisIndex

rng = np.random.RandomState(0)


def readings(n, drift=0.0):
    """Three channels with different units; the first two are strongly correlated."""
    t = rng.randn(n)
    temp_c = 20 + 5 * t + drift
    pressure_pa = 101_300 + 400 * t + 50 * rng.randn(n)
    flow_lpm = 3 + 0.5 * rng.randn(n)
    return np.column_stack([temp_c, pressure_pa, flow_lpm])


history = readings(3_600)
idx = MahalanobisIndex(cov_window=5_000).fit(history)

# stream: insert one reading, ask for the 5 most similar past readings
for step in range(10_000):
    x = readings(1, drift=step / 2_000)[0]
    d, i = idx.kneighbors(x, k=5)
    idx.insert(x)
    if step % 2_500 == 0:
        print(f"step {step:>5}  n={len(idx):>6}  nearest={d[0, 0]:.3f}  drift={idx.drift():.3f}")

print(f"\ncomponents before consolidate: {idx.backend.n_components}")
idx.consolidate()
print(f"components after  consolidate: {idx.backend.n_components}")

print(f"\ndrift {idx.drift():.3f} -> refit ->", end=" ")
idx.refit()
print(f"{idx.drift():.3f}")
