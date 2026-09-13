# -*- coding: utf-8 -*-
"""Interleaved runbook: the control-loop workload. Every step does
    insert 1 new point -> delete the oldest point -> search 1 query
and every search must reflect all inserts so far (exact semantics, no staleness).

The window stays at n0 points. Methods that rebuild must rebuild at every step; they are
run for a short prefix and the per-step cost is extrapolated, marked as such.

    python bench_interleaved.py [dataset ...]

Writes CSV, a markdown summary and plots to results/ next to this script:
    interleaved_<dataset>.csv, interleaved_summary.md, interleaved_steps.png
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
from bench_streaming import (FaissFlat, NumpyBrute, RebuildBallTree, RebuildCKD,  # noqa: E402
                             WhiteTree, HAVE_FAISS, HAVE_SK, K)

N0 = 200_000
STEPS = 20_000              # full run for methods that support updates
STEPS_REBUILD = 200         # prefix for rebuild-per-query baselines, then extrapolated
CHECK_EVERY = 100           # recall is checked at every 100th step against exact truth
DATASETS = sys.argv[1:] or ["road3d", "household"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def run_dataset(name):
    X, desc = datasets.get(name, n_max=N0 + STEPS + STEPS)
    if len(X) < N0 + 2 * STEPS:
        raise SystemExit(f"{name}: need {N0 + 2 * STEPS:,} rows")
    Q, X = X[:STEPS], X[STEPS:]
    C = np.cov(X, rowvar=False)
    W = np.linalg.inv(np.linalg.cholesky(C))
    Xw, Qw = X @ W.T, Q @ W.T
    d = X.shape[1]
    print(f"\n{desc}: window {N0:,}, {STEPS:,} steps of insert 1 / delete oldest / search 1")

    methods = [(WhiteTree, STEPS)]
    if HAVE_FAISS:
        methods.append((FaissFlat, STEPS))
    methods.append((NumpyBrute, STEPS))
    methods.append((RebuildCKD, STEPS_REBUILD))
    if HAVE_SK:
        methods.append((RebuildBallTree, STEPS_REBUILD))

    rows = []
    for cls, steps in methods:
        m = cls(C, W, d)
        t0 = time.perf_counter()
        m.insert(X[:N0], np.arange(N0))
        t_build = time.perf_counter() - t0
        ok = tot = 0
        t_steps = 0.0
        for s in range(steps):
            pid = N0 + s
            t0 = time.perf_counter()
            m.insert(X[pid:pid + 1], [pid])
            m.delete([s])
            if hasattr(m, "round_start"):
                m.round_start()
            got = np.asarray(m.search_single(Q[s]))
            t_steps += time.perf_counter() - t0
            if s % CHECK_EVERY == 0:
                live = np.arange(s + 1, pid + 1)
                truth = cKDTree(Xw[live]).query(Qw[s], k=K)[0][K - 1]
                g = got[got >= 0]
                dd = np.sqrt(((Xw[g] - Qw[s]) ** 2).sum(1))
                ok += int((dd <= truth * (1 + 1e-9)).sum())
                tot += K
        per_step = t_steps / steps
        row = dict(dataset=name, method=m.name, build_s=t_build, steps_run=steps,
                   per_step_ms=per_step * 1e3, steps_per_s=1 / per_step,
                   est_total_s=per_step * STEPS, extrapolated=steps < STEPS,
                   recall=ok / tot)
        rows.append(row)
        tag = " (extrapolated from %d steps)" % steps if steps < STEPS else ""
        print(f"  {m.name:<48} {row['per_step_ms']:8.3f} ms/step  {row['steps_per_s']:>8,.0f} "
              f"steps/s  {STEPS:,} steps = {row['est_total_s']:8.1f}s  recall {row['recall']:.4f}"
              f"{tag}")
    return rows


def main():
    all_rows = []
    for name in DATASETS:
        rows = run_dataset(name)
        all_rows += rows
        with open(os.path.join(OUT, f"interleaved_{name}.csv"), "w", encoding="utf-8") as f:
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                                 for k in keys) + "\n")

    lines = [f"Machine: {platform.processor() or platform.machine()}, {platform.system()} "
             f"{platform.release()}, Python {platform.python_version()}, single thread",
             f"Runbook: window of {N0:,} points; each step = insert 1, delete the oldest, "
             f"search 1 (k={K}); every search must reflect every insert so far. "
             f"Rebuild-per-query baselines run {STEPS_REBUILD} steps and are extrapolated.", ""]
    for name in DATASETS:
        rs = [r for r in all_rows if r["dataset"] == name]
        lines += [f"### {name}", "",
                  "| method | ms / step | steps / s | time for 20,000 steps | recall@10 |",
                  "|---|---|---|---|---|"]
        for r in rs:
            est = f"{r['est_total_s']:,.1f} s" + (" (extrapolated)" if r["extrapolated"] else "")
            lines.append(f"| {r['method']} | {r['per_step_ms']:.3f} | {r['steps_per_s']:,.0f} | "
                         f"{est} | {r['recall']:.4f} |")
        lines.append("")
    with open(os.path.join(OUT, "interleaved_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["method"] for r in all_rows if r["dataset"] == DATASETS[0]]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 3.8))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, DATASETS):
        rs = {r["method"]: r for r in all_rows if r["dataset"] == name}
        vals = [rs[m]["steps_per_s"] for m in names]
        colors = ["#d62728" if m == "whitetree" else ("#c7c7c7" if rs[m]["extrapolated"]
                  else "#7f7f7f") for m in names]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("steps / s (log); light grey = extrapolated", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Interleaved: insert 1 / delete 1 / query 1 per step, window {N0:,}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "interleaved_steps.png"), dpi=130)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
