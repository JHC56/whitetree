# -*- coding: utf-8 -*-
"""Loaders for the two UCI sensor logs used by the benchmarks. Expected layout:

    data/household/household_power_consumption.txt
        https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
    data/airquality/AirQualityUCI.csv
        https://archive.ics.uci.edu/static/public/360/air+quality.zip
    data/road3d/3D_spatial_network.txt   (loaded by datasets.py)
        https://archive.ics.uci.edu/static/public/246/3d+road+network+north+jutland+denmark.zip
"""
from __future__ import annotations

import io
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_household(max_rows=None, with_hour=False):
    """2.05M rows x 7 channels at 1-minute sampling. Rows with '?' (missing) are dropped."""
    path = os.path.join(DATA, "household", "household_power_consumption.txt")
    cols, rows, hours = None, [], []
    with io.open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(";")
        cols = header[2:]                       # drop Date, Time
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) != len(header) or "?" in parts[2:]:
                continue
            rows.append([float(v) for v in parts[2:]])
            if with_hour:
                hours.append(int(parts[1][:2]))
            if max_rows and len(rows) >= max_rows:
                break
    X = np.asarray(rows)
    if with_hour:
        return X, cols, np.asarray(hours, dtype=np.int64)
    return X, cols


def load_airquality():
    """9,358 hourly rows. -200 marks missing; channels missing in >20% of rows are dropped,
    then rows with any remaining missing value."""
    path = os.path.join(DATA, "airquality", "AirQualityUCI.csv")
    with io.open(path, encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().strip().split(";")]
        ncol = len([h for h in header if h])
        cols = header[2:ncol]
        rows = []
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(";")]
            if len(parts) < ncol or not parts[0]:
                continue
            vals = []
            for p in parts[2:ncol]:
                p = p.replace(",", ".")
                vals.append(np.nan if p == "" else float(p))
            rows.append(vals)
    arr = np.asarray(rows, dtype=float)
    arr[arr == -200] = np.nan
    miss = np.isnan(arr).mean(0)
    keep = miss < 0.2
    dropped = [(c, f"{m*100:.0f}% missing") for c, m, kp in zip(cols, miss, keep) if not kp]
    if dropped:
        print(f"  air quality: dropped channels {dropped}")
    arr = arr[:, keep]
    cols = [c for c, kp in zip(cols, keep) if kp]
    arr = arr[~np.isnan(arr).any(1)]
    return arr, cols
