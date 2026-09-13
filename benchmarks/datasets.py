# -*- coding: utf-8 -*-
"""Real low-dimensional datasets for the promotional benchmarks.

    household   UCI Individual Household Electric Power Consumption, 7 channels, 1-min sampling
    road3d      UCI 3D Road Network (North Jutland), lon/lat/altitude, 434,874 points
    airquality  UCI Air Quality, 12 channels, hourly

All from https://archive.ics.uci.edu (ids 235, 246, 360). Files expected under ../data/.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from uci_data import load_airquality, load_household  # noqa: E402

DATA = os.path.join(ROOT, "data")


def load_road3d():
    path = os.path.join(DATA, "road3d", "3D_spatial_network.txt")
    return np.loadtxt(path, delimiter=",")[:, 1:]        # drop OSM id


def get(name, n_max=None, seed=0):
    """Return (X, description). Rows are shuffled so a prefix is a random sample."""
    if name == "household":
        X, _ = load_household()
        desc = "UCI household power, 7 ch"
    elif name == "road3d":
        X = load_road3d()
        desc = "UCI 3D road network, 3 ch"
    elif name == "airquality":
        X, _ = load_airquality()
        desc = "UCI air quality, 12 ch"
    elif name.startswith("gaussian"):
        d = int(name.split("-")[1])
        rng = np.random.RandomState(seed)
        A = rng.randn(d, d)
        X = rng.randn(n_max or 500_000, d) @ A.T
        desc = f"synthetic gaussian, {d} ch"
    else:
        raise ValueError(name)
    rng = np.random.RandomState(seed)
    X = X[rng.permutation(len(X))]
    if n_max:
        X = X[:n_max]
    return np.ascontiguousarray(X, dtype=np.float64), desc
