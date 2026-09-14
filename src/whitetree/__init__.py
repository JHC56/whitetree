"""Exact Mahalanobis nearest neighbour search with insert and delete.

    from whitetree import MahalanobisIndex

    idx = MahalanobisIndex().fit(X)
    d, i = idx.kneighbors(q, k=5)
    pid = idx.insert(new_row)
    idx.delete(pid)

`DynamicKDTree` is the Euclidean backend and can be used on its own.
"""
from .dynamic_kdtree import DynamicKDTree
from .mahalanobis import MahalanobisIndex

__all__ = ["MahalanobisIndex", "DynamicKDTree"]
__version__ = "0.1.1"
