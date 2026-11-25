import math
import numpy as np
from sklearn.neighbors import KDTree

MULTIPLIER = 64.0


def numerical_scale(X):
    centered = X - X.mean(axis=0)
    return float(np.max(np.linalg.norm(centered, axis=1)))


class RankNeighborhoods:
    def __init__(self, X, max_rank, tree=None):
        self.X = np.asarray(X, dtype=np.float64)
        self.tree = KDTree(self.X) if tree is None else tree
        self.scale = numerical_scale(self.X)
        q = min(len(X), int(max_rank) + 2)
        (distances, indices) = self.tree.query(self.X, k=q)
        (rows, values) = ([], [])
        for i, (ds, ns) in enumerate(zip(distances, indices)):
            other = ns != i
            rows.append(ns[other].astype(np.int64))
            values.append(ds[other])
        (self.indices, self.distances) = (rows, values)

    def row(self, i, rank):
        rank = min(len(self.X) - 1, max(1, int(rank)))
        (distances, indices) = (self.distances[i], self.indices[i])
        assert len(distances) >= rank
        boundary = float(distances[rank - 1])
        tolerance = MULTIPLIER * np.finfo(np.float64).eps * max(self.scale, boundary)
        upper = np.nextafter(boundary + tolerance, np.inf)
        stop = int(np.searchsorted(distances, upper, side="right"))
        if stop == len(distances) and len(distances) < len(self.X) - 1:
            indices = self.tree.query_radius(self.X[i : i + 1], r=upper)[0]
            indices = indices[indices != i]
            actual = np.linalg.norm(self.X[indices] - self.X[i], axis=1)
            indices = indices[actual <= upper]
        else:
            indices = indices[:stop]
        assert len(indices) >= rank and i not in indices
        return np.sort(indices).astype(np.int64)

    def rows(self, ranks):
        ranks = np.broadcast_to(np.asarray(ranks, int), (len(self.X),))
        return [self.row(i, rank) for (i, rank) in enumerate(ranks)]


def natural_neighbor_search(data):
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    tree = KDTree(data)
    if n <= 1:
        return (0, [set() for _ in range(n)], [set() for _ in range(n)], np.zeros(n, int), tree)
    kmax = min(n - 1, max(8, int(math.sqrt(n) * 4)))
    ranked = RankNeighborhoods(data, kmax, tree)
    (NN, RNN) = ([set() for _ in data], [set() for _ in data])
    zero_history = []
    for r in range(1, n):
        if r > kmax:
            kmax = min(n - 1, 2 * kmax)
            ranked = RankNeighborhoods(data, kmax, tree)
        for i in range(n):
            next_neighbors = set(map(int, ranked.row(i, r)))
            added = next_neighbors - NN[i]
            assert not NN[i] - next_neighbors
            NN[i] = next_neighbors
            for j in added:
                RNN[j].add(i)
        nb = np.asarray(list(map(len, RNN)), np.int64)
        zero = int(np.count_nonzero(nb == 0))
        if zero == 0:
            break
        zero_history.append(zero)
        if len(zero_history) >= 3 and zero_history[-1] == zero_history[-2] == zero_history[-3]:
            break
    return (r, NN, RNN, np.asarray(list(map(len, RNN)), np.int64), tree)


def variable_base(X, ranks, core, derive):
    ranks = np.asarray(ranks, int)
    assert ranks.shape == (len(X),) and np.all((ranks >= 1) & (ranks < len(X)))
    rows = RankNeighborhoods(X, int(ranks.max())).rows(ranks)
    NN = [set(map(int, row)) for row in rows]
    RNN = [set() for _ in X]
    for i, neighbors in enumerate(NN):
        for j in neighbors:
            RNN[j].add(i)
    return derive(X, NN, RNN, int(ranks.max()), core)
