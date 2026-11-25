import time
import numpy as np
from .restoration_candidates import apply as restore
from .assignment_candidates import prepare_neighbors, apply as assign_active
from .single_c5 import _core

R6 = "R6_FULL_LOCAL_PROP"


def local_neighbors(X, pre):
    gamma = [
        set(map(int, pre["gamma_indices"][pre["gamma_indptr"][i] : pre["gamma_indptr"][i + 1]]))
        for i in range(len(X))
    ]
    mu = _core.compute_local_scale_from_gamma(X, gamma)
    ptr = pre["NN_indptr"].astype(np.int64)
    indices = pre["NN_indices"].astype(np.int64)
    hosts = np.repeat(np.arange(len(X)), np.diff(ptr))
    raw = np.linalg.norm(X[indices] - X[hosts], axis=1)
    scaled = raw / (np.sqrt(mu[hosts] * mu[indices]) + _core.EPS)
    reverse = [[] for _ in X]
    for i, j in zip(hosts, indices):
        reverse[int(j)].append(int(i))
    neighbors = dict(indptr=ptr, indices=indices, raw=raw, local=scaled, reverse=reverse)
    shared = dict(
        r=pre["r"],
        NN_indptr=ptr,
        NN_indices=indices,
        original_scale=mu,
        gamma_indptr=pre["gamma_indptr"],
        gamma_indices=pre["gamma_indices"],
    )
    return (shared, neighbors)


def suffix(X, a8, pre):
    start = time.perf_counter()
    (shared, neighbors) = local_neighbors(X, pre)
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    (output, trace) = restore(a8, neighbors, R6)
    restore_s = time.perf_counter() - start
    return (
        output,
        trace,
        shared,
        dict(
            local_setup_seconds=setup_s,
            R6_restoration_seconds=restore_s,
            unused_full_pre_sigma_G_recomputed=False,
            other_restoration_variants_computed=False,
        ),
    )
