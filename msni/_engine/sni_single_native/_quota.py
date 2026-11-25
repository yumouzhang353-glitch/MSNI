import numpy as np
from . import _core, _screening

VERSIONS = ["NQ1_BLOCK", "NQ2_SUPPORT", "NQ3_WEAK_SUPPORT"]


def score_block(scores, candidates, capacity):
    selected = np.zeros(len(scores), bool)
    if capacity <= 0 or not candidates.any():
        return selected
    if int(candidates.sum()) <= capacity:
        return candidates.copy()
    cut = float(np.partition(scores[candidates], capacity - 1)[capacity - 1])
    tolerance = 64 * np.finfo(float).eps * max(abs(cut), np.finfo(float).tiny)
    below = candidates & (scores < cut - tolerance)
    equal = candidates & (np.abs(scores - cut) <= tolerance)
    selected |= below
    if int((below | equal).sum()) <= capacity:
        selected |= equal
    assert selected.sum() <= capacity
    return selected


def quota_mask(nb, peer_nb, relative_density, weak, version):
    assert version in VERSIONS
    nb = np.asarray(nb, int)
    n = len(nb)
    score = np.asarray(relative_density) * (nb + 1) / (np.asarray(peer_nb) + 1)
    quota = int(0.3 * n)
    positive = nb[nb > _core.EPS]
    k_auto = max(1, int(np.percentile(positive, 15))) if len(positive) else 0
    candidates = nb <= k_auto
    mask = np.zeros(n, bool)
    cut = -1
    if n <= 5:
        candidates[:] = False
    elif int(candidates.sum()) <= quota:
        mask = candidates.copy()
    else:
        cut = int(np.partition(nb[candidates], quota - 1)[quota - 1])
        lower = candidates & (nb < cut)
        boundary = candidates & (nb == cut)
        capacity = quota - int(lower.sum())
        mask |= lower
        if int(boundary.sum()) <= capacity:
            mask |= boundary
        elif version == "NQ2_SUPPORT":
            mask |= score_block(score, boundary, capacity)
        elif version == "NQ3_WEAK_SUPPORT":
            admissible = boundary & (
                weak | (relative_density < _screening.P["density_deficit_ratio"])
            )
            mask |= score_block(score, admissible, capacity)
    assert not (mask & ~candidates).any() and int(mask.sum()) <= quota
    return (
        mask,
        dict(
            quota_nb=nb,
            quota_peer_nb=np.asarray(peer_nb),
            quota_relative_density=np.asarray(relative_density),
            quota_score=score,
            quota_weak=np.asarray(weak, bool),
            quota_candidates=candidates,
            quota_capacity=np.asarray(quota),
            quota_k_auto=np.asarray(k_auto),
            quota_boundary_nb=np.asarray(cut),
        ),
    )


def quota_pre_state(X, core, version):
    pre = _screening.pre_state(X, core)
    nb = pre["graph"]["nb"]
    peer = np.asarray(
        [
            np.median(nb[np.asarray(sorted(ns), int)]) if ns else nb[i]
            for (i, ns) in enumerate(pre["graph"]["NN"])
        ]
    )
    (mask, arrays) = quota_mask(nb, peer, pre["ev"]["relative_density"], pre["weak"], version)
    arrays["quota_original_B"] = pre["B"].copy()
    pre["B"] = mask
    pre["quota_arrays"] = arrays
    return pre
