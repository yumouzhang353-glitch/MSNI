import numpy as np

CONTACT_ENDPOINTS = 2
TIE_MULTIPLIER = 64.0


def contacts(pairs, ids, mask):
    groups = {}
    for k in np.flatnonzero(mask):
        (a, b) = map(int, pairs[k])
        (ca, cb) = (int(ids[a]), int(ids[b]))
        if ca > cb:
            (ca, cb, a, b) = (cb, ca, b, a)
        groups.setdefault((ca, cb), []).append((int(k), a, b))
    (left, right) = (np.zeros(len(pairs), np.int64), np.zeros(len(pairs), np.int64))
    for entries in groups.values():
        positions = [k for (k, _, _) in entries]
        left[positions] = len({a for (_, a, _) in entries})
        right[positions] = len({b for (_, _, b) in entries})
    return (left, right, (left >= CONTACT_ENDPOINTS) & (right >= CONTACT_ENDPOINTS))


def component_evidence(pairs, ids, distance, forward, backward, eligible, support):
    (a, b) = (pairs[:, 0], pairs[:, 1])
    ncomp = int(ids.max()) + 1
    values = [[] for _ in range(ncomp)]
    internal = ids[a] == ids[b]
    for k in np.flatnonzero(internal):
        count = int(forward[k]) + int(backward[k])
        values[int(ids[a[k]])].extend([float(distance[k])] * count)
    counts = np.asarray(list(map(len, values)), np.int64)
    scale50 = np.asarray([np.percentile(v, 50) if v else 0.0 for v in values])
    scale75 = np.asarray([np.percentile(v, 75) if v else 0.0 for v in values])
    closest = np.full(ncomp, np.inf)
    pair_minimum = {}
    for k in np.flatnonzero(eligible):
        (ca, cb) = sorted((int(ids[a[k]]), int(ids[b[k]])))
        pair_minimum[ca, cb] = min(pair_minimum.get((ca, cb), np.inf), float(distance[k]))
    for (ca, cb), value in pair_minimum.items():
        (closest[ca], closest[cb]) = (min(closest[ca], value), min(closest[cb], value))
    edge_pair_distance = np.full(len(pairs), np.inf)
    reciprocal = np.zeros(len(pairs), bool)
    for k in np.flatnonzero(eligible):
        (ca, cb) = sorted((int(ids[a[k]]), int(ids[b[k]])))
        value = pair_minimum[ca, cb]
        edge_pair_distance[k] = value
        tolerance = (
            TIE_MULTIPLIER
            * np.finfo(np.float64).eps
            * max(abs(value), abs(closest[ca]), abs(closest[cb]), np.finfo(np.float64).tiny)
        )
        reciprocal[k] = value <= closest[ca] + tolerance and value <= closest[cb] + tolerance
    arrays = dict(
        receiver_component_sizes=np.bincount(ids, minlength=ncomp),
        receiver_component_internal_counts=counts,
        receiver_component_scale50=scale50,
        receiver_component_scale75=scale75,
        receiver_component_closest=closest,
        receiver_component_pair_distance=edge_pair_distance,
        receiver_component_reciprocal=reciprocal,
    )
    for tag, scales in [("50", scale50), ("75", scale75)]:
        candidate = eligible & support & (distance <= scales[ids[a]]) & (distance <= scales[ids[b]])
        (left, right, quorum) = contacts(pairs, ids, candidate)
        arrays["receiver_component_candidate" + tag] = candidate
        arrays["receiver_component_left" + tag] = left
        arrays["receiver_component_right" + tag] = right
        arrays["receiver_component_quorum" + tag] = quorum
        arrays["receiver_component_extra" + tag] = candidate & quorum
    arrays["receiver_component_extra_reciprocal75"] = (
        arrays["receiver_component_extra75"] & reciprocal
    )
    return arrays
