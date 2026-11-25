import numpy as np
from .geometry_operator167 import quadratic
from .geometry_adapter167 import validate_shared_geometry, require_canonical_pairs

MODES = ("D2", "E2", "E3", "B2")
PARAMETERS = dict(
    dimension_max=1.5,
    span_max=2.0,
    outward_fraction=0.75,
    internal_direction_percentile=75,
    numerical_tie_multiplier=64,
)


def propose(X, shared, core):
    n = len(X)
    if X.ndim != 2 or n < 2:
        raise ValueError("Fixed bridge path requires at least two points")
    validate_shared_geometry(X, shared)
    raw = shared["raw_ids"]
    source_q = shared["rank_repair_source_ranks"]
    if raw.shape != (n,) or raw.dtype.kind not in "iu" or (raw < 0).any():
        raise ValueError("Invalid raw component identities")
    if source_q.shape != (n,) or source_q.dtype.kind not in "iu" or (source_q < 0).any():
        raise ValueError("Invalid source neighbour ranks")
    (ptr, indices) = (shared["rank_repair_old_NN_indptr"], shared["rank_repair_old_NN_indices"])
    if (
        ptr.dtype.kind not in "iu"
        or ptr.shape != (n + 1,)
        or ptr[0] != 0
        or (ptr[-1] != len(indices))
        or (ptr[1:] < ptr[:-1]).any()
    ):
        raise ValueError("Invalid old-NN CSR pointers")
    if (
        indices.dtype.kind not in "iu"
        or indices.ndim != 1
        or (indices < 0).any()
        or (indices >= n).any()
    ):
        raise ValueError("Invalid old-NN CSR endpoints")
    if not np.array_equal(np.unique(raw), np.arange(int(raw.max()) + 1)):
        raise ValueError("Raw component IDs must be compact")
    old_nn = [indices[ptr[i] : ptr[i + 1]] for i in range(n)]
    radius = np.asarray(
        [
            max((float(np.linalg.norm(X[j] - X[i])) for j in ns), default=0.0)
            for (i, ns) in enumerate(old_nn)
        ]
    )
    inverse = shared["geometry_inverse"]
    dimensions = shared["geometry_dimensions"]
    (q2, q3) = (np.minimum(n - 1, 2 * source_q), np.minimum(n - 1, 3 * source_q))
    if int(q3.max()) == 0:
        neighbors = np.empty((n, 0), dtype=np.int64)
    else:
        (neighbors, _) = core._query_knn_excluding_self(X, core.KDTree(X), int(q3.max()))
    nn2 = [set(map(int, row[:q])) for (row, q) in zip(neighbors, q2)]
    nn3 = [set(map(int, row[:q])) for (row, q) in zip(neighbors, q3)]
    pairs = np.asarray(
        [
            (i, j)
            for (i, ns) in enumerate(nn3)
            for j in sorted(ns)
            if i < j and i in nn3[j] and (raw[i] != raw[j])
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    require_canonical_pairs(pairs, n)
    (a, b) = pairs.T
    delta = X[b] - X[a]
    distance = np.linalg.norm(delta, axis=1)
    unit = np.divide(
        delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 1e-12
    )
    factors = np.asarray(
        [np.stack([quadratic(inv, a, unit), quadratic(inv, b, unit)], axis=1) for inv in inverse]
    )
    internal_pairs = sorted(
        {tuple(sorted((i, int(j)))) for (i, ns) in enumerate(old_nn) for j in ns if i != j}
    )
    internal = np.asarray(internal_pairs, dtype=np.int64).reshape(-1, 2)
    (ia, ib) = internal.T
    displacement = X[ib] - X[ia]
    length = np.linalg.norm(displacement, axis=1)
    iu = np.divide(
        displacement,
        length[:, None],
        out=np.zeros_like(displacement),
        where=length[:, None] > 1e-12,
    )
    limits = np.ones((len(inverse), int(raw.max()) + 1), float)
    for level, inv in enumerate(inverse):
        value = np.maximum(quadratic(inv, ia, iu), quadratic(inv, ib, iu))
        for cid in range(limits.shape[1]):
            keep = (raw[ia] == cid) & (length > 1e-12)
            if keep.any():
                limits[level, cid] = max(
                    float(np.percentile(value[keep], PARAMETERS["internal_direction_percentile"])),
                    1e-12,
                )
    direction_limit = np.stack([limits[:, raw[a]], limits[:, raw[b]]], axis=2)
    direction_ratio = factors / direction_limit
    low_dimension = (dimensions[:, a] <= PARAMETERS["dimension_max"]).all(axis=0) & (
        dimensions[:, b] <= PARAMETERS["dimension_max"]
    ).all(axis=0)
    direction_pass = (direction_ratio <= 1).all(axis=(0, 2))
    denominator = np.sqrt(radius[a] * radius[b])
    span = np.divide(
        distance, denominator, out=np.full(len(pairs), np.inf), where=denominator > 1e-12
    )
    outward = np.zeros((len(pairs), 2), float)
    for k, (i, j) in enumerate(pairs):
        for side, (endpoint, sign) in enumerate([(int(i), 1), (int(j), -1)]):
            v = X[old_nn[endpoint]] - X[endpoint]
            tolerance = (
                PARAMETERS["numerical_tie_multiplier"]
                * np.finfo(float).eps
                * max(radius[endpoint], 1.0)
            )
            if len(v):
                outward[k, side] = float((v @ (sign * unit[k]) <= tolerance).mean())
    mutual2 = np.asarray(
        [int(j) in nn2[int(i)] and int(i) in nn2[int(j)] for (i, j) in pairs], bool
    )
    common = (distance > 1e-12) & low_dimension & direction_pass & (span <= PARAMETERS["span_max"])
    endpoint = (outward >= PARAMETERS["outward_fraction"]).all(axis=1)
    e2 = common & endpoint & mutual2
    cost = span * np.sqrt(np.maximum(direction_ratio.max(axis=(0, 2)), 0.0))
    raw_pair = np.sort(np.c_[raw[a], raw[b]], axis=1)
    contact_scores = {}
    for k in np.flatnonzero(e2):
        key = tuple(map(int, raw_pair[k]))
        contact_scores[key] = min(contact_scores.get(key, np.inf), float(cost[k]))
    best_cost = np.full(limits.shape[1], np.inf)
    for (ra, rb), value in contact_scores.items():
        best_cost[ra] = min(best_cost[ra], value)
        best_cost[rb] = min(best_cost[rb], value)
    best = np.zeros(len(pairs), bool)
    for k in np.flatnonzero(e2):
        (ra, rb) = map(int, raw_pair[k])
        value = contact_scores[ra, rb]
        tolerance = PARAMETERS["numerical_tie_multiplier"] * np.finfo(float).eps * max(1.0, value)
        best[k] = value <= best_cost[ra] + tolerance and value <= best_cost[rb] + tolerance
    decisions = dict(D2=common & mutual2, E2=e2, E3=common & endpoint, B2=e2 & best)
    arrays = dict(
        pairs=pairs,
        distance=distance,
        span=span,
        old_NN_radius=radius,
        query_neighbors=neighbors,
        q_graph_global2=q2,
        q_graph_global3=q3,
        endpoint_direction=factors,
        raw_internal_direction_limit=limits,
        direction_ratio=direction_ratio,
        low_dimension=low_dimension,
        direction_pass=direction_pass,
        endpoint_outward_fraction=outward,
        mutual_query2=mutual2,
        endpoint_pass=endpoint,
        contact_cost=cost,
        bilateral_best_contact=best,
        source_geometry_inverse=inverse,
        source_geometry_dimensions=dimensions,
        source_raw_ids=raw,
        source_NN_indptr=ptr,
        source_NN_indices=indices,
        source_q_graph=source_q,
    )
    for mode in MODES:
        arrays["admitted_" + mode] = decisions[mode]
    assert all((np.all(raw[pairs[keep, 0]] != raw[pairs[keep, 1]]) for keep in decisions.values()))
    return ({mode: pairs[decisions[mode]] for mode in MODES}, arrays)
