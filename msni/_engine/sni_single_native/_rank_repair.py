import numpy as np
from . import _pipeline

MODES = ("Q2", "Q3", "L2", "L3", "L3Q")


def rebuild(X, ranks, source, core):
    (neighbors, _) = core._query_knn_excluding_self(X, core.KDTree(X), int(ranks.max()))
    NN = []
    for row, q, old in zip(neighbors, ranks, source["NN"]):
        ns = set(map(int, row[:q]))
        if not old <= ns:
            ns = set(old)
            for j in row:
                if len(ns) >= q:
                    break
                ns.add(int(j))
        assert len(ns) == q and old <= ns
        NN.append(ns)
    RNN = [set() for _ in X]
    for i, ns in enumerate(NN):
        for j in ns:
            RNN[j].add(i)
    return _pipeline.derive_base(X, NN, RNN, int(ranks.max()), core)


def repair(X, ranks, base, mode, core):
    assert mode in MODES
    factor = 2 if mode in ("Q2", "L2") else 3
    endpoints = 2 if mode == "L3Q" else 1
    n = len(X)
    ranks = np.asarray(ranks, dtype=np.int64)
    target = np.minimum(n - 1, factor * ranks)
    local = mode.startswith("L")
    ids = base["component_ids_raw"]
    radii = np.asarray(
        [
            max((float(np.linalg.norm(X[i] - X[j])) for j in ns), default=0.0)
            for (i, ns) in enumerate(base["NN"])
        ]
    )
    (neighbors, _) = core._query_knn_excluding_self(X, core.KDTree(X), int(target.max()))
    query = np.asarray(neighbors, dtype=np.int64)
    expanded = [set(map(int, row[:q])) for (row, q) in zip(query, target)]
    pairs = np.asarray(
        [
            (i, j)
            for (i, ns) in enumerate(expanded)
            for j in sorted(ns)
            if i < j and i in expanded[j] and (ids[i] != ids[j])
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    distance = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    threshold = 2 * np.sqrt(radii[pairs[:, 0]] * radii[pairs[:, 1]])
    eligible = distance <= threshold
    groups = {}
    for k in np.flatnonzero(eligible):
        (i, j) = map(int, pairs[k])
        (ci, cj) = (int(ids[i]), int(ids[j]))
        if ci > cj:
            (ci, cj, i, j) = (cj, ci, j, i)
        groups.setdefault((ci, cj), []).append((int(k), i, j))
    left = np.zeros(len(pairs), dtype=np.int64)
    right = np.zeros(len(pairs), dtype=np.int64)
    for rows in groups.values():
        index = [k for (k, _, _) in rows]
        left[index] = len({i for (_, i, _) in rows})
        right[index] = len({j for (_, _, j) in rows})
    admitted = eligible & (left >= endpoints) & (right >= endpoints)
    touched = np.zeros(n, bool)
    touched[np.unique(pairs[admitted])] = True
    changed = np.where(touched, target, ranks) if local else target
    (oldptr, oldidx) = _pipeline.csr(base["NN"])
    arrays = dict(
        rank_repair_source_ranks=ranks,
        rank_repair_final_ranks=changed,
        rank_repair_old_raw_ids=ids,
        rank_repair_old_NN_indptr=oldptr,
        rank_repair_old_NN_indices=oldidx,
        rank_repair_radius=radii,
        rank_repair_query_neighbors=query,
        rank_repair_contact_pairs=pairs,
        rank_repair_contact_distance=distance,
        rank_repair_contact_threshold=threshold,
        rank_repair_contact_eligible=eligible,
        rank_repair_contact_left=left,
        rank_repair_contact_right=right,
        rank_repair_contact_admitted=admitted,
        rank_repair_touched=touched,
        rank_repair_multiplier=np.asarray(factor),
        rank_repair_endpoint_quorum=np.asarray(endpoints),
        rank_repair_local=np.asarray(local),
    )
    assert np.all(changed >= ranks) and np.all(changed < n)
    return (changed, arrays)
