import numpy as np
from . import _pipeline


def expand(X, ranks, base, mode, core):
    assert mode in ("I2", "I3")
    factor = int(mode[1])
    source = np.asarray(ranks, dtype=np.int64)
    ids = base["component_ids_raw"]
    sizes = np.bincount(ids)
    target = np.minimum(factor * source, sizes[ids] - 1)
    assert np.all(target >= source)
    NN = [set() for _ in X]
    probes = [[] for _ in X]
    for raw, component in enumerate(base["components_raw"]):
        points = np.asarray(sorted(component), dtype=np.int64)
        if len(points) <= 1:
            for i in points:
                NN[int(i)] = set(base["NN"][int(i)])
            continue
        current = X[points]
        (neighbors, _) = core._query_knn_excluding_self(
            current, core.KDTree(current), int(target[points].max())
        )
        for position, row in enumerate(neighbors):
            i = int(points[position])
            q = int(target[i])
            candidates = points[np.asarray(row, dtype=np.int64)]
            probes[i] = list(map(int, candidates))
            ns = set(map(int, candidates[:q]))
            old = base["NN"][i]
            if not old <= ns:
                ns = set(old)
                for j in candidates:
                    if len(ns) >= q:
                        break
                    ns.add(int(j))
            assert len(ns) == q and old <= ns and all((ids[j] == raw for j in ns))
            NN[i] = ns
    (oldptr, oldidx) = _pipeline.csr(base["NN"])
    ptr = np.r_[0, np.cumsum([len(row) for row in probes])].astype(np.int64)
    indices = np.asarray([j for row in probes for j in row], dtype=np.int64)
    arrays = dict(
        rank_repair_source_ranks=source,
        rank_repair_final_ranks=target,
        rank_repair_old_raw_ids=ids,
        rank_repair_old_NN_indptr=oldptr,
        rank_repair_old_NN_indices=oldidx,
        intra_raw_sizes=sizes,
        intra_query_indptr=ptr,
        intra_query_indices=indices,
        intra_rank_multiplier=np.asarray(factor),
        intra_restricted_to_old_raw=np.asarray(True),
    )
    if np.array_equal(source, target):
        return (base, target, arrays)
    RNN = [set() for _ in X]
    for i, ns in enumerate(NN):
        for j in ns:
            RNN[j].add(i)
    revised = _pipeline.derive_base(X, NN, RNN, int(target.max()), core)
    assert np.array_equal(revised["component_ids_raw"], ids)
    return (revised, target, arrays)


def original_anchor(X, base, geometry_q, H, active, core, bridge_mode, quota_mode, tie_geometry):
    from . import _field, _geometry
    from ._budget_adapter import prepare_anchor

    ev = _field.evidence(X, base)
    f0 = _field.interaction(X, base, ev, "F0", core)
    field = _field.interaction(X, base, ev, "C1", core)
    shape = _geometry.prepare(
        X, int(geometry_q), base["reliability"], weighted=True, tie_closed=tie_geometry
    )
    field = _geometry.correct(X, base, ev, field, shape, "LW_GM", core)
    context = _field.threshold_context(ev, f0, field, len(X), "T8")
    (strong, _) = _field.build_strong_graph(ev, field, len(X), context)
    return prepare_anchor(X, base, strong, ev["pairs"], H, active, core, bridge_mode, quota_mode)
