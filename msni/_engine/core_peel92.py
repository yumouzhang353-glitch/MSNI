import numpy as np

MODES = ("C0", "P_G", "P_R", "P_C", "P_N", "P_TC", "P_GC")


def graph_from(n, pairs, keep):
    graph = [set() for _ in range(n)]
    for a, b in pairs[keep]:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    return graph


def nn_arrays(neighbors):
    rows = [sorted(map(int, row)) for row in neighbors]
    ptr = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64)
    indices = np.asarray([j for row in rows for j in row], np.int64)
    return (ptr, indices)


def node_level(mode, base, ev, field):
    n = len(base["NN"])
    counts = np.asarray(list(map(len, base["NN"])), np.int64)
    valid = counts > 0
    if mode == "C0":
        return (np.zeros(n), valid, np.zeros(n, bool), "none")
    value = ev["distance"] if mode in ("P_G", "P_R", "P_C", "P_N") else field["t"]
    (total, weighted, mass) = (np.zeros(n), np.zeros(n), np.zeros(n))
    for k, (a, b) in enumerate(ev["pairs"]):
        (a, b) = (int(a), int(b))
        for host, other, weight in ((a, b, field["G_ij"][k]), (b, a, field["G_ji"][k])):
            if other not in base["NN"][host]:
                continue
            total[host] += float(value[k])
            if mode == "P_GC":
                mass[host] += float(weight)
                weighted[host] += float(weight) * float(value[k])
    level = np.divide(total, counts, out=np.zeros(n), where=valid)
    fallback = np.zeros(n, bool)
    if mode == "P_GC":
        fallback = valid & (mass == 0)
        np.divide(weighted, mass, out=level, where=mass > 0)
    assert np.isfinite(level).all() and (level >= 0).all()
    name = (
        "raw_NN_mean"
        if mode in ("P_G", "P_R", "P_C", "P_N")
        else "final_t_NN_mean"
        if mode == "P_TC"
        else "final_G_weighted_t_NN_mean"
    )
    return (level, valid, fallback, name)


def background(mode, level, valid, raw_ids, reference_ids, NN):
    n = len(level)
    threshold = level.copy()
    if mode == "P_N":
        for i, row in enumerate(NN):
            neighbors = np.asarray(sorted((j for j in row if valid[int(j)])), np.int64)
            if len(neighbors):
                threshold[i] = float(level[neighbors].mean())
        return (threshold, np.full(n, -1, np.int64), "directed_NN")
    ids = (
        np.zeros(n, np.int64)
        if mode in ("C0", "P_G")
        else np.asarray(raw_ids, np.int64)
        if mode == "P_R"
        else np.asarray(reference_ids, np.int64)
    )
    for group in np.unique(ids):
        members = (ids == group) & valid
        if members.any():
            threshold[ids == group] = float(level[members].mean())
    name = (
        "none"
        if mode == "C0"
        else "global"
        if mode == "P_G"
        else "raw"
        if mode == "P_R"
        else "reference_C4Q"
    )
    return (threshold, ids.copy(), name)


def prepare(mode, base, ev, field, reference_keep, target_keep, core):
    assert mode in MODES
    n = len(base["NN"])
    pairs = ev["pairs"]
    assert pairs.shape[1:] == (2,) and np.all(~reference_keep | target_keep)
    (reference_ids, _) = core.weak_connected_components(graph_from(n, pairs, reference_keep))
    (level, valid, fallback, name) = node_level(mode, base, ev, field)
    (threshold, domain_ids, domain_kind) = background(
        mode, level, valid, base["component_ids_raw"], reference_ids, base["NN"]
    )
    candidate = valid & (level > threshold) if mode != "C0" else np.zeros(n, bool)
    (ptr, indices) = nn_arrays(base["NN"])
    return dict(
        mode=mode,
        level_name=name,
        domain_kind=domain_kind,
        level=level,
        threshold=threshold,
        valid_NN=valid,
        G_zero_mass_fallback=fallback,
        domain_ids=domain_ids,
        candidate_peel=candidate,
        source_reference_component_ids=reference_ids,
        source_raw_ids=base["component_ids_raw"].copy(),
        source_NN_indptr=ptr,
        source_NN_indices=indices,
        pairs=pairs,
        source_distance=ev["distance"],
        source_final_t=field["t"],
        source_final_G_ij=field["G_ij"],
        source_final_G_ji=field["G_ji"],
        source_reference_keep=reference_keep.copy(),
        source_target_keep=target_keep.copy(),
        coordinates_moved=False,
        active_nodes_removed=False,
        levels_frozen_before_peeling=True,
    )


def suffix(X, base, reference_before, record, core, quota_mode, budget_mode, anchor, _suffix=None):
    if _suffix is None:
        from .sni_single_native._quota_suffix import suffix_local

        _suffix = suffix_local
    n = len(X)
    pairs = record["pairs"]
    raw_ids = base["component_ids_raw"]
    effective = record["candidate_peel"].copy()
    rescued = np.zeros(len(base["components_raw"]), bool)
    history = []
    before = graph_from(n, pairs, reference_before)
    for iteration in range(len(rescued) + 1):
        node_keep = ~effective[pairs].any(axis=1)
        ref_keep = record["source_reference_keep"] & node_keep
        target_keep = record["source_target_keep"] & node_keep
        reference_graph = graph_from(n, pairs, ref_keep)
        (clean, selected, summary) = _suffix(
            X,
            base,
            before,
            reference_graph,
            core,
            quota_mode,
            budget_mode=budget_mode,
            anchor=anchor,
        )
        conflicting = effective & (selected["selected_seed_labels"] >= 0)
        bad = np.unique(raw_ids[conflicting]).astype(np.int64)
        history.append(
            dict(
                iteration=iteration,
                effective_peel=effective.copy(),
                reference_keep=ref_keep.copy(),
                target_keep=target_keep.copy(),
                provisional_component_ids=selected["component_ids"].copy(),
                provisional_reference_seeds=selected["selected_seed_labels"].copy(),
                caps=dict(summary["caps"]),
                conflicting_raw_ids=bad.copy(),
            )
        )
        if not len(bad):
            break
        assert not rescued[bad].any(), "budget rescue must make monotonic progress"
        rescued[bad] = True
        effective[np.isin(raw_ids, bad)] = False
    else:
        raise AssertionError("finite raw-component rescue bound")
    assert not np.any(effective & (selected["selected_seed_labels"] >= 0))
    assert np.all(~ref_keep | record["source_reference_keep"]) and np.all(
        ~target_keep | record["source_target_keep"]
    )
    assert np.all(~ref_keep | target_keep)
    final = dict(
        record,
        effective_peel=effective,
        rescued_raw_mask=rescued,
        rescue_history=history,
        budget_suffix_calls=len(history),
        reference_keep=ref_keep,
        target_keep=target_keep,
        reference_before_keep=reference_before.copy(),
        reference_seed_conflicts_final=0,
    )
    return (clean, selected, summary, reference_graph, graph_from(n, pairs, target_keep), final)
