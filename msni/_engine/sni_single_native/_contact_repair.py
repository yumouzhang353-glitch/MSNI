import numpy as np
from . import _geometry

MODES = ("ALL", "C8", "C4", "C8Q", "C4Q", "C8G", "C8B")
EXTENDED = ("W4H", "W8H", "W8M", "T_L", "T_A", "W4T", "W8T")
MODES += EXTENDED


def apply(X, ev, reference, native_keep, native_before, mode, core):
    assert mode in MODES
    n = len(X)
    pairs = ev["pairs"]
    refpairs = reference["ev"]["pairs"]
    old_keep = reference["original_U_keep"]
    old_graph = [set() for _ in X]
    for a, b in refpairs[old_keep]:
        old_graph[int(a)].add(int(b))
        old_graph[int(b)].add(int(a))
    (ids, components) = core.weak_connected_components(old_graph)
    sizes = np.asarray(list(map(len, components)), dtype=np.int64)
    ref_positions = {tuple(map(int, edge)): k for (k, edge) in enumerate(refpairs)}
    mapped = np.asarray(
        [ref_positions.get(tuple(map(int, edge)), -1) for edge in pairs], dtype=np.int64
    )
    original = mapped >= 0
    preserved = np.zeros(len(pairs), dtype=bool)
    preserved[original] = old_keep[mapped[original]]
    assert int(preserved.sum()) == int(old_keep.sum())
    direction = _geometry.factors(X, pairs, reference["shape"])[-1].max(axis=0)
    old_direction = reference["field"]["edge_direction_factor"]
    limits = np.ones(len(components), dtype=np.float64)
    for cid in range(len(components)):
        internal = old_keep & (ids[refpairs[:, 0]] == cid)
        if internal.any():
            limits[cid] = float(np.percentile(old_direction[internal], 75))
    groups = {}
    for k in np.flatnonzero(native_keep):
        (a, b) = map(int, pairs[k])
        (ca, cb) = (int(ids[a]), int(ids[b]))
        if ca == cb:
            continue
        if ca > cb:
            (ca, cb, a, b) = (cb, ca, b, a)
        groups.setdefault((ca, cb), []).append((int(k), a, b))
    keys = np.asarray(sorted(groups), dtype=np.int64).reshape(-1, 2)
    pair_group = np.full(len(pairs), -1, dtype=np.int64)
    count = len(keys)
    edge_counts = np.zeros(count, dtype=np.int64)
    endpoints = np.zeros((count, 2), dtype=np.int64)
    old_endpoints = np.zeros((count, 2), dtype=np.int64)
    old_counts = np.zeros(count, dtype=np.int64)
    coverage = np.zeros(count, dtype=np.float64)
    median_direction = np.zeros(count, dtype=np.float64)
    median_third = np.zeros(count, dtype=np.float64)
    geometry_limits = np.zeros(count, dtype=np.float64)
    score = np.zeros(count, dtype=np.float64)
    for gid, key in enumerate(keys):
        (ca, cb) = map(int, key)
        entries = groups[ca, cb]
        ks = np.asarray([k for (k, _, _) in entries], dtype=np.int64)
        pair_group[ks] = gid
        (left, right) = ({a for (_, a, _) in entries}, {b for (_, _, b) in entries})
        prior = [(a, b) for (k, a, b) in entries if original[k]]
        (old_left, old_right) = ({a for (a, _) in prior}, {b for (_, b) in prior})
        edge_counts[gid] = len(entries)
        endpoints[gid] = (len(left), len(right))
        old_endpoints[gid] = (len(old_left), len(old_right))
        old_counts[gid] = len(prior)
        (fa, fb) = (len(left) / sizes[ca], len(right) / sizes[cb])
        coverage[gid] = fa if sizes[ca] <= sizes[cb] else fb
        median_direction[gid] = float(np.median(direction[ks]))
        median_third[gid] = float(np.median(ev["third_support"][ks]))
        geometry_limits[gid] = max(limits[ca], limits[cb])
        score[gid] = np.sqrt(fa * fb) * median_third[gid] / (1 + median_direction[gid])
    quorum = (endpoints >= 2).all(axis=1)
    c8 = quorum & (coverage >= 1 / 8)
    c4 = quorum & (coverage >= 1 / 4)
    old_quorum = (old_endpoints >= 2).all(axis=1)
    geometry_pass = median_direction <= geometry_limits
    best_scores = np.zeros(len(components), dtype=np.float64)
    for gid in np.flatnonzero(c8):
        (ca, cb) = keys[gid]
        best_scores[ca] = max(best_scores[ca], score[gid])
        best_scores[cb] = max(best_scores[cb], score[gid])
    tol = 64 * np.finfo(np.float64).eps
    best = np.zeros(count, dtype=bool)
    for gid in np.flatnonzero(c8):
        (ca, cb) = keys[gid]
        best[gid] = (
            score[gid] + tol * max(1.0, best_scores[ca]) >= best_scores[ca]
            and score[gid] + tol * max(1.0, best_scores[cb]) >= best_scores[cb]
        )
    decisions = dict(
        ALL=np.ones(count, dtype=bool),
        C8=c8,
        C4=c4,
        C8Q=c8 & old_quorum,
        C4Q=c4 & old_quorum,
        C8G=c8 & geometry_pass,
        C8B=best,
    )
    if mode in EXTENDED:
        dimensions = reference["shape"]["dimensions"]
        endpoint_dimensions = np.full((len(dimensions), count, 2), np.nan, dtype=np.float64)
        for gid, key in enumerate(keys):
            entries = groups[tuple(map(int, key))]
            if len(entries) == 1:
                (_, a, b) = entries[0]
                endpoint_dimensions[:, gid] = dimensions[:, [a, b]]
        ratio = median_direction / geometry_limits
        weak = (old_endpoints >= 1).all(axis=1)
        w4h = c4 & weak & (ratio <= 1 / 2)
        w8h = c8 & weak & (ratio <= 1 / 2)
        w8m = c8 & weak & (ratio <= 2 / 3)
        single = (edge_counts == 1) & (old_counts >= 1)
        low_large = (endpoint_dimensions[-1] <= 1.5).all(axis=1)
        low_all = (endpoint_dimensions <= 1.5).all(axis=(0, 2))
        tl = single & low_large & (ratio <= 1 / 2)
        ta = single & low_all & (ratio <= 1 / 2)
        base_decision = decisions["C4Q"]
        decisions.update(
            W4H=base_decision | w4h,
            W8H=base_decision | w8h,
            W8M=base_decision | w8m,
            T_L=base_decision | tl,
            T_A=base_decision | ta,
            W4T=base_decision | w4h | tl,
            W8T=base_decision | w8h | tl,
        )
    accepted = decisions[mode]
    crossing = pair_group >= 0
    allowed = np.zeros(len(pairs), dtype=bool)
    allowed[crossing] = accepted[pair_group[crossing]]
    same = ids[pairs[:, 0]] == ids[pairs[:, 1]]
    final = preserved | native_keep & (same | allowed)
    before = native_before | preserved
    assert not np.any(final & ~before)
    assert not np.any(preserved & ~final)
    arrays = dict(
        repair_mode_code=np.asarray(MODES.index(mode)),
        repair_native_keep=native_keep.copy(),
        repair_native_before=native_before.copy(),
        repair_before_keep=before,
        repair_preserved_U_keep=preserved,
        repair_original_pair_index=mapped,
        repair_reference_pairs=refpairs,
        repair_reference_U_keep=old_keep,
        repair_reference_component_ids=ids,
        repair_reference_component_sizes=sizes,
        repair_original_direction=direction,
        repair_reference_direction_limits=limits,
        repair_contact_keys=keys,
        repair_pair_group_id=pair_group,
        repair_contact_edges=edge_counts,
        repair_contact_endpoints=endpoints,
        repair_contact_old_endpoints=old_endpoints,
        repair_contact_old_edges=old_counts,
        repair_contact_coverage=coverage,
        repair_contact_median_direction=median_direction,
        repair_contact_direction_limit=geometry_limits,
        repair_contact_median_third=median_third,
        repair_contact_score=score,
        repair_contact_quorum=quorum,
        repair_contact_original_quorum=old_quorum,
        repair_contact_geometry_pass=geometry_pass,
        repair_contact_best=best,
        repair_component_best_score=best_scores,
        repair_contact_accepted=accepted,
        repair_restored_U_edges=preserved & ~native_keep,
        repair_removed_native_edges=native_keep & ~final,
        repair_final_keep=final,
    )
    if mode in EXTENDED:
        arrays.update(
            repair_reference_dimensions=dimensions,
            repair_contact_endpoint_dimensions=endpoint_dimensions,
            repair_contact_direction_ratio=ratio,
            repair_contact_weak_original_support=weak,
            repair_contact_single_original_supported=single,
            repair_contact_low_dimension_large=low_large,
            repair_contact_low_dimension_all=low_all,
            repair_contact_W4H=w4h,
            repair_contact_W8H=w8h,
            repair_contact_W8M=w8m,
            repair_contact_T_L=tl,
            repair_contact_T_A=ta,
        )
    return (final, before, arrays)
