import numpy as np

MODES = ("C0", "Q2", "R2", "Q2O", "R2O", "Q2S", "Q2L")


def qualify(q1, shared, target, c5_audit, mode):
    assert mode in MODES
    (labels, selected) = (q1["active_labels"], q1["selected_mask"])
    (raw, contacts, gate) = (shared["raw_edges"], c5_audit["CONTACTS"], c5_audit["GATE"])
    positions = np.flatnonzero(contacts["admitted_C4"])
    pairs = gate["pairs"]
    passed = gate["admitted_C5"]
    assert pairs.shape == (len(positions), 2) and np.array_equal(raw[positions], pairs)
    assert passed.dtype == bool and passed.shape == (len(pairs),)
    assert selected.dtype == bool and selected.shape == labels.shape
    assert (labels >= 0).all() and (pairs >= 0).all() and (pairs < len(labels)).all()
    assert len(np.unique(np.sort(pairs, axis=1), axis=0)) == len(pairs)
    repair = target["repair_pair_group_id"][positions]
    assert (repair >= 0).all()
    endpoint_groups = labels[pairs]
    assert (endpoint_groups[:, 0] != endpoint_groups[:, 1]).all()
    swap = endpoint_groups[:, 0] > endpoint_groups[:, 1]
    oriented = pairs.copy()
    oriented[swap] = oriented[swap, ::-1]
    group_pair = np.sort(endpoint_groups, axis=1)
    source_original = target["repair_original_pair_index"][positions] >= 0
    source_selected = selected[oriented]
    dimensions = target["repair_reference_dimensions"][-1, pairs]
    assert dimensions.dtype.kind == "f" and np.isfinite(dimensions).all()
    ref_ids = target["repair_reference_component_ids"][pairs]
    limits = target["repair_reference_direction_limits"][ref_ids].max(axis=1)
    direction = target["repair_original_direction"][positions]
    direction_ratio = np.divide(
        direction, limits, out=np.full(len(pairs), np.inf), where=limits > 0
    )
    edge_low = (dimensions <= 1.5).all(axis=1)
    domain_repair = repair if mode in ("R2", "R2O") else np.full(len(pairs), -1, np.int64)
    (keys, inverse) = np.unique(np.c_[group_pair, domain_repair], axis=0, return_inverse=True)
    k = len(keys)
    (candidate_edges, passed_edges, effective_edges) = [np.zeros(k, np.int64) for _ in range(3)]
    (candidate_endpoints, passed_endpoints, original_endpoints, selected_endpoints) = [
        np.zeros((k, 2), np.int64) for _ in range(4)
    ]
    (quorum, original_quorum, selected_quorum) = [np.zeros(k, bool) for _ in range(3)]
    (sole_original, sole_low, sole_direction, rescue, admitted) = [
        np.zeros(k, bool) for _ in range(5)
    ]
    effective = np.zeros(len(pairs), bool)
    for gid in range(k):
        ix = np.flatnonzero(inverse == gid)
        good = ix[passed[ix]]
        prior = good[source_original[good]]
        (candidate_edges[gid], passed_edges[gid]) = (len(ix), len(good))
        for side in range(2):
            candidate_endpoints[gid, side] = len(np.unique(oriented[ix, side]))
            passed_endpoints[gid, side] = len(np.unique(oriented[good, side]))
            original_endpoints[gid, side] = len(np.unique(oriented[prior, side]))
            selected_endpoints[gid, side] = len(
                np.unique(oriented[good[source_selected[good, side]], side])
            )
        quorum[gid] = (passed_endpoints[gid] >= 2).all()
        original_quorum[gid] = (original_endpoints[gid] >= 1).all()
        selected_quorum[gid] = (selected_endpoints[gid] >= 1).all()
        if len(good) == 1:
            ix1 = int(good[0])
            sole_original[gid] = source_original[ix1]
            sole_low[gid] = edge_low[ix1]
            sole_direction[gid] = direction_ratio[ix1] <= 0.5
        rescue[gid] = sole_original[gid] & sole_low[gid] & sole_direction[gid]
        if mode == "C0":
            admitted[gid] = True
        elif mode in ("Q2", "R2"):
            admitted[gid] = quorum[gid]
        elif mode in ("Q2O", "R2O"):
            admitted[gid] = quorum[gid] & original_quorum[gid]
        elif mode == "Q2S":
            admitted[gid] = quorum[gid] & selected_quorum[gid]
        else:
            admitted[gid] = quorum[gid] | rescue[gid]
        if admitted[gid]:
            effective[good] = True
        effective_edges[gid] = int(effective[ix].sum())
    assert not (effective & ~passed).any()
    if mode == "C0":
        assert np.array_equal(effective, passed)
    return dict(
        mode=mode,
        domain_kind="R" if mode in ("R2", "R2O") else "Q",
        pairs=pairs,
        raw_positions=positions,
        repair_contact_group=repair,
        oriented_endpoints=oriented,
        Q1_group_pairs=group_pair,
        domain_keys=keys,
        pair_domain=inverse,
        source_C5_pass=passed,
        effective_pass=effective,
        original_raw_source=source_original,
        selected_endpoints=source_selected,
        reference_endpoint_dimensions=dimensions,
        reference_direction_ratio=direction_ratio,
        edge_low_dimension=edge_low,
        candidate_edges=candidate_edges,
        passed_edges=passed_edges,
        effective_edges=effective_edges,
        candidate_endpoint_counts=candidate_endpoints,
        passed_endpoint_counts=passed_endpoints,
        passed_original_endpoint_counts=original_endpoints,
        passed_selected_endpoint_counts=selected_endpoints,
        quorum=quorum,
        original_quorum=original_quorum,
        selected_quorum=selected_quorum,
        sole_original=sole_original,
        sole_low_dimension=sole_low,
        sole_direction_pass=sole_direction,
        single_edge_rescue=rescue,
        domain_admitted=admitted,
    )
