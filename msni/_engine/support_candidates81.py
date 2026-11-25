import numpy as np
from .evidence80 import features
from .post_quotient79 import coarsen
from .single_r6 import _core
from .sni_single_native import _pipeline


def prepare_evidence(X, a):
    (rows, _) = features(X, a)
    (shared, ref) = (a["SHARED"], a["REFERENCE"])
    (ptr, indices) = (shared["NN_indptr"], shared["NN_indices"])
    hosts = np.repeat(np.arange(len(shared["active"])), np.diff(ptr))
    original = ref["selected_seed_labels"] >= 0
    direct = np.bincount(
        ref["component_ids"][hosts], weights=original[indices], minlength=len(rows)
    )
    return (rows, direct)


def choose_supports(base, rows, direct, kind):
    if kind is None:
        return (np.empty(0, np.int64), np.zeros(len(rows), bool))
    admitted = np.zeros(len(rows), bool)
    scores = np.zeros(len(rows))
    for row in rows:
        cid = row["cut_id"]
        n = row["cut_size"]
        cap = row["original_cap"]
        valid = not row["original_selected"] and cap > 0 and (n >= 3 * row["q_graph_max"])
        valid &= row["NN_valid_points"] == row["field_valid_points"] == n
        limit = 0.95 if kind == "S95" else 0.99
        valid &= row["NN_internal_mean"] >= limit and row["field_internal_mean"] >= limit
        if kind in ("S95", "S99"):
            valid &= cap == 1
        if kind in ("S99", "U99"):
            valid &= row["NN_internal_q25"] == 1.0
        if kind == "U99":
            valid &= direct[cid] == 0
        admitted[cid] = valid
        scores[cid] = n * row["NN_internal_mean"] * row["field_internal_mean"]
    choices = []
    raw_for_cut = np.asarray([r["raw_id"] for r in rows], np.int64)
    for raw in range(len(base["raw_parent"])):
        candidates = np.flatnonzero(admitted & (raw_for_cut == raw))
        if len(candidates):
            choices.append(int(sorted(candidates, key=lambda c: (-scores[c], int(c)))[0]))
    return (np.asarray(choices, np.int64), admitted)


def rebuild_baseline(X, a, chosen):
    old = a["BASELINE"]
    ref = a["REFERENCE"]
    ids = old["component_ids"]
    raw = old["raw_ids"]
    current = X[old["active"]]
    before = current.tobytes()
    reference = ref["selected_seed_labels"].copy()
    assert np.array_equal(reference, old["reference_seeds"])
    total = len(old["seed_quotient"])
    caps = old["reference_caps"].copy()
    owners = []
    for offset, cid in enumerate(chosen):
        points = np.flatnonzero(ref["component_ids"] == cid)
        owner = np.unique(raw[points])
        assert len(owner) == 1 and len(points) > 0 and (reference[points] < 0).all()
        rid = int(owner[0])
        assert caps[rid] > 0
        caps[rid] += 1
        owners.append(rid)
        reference[points] = total + offset
    assert len(owners) == len(set(owners))
    assert np.array_equal(
        reference[old["reference_seeds"] >= 0], old["reference_seeds"][old["reference_seeds"] >= 0]
    )
    quotient = np.full(total + len(chosen), -1, np.int64)
    supported = []
    for cid in np.unique(ids):
        points = np.flatnonzero(ids == cid)
        labels = np.unique(reference[points])
        labels = labels[labels >= 0]
        if len(labels):
            root = int(labels.min())
            assert (quotient[labels] < 0).all()
            quotient[labels] = root
            supported.append((root, points))
    assert (quotient >= 0).all()
    seeds = np.full(len(current), -1, np.int64)
    for label, (_, points) in enumerate(sorted(supported, key=lambda t: t[0])):
        seeds[points] = label
    parents = {rid: int(parent) for (rid, parent) in enumerate(old["raw_parent"])}
    clean = _core.dpc_assign_remaining_clean_points_v94(
        current, seeds, old["rho"], ids, raw, parents
    )
    (labels, witness) = _pipeline.assign(X, old["active"], clean, _core)
    uses = np.asarray(
        [len(np.unique(seeds[(raw == rid) & (seeds >= 0)])) for rid in range(len(caps))], np.int64
    )
    assert (
        np.all(uses <= caps)
        and np.all(caps >= old["reference_caps"])
        and np.all(caps <= old["reference_caps"] + 1)
    )
    assert np.array_equal(clean[seeds >= 0], seeds[seeds >= 0]) and np.all(clean >= 0)
    assert np.all((old["seeds"] < 0) | (seeds >= 0)) and current.tobytes() == before
    result = dict(old)
    result.update(
        labels=labels,
        active_labels=clean,
        seeds=seeds,
        selected_mask=seeds >= 0,
        reference_seeds=reference,
        seed_quotient=quotient,
        reference_caps=caps,
        used_caps=uses,
        restoration_witness=witness,
        original_reference_seeds=old["reference_seeds"],
        original_reference_caps=old["reference_caps"],
        promotion_cut_ids=chosen,
        promotion_raw_ids=np.asarray(owners, np.int64),
        promotion_new_reference_ids=np.arange(total, total + len(chosen), dtype=np.int64),
    )
    return result


def small_contacts(base, geometry, kind):
    pairs = geometry["pairs"]
    raw = base["raw_ids"]
    nraw = len(base["raw_parent"])
    counts = np.zeros(nraw, np.int64)
    unique = np.full(nraw, -1, np.int64)
    sizes = np.bincount(raw, minlength=nraw)
    qs = np.zeros(nraw, np.int64)
    for rid in range(nraw):
        values = np.unique(base["seeds"][(raw == rid) & base["selected_mask"]])
        counts[rid] = len(values)
        if len(values) == 1:
            unique[rid] = values[0]
        qs[rid] = geometry["source_q_graph"][raw == rid].max()
    current = (counts[raw[pairs]] == 1).all(axis=1)
    current &= (base["active_labels"][pairs] == unique[raw[pairs]]).all(axis=1)
    current &= base["active_labels"][pairs[:, 0]] != base["active_labels"][pairs[:, 1]]
    distance_pass = geometry["distance"] > 1e-12
    direction_pass = (geometry["direction_ratio"] <= 1.0).all(axis=(0, 2))
    assert np.array_equal(direction_pass, geometry["direction_pass"])
    span_pass = geometry["span"] <= 2.0
    outward_pass = (
        np.ones(len(pairs), bool)
        if kind is None
        else (geometry["endpoint_outward_fraction"] >= (0.75 if kind == "X75" else 0.5)).all(axis=1)
    )
    geometry_pass = distance_pass & direction_pass & span_pass & outward_pass
    keys = np.sort(raw[pairs], axis=1)
    candidates = []
    diagnostics = []
    for ra, rb in np.unique(keys, axis=0):
        (small, big) = (int(ra), int(rb)) if sizes[ra] <= sizes[rb] else (int(rb), int(ra))
        scope = (keys == [ra, rb]).all(axis=1) & current
        ps = pairs[scope]
        small_points = np.unique(ps[raw[ps] == small])
        big_points = np.unique(ps[raw[ps] == big])
        bounded = sizes[small] <= 3 * qs[small] and sizes[big] >= 3 * sizes[small]
        quorum = (
            len(small_points) >= 2
            and len(big_points) >= 2
            and (len(small_points) / sizes[small] >= 0.125)
        )
        positions = np.flatnonzero(scope & geometry_pass)
        diagnostics.append(
            [
                small,
                big,
                int(scope.sum()),
                len(small_points),
                len(big_points),
                len(small_points) / sizes[small],
                int(bounded),
                int(quorum),
                len(positions),
                float(geometry["contact_cost"][positions].min()) if len(positions) else np.inf,
            ]
        )
        if kind is not None and bounded and quorum and len(positions):
            candidates.append(
                (small, big, float(geometry["contact_cost"][positions].min()), positions)
            )
    accepted = np.zeros(len(pairs), bool)
    best = np.full(nraw, np.inf)
    for small, _, cost, _ in candidates:
        best[small] = min(best[small], cost)
    for small, _, cost, positions in candidates:
        tolerance = 64 * np.finfo(float).eps * max(1.0, abs(cost))
        if cost <= best[small] + tolerance:
            accepted[positions] = True
    return (
        pairs[accepted],
        dict(
            pairs=pairs,
            geometry_pass=geometry_pass,
            current_group_eligibility=current,
            distance_pass=distance_pass,
            direction_pass=direction_pass,
            span_pass=span_pass,
            outward_pass=outward_pass,
            raw_pair_diagnostics=np.asarray(diagnostics, float).reshape(-1, 10),
            accepted=accepted,
            current_raw_counts=counts,
            current_raw_unique=unique,
            raw_sizes=sizes,
            raw_q_graph=qs,
            best_small_contact_cost=best,
        ),
    )
