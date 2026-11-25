import time
import numpy as np
from .core_peel92 import prepare as original_prepare, graph_from

MODES = ("C0", "B_G", "BP_G", "BR_G", "B_C", "BP_C", "BR_C")


def group_support(base, pairs, field, ids, members, q_graph):
    n = len(ids)
    groups = int(ids.max()) + 1 if n else 0
    for key in ("G_ij", "G_ji"):
        assert (
            field[key].shape == (len(pairs),)
            and np.isfinite(field[key]).all()
            and (field[key] >= 0).all()
        )
    (inside_count, outside_count) = (np.zeros(n, np.int64), np.zeros(n, np.int64))
    (inside_mass, outside_mass) = (np.zeros(n), np.zeros(n))
    for k, (a, b) in enumerate(pairs):
        (a, b) = (int(a), int(b))
        for host, other, mass in ((a, b, field["G_ij"][k]), (b, a, field["G_ji"][k])):
            if not members[host] or other not in base["NN"][host]:
                continue
            if members[other] and ids[host] == ids[other]:
                inside_count[host] += 1
                inside_mass[host] += float(mass)
            else:
                outside_count[host] += 1
                outside_mass[host] += float(mass)
    active = np.flatnonzero(members)
    owners = ids[active]

    def add(values):
        return np.bincount(owners, weights=values[active], minlength=groups)

    sizes = np.bincount(owners, minlength=groups)
    maxima = np.zeros(groups, np.int64)
    np.maximum.at(maxima, owners, q_graph[active])
    valid = np.asarray([bool(row) for row in base["NN"]])
    (counts_in, counts_out) = (
        add(inside_count).astype(np.int64),
        add(outside_count).astype(np.int64),
    )
    (mass_in, mass_out) = (add(inside_mass), add(outside_mass))
    nn_valid = add(valid).astype(np.int64)
    field_valid = add(inside_mass + outside_mass > 0).astype(np.int64)
    tolerance = 64 * np.finfo(float).eps * (mass_in + mass_out)
    qualified = (sizes > 0) & (sizes >= 3 * maxima) & (nn_valid == sizes) & (field_valid == sizes)
    qualified &= (counts_in >= counts_out) & (mass_in + tolerance >= mass_out) & (mass_in > 0)
    return dict(
        size=sizes,
        q_graph_max=maxima,
        NN_valid_points=nn_valid,
        field_valid_points=field_valid,
        internal_NN_count=counts_in,
        external_NN_count=counts_out,
        internal_G_mass=mass_in,
        external_G_mass=mass_out,
        G_dominance_tolerance=tolerance,
        qualified=qualified,
    )


def regional_guard(base, ev, field, reference_keep, target_keep, candidate, q_graph, core, protect):
    n = len(candidate)
    pairs = ev["pairs"]
    assert (
        q_graph.shape == candidate.shape
        and np.issubdtype(q_graph.dtype, np.integer)
        and (q_graph >= 0).all()
    )
    core_keep = reference_keep & (~candidate[pairs]).all(axis=1)
    region_keep = target_keep & candidate[pairs].all(axis=1)
    (core_ids, _) = core.weak_connected_components(graph_from(n, pairs, core_keep))
    (region_ids, _) = core.weak_connected_components(graph_from(n, pairs, region_keep))
    cores = group_support(base, pairs, field, core_ids, ~candidate, q_graph)
    regions = group_support(base, pairs, field, region_ids, candidate, q_graph)
    contacts = {}
    for a, b in pairs[target_keep & (candidate[pairs].sum(axis=1) == 1)]:
        (a, b) = (int(a), int(b))
        (host, other) = (a, b) if candidate[a] else (b, a)
        (rid, cid) = (int(region_ids[host]), int(core_ids[other]))
        if not cores["qualified"][cid]:
            continue
        (left, right) = contacts.setdefault((rid, cid), (set(), set()))
        left.add(host)
        right.add(other)
    rows = np.asarray(
        [
            [rid, cid, len(left), len(right)]
            for ((rid, cid), (left, right)) in sorted(contacts.items())
        ],
        np.int64,
    ).reshape(-1, 4)
    admitted = (rows[:, 2] >= 2) & (rows[:, 3] >= 2)
    counts = np.bincount(rows[admitted, 0], minlength=len(regions["size"]))
    bridge = (regions["size"] > 0) & (counts >= 2)
    protected = bridge & regions["qualified"] if protect else np.zeros(len(bridge), bool)
    selected = candidate & bridge[region_ids] & ~protected[region_ids]
    return (
        selected,
        dict(
            guard_enabled=True,
            core_role_ids=core_ids,
            region_role_ids=region_ids,
            temporary_core_keep=core_keep,
            temporary_region_keep=region_keep,
            core_support=cores,
            region_support=regions,
            contact_rows=rows,
            contact_admitted=admitted,
            independent_core_contacts=counts,
            bridge_region=bridge,
            protected_region=protected,
            sparse_region_protection_enabled=bool(protect),
        ),
    )


def prepare(mode, base, ev, field, reference_keep, target_keep, core, q_graph):
    assert mode in MODES
    level_mode = "C0" if mode == "C0" else "P_G" if mode.endswith("_G") else "P_C"
    level_started = time.perf_counter()
    record = original_prepare(level_mode, base, ev, field, reference_keep, target_keep, core)
    record["raw_level_prepare_seconds"] = time.perf_counter() - level_started
    candidate = record["candidate_peel"].copy()
    q_graph = np.asarray(q_graph)
    assert (
        q_graph.shape == candidate.shape
        and np.issubdtype(q_graph.dtype, np.integer)
        and (q_graph >= 0).all()
    )
    q_graph = q_graph.astype(np.int64, copy=False)
    record.update(
        mode=mode,
        level_mode=level_mode,
        level_candidate_peel=candidate,
        source_q_graph=q_graph.copy(),
        source_rho=np.asarray(base["rho"]).copy(),
    )
    if mode == "C0":
        record.update(guard_enabled=False, original_core_budget_calls=0, regional_guard_seconds=0.0)
    else:
        regional_started = time.perf_counter()
        (filtered, guard) = regional_guard(
            base,
            ev,
            field,
            reference_keep,
            target_keep,
            candidate,
            q_graph,
            core,
            mode.startswith(("BP_", "BR_")),
        )
        record.update(
            guard,
            candidate_peel=filtered,
            original_core_budget_calls=0,
            regional_guard_seconds=time.perf_counter() - regional_started,
        )
    return record
