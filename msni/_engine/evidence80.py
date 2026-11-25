import numpy as np
from .single_r6 import prepare_neighbors


def connected_check(ids, edges):
    graph = [set() for _ in ids]
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    unseen = set(range(len(ids)))
    represented = set()
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo = [start]
        found = {start}
        while todo:
            for j in graph[todo.pop()] & unseen:
                unseen.remove(j)
                found.add(j)
                todo.append(j)
        values = set(map(int, ids[list(found)]))
        assert len(values) == 1
        group = next(iter(values))
        assert group not in represented
        represented.add(group)
    assert represented == set(map(int, ids))


def features(X, a):
    (ref, shared, baseline) = (a["REFERENCE"], a["SHARED"], a["BASELINE"])
    assert np.array_equal(shared["active"], baseline["active"])
    current = X[baseline["active"]]
    ids = ref["component_ids"]
    raw = shared["raw_ids"]
    connected_check(ids, ref["strong_edges"])
    selected = ref["selected_seed_labels"] >= 0
    assert np.array_equal(ref["selected_mask"], selected)
    nn = prepare_neighbors(current, shared)
    hosts = np.repeat(np.arange(len(current)), np.diff(nn["indptr"]))
    peers = nn["indices"]
    internal = ids[hosts] == ids[peers]
    total = np.diff(nn["indptr"])
    internal_count = np.bincount(hosts, weights=internal, minlength=len(current))
    internal_share = np.divide(internal_count, total, out=np.zeros(len(current)), where=total > 0)
    total_field = np.bincount(hosts, weights=nn["field"], minlength=len(current))
    inside_field = np.bincount(hosts, weights=nn["field"] * internal, minlength=len(current))
    field_share = np.divide(
        inside_field, total_field, out=np.zeros(len(current)), where=total_field > 0
    )
    rows = []
    for cid in np.unique(ids):
        points = np.flatnonzero(ids == cid)
        raws = np.unique(raw[points])
        assert len(raws) == 1
        rid = int(raws[0])
        supports = np.unique(ref["selected_seed_labels"][points])
        supports = supports[supports >= 0]
        assert len(supports) <= 1 and (
            selected[points].all() if len(supports) else not selected[points].any()
        )
        outgoing = (ids[hosts] == cid) & ~internal
        inside = (ids[hosts] == cid) & internal
        rows.append(
            dict(
                cut_id=int(cid),
                raw_id=rid,
                cut_size=len(points),
                raw_size=int((raw == rid).sum()),
                original_selected=bool(len(supports)),
                original_cap=int(baseline["reference_caps"][rid]),
                original_significant_min=float(ref["budget_current_significant_min"][rid]),
                q_graph_max=int(shared["rank_repair_source_ranks"][points].max()),
                NN_valid_points=int((total[points] > 0).sum()),
                NN_internal_mean=float(internal_share[points].mean()),
                NN_internal_q25=float(np.quantile(internal_share[points], 0.25)),
                NN_all_internal_fraction=float((internal_share[points] == 1).mean()),
                field_valid_points=int((total_field[points] > 0).sum()),
                field_internal_mean=float(field_share[points].mean()),
                rho_median=float(np.median(shared["rho"][points])),
                rho_q25=float(np.quantile(shared["rho"][points], 0.25)),
                scale_median=float(np.median(shared["original_scale"][points])),
                internal_local_q95=float(np.quantile(nn["local"][inside], 0.95))
                if inside.any()
                else None,
                outgoing_edges=int(outgoing.sum()),
                outgoing_local_min=float(nn["local"][outgoing].min()) if outgoing.any() else None,
                outgoing_local_median=float(np.median(nn["local"][outgoing]))
                if outgoing.any()
                else None,
            )
        )
    g = a["OLD_CANDIDATES"]
    pairs = g["pairs"]
    raw_pairs = np.sort(raw[pairs], axis=1)
    distances = np.linalg.norm(current[pairs[:, 0]] - current[pairs[:, 1]], axis=1)
    local = distances / (
        np.sqrt(shared["original_scale"][pairs[:, 0]] * shared["original_scale"][pairs[:, 1]])
        + 1e-12
    )
    contacts = []
    for ra, rb in np.unique(raw_pairs, axis=0):
        use = (raw_pairs == [ra, rb]).all(axis=1)
        ps = pairs[use]
        pa = np.unique(ps[raw[ps] == ra])
        pb = np.unique(ps[raw[ps] == rb])
        sizea = int((raw == ra).sum())
        sizeb = int((raw == rb).sum())
        contacts.append(
            dict(
                raw_a=int(ra),
                raw_b=int(rb),
                size_a=sizea,
                size_b=sizeb,
                q_graph_a_max=int(shared["rank_repair_source_ranks"][raw == ra].max()),
                q_graph_b_max=int(shared["rank_repair_source_ranks"][raw == rb].max()),
                contact_edges=int(use.sum()),
                endpoints_a=len(pa),
                endpoints_b=len(pb),
                coverage_a=len(pa) / sizea,
                coverage_b=len(pb) / sizeb,
                local_min=float(local[use].min()),
                local_median=float(np.median(local[use])),
                dimension_a_median=float(np.median(shared["geometry_dimensions"][:, raw == ra])),
                dimension_b_median=float(np.median(shared["geometry_dimensions"][:, raw == rb])),
                reference_groups_a=int(
                    len(np.unique(ref["selected_seed_labels"][(raw == ra) & selected]))
                ),
                reference_groups_b=int(
                    len(np.unique(ref["selected_seed_labels"][(raw == rb) & selected]))
                ),
                C5_groups_a=int(
                    len(np.unique(a["C5"]["seeds"][(raw == ra) & a["C5"]["selected_mask"]]))
                ),
                C5_groups_b=int(
                    len(np.unique(a["C5"]["seeds"][(raw == rb) & a["C5"]["selected_mask"]]))
                ),
                E2_edges=int(g["admitted_E2"][use].sum()),
                B2_edges=int(g["admitted_B2"][use].sum()),
                D2_edges=int(g["admitted_D2"][use].sum()),
                E3_edges=int(g["admitted_E3"][use].sum()),
            )
        )
    return (rows, contacts)
