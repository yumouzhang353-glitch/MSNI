def require(ok, message):
    if not ok:
        raise ValueError(message)


def components(n, edges):
    import numpy as np

    parent = np.arange(n, dtype=np.int64)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = int(parent[k])
        return k

    for a, b in edges:
        (x, y) = (find(int(a)), find(int(b)))
        if x != y:
            parent[max(x, y)] = min(x, y)
    if not n:
        return np.zeros(0, np.int64)
    return np.unique([find(k) for k in range(n)], return_inverse=True)[1].astype(np.int64)


def effective(weights):
    import numpy as np

    v = np.asarray(weights, np.float64)
    v = v[v > 0]
    return min(float(v.sum() ** 2 / np.square(v).sum()), float(len(v))) if len(v) else 0.0


def endpoint_evidence(p, w, edges, left_mask):
    import numpy as np

    by = [{}, {}]
    for e in edges:
        (a, b) = map(int, p[e])
        (left, right) = (a, b) if left_mask[a] else (b, a)
        for side, node in enumerate((left, right)):
            by[side][node] = by[side].get(node, 0.0) + float(w[e])
    values = [[v for (_, v) in sorted(table.items())] for table in by]
    neff = [effective(v) for v in values]
    eps64 = np.finfo(np.float64).eps
    return dict(
        source_U_edges=len(edges),
        positive_U_edges=sum((w[e] > 0 for e in edges)),
        endpoint_counts=[len(t) for t in by],
        positive_endpoint_counts=[sum((v > 0 for v in t)) for t in values],
        effective_endpoints=neff,
        neff2_pass=bool(all((v + 64 * eps64 * max(1.0, v) >= 2 for v in neff))),
        source_U_force=float(sum((w[e] for e in edges))),
    )


def projection(n, p, B, added, base, supported):
    import numpy as np

    labels = components(n, p[B | added])
    supported_ids = np.flatnonzero(supported)
    representatives = np.asarray([np.flatnonzero(base == k)[0] for k in supported_ids], np.int64)
    groups = {}
    for k, node in zip(supported_ids, representatives):
        groups.setdefault(int(labels[node]), []).append(int(k))
    multi = {k: v for (k, v) in groups.items() if len(v) > 1}
    return (
        labels,
        dict(
            edges=int((B | added).sum()),
            added_proposals=int(added.sum()),
            new_edges_not_in_B=int((added & ~B).sum()),
            connected_components=int(len(np.unique(labels))),
            components_joining_multiple_supported_base_components=len(multi),
            multiple_supported_examples=[
                dict(component=k, base_components=v) for (k, v) in list(multi.items())[:20]
            ],
            interpretation="Fixed post-bridge B projection, not rebridged model prediction",
        ),
    )


def diagnose(a, B, current_post, base_name, representation):
    import numpy as np

    (p, w) = (a["pairs"], a["normalized_weight"])
    n = len(a["active"])
    m = len(p)
    require(B.dtype == bool and B.shape == (m,), "Base mask domain")
    b = components(n, p[B])
    nb = int(b.max()) + 1 if n else 0
    bp = a["node_block_position"]
    node_H = a["internal_force"][bp] > 0
    supported = np.bincount(b[node_H], minlength=nb) > 0
    bpair = b[p]
    same = bpair[:, 0] == bpair[:, 1]
    touch = supported[bpair]
    considered = a["unknown_proposal_edge"].copy()
    if representation == "POS_U":
        considered &= w > 0
    else:
        require(representation == "U", "Representation fixed before labels")
    role = np.zeros(m, np.int8)
    role[considered & same] = 1
    role[considered & ~same & touch.all(axis=1)] = 2
    role[considered & ~same & (touch[:, 0] ^ touch[:, 1])] = 3
    role[considered & ~same & ~touch.any(axis=1)] = 4
    require(np.array_equal(role > 0, considered), "Every considered edge has one role")
    uunits = np.flatnonzero(~supported)
    reg_roots = components(nb, bpair[role == 4])
    reg_of_b = np.full(nb, -1, np.int64)
    if len(uunits):
        (_, inv) = np.unique(reg_roots[uunits], return_inverse=True)
        reg_of_b[uunits] = inv
    rnode = reg_of_b[b]
    nr = int(rnode.max()) + 1 if len(uunits) else 0
    contacts = {}
    direct = {}
    for e in np.flatnonzero(role == 3):
        (x, y) = map(int, p[e])
        (left, right) = (x, y) if rnode[x] >= 0 else (y, x)
        contacts.setdefault((int(rnode[left]), int(b[right])), []).append(int(e))
    for e in np.flatnonzero(role == 2):
        direct.setdefault(tuple(sorted(map(int, bpair[e]))), []).append(int(e))
    targets = [[] for _ in range(nr)]
    for region, target in sorted(contacts):
        targets[region].append(target)
    contact_count = np.asarray([len(v) for v in targets], np.int64)
    auxiliary = [
        dict(
            all_U_targets=set(),
            all_U_boundary_edges=0,
            all_U_boundary_force=0.0,
            all_U_to_other_unanchored_regions=0,
            all_U_S_targets=set(),
            all_U_S_boundary_edges=0,
            all_U_S_boundary_force=0.0,
            all_U_S_to_other_unanchored_regions=0,
        )
        for _ in range(nr)
    ]
    for label, mask in [("all_U", a["proposed_U"]), ("all_U_S", a["proposed_U_S"])]:
        for e in np.flatnonzero(mask):
            (x, y) = map(int, p[e])
            for node, other in ((x, y), (y, x)):
                region = int(rnode[node])
                if region < 0 or rnode[other] == region:
                    continue
                aux = auxiliary[region]
                if supported[b[other]]:
                    aux[label + "_targets"].add(int(b[other]))
                    aux[label + "_boundary_edges"] += 1
                    aux[label + "_boundary_force"] += float(w[e])
                else:
                    aux[label + "_to_other_unanchored_regions"] += 1
    strength = np.zeros(n)
    np.add.at(strength, p[:, 0], w)
    np.add.at(strength, p[:, 1], w)
    base_volume = np.bincount(b, weights=strength, minlength=nb)
    region_volume = np.bincount(rnode[rnode >= 0], weights=strength[rnode >= 0], minlength=nr)
    region_internal = np.zeros(nr)
    inside = (rnode[p[:, 0]] >= 0) & (rnode[p[:, 0]] == rnode[p[:, 1]])
    np.add.at(region_internal, rnode[p[inside, 0]], 2 * w[inside])
    (raw_keys, raw_pos) = np.unique(a["raw_ids"], return_inverse=True)
    require(np.array_equal(raw_keys, a["raw_keys"]), "Raw identity")
    base_raw = np.empty(nb, np.int64)
    for k in range(nb):
        vals = np.unique(raw_pos[b == k])
        require(len(vals) == 1, "B component single raw")
        base_raw[k] = vals[0]
    region_raw = np.empty(nr, np.int64)
    for k in range(nr):
        vals = np.unique(raw_pos[rnode == k])
        require(len(vals) == 1, "Region single raw")
        region_raw[k] = vals[0]
    cross = {}
    base_cross = {}
    for e, (x, y) in enumerate(p):
        (x, y) = (int(x), int(y))
        (rx, ry) = (int(rnode[x]), int(rnode[y]))
        (bx, by) = (int(b[x]), int(b[y]))
        if rx >= 0 and supported[by]:
            cross[rx, by] = cross.get((rx, by), 0.0) + float(w[e])
        if ry >= 0 and supported[bx]:
            cross[ry, bx] = cross.get((ry, bx), 0.0) + float(w[e])
        if bx != by and supported[bx] and supported[by]:
            key = tuple(sorted((bx, by)))
            base_cross[key] = base_cross.get(key, 0.0) + float(w[e])

    def force_test(C, v1, v2, r):
        T = float(a["raw_total_force"][r])
        lhs = float(C * T)
        rhs = float(v1 * v2)
        return dict(
            complete_E0_cross_force=float(C),
            left_volume=float(v1),
            right_volume=float(v2),
            raw_total_force=T,
            positive_volume=bool(v1 > 0 and v2 > 0 and (T > 0)),
            enrichment_lhs=lhs,
            enrichment_rhs=rhs,
            enriched=bool(v1 > 0 and v2 > 0 and (T > 0) and (lhs > rhs)),
        )

    contact_records = []
    for (region, target), edges in sorted(contacts.items()):
        require(region_raw[region] == base_raw[target], "Boundary single raw")
        contact_records.append(
            dict(
                region=region,
                target_base_component=target,
                **force_test(
                    cross.get((region, target), 0.0),
                    region_volume[region],
                    base_volume[target],
                    region_raw[region],
                ),
                **endpoint_evidence(p, w, edges, rnode == region),
            )
        )
    direct_records = []
    for (left, right), edges in sorted(direct.items()):
        require(base_raw[left] == base_raw[right], "Direct supported contact single raw")
        direct_records.append(
            dict(
                left_base_component=left,
                right_base_component=right,
                **force_test(
                    base_cross.get((left, right), 0.0),
                    base_volume[left],
                    base_volume[right],
                    base_raw[left],
                ),
                **endpoint_evidence(p, w, edges, b == left),
            )
        )
    quotient_edges = [{} for _ in range(nr)]
    for e in np.flatnonzero(role == 4):
        (left, right) = sorted(map(int, bpair[e]))
        region = int(reg_of_b[left])
        require(
            region == reg_of_b[right] and region >= 0,
            "Unknown internal proposal stays in its frozen region",
        )
        key = (left, right)
        quotient_edges[region][key] = quotient_edges[region].get(key, 0) + 1
    regions = []
    for region in range(nr):
        units = np.flatnonzero(reg_of_b == region)
        nodes = np.flatnonzero(rnode == region)
        edges = quotient_edges[region]
        cycles = len(edges) - len(units) + 1
        require(cycles >= 0, "Connected region quotient cycle rank")
        regions.append(
            dict(
                region=region,
                nodes=len(nodes),
                base_units=len(units),
                supported_targets=targets[region],
                target_count=len(targets[region]),
                raw_key=int(raw_keys[region_raw[region]]),
                internal_complete_E0_force=float(region_internal[region]),
                volume=float(region_volume[region]),
                quotient_distinct_edges=len(edges),
                quotient_cycle_rank=cycles,
                parallel_U_proposal_excess=sum(edges.values()) - len(edges),
            )
        )
        regions[-1]["auxiliary_all_source_contacts"] = {
            k: sorted(v) if isinstance(v, set) else v for (k, v) in auxiliary[region].items()
        }
    masks = {"SAME": considered & same, "LEAF": (role == 1) | (role == 4), "ALL": considered.copy()}
    for (region, target), edges in contacts.items():
        if contact_count[region] == 1:
            masks["LEAF"][edges] = True
    require(
        not (masks["SAME"] & ~masks["LEAF"]).any()
        and (not (masks["LEAF"] & ~masks["ALL"]).any())
        and (not (masks["ALL"] & ~a["unknown_proposal_edge"]).any()),
        "Nested projections of original unknown U",
    )
    arrays = dict(
        base_component=b,
        base_supported=supported,
        region_of_base=reg_of_b,
        region_of_node=rnode,
        considered=considered,
        edge_role=role,
        base_volume=base_volume,
        region_volume=region_volume,
        region_internal_force=region_internal,
        region_raw_position=region_raw,
        region_contact_count=contact_count,
        node_positive_internal=node_H,
        base_keep=B.copy(),
    )
    projections = {}
    for name, mask in masks.items():
        (lab, info) = projection(n, p, B, mask, b, supported)
        arrays["projection_" + name] = mask
        arrays["components_" + name] = lab
        projections[name] = info
        if name in ("SAME", "LEAF"):
            require(
                info["components_joining_multiple_supported_base_components"] == 0,
                "Fixed-B leaf property",
            )
    size = np.bincount(bp, minlength=len(a["block_keys"]))
    unknown_blocks = a["internal_force"] == 0
    summary = dict(
        base=base_name,
        representation=representation,
        n_active=n,
        base_components=nb,
        supported_base_components=int(supported.sum()),
        supported_node_count=int(supported[b].sum()),
        original_zero_internal_nodes_already_in_supported_base=int((~node_H & supported[b]).sum()),
        original_zero_internal_blocks=int(unknown_blocks.sum()),
        original_zero_internal_multinode_blocks=int((unknown_blocks & (size > 1)).sum()),
        source_unknown_U=int(a["unknown_proposal_edge"].sum()),
        considered_unknown_U=int(considered.sum()),
        excluded_zero_force_U=int((a["unknown_proposal_edge"] & ~considered).sum()),
        roles={
            name: int((role == k).sum())
            for (k, name) in [
                (1, "same_B"),
                (2, "different_supported_B"),
                (3, "supported_to_unanchored_B"),
                (4, "different_unanchored_B"),
            ]
        },
        regions=nr,
        region_contacts={
            name: dict(
                regions=int(mask.sum()),
                nodes=int(sum((v["nodes"] for v in regions if mask[v["region"]]))),
                boundary_U_edges=int(sum((len(e) for ((r, _), e) in contacts.items() if mask[r]))),
            )
            for (name, mask) in [
                ("zero", contact_count == 0),
                ("single", contact_count == 1),
                ("multiple", contact_count > 1),
            ]
        },
        full_projection_vs_actual_CURRENT=dict(
            edge_mask_equal=bool(np.array_equal(B | considered, current_post)),
            edge_mask_difference=int(np.count_nonzero((B | considered) ^ current_post)),
        ),
        projections=projections,
        interpretation="Frozen feature graph; supported components are not true clusters; no prediction or rebridge",
    )
    return (
        dict(
            summary=summary,
            regions=regions,
            contacts=contact_records,
            direct_supported_pairs=direct_records,
        ),
        arrays,
    )
