import time
import numpy as np
from .core_partition94 import (
    partition as original_partition,
    regional_prepare,
    field_weights,
    core_scales,
    adjacency,
    compatibility,
    owners,
    tree_edges,
    union_find,
)

MODES = ("C0", "W_RAW_MIN", "W_UNIT_MIN", "W_RAW_MIN_EVID", "W_RAW_MIN_UCORE", "W_RAW_MIN_FULL")


def component_ids(n, pairs, keep):
    (root, merge) = union_find(n)
    for left, right in pairs[keep]:
        merge(left, right)
    roots = np.asarray([root(point) for point in range(n)], np.int64)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


def mapped_graph(source, points, core_ids, weights):
    pairs = source["pairs"]
    points = np.asarray(points, np.int64)
    cids = sorted(map(int, core_ids))
    index = {int(point): i for (i, point) in enumerate(points)}
    terminals = {cid: len(points) + i for (i, cid) in enumerate(cids)}
    (initial, roles) = (source["level_candidate_peel"], source["core_role_ids"])
    assert len(index) == len(points) and (
        not any((not initial[p] and int(roles[p]) in terminals for p in points))
    )
    edges = {}

    def local(point):
        if point in index:
            return index[point]
        if not initial[point]:
            return terminals.get(int(roles[point]))
        return None

    for edge in np.flatnonzero(source["source_target_keep"]):
        (left, right) = map(int, pairs[edge])
        (x, y) = (local(left), local(right))
        if x is None or y is None or x == y:
            continue
        key = tuple(sorted((x, y)))
        (weight, length) = (float(weights[edge]), float(source["source_final_t"][edge]))
        old = edges.get(key)
        if (
            old is None
            or weight > old[0]
            or (weight == old[0] and (length, int(edge)) < (old[1], old[2]))
        ):
            edges[key] = (weight, length, int(edge))
    keys = sorted(edges)
    return dict(
        points=points,
        terminals=terminals,
        pairs=np.asarray(keys, np.int64).reshape(-1, 2),
        weights=np.asarray([edges[key][0] for key in keys]),
        lengths=np.asarray([edges[key][1] for key in keys]),
        point_pair_ids=np.asarray([edges[key][2] for key in keys], np.int64),
    )


def admitted_contacts(source, point_mask):
    (pairs, roles) = (source["pairs"], source["core_role_ids"])
    qualified = ~source["level_candidate_peel"] & source["core_support"]["qualified"][roles]
    contacts = {}
    crossing = source["source_target_keep"] & (point_mask[pairs].sum(axis=1) == 1)
    for left, right in pairs[crossing]:
        (left, right) = (int(left), int(right))
        (point, endpoint) = (left, right) if point_mask[left] else (right, left)
        if qualified[endpoint]:
            cid = int(roles[endpoint])
            (inside, outside) = contacts.setdefault(cid, (set(), set()))
            inside.add(point)
            outside.add(endpoint)
    rows = np.asarray(
        [[cid, len(a), len(b)] for (cid, (a, b)) in sorted(contacts.items())], np.int64
    ).reshape(-1, 3)
    admitted = rows[(rows[:, 1] >= 2) & (rows[:, 2] >= 2), 0]
    return (rows, admitted)


def scopes(mode, source):
    (n, pairs) = (len(source["level_candidate_peel"]), source["pairs"])
    initial = source["level_candidate_peel"]
    seed = source["candidate_peel"]
    unqualified = ~initial & ~source["core_support"]["qualified"][source["core_role_ids"]]
    target_ids = component_ids(n, pairs, source["source_target_keep"])
    (prepared, skipped) = ([], [])
    if mode == "W_RAW_MIN_EVID":
        rows = source["contact_rows"]
        for rid in np.flatnonzero(source["bridge_region"]):
            points = np.flatnonzero(initial & (source["region_role_ids"] == rid))
            cids = np.asarray(
                sorted(map(int, rows[(rows[:, 0] == rid) & source["contact_admitted"], 1])),
                np.int64,
            )
            owner_cores = ~initial & np.isin(source["core_role_ids"], cids)
            component = int(target_ids[points[0]])
            assert (target_ids[points] == component).all()
            evidence_points = np.flatnonzero((target_ids == component) & ~owner_cores)
            prepared.append(
                dict(
                    region_id=int(rid),
                    points=points,
                    core_ids=cids,
                    evidence_points=evidence_points,
                    original_bridge_regions=np.asarray([rid], np.int64),
                    source_target_component=component,
                    expanded_contact_rows=np.empty((0, 3), np.int64),
                )
            )
        return (prepared, skipped)
    allowed = seed | unqualified if mode == "W_RAW_MIN_UCORE" else initial | unqualified
    if mode == "W_RAW_MIN_FULL":
        allowed &= np.isin(target_ids, np.unique(target_ids[seed]))
    ids = component_ids(n, pairs, source["source_target_keep"] & allowed[pairs].all(axis=1))
    for scope in np.unique(ids[seed]):
        point_mask = allowed & (ids == scope)
        points = np.flatnonzero(point_mask)
        (rows, cids) = admitted_contacts(source, point_mask)
        original = np.unique(source["region_role_ids"][points[seed[points]]]).astype(np.int64)
        assert len(original) and source["bridge_region"][original].all()
        if len(cids) < 2:
            skipped.append(
                dict(
                    scope_id=int(scope),
                    points=points,
                    contact_rows=rows,
                    reason="fewer_than_two_dual_endpoint_qualified_terminals",
                )
            )
            continue
        component = int(target_ids[points[0]])
        assert (target_ids[points] == component).all()
        prepared.append(
            dict(
                region_id=int(scope),
                points=points,
                core_ids=cids,
                evidence_points=points.copy(),
                original_bridge_regions=original,
                source_target_component=component,
                expanded_contact_rows=rows,
            )
        )
    return (prepared, skipped)


def partition(mode, base, ev, field, reference_keep, target_keep, core, q_graph):
    assert mode in MODES
    if mode in ("C0", "W_RAW_MIN", "W_UNIT_MIN"):
        return original_partition(mode, base, ev, field, reference_keep, target_keep, core, q_graph)
    started = time.perf_counter()
    source = regional_prepare("B_C", base, ev, field, reference_keep, target_keep, core, q_graph)
    (weights, mass) = field_weights("W_RAW_MIN", source)
    (references, core_proof) = core_scales(source, weights, "MIN")
    (regions, skipped) = scopes(mode, source)
    removed = np.zeros(len(ev["pairs"]), bool)
    (saved, assigned_domains) = ([], np.zeros(len(base["NN"]), bool))
    for scope in regions:
        (points, cids) = (scope["points"], scope["core_ids"])
        assert len(cids) >= 2 and (not assigned_domains[points].any())
        assigned_domains[points] = True
        local = mapped_graph(source, points, cids, weights)
        evidence = mapped_graph(source, scope["evidence_points"], cids, weights)
        count = len(evidence["points"]) + len(cids)
        (labels, matrix, admitted, evidence_tree) = compatibility(
            count, evidence["pairs"], evidence["weights"], evidence["terminals"], references
        )
        graph = adjacency(
            len(points) + len(cids), local["pairs"], local["weights"], local["lengths"]
        )
        (assignment, widths, distances) = owners(graph, local["terminals"], labels, len(points))
        point_owner = {int(point): int(owner) for (point, owner) in zip(points, assignment)}
        core_owner = {int(cid): int(owner) for (cid, owner) in zip(cids, labels)}
        cuts = []
        if len(np.unique(labels)) > 1:
            for edge in np.flatnonzero(target_keep):
                (left, right) = map(int, ev["pairs"][edge])
                (ia, ib) = (left in point_owner, right in point_owner)
                if not ia and (not ib):
                    continue

                def owner(point, inside):
                    if inside:
                        return point_owner[point]
                    if not source["level_candidate_peel"][point]:
                        return core_owner.get(int(source["core_role_ids"][point]), -1)
                    return -1

                (a, b) = (owner(left, ia), owner(right, ib))
                if a >= 0 and b >= 0 and (a != b):
                    cuts.append(int(edge))
        cut = np.asarray(cuts, np.int64)
        removed[cut] = True
        local_tree = tree_edges(len(points) + len(cids), local["pairs"], local["weights"])
        saved.append(
            dict(
                **scope,
                terminal_local_ids=np.asarray([local["terminals"][int(c)] for c in cids]),
                local_pairs=local["pairs"],
                local_weights=local["weights"],
                local_t=local["lengths"],
                local_point_pair_ids=local["point_pair_ids"],
                local_tree_edge_ids=local_tree,
                compatibility_graph=evidence,
                compatibility_tree_edge_ids=evidence_tree,
                core_bottleneck_matrix=matrix,
                core_compatibility=admitted,
                core_group_labels=labels,
                point_group_labels=assignment,
                group_width_scores=widths,
                group_independent_t_distances=distances,
                removed_point_pair_ids=cut,
                groups_after_compatibility=len(np.unique(labels)),
                unresolved_points=int((assignment < 0).sum()),
            )
        )
    (final_reference, final_target) = (reference_keep & ~removed, target_keep & ~removed)
    assert np.all(~final_reference | final_target) and np.all(~removed | target_keep)
    qualified_nodes = (
        ~source["level_candidate_peel"]
        & source["core_support"]["qualified"][source["core_role_ids"]]
    )
    assert not (assigned_domains & qualified_nodes).any()
    pairs = ev["pairs"]
    assert not (removed & source["temporary_core_keep"] & qualified_nodes[pairs].all(axis=1)).any()
    return (
        final_reference,
        final_target,
        dict(
            mode=mode,
            enabled=True,
            scope_mode=mode.rsplit("_", 1)[1],
            source_roles=source,
            field_weights=weights,
            source_M_outgoing_mass=mass,
            core_references=references,
            core_tree_proof=core_proof,
            regions=saved,
            skipped_scopes=skipped,
            ownership_scope_nodes=assigned_domains,
            newly_scoped_original_non_candidate_nodes=assigned_domains
            & ~source["level_candidate_peel"],
            removed_pair_mask=removed,
            source_reference_keep=reference_keep.copy(),
            source_target_keep=target_keep.copy(),
            reference_keep=final_reference.copy(),
            target_keep=final_target.copy(),
            coordinates_moved=False,
            active_nodes_removed=False,
            G_recomputed=False,
            second_path_is_independent_t_shortest_not_shortest_widest=True,
            compatibility_and_ownership_graphs_are_same_scope=mode != "W_RAW_MIN_EVID",
            seconds=time.perf_counter() - started,
        ),
    )
