import heapq
import time
import numpy as np
from .core_guard93 import prepare as regional_prepare

MODES = ("C0", "W_RAW_MIN", "W_RAW_MED", "W_UNIT_MIN", "W_UNIT_MED")
ROUND = 64 * np.finfo(float).eps


def union_find(n):
    parent = list(range(n))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def merge(a, b):
        (a, b) = (root(int(a)), root(int(b)))
        if a == b:
            return False
        parent[max(a, b)] = min(a, b)
        return True

    return (root, merge)


def tree_edges(n, pairs, weights):
    assert pairs.shape == (len(weights), 2) and np.isfinite(weights).all() and (weights >= 0).all()
    (root, merge) = union_find(n)
    chosen = []
    for k in np.lexsort((pairs[:, 1], pairs[:, 0], -weights)):
        if merge(*pairs[k]):
            chosen.append(int(k))
    return np.asarray(chosen, np.int64)


def adjacency(n, pairs, weights, lengths):
    graph = [[] for _ in range(n)]
    for k, (a, b) in enumerate(pairs):
        (a, b) = (int(a), int(b))
        w = float(weights[k])
        t = float(lengths[k])
        assert a != b and 0 <= a < n and (0 <= b < n) and np.isfinite(t) and (t >= 0)
        graph[a].append((b, w, t))
        graph[b].append((a, w, t))
    return graph


def widest(graph, sources):
    width = np.full(len(graph), -1.0)
    heap = []
    for source in sources:
        width[source] = np.inf
        heapq.heappush(heap, (-np.inf, int(source)))
    while heap:
        (negative, node) = heapq.heappop(heap)
        current = -negative
        if current != width[node]:
            continue
        for other, w, _ in graph[node]:
            value = min(current, w)
            if value > width[other]:
                width[other] = value
                heapq.heappush(heap, (-value, other))
    return width


def shortest(graph, sources):
    distance = np.full(len(graph), np.inf)
    heap = []
    for source in sources:
        distance[source] = 0.0
        heapq.heappush(heap, (0.0, int(source)))
    while heap:
        (current, node) = heapq.heappop(heap)
        if current != distance[node]:
            continue
        for other, _, length in graph[node]:
            value = current + length
            if value < distance[other]:
                distance[other] = value
                heapq.heappush(heap, (value, other))
    return distance


def field_weights(mode, record):
    left = record["source_final_G_ij"]
    right = record["source_final_G_ji"]
    pairs = record["pairs"]
    n = len(record["source_raw_ids"])
    assert (
        np.isfinite(left).all()
        and np.isfinite(right).all()
        and (left >= 0).all()
        and (right >= 0).all()
    )
    mass = np.zeros(n)
    np.add.at(mass, pairs[:, 0], left)
    np.add.at(mass, pairs[:, 1], right)
    assert np.isfinite(mass).all()
    if mode.startswith("W_UNIT_"):
        left = np.divide(
            left, mass[pairs[:, 0]], out=np.zeros_like(left), where=mass[pairs[:, 0]] > 0
        )
        right = np.divide(
            right, mass[pairs[:, 1]], out=np.zeros_like(right), where=mass[pairs[:, 1]] > 0
        )
    weights = np.sqrt(left) * np.sqrt(right)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    return (weights, mass)


def core_scales(record, weights, statistic):
    ids = record["core_role_ids"]
    initial = record["level_candidate_peel"]
    pairs = record["pairs"]
    result = {}
    proof = []
    for cid in np.flatnonzero(record["core_support"]["qualified"]):
        points = np.flatnonzero((ids == cid) & ~initial)
        index = {int(p): i for (i, p) in enumerate(points)}
        edge_ids = np.flatnonzero(record["temporary_core_keep"] & (ids[pairs[:, 0]] == cid))
        local = np.asarray(
            [[index[int(a)], index[int(b)]] for (a, b) in pairs[edge_ids]], np.int64
        ).reshape(-1, 2)
        chosen = tree_edges(len(points), local, weights[edge_ids])
        assert len(chosen) == len(points) - 1 and len(chosen) > 0
        values = weights[edge_ids[chosen]]
        reference = float(values.min() if statistic == "MIN" else np.median(values))
        result[int(cid)] = reference
        proof.append(
            dict(
                core_id=int(cid),
                points=points,
                tree_point_pair_ids=edge_ids[chosen],
                reference=reference,
            )
        )
    return (result, proof)


def local_graph(record, region, core_ids, weights):
    pairs = record["pairs"]
    initial = record["level_candidate_peel"]
    rids = record["region_role_ids"]
    cids = record["core_role_ids"]
    points = np.flatnonzero(initial & (rids == region))
    mapping = {int(p): i for (i, p) in enumerate(points)}
    terminals = {int(cid): len(points) + i for (i, cid) in enumerate(core_ids)}
    edges = {}

    def local(point):
        if int(point) in mapping:
            return mapping[int(point)]
        if not initial[point] and int(cids[point]) in terminals:
            return terminals[int(cids[point])]
        return None

    for k in np.flatnonzero(record["source_target_keep"]):
        (a, b) = pairs[k]
        (x, y) = (local(a), local(b))
        if x is None or y is None or x == y:
            continue
        key = tuple(sorted((x, y)))
        w = float(weights[k])
        t = float(record["source_final_t"][k])
        old = edges.get(key)
        if old is None or w > old[0] or (w == old[0] and (t, int(k)) < (old[1], old[2])):
            edges[key] = (w, t, int(k))
    keys = sorted(edges)
    local_pairs = np.asarray(keys, np.int64).reshape(-1, 2)
    w = np.asarray([edges[k][0] for k in keys])
    t = np.asarray([edges[k][1] for k in keys])
    point_edges = np.asarray([edges[k][2] for k in keys], np.int64)
    return (points, terminals, local_pairs, w, t, point_edges)


def compatibility(n, pairs, weights, terminals, references):
    chosen = tree_edges(n, pairs, weights)
    tree = adjacency(n, pairs[chosen], weights[chosen], np.zeros(len(chosen)))
    cids = sorted(terminals)
    matrix = np.empty((len(cids), len(cids)))
    for i, cid in enumerate(cids):
        matrix[i] = widest(tree, [terminals[cid]])[[terminals[c] for c in cids]]
    (root, merge) = union_find(len(cids))
    admitted = np.zeros_like(matrix, bool)
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            value = matrix[i, j]
            reference = min(references[cids[i]], references[cids[j]])
            tolerance = ROUND * max(abs(value), abs(reference))
            valid = value > 0 and value + tolerance >= reference
            admitted[i, j] = admitted[j, i] = valid
            if valid:
                merge(i, j)
    labels = np.asarray([root(i) for i in range(len(cids))], np.int64)
    return (labels, matrix, admitted, chosen)


def owners(graph, terminals, compat, point_count):
    cids = sorted(terminals)
    groups = sorted(set(map(int, compat)))
    widths = []
    distances = []
    for group in groups:
        sources = [terminals[cids[i]] for i in range(len(cids)) if compat[i] == group]
        widths.append(widest(graph, sources)[:point_count])
        distances.append(shortest(graph, sources)[:point_count])
    widths = np.asarray(widths)
    distances = np.asarray(distances)
    assignment = np.full(point_count, -1, np.int64)
    for point in range(point_count):
        column = widths[:, point]
        best = float(column.max())
        if best <= 0 or not np.isfinite(best):
            continue
        candidates = np.flatnonzero(best - column <= ROUND * abs(best))
        length = float(distances[candidates, point].min())
        if not np.isfinite(length):
            continue
        selected = candidates[np.abs(distances[candidates, point] - length) <= ROUND * abs(length)]
        if len(selected) == 1:
            assignment[point] = groups[int(selected[0])]
    return (assignment, widths, distances)


def partition(mode, base, ev, field, reference_keep, target_keep, core, q_graph):
    assert mode in MODES
    started = time.perf_counter()
    n = len(base["NN"])
    if mode == "C0":
        source = regional_prepare("C0", base, ev, field, reference_keep, target_keep, core, q_graph)
        return (
            reference_keep.copy(),
            target_keep.copy(),
            dict(
                mode=mode,
                enabled=False,
                source_roles=source,
                regions=[],
                core_references={},
                core_tree_proof=[],
                removed_pair_mask=np.zeros(len(ev["pairs"]), bool),
                source_reference_keep=reference_keep.copy(),
                source_target_keep=target_keep.copy(),
                reference_keep=reference_keep.copy(),
                target_keep=target_keep.copy(),
                coordinates_moved=False,
                active_nodes_removed=False,
                G_recomputed=False,
                seconds=time.perf_counter() - started,
            ),
        )
    record = regional_prepare("B_C", base, ev, field, reference_keep, target_keep, core, q_graph)
    (weights, mass) = field_weights(mode, record)
    (references, core_proof) = core_scales(record, weights, mode.rsplit("_", 1)[1])
    removed = np.zeros(len(ev["pairs"]), bool)
    regions = []
    pairs = record["pairs"]
    initial = record["level_candidate_peel"]
    for rid in np.flatnonzero(record["bridge_region"]):
        rows = record["contact_rows"]
        cids = np.asarray(
            sorted(map(int, rows[(rows[:, 0] == rid) & record["contact_admitted"], 1])), np.int64
        )
        assert len(cids) >= 2
        (points, terminals, local_pairs, w, t, point_edges) = local_graph(
            record, rid, cids, weights
        )
        count = len(points) + len(terminals)
        (compat, matrix, admitted, chosen) = compatibility(
            count, local_pairs, w, terminals, references
        )
        graph = adjacency(count, local_pairs, w, t)
        (assignment, widths, distances) = owners(graph, terminals, compat, len(points))
        point_owner = {int(p): int(label) for (p, label) in zip(points, assignment)}
        core_owner = {int(c): int(label) for (c, label) in zip(sorted(terminals), compat)}
        cuts = []
        if len(np.unique(compat)) > 1:
            for k in np.flatnonzero(target_keep):
                (a, b) = map(int, pairs[k])
                (ia, ib) = (a in point_owner, b in point_owner)
                if not ia and (not ib):
                    continue

                def label(point, inside):
                    if inside:
                        return point_owner[point]
                    return (
                        core_owner.get(int(record["core_role_ids"][point]), -1)
                        if not initial[point]
                        else -1
                    )

                (la, lb) = (label(a, ia), label(b, ib))
                if la >= 0 and lb >= 0 and (la != lb):
                    cuts.append(int(k))
        cut = np.asarray(cuts, np.int64)
        removed[cut] = True
        regions.append(
            dict(
                region_id=int(rid),
                points=points,
                core_ids=cids,
                terminal_local_ids=np.asarray([terminals[int(c)] for c in cids]),
                local_pairs=local_pairs,
                local_weights=w,
                local_t=t,
                local_point_pair_ids=point_edges,
                local_tree_edge_ids=chosen,
                core_bottleneck_matrix=matrix,
                core_compatibility=admitted,
                core_group_labels=compat,
                point_group_labels=assignment,
                group_width_scores=widths,
                group_independent_t_distances=distances,
                removed_point_pair_ids=cut,
                groups_after_compatibility=len(np.unique(compat)),
                unresolved_points=int((assignment < 0).sum()),
            )
        )
    final_ref = reference_keep & ~removed
    final_target = target_keep & ~removed
    assert np.all(~final_ref | final_target) and np.all(~removed | target_keep)
    audit = dict(
        mode=mode,
        enabled=True,
        source_roles=record,
        field_weights=weights,
        source_M_outgoing_mass=mass,
        core_references=references,
        core_tree_proof=core_proof,
        regions=regions,
        removed_pair_mask=removed,
        source_reference_keep=reference_keep.copy(),
        source_target_keep=target_keep.copy(),
        reference_keep=final_ref.copy(),
        target_keep=final_target.copy(),
        coordinates_moved=False,
        active_nodes_removed=False,
        G_recomputed=False,
        second_path_is_independent_t_shortest_not_shortest_widest=True,
        seconds=time.perf_counter() - started,
    )
    return (final_ref, final_target, audit)
