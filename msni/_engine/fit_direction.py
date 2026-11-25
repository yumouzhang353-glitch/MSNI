from contextlib import contextmanager
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .locate import locate, neighborhood
from . import trace171_nd as pipeline


def detect(x):
    if x.shape[1] != 2 or len(x) < 32:
        return {"decision": "fallback_dimension_or_size"}
    out = locate(x)
    if len(out["routes"]) != 1:
        return {"decision": "fallback_route_count", "routes": len(out["routes"])}
    rt = out["routes"][0]
    if any((r is None for r in rt["readings"])):
        return {"decision": "fallback_no_reading", "route": rt}
    scores = np.array([v["score"] for v in rt["readings"]])
    decision = (
        "within"
        if np.all(scores >= 0.2)
        else "cross"
        if np.all(scores <= -0.2)
        else "fallback_weak_or_conflicting"
    )
    return {"decision": decision, "route": rt}


def components(n, pairs):
    if len(pairs) == 0:
        return np.arange(n)
    return connected_components(
        coo_matrix((np.ones(len(pairs)), pairs.T), shape=(n, n)), directed=False
    )[1]


def apply_decision(X, active, roles, ref, target, audit, trace):
    assert "input_sha" not in trace and np.array_equal(
        roles["pairs"], audit["source_roles"]["pairs"]
    )
    pairs = roles["pairs"]
    trace.update(
        input_sha=hashlib.sha256(X.tobytes(order="C")).hexdigest(),
        native_ref=ref.copy(),
        native_target=target.copy(),
        native_removed=audit["removed_pair_mask"].copy(),
        link=[],
        cuts=[],
        decision="pending",
    )
    route = detect(X)
    trace.update(route)
    decision = trace["decision"]
    if decision.startswith("fallback"):
        return (ref, target)
    rt = trace["route"]
    (p, q) = np.asarray(rt["ends"])
    (_, dist, knn) = neighborhood(X)
    k = (knn.shape[1] - 1) // 2
    centers = np.asarray([v["center"] for v in rt["readings"]])
    if decision == "within":
        comp = components(len(active), pairs[target])
        sizes = np.bincount(comp)
        chosen = []
        for center in centers:
            lengths = np.linalg.norm(X - center, axis=1)
            root = np.argmin(lengths)
            assert lengths[root] == 0
            rows = np.flatnonzero(lengths[active] <= dist[root, k] * (1 + 1e-12))
            if len(rows) == 0:
                trace["decision"] = "fallback_no_retained_endpoint"
                return (ref, target)
            order = np.lexsort((active[rows], lengths[active[rows]], -sizes[comp[rows]]))
            chosen.append(int(rows[order[0]]))
        trace["representatives"] = active[chosen].tolist()
        if comp[chosen[0]] != comp[chosen[1]]:
            trace["link"] = sorted(chosen)
        return (ref, target)
    axis = (q - p) / np.linalg.norm(q - p)
    normal = np.array([-axis[1], axis[0]])
    mid = (p + q) * 0.5
    u = (X[active] - mid) @ axis
    v = (X[active] - mid) @ normal
    widths = [dist[np.argmin(np.linalg.norm(X - tip, axis=1)), k] for tip in (p, q)]
    width = np.mean(widths)
    (a, b) = pairs.T
    ids = np.flatnonzero(target & ((u[a] < 0) != (u[b] < 0)))
    t = -u[a[ids]] / (u[b[ids]] - u[a[ids]])
    offset = v[a[ids]] + t * (v[b[ids]] - v[a[ids]])
    cut = ids[np.abs(offset) <= width]
    trace["cuts"] = cut.tolist()
    (ref, target) = (ref.copy(), target.copy())
    ref[cut] = False
    target[cut] = False
    audit["reference_keep"] = ref
    audit["target_keep"] = target
    audit["removed_pair_mask"] = audit["removed_pair_mask"].copy()
    audit["removed_pair_mask"][cut] = True
    assert not np.any(ref & ~target)
    return (ref, target)


@contextmanager
def before_support(trace):

    def apply(X, active, ev, ref, target, audit):
        return apply_decision(X, active, ev, ref, target, audit, trace)

    def link(graph):
        if trace["link"]:
            (i, j) = trace["link"]
            assert j not in graph[i] and i not in graph[j]
            graph[i].add(j)
            graph[j].add(i)

    assert pipeline._interface_masks is None and pipeline._interface_link is None
    pipeline._interface_masks = apply
    pipeline._interface_link = link
    try:
        yield
    finally:
        pipeline._interface_masks = None
        pipeline._interface_link = None
