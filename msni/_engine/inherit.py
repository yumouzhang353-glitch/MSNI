from contextlib import contextmanager
from itertools import combinations
import sys
import numpy as np
from scipy.spatial import cKDTree
from . import fit_direction as prior
from . import trace171_nd as mod
from . import assignment_candidates as assignment
from . import support_candidates81 as promotions
from .locate import neighborhood, get_bodies, find_route


def geometry(X, active, pairs, target, route):
    (k, d, nn) = neighborhood(X)
    (bodies, thin) = get_bodies(X, k, nn)
    tips = np.asarray(route["ends"])
    matching = []
    for a, b in combinations(range(len(bodies)), 2):
        current = find_route(X, (bodies[a], bodies[b]), thin, d[:, k], k)
        if current and np.array_equal(current["ends"], tips):
            matching.append((a, b))
    assert len(matching) == 1
    ids = np.concatenate(bodies)
    owners = np.repeat(np.arange(len(bodies)), [len(b) for b in bodies])
    (distance, i) = cKDTree(X[ids]).query(X)
    (p, q) = tips
    axis = (q - p) / np.linalg.norm(q - p)
    t = (X - (p + q) / 2) @ axis
    normal = np.array([-axis[1], axis[0]])
    v = (X - (p + q) / 2) @ normal
    width = np.mean([d[np.argmin(np.linalg.norm(X - tip, axis=1)), k] for tip in tips])
    side = np.zeros(len(X), dtype=int)
    covered = distance <= 2 * d[ids[i], k]
    epsilon = 64 * np.finfo(float).eps * np.linalg.norm(q - p)
    tube = (np.abs(v) <= width) & (np.abs(t) <= 0.5 * np.linalg.norm(q - p) + 2 * width)
    for sign, idx in zip((-1, 1), matching[0]):
        side[(sign * t > epsilon) & (tube | covered & (owners[i] == idx))] = sign
    centers = np.array([e["center"] for e in route["readings"]])
    reach = [d[np.argmin(np.linalg.norm(X - center, axis=1)), k] for center in centers]
    return dict(
        side=side[active],
        active=active.copy(),
        X=X[active],
        centers=centers,
        reach=reach,
        target=prior.components(len(active), pairs[target]),
        full_side=side,
    )


def label_sides(seeds, side):
    valid = seeds >= 0
    owners = np.zeros(max(seeds) + 1, int)
    left = np.unique(seeds[valid & (side < 0)])
    right = np.unique(seeds[valid & (side > 0)])
    if not len(left) or not len(right) or np.intersect1d(left, right).size:
        return None
    owners[left] = -1
    owners[right] = 1
    return owners


def reserve(geometry, parts, raw, caps, native):
    return (native, dict(feasible=True, swaps=0, disabled=True))


def bounded_dpc(original, args, kwargs, geometry, audit):
    audit.append(dict(feasible=False, disabled=True))
    return original(*args, **kwargs)


@contextmanager
def inherited(audit):
    core = mod._core
    (decide, suffix, dpc, allocate, identify) = (
        prior.apply_decision,
        mod.suffix_local,
        core.dpc_assign_remaining_clean_points_v94,
        assignment.allocate,
        assignment.identify,
    )
    live = dict(g=None, vote=None)

    def decision(X, active, ev, ref, target, bound, trace):
        (r, t) = decide(X, active, ev, ref, target, bound, trace)
        g = (
            geometry(X, active, ev["pairs"], t, trace["route"])
            if trace["decision"] == "cross"
            else None
        )
        live.update(g=g, vote=None, reserved=False)
        rec = dict(decision=trace["decision"], support=[], dpc=[], a8=[])
        if g is not None:
            rec["owned_counts"] = [int(np.sum(g["side"] == s)) for s in (-1, 0, 1)]
            rec["_full_side"] = g["full_side"]
        audit.append(rec)
        return (r, t)

    def support(X, base, before, strong, scoped, quota_mode, **kwargs):
        if live["g"] is None or sys._getframe(1).f_code is not mod.run_single.__code__:
            return suffix(X, base, before, strong, scoped, quota_mode, **kwargs)

        class BudgetCore:
            def __getattr__(self, name):
                return getattr(scoped, name)

            def select_components_by_budget(self, parts, raw, rawgroups, caps):
                native = scoped.select_components_by_budget(parts, raw, rawgroups, caps)
                if sys._getframe(1).f_code is not suffix.__code__:
                    return native
                (result, rec) = reserve(live["g"], parts, raw, caps, native)
                audit[-1]["support"].append(rec)
                live["reserved"] = rec["feasible"]
                return result

        return suffix(X, base, before, strong, BudgetCore(), quota_mode, **kwargs)

    def assignment_dpc(*args, **kwargs):
        caller = sys._getframe(1).f_code
        if (
            live["g"] is None
            or not live["reserved"]
            or caller not in (mod.run_single.__code__, promotions.rebuild_baseline.__code__)
        ):
            return dpc(*args, **kwargs)
        return bounded_dpc(dpc, args, kwargs, live["g"], audit[-1]["dpc"])

    def run_allocate(seeds, old, neighbors, mode):
        g = live["g"]
        if g is None or not live["reserved"]:
            return allocate(seeds, old, neighbors, mode)
        owners = label_sides(seeds, g["side"])
        rec = dict(feasible=owners is not None, removed_votes=0, fallback_replaced=0)
        audit[-1]["a8"].append(rec)
        if owners is None:
            return allocate(seeds, old, neighbors, mode)
        fixed = seeds >= 0
        fallback = old.copy()
        for s in (-1, 1):
            ids = np.flatnonzero(~fixed & (g["side"] == s) & (owners[old] != s))
            if len(ids):
                anchors = np.flatnonzero(
                    fixed & (g["side"] == s) & (owners[np.maximum(seeds, 0)] == s)
                )
                assert len(anchors)
                for i in ids:
                    d = g["X"][anchors] - g["X"][i]
                    fallback[i] = seeds[anchors[np.argmin(np.einsum("ij,ij->i", d, d))]]
                rec["fallback_replaced"] += len(ids)
        live["vote"] = (owners, rec)
        try:
            (clean, trace) = allocate(seeds, fallback, neighbors, mode)
            use = g["side"] != 0
            assert np.array_equal(owners[clean[use]], g["side"][use])
            return (clean, trace)
        finally:
            live["vote"] = None

    def guarded_vote(i, known, fixed, neighbors, kind):
        if live["vote"] is None or not live["g"]["side"][i]:
            return identify(i, known, fixed, neighbors, kind)
        (owners, rec) = live["vote"]
        js = neighbors["indices"][neighbors["indptr"][i] : neighbors["indptr"][i + 1]]
        values = known[js]
        bad = js[(values >= 0) & (owners[np.maximum(values, 0)] != live["g"]["side"][i])]
        if not len(bad):
            return identify(i, known, fixed, neighbors, kind)
        saved = known[bad].copy()
        known[bad] = -1
        try:
            rec["removed_votes"] += len(bad)
            return identify(i, known, fixed, neighbors, kind)
        finally:
            known[bad] = saved

    try:
        prior.apply_decision = decision
        mod.suffix_local = support
        core.dpc_assign_remaining_clean_points_v94 = assignment_dpc
        (assignment.allocate, assignment.identify) = (run_allocate, guarded_vote)
        yield
    finally:
        prior.apply_decision = decide
        mod.suffix_local = suffix
        core.dpc_assign_remaining_clean_points_v94 = dpc
        (assignment.allocate, assignment.identify) = (allocate, identify)
