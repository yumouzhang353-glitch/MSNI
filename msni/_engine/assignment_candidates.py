import heapq
import numpy as np
from .single_c5 import _core
from .sni_single_native import _pipeline

MODES = {
    "A1_NN_COUNT_FIXED": ("count", False),
    "A2_NN_COUNT_PROP": ("count", True),
    "A3_NN_RAW_FIXED": ("raw", False),
    "A4_NN_RAW_PROP": ("raw", True),
    "A5_NN_LOCAL_FIXED": ("local", False),
    "A6_NN_LOCAL_PROP": ("local", True),
    "A7_NN_FIELD_FIXED": ("field", False),
    "A8_NN_FIELD_PROP": ("field", True),
}
(RTOL, ATOL) = (1e-12, 1e-15)


def prepare_neighbors(X, shared):
    ptr = shared["NN_indptr"].astype(np.int64)
    indices = shared["NN_indices"].astype(np.int64)
    hosts = np.repeat(np.arange(len(X)), np.diff(ptr))
    assert np.all(hosts != indices) and np.all((indices >= 0) & (indices < len(X)))
    distance = np.linalg.norm(X[indices] - X[hosts], axis=1)
    mu = shared["original_scale"]
    local = distance / (np.sqrt(mu[hosts] * mu[indices]) + _core.EPS)
    pair_lookup = {}
    for (a, b), ab, ba in zip(shared["raw_edges"], shared["G_ij"], shared["G_ji"]):
        pair_lookup[int(a), int(b)] = float(ab)
        pair_lookup[int(b), int(a)] = float(ba)
    field = np.asarray([pair_lookup.get((int(i), int(j)), 0.0) for (i, j) in zip(hosts, indices)])
    assert np.isfinite(distance).all() and np.isfinite(local).all()
    assert np.isfinite(field).all() and np.all(field >= 0)
    reverse = [[] for _ in X]
    for i, j in zip(hosts, indices):
        reverse[int(j)].append(int(i))
    return dict(
        indptr=ptr, indices=indices, raw=distance, local=local, field=field, reverse=reverse
    )


def identify(i, known, original_support, neighbors, kind):
    sl = slice(int(neighbors["indptr"][i]), int(neighbors["indptr"][i + 1]))
    js = neighbors["indices"][sl]
    values = known[js]
    valid = values >= 0
    record = dict(
        reason=2,
        winner=-1,
        confidence=0.0,
        runner_up=0.0,
        labeled_count=int(valid.sum()),
        winning_count=0,
        original_contributors=0,
        propagated_contributors=0,
    )
    if not valid.any():
        return record
    (groups, inv) = np.unique(values[valid], return_inverse=True)
    counts = np.bincount(inv, minlength=len(groups))
    if kind == "count":
        scores = counts.astype(float) / max(len(js), 1)
    elif kind in ("raw", "local"):
        sums = np.bincount(inv, weights=neighbors[kind][sl][valid], minlength=len(groups))
        scores = counts / max(len(js), 1) * np.exp(-sums / counts)
    elif kind == "field":
        weights = neighbors["field"][sl]
        denominator = float(weights.sum())
        scores = (
            np.bincount(inv, weights=weights[valid], minlength=len(groups)) / denominator
            if denominator > 0
            else np.zeros(len(groups))
        )
    else:
        raise ValueError(kind)
    best = int(np.argmax(scores))
    top = float(scores[best])
    second = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.0
    record.update(confidence=top, runner_up=second)
    if top <= 0:
        record["reason"] = 3
        return record
    if np.count_nonzero(np.isclose(scores, top, rtol=RTOL, atol=ATOL)) != 1:
        record["reason"] = 4
        return record
    contributors = js[valid][inv == best]
    original = int(original_support[contributors].sum())
    record.update(
        reason=1,
        winner=int(groups[best]),
        winning_count=int(counts[best]),
        original_contributors=original,
        propagated_contributors=int(counts[best]) - original,
    )
    return record


def allocate(seeds, old_labels, neighbors, mode):
    (kind, propagate) = MODES[mode]
    selected = seeds >= 0
    assert selected.any()
    known = seeds.copy()
    clean = seeds.copy()
    remaining = np.flatnonzero(~selected)
    trace = {
        key: np.zeros(
            len(seeds), dtype=np.int64 if key not in ("confidence", "runner_up") else float
        )
        for key in [
            "reason",
            "winner",
            "confidence",
            "runner_up",
            "labeled_count",
            "winning_count",
            "original_contributors",
            "propagated_contributors",
        ]
    }
    trace["winner"][:] = -1
    trace["order"] = np.full(len(seeds), -1, np.int64)
    trace["order"][selected] = -2
    trace["winner"][selected] = seeds[selected]

    def record_at(i, record):
        for key, value in record.items():
            trace[key][i] = value

    if not propagate:
        step = 0
        for i in remaining:
            record = identify(int(i), known, selected, neighbors, kind)
            record_at(i, record)
            if record["reason"] == 1:
                clean[i] = record["winner"]
                trace["order"][i] = step
                step += 1
            else:
                clean[i] = old_labels[i]
    else:
        versions = np.zeros(len(seeds), np.int64)
        queue = []
        records = {}

        def update(i):
            versions[i] += 1
            record = identify(i, known, selected, neighbors, kind)
            records[i] = record
            if record["reason"] == 1:
                heapq.heappush(queue, (-record["confidence"], i, int(versions[i])))

        for i in remaining:
            update(int(i))
        step = 0
        while queue:
            (neg_confidence, i, version) = heapq.heappop(queue)
            if known[i] >= 0 or version != versions[i]:
                continue
            record = records[i]
            assert record["reason"] == 1 and neg_confidence == -record["confidence"]
            record_at(i, record)
            clean[i] = known[i] = record["winner"]
            trace["order"][i] = step
            step += 1
            for host in neighbors["reverse"][i]:
                if known[host] < 0:
                    update(host)
        for i in np.flatnonzero(known < 0):
            record = identify(int(i), known, selected, neighbors, kind)
            assert record["reason"] != 1
            record_at(i, record)
            clean[i] = old_labels[i]
    assert np.all(clean >= 0) and np.array_equal(clean[selected], seeds[selected])
    assert set(map(int, clean)) <= set(map(int, seeds[selected]))
    return (clean, trace)


def apply(X, c5, neighbors, mode):
    (clean, trace) = allocate(c5["seeds"], c5["active_labels"], neighbors, mode)
    (labels, witness) = _pipeline.assign(X, c5["active"], clean, _core)
    output = dict(c5)
    output.update(
        labels=labels,
        active_labels=clean,
        restoration_witness=witness,
        assignment_recomputed=np.asarray(True),
        fragment_assignment_recomputed=np.asarray(True),
        multi_neighbor_assignment=np.asarray(True),
        propagated=np.asarray(MODES[mode][1]),
    )
    trace.update(
        changed_active=clean != c5["active_labels"], original_C5_active_labels=c5["active_labels"]
    )
    assert np.array_equal(labels[c5["active"]], clean)
    assert np.array_equal(clean[c5["selected_mask"]], c5["seeds"][c5["selected_mask"]])
    mask = c5["screening_mask"]
    assert np.array_equal(labels[mask], labels[witness[mask]])
    return (output, trace)
