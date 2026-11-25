import numpy as np
from .assignment_candidates import MODES as ACTIVE_MODES, allocate

MODES = {a.replace("A", "R", 1).replace("_NN_", "_FULL_"): a for a in ACTIVE_MODES}


def witnesses(result, labels, trace, neighbors, mode):
    propagate = ACTIVE_MODES[MODES[mode]][1]
    active_mask = ~result["screening_mask"]
    root = np.full(len(labels), -1, np.int64)
    root[active_mask] = np.flatnonzero(active_mask)
    parent = np.full(len(labels), -1, np.int64)
    accepted = np.flatnonzero(trace["order"] >= 0)
    for i in accepted[np.argsort(trace["order"][accepted])]:
        sl = slice(int(neighbors["indptr"][i]), int(neighbors["indptr"][i + 1]))
        js = neighbors["indices"][sl]
        allowed = active_mask[js]
        if propagate:
            allowed |= (trace["order"][js] >= 0) & (trace["order"][js] < trace["order"][i])
        allowed &= labels[js] == labels[i]
        if ACTIVE_MODES[MODES[mode]][0] == "field":
            allowed &= neighbors["field"][sl] > 0
        positions = np.flatnonzero(allowed)
        assert len(positions)
        pos = min(positions, key=lambda pos: (float(neighbors["raw"][sl][pos]), int(js[pos])))
        j = int(js[pos])
        assert root[j] >= 0 and active_mask[root[j]] and (labels[root[j]] == labels[i])
        parent[i] = j
        root[i] = root[j]
    fallback = result["screening_mask"] & (trace["reason"] != 1)
    parent[fallback] = root[fallback] = result["restoration_witness"][fallback]
    assert np.all(root[result["screening_mask"]] >= 0)
    assert active_mask[root[result["screening_mask"]]].all()
    assert np.array_equal(labels[result["screening_mask"]], labels[root[result["screening_mask"]]])
    root[active_mask] = -1
    return (root, parent)


def apply(result, neighbors, mode):
    anchors = np.full(len(result["labels"]), -1, np.int64)
    anchors[result["active"]] = result["active_labels"]
    (labels, trace) = allocate(anchors, result["labels"], neighbors, MODES[mode])
    (root, parent) = witnesses(result, labels, trace, neighbors, mode)
    output = dict(result)
    output.update(
        labels=labels,
        restoration_witness=root,
        restoration_parent=parent,
        multi_neighbor_restoration=np.asarray(True),
        restoration_propagated=np.asarray(ACTIVE_MODES[MODES[mode]][1]),
        restoration_witness_is_nearest_active=np.asarray(False),
    )
    trace.update(original_A8_labels=result["labels"], changed_full=labels != result["labels"])
    assert np.array_equal(labels[result["active"]], result["active_labels"])
    return (output, trace)
