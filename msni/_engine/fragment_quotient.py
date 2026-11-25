import numpy as np


def coarsen(q1, edges):
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    groups = np.unique(q1["labels"])
    parent = np.arange(len(groups))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    pairs = q1["active_labels"][edges]
    for a, b in np.searchsorted(groups, pairs):
        (ra, rb) = (find(int(a)), find(int(b)))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.asarray([find(i) for i in range(len(groups))])
    (_, image) = np.unique(roots, return_inverse=True)

    def convert(labels):
        return image[np.searchsorted(groups, labels)]

    (labels, clean) = (convert(q1["labels"]), convert(q1["active_labels"]))
    seeds = q1["seeds"].copy()
    selected = seeds >= 0
    seeds[selected] = convert(seeds[selected])
    reference_image = np.asarray(
        [
            int(np.unique(clean[q1["reference_seeds"] == i]).item())
            for i in range(len(q1["seed_quotient"]))
        ]
    )
    reference_quotient = np.asarray(
        [np.flatnonzero(reference_image == group).min() for group in reference_image]
    )
    uses = np.asarray(
        [
            len(np.unique(seeds[(q1["raw_ids"] == cid) & selected]))
            for cid in range(len(q1["raw_parent"]))
        ],
        np.int64,
    )
    old = q1["old_final_group_to_final_group"]
    output = dict(q1)
    output.update(
        labels=labels,
        active_labels=clean,
        seeds=seeds,
        seed_quotient=reference_quotient,
        used_caps=uses,
        per_raw_group_uses=uses,
        global_seed_groups=np.asarray(len(np.unique(seeds[selected]))),
        old_final_group_to_final_group=np.c_[old[:, 0], convert(old[:, 1])],
        fragment_merge_edges=edges,
        fragment_quotient_group_edges=pairs,
        Q1_group_to_final_group=np.c_[groups, image],
        source_Q1_labels=q1["labels"],
        source_Q1_active_labels=q1["active_labels"],
        source_Q1_seeds=q1["seeds"],
        source_Q1_seed_quotient=q1["seed_quotient"],
        fragment_support_expanded=np.asarray(False),
        fragment_assignment_recomputed=np.asarray(False),
    )
    assert len(np.unique(labels)) <= len(groups)
    assert np.array_equal(labels[q1["active"]], clean)
    assert np.array_equal(output["selected_mask"], q1["selected_mask"])
    assert np.array_equal(clean[selected], seeds[selected])
    mask = q1["screening_mask"]
    assert np.array_equal(labels[mask], labels[q1["restoration_witness"][mask]])
    return output
