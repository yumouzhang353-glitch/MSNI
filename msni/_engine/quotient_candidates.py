import numpy as np


def eligibility(base, edges):
    raw = base["raw_ids"]
    reference = base["reference_seeds"]
    counts = np.zeros(int(raw.max()) + 1, np.int64)
    unique_group = np.full(len(counts), -1, np.int64)
    for cid in range(len(counts)):
        labels = np.unique(reference[(raw == cid) & (reference >= 0)])
        counts[cid] = len(labels)
        if len(labels) == 1:
            image = np.unique(base["active_labels"][(raw == cid) & (reference == labels[0])])
            assert len(image) == 1
            unique_group[cid] = int(image[0])
    (a, b) = edges.T
    unique_raw = (counts[raw[a]] == 1) & (counts[raw[b]] == 1)
    consistent = (base["active_labels"][a] == unique_group[raw[a]]) & (
        base["active_labels"][b] == unique_group[raw[b]]
    )
    selected = (base["seeds"][a] >= 0) & (base["seeds"][b] >= 0)
    return (
        dict(Q0=np.ones(len(edges), bool), Q1=unique_raw, Q2=unique_raw & consistent, Q3=selected),
        counts,
        unique_group,
    )


def merge(base, admitted, allowed, counts, unique_group):
    groups = np.unique(base["labels"])
    lookup = {int(group): i for (i, group) in enumerate(groups)}
    parent = np.arange(len(groups))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    edges = admitted[allowed]
    source_pairs = base["active_labels"][edges]
    for a, b in source_pairs:
        (ra, rb) = (find(lookup[int(a)]), find(lookup[int(b)]))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.asarray([find(i) for i in range(len(groups))])
    (_, image) = np.unique(roots, return_inverse=True)

    def convert(labels):
        return image[np.searchsorted(groups, labels)]

    seeds = base["seeds"].copy()
    selected = seeds >= 0
    seeds[selected] = convert(seeds[selected])
    labels = convert(base["labels"])
    clean = convert(base["active_labels"])
    reference_image = []
    for label in range(len(base["seed_quotient"])):
        images = np.unique(clean[base["reference_seeds"] == label])
        assert len(images) == 1
        reference_image.append(int(images[0]))
    reference_image = np.asarray(reference_image)
    reference_quotient = np.asarray(
        [np.flatnonzero(reference_image == group).min() for group in reference_image]
    )
    uses = np.asarray(
        [
            len(np.unique(seeds[(base["raw_ids"] == cid) & selected]))
            for cid in range(len(base["raw_parent"]))
        ],
        np.int64,
    )
    output = dict(base)
    output.update(
        labels=labels,
        active_labels=clean,
        seeds=seeds,
        seed_quotient=reference_quotient,
        used_caps=uses,
        per_raw_group_uses=uses,
        global_seed_groups=np.asarray(len(np.unique(seeds[selected]))),
        added_cross_raw_edges=edges,
        contact_allowed_mask=allowed,
        old_final_group_to_final_group=np.c_[groups, image],
        quotient_group_edges=source_pairs,
        raw_reference_counts=counts,
        raw_unique_reference_group=unique_group,
        source_final_labels=base["labels"],
        source_active_labels=base["active_labels"],
        source_seeds=base["seeds"],
        source_seed_quotient=base["seed_quotient"],
        support_expanded=np.asarray(False),
        assignment_recomputed=np.asarray(False),
    )
    assert np.array_equal(output["selected_mask"], base["selected_mask"])
    assert np.array_equal(output["strong_edges"], base["strong_edges"])
    assert len(np.unique(labels)) <= len(groups)
    assert np.array_equal(labels[base["active"]], clean)
    mask = base["screening_mask"]
    assert np.array_equal(labels[mask], labels[base["restoration_witness"][mask]])
    return output
