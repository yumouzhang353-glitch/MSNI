import numpy as np


def coarsen(base, edges):
    groups = np.unique(base["labels"])
    parent = np.arange(len(groups))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    group_edges = base["active_labels"][edges]
    for a, b in np.searchsorted(groups, group_edges):
        (ra, rb) = (find(int(a)), find(int(b)))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.asarray([find(i) for i in range(len(groups))])
    (_, image) = np.unique(roots, return_inverse=True)

    def convert(labels):
        return image[np.searchsorted(groups, labels)]

    selected = base["seeds"] >= 0
    seeds = base["seeds"].copy()
    seeds[selected] = convert(seeds[selected])
    clean = convert(base["active_labels"])
    labels = convert(base["labels"])
    ref = base["reference_seeds"]
    ref_images = []
    for seed in range(len(base["seed_quotient"])):
        members = np.unique(clean[ref == seed])
        assert len(members) == 1
        ref_images.append(int(members[0]))
    ref_images = np.asarray(ref_images)
    quotient = np.asarray(
        [np.flatnonzero(ref_images == group)[0] for group in ref_images], np.int64
    )
    uses = np.asarray(
        [
            len(np.unique(seeds[(base["raw_ids"] == cid) & selected]))
            for cid in range(len(base["raw_parent"]))
        ],
        np.int64,
    )
    assert np.all(uses <= base["used_caps"])
    output = dict(base)
    output.update(
        labels=labels,
        active_labels=clean,
        seeds=seeds,
        seed_quotient=quotient,
        used_caps=uses,
        per_raw_group_uses=uses,
        global_seed_groups=np.asarray(len(np.unique(seeds[selected]))),
        post_merge_edges=edges,
        post_group_edges=group_edges,
        post_source_group_to_final_group=np.c_[groups, image],
    )
    for key in ["old_final_group_to_final_group", "Q1_group_to_final_group"]:
        if key in base:
            output["prefix_" + key] = base[key]
            output[key] = np.c_[base[key][:, 0], convert(base[key][:, 1])]
    assert np.array_equal(labels[base["active"]], clean) and np.array_equal(
        clean[selected], seeds[selected]
    )
    assert np.array_equal(output["selected_mask"], selected) and np.array_equal(
        output["reference_caps"], base["reference_caps"]
    )
    mask = base["screening_mask"]
    assert np.array_equal(labels[mask], labels[base["restoration_witness"][mask]])
    return output
