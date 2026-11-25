import numpy as np


def contact_candidates(q1, shared, target):
    ids = target["repair_reference_component_ids"]
    counts = np.asarray(
        [
            len(np.unique(q1["reference_seeds"][(ids == c) & (q1["reference_seeds"] >= 0)]))
            for c in range(len(target["repair_reference_component_sizes"]))
        ]
    )
    keys = target["repair_contact_keys"]
    base = (
        target["repair_contact_quorum"]
        & (target["repair_contact_coverage"] >= 0.25)
        & target["repair_contact_original_quorum"]
    )
    mask = dict(
        C1=base | target["repair_contact_W4H"],
        C2=base | target["repair_contact_W8H"],
        C3=base | target["repair_contact_W8M"],
    )
    mask["C4"] = (target["repair_contact_endpoints"] >= 2).all(axis=1) & (
        target["repair_contact_old_endpoints"] >= 1
    ).all(axis=1)
    mask["C4"] &= (target["repair_contact_direction_ratio"] <= 1) & (counts[keys] == 1).all(axis=1)
    pairs = shared["raw_edges"]
    group = target["repair_pair_group_id"]
    crossing = group >= 0
    different = q1["active_labels"][pairs[:, 0]] != q1["active_labels"][pairs[:, 1]]
    additions = {}
    arrays = dict(
        pairs=pairs,
        contact_group=group,
        reference_component_ids=ids,
        reference_seed_counts=counts,
        contact_keys=keys,
        native_keep=target["repair_native_keep"],
        different_Q1_group=different,
    )
    for mode, decision in mask.items():
        accepted = np.zeros(len(pairs), bool)
        accepted[crossing] = decision[group[crossing]]
        keep = accepted & target["repair_native_keep"] & different
        additions[mode] = pairs[keep]
        arrays["contact_" + mode] = decision
        arrays["admitted_" + mode] = keep
    return (additions, arrays)
