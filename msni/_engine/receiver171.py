import numpy as np
from .sni_single_native._config import EPS

MODES = (
    "BASE_G",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "U",
    "C50",
    "C75",
    "CR75",
    "NN1",
    "NN2",
    "NN3",
    "NN4",
    "NN5",
    "U_S",
)
ROW_PERCENTILE = 30.0
CONTACT_ENDPOINTS = 2


def row_positive_threshold(pairs, values, n):
    rows = [[] for _ in range(n)]
    for k, (i, j) in enumerate(pairs):
        if values[k] > EPS:
            rows[int(i)].append(float(values[k]))
            rows[int(j)].append(float(values[k]))
    return np.asarray([np.percentile(row, ROW_PERCENTILE) if row else np.inf for row in rows])


def admit(X, base, ev, graph, original_keep, mode, core):
    assert mode in MODES
    pairs = ev["pairs"]
    (i, j) = (pairs[:, 0], pairs[:, 1])
    (ids, components) = core.weak_connected_components(graph)
    distance = ev["distance"]
    scales = np.asarray(
        [
            float(np.mean(np.linalg.norm(X[sorted(ns)] - X[k], axis=1))) if ns else 0.0
            for (k, ns) in enumerate(base["NN"])
        ]
    )
    forward = np.asarray([int(b) in base["NN"][int(a)] for (a, b) in pairs], bool)
    backward = np.asarray([int(a) in base["NN"][int(b)] for (a, b) in pairs], bool)
    assert np.all(forward | backward)
    assert np.all(base["component_ids_raw"][i] == base["component_ids_raw"][j])
    rejected_cross = ~original_keep & (ids[i] != ids[j])
    sigma = ev["original_sigma"]
    third = ev["third_support"]
    third_rows = row_positive_threshold(pairs, third, len(X))
    sigma_rows = row_positive_threshold(pairs, sigma, len(X))
    third_threshold = np.sqrt(third_rows[i] * third_rows[j])
    sigma_threshold = np.sqrt(sigma_rows[i] * sigma_rows[j])
    one = forward & (distance <= scales[j]) | backward & (distance <= scales[i])
    both = forward & backward & (distance <= scales[i]) & (distance <= scales[j])
    geometry = distance <= np.sqrt(scales[i] * scales[j])
    support = (third > EPS) & (third >= third_threshold)
    relative = sigma >= sigma_threshold
    eligible = rejected_cross & (sigma > EPS)
    r3 = eligible & one & support
    groups = {}
    for k in np.flatnonzero(r3):
        (a, b) = map(int, pairs[k])
        (ca, cb) = (int(ids[a]), int(ids[b]))
        if ca > cb:
            (ca, cb, a, b) = (cb, ca, b, a)
        groups.setdefault((ca, cb), []).append((int(k), a, b))
    contact_left = np.zeros(len(pairs), np.int64)
    contact_right = np.zeros(len(pairs), np.int64)
    contact_edges = np.zeros(len(pairs), np.int64)
    for entries in groups.values():
        (left, right) = (len({a for (_, a, _) in entries}), len({b for (_, _, b) in entries}))
        positions = [k for (k, _, _) in entries]
        (contact_left[positions], contact_right[positions]) = (left, right)
        contact_edges[positions] = len(entries)
    quorum = (contact_left >= CONTACT_ENDPOINTS) & (contact_right >= CONTACT_ENDPOINTS)
    decisions = {
        "BASE_G": np.zeros(len(pairs), bool),
        "R1": eligible & one,
        "R2": eligible & both,
        "R3": r3,
        "R4": eligible & geometry & support,
        "R5": r3 & quorum,
        "R6": eligible & one & relative,
    }
    from .sni_single_native._component_receiver import component_evidence

    extra = component_evidence(pairs, ids, distance, forward, backward, eligible, support)
    union = decisions["R2"] | decisions["R5"]
    extra["receiver_union_admitted"] = union
    decisions.update(
        U=union,
        U_S=decisions["R2"] & support | decisions["R5"],
        C50=union | extra["receiver_component_extra50"],
        C75=union | extra["receiver_component_extra75"],
        CR75=union | extra["receiver_component_extra_reciprocal75"],
    )
    from .sni_single_native._nn_envelope import envelope_evidence

    (envelope, envelope_decisions) = envelope_evidence(
        pairs,
        ids,
        distance,
        forward,
        backward,
        eligible,
        support,
        relative,
        extra["receiver_component_reciprocal"],
        len(X),
    )
    extra.update(envelope)
    decisions.update({tag: union | mask for (tag, mask) in envelope_decisions.items()})
    accepted = decisions[mode]
    for a, b in pairs[accepted]:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    arrays = dict(
        receiver_original_component_ids=ids,
        receiver_nn_scale=scales,
        receiver_distance=distance,
        receiver_forward=forward,
        receiver_backward=backward,
        receiver_rejected_cross=rejected_cross,
        receiver_eligible=eligible,
        receiver_one_direction=one,
        receiver_two_direction=both,
        receiver_geometry_pass=geometry,
        receiver_third_support=third,
        receiver_third_row=third_rows,
        receiver_third_threshold=third_threshold,
        receiver_third_pass=support,
        receiver_sigma_row=sigma_rows,
        receiver_sigma_threshold=sigma_threshold,
        receiver_sigma_pass=relative,
        receiver_contact_left=contact_left,
        receiver_contact_right=contact_right,
        receiver_contact_edges=contact_edges,
        receiver_contact_quorum=quorum,
        receiver_admitted=accepted,
        combined_pre_bridge_keep=original_keep | accepted,
    )
    arrays.update(extra)
    arrays.update(
        receiver_mode=mode,
        receiver_R2=decisions["R2"].copy(),
        receiver_R5=decisions["R5"].copy(),
        receiver_U=union.copy(),
        receiver_U_S=(decisions["R2"] & support | decisions["R5"]).copy(),
        receiver_R2_support_rejected=decisions["R2"] & ~support,
        receiver_R2_support_incremental_rejected=decisions["R2"] & ~support & ~decisions["R5"],
        receiver_R2_R5_overlap=decisions["R2"] & decisions["R5"],
    )
    return (graph, arrays)
