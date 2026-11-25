from collections import defaultdict
import numpy as np

CONFIGURATIONS = {
    "native": (),
    "bridge": ("bridge_removal", "bridge_protection"),
    "topology": ("reference", "target", "bridge_removal", "bridge_protection"),
    "geometry": ("geometry",),
    "transfer": (
        "geometry",
        "reference",
        "target",
        "bridge_removal",
        "bridge_protection",
        "initial_support",
        "final_support",
        "protected_continuation",
        "reliability",
        "propagation",
        "restoration",
    ),
}


def edge_ids(pairs, n):
    canonical = np.sort(pairs, axis=1)
    if (
        (canonical[:, 0] == canonical[:, 1]).any()
        or (canonical < 0).any()
        or (canonical >= n).any()
    ):
        raise ValueError("invalid undirected edge endpoints")
    return canonical[:, 0] * n + canonical[:, 1]


def align_edges(global_pairs, local_pairs, n):
    keys = edge_ids(local_pairs, n)
    order = np.argsort(keys)
    keys = keys[order]
    if len(keys) == 0:
        return (np.zeros(len(global_pairs), dtype=bool), np.empty(0, dtype=int))
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError("duplicate undirected Core edge")
    target = edge_ids(global_pairs, n)
    positions = np.searchsorted(keys, target)
    possible = np.minimum(positions, len(keys) - 1)
    matched = (positions < len(keys)) & (keys[possible] == target)
    return (matched, order[possible[matched]])


def evidence(pairs, core, n):
    (matched, indices) = align_edges(pairs, core["pairs"], n)
    (left, right) = pairs.T
    terms = defaultdict(lambda: np.zeros(len(pairs), dtype=float))

    def on_matched(name, values):
        terms[name][matched] = values[indices]

    target = core["target"]
    reference = core["reference"]
    removed = core["bridge_removed"] | core["first_bridge_removed"]
    on_matched("target", 0.12 * target)
    on_matched("reference", 0.14 * reference)
    on_matched("bridge_removal", -np.log(2.0) * removed)
    on_matched("bridge_protection", 0.12 * (core["bridge_protected"] & target & ~removed))
    ratio = np.zeros(len(target), dtype=float)
    valid = (core["base_force"] > 0) & (core["corrected_force"] > 0)
    if valid.any():
        ratio[valid] = np.log(core["corrected_force"][valid]) - np.log(core["base_force"][valid])
        ratio[valid] -= np.median(ratio[valid])
    on_matched("geometry", 0.25 * np.clip(ratio, -np.log(2), np.log(2)))
    active = np.zeros(n, dtype=bool)
    active[core["active"]] = True
    retained_reliability = np.zeros(n)
    retained_reliability[core["active"]] = np.clip(core["reliability"], 0, 1)
    a8 = np.zeros(n)
    a8[core["active"]] = np.clip(core["propagation"], 0, 1)
    r6 = np.clip(core["restoration"], 0, 1)
    vetted = terms["target"] > 0
    both_in_reference = terms["reference"] > 0
    for name, source, coefficient in (
        ("initial_support", "selected", 0.07),
        ("final_support", "final_supports", 0.07),
    ):
        point_mask = np.zeros(n, dtype=bool)
        point_mask[core[source]] = True
        terms[name] = (
            coefficient * both_in_reference * (0.5 * point_mask[left] + 0.5 * point_mask[right])
        )
    protected = core["protected"]
    terms["protected_continuation"] = 0.1 * vetted * (protected[left] | protected[right])
    terms["reliability"] = (
        0.08 * vetted * np.sqrt(retained_reliability[left] * retained_reliability[right])
    )
    terms["propagation"] = 0.08 * vetted * np.sqrt(a8[left] * a8[right])
    terms["restoration"] = (
        0.08
        * (active[left] ^ active[right])
        * (r6[left] * retained_reliability[right] + r6[right] * retained_reliability[left])
    )
    return (dict(terms), int(matched.sum()))
