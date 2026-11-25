import numpy as np
from ._local_budget import state_for, resolve_parents

BUDGET_MODES = ("D", "A", "F", "E")


def prepare_anchor(X, base, graph, pairs, H, active, core, bridge_mode, quota_mode):
    from ._bridge import deleted_bridges
    from ._paths import protected_bridges
    from ._quota_suffix import suffix_local

    before = [set(neighbors) for neighbors in graph]
    strong = [set(neighbors) for neighbors in graph]
    candidates = deleted_bridges(base["M"], strong, H, active, "SB_DUAL")
    candidate = np.asarray([tuple(map(int, edge)) in candidates for edge in pairs], bool)
    (protection, path_summary) = protected_bridges(
        base, H, active, pairs, candidate, bridge_mode, core
    )
    removed = candidate & ~protection
    for a, b in pairs[removed]:
        strong[int(a)].remove(int(b))
        strong[int(b)].remove(int(a))
    (clean, arrays, summary) = suffix_local(X, base, before, strong, core, quota_mode)
    return dict(
        before=before,
        strong=strong,
        clean=clean,
        arrays=arrays,
        summary=summary,
        candidate=candidate,
        protected=protection,
        removed=removed,
        path_summary=path_summary,
        source="same_complete_X_call_original_field_graph",
        reference_labels_received=False,
    )


def anchored_states(X, base, before_components, components, dynamic_state, anchor, core):
    original = anchor["summary"]
    old_reference = float(original["old_reference_size"])
    before = state_for(X, base, before_components, old_reference, core)
    cache = {}
    (caps, proposed, refs, minima) = ({}, {}, {}, {})
    for raw in dynamic_state["caps"]:
        ref = float(original["reference_size"][raw])
        if ref not in cache:
            cache[ref] = state_for(X, base, components, ref, core)
        current = cache[ref]
        caps[raw] = current["caps"][raw]
        proposed[raw] = current["parents"][raw]
        refs[raw] = ref
        minima[raw] = current["significant_cut_min"]
    raw_sizes = list(map(len, base["components_raw"]))
    (parents, closure) = resolve_parents(caps, proposed, raw_sizes)
    state = dict(
        caps=caps,
        parents=parents,
        reference_size=refs,
        significant_cut_min=minima,
        reference_scope="per_raw_component",
        proposed_parents=proposed,
        parent_closure=closure,
        touched_raw=dynamic_state["touched_raw"].copy(),
        old_reference_size=old_reference,
        new_global_reference_size=float(core.robust_reference_component_size(components)),
    )
    return (state, before)


def seed_tracking(base, ids, components, anchor):
    seeds = anchor["arrays"]["selected_seed_labels"]
    total = int(anchor["summary"]["budget_total"])
    assert len(np.unique(seeds[seeds >= 0])) == total
    owners = np.full(total, -1, np.int64)
    for label in range(total):
        raw = np.unique(base["component_ids_raw"][seeds == label])
        assert len(raw) == 1
        owners[label] = raw[0]
    parents = np.arange(total)

    def root(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]
            k = int(parents[k])
        return k

    occupied = np.zeros(len(base["components_raw"]), np.int64)
    for component in components:
        labels = np.unique(seeds[list(component)])
        labels = labels[labels >= 0]
        if not len(labels):
            continue
        raw = int(base["component_ids_raw"][min(component)])
        assert np.all(owners[labels] == raw)
        occupied[raw] += 1
        for label in labels[1:]:
            (x, y) = (root(int(labels[0])), root(int(label)))
            parents[max(x, y)] = min(x, y)
    equivalents = np.asarray([root(k) for k in range(total)], np.int64)
    seats = np.zeros(len(base["components_raw"]), np.int64)
    for raw in range(len(seats)):
        seats[raw] = len(np.unique(equivalents[owners == raw]))
    split = np.asarray([len(np.unique(ids[seeds == label])) > 1 for label in range(total)], bool)
    assert np.all(seats <= occupied)
    return dict(
        budget_tracked_seed_owners=owners,
        budget_tracked_seed_equivalence=equivalents,
        budget_tracked_seats=seats,
        budget_tracked_occupied_cuts=occupied,
        budget_tracked_split_seed_mask=split,
    )


def adapt(
    X,
    base,
    old_ids,
    old_components,
    ids,
    components,
    dynamic_state,
    dynamic_old_state,
    dynamic_caps,
    anchor,
    mode,
    core,
    quota_mode,
):
    from ._quota_suffix import constrained_caps, QUOTA_PARAMETERS

    assert mode in BUDGET_MODES and anchor is not None
    original = anchor["summary"]
    tracking = seed_tracking(base, ids, components, anchor)
    (state, before_state) = (dynamic_state, dynamic_old_state)
    caps = dict(dynamic_caps)
    if mode == "A":
        (state, before_state) = anchored_states(
            X, base, old_components, components, dynamic_state, anchor, core
        )
        (caps, eligible, floors) = constrained_caps(
            base,
            old_ids,
            old_components,
            components,
            state,
            before_state,
            QUOTA_PARAMETERS[quota_mode],
        )
    else:
        (_, eligible, floors) = constrained_caps(
            base,
            old_ids,
            old_components,
            components,
            state,
            before_state,
            QUOTA_PARAMETERS[quota_mode],
        )
    if mode in ["F", "E"]:
        seats = tracking["budget_tracked_seats"]
        for raw, current_cap in caps.items():
            support_cap = int(seats[raw])
            if mode == "F":
                caps[raw] = max(current_cap, support_cap)
            elif support_cap:
                caps[raw] = support_cap
        proposed = dict(state["parents"])
        for raw, cap in caps.items():
            if cap > 0:
                proposed[raw] = raw
        (parents, closure) = resolve_parents(caps, proposed, list(map(len, base["components_raw"])))
        state = dict(state, parents=parents, proposed_parents=proposed, parent_closure=closure)
    assert all(
        (
            0 <= cap <= sum((base["component_ids_raw"][min(c)] == raw for c in components))
            for (raw, cap) in caps.items()
        )
    )
    assert all((caps[parent] > 0 for parent in state["parents"].values()))
    raws = sorted(caps)
    arrays = dict(
        budget_anchor_component_ids=anchor["arrays"]["component_ids"],
        budget_anchor_strong_edges=anchor["arrays"]["strong_edges"],
        budget_anchor_seed_labels=anchor["arrays"]["selected_seed_labels"],
        budget_anchor_active_labels=anchor["arrays"]["active_labels"],
        budget_anchor_candidate=anchor["candidate"],
        budget_anchor_protected=anchor["protected"],
        budget_anchor_removed=anchor["removed"],
        budget_anchor_reference_size=np.asarray([original["reference_size"][raw] for raw in raws]),
        budget_anchor_significant_min=np.asarray(
            [original["significant_cut_min"][raw] for raw in raws]
        ),
        budget_anchor_original_caps=np.asarray(
            [original["quota_original_caps"][raw] for raw in raws]
        ),
        budget_anchor_G0_caps=np.asarray([original["quota_G0_caps"][raw] for raw in raws]),
        budget_anchor_caps=np.asarray([original["caps"][raw] for raw in raws]),
        budget_anchor_old_reference=np.asarray(original["old_reference_size"]),
        budget_dynamic_caps=np.asarray([dynamic_caps[raw] for raw in raws]),
        budget_dynamic_reference_size=np.asarray(
            [dynamic_state["reference_size"][raw] for raw in raws]
        ),
        budget_dynamic_significant_min=np.asarray(
            [dynamic_state["significant_cut_min"][raw] for raw in raws]
        ),
        budget_current_reference_size=np.asarray([state["reference_size"][raw] for raw in raws]),
        budget_current_significant_min=np.asarray(
            [state["significant_cut_min"][raw] for raw in raws]
        ),
        budget_mode_code=np.asarray(BUDGET_MODES.index(mode)),
    )
    arrays.update(tracking)
    return (state, before_state, caps, eligible, floors, arrays)
