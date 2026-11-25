import numpy as np
from . import _pipeline
from ._local_budget import local_states


def suffix_local(X, base, before, strong, core, quota_mode, budget_mode="D", anchor=None):
    (old_ids, old_components) = core.weak_connected_components(before)
    (ids, components) = core.weak_connected_components(strong)
    touched = np.zeros(len(base["components_raw"]), bool)
    for a, b in _pipeline.edges(before):
        if int(b) not in strong[int(a)]:
            raw = int(base["component_ids_raw"][int(a)])
            assert raw == int(base["component_ids_raw"][int(b)])
            touched[raw] = True
    (policies, old_state, _) = local_states(X, base, old_components, components, touched, core)
    original_state = policies["L2"]
    assert quota_mode in QUOTA_PARAMETERS
    (caps, eligible, floors) = constrained_caps(
        base,
        old_ids,
        old_components,
        components,
        original_state,
        old_state,
        QUOTA_PARAMETERS[quota_mode],
    )
    budget_arrays = {}
    if anchor is not None:
        from ._budget_adapter import adapt

        (original_state, old_state, caps, eligible, floors, budget_arrays) = adapt(
            X,
            base,
            old_ids,
            old_components,
            ids,
            components,
            original_state,
            old_state,
            caps,
            anchor,
            budget_mode,
            core,
            quota_mode,
        )
    else:
        assert budget_mode == "D"
    state = dict(original_state, caps=caps)
    selected = core.select_components_by_budget(
        components, base["component_ids_raw"], base["components_raw"], state["caps"]
    )
    seeds = np.full(len(X), -1, np.int64)
    selected_mask = np.zeros(len(X), bool)
    for label, record in enumerate(selected):
        points = np.asarray(list(components[record["cut_cid"]]), np.int64)
        seeds[points] = label
        selected_mask[points] = True
    clean = core.dpc_assign_remaining_clean_points_v94(
        data=X,
        labels=seeds,
        rho=base["rho"],
        component_ids_cut=ids,
        component_ids_raw=base["component_ids_raw"],
        raw_parent=state["parents"],
    )
    assert (clean >= 0).all()
    arrays = dict(
        component_ids=ids,
        active_labels=clean,
        selected_mask=selected_mask,
        selected_seed_labels=seeds,
        strong_edges=_pipeline.edges(strong),
    )
    summary = dict(
        state,
        selected=selected,
        cut_sizes=list(map(len, components)),
        raw_sizes=list(map(len, base["components_raw"])),
        graph_reference_size=float(core.robust_reference_component_size(components)),
        budget_total=int(sum(state["caps"].values())),
        full_coverage=True,
    )
    arrays.update(
        quota_support_eligible_cut=eligible,
        quota_support_minimum_cut_size=floors,
        quota_support_original_caps=np.asarray(list(original_state["caps"].values())),
        quota_support_G0_caps=np.asarray(list(old_state["caps"].values())),
        quota_support_caps=np.asarray(list(caps.values())),
    )
    summary.update(
        quota_policy=quota_mode,
        quota_fraction=QUOTA_PARAMETERS[quota_mode]["fraction"],
        quota_original_caps=original_state["caps"],
        quota_G0_caps=old_state["caps"],
    )
    arrays.update(budget_arrays)
    if anchor is not None:
        summary.update(
            budget_adaptation=budget_mode,
            anchor_source=anchor["source"],
            anchor_budget_total=int(anchor["summary"]["budget_total"]),
            tracked_seats=int(budget_arrays["budget_tracked_seats"].sum()),
            original_seed_groups_split=int(budget_arrays["budget_tracked_split_seed_mask"].sum()),
        )
    return (clean, arrays, summary)


QUOTA_PARAMETERS = {
    "PG16": {"reference": "G0_parent_cut", "fraction": 0.0625, "growth_only": True},
    "PG8": {"reference": "G0_parent_cut", "fraction": 0.125, "growth_only": True},
    "PG4": {"reference": "G0_parent_cut", "fraction": 0.25, "growth_only": True},
}


def constrained_caps(base, old_ids, old_components, components, state, old_state, policy):
    eligible = np.zeros(len(components), bool)
    floors = np.zeros(len(components), float)
    counts = {raw: 0 for raw in state["caps"]}
    largest = {
        raw: max((len(c) for c in components if base["component_ids_raw"][min(c)] == raw))
        for raw in counts
    }
    for cid, component in enumerate(components):
        raw = int(base["component_ids_raw"][min(component)])
        original_ids = np.unique(old_ids[list(component)])
        assert len(original_ids) == 1
        support_size = (
            len(old_components[int(original_ids[0])])
            if policy["reference"] == "G0_parent_cut"
            else largest[raw]
        )
        threshold = max(state["significant_cut_min"][raw], policy["fraction"] * support_size)
        floors[cid] = threshold
        eligible[cid] = len(component) >= threshold
        counts[raw] += int(eligible[cid])
    caps = dict(state["caps"])
    for raw, cap in caps.items():
        if cap <= 0 or not state["touched_raw"][raw]:
            continue
        quota = max(1, counts[raw], old_state["caps"][raw] if policy["growth_only"] else 0)
        caps[raw] = min(cap, quota)
    assert all((0 <= caps[raw] <= state["caps"][raw] for raw in caps))
    assert all((caps[raw] == state["caps"][raw] for raw in caps if not state["touched_raw"][raw]))
    assert all((caps[parent] > 0 for parent in state["parents"].values()))
    return (caps, eligible, floors)
