def state_for(X, base, components, reference, core):
    (caps, info) = core.compute_raw_component_center_caps_v94(
        data=X,
        components_raw=base["components_raw"],
        components_cut=components,
        component_ids_raw=base["component_ids_raw"],
        ref_size=reference,
        local_scale=base["local_scale"],
    )
    return dict(
        caps=caps,
        parents=info["raw_parent"],
        reference_size=float(reference),
        significant_cut_min=float(info["significant_cut_min"]),
    )


def resolve_parents(caps, parents, raw_sizes):
    result = dict(parents)
    changes = []
    for raw, cap in caps.items():
        if cap > 0:
            assert parents[raw] == raw
            continue
        seen = {raw}
        parent = parents[raw]
        previous = raw
        chain = [raw]
        while True:
            assert parent in caps and parent not in seen, (raw, chain, parent)
            assert raw_sizes[parent] > raw_sizes[previous], (raw, previous, parent)
            chain.append(parent)
            if caps[parent] > 0:
                break
            seen.add(parent)
            (previous, parent) = (parent, parents[parent])
        result[raw] = parent
        if result[raw] != parents[raw]:
            changes.append(
                dict(raw_id=raw, proposed_parent=parents[raw], final_parent=parent, chain=chain)
            )
    assert all((caps[result[raw]] > 0 for raw in caps))
    return (result, changes)


def local_states(X, base, old_components, new_components, touched, core):
    raw_ids = base["component_ids_raw"]
    old_ref = core.robust_reference_component_size(old_components)
    new_ref = core.robust_reference_component_size(new_components)
    old = state_for(X, base, old_components, old_ref, core)
    new = state_for(X, base, new_components, new_ref, core)
    (old_parts, new_parts) = ({}, {})
    for components, target in [(old_components, old_parts), (new_components, new_parts)]:
        for component in components:
            owners = set(map(int, raw_ids[list(component)]))
            assert len(owners) == 1
            target.setdefault(next(iter(owners)), []).append(component)
    raw_sizes = list(map(len, base["components_raw"]))
    cache = {float(new_ref): new}
    policies = {}
    for policy in ["L1", "L2"]:
        (caps, proposed, refs, minima) = ({}, {}, {}, {})
        for raw in range(len(raw_sizes)):
            if not touched[raw]:
                source = old
            elif policy == "L1":
                source = new
            else:
                hypothetical = [
                    component
                    for (owner, components) in old_parts.items()
                    if owner != raw
                    for component in components
                ]
                hypothetical += new_parts[raw]
                local_ref = float(core.robust_reference_component_size(hypothetical))
                assert sum(map(len, hypothetical)) == len(X)
                if local_ref not in cache:
                    cache[local_ref] = state_for(X, base, new_components, local_ref, core)
                source = cache[local_ref]
            caps[raw] = source["caps"][raw]
            proposed[raw] = source["parents"][raw]
            refs[raw] = source["reference_size"]
            minima[raw] = source["significant_cut_min"]
        (parents, closure) = resolve_parents(caps, proposed, raw_sizes)
        assert all((caps[raw] == old["caps"][raw] for raw in caps if not touched[raw]))
        policies[policy] = dict(
            caps=caps,
            parents=parents,
            reference_size=refs,
            significant_cut_min=minima,
            reference_scope="per_raw_component",
            proposed_parents=proposed,
            parent_closure=closure,
            touched_raw=touched.astype(bool),
            old_reference_size=float(old_ref),
            new_global_reference_size=float(new_ref),
        )
    if int(touched.sum()) <= 1:
        assert policies["L1"]["caps"] == policies["L2"]["caps"]
        assert policies["L1"]["proposed_parents"] == policies["L2"]["proposed_parents"]
        assert policies["L1"]["parents"] == policies["L2"]["parents"]
    return (policies, old, new)
