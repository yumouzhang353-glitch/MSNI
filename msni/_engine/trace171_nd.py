_interface_masks = None
_interface_link = None
import numpy as np
import time
from .sni_single_native._config import PARAMETERS
from .sni_single_native import _core, _screening, _field, _pipeline
from . import geometry_adapter167 as _geometry
from . import geometry_support167 as _support
from .sni_single_native._policy import quota_pre_state, intervention_policy
from .sni_single_native._bridge import deleted_bridges
from .sni_single_native._paths import protected_bridges
from .sni_single_native._quota_suffix import suffix_local
from .receiver171 import admit
from .sni_single_native._budget_adapter import prepare_anchor


def run_single(X, boundary_mode="C0", original_receiver="U", native_receiver="U"):
    X = np.asarray(X, dtype=np.float64, order="C")
    assert (
        X.ndim == 2 and 2 <= len(X) <= 10000 and (1 <= X.shape[1] <= 552) and np.isfinite(X).all()
    )
    (version, bridge_mode, quota_mode) = ("QR3_GRAPH", "P4_NB", "PG8")
    (tie_geometry, tie_pre, tie_active) = (True, False, False)
    (receiver_mode, budget_mode) = (native_receiver, "F")
    (rank_repair_mode, budget_anchor_scope) = ("I2", "ORIGINAL")
    (field_guard_mode, contact_repair_mode) = ("NONE", "C4Q")
    from .sni_single_native._scoped_core import ScopedCore

    core = ScopedCore(_core, tie_active)
    pre_core = ScopedCore(_core, tie_pre)
    protocol = PARAMETERS
    config = dict(stable=True, local=False)
    pre = quota_pre_state(X, pre_core, version)
    parent_mask = _screening.screen(X, "N4_B", core, pre)
    (candidates, extra) = _support.continuation(
        X, pre, parent_mask, core, protocol["continuity_parameters"], tie_geometry=tie_geometry
    )
    (protected, _, extra) = _support.gated_support(
        X,
        pre,
        parent_mask,
        candidates,
        extra,
        core,
        protocol,
        stable=False,
        tie_geometry=tie_geometry,
    )
    search_gate = extra["anchor_stable_gate"] if config["stable"] else extra["anchor_median_gate"]
    search_allowed = search_gate[extra["fragment_component_ids"]]
    extra.update(
        anchor_selected_search_gate=search_gate,
        support_search_allowed=search_allowed,
        support_search_stable=np.asarray(config["stable"]),
    )
    restoration = bool(protected.any())
    local_scope = bool(config["local"] and (not restoration))
    extra.update(
        support_restoration_trigger=np.asarray(restoration),
        support_local_domain=np.asarray(local_scope),
    )
    mask = parent_mask & ~protected
    active = np.flatnonzero(~mask)
    current = X[active]
    anchor_base = _pipeline.build_base(X[~parent_mask], core)
    natural = (
        anchor_base if np.array_equal(mask, parent_mask) else _pipeline.build_base(current, core)
    )
    (natural_r, anchor_r) = (int(natural["r"]), int(anchor_base["r"]))
    baseline = min(len(current) - 1, natural_r)
    target = min(len(current) - 1, max(baseline, int(pre["r"])))
    if restoration:
        expansion = np.full(len(active), target > baseline, dtype=bool)
    else:
        expansion = np.full(
            len(active), search_allowed[active].any() and target > baseline, dtype=bool
        )
    ranks = np.where(expansion, target, baseline).astype(int)
    legacy_ranks = ranks.copy()
    (ranks, geometry_q, policy_arrays) = intervention_policy(pre, ranks, version, active)
    expansion = expansion | (ranks > legacy_ranks)
    variable_builder = _pipeline.variable_base if tie_active else _pipeline._original_variable_base
    base = dict(natural) if np.all(ranks == natural_r) else variable_builder(current, ranks, core)
    rank_repair_arrays = {}
    prior_anchor = None
    guard_reference = None
    if field_guard_mode != "NONE" or contact_repair_mode != "NONE":
        from .geometry_contact_guard171 import prepare_reference

        guard_reference = prepare_reference(
            current,
            base,
            geometry_q,
            pre["H"],
            active,
            core,
            bridge_mode,
            quota_mode,
            tie_geometry,
            receiver_mode=original_receiver,
        )
    assert budget_anchor_scope in ("CURRENT", "ORIGINAL")
    if budget_anchor_scope == "ORIGINAL":
        assert rank_repair_mode in ("I2", "I3") and budget_mode == "F"
        from .sni_single_native._raw_rank import original_anchor

        prior_anchor = (
            guard_reference["anchor"]
            if guard_reference is not None
            else original_anchor(
                current,
                base,
                geometry_q,
                pre["H"],
                active,
                core,
                bridge_mode,
                quota_mode,
                tie_geometry,
            )
        )
    if rank_repair_mode in ("I2", "I3"):
        from .sni_single_native._raw_rank import expand

        (base, ranks, rank_repair_arrays) = expand(current, ranks, base, rank_repair_mode, core)
    elif rank_repair_mode != "NONE":
        from .sni_single_native._rank_repair import repair, rebuild

        (ranks, rank_repair_arrays) = repair(current, ranks, base, rank_repair_mode, core)
        if np.any(ranks != rank_repair_arrays["rank_repair_source_ranks"]):
            base = rebuild(current, ranks, base, core)
    ev = _field.evidence(current, base)
    f0 = _field.interaction(current, base, ev, "F0", core)
    field = _field.interaction(current, base, ev, "C1", core)
    guard_uncorrected_field = field
    shape = _geometry.prepare(
        current, int(geometry_q), base["reliability"], weighted=True, tie_closed=tie_geometry
    )
    field = _geometry.correct(current, base, ev, field, shape, "LW_GM", core)
    context = _field.threshold_context(ev, f0, field, len(active), "T8")
    (strong, branches) = _field.build_strong_graph(ev, field, len(active), context)
    field_predicate_keep = branches["keep"].copy()
    budget_anchor = prepare_anchor(
        current, base, strong, ev["pairs"], pre["H"], active, core, bridge_mode, quota_mode
    )
    if prior_anchor is not None:
        budget_anchor = prior_anchor
    (strong, receiver_arrays) = admit(
        current, base, ev, strong, field_predicate_keep, receiver_mode, core
    )
    branches["keep"] = receiver_arrays["combined_pre_bridge_keep"].copy()
    before_bridge = [set(neighbors) for neighbors in strong]
    original_keep = branches["keep"].copy()
    candidates_to_delete = deleted_bridges(base["M"], strong, pre["H"], active, "SB_DUAL")
    candidate = np.asarray([tuple(map(int, e)) in candidates_to_delete for e in ev["pairs"]], bool)
    (protection, path_summary) = protected_bridges(
        base, pre["H"], active, ev["pairs"], candidate, bridge_mode, core
    )
    removed = candidate & ~protection
    deleted = {tuple(map(int, ev["pairs"][k])) for k in np.flatnonzero(removed)}
    pre["path_decisions"] = dict(candidate=candidate, protected=protection)
    pre["path_summary"] = path_summary
    for a, b in deleted:
        strong[a].remove(b)
        strong[b].remove(a)
    branches["keep"] = original_keep & ~removed
    guard_arrays = {}
    if field_guard_mode != "NONE":
        from .geometry_contact_guard167 import apply

        (guarded_keep, guard_arrays) = apply(
            current,
            base,
            ev,
            f0,
            guard_uncorrected_field,
            field,
            shape,
            guard_reference,
            branches["keep"],
            field_guard_mode,
            core,
        )
        for a, b in ev["pairs"][branches["keep"] & ~guarded_keep]:
            strong[int(a)].remove(int(b))
            strong[int(b)].remove(int(a))
        branches["keep"] = guarded_keep
    from .geometry_contact_repair167 import apply as repair_contacts

    def graph_from(keep):
        graph = [set() for _ in current]
        for a, b in ev["pairs"][keep]:
            graph[int(a)].add(int(b))
            graph[int(b)].add(int(a))
        return graph

    (reference_keep, reference_before, reference_repair) = repair_contacts(
        current,
        ev,
        guard_reference,
        branches["keep"],
        receiver_arrays["combined_pre_bridge_keep"],
        "C4Q",
        core,
    )
    (target_keep, target_before, target_repair) = repair_contacts(
        current,
        ev,
        guard_reference,
        branches["keep"],
        receiver_arrays["combined_pre_bridge_keep"],
        "T_L",
        core,
    )
    assert np.all(~reference_keep | target_keep)
    from .core_scope96 import partition

    boundary_started = time.perf_counter()
    (reference_final, target_final, boundary_record) = partition(
        boundary_mode, base, ev, field, reference_keep, target_keep, core, ranks.copy()
    )
    (reference_final, target_final) = _interface_masks(
        X, active, ev, reference_final, target_final, boundary_record
    )
    (strong_reference, strong_target) = (graph_from(reference_final), graph_from(target_final))
    _interface_link(strong_target)
    suffix_started = time.perf_counter()
    (reference_clean, reference_suffix, reference_summary) = suffix_local(
        current,
        base,
        graph_from(reference_before),
        strong_reference,
        core,
        quota_mode,
        budget_mode=budget_mode,
        anchor=budget_anchor,
    )
    boundary_record.update(
        reference_before_keep=reference_before.copy(),
        suffix_seconds=time.perf_counter() - suffix_started,
        boundary_and_budget_seconds=time.perf_counter() - boundary_started,
        budget_suffix_calls=1,
        caps=dict(reference_summary["caps"]),
    )
    (ids, groups) = core.weak_connected_components(strong_target)
    reference_seeds = reference_suffix["selected_seed_labels"].copy()
    total = int(reference_seeds.max()) + 1
    quotient = np.full(total, -1, np.int64)
    supported = []
    for group in groups:
        points = np.asarray(list(group), np.int64)
        labels = np.unique(reference_seeds[points])
        labels = labels[labels >= 0]
        if len(labels):
            root = int(labels.min())
            assert np.all(quotient[labels] < 0)
            quotient[labels] = root
            supported.append((root, points))
    assert np.all(quotient >= 0)
    seeds = np.full(len(active), -1, np.int64)
    for label, (_, points) in enumerate(sorted(supported, key=lambda item: item[0])):
        seeds[points] = label
    raw_ids = base["component_ids_raw"]
    parents = reference_summary["parents"]
    clean = core.dpc_assign_remaining_clean_points_v94(
        current, seeds, base["rho"], ids, raw_ids, parents
    )
    assert np.array_equal(clean[seeds >= 0], seeds[seeds >= 0]) and np.all(clean >= 0)
    (labels, witness) = _pipeline.assign(X, active, clean, core)
    output = dict(
        labels=labels,
        active=active,
        screening_mask=mask,
        seeds=seeds,
        selected_mask=seeds >= 0,
        active_labels=clean,
        component_ids=ids,
        raw_ids=raw_ids,
        rho=base["rho"],
        support_topology_edges=_pipeline.edges(strong_target),
        strong_edges=ev["pairs"][target_final].copy(),
        reference_seeds=reference_seeds,
        seed_quotient=quotient,
        reference_caps=np.asarray(
            [reference_summary["caps"][raw] for raw in range(len(base["components_raw"]))]
        ),
        used_caps=np.asarray(
            [
                len(np.unique(seeds[(raw_ids == raw) & (seeds >= 0)]))
                for raw in range(len(base["components_raw"]))
            ]
        ),
        raw_parent=np.asarray([parents[raw] for raw in range(len(base["components_raw"]))]),
        restoration_witness=witness,
    )
    shared = _pipeline.base_arrays(base, active)
    shared.update(
        support_requested_rank=ranks,
        support_expanded_nodes=expansion,
        support_natural_r=np.asarray(natural_r),
        support_anchor_r=np.asarray(anchor_r),
        support_baseline_rank=np.asarray(baseline),
        support_target_rank=np.asarray(target),
        geometry_ranks=np.asarray(shape["ranks"]),
        geometry_neighbors=shape["neighbors"],
        geometry_dimensions=shape["dimensions"],
        geometry_inverse=shape["inverse"],
        geometry_scales=shape["scales"],
        edge_direction_factor=field["edge_direction_factor"],
        graph_used_rank=np.asarray(base["r"]),
    )
    shared.update(policy_arrays)
    shared.update(rank_repair_arrays)
    shared.update(
        geometry_neighbor_valid=shape["neighbor_valid"],
        geometry_actual_counts=shape["actual_counts"],
        geometry_neighbors_per_rank=shape["neighbors_per_rank"],
    )
    shared.update(pre["quota_arrays"])
    shared.update(geometry_compact_sidecar=_geometry.geometry_sidecar(shape))
    pre_arrays = _pipeline.pre_arrays(pre)
    reference_audit = dict(reference_suffix, **reference_repair)
    reference_audit["restoration_witness"] = witness
    return (
        output,
        dict(
            core_boundary=boundary_record,
            receiver_study=dict(
                original=guard_reference["receiver_study"],
                native=dict(
                    pairs=ev["pairs"].copy(),
                    field_only_keep=field_predicate_keep.copy(),
                    receiver=receiver_arrays,
                    bridge_candidate=candidate.copy(),
                    bridge_protected=protection.copy(),
                    bridge_removed=removed.copy(),
                    post_bridge_keep=branches["keep"].copy(),
                    receiver_policy=receiver_mode,
                    branch="expanded_native",
                ),
            ),
            support=extra,
            shared=shared,
            reference=reference_audit,
            target=target_repair,
            pre=pre_arrays,
            screening_and_support_passes=1,
            expanded_field_passes=1,
            geometry_passes_after_expansion=1,
            reference_suffix_passes=1,
            target_budget_passes=0,
            final_assignment_passes=1,
            natural_r=int(pre["r"]),
            active_natural_r=natural_r,
            labels_received=False,
            coordinates_moved=False,
            historical_prefix_input=False,
        ),
    )
