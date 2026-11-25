from contextlib import contextmanager
from .field_gate175 import fingerprint, qualify as qualify175, components

MASK_KEYS = {
    "CURRENT_REPLAY": "proposed_U",
    "U_KNOWN_ONLY": "accepted_known",
    "EN_KNOWN_ONLY": "accepted_known_en",
    "EN_NEFF2_KNOWN_ONLY": "accepted_known_neff",
    "EN_POSITIVE_VOLUME": "accepted_positive_en",
}


def qualify(pairs, gij, gji, block_ids, raw_ids, proposal, eps):
    import numpy as np

    a = qualify175(pairs, gij, gji, block_ids, raw_ids, proposal, eps)
    a["schema"] = "176_FIXED_BLOCK_RECEIVER_ABLATION_V1"
    a["domain_zero_internal_support"] = a.pop("domain_unknown_fallback")
    a["unknown_proposal_edge"] = a.pop("unknown_edge")
    a["diagnostic175_accepted_enrich"] = a.pop("accepted_enrich")
    a["diagnostic175_accepted_neff"] = a.pop("accepted_neff")
    d = a["domain_block_positions"]
    r = a["domain_raw_position"]
    positive = (a["block_volume"][d] > 0).all(axis=1) & (a["raw_total_force"][r] > 0)
    a["domain_positive_volume"] = positive
    a["domain_positive_en"] = a["domain_has_proposal"] & positive & a["domain_enriched"]
    cross = a["edge_domain"] >= 0
    same = a["same_block_U"]

    def edges(domain):
        mask = np.zeros(len(pairs), bool)
        mask[cross] = proposal[cross] & domain[a["edge_domain"][cross]]
        return mask

    a["known_proposal_edge"] = edges(a["domain_known"])
    a["positive_volume_en_edge"] = edges(a["domain_positive_en"])
    a["zero_internal_positive_en_edge"] = a["positive_volume_en_edge"] & a["unknown_proposal_edge"]
    a["accepted_known"] = same | a["known_proposal_edge"]
    a["accepted_known_en"] = same | a["certified_enrich_edge"]
    a["accepted_known_neff"] = same | a["certified_neff_edge"]
    a["accepted_positive_en"] = same | a["positive_volume_en_edge"]
    for left, right in [
        ("accepted_known_neff", "accepted_known_en"),
        ("accepted_known_en", "accepted_known"),
        ("accepted_known", "proposed_U"),
        ("accepted_known_en", "accepted_positive_en"),
        ("accepted_positive_en", "proposed_U"),
    ]:
        assert not (a[left] & ~a[right]).any(), (left, right)
    for key in ("accepted_known", "accepted_known_en", "accepted_known_neff"):
        assert not (a[key] & a["unknown_proposal_edge"]).any()
    assert np.array_equal(
        a["accepted_positive_en"] & a["unknown_proposal_edge"], a["zero_internal_positive_en_edge"]
    )
    return a


def summarize_paths(a, seeds):
    import numpy as np

    (p, F, A, P) = (
        a["pairs"],
        a["field_only_keep"],
        a["actual_admitted"],
        a["actual_post_bridge_keep"],
    )
    anchor_positions = {tuple(map(int, e)) for e in a["anchor_strong_edges"]}
    anchor_mask = np.asarray([tuple(map(int, e)) in anchor_positions for e in p], bool)
    base = (F | a["same_block_U"]) & P
    masks = dict(
        anchor=anchor_mask,
        F_predicate=F,
        original_U_before_bridge=F | a["proposed_U"],
        original_U_post_bridge=a["original_U_post_bridge_keep"],
        actual_before_bridge=F | A,
        actual_post_bridge=P,
        actual_F_only=F & P,
        actual_base=base,
        actual_base_plus_internal_support=base | a["actual_internal_support_edge"] & P,
        actual_base_plus_zero_internal_EN=base | a["actual_zero_internal_positive_en_edge"] & P,
        actual_base_plus_unconditional_unknown=base | a["actual_unconditional_unknown_edge"] & P,
    )
    roots = {k: components(len(seeds), p, v) for (k, v) in masks.items()}
    summary = {}
    tables = {}
    selected = seeds >= 0
    for key, root in roots.items():
        table = {}
        for component, label in np.unique(np.c_[root[selected], seeds[selected]], axis=0):
            table.setdefault(int(component), []).append(int(label))
        tables[key] = table
        multi = {k: v for (k, v) in table.items() if len(v) > 1}
        summary[key] = dict(
            edges=int(masks[key].sum()),
            components=int(len(np.unique(root))),
            components_joining_multiple_anchor_seeds=len(multi),
            examples=[
                dict(component=k, anchor_seed_labels=v) for (k, v) in list(multi.items())[:20]
            ],
        )
    categories = {
        k: 0
        for k in (
            "F_only_witness",
            "base_only_witness",
            "internal_support_only_witness",
            "zero_internal_EN_only_witness",
            "unconditional_unknown_only_witness",
            "multiple_separate_support_witnesses",
            "mixed_or_other_path",
        )
    }
    for root, owners in tables["actual_post_bridge"].items():
        if len(owners) <= 1:
            continue
        nodes = roots["actual_post_bridge"] == root

        def witness(key):
            return any((len(tables[key].get(int(k), [])) > 1 for k in np.unique(roots[key][nodes])))

        if witness("actual_F_only"):
            category = "F_only_witness"
        elif witness("actual_base"):
            category = "base_only_witness"
        else:
            found = [
                label
                for (label, key) in [
                    ("internal_support", "actual_base_plus_internal_support"),
                    ("zero_internal_EN", "actual_base_plus_zero_internal_EN"),
                    ("unconditional_unknown", "actual_base_plus_unconditional_unknown"),
                ]
                if witness(key)
            ]
            category = (
                found[0] + "_only_witness"
                if len(found) == 1
                else "multiple_separate_support_witnesses"
                if len(found) > 1
                else "mixed_or_other_path"
            )
        categories[category] += 1
    return dict(
        graphs=summary,
        actual_multi_anchor_component_path_witnesses=categories,
        interpretation="Feature-anchor path sufficiency witnesses; F then base take precedence; only means one support type suffices, not exclusive causal attribution or reference-class purity",
    )


@contextmanager
def observe_gate(guard, method, audits, eps):
    import numpy as np

    assert method in MASK_KEYS
    original = guard.prepare_reference

    def wrapped(
        X,
        base,
        geometry_q,
        H,
        active,
        core,
        bridge_mode,
        quota_mode,
        tie_geometry,
        receiver_mode="U",
    ):
        assert receiver_mode == "U"
        ref = original(
            X,
            base,
            geometry_q,
            H,
            active,
            core,
            bridge_mode,
            quota_mode,
            tie_geometry,
            receiver_mode,
        )
        old = ref["receiver_study"]
        rx = old["receiver"]
        anchor = ref["anchor"]
        a = qualify(
            ref["ev"]["pairs"],
            ref["field"]["G_ij"],
            ref["field"]["G_ji"],
            anchor["arrays"]["component_ids"],
            base["component_ids_raw"],
            rx["receiver_U"],
            eps,
        )
        a.update(
            active=active.copy(),
            field_only_keep=old["field_only_keep"].copy(),
            anchor_strong_edges=anchor["arrays"]["strong_edges"].copy(),
            anchor_seed_labels=anchor["arrays"]["selected_seed_labels"].copy(),
            proposed_U_S=rx["receiver_U_S"].copy(),
            original_U_post_bridge_keep=old["post_bridge_keep"].copy(),
            original_bridge_candidate=old["bridge_candidate"].copy(),
            original_bridge_protected=old["bridge_protected"].copy(),
            original_bridge_removed=old["bridge_removed"].copy(),
            method=method,
        )
        a["entry_fingerprint"] = fingerprint(
            dict(
                X=X,
                active=active,
                pairs=a["pairs"],
                field_F=a["field_only_keep"],
                G_ij=a["G_ij"],
                G_ji=a["G_ji"],
                raw_ids=a["raw_ids"],
                anchor_arrays=anchor["arrays"],
                anchor_candidate=anchor["candidate"],
                anchor_protected=anchor["protected"],
                anchor_removed=anchor["removed"],
                anchor_caps=anchor["summary"]["caps"],
                U=a["proposed_U"],
                U_S=a["proposed_U_S"],
            )
        )
        admitted = a[MASK_KEYS[method]]
        if method == "CURRENT_REPLAY":
            policy = "U"
            result = ref
        else:
            policy = "U__" + method
            if np.array_equal(admitted, a["proposed_U"]):
                candidate = old["bridge_candidate"].copy()
                protected = old["bridge_protected"].copy()
                removed = old["bridge_removed"].copy()
                post = old["post_bridge_keep"].copy()
            else:
                graph = [set() for _ in range(len(X))]
                for mask in (a["field_only_keep"], admitted):
                    for i, j in a["pairs"][mask]:
                        graph[int(i)].add(int(j))
                        graph[int(j)].add(int(i))
                chosen = guard.deleted_bridges(base["M"], graph, H, active, "SB_DUAL")
                candidate = np.asarray([tuple(map(int, e)) in chosen for e in a["pairs"]], bool)
                (protected, _) = guard.protected_bridges(
                    base, H, active, a["pairs"], candidate, bridge_mode, core
                )
                removed = candidate & ~protected
                post = (a["field_only_keep"] | admitted) & ~removed
            new_rx = dict(
                rx,
                receiver_mode=policy,
                receiver_admitted=admitted.copy(),
                combined_pre_bridge_keep=a["field_only_keep"] | admitted,
            )
            study = dict(
                old,
                receiver=new_rx,
                bridge_candidate=candidate,
                bridge_protected=protected,
                bridge_removed=removed,
                post_bridge_keep=post,
                receiver_policy=policy,
            )
            result = dict(ref, original_U_keep=post, receiver_study=study)
        actual = result["receiver_study"]
        unknown = a["unknown_proposal_edge"]
        known = a["known_proposal_edge"]
        applied_en = method in ("EN_KNOWN_ONLY", "EN_NEFF2_KNOWN_ONLY", "EN_POSITIVE_VOLUME")
        a.update(
            actual_admitted=admitted.copy(),
            actual_internal_support_edge=admitted & known,
            actual_known_EN_certificate_edge=admitted & a["certified_enrich_edge"]
            if applied_en
            else np.zeros(len(admitted), bool),
            actual_known_NEFF2_certificate_edge=admitted & a["certified_neff_edge"]
            if method == "EN_NEFF2_KNOWN_ONLY"
            else np.zeros(len(admitted), bool),
            actual_zero_internal_positive_en_edge=admitted & a["zero_internal_positive_en_edge"]
            if method == "EN_POSITIVE_VOLUME"
            else np.zeros(len(admitted), bool),
            actual_unconditional_unknown_edge=admitted & unknown
            if method == "CURRENT_REPLAY"
            else np.zeros(len(admitted), bool),
            actual_sameblock_edge=admitted & a["same_block_U"],
            actual_bridge_candidate=actual["bridge_candidate"].copy(),
            actual_bridge_protected=actual["bridge_protected"].copy(),
            actual_bridge_removed=actual["bridge_removed"].copy(),
            actual_post_bridge_keep=actual["post_bridge_keep"].copy(),
            actual_policy=policy,
        )
        assert np.array_equal(
            admitted,
            a["actual_internal_support_edge"]
            | a["actual_zero_internal_positive_en_edge"]
            | a["actual_unconditional_unknown_edge"]
            | a["actual_sameblock_edge"],
        )
        assert np.array_equal(
            a["actual_bridge_removed"], a["actual_bridge_candidate"] & ~a["actual_bridge_protected"]
        )
        assert np.array_equal(
            a["actual_post_bridge_keep"],
            (a["field_only_keep"] | admitted) & ~a["actual_bridge_removed"],
        )
        a["path_summary"] = summarize_paths(a, a["anchor_seed_labels"])
        audits.append(a)
        return result

    guard.prepare_reference = wrapped
    try:
        yield
    finally:
        assert guard.prepare_reference is wrapped
        guard.prepare_reference = original
