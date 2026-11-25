from contextlib import contextmanager
from .field_gate176 import observe_gate as observe_known
from .field_gate175 import fingerprint
from .unknown_regions177 import diagnose, components

BASE_MAP = {
    "EN_KNOWN_ONLY_REPLAY": "EN_KNOWN_ONLY",
    "EN_KNOWN_LEAF": "EN_KNOWN_ONLY",
    "EN_NEFF2_KNOWN_ONLY_REPLAY": "EN_NEFF2_KNOWN_ONLY",
    "EN_NEFF2_KNOWN_LEAF": "EN_NEFF2_KNOWN_ONLY",
}
LEAF_METHODS = {"EN_KNOWN_LEAF", "EN_NEFF2_KNOWN_LEAF"}


def support_membership(labels, base_ids, supported):
    import numpy as np

    selected = supported[base_ids]
    pairs = np.unique(np.c_[labels[selected], base_ids[selected]], axis=0)
    (roots, counts) = np.unique(pairs[:, 0], return_counts=True)
    (origins, parts) = np.unique(pairs[:, 1], return_counts=True)
    return dict(
        components_joining_multiple_supported_B=int((counts > 1).sum()),
        supported_B_split=int((parts > 1).sum()),
        multiple_supported_examples=[
            dict(component=int(r), source_B=pairs[pairs[:, 0] == r, 1].astype(int).tolist())
            for r in roots[counts > 1][:20]
        ],
        split_supported_examples=[
            dict(source_B=int(b), pieces=int(c))
            for (b, c) in zip(origins[parts > 1][:20], parts[parts > 1][:20])
        ],
        interpretation="All nodes of supported origin-B components; structure roles, not reference classes",
    )


def finish_reference(guard, ref, baseline, admitted, policy, X, base, H, active, core, bridge_mode):
    import numpy as np

    old = ref["receiver_study"]
    F = baseline["field_only_keep"]
    pairs = baseline["pairs"]
    if np.array_equal(admitted, baseline["actual_admitted"]):
        candidate = old["bridge_candidate"].copy()
        protected = old["bridge_protected"].copy()
        removed = old["bridge_removed"].copy()
        post = old["post_bridge_keep"].copy()
        route = "same_call_base_bridge"
    elif np.array_equal(admitted, baseline["proposed_U"]):
        candidate = baseline["original_bridge_candidate"].copy()
        protected = baseline["original_bridge_protected"].copy()
        removed = baseline["original_bridge_removed"].copy()
        post = baseline["original_U_post_bridge_keep"].copy()
        route = "same_call_original_U_bridge"
    else:
        graph = [set() for _ in range(len(X))]
        for mask in (F, admitted):
            for a, b in pairs[mask]:
                graph[int(a)].add(int(b))
                graph[int(b)].add(int(a))
        chosen = guard.deleted_bridges(base["M"], graph, H, active, "SB_DUAL")
        candidate = np.asarray([tuple(map(int, e)) in chosen for e in pairs], bool)
        (protected, _) = guard.protected_bridges(
            base, H, active, pairs, candidate, bridge_mode, core
        )
        removed = candidate & ~protected
        post = (F | admitted) & ~removed
        route = "recomputed_F_plus_final_A"
    assert np.array_equal(removed, candidate & ~protected) and np.array_equal(
        post, (F | admitted) & ~removed
    )
    if np.array_equal(admitted, baseline["actual_admitted"]):
        return (ref, route)
    rx = dict(
        old["receiver"],
        receiver_mode=policy,
        receiver_admitted=admitted.copy(),
        combined_pre_bridge_keep=F | admitted,
    )
    study = dict(
        old,
        receiver=rx,
        bridge_candidate=candidate,
        bridge_protected=protected,
        bridge_removed=removed,
        post_bridge_keep=post,
        receiver_policy=policy,
    )
    return (dict(ref, original_U_keep=post, receiver_study=study), route)


@contextmanager
def observe_gate(guard, method, audits, eps):
    import numpy as np

    assert method in BASE_MAP
    base_audits = []
    with observe_known(guard, BASE_MAP[method], base_audits, eps):
        base_function = guard.prepare_reference

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
            ref = base_function(
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
            assert len(base_audits) == 1
            b = base_audits[0]
            B = b["actual_post_bridge_keep"]
            A = b["actual_admitted"]
            U = b["proposed_U"]
            unknown = b["unknown_proposal_edge"]
            (view, values) = diagnose(b, B, b["original_U_post_bridge_keep"], BASE_MAP[method], "U")
            leaf = values["projection_LEAF"]
            assert not (leaf & ~unknown).any() and (not (leaf & A).any())
            admitted = A | leaf if method in LEAF_METHODS else A.copy()
            assert not (A & ~admitted).any() and (not (admitted & ~U).any())
            assert (
                view["summary"]["projections"]["LEAF"][
                    "components_joining_multiple_supported_base_components"
                ]
                == 0
            )
            if method in LEAF_METHODS:
                (result, route) = finish_reference(
                    guard, ref, b, admitted, "U__" + method, X, base, H, active, core, bridge_mode
                )
            else:
                result = ref
                route = "exact_known_control"
            study = result["receiver_study"]
            post = study["post_bridge_keep"]
            actual_roots = components(len(active), b["pairs"][post])
            fixed = B | (leaf if method in LEAF_METHODS else np.zeros(len(leaf), bool))
            fixed_roots = components(len(active), b["pairs"][fixed])
            actual = dict(
                admitted=admitted.copy(),
                field_only_keep=b["field_only_keep"].copy(),
                combined_pre_bridge=b["field_only_keep"] | admitted,
                bridge_candidate=study["bridge_candidate"].copy(),
                bridge_protected=study["bridge_protected"].copy(),
                bridge_removed=study["bridge_removed"].copy(),
                post_bridge_keep=post.copy(),
                policy=study["receiver_policy"],
                route=route,
                roots=actual_roots,
                fixed_projection_keep=fixed,
                fixed_projection_roots=fixed_roots,
                edge_difference_from_fixed_projection=int(np.count_nonzero(post ^ fixed)),
                same_partition_as_fixed_projection=bool(np.array_equal(actual_roots, fixed_roots)),
                support_membership=support_membership(
                    actual_roots, values["base_component"], values["base_supported"]
                ),
            )
            assert np.array_equal(
                actual["bridge_removed"], actual["bridge_candidate"] & ~actual["bridge_protected"]
            )
            assert np.array_equal(
                post, (b["field_only_keep"] | admitted) & ~actual["bridge_removed"]
            )
            assert np.array_equal(admitted, study["receiver"]["receiver_admitted"])
            audits.append(
                dict(
                    schema="178_SAME_CALL_KNOWN_BASE_PLUS_LEAF_V1",
                    method=method,
                    base_method=BASE_MAP[method],
                    baseline=b,
                    decision=view,
                    decision_arrays=values,
                    final=actual,
                    baseline_fingerprint=fingerprint(b),
                )
            )
            return result

        guard.prepare_reference = wrapped
        try:
            yield
        finally:
            assert guard.prepare_reference is wrapped
            guard.prepare_reference = base_function
