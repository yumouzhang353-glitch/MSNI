from contextlib import contextmanager
import numpy as np
from .field_gate175 import qualify as field_qualification, fingerprint

POLICIES = ("ORIGINAL", "OFF", "QUORUM", "FIELD_EN", "FIELD_EN2")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def marker_blocks(raw, seeds):
    n = len(raw)
    need(
        raw.dtype == seeds.dtype == np.int64
        and raw.shape == seeds.shape == (n,)
        and (raw >= 0).all()
        and (seeds >= -1).all(),
        "Raw/seed integer point domain",
    )
    selected = seeds >= 0
    keys = np.c_[raw, selected.astype(np.int64), np.where(selected, seeds, np.arange(n))]
    (_, blocks) = np.unique(keys, axis=0, return_inverse=True)
    return blocks.astype(np.int64)


def map_pairs(pool, proposals):
    need(
        pool.dtype == proposals.dtype == np.int64
        and pool.ndim == proposals.ndim == 2
        and (pool.shape[1] == proposals.shape[1] == 2),
        "Pair matrices",
    )
    need(
        (pool[:, 0] < pool[:, 1]).all() and (proposals[:, 0] < proposals[:, 1]).all(),
        "Canonical pair orientation",
    )
    lookup = {tuple(map(int, pair)): i for (i, pair) in enumerate(pool)}
    need(
        len(lookup) == len(pool) and len(np.unique(proposals, axis=0)) == len(proposals),
        "Unique pair domains",
    )
    need(
        all((tuple(map(int, pair)) in lookup for pair in proposals)),
        "Every C5 pair belongs to the exact current field pool",
    )
    return np.asarray([lookup[tuple(map(int, pair))] for pair in proposals], np.int64)


def qualify_selected(q1, shared, roles, d25, policy, eps):
    need(policy in ("FIELD_EN", "FIELD_EN2"), "New C5 policy")
    n = len(q1["seeds"])
    selected = q1["seeds"] >= 0
    need(
        np.array_equal(selected, q1["selected_mask"])
        and np.array_equal(q1["active_labels"][selected], q1["seeds"][selected]),
        "Actual frozen Q1 marker identity",
    )
    pool = roles["pairs"]
    need(
        np.array_equal(pool, shared["raw_edges"])
        and np.array_equal(roles["source_raw_ids"], q1["raw_ids"])
        and np.array_equal(shared["active"], q1["active"])
        and (len(shared["active"]) == n),
        "Current field/raw/active identity",
    )
    (ij, ji) = (roles["source_final_G_ij"], roles["source_final_G_ji"])
    need(np.array_equal(ij, ji), "Symmetric corrected field, one effective direction test")
    index = map_pairs(pool, d25["pairs"])
    proposed = np.zeros(len(pool), bool)
    proposed[index] = d25["effective_pass"]
    blocks = marker_blocks(q1["raw_ids"], q1["seeds"])
    certificate = field_qualification(pool, ij, ji, blocks, q1["raw_ids"], proposed, eps)
    accepted = certificate[
        "certified_enrich_edge" if policy == "FIELD_EN" else "certified_neff_edge"
    ]
    accepted = accepted & certificate["valid_positive_force"] & selected[pool].all(axis=1)
    effective = d25["effective_pass"] & accepted[index]
    need(
        not (accepted & ~proposed).any() and (not (effective & ~d25["effective_pass"]).any()),
        "C5 intervention only removes original D25 edges",
    )
    need(
        not (accepted & certificate["unknown_edge"]).any(),
        "No unknown fallback in C5 structural certification",
    )
    return (
        effective,
        dict(
            schema="187_FIXED_Q1_SELECTED_FIELD_C5_V1",
            policy=policy,
            selected_marker_blocks=blocks,
            source_corrected_field_roles=roles,
            source_D25=d25,
            C5_pair_to_field_index=index,
            field_qualification=certificate,
            final_pool_keep=accepted,
            final_C5_keep=effective,
            effective_direction_tests=1,
            unknown_policy="reject",
            reference_values_received=False,
            qualification_recomputed_after_each_merge=False,
        ),
    )


@contextmanager
def intervene(nd100, policy, audits, eps):
    need(policy in POLICIES, "Declared C5 policy")
    (compose, qualify) = (nd100.old.compose_one, nd100.qualify)
    captured = []

    def capture(*args, **kwargs):
        result = compose(*args, **kwargs)
        need(not captured, "One same-call complete prefix")
        captured.append(result)
        return result

    def changed(labels, seeds, rho, ptr, indices, pairs, source_pass, rule):
        need(
            len(captured) == 1 and rule == "D25_ANY" and (not audits),
            "Only one last D25 qualification is replaced",
        )
        payload = captured[0]
        q1 = payload["checkpoints"]["rebuilt_Q1"]
        shared = payload["a"]["SHARED"]
        roles = payload["a"]["CORE_BOUNDARY"]["source_roles"]
        for value, expected in (
            (labels, q1["active_labels"]),
            (seeds, q1["seeds"]),
            (rho, shared["rho"]),
            (ptr, shared["NN_indptr"]),
            (indices, shared["NN_indices"]),
        ):
            need(np.array_equal(value, expected), "Same-call Q1 and field inputs")
        immutable = dict(
            q1={
                key: q1[key]
                for key in ("active", "active_labels", "seeds", "selected_mask", "raw_ids")
            },
            shared={
                key: shared[key]
                for key in ("active", "rho", "raw_edges", "NN_indptr", "NN_indices")
            },
            roles={
                key: roles[key]
                for key in ("pairs", "source_raw_ids", "source_final_G_ij", "source_final_G_ji")
            },
        )
        before = fingerprint(immutable)
        original = qualify(labels, seeds, rho, ptr, indices, pairs, source_pass, "D25_ANY")
        if policy == "ORIGINAL":
            (post, evidence) = (original, None)
        elif policy in ("OFF", "QUORUM"):
            post = qualify(
                labels,
                seeds,
                rho,
                ptr,
                indices,
                pairs,
                source_pass,
                "NO_C5" if policy == "OFF" else "D25_QUORUM",
            )
            evidence = None
        else:
            (effective, evidence) = qualify_selected(q1, shared, roles, original, policy, eps)
            counts = np.bincount(
                original["edge_domain"][effective], minlength=len(original["domain_keys"])
            )
            post = dict(
                original,
                rule="D25_" + policy,
                effective_pass=effective,
                domain_admitted=counts > 0,
                domain_certified_edges=counts,
                upstream_rule="D25_ANY",
                upstream_domain_unknown_fallback=original["domain_unknown_fallback"],
                domain_unknown_fallback=np.zeros(len(counts), bool),
                unknown_fallback="reject_unknown_field_support",
                retained_edges="certified_positive_selected_subset_of_original_D25",
            )
        need(
            fingerprint(immutable) == before,
            "Immutable Q1/shared/corrected field inputs actually used",
        )

        def group_pairs(mask):
            return np.unique(np.sort(labels[pairs[mask]], axis=1), axis=0).astype(np.int64)

        audits.append(
            dict(
                policy=policy,
                original_D25=original,
                actual_post=post,
                evidence=evidence,
                source_group_pairs=group_pairs(original["effective_pass"]),
                accepted_group_pairs=group_pairs(post["effective_pass"]),
                Q1_input_fingerprint=before,
                prefix_capture_calls=1,
                final_qualification_calls=1,
            )
        )
        return post

    (nd100.old.compose_one, nd100.qualify) = (capture, changed)
    try:
        yield
    finally:
        need(nd100.old.compose_one is capture and nd100.qualify is changed, "C5 hook ownership")
        (nd100.old.compose_one, nd100.qualify) = (compose, qualify)
