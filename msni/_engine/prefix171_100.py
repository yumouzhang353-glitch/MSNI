import time
import numpy as np
from . import prefix171_96 as old
from .contact_density100 import MODES, qualify


def compose_one(X, version, original_receiver="U", native_receiver="U"):
    if version not in MODES:
        raise ValueError(version)
    boundary = version.split("__")[0]
    tick = time.perf_counter()
    p = old.compose_one(X, boundary, original_receiver, native_receiver)
    initial_seconds = time.perf_counter() - tick
    p.update(
        version=version,
        scientific_source="newly_inferred_from_X",
        receiver_configuration=dict(original=original_receiver, native=native_receiver),
    )
    if "__" not in version:
        p["contact_record"] = dict(
            enabled=False,
            rule="CONTROL",
            rejected_edges=0,
            source_groups=int(
                len(
                    np.unique(
                        p["checkpoints"]["rebuilt_Q1"]["seeds"][
                            p["checkpoints"]["rebuilt_Q1"]["selected_mask"]
                        ]
                    )
                )
            ),
        )
        p["phase100_timing"] = dict(
            original_full_pipeline_seconds=initial_seconds,
            qualification_seconds=0.0,
            fresh_C5_A8_R6_X50_seconds=0.0,
        )
        return p
    rule = version.split("__")[1]
    prior = dict(p)
    prior["version"] = boundary
    cp = p["checkpoints"]
    q1 = cp["rebuilt_Q1"]
    shared = p["a"]["SHARED"]
    audit = cp["actual_C5_audit"]
    rho = shared["rho"]
    tick = time.perf_counter()
    post = qualify(
        q1["active_labels"],
        q1["seeds"],
        rho,
        shared["NN_indptr"],
        shared["NN_indices"],
        audit["GATE"]["pairs"],
        audit["GATE"]["admitted_C5"],
        rule,
    )
    qualify_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    c5 = old.previous.c5_coarsen(q1, post["pairs"][post["effective_pass"]])
    neighbors = old.previous.prepare_neighbors(X[c5["active"]], shared)
    (a8, a8_votes) = old.previous.assign_active(X, c5, neighbors, "A8_NN_FIELD_PROP")
    (r6, r6_votes, local, r6_meta) = old.previous.restore(X, a8, p["a"]["PRE"])
    (edges, ticket) = old.previous.small_contacts(r6, p["a"]["OLD_CANDIDATES"], "X50")
    output = old.previous.coarsen(r6, edges)
    for key, value in p["ticket"].items():
        if key.startswith(
            ("promotion_", "source_reference_", "proposed_reference_", "target_baseline_")
        ):
            ticket[key] = value
    ticket.update(
        prefix_labels=r6["labels"],
        prefix_active_labels=r6["active_labels"],
        prefix_seeds=r6["seeds"],
        prefix_reference_seeds=r6["reference_seeds"],
        prefix_caps=r6["reference_caps"],
        prefix_used_caps=r6["used_caps"],
        prefix_restoration_witness=r6["restoration_witness"],
        accepted_edges=edges,
        post_mapping=output["post_source_group_to_final_group"],
    )
    actual = dict(cp)
    actual.update(
        rebuilt_C5=c5,
        actual_A8=a8,
        actual_R6=r6,
        final_X50=output,
        actual_A8_votes=a8_votes,
        actual_R6_votes=r6_votes,
        actual_local_shared=local,
        A8_seeds=c5["seeds"],
        actual_contact_density=post,
        actual_post=post,
        actual_post_schema="phase100_density_contact_domain",
        pre_contact_actual_post=cp["actual_post"],
    )
    p.update(
        checkpoints=actual,
        output=output,
        ticket=ticket,
        pre_contact_payload=prior,
        contact_record=dict(
            enabled=True,
            rule=rule,
            source_labels=q1["active_labels"].copy(),
            source_seeds=q1["seeds"].copy(),
            source_full_ids=q1["active"].copy(),
            qualification=post,
            rejected_edges=int((post["source_pass"] & ~post["effective_pass"]).sum()),
            original_C5_groups=int(
                len(np.unique(cp["rebuilt_C5"]["seeds"][cp["rebuilt_C5"]["selected_mask"]]))
            ),
            actual_C5_groups=int(len(np.unique(c5["seeds"][c5["selected_mask"]]))),
            actual_C5_calls=1,
            A8_calls=1,
            R6_calls=1,
            X50_contact_calls=1,
            old_A8_or_R6_not_reused=True,
            S99_choices_unchanged=True,
            F_ORIGINAL_anchor_and_caps_unchanged=True,
            R6_metadata=r6_meta,
        ),
    )
    p["phase100_timing"] = dict(
        original_full_pipeline_seconds=initial_seconds,
        qualification_seconds=qualify_seconds,
        fresh_C5_A8_R6_X50_seconds=time.perf_counter() - tick,
    )
    return p
