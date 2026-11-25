from .single_r6 import _core, prepare_neighbors, assign_active, suffix as restore
from .single_c5 import eligibility, merge, suffix as c5_suffix
from .fragment_quotient import coarsen as c5_coarsen
from .support_candidates81 import (
    prepare_evidence,
    choose_supports,
    rebuild_baseline,
    small_contacts,
    coarsen,
)
from .post_contact91 import qualify


def finish(X, a, mode, post_native):
    evidence = prepare_evidence(X, a)
    (chosen, eligible) = choose_supports(a["BASELINE"], *evidence, "S99")
    if len(chosen):
        baseline = rebuild_baseline(X, a, chosen)
        geom = a["OLD_CANDIDATES"]
        e2 = geom["pairs"][geom["admitted_E2"]]
        (masks, counts, unique) = eligibility(baseline, e2)
        q1 = merge(baseline, e2, masks["Q1"], counts, unique)
        (original_c5, c5_audit) = c5_suffix(X[q1["active"]], q1, a["SHARED"], a["TARGET"])
        post = qualify(q1, a["SHARED"], a["TARGET"], c5_audit, mode)
        c5 = original_c5 if mode == "C0" else c5_coarsen(q1, post["pairs"][post["effective_pass"]])
        neighbors = prepare_neighbors(X[c5["active"]], a["SHARED"])
        (a8, a8_votes) = assign_active(X, c5, neighbors, "A8_NN_FIELD_PROP")
        (prefix, r6_votes, local, _) = restore(X, a8, a["PRE"])
    else:
        (baseline, q1, c5) = (a["BASELINE"], a["Q1"], a["C5"])
        c5_audit = {k: a[k] for k in ("CONTACTS", "TRUSTED", "GATE")}
        post = post_native
        (a8, prefix) = (a["A8_NN_FIELD_PROP"], a["R6_FULL_LOCAL_PROP"])
        (a8_votes, r6_votes, local) = (a["A8_VOTES"], a["R6_VOTES"], a["LOCAL_SHARED"])
    (edges, ticket) = small_contacts(prefix, a["OLD_CANDIDATES"], "X50")
    output = coarsen(prefix, edges)
    ticket.update(
        promotion_cut_ids=chosen,
        promotion_eligible=eligible,
        source_reference_seeds=a["BASELINE"]["reference_seeds"],
        source_reference_caps=a["BASELINE"]["reference_caps"],
        proposed_reference_seeds=baseline["reference_seeds"],
        proposed_reference_caps=baseline["reference_caps"],
        target_baseline_seeds=baseline["seeds"],
        target_baseline_active_labels=baseline["active_labels"],
        target_baseline_labels=baseline["labels"],
        target_baseline_restoration_witness=baseline["restoration_witness"],
        target_baseline_quotient=baseline["seed_quotient"],
        prefix_labels=prefix["labels"],
        prefix_active_labels=prefix["active_labels"],
        prefix_seeds=prefix["seeds"],
        prefix_reference_seeds=prefix["reference_seeds"],
        prefix_caps=prefix["reference_caps"],
        prefix_used_caps=prefix["used_caps"],
        prefix_restoration_witness=prefix["restoration_witness"],
        accepted_edges=edges,
        post_mapping=output["post_source_group_to_final_group"],
    )
    checkpoints = dict(
        S99_baseline=baseline,
        rebuilt_Q1=q1,
        rebuilt_C5=c5,
        actual_A8=a8,
        actual_R6=prefix,
        final_X50=output,
        actual_C5_audit=c5_audit,
        actual_post=post,
        actual_A8_votes=a8_votes,
        actual_R6_votes=r6_votes,
        actual_local_shared=local,
        S99_triggered=bool(len(chosen)),
        A8_seeds=c5["seeds"],
    )
    return dict(
        a=a,
        post_native=post_native,
        output=output,
        ticket=ticket,
        checkpoints=checkpoints,
        support_evidence=evidence,
    )
