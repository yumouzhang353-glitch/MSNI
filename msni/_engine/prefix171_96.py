import time
from . import pipeline91_v3 as previous
from .trace171_nd import run_single


def native(X, mode, original_receiver="U", native_receiver="U"):
    started = time.perf_counter()
    (baseline, audit) = run_single(X, mode, original_receiver, native_receiver)
    trace_seconds = time.perf_counter() - started
    started = time.perf_counter()
    from .single_c5 import eligibility, merge, suffix as c5_suffix
    from .geometry_bridge167 import propose as old_propose

    (old_edges, old_candidates) = old_propose(
        X[baseline["active"]], audit["shared"], previous._core
    )
    (masks, counts, unique) = eligibility(baseline, old_edges["E2"])
    q1 = merge(baseline, old_edges["E2"], masks["Q1"], counts, unique)
    (c5, c5_audit) = c5_suffix(X[q1["active"]], q1, audit["shared"], audit["target"])
    a = dict(
        BASELINE=baseline,
        Q1=q1,
        C5=c5,
        OLD_CANDIDATES=old_candidates,
        **c5_audit,
        RECEIVER_STUDY=audit["receiver_study"],
        **{
            key.upper(): audit[key]
            for key in ("pre", "shared", "reference", "target", "support", "core_boundary")
        },
    )
    q1_c5_seconds = time.perf_counter() - started
    started = time.perf_counter()
    neighbors = previous.prepare_neighbors(X[c5["active"]], a["SHARED"])
    (a8, a8_votes) = previous.assign_active(X, c5, neighbors, "A8_NN_FIELD_PROP")
    (r6, r6_votes, local, r6_meta) = previous.restore(X, a8, a["PRE"])
    a.update(
        A8_NN_FIELD_PROP=a8,
        R6_FULL_LOCAL_PROP=r6,
        A8_VOTES=a8_votes,
        R6_VOTES=r6_votes,
        LOCAL_SHARED=local,
    )
    return (
        a,
        dict(
            trace_seconds=trace_seconds,
            Q1_C5_seconds=q1_c5_seconds,
            native_A8_R6_seconds=time.perf_counter() - started,
            R6_metadata=r6_meta,
            independent_full_X_prefix=True,
            no_previous_inference_arrays=True,
        ),
    )


def compose_one(X, mode, original_receiver="U", native_receiver="U"):
    started = time.perf_counter()
    (a, metadata) = native(X, mode, original_receiver, native_receiver)
    post = previous.qualify(a["Q1"], a["SHARED"], a["TARGET"], a, "C0")
    payload = previous.finish(X, a, "C0", post)
    payload.update(boundary_mode=mode, boundary_prefix_metadata=metadata)
    metadata["S99_X50_finish_seconds"] = (
        time.perf_counter()
        - started
        - sum((metadata[k] for k in ("trace_seconds", "Q1_C5_seconds", "native_A8_R6_seconds")))
    )
    return payload
