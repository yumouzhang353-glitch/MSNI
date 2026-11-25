from threading import RLock

import numpy as np
from threadpoolctl import threadpool_limits

from . import fit_direction as channel
from . import geometry_contact_guard171 as contacts
from . import prefix171_100 as consolidation
from .c5_field187 import intervene
from .fusion import fuse, solve
from .guarded_a8 import protect
from .inherit import inherited
from .leaf_gate178 import observe_gate
from .natural_graph import build_graph, static_weights
from .quota_fit import InfeasibleSupport, native_quota_error
from .sni_single_native._config import EPS
from .trial import continuation


_LEIDEN = dict(
    partition="RBConfigurationVertexPartition",
    resolution_parameter=1.0,
    n_iterations=2,
    seed=187006,
    initial_membership=None,
    max_comm_size=0,
)
_FIT_LOCK = RLock()


def _consolidate(X):
    reference_checks, merge_checks = [], []
    with observe_gate(contacts, "EN_KNOWN_ONLY_REPLAY", reference_checks, EPS):
        with intervene(consolidation, "OFF", merge_checks, EPS):
            payload = consolidation.compose_one(X, "W_RAW_MIN__D25_ANY", "U", "U")
    if len(reference_checks) != 1 or len(merge_checks) != 1:
        raise RuntimeError("Incomplete structural consolidation")
    if merge_checks[0]["actual_post"]["effective_pass"].any():
        raise RuntimeError("Final internal-contact policy differs from the fixed method")
    return payload


def _bounded_consolidation(X):
    apply_decision, detect = channel.apply_decision, channel.detect
    latest = {}

    def observe(*args):
        result = apply_decision(*args)
        latest["trace"] = args[-1]
        return result

    def run():
        trace = {}
        np.random.seed(187006)
        with channel.before_support(trace):
            payload = _consolidate(X)
        for stage, baseline in (
            ("initial", payload["a"]["BASELINE"]),
            ("S99", payload["checkpoints"]["S99_baseline"]),
        ):
            used, cap = baseline["used_caps"], baseline["reference_caps"]
            if np.any(used > cap) and (trace["link"] or trace["cuts"]):
                raise InfeasibleSupport(stage, used, cap)
        return payload

    try:
        channel.apply_decision = observe
        try:
            return run()
        except AssertionError as error:
            failure = native_quota_error(error)
            trace = latest.get("trace", {})
            if failure is None or not (trace.get("link") or trace.get("cuts")):
                raise
        except InfeasibleSupport:
            pass

        attempted = latest["trace"]

        def use_standard_path(_):
            return dict(
                decision="fallback_native_quota",
                route=attempted["route"],
                attempted_decision=attempted["decision"],
            )

        channel.detect = use_standard_path
        return run()
    finally:
        channel.apply_decision = apply_decision
        channel.detect = detect


def _structural_evidence(payload):
    state = payload["a"]
    out = payload["output"]
    baseline = state["BASELINE"]
    bridge = state["CORE_BOUNDARY"]
    receiver = state["RECEIVER_STUDY"]["native"]
    shared = state["SHARED"]
    active = baseline["active"]
    pairs = bridge["source_roles"]["pairs"]
    assert np.array_equal(active, out["active"])
    assert np.array_equal(pairs, receiver["pairs"])
    assert np.array_equal(pairs[bridge["target_keep"]], baseline["strong_edges"])

    return dict(
        active=active,
        pairs=active[pairs],
        base_force=np.sqrt(shared["G_ij"]) * np.sqrt(shared["G_ji"]),
        corrected_force=bridge["field_weights"],
        bridge_protected=receiver["bridge_protected"],
        first_bridge_removed=receiver["bridge_removed"],
        bridge_removed=bridge["removed_pair_mask"],
        reference=bridge["reference_keep"],
        target=bridge["target_keep"],
        selected=active[baseline["selected_mask"]],
        final_supports=out["active"][out["selected_mask"]],
        reliability=shared["reliability"],
        protected=state["SUPPORT"]["support_applied_protect"],
        propagation=state["A8_VOTES"]["confidence"],
        restoration=state["R6_VOTES"]["confidence"],
    )


def _community_recovery(X, payload):
    graph = build_graph(X)
    pairs = graph["pairs"]
    affinity, _, _ = static_weights(pairs, graph["gravity"], len(X), "R_NAT_GSYM")
    positive = affinity > 0
    pairs, weights, _ = fuse(
        len(X),
        pairs[positive],
        affinity[positive],
        _structural_evidence(payload),
        0.20,
        True,
    )
    labels, _ = solve(len(X), pairs, weights, _LEIDEN)
    assert len(labels) == len(X) and np.all(labels >= 0)
    return labels


def fit_predict(X):
    features = np.array(X, dtype=np.float64, order="C", copy=True)
    if features.ndim != 2 or not 2 <= len(features) <= 10_000 or not 1 <= features.shape[1] <= 552:
        raise ValueError("Expected 2–10,000 rows and 1–552 features")
    if not np.isfinite(features).all():
        raise ValueError("MSNI requires finite feature values")
    features.setflags(write=False)

    with _FIT_LOCK, threadpool_limits(1):
        rng = np.random.get_state()
        try:
            support_change = dict(trigger=False, calls=[])
            with protect(support_change, []), continuation(support_change), inherited([]):
                payload = _bounded_consolidation(features)
            labels = payload["output"]["labels"]
            if np.unique(labels).size == 1:
                proposal = _community_recovery(features, payload)
                if np.unique(proposal).size > 1:
                    labels = proposal
            return labels.copy()
        finally:
            np.random.set_state(rng)
