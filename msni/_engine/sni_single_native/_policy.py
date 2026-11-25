import numpy as np
from . import _quota as quota

VERSIONS = ["QR3_GRAPH"]


def quota_pre_state(X, core, version):
    assert version in VERSIONS
    return quota.quota_pre_state(X, core, "NQ1_BLOCK")


def intervention_policy(pre, ranks, version, active):
    assert version in VERSIONS
    arrays = pre["quota_arrays"]
    nb = arrays["quota_nb"]
    (cut, capacity) = (int(arrays["quota_boundary_nb"]), int(arrays["quota_capacity"]))
    boundary = arrays["quota_candidates"] & (nb == cut) if cut >= 0 else np.zeros(len(nb), bool)
    omitted = boundary & ~pre["B"]
    retained_boundary = omitted[active]
    trigger = bool(cut >= 0 and int(boundary.sum()) >= capacity and retained_boundary.any())
    q0 = int(np.max(ranks))
    q = q0
    geometry_q = q0
    changed = np.asarray(ranks, int).copy()
    if trigger:
        if version == "QR2_GRAPH":
            q = geometry_q = min(len(active) - 1, 2 * q0)
            changed[:] = q
        elif version == "QR3_GRAPH":
            q = geometry_q = min(len(active) - 1, 3 * q0)
            changed[:] = q
        else:
            geometry_q = min(len(active) - 1, 2 * q0)
    assert np.all((changed >= 1) & (changed < len(active))) and 1 <= geometry_q < len(active)
    return (
        changed,
        geometry_q,
        dict(
            quota_intervention_trigger=np.asarray(trigger),
            quota_intervention_boundary_size=np.asarray(int(boundary.sum())),
            quota_intervention_retained_boundary=np.asarray(int(retained_boundary.sum())),
            quota_intervention_legacy_ranks=np.asarray(ranks, int),
            quota_intervention_graph_rank=np.asarray(q),
            quota_intervention_geometry_rank=np.asarray(geometry_q),
        ),
    )
