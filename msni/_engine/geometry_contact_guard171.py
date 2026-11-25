import numpy as np
from .sni_single_native import _field
from . import geometry_adapter167 as _geometry
from .sni_single_native._budget_adapter import prepare_anchor
from .receiver171 import admit
from .sni_single_native._bridge import deleted_bridges
from .sni_single_native._paths import protected_bridges


def prepare_reference(
    X, base, geometry_q, H, active, core, bridge_mode, quota_mode, tie_geometry, receiver_mode="U"
):
    ev = _field.evidence(X, base)
    _geometry.require_canonical_pairs(ev["pairs"], len(X))
    f0 = _field.interaction(X, base, ev, "F0", core)
    uncorrected = _field.interaction(X, base, ev, "C1", core)
    shape = _geometry.prepare(
        X, int(geometry_q), base["reliability"], weighted=True, tie_closed=tie_geometry
    )
    field = _geometry.correct(X, base, ev, uncorrected, shape, "LW_GM", core)
    (f0ctx, currentctx) = (_field.context(ev, f0, len(X)), _field.context(ev, field, len(X)))
    context = _field.threshold_context(
        ev, f0, field, len(X), "T8", f0ctx=f0ctx, currentctx=currentctx
    )
    (strong, branches) = _field.build_strong_graph(ev, field, len(X), context)
    anchor = prepare_anchor(X, base, strong, ev["pairs"], H, active, core, bridge_mode, quota_mode)
    (strong, receiver) = admit(X, base, ev, strong, branches["keep"], receiver_mode, core)
    candidates = deleted_bridges(base["M"], strong, H, active, "SB_DUAL")
    candidate = np.asarray([tuple(map(int, e)) in candidates for e in ev["pairs"]], dtype=bool)
    (protection, _) = protected_bridges(base, H, active, ev["pairs"], candidate, bridge_mode, core)
    original_U_keep = receiver["combined_pre_bridge_keep"] & ~(candidate & ~protection)
    return dict(
        base=base,
        ev=ev,
        f0=f0,
        uncorrected=uncorrected,
        field=field,
        shape=shape,
        f0ctx=f0ctx,
        currentctx=currentctx,
        context=context,
        anchor=anchor,
        original_U_keep=original_U_keep,
        receiver_study=dict(
            pairs=ev["pairs"].copy(),
            field_only_keep=branches["keep"].copy(),
            receiver=receiver,
            bridge_candidate=candidate.copy(),
            bridge_protected=protection.copy(),
            bridge_removed=candidate & ~protection,
            post_bridge_keep=original_U_keep.copy(),
            receiver_policy=receiver_mode,
            branch="original_pre_I2",
        ),
    )
