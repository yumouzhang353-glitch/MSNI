import numpy as np
from .sni_single_native import _field
from . import geometry_adapter167 as _geometry

GUARDS = ("OLD_REJECT", "THRESHOLD", "SCALE", "STRUCTURE", "RELIABILITY", "GEOMETRY", "ALL")


def view(X, base, ev, f0, uncorrected, field, shape, reference, mode, core):
    assert mode in GUARDS and mode != "OLD_REJECT"
    refbase = reference["base"]
    (vev, vbase, vshape, vf0, vunc) = (ev, base, shape, f0, uncorrected)
    if mode == "THRESHOLD":
        vfield = field
    elif mode == "SCALE":
        scale = reference["uncorrected"]["scale"]
        (a, b) = ev["pairs"].T
        from .sni_single_native._config import EPS

        vunc = dict(
            uncorrected, scale=scale, t=ev["distance"] / (np.sqrt(scale[a] * scale[b]) + EPS)
        )
        vfield = _geometry.correct(X, base, ev, vunc, shape, "LW_GM", core)
    elif mode == "STRUCTURE":
        sigma = core.compute_structure_similarity(base["M"], refbase["gamma"], base["reliability"])
        values = np.asarray([sigma[tuple(map(int, e))] for e in ev["pairs"]])
        vunc = dict(uncorrected, sigma=values)
        vfield = _geometry.correct(X, base, ev, vunc, shape, "LW_GM", core)
    elif mode == "RELIABILITY":
        reliability = refbase["reliability"]
        sigma = core.compute_structure_similarity(base["M"], base["gamma"], reliability)
        vbase = dict(base, reliability=reliability, sigma=sigma)
        vbase["G"] = core.compute_gravity(X, vbase["M"], sigma, reliability, vbase["local_scale"])
        vev = dict(ev, original_sigma=np.asarray([sigma[tuple(map(int, e))] for e in ev["pairs"]]))
        vf0 = _field.interaction(X, vbase, vev, "F0", core)
        vunc = _field.interaction(X, vbase, vev, "C1", core)
        vshape = reference["shape"]
        vfield = _geometry.correct(X, vbase, vev, vunc, vshape, "LW_GM", core)
    elif mode == "GEOMETRY":
        vshape = reference["shape"]
        vfield = _geometry.correct(X, base, ev, uncorrected, vshape, "LW_GM", core)
    else:
        assert mode == "ALL"
        sigma = core.compute_structure_similarity(
            base["M"], refbase["gamma"], refbase["reliability"]
        )
        vbase = dict(refbase, M=base["M"], sigma=sigma)
        vbase["G"] = core.compute_gravity(
            X, vbase["M"], sigma, vbase["reliability"], vbase["local_scale"]
        )
        vev = _field.evidence(X, vbase)
        assert np.array_equal(vev["pairs"], ev["pairs"])
        vf0 = _field.interaction(X, vbase, vev, "F0", core)
        vunc = _field.interaction(X, vbase, vev, "C1", core)
        vshape = reference["shape"]
        vfield = _geometry.correct(X, vbase, vev, vunc, vshape, "LW_GM", core)
    args = (
        dict(f0ctx=reference["f0ctx"], currentctx=reference["currentctx"])
        if mode in ("THRESHOLD", "ALL")
        else {}
    )
    context = _field.threshold_context(vev, vf0, vfield, len(X), "T8", **args)
    (_, decisions) = _field.build_strong_graph(vev, vfield, len(X), context)
    return (vev, vf0, vunc, vfield, context, decisions)


def apply(X, base, ev, f0, uncorrected, field, shape, reference, native_keep, mode, core):
    assert mode in GUARDS
    _geometry.require_canonical_pairs(ev["pairs"], len(X))
    _geometry.require_canonical_pairs(reference["ev"]["pairs"], len(X))
    original_positions = {tuple(map(int, e)): k for (k, e) in enumerate(reference["ev"]["pairs"])}
    mapped = np.asarray(
        [original_positions.get(tuple(map(int, e)), -1) for e in ev["pairs"]], dtype=np.int64
    )
    original = mapped >= 0
    original_U = np.zeros(len(mapped), dtype=bool)
    original_U[original] = reference["original_U_keep"][mapped[original]]
    arrays = dict(
        guard_mode_code=np.asarray(GUARDS.index(mode)),
        guard_native_keep=native_keep.copy(),
        guard_original_pair_index=mapped,
        guard_original_U_keep_mapped=original_U,
        guard_reference_pairs=reference["ev"]["pairs"],
        guard_reference_U_keep=reference["original_U_keep"],
        guard_reference_reliability=reference["base"]["reliability"],
        guard_reference_C1_scale=reference["uncorrected"]["scale"],
        guard_reference_geometry_inverse=reference["shape"]["inverse"],
        guard_reference_row=reference["context"]["row_threshold"],
        guard_reference_f0_row=reference["f0ctx"]["row_threshold"],
        guard_reference_current_row=reference["currentctx"]["row_threshold"],
    )
    for name in ["tau95", "tau85"]:
        arrays["guard_reference_f0_" + name] = np.asarray(reference["f0ctx"][name])
        arrays["guard_reference_current_" + name] = np.asarray(reference["currentctx"][name])
    if mode == "OLD_REJECT":
        passing = ~original | original_U
    else:
        (vev, vf0, vunc, vfield, context, decisions) = view(
            X, base, ev, f0, uncorrected, field, shape, reference, mode, core
        )
        passing = decisions["keep"]
        arrays.update(
            guard_view_sigma=vfield["sigma"],
            guard_view_scale=vfield["scale"],
            guard_view_t=vfield["t"],
            guard_view_f0_t=vf0["t"],
            guard_view_G_ij=vfield["G_ij"],
            guard_view_G_ji=vfield["G_ji"],
            guard_view_direction=vfield["edge_direction_factor"],
            guard_view_mutual=vev["mutual"],
            guard_view_row=context["row_threshold"],
            guard_view_mutual_distance=context["mutual_distance_pass"],
            guard_view_strong_distance=context["strong_distance_pass"],
            guard_view_mutual_branch=decisions["mutual_branch"],
            guard_view_both_branch=decisions["both_branch"],
        )
        for name in ["tau95", "tau85", "f0_tau95", "f0_tau85", "current_tau95", "current_tau85"]:
            arrays["guard_view_" + name] = np.asarray(context[name])
    final = native_keep & passing
    arrays.update(
        guard_view_keep=passing, guard_final_keep=final, guard_rejected=native_keep & ~passing
    )
    assert not np.any(final & ~native_keep)
    return (final, arrays)
