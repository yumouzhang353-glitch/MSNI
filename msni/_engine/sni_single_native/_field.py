import math
import numpy as np
from ._config import EPS, PARAMETERS
from ._pipeline import edges

PARAMS = {"candidate_parameters": PARAMETERS["field_parameters"]}
FIELD_BACKENDS = ["F0", "L1"] + [f"C{i}" for i in range(1, 8)]
CANDIDATES = [f"T{i}" for i in range(1, 10)]
SPEC = PARAMETERS["threshold_versions"]


def alternative_scale(X, gamma, method="L1"):
    assert method == "L1"
    scales = np.zeros(len(X), dtype=float)
    for i, ns in enumerate(gamma):
        ds = np.asarray([float(np.linalg.norm(X[i] - X[int(j)])) for j in ns if int(j) != i])
        ds = ds[ds > EPS]
        if len(ds):
            scales[i] = float(np.median(ds))
    p = scales[scales > 0]
    scales[scales <= EPS] = max(float(np.median(p)) if len(p) else 1.0, EPS)
    return scales


def evidence(X, base):
    pairs = edges(base["M"])
    distances = np.asarray([float(np.linalg.norm(X[i] - X[j])) for (i, j) in pairs])
    sigma = np.asarray([base["sigma"].get((int(i), int(j)), 0.0) for (i, j) in pairs])
    (q, overlap, mass_i, mass_j, mass_common) = ([], [], [], [], [])
    mutual = []
    for i, j in pairs:
        a = base["gamma"][i] - {int(i), int(j)}
        b = base["gamma"][j] - {int(i), int(j)}
        ai = sum((float(base["reliability"][u]) for u in sorted(a)))
        bj = sum((float(base["reliability"][u]) for u in sorted(b)))
        c = sum((float(base["reliability"][u]) for u in sorted(a & b)))
        q.append(c / (math.sqrt(ai * bj) + EPS))
        overlap.append(c / (min(ai, bj) + EPS))
        mass_i.append(ai)
        mass_j.append(bj)
        mass_common.append(c)
        mutual.append(int(j) in base["S"][i])
    mu = base["local_scale"]
    median = alternative_scale(X, base["gamma"])
    (i, j) = (pairs[:, 0], pairs[:, 1])
    out = {
        "pairs": pairs,
        "distance": distances,
        "original_sigma": sigma,
        "third_support": np.clip(np.asarray(q), 0.0, 1.0),
        "third_overlap": np.clip(np.asarray(overlap), 0.0, 1.0),
        "third_mass_i": np.asarray(mass_i),
        "third_mass_j": np.asarray(mass_j),
        "third_common_mass": np.asarray(mass_common),
        "mutual": np.asarray(mutual, dtype=bool),
        "median": median,
        "t_mu": distances / (np.sqrt(mu[i] * mu[j]) + EPS),
        "t_median": distances / (np.sqrt(median[i] * median[j]) + EPS),
    }
    return out


def weighted_scale(ev, mu, weights):
    good = ev["distance"] > EPS
    p = ev["pairs"][good]
    weights = weights[good]
    (numerator, denominator) = (np.zeros(len(mu)), np.zeros(len(mu)))
    for side in [0, 1]:
        np.add.at(numerator, p[:, side], weights * ev["distance"][good])
        np.add.at(denominator, p[:, side], weights)
    result = mu.copy()
    np.divide(numerator, denominator, out=result, where=denominator > EPS)
    return result


def specification(backend):
    (name, _, setting) = backend.partition("_")
    p = PARAMS["candidate_parameters"].get(name)
    if setting in ["LOW", "HIGH"]:
        p = PARAMS["sensitivity_parameters"][name][int(setting == "HIGH")]
    assert name in FIELD_BACKENDS and setting in ["", "LOW", "HIGH", "THR", "CAP"]
    return (name, p, setting)


def interaction(X, base, ev, backend, core, parameter_override=None):
    (name, p, mode) = specification(backend)
    if parameter_override is not None:
        p = parameter_override
    (mu, median) = (base["local_scale"], ev["median"])
    scale = mu.copy()
    sigma = ev["original_sigma"].copy()
    t = ev["t_mu"].copy()
    (i, j) = (ev["pairs"][:, 0], ev["pairs"][:, 1])
    role = "node"
    if name == "L1":
        scale = median.copy()
    elif name == "C1":
        scale = (
            mu.copy()
            if p == 0
            else median.copy()
            if p == 1
            else np.sqrt(mu * median)
            if p == 0.5
            else mu ** (1 - p) * median**p
        )
    elif name == "C2":
        scale = weighted_scale(ev, mu, sigma**p)
    elif name == "C3":
        role = "edge"
        t = t + p * (1 - ev["third_support"]) * (ev["t_median"] - t)
    elif name == "C4":
        sigma = (1 - p) * sigma + p * ev["third_support"]
    elif name == "C5":
        scale = weighted_scale(ev, mu, sigma * (1 + p * ev["mutual"]))
    elif name == "C6":
        sigma = (1 - p) * sigma + p * ev["third_overlap"]
    elif name == "C7":
        role = "edge"
        geometric = np.sqrt(mu[i] * mu[j])
        mismatch = np.abs(mu[i] - mu[j]) / (mu[i] + mu[j] + EPS)
        a = geometric + p * mismatch * (np.maximum(mu[i], mu[j]) - geometric)
        t = ev["distance"] / (a + EPS)
    if role == "node":
        t = ev["distance"] / (np.sqrt(scale[i] * scale[j]) + EPS)
    if name in ["F0", "L1", "C1", "C2", "C5"]:
        G = (
            base["G"]
            if np.array_equal(scale, mu)
            else core.compute_gravity(X, base["M"], base["sigma"], base["reliability"], scale)
        )
        gij = np.asarray([G.get((int(a), int(b)), 0.0) for (a, b) in ev["pairs"]])
        gji = np.asarray([G.get((int(b), int(a)), 0.0) for (a, b) in ev["pairs"]])
    else:
        vals = []
        for k, (a, b) in enumerate(ev["pairs"]):
            alpha = core.adaptive_alpha(
                int(a), int(b), base["reliability"], alpha_min=1.0, alpha_max=2.0
            )
            beta = 1.0 + (1.0 - min(base["reliability"][a], base["reliability"][b]))
            vals.append(
                float(
                    sigma[k] ** beta * math.exp(-(float(t[k]) ** 2)) / (float(t[k]) + EPS) ** alpha
                )
                if sigma[k] > EPS
                else 0.0
            )
        gij = np.asarray(vals)
        gji = gij.copy()
        G = {}
        for k, (a, b) in enumerate(ev["pairs"]):
            G[int(a), int(b)] = G[int(b), int(a)] = float(gij[k])
    assert np.isfinite(scale).all() and (scale >= EPS).all()
    assert np.isfinite(t).all() and (t >= 0).all()
    assert np.isfinite(sigma).all() and (sigma >= 0).all() and (sigma <= 1.0 + 1e-12).all()
    assert np.isfinite(gij).all() and np.isfinite(gji).all()
    return {
        "scale": scale,
        "sigma": sigma,
        "t": t,
        "G": G,
        "G_ij": gij,
        "G_ji": gji,
        "scale_role": role,
        "name": name,
        "parameter": p,
        "mode": mode,
    }


def context(ev, field, n):
    nodes = [[] for _ in range(n)]
    for k, (i, j) in enumerate(ev["pairs"]):
        if field["G_ij"][k] > EPS:
            nodes[i].append(field["G_ij"][k])
        if field["G_ji"][k] > EPS:
            nodes[j].append(field["G_ji"][k])
    row = np.asarray([float(np.percentile(v, 30)) if v else np.inf for v in nodes])
    sd = np.repeat(field["t"][ev["mutual"]], 2)
    return {
        "row_threshold": row,
        "tau95": float(np.percentile(sd, 95)) if len(sd) else 1.8,
        "tau85": float(np.percentile(sd, 85)) if len(sd) else 1.5,
    }


def geometric_rows(h0, hc):
    out = np.full_like(h0, np.inf, dtype=float)
    finite = np.isfinite(h0) & np.isfinite(hc)
    out[finite] = np.sqrt(h0[finite] * hc[finite])
    return out


def relative_distance(t, tau):
    assert np.isfinite(tau) and tau >= 0
    if tau == 0:
        return np.where(t == 0, 0.0, np.inf)
    return t / tau


def distance_passes(t0, tc, tau0, tauc, mode):
    if mode == "tc_vs_tc_quantile":
        return tc <= tauc
    if mode == "tc_vs_t0_quantile":
        return tc <= tau0
    if mode == "tc_vs_mean_quantile":
        return tc <= (tau0 + tauc) / 2.0
    if mode == "t0_vs_t0_quantile":
        return t0 <= tau0
    if mode == "mean_relative_distance_le_one":
        return (relative_distance(t0, tau0) + relative_distance(tc, tauc)) / 2.0 <= 1.0
    if mode == "both_native_distance_predicates":
        return (t0 <= tau0) & (tc <= tauc)
    raise ValueError(mode)


def threshold_context(ev, f0, field, n, backend, f0ctx=None, currentctx=None):
    f0ctx = context(ev, f0, n) if f0ctx is None else f0ctx
    currentctx = context(ev, field, n) if currentctx is None else currentctx
    spec = (
        SPEC[backend] if backend in CANDIDATES else {"row": "c1", "distance": "tc_vs_tc_quantile"}
    )
    (h0, hc) = (f0ctx["row_threshold"], currentctx["row_threshold"])
    row = (
        h0.copy()
        if spec["row"] == "f0"
        else geometric_rows(h0, hc)
        if spec["row"] == "geometric_blend"
        else hc.copy()
    )
    mode = spec["distance"]
    used = {}
    passes = {}
    for key, label in [("tau95", "mutual"), ("tau85", "strong")]:
        (a, b) = (f0ctx[key], currentctx[key])
        passes[label] = distance_passes(f0["t"], field["t"], a, b, mode)
        used[key] = (
            a
            if mode in ["tc_vs_t0_quantile", "t0_vs_t0_quantile"]
            else (a + b) / 2
            if mode == "tc_vs_mean_quantile"
            else b
        )
    return {
        **used,
        "row_threshold": row,
        "normal_row_threshold": hc,
        "distance_mode": mode,
        "row_mode": spec["row"],
        "mutual_distance_pass": passes["mutual"],
        "strong_distance_pass": passes["strong"],
        "f0_tau95": f0ctx["tau95"],
        "f0_tau85": f0ctx["tau85"],
        "current_tau95": currentctx["tau95"],
        "current_tau85": currentctx["tau85"],
        "f0_row_threshold": h0,
        "current_row_threshold": hc,
    }


def build_strong_graph(ev, field, n, ctx):
    (i, j) = (ev["pairs"][:, 0], ev["pairs"][:, 1])
    (gi, gj, h) = (field["G_ij"], field["G_ji"], ctx["row_threshold"])
    mutual_branch = (
        ev["mutual"] & ctx["mutual_distance_pass"] & (np.maximum(gi, gj) >= np.minimum(h[i], h[j]))
    )
    both_branch = (gi >= h[i]) & (gj >= h[j]) & ctx["strong_distance_pass"]
    keep = mutual_branch | both_branch
    graph = [set() for _ in range(n)]
    for a, b in ev["pairs"][keep]:
        graph[a].add(int(b))
        graph[b].add(int(a))
    return (graph, {"keep": keep, "mutual_branch": mutual_branch, "both_branch": both_branch})
