import hashlib
import json


def fingerprint(value):
    import numpy as np

    def enc(v):
        if isinstance(v, np.ndarray):
            return ["array", v.dtype.str, list(v.shape), hashlib.sha256(v.tobytes()).hexdigest()]
        if isinstance(v, np.generic):
            return ["scalar", v.dtype.str, enc(v.item())]
        if isinstance(v, dict):
            return [
                "dict",
                sorted([[enc(k), enc(w)] for (k, w) in v.items()], key=lambda p: repr(p[0])),
            ]
        if isinstance(v, (list, tuple, set)):
            return [
                type(v).__name__,
                [enc(w) for w in (sorted(v, key=repr) if isinstance(v, set) else v)],
            ]
        if isinstance(v, float):
            return ["float", v.hex()]
        if v is None or isinstance(v, (str, int, bool)):
            return [type(v).__name__, v]
        raise TypeError(type(v).__name__)

    return hashlib.sha256(
        json.dumps(enc(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def qualify(pairs, gij, gji, block_ids, raw_ids, proposal, eps):
    import numpy as np

    n = len(block_ids)
    m = len(pairs)
    eps64 = np.finfo(np.float64).eps
    assert pairs.dtype == np.int64 and pairs.shape == (m, 2) and ((pairs >= 0) & (pairs < n)).all()
    assert (pairs[:, 0] < pairs[:, 1]).all() and len(np.unique(pairs, axis=0)) == m
    assert (
        block_ids.dtype == raw_ids.dtype == np.int64
        and block_ids.shape == raw_ids.shape == (n,)
        and (block_ids >= 0).all()
        and (raw_ids >= 0).all()
    )
    assert (
        gij.dtype == gji.dtype == np.float64
        and gij.shape == gji.shape == (m,)
        and np.isfinite(gij).all()
        and np.isfinite(gji).all()
        and (gij >= 0).all()
        and (gji >= 0).all()
    )
    assert np.array_equal(gij, gji), (
        "This frozen field is symmetric; directed extension is out of scope"
    )
    assert proposal.dtype == bool and proposal.shape == (m,)
    (blocks, bpos) = np.unique(block_ids, return_inverse=True)
    (raws, rpos) = np.unique(raw_ids, return_inverse=True)
    block_raw = np.full(len(blocks), -1, np.int64)
    for k in range(len(blocks)):
        owners = np.unique(rpos[bpos == k])
        assert len(owners) == 1
        block_raw[k] = owners[0]
    edge_raw = rpos[pairs[:, 0]]
    assert np.array_equal(edge_raw, rpos[pairs[:, 1]])
    valid = gij > eps
    positive = np.where(valid, gij, 0.0)
    raw_scale = np.zeros(len(raws), np.float64)
    np.maximum.at(raw_scale, edge_raw, positive)
    weight = np.divide(
        positive, raw_scale[edge_raw], out=np.zeros(m, np.float64), where=raw_scale[edge_raw] > 0
    )
    endpoints = bpos[pairs]
    internal = endpoints[:, 0] == endpoints[:, 1]
    volume = np.zeros(len(blocks), np.float64)
    inner = np.zeros(len(blocks), np.float64)
    total = np.zeros(len(raws), np.float64)
    np.add.at(volume, endpoints[:, 0], weight)
    np.add.at(volume, endpoints[:, 1], weight)
    np.add.at(inner, endpoints[internal, 0], 2 * weight[internal])
    np.add.at(total, edge_raw, 2 * weight)
    conservation = np.bincount(block_raw, weights=volume, minlength=len(raws))
    assert np.allclose(conservation, total, rtol=256 * eps64, atol=256 * eps64)
    (domain_pos, domain_inverse) = np.unique(
        np.sort(endpoints[~internal], axis=1), axis=0, return_inverse=True
    )
    d = len(domain_pos)
    edge_domain = np.full(m, -1, np.int64)
    edge_domain[~internal] = domain_inverse
    cross_force = np.zeros(d, np.float64)
    proposal_count = np.zeros(d, np.int64)
    np.add.at(cross_force, domain_inverse, weight[~internal])
    np.add.at(proposal_count, edge_domain[proposal & ~internal], 1)
    domain_raw = block_raw[domain_pos[:, 0]] if d else np.zeros(0, np.int64)
    assert not d or np.array_equal(domain_raw, block_raw[domain_pos[:, 1]])
    known = (inner[domain_pos] > 0).all(axis=1) & (total[domain_raw] > 0)
    lhs = cross_force * total[domain_raw]
    rhs = volume[domain_pos[:, 0]] * volume[domain_pos[:, 1]]
    enrichment = lhs > rhs
    has_proposal = proposal_count > 0
    unknown = has_proposal & ~known
    endpoint_maps = [[{}, {}] for _ in range(d)]
    counts = np.zeros((d, 2), np.int64)
    neff = np.zeros((d, 2), np.float64)
    for e in np.flatnonzero(proposal & ~internal):
        domain = int(edge_domain[e])
        (a, b) = map(int, pairs[e])
        (left, right) = (a, b) if bpos[a] == domain_pos[domain, 0] else (b, a)
        for side, node in enumerate((left, right)):
            endpoint_maps[domain][side][node] = endpoint_maps[domain][side].get(node, 0.0) + float(
                weight[e]
            )
    positive_counts = np.zeros((d, 2), np.int64)
    for domain in range(d):
        for side in range(2):
            vals = np.asarray(
                [w for (_, w) in sorted(endpoint_maps[domain][side].items())], np.float64
            )
            counts[domain, side] = len(vals)
            vals = vals[vals > 0]
            positive_counts[domain, side] = len(vals)
            if len(vals):
                value = float(vals.sum() ** 2 / np.square(vals).sum())
                neff[domain, side] = min(value, float(len(vals)))
    neff_pass = (neff + 64 * eps64 * np.maximum(1.0, neff) >= 2).all(axis=1)
    enrich_cert = has_proposal & known & enrichment
    neff_cert = enrich_cert & neff_pass
    enrich_domain = unknown | enrich_cert
    neff_domain = unknown | neff_cert
    accepted_enrich = proposal.copy()
    accepted_neff = proposal.copy()
    accepted_enrich[~internal] &= enrich_domain[domain_inverse]
    accepted_neff[~internal] &= neff_domain[domain_inverse]
    assert not (accepted_neff & ~accepted_enrich).any() and (
        not (accepted_enrich & ~proposal).any()
    )
    unknown_edge = np.zeros(m, bool)
    cert_edge = np.zeros(m, bool)
    cert_neff_edge = np.zeros(m, bool)
    unknown_edge[~internal] = proposal[~internal] & unknown[domain_inverse]
    cert_edge[~internal] = proposal[~internal] & enrich_cert[domain_inverse]
    cert_neff_edge[~internal] = proposal[~internal] & neff_cert[domain_inverse]
    assert np.array_equal(accepted_enrich[unknown_edge], proposal[unknown_edge]) and np.array_equal(
        accepted_neff[unknown_edge], proposal[unknown_edge]
    )
    return dict(
        schema="175_FIXED_BLOCK_SYMMETRIC_FORCE_V1",
        pairs=pairs.copy(),
        G_ij=gij.copy(),
        G_ji=gji.copy(),
        EPS=np.asarray(eps, np.float64),
        block_ids=block_ids.copy(),
        raw_ids=raw_ids.copy(),
        block_keys=blocks,
        raw_keys=raws,
        node_block_position=bpos.astype(np.int64),
        block_raw_position=block_raw,
        edge_raw_position=edge_raw,
        valid_positive_force=valid,
        raw_force_scale=raw_scale,
        normalized_weight=weight,
        internal_force=inner,
        block_volume=volume,
        raw_total_force=total,
        conservation_by_raw=conservation,
        domain_keys=blocks[domain_pos],
        domain_block_positions=domain_pos,
        domain_raw_position=domain_raw,
        edge_domain=edge_domain,
        domain_cross_force=cross_force,
        domain_source_U_edges=proposal_count,
        domain_known=known,
        domain_has_proposal=has_proposal,
        domain_unknown_fallback=unknown,
        enrichment_lhs=lhs,
        enrichment_rhs=rhs,
        domain_enriched=enrichment,
        endpoint_counts=counts,
        positive_endpoint_counts=positive_counts,
        effective_endpoint_counts=neff,
        domain_neff_pass=neff_pass,
        proposed_U=proposal.copy(),
        accepted_enrich=accepted_enrich,
        accepted_neff=accepted_neff,
        unknown_edge=unknown_edge,
        certified_enrich_edge=cert_edge,
        certified_neff_edge=cert_neff_edge,
        same_block_U=proposal & internal,
        force_symmetric=True,
        effective_direction_tests=1,
        neff_comparison="neff+64*float64_epsilon*max(1,neff)>=2",
    )


def components(n, pairs, mask):
    import numpy as np

    parent = np.arange(n, dtype=np.int64)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = int(parent[k])
        return k

    for a, b in pairs[mask]:
        (ra, rb) = (find(int(a)), find(int(b)))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.asarray([find(i) for i in range(n)], np.int64)
