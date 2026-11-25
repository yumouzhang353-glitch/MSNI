from contextlib import contextmanager
import numpy as np
from . import single_c5
from . import pipeline91_v3 as previous
from . import fit_direction as prior


@contextmanager
def continuation(log):
    merge = single_c5.merge
    assert previous.merge is merge
    decide = prior.apply_decision
    state = {"pending": None, "live": None}

    def observe(X, active, ev, ref, target, bound, trace):
        out = decide(X, active, ev, ref, target, bound, trace)
        state["live"] = None
        if trace["decision"] == "within" and trace["link"]:
            state["pending"] = tuple(sorted(trace["representatives"]))
            assert sorted(active[trace["link"]]) == list(state["pending"])
        elif (
            trace["decision"] == "fallback_native_quota" and trace["attempted_decision"] == "within"
        ):
            assert state["pending"] is not None
            state["live"] = state["pending"]
            log["trigger"] = True
            log["representatives"] = list(state["live"])
        return out

    def group_merge(base, edges, allowed, counts, unique):
        if state["live"] is None:
            return merge(base, edges, allowed, counts, unique)
        active = base["active"]
        indices = [int(np.flatnonzero(active == i).item()) for i in state["live"]]
        labels = base["active_labels"][indices]
        funded = np.unique(base["seeds"][base["seeds"] >= 0])
        eligible = len(np.unique(labels)) == 2 and np.isin(labels, funded).all()
        if eligible:
            edges = np.vstack((edges, np.asarray(indices, dtype=edges.dtype)))
            allowed = np.r_[allowed, True]
        out = merge(base, edges, allowed, counts, unique)
        assert np.array_equal(base["seeds"] >= 0, out["seeds"] >= 0)
        assert np.array_equal(base["reference_caps"], out["reference_caps"])
        assert np.all(out["used_caps"] <= base["used_caps"]) and np.all(
            out["used_caps"] <= out["reference_caps"]
        )
        if eligible:
            assert len(np.unique(out["active_labels"][indices])) == 1
        log["calls"].append(
            dict(
                before=labels.tolist(),
                funded=bool(np.isin(labels, funded).all()),
                allowed=bool(eligible),
                before_k=len(np.unique(base["labels"])),
                after_k=len(np.unique(out["labels"])),
            )
        )
        return out

    try:
        prior.apply_decision = observe
        single_c5.merge = previous.merge = group_merge
        yield
    finally:
        prior.apply_decision = decide
        single_c5.merge = previous.merge = merge
