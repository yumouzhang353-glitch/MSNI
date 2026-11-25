from contextlib import contextmanager
import numpy as np
from . import assignment_candidates as ac
from . import fit_direction as prior


def free_component(start, old, seeds, neighbors):
    inside = (seeds < 0) & (old == old[start])
    assert inside[start]
    ind = neighbors["indices"]
    ptr = neighbors["indptr"]
    hosts = np.repeat(np.arange(len(seeds)), np.diff(ptr))
    active = (neighbors["field"] > 0) & inside[hosts] & inside[ind]
    comp = prior.components(len(seeds), np.column_stack((hosts[active], ind[active])))
    return inside & (comp == comp[start])


@contextmanager
def protect(qlog, events):
    decide = prior.apply_decision
    (allocate, identify) = (ac.allocate, ac.identify)
    state = {"active": None, "review": None}

    def observe(X, active, ev, ref, target, bound, trace):
        out = decide(X, active, ev, ref, target, bound, trace)
        state["active"] = active
        return out

    def vote(i, known, fixed, nb, kind):
        record = identify(i, known, fixed, nb, kind)
        review = state["review"]
        if (
            review is None
            or not review["scope"][i]
            or record["reason"] != 1
            or (record["winner"] == review["joined"])
        ):
            return record
        sl = slice(nb["indptr"][i], nb["indptr"][i + 1])
        if np.any((known[nb["indices"][sl]] == review["joined"]) & (nb["field"][sl] > 0)):
            return record
        review["blocked"].add(int(i))
        return dict(
            record,
            reason=5,
            winner=-1,
            confidence=0.0,
            runner_up=0.0,
            winning_count=0,
            original_contributors=0,
            propagated_contributors=0,
        )

    def run(seeds, old, nb, mode):
        native = allocate(seeds, old, nb, mode)
        if (
            mode != "A8_NN_FIELD_PROP"
            or not qlog["trigger"]
            or (not qlog["calls"])
            or (not qlog["calls"][-1]["allowed"])
        ):
            return native
        active = state["active"]
        reps = np.asarray(
            [int(np.flatnonzero(active == j).item()) for j in qlog["representatives"]]
        )
        (z0, t0) = native
        moving = seeds[reps] < 0
        record = dict(
            representatives=active[reps].tolist(),
            fixed=(~moving).tolist(),
            before=old[reps].tolist(),
            native=z0[reps].tolist(),
            applied=False,
        )
        events.append(record)
        if sum(moving) != 1 or old[reps[0]] != old[reps[1]] or z0[reps[0]] == z0[reps[1]]:
            return native
        start = int(reps[moving][0])
        joined = int(old[start])
        scope = free_component(start, old, seeds, nb)
        review = dict(scope=scope, joined=joined, blocked=set())
        assert state["review"] is None and z0[reps[~moving][0]] == joined
        state["review"] = review
        try:
            (clean, trace) = allocate(seeds, old, nb, mode)
        finally:
            state["review"] = None
        assert np.array_equal(clean[seeds >= 0], seeds[seeds >= 0])
        hold = trace["reason"] == 5
        assert (
            np.all(clean[hold] == old[hold])
            and np.all(trace["order"][hold] == -1)
            and np.all(scope[hold])
        )
        record.update(
            applied=True,
            scope=int(scope.sum()),
            vetoed_hosts=len(review["blocked"]),
            held=int(hold.sum()),
            after=clean[reps].tolist(),
            changed=int(np.sum(clean != z0)),
            changed_outside=int(np.sum((clean != z0) & ~scope)),
        )
        return (clean, trace)

    try:
        prior.apply_decision = observe
        (ac.allocate, ac.identify) = (run, vote)
        yield
    finally:
        prior.apply_decision = decide
        (ac.allocate, ac.identify) = (allocate, identify)
