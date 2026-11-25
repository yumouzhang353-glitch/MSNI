import traceback
import numpy as np


class InfeasibleSupport(Exception):
    def __init__(self, stage, use, cap):
        bad = np.flatnonzero(use > cap)
        self.record = {
            "stage": stage,
            "raw": bad.tolist(),
            "used": use[bad].tolist(),
            "cap": cap[bad].tolist(),
        }


def native_quota_error(error):
    from . import support_candidates81

    frame = list(traceback.walk_tb(error.__traceback__))[-1][0]
    if frame.f_code is not support_candidates81.rebuild_baseline.__code__:
        return None
    d = frame.f_locals
    if not {"uses", "caps", "old"} <= d.keys():
        return None
    (caps, old, uses) = (d["caps"], d["old"]["reference_caps"], d["uses"])
    if np.any(uses > caps) and np.all(caps >= old) and np.all(caps <= old + 1):
        return InfeasibleSupport("S99_rebuild", uses, caps)
    return None
