import numpy as np
from .sni_single_native._config import EPS


def threshold(mean, std):
    ratio = np.divide(0.3 * mean, std, out=np.zeros_like(np.asarray(mean, float)), where=std > EPS)
    adaptive = 1.4 + 3.0 / (1.0 + np.exp(np.minimum(ratio, 700.0)))
    return np.where((std > EPS) & (mean > EPS), adaptive, 2.7)
