"""Cheap features computed from one window of requests (arrival order)."""
import numpy as np
from schedulers import MAX_CYL

FEATURES = ["std", "mean_jump", "frac_increasing", "span", "head_offset", "n_clusters"]


def extract(head, reqs):
    r = np.asarray(reqs, dtype=float)
    d = np.diff(r)
    s = np.sort(r)
    gaps = np.diff(s)
    return [
        r.std() / MAX_CYL,                      # spread of track positions
        np.abs(d).mean() / MAX_CYL,             # burstiness: avg jump between consecutive arrivals
        (d > 0).mean(),                         # monotonicity (sequential streams ~1)
        (s[-1] - s[0]) / MAX_CYL,               # span covered
        (r.mean() - head) / MAX_CYL,            # where requests sit relative to the head
        1 + int((gaps > 300).sum()),            # number of separated clusters
    ]
