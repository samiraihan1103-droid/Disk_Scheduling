"""Synthetic request-window generators. One window = W requests in arrival order."""
import numpy as np
from schedulers import MAX_CYL

W = 40  # requests per window


def sequential(rng, head):
    """Mostly increasing cylinders near the head (e.g. a large file read)."""
    start = int(np.clip(head + rng.integers(-50, 50), 0, MAX_CYL))
    step = rng.integers(5, 40)
    x = start + np.arange(W) * step + rng.integers(-3, 4, W)
    return list(np.clip(x, 0, MAX_CYL).astype(int))


def random_(rng, head):
    """Uniformly random cylinders (many independent processes)."""
    return list(rng.integers(0, MAX_CYL + 1, W))


def bursty(rng, head):
    """Requests arrive in bursts around 2-3 hotspots, interleaved in time."""
    k = rng.integers(2, 4)
    spots = rng.integers(0, MAX_CYL + 1, k)
    seq, s = [], rng.integers(k)
    while len(seq) < W:
        burst = rng.integers(3, 9)
        seq += list(spots[s] + rng.normal(0, 60, burst).astype(int))
        s = rng.integers(k)  # jump to a (possibly different) hotspot
    return list(np.clip(seq[:W], 0, MAX_CYL).astype(int))


GENERATORS = {"sequential": sequential, "random": random_, "bursty": bursty}
