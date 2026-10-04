"""Fixed benchmark workload for the Ubuntu/VMware resource experiments.

It builds an in-memory request trace of --trace-mb megabytes (int32 cylinders),
then services --windows request windows taken from RANDOM offsets of that trace:
each window is scheduled by all four algorithms and by the learned selector.
The total work is identical in every run (same seeds, same window offsets);
only the VM's RAM / vCPU settings change. Work is split across --workers processes
(default = number of vCPUs visible to the guest).

Random offsets into a large trace give poor locality: when the trace no longer fits
in guest RAM, the kernel must page it to swap -> major page faults & longer runtime.
"""
import argparse, os, time, json
import multiprocessing as mp
import numpy as np
from schedulers import run_all, NAMES, MAX_CYL
import selector as S

ap = argparse.ArgumentParser()
ap.add_argument("--trace-mb", type=int, default=768)
ap.add_argument("--windows", type=int, default=20000)
ap.add_argument("--workers", type=int, default=os.cpu_count())
ap.add_argument("--json", default=None, help="write timing summary here")
a = ap.parse_args()
W = 40
TRACE = None  # filled before fork; children share it copy-on-write
CLF = None


def build_trace(mb):
    n = mb * 1024 * 1024 // 4
    t = np.empty(n, dtype=np.int32)
    rng = np.random.default_rng(123)
    chunk = 1 << 22
    for i in range(0, n, chunk):              # chunked: avoids a 2x temporary
        t[i:i + chunk] = rng.integers(0, MAX_CYL + 1, min(chunk, n - i), dtype=np.int32)
    return t


def work(offsets):
    total = {n: 0 for n in NAMES + ["Selector"]}
    head = MAX_CYL // 2
    for off in offsets:
        reqs = TRACE[off:off + W].tolist()     # touches a random page of the trace
        res = run_all(head, reqs)
        pick, _ = S.predict(CLF, head, reqs)
        for n in NAMES: total[n] += res[n][1]
        total["Selector"] += res[pick][1]
        head = res[pick][0][-1]
    return total


if __name__ == "__main__":
    t0 = time.perf_counter()
    Xtr, ytr, *_ = S.make_dataset(300, seed=1)
    CLF = S.train(Xtr, ytr)
    TRACE = build_trace(a.trace_mb)
    t1 = time.perf_counter()
    offs = np.random.default_rng(99).integers(0, len(TRACE) - W, a.windows)
    parts = np.array_split(offs, a.workers)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        results = pool.map(work, parts)
    t2 = time.perf_counter()
    tot = {n: sum(r[n] for r in results) for n in results[0]}
    out = {"trace_mb": a.trace_mb, "windows": a.windows, "workers": a.workers,
           "build_s": round(t1 - t0, 3), "schedule_s": round(t2 - t1, 3), "total_s": round(t2 - t0, 3),
           "windows_per_s": round(a.windows / (t2 - t1), 1), "seek_totals": tot}
    print(json.dumps(out))
    if a.json: json.dump(out, open(a.json, "w"))
