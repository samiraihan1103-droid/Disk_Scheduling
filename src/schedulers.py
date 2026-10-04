"""Classical disk-scheduling algorithms (Silberschatz et al., Ch. 11).

Every function takes the current head position and a list of requested
cylinders (in arrival order) and returns (service_order, total_head_movement).
Total head movement in cylinders is used as the seek-time metric.
"""
MAX_CYL = 4999  # cylinders 0..4999


def fcfs(head, reqs):
    order, total, pos = list(reqs), 0, head
    for r in order:
        total += abs(r - pos)
        pos = r
    return order, total


def sstf(head, reqs):
    pending, order, total, pos = list(reqs), [], 0, head
    while pending:
        i = min(range(len(pending)), key=lambda k: abs(pending[k] - pos))
        r = pending.pop(i)
        total += abs(r - pos)
        pos = r
        order.append(r)
    return order, total


def scan(head, reqs, direction="up", max_cyl=MAX_CYL):
    """Elevator: sweep to the end of the disk, then reverse (arm stops early
    only if nothing is waiting behind it)."""
    up = sorted(r for r in reqs if r >= head)
    down = sorted((r for r in reqs if r < head), reverse=True)
    if direction == "up":
        order = up + down
        total = (max_cyl - head) + (max_cyl - down[-1]) if down else (up[-1] - head if up else 0)
    else:
        order = down + up
        total = head + up[-1] if up else (head - down[-1] if down else 0)
    return order, total


def cscan(head, reqs, max_cyl=MAX_CYL):
    """Circular SCAN: sweep up to the end, jump to cylinder 0, sweep up again.
    The return jump (MAX_CYL cylinders) is counted, as in the textbook example."""
    up = sorted(r for r in reqs if r >= head)
    low = sorted(r for r in reqs if r < head)
    order = up + low
    if low:
        total = (max_cyl - head) + max_cyl + low[-1]
    else:
        total = up[-1] - head if up else 0
    return order, total


SCHEDULERS = {"FCFS": fcfs, "SSTF": sstf, "SCAN": scan, "C-SCAN": cscan}
NAMES = list(SCHEDULERS)


def run_all(head, reqs):
    """Return {name: (order, total)} for all four schedulers."""
    return {n: f(head, reqs) for n, f in SCHEDULERS.items()}
