"""Textbook check (Silberschatz, Operating System Concepts, disk-scheduling example):
queue 98,183,37,122,14,124,65,67, head 53, cylinders 0-199."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schedulers import fcfs, sstf, scan, cscan
Q, H = [98, 183, 37, 122, 14, 124, 65, 67], 53
assert fcfs(H, Q)[1] == 640
assert sstf(H, Q)[1] == 236
assert scan(H, Q, "down", max_cyl=199)[1] == 236          # toward 0 first
assert scan(H, Q, "up", max_cyl=199)[1] == 146 + 185      # 53->199->14
assert cscan(H, Q, max_cyl=199)[1] == 146 + 199 + 37      # 382, return jump counted
print("all textbook checks passed: FCFS 640, SSTF 236, SCAN 236, C-SCAN 382")
