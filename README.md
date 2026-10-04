# Learned Scheduler Selection for Disk Scheduling under Workload Shift
**CSE-307 Operating Systems — Term Paper, Section B, Track 2**
**Name:** Lt Sami
**Student ID:** 202314210

FCFS, SSTF, SCAN and C-SCAN, plus a depth-4 decision tree that looks at a window of
40 requests and predicts which scheduler gives the least head movement, with a confidence
score. Tested on sequential / random / bursty workloads, a timeline that shifts between them,
inside a controlled Python simulation environment.

## Setup
```bash
sudo apt install python3-pip time            # /usr/bin/time is needed for the VM runs
pip install -r requirements.txt               # numpy, pandas, scikit-learn, matplotlib
```

## Run
```bash
python3 tests/test_schedulers.py              # textbook check: FCFS 640, SSTF 236, SCAN 236, C-SCAN 382
python3 src/experiment.py                     # E1 + E2 (~5 s, deterministic) -> results/
```

### Report
```bash
cd report && latexmk -pdf report.tex          # tables/figures are pulled from results/ automatically
```
The report and results are self-contained; no personal placeholders remain.

## Layout
| Path | Contents |
|---|---|
| `src/schedulers.py` | FCFS, SSTF, SCAN, C-SCAN (head movement in cylinders, 0–4999) |
| `src/workload.py` | sequential / random / bursty window generators |
| `src/features.py` | 6 window features (std, mean jump, monotonicity, span, head offset, clusters) |
| `src/selector.py` | oracle labelling, decision tree, prediction + confidence |
| `src/experiment.py` | E1 (static test), E2 (shift timeline, incl. shift-naive model), calibration, tables, figures |
| `results/` | raw CSVs, summary JSON, LaTeX tables, `figures/` |
| `report/` | `report.tex`, `references.bib`, `report.pdf` |

## Results summary (E1, E2)
- SSTF is the best fixed policy overall; FCFS wins on sequential windows; C-SCAN never wins on seek distance.
- Selector: 90.9% accuracy on held-out windows, 4.2% less seek than SSTF, within 1.3% of the oracle.
- Calibration: ECE 0.011; mean confidence 0.95 when right vs 0.67 when wrong.
- Shift: FCFS degrades most (≈13.6× SSTF in the random phase). The full selector switches FCFS→SSTF at
  the first random window. A selector never trained on bursty traffic stays ~98% confident but is right on
  only 5% of bursty windows (3× oracle seek) — confidence is only trustworthy in distribution.

## AI assistance disclosure
Claude (Anthropic) was used to help write the code scaffolding, scripts and the LaTeX template.
The choice of VM configurations, the runs, the measured results and the final analysis are my own.
