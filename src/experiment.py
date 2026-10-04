"""Main experiment: train/test the selector, then run the workload-shift timeline.
Deterministic (fixed seeds) so every run - on any machine/VM - uses the SAME workload.
Usage: python src/experiment.py [--out results] [--no-plots]"""
import argparse, json, os, csv
import numpy as np, pandas as pd
from schedulers import run_all, NAMES, MAX_CYL
from workload import GENERATORS
import selector as S

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
ap.add_argument("--no-plots", action="store_true")
a = ap.parse_args()
OUT = os.path.abspath(a.out); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

# ---------- 1. Train / test on mixed workloads ----------
Xtr, ytr, _, _, _ = S.make_dataset(1000, seed=1)
Xte, yte, wte, hte, rte = S.make_dataset(500, seed=2)
clf = S.train(Xtr, ytr)
open(os.path.join(OUT, "decision_tree.txt"), "w").write(S.describe(clf))

rows = []
for h, r, wl, lab in zip(hte, rte, wte, yte):
    res = run_all(h, r)
    pick, conf = S.predict(clf, h, r)
    row = {"workload": wl, "best": lab, "pick": pick, "conf": conf, "correct": pick == lab}
    row.update({n: res[n][1] for n in NAMES})
    row["Selector"] = res[pick][1]; row["Oracle"] = res[lab][1]
    rows.append(row)
te = pd.DataFrame(rows); te.to_csv(os.path.join(OUT, "test_windows.csv"), index=False)

cols = NAMES + ["Selector", "Oracle"]
by_wl = te.groupby("workload")[cols].mean().round(1)
by_wl.loc["all"] = te[cols].mean().round(1)
by_wl.to_csv(os.path.join(OUT, "seek_by_workload.csv"))
acc = te.groupby("workload")["correct"].mean().round(3).to_dict(); acc["all"] = round(te.correct.mean(), 3)

# calibration
bins = [0, .5, .6, .7, .8, .9, 1.0001]
te["bin"] = pd.cut(te.conf, bins, right=False)
cal = te.groupby("bin", observed=True).agg(n=("correct", "size"), mean_conf=("conf", "mean"), accuracy=("correct", "mean")).reset_index()
cal["bin"] = cal["bin"].astype(str); cal.round(3).to_csv(os.path.join(OUT, "calibration.csv"), index=False)
ece = float((cal.n * (cal.mean_conf - cal.accuracy).abs()).sum() / cal.n.sum())
conf_ok = te[te.correct].conf.mean(); conf_bad = te[~te.correct].conf.mean() if (~te.correct).any() else float("nan")

# ---------- 2. Workload-shift timeline: sequential -> random -> bursty ----------
PH = [("sequential", 40), ("random", 40), ("bursty", 40)]
rng = np.random.default_rng(7)
windows, phase = [], []
stream = MAX_CYL // 2
for t, n in PH:
    for _ in range(n):
        r = GENERATORS[t](rng, stream); stream = int(r[-1])
        windows.append(r); phase.append(t)

def simulate(policy, model=None):
    """policy: scheduler name, 'Selector' or 'Oracle'. Each policy keeps its own head."""
    head, out = MAX_CYL // 2, []
    for i, r in enumerate(windows):
        res = run_all(head, r)
        best = min(NAMES, key=lambda n: res[n][1])
        conf = np.nan
        if policy == "Selector": pick, conf = S.predict(model, head, r)
        elif policy == "Oracle": pick = best
        else: pick = policy
        order, cost = res[pick]
        out.append({"window": i, "phase": phase[i], "policy": policy, "pick": pick, "best": best,
                    "conf": conf, "correct": pick == best, "seek": cost})
        head = order[-1]
    return out

# shift-naive model: never saw bursty traffic during training
Xn, yn, _, _, _ = S.make_dataset(1500, seed=1, types=["sequential", "random"])
naive = S.train(Xn, yn)

tl = []
for p in NAMES + ["Oracle"]: tl += simulate(p)
tl += simulate("Selector", clf)
for d in simulate("Selector", naive): d["policy"] = "Selector-naive"; tl.append(d)
tl = pd.DataFrame(tl); tl.to_csv(os.path.join(OUT, "shift_timeline.csv"), index=False)

pol = NAMES + ["Selector", "Selector-naive", "Oracle"]
phase_sum = tl.pivot_table(index="policy", columns="phase", values="seek", aggfunc="sum")[[t for t, _ in PH]]
phase_sum["total"] = phase_sum.sum(axis=1); phase_sum = phase_sum.loc[pol]
phase_sum.to_csv(os.path.join(OUT, "shift_phase_seek.csv"))
sel = tl[tl.policy.str.startswith("Selector")]
phase_acc = sel.pivot_table(index="policy", columns="phase", values="correct", aggfunc="mean")[[t for t, _ in PH]].round(3)
phase_conf = sel.pivot_table(index="policy", columns="phase", values="conf", aggfunc="mean")[[t for t, _ in PH]].round(3)
phase_acc.to_csv(os.path.join(OUT, "shift_selector_accuracy.csv")); phase_conf.to_csv(os.path.join(OUT, "shift_selector_confidence.csv"))

summary = {"test_accuracy": acc, "ECE": round(ece, 4), "mean_conf_correct": round(conf_ok, 3),
           "mean_conf_wrong": round(conf_bad, 3), "label_distribution_train": pd.Series(ytr).value_counts().to_dict(),
           "selector_vs_best_classical_total": {
               "selector": int(phase_sum.loc["Selector", "total"]),
               "best_classical": phase_sum.loc[NAMES, "total"].idxmin(),
               "best_classical_total": int(phase_sum.loc[NAMES, "total"].min()),
               "oracle": int(phase_sum.loc["Oracle", "total"])}}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)

# ---------- 3. LaTeX tables (auto-included by report/report.tex) ----------
def tex_table(df, path, fmt="{:,.0f}"):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(df.columns) + "}\\toprule\n")
        f.write(" & ".join([""] + [str(c) for c in df.columns]) + "\\\\\\midrule\n")
        for i, row in df.iterrows():
            f.write(" & ".join([str(i)] + [fmt.format(v) for v in row]) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
tex_table(by_wl, os.path.join(OUT, "tab_seek_by_workload.tex"))
tex_table(phase_sum, os.path.join(OUT, "tab_shift_phase.tex"))
pa = phase_acc.copy(); pa.columns = [c + " acc" for c in pa.columns]
pc = phase_conf.copy(); pc.columns = [c + " conf" for c in pc.columns]
tex_table(pd.concat([pa, pc], axis=1), os.path.join(OUT, "tab_shift_selector.tex"), "{:.2f}")
with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write(f"\\newcommand{{\\TestAcc}}{{{acc['all']*100:.1f}\\%}}\n")
    f.write(f"\\newcommand{{\\ECE}}{{{ece:.3f}}}\n")
    f.write(f"\\newcommand{{\\ConfOK}}{{{conf_ok:.2f}}}\n\\newcommand{{\\ConfBad}}{{{conf_bad:.2f}}}\n")

# ---------- 4. Figures ----------
if not a.no_plots:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    ax = by_wl.drop("all").T.plot.bar(figsize=(6, 3)); ax.set_ylabel("mean seek / window (cyl)")
    ax.set_title("Seek distance by workload type (test set)"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_seek_by_workload.png"), dpi=200); plt.close()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for p in pol:
        s = tl[tl.policy == p].seek.cumsum().values
        a1.plot(s, label=p, lw=1.8 if p.startswith("Selector") else 1, ls="--" if p == "Oracle" else "-")
    for x in (40, 80): a1.axvline(x, c="k", lw=.6); a2.axvline(x, c="k", lw=.6)
    a1.set_ylabel("cumulative seek (cyl, log)"); a1.set_yscale("log"); a1.legend(fontsize=7, ncol=2)
    a1.set_title("Workload shift: sequential | random | bursty")
    for p, m in (("Selector", "o"), ("Selector-naive", "x")):
        d = tl[tl.policy == p]
        a2.scatter(d.window, d.conf, s=8, marker=m, c=np.where(d.correct, "tab:green", "tab:red"), label=p)
    a2.set_ylabel("confidence"); a2.set_xlabel("window index"); a2.set_ylim(0, 1.05); a2.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_shift_timeline.png"), dpi=200); plt.close()

    plt.figure(figsize=(3.2, 3)); plt.plot([0, 1], [0, 1], "k:", lw=.8)
    plt.plot(cal.mean_conf, cal.accuracy, "o-"); plt.xlabel("mean confidence"); plt.ylabel("accuracy")
    plt.title(f"Reliability (ECE={ece:.3f})"); plt.xlim(.3, 1.02); plt.ylim(.3, 1.02); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_calibration.png"), dpi=200); plt.close()

print(json.dumps(summary, indent=2, default=str))
print(phase_sum); print(phase_acc); print(phase_conf)
