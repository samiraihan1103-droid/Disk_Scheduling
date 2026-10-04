"""Summarise results/vm_runs/runs.csv -> results/vm_summary.csv, results/tab_vm.tex,
results/figures/fig_vm.png.  Run after all VM configurations are done."""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
df = pd.read_csv(os.path.join(R, "vm_runs", "runs.csv"))
order = list(dict.fromkeys(df.label))            # keep the order you ran them in
g = df.groupby("label", sort=False).agg(
    vCPU=("vcpus", "first"), RAM_MB=("mem_total_mb", "first"),
    time_s=("elapsed_s", "mean"), time_sd=("elapsed_s", "std"),
    win_per_s=("windows_per_s", "mean"), RSS_MB=("max_rss_mb", "mean"),
    major_flt=("major_faults", "mean"), swap_in=("pswpin", "mean"), swap_out=("pswpout", "mean"),
    inv_cs=("invol_cs", "mean")).loc[order]
g = g.fillna(0).round(2)
g.to_csv(os.path.join(R, "vm_summary.csv"))
with open(os.path.join(R, "tab_vm.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\\toprule\n")
    f.write("Config & vCPU & RAM (MB) & Time (s) & SD & Major flt & Swap-in (pg) & Invol.\\ CS\\\\\\midrule\n")
    for i, x in g.iterrows():
        f.write(f"{i} & {x.vCPU:.0f} & {x.RAM_MB:,.0f} & {x.time_s:.2f} & {x.time_sd:.2f} & "
                f"{x.major_flt:,.0f} & {x.swap_in:,.0f} & {x.inv_cs:,.0f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
a1.bar(g.index, g.time_s, yerr=g.time_sd, capsize=3); a1.set_ylabel("wall time (s)")
a2.bar(g.index, g.major_flt, color="tab:red"); a2.set_ylabel("major page faults")
for ax in (a1, a2): ax.tick_params(axis="x", rotation=30, labelsize=7)
plt.tight_layout(); os.makedirs(os.path.join(R, "figures"), exist_ok=True)
plt.savefig(os.path.join(R, "figures", "fig_vm.png"), dpi=200)
print(g.to_string())
