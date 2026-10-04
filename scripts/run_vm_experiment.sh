#!/usr/bin/env bash
# Run the SAME benchmark under the current VM configuration and log OS measurements.
# Usage:  ./scripts/run_vm_experiment.sh <config-label> [repeats]
#   e.g.  ./scripts/run_vm_experiment.sh 2vCPU-2GB 3
# Between configurations: power off the VM, change RAM/vCPU in VMware
# (VM > Settings > Memory / Processors), boot, and run again with a new label.
# Every run appends one row to results/vm_runs/runs.csv.
set -euo pipefail
LABEL=${1:?config label required, e.g. 2vCPU-2GB}
REPS=${2:-3}
TRACE_MB=${TRACE_MB:-768}       # keep identical across ALL configs
WINDOWS=${WINDOWS:-20000}       # keep identical across ALL configs
cd "$(dirname "$0")/.."
OUT=results/vm_runs; mkdir -p "$OUT"
CSV=$OUT/runs.csv
[ -f "$CSV" ] || echo "label,rep,vcpus,mem_total_mb,swap_total_mb,trace_mb,windows,elapsed_s,build_s,schedule_s,windows_per_s,max_rss_mb,major_faults,minor_faults,vol_cs,invol_cs,pswpin,pswpout,cpu_pct" > "$CSV"

VCPU=$(nproc)
MEM=$(awk '/MemTotal/{printf "%d",$2/1024}' /proc/meminfo)
SWP=$(awk '/SwapTotal/{printf "%d",$2/1024}' /proc/meminfo)
echo "Config $LABEL: $VCPU vCPU, ${MEM} MB RAM, ${SWP} MB swap"

vm(){ awk -v k="$1" '$1==k{print $2}' /proc/vmstat; }
for r in $(seq 1 "$REPS"); do
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true   # cold page cache if allowed
  IN0=$(vm pswpin); OUT0=$(vm pswpout)
  T="$OUT/time_${LABEL}_$r.txt"; J="$OUT/bench_${LABEL}_$r.json"
  /usr/bin/time -v -o "$T" python3 src/bench.py --trace-mb "$TRACE_MB" --windows "$WINDOWS" --json "$J" > /dev/null
  IN1=$(vm pswpin); OUT1=$(vm pswpout)
  g(){ grep "$1" "$T" | awk -F': ' '{print $2}'; }
  EL=$(g "Elapsed (wall clock)" | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }')
  RSS=$(( $(g "Maximum resident set size") / 1024 ))
  read BS SS WPS <<< "$(python3 -c "import json;d=json.load(open('$J'));print(d['build_s'],d['schedule_s'],d['windows_per_s'])")"
  echo "$LABEL,$r,$VCPU,$MEM,$SWP,$TRACE_MB,$WINDOWS,$EL,$BS,$SS,$WPS,$RSS,$(g 'Major'),$(g 'Minor'),$(g 'Voluntary context'),$(g 'Involuntary context'),$((IN1-IN0)),$((OUT1-OUT0)),$(g 'Percent of CPU' | tr -d '%')" >> "$CSV"
  echo "  rep $r: ${EL}s | major faults $(g 'Major') | swap-in $((IN1-IN0)) pages | swap-out $((OUT1-OUT0)) pages"
done
