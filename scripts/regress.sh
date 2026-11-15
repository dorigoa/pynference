#!/usr/bin/env bash
# C10 regression: rerun the S6 checks and compare against the golden baseline.
# Run only inside a GPU job (2.4). Prints PASS or FAIL, exits 0 or 1.
#   usage: scripts/regress.sh gemma4
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

case "${1:-}" in
  gemma4) MODEL=/nvme/gguf/gemma-4-31B/gemma-4-31B-it-Q8_0.gguf; G=tests/golden/gemma4 ;;
  *) echo "usage: scripts/regress.sh gemma4"; exit 2 ;;
esac
if [ ! -f "$MODEL" ] || [ ! -f "$G/ours_baseline.json" ] || [ ! -f "$G/ours_ppl.txt" ]; then
  echo "FAIL: missing model or golden files"; exit 1
fi
OUT=$(mktemp -d /tmp/pynference-regress-XXXX)
trap 'rm -rf "$OUT"' EXIT

echo "== regress $1: ours --ref (graph) $(date +%T)"
timeout 900 torchrun --nproc_per_node=2 bench/compare_probs.py ours --ref "$G/ref_probs.json" -m "$MODEL" \
    --decode graph -o "$OUT/ours_now.json" 2>&1 | grep -v -E "CCL_WARN|^W1" | tail -n 6
echo "== ours exit ${PIPESTATUS[0]} ($(date +%T))"

echo "== regress $1: ppl $(date +%T)"
timeout 900 torchrun --nproc_per_node=2 bench/ppl.py -m "$MODEL" -c 512 --chunks 8 > "$OUT/ppl.log" 2>&1
echo "== ppl exit $? ($(date +%T))"
grep -E "Final estimate" "$OUT/ppl.log" | tail -n 1

python - "$G/ours_baseline.json" "$OUT/ours_now.json" "$G/ours_ppl.txt" "$OUT/ppl.log" <<'PYEOF'
import json, re, sys

base_path, now_path, ppl_base_path, ppl_now_path = sys.argv[1:5]
problems = []

base = json.load(open(base_path))
now = json.load(open(now_path))
if len(base) != len(now):
    problems.append(f"prompt count: {len(base)} baseline, {len(now)} now")
for b, n in zip(base, now):
    if b["name"] != n["name"]:
        problems.append(f"prompt order: {b['name']} vs {n['name']}")
        continue
    if b["prompt"] != n["prompt"]:
        problems.append(f"{b['name']}: prompt ids differ")
        continue
    if len(b["steps"]) != len(n["steps"]):
        problems.append(f"{b['name']}: step count {len(b['steps'])} vs {len(n['steps'])}")
    max_dp = 0.0
    for i, (bs, ns) in enumerate(zip(b["steps"], n["steps"])):
        if bs["id"] != ns["id"]:
            problems.append(f"{b['name']}: greedy token differs at step {i}")
            break
        pb = {t["id"]: t["p"] for t in bs["top"]}
        pn = {t["id"]: t["p"] for t in ns["top"]}
        if pb.keys() != pn.keys():
            problems.append(f"{b['name']}: top-10 ids differ at step {i}")
            break
        max_dp = max([max_dp] + [abs(pb[k] - pn[k]) for k in pb])
    print(f"{b['name']}: max |dp| {max_dp:.2e} over top-10")
    if max_dp > 1e-3:
        problems.append(f"{b['name']}: max |dp| {max_dp:.2e} > 1e-3")

pp = re.compile(r"PPL = ([0-9.]+)")
b_ppl = float(pp.search(open(ppl_base_path).read()).group(1))
n_ppl = float(pp.search(open(ppl_now_path).read()).group(1))
rel = abs(n_ppl - b_ppl) / b_ppl
print(f"ppl: baseline {b_ppl:.4f}, now {n_ppl:.4f}, rel diff {rel * 100:.3f}%")
if rel > 1e-3:
    problems.append(f"ppl rel diff {rel * 100:.3f}% > 0.1%")

if problems:
    for p in problems:
        print("FAIL:", p)
    sys.exit(1)
print("PASS")
PYEOF
rc=$?
[ $rc -eq 0 ] && echo "PASS" || echo "FAIL"
exit $rc
