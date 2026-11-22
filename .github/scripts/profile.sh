#!/bin/bash
# Profile an unstripped native build on the runner with perf.
set -eux
sudo apt-get update -qq
sudo apt-get install -y -qq ninja-build libjemalloc2 "linux-tools-$(uname -r)" linux-tools-generic >/dev/null || \
  sudo apt-get install -y -qq ninja-build libjemalloc2 linux-tools-generic >/dev/null
python3 -m pip install --quiet --break-system-packages pyarrow==23.0.1
PERF=$(ls /usr/lib/linux-tools/*/perf | head -1)
cmake -S . -B "$WORK/build-g" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG -fno-omit-frame-pointer"
target=shopping-sessions-avx2
grep -qw avx512bw /proc/cpuinfo && target=shopping-sessions-avx512
grep -qw avx512_vbmi2 /proc/cpuinfo && target=shopping-sessions-vbmi2
cmake --build "$WORK/build-g" --target "$target" --parallel 4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 MALLOC_CONF=thp:always,metadata_thp:always,narenas:4
export DATA_DIR="$WORK/practice" NCPU=$(nproc) PROFILE=1
set +x
for i in 1 2 3; do rm -rf "$WORK/out"; OUTPUT_DIR="$WORK/out" "$WORK/build-g/$target"; done
rm -rf "$WORK/out"
sudo -E OUTPUT_DIR="$WORK/out" "$PERF" record -e cpu-clock -F 4000 -g -o "$WORK/perf.data" "$WORK/build-g/$target"
sudo chmod a+r "$WORK/perf.data"
echo "=== flat profile ==="
"$PERF" report -i "$WORK/perf.data" --no-children --sort sym -g none --percent-limit 0.25 2>/dev/null | grep '%' | cut -c1-170
echo "=== inclusive ==="
"$PERF" report -i "$WORK/perf.data" --children --sort sym -g none --percent-limit 1.5 2>/dev/null | grep '%' | cut -c1-170
echo "=== hot lines in session functions ==="
"$PERF" annotate -i "$WORK/perf.data" --stdio -l 2>/dev/null > "$WORK/annotate.txt" || true
python3 - "$WORK/annotate.txt" <<'PY'
import re, sys
cur = None; funcs = {}
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\s*:\s*\d+\s+[0-9a-f]+ <(.*)>:$", line)
    if m:
        cur = m.group(1)[:110]; funcs.setdefault(cur, {}); continue
    m = re.match(r"\s*([0-9.]+) :\s+[0-9a-f]+:\s+(.*?)\s*(//\s*(\S+))?$", line)
    if m and cur:
        funcs[cur][m.group(4) or "?"] = funcs[cur].get(m.group(4) or "?", 0) + float(m.group(1))
for name, lines in funcs.items():
    if not re.search(r"group_by_raw_id|SessionState|sessionize|process|CampaignTotals|push_back|radix|sort_", name):
        continue
    print("==", name)
    for src, pct in sorted(lines.items(), key=lambda x: -x[1])[:10]:
        print(f"   {pct:6.2f} {src}")
PY
