"""Split container lifetime into startup, program and exit for image variants."""
from datetime import datetime
from pathlib import Path
import os, re, statistics, subprocess, tempfile

def docker(*args):
    return subprocess.run(["docker", *map(str, args)], check=True, text=True, capture_output=True).stdout.strip()

def stamp(text):
    # Docker reports nanoseconds; keep microseconds for datetime.
    text = re.sub(r"(\.\d{6})\d*", r"\1", text.replace("Z", "+00:00"))
    return datetime.fromisoformat(text).timestamp()

def run(label, args, data, work):
    out = Path(tempfile.mkdtemp(dir=work)); out.chmod(0o777)
    for name in ("user_daily", "campaign_daily"):
        (out / name).mkdir(); (out / name).chmod(0o777)
    cid = docker("create", "--network", "none", "--read-only", "--memory=12g", "--memory-swap=12g",
                 "--user", "1000:1000", "-v", f"{data}:/data:ro", "-v", f"{out}:/output", *args)
    try:
        subprocess.run(["docker", "start", "-a", cid], text=True, capture_output=True)
        log = subprocess.run(["docker", "logs", cid], text=True, capture_output=True).stderr
        started, finished = docker("inspect", "-f", "{{.State.StartedAt}} {{.State.FinishedAt}}", cid).split()
    finally:
        docker("rm", cid)
    s, f = stamp(started), stamp(finished)
    fields = dict(re.findall(r"(\w+)=([0-9.]+)", log))
    row = {"total": f - s}
    if "wall_main" in fields:
        main, done = int(fields["wall_main"]) / 1e9, int(fields["wall_done"]) / 1e9
        first = int(fields.get("wall_dispatch", fields["wall_main"])) / 1e9
        row.update(start=first - s, dispatch=main - first, program=done - main, exit=f - done)
        for key in ("compute", "parquet"):
            if key in fields:
                row[key] = float(fields[key])
    return row

def main():
    work = Path(os.environ["WORK"]).resolve()
    data = work / "practice"
    cpus = str(len(os.sched_getaffinity(0)))
    variant = "vbmi2" if "avx512_vbmi2" in open("/proc/cpuinfo").read() else "avx2"
    base = ["-e", f"NCPU={cpus}", "-e", "PROFILE=1"]
    variants = {
        "empty container": ["--entrypoint", "/bin/true", "shopping-sessions"],
        "scored (--cpus)": ["--cpus", cpus, *base, "shopping-sessions"],
        "no cpu quota": [*base, "shopping-sessions"],
        "no jemalloc": ["--cpus", cpus, *base, "-e", "LD_PRELOAD=", "shopping-sessions"],
        "jemalloc no thp": ["--cpus", cpus, *base, "-e", "MALLOC_CONF=narenas:4", "shopping-sessions"],
        "direct variant": ["--cpus", cpus, *base, "--entrypoint", f"/opt/pipeline/bin/shopping-sessions-{variant}",
                           "shopping-sessions"],
    }
    run("warmup", variants["scored (--cpus)"], data, work)
    results = {name: [] for name in variants}
    for round_ in range(5):
        for name, args in variants.items():
            results[name].append(run(name, args, data, work))
    for name, rows in results.items():
        keys = rows[0].keys()
        print(f"{name:18}", "  ".join(f"{k}={statistics.median(r[k] for r in rows):.4f}" for k in keys), flush=True)

main()
