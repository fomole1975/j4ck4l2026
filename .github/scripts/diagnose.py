"""Compare per-phase timings of this branch and compare_commit on one runner."""
from datetime import datetime
from pathlib import Path
import os, re, statistics, subprocess, tempfile

def docker(*args):
    return subprocess.run(["docker", *map(str, args)], check=True, text=True, capture_output=True).stdout.strip()

def stamp(text):
    text = re.sub(r"(\.\d{6})\d*", r"\1", text.replace("Z", "+00:00"))
    return datetime.fromisoformat(text).timestamp()

def run(image, data, work, cpus):
    out = Path(tempfile.mkdtemp(dir=work)); out.chmod(0o777)
    for name in ("user_daily", "campaign_daily"):
        (out / name).mkdir(); (out / name).chmod(0o777)
    cid = docker("create", "--network", "none", "--read-only", "--memory=12g", "--memory-swap=12g",
                 "--user", "1000:1000", "--cpus", cpus, "-e", f"NCPU={cpus}", "-e", "PROFILE=1",
                 "-v", f"{data}:/data:ro", "-v", f"{out}:/output", image)
    try:
        subprocess.run(["docker", "start", "-a", cid], text=True, capture_output=True)
        log = subprocess.run(["docker", "logs", cid], text=True, capture_output=True).stderr
        code, started, finished = docker("inspect", "-f", "{{.State.ExitCode}} {{.State.StartedAt}} {{.State.FinishedAt}}", cid).split()
    finally:
        docker("rm", cid)
    if code != "0":
        raise RuntimeError(f"{image} exited {code}: {log}")
    row = {"container": stamp(finished) - stamp(started)}
    for key, value in re.findall(r"([\w+]+)=([0-9.]+)s?\b", log):
        if not key.startswith("wall") and key != "isa":
            row[key] = float(value)
    return row

def main():
    work = Path(os.environ["WORK"]).resolve()
    cpus = str(len(os.sched_getaffinity(0)))
    images = ["shopping-sessions"] + (["shopping-baseline"] if os.environ.get("BASE_SHA") else [])
    for dataset in ("practice", "final"):
        data = work / dataset
        for image in images:
            run(image, data, work, cpus)  # warm-up
        rows = {image: [] for image in images}
        for round_ in range(6):
            for image in (images if round_ % 2 == 0 else images[::-1]):
                rows[image].append(run(image, data, work, cpus))
        for image in images:
            keys = [k for k in rows[image][0] if all(k in r for r in rows[image])]
            print(f"{dataset:8} {image:18}", "  ".join(f"{k}={statistics.median(r[k] for r in rows[image]):.4f}" for k in keys), flush=True)

main()
