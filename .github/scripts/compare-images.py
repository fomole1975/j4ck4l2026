"""Time two actual submission images on one runner and compare their output."""
from datetime import datetime
from pathlib import Path
import os
import shutil
import statistics
import subprocess
import tempfile

import polars as pl


def docker(*args, capture=False):
    command = ["docker", *map(str, args)]
    if capture:
        return subprocess.check_output(command, text=True).strip()
    subprocess.run(command, check=True)


def time_image(image, data, output, workers):
    output.mkdir(mode=0o777)
    output.chmod(0o777)
    for name in ("user_daily", "campaign_daily"):
        directory = output / name
        directory.mkdir()
        directory.chmod(0o777)
    container = docker(
        "create", "--network", "none", "--read-only", "--memory=12g",
        "--memory-swap=12g", "--user", "1000:1000", "--cpus", workers,
        "-e", f"NCPU={workers}", "-e", "PROFILE=1", "-v", f"{data}:/data:ro",
        "-v", f"{output}:/output", image, capture=True,
    )
    try:
        docker("start", "-a", container)
        state = docker("inspect", "-f", "{{.State.ExitCode}} {{.State.StartedAt}} {{.State.FinishedAt}}",
                       container, capture=True)
        code, start, end = state.split()
        if code != "0":
            raise RuntimeError(f"{image} exited with {code}")
        return (datetime.fromisoformat(end.replace("Z", "+00:00")) -
                datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    finally:
        docker("rm", container, capture=True)


def compare_table(left, right, table, keys):
    def load(folder):
        return pl.read_parquet(folder / table / "*.parquet").sort(keys)

    a, b = load(left), load(right)
    if a.height != b.height or set(a.columns) != set(b.columns):
        raise ValueError(f"{table} differs in shape")
    if not a.equals(b.select(a.columns)):
        raise ValueError(f"{table} differs in values")
    print(f"{table} matched all {a.height:,} rows", flush=True)


def compare(work, workload, workers):
    images = ["shopping-baseline", "shopping-sessions"]
    times = {name: [] for name in images}
    # The first pair warms both images and the shared read-only input. Each
    # subsequent pair reverses order to distribute any changes in host load.
    order = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        kept = {}
        for run, index in enumerate(order):
            image = images[index]
            output = Path(tmp) / f"out-{run}"
            elapsed = time_image(image, work / workload, output, workers)
            label = "warmup" if run < 2 else "measured"
            print(f"{workload} {image} {label} {elapsed:.4f}s", flush=True)
            if run >= 2:
                times[image].append(elapsed)
            if image in kept:
                shutil.rmtree(kept[image])
            kept[image] = output
        compare_table(kept[images[0]], kept[images[1]], "user_daily", ["user_id", "day"])
        compare_table(kept[images[0]], kept[images[1]], "campaign_daily", ["campaign_id", "day"])
    baseline, candidate = (statistics.median(times[name]) for name in images)
    print(f"{workload} median {baseline:.4f}s -> {candidate:.4f}s; "
          f"{100*(1-candidate/baseline):.2f}% less time", flush=True)


if __name__ == "__main__":
    root = Path(os.environ["WORK"]).resolve()
    threads = len(os.sched_getaffinity(0))
    for dataset in ("practice", "holdout", "final"):
        compare(root, dataset, threads)
