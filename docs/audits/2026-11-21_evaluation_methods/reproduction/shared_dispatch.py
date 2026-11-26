"""Use the historical dispatcher with a short-lived reservation for another local study."""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path


def process_start(pid, proc=Path("/proc")):
    """Identify a live process across PID reuse using Linux start ticks."""
    try:
        stat = (proc / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return int(stat[19]) if stat[0] not in {"Z", "X"} else None
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def requested_growth(file, now=None, proc=Path("/proc")):
    """Only the coordinating process may reserve up to 2.5 GiB of future growth."""
    try:
        with file.open() as stream:
            age = (time.time() if now is None else now) - os.fstat(stream.fileno()).st_mtime
            if not 0 <= age <= 15:
                return 0.0
            request = json.load(stream)
        start = process_start(int(request["pid"]), proc)
        if start is None or start != int(request["proc_start_ticks"]):
            return 0.0
        gib = float(request["required_gib"])
        return min(2.5, max(0.0, gib)) if math.isfinite(gib) else 0.0
    except (OSError, ValueError, TypeError, KeyError):
        return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("dispatcher", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    config, args = parser.parse_known_args()
    root = config.root.resolve()
    spec = importlib.util.spec_from_file_location("historical_dispatch", config.dispatcher)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    capacity, atomic = d.capacity, d.atomic_json

    def shared_capacity(previous, current, settings, worker_peaks=None, unseen=()):
        growth = requested_growth(root / "parallel-budget.json")
        # Debit only uncharged Python growth. Both studies use MemAvailable and
        # one common host reserve; R's own growth field remains directly usable.
        adjusted = argparse.Namespace(**vars(settings))
        adjusted.reserve_gib += growth
        slots, usage = capacity(previous, current, adjusted, worker_peaks, unseen)
        usage["parallel_growth_reserved_gib"] = round(growth, 5)
        return slots, usage

    # R owns the per-partition claims, so existing workers keep running and
    # continue to be budgeted by the original scheduler after a handover.
    claims = d.claim_jobs([root / "checkpoints"])
    inherited = {}
    for pid, names in claims.items():
        job = d.job_from_environ(Path("/proc") / str(pid), root / "dispatcher/jobs")
        born = process_start(pid)
        if job in names and born is not None:
            inherited[job] = (pid, born)

    def with_inherited(path, value):
        if path.name == "status.json":
            for job, (pid, born) in list(inherited.items()):
                if process_start(pid) == born:
                    value["active"][job] = pid
                else:
                    inherited.pop(job)
                    # An orphan is no longer a child we can waitpid(). Absence
                    # of a valid final file is the equivalent failure signal.
                    if not d.valid_joint(root / "checkpoints" / (job + ".csv"), d.JOBS[job]):
                        failures = value["failure_counts"]
                        failures[job] = failures.get(job, 0) + 1
                        atomic(path.with_name("failures.json"), failures)
            value["high_water"]["dispatcher_workers"] = max(
                value["high_water"]["dispatcher_workers"], len(value["active"])
            )
        atomic(path, value)

    d.capacity = shared_capacity
    d.atomic_json = with_inherited
    sys.argv = [str(config.dispatcher), "--root", str(root), *args]
    print(f"Tracking {len(inherited)} live R suites from the previous coordinator.", flush=True)
    raise SystemExit(d.main())


if __name__ == "__main__":
    main()
