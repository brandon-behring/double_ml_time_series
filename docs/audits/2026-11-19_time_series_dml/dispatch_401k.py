"""Resume the 600 independent joint 401(k) refits with a host-wide resource budget."""

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2] / "results/audit_2026-11-19/401k"
SOURCE = "DMLonGitHub-b91cbf96c01eccd73367fbd6601ecdd7aa78403b"
METHODS = {"RLasso", "Trees", "Forest", "Boosting", "Nnet", "Ensemble", "best"}
PANELS = ("plinear", "interactive", "LATE")
JOBS = {f"all-{p}-k{k}-r{r:03d}": (p, k, r) for r in range(1, 101) for k in (2, 5) for p in PANELS}
GIB = 1024**3


def atomic_json(path, obj):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(obj, out, indent=2, sort_keys=True)
        out.write("\n")
    os.replace(out.name, path)


def valid_joint(path, job):
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return (
            len(rows) == len(METHODS)
            and {row["method"] for row in rows} == METHODS
            and all(
                (row["panel"], int(row["folds"]), int(row["partition"])) == job
                and all(math.isfinite(float(row[f])) for f in ("estimate", "se", "seconds"))
                and float(row["se"]) >= 0
                and float(row["seconds"]) >= 0
                for row in rows
            )
        )
    except (OSError, KeyError, TypeError, ValueError, csv.Error):
        return False


def reservations(path):
    fields = [value for value in re.split(r"[,\s]+", path.read_text().strip()) if value]
    numbers = {int(value) for value in fields}
    if not numbers <= set(range(1, 101)):
        raise ValueError("reserved partitions must be integers in [1, 100]")
    return numbers


def copy_missing(source, target):
    """Publish a complete copy without replacing another writer's atomic output."""
    if target.exists():
        return
    target.parent.mkdir(exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".incoming-", delete=False
        ) as tmp:
            stage = Path(tmp.name)
            try:
                with source.open("rb") as inp:
                    shutil.copyfileobj(inp, tmp)
                tmp.flush()
                os.link(stage, target)
            finally:
                stage.unlink(missing_ok=True)
    except (FileExistsError, FileNotFoundError):
        pass


def mirror(legacy, checkpoints):
    if not legacy.is_dir():
        return
    for file in legacy.glob("*.csv"):
        if file.stem.startswith("all-"):
            job = JOBS.get(file.stem)
            if not job or job[2] != 1 or not valid_joint(file, job):
                continue
        elif not re.fullmatch(
            r"(?:component-)?(?:RLasso|Trees|Forest|Boosting|Nnet|Ensemble|best)-(?:plinear|interactive|LATE)-k[25]-r\d{3}",
            file.stem,
        ):
            continue
        copy_missing(file, checkpoints / file.name)
    for old in legacy.glob("partial-all-*"):
        name = old.name.removeprefix("partial-")
        if name not in JOBS or (checkpoints / f"{name}.csv").exists():
            continue
        for fit in old.glob("fit-*.rds"):
            if (
                re.fullmatch(r"fit-\d+\.rds", fit.name)
                and not (checkpoints / f"{name}.csv").exists()
            ):
                copy_missing(fit, checkpoints / old.name / fit.name)


def stale_local_claim(claim, grace=45):
    """Only reclaim an identified, dead local owner in the host PID namespace."""
    try:
        owner = claim / "owner"
        if time.time() - owner.stat().st_mtime < grace:
            return False
        host, pid, *_ = owner.read_text().splitlines()
        if (
            host != os.uname().nodename
            or not pid.isdecimal()
            or Path("/proc/1/comm").read_text().strip() != "systemd"
        ):
            return False
        stat = Path("/proc") / pid / "stat"
        try:
            fields = stat.read_text().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            return True
        if fields[0] in ("Z", "X"):
            return True
        # A currently live process born after the ownership record reused its PID.
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        birth = time.time() - uptime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return birth > owner.stat().st_mtime + 5
    except (ValueError, IndexError, FileNotFoundError, PermissionError):
        return False


def physical_cores():
    pairs = set()
    for cpu in os.sched_getaffinity(0):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        if (path / "core_id").exists():
            pairs.add(((path / "physical_package_id").read_text(), (path / "core_id").read_text()))
    return len(pairs) or len(os.sched_getaffinity(0))


def resources(local_pids):
    busy = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    ticks = sum(busy) - busy[3] - busy[4]
    available = (
        int(re.search(r"^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)[1]) * 1024
    )
    workers = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            if (proc / "comm").read_text().strip() not in (
                "R",
                "Rscript",
            ) and pid not in local_pids:
                continue
            cmd = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").lower()
            if pid not in local_pids and b"401k" not in cmd:
                continue
            stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            if stat[0] in ("Z", "X"):
                continue
            # A sleeping mclapply coordinator waits for its forked R workers.
            # Its current RSS is already reflected in MemAvailable. Its children
            # each need a fit-sized growth allowance, while the coordinator does not.
            if stat[0] in ("S", "T", "t") and b"replicate_401k.r" in cmd:
                children = proc / "task" / proc.name / "children"
                if children.read_text().strip():
                    continue
            workers[pid] = (
                int(stat[11]) + int(stat[12]),
                int(stat[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    return time.monotonic(), ticks, available, workers


def fit_group(name):
    job = JOBS.get(name)
    return f"{job[0]}-k{job[1]}" if job else "unknown"


def claim_jobs(directories):
    """Include live legacy batch children, which need not have a one-row job file."""
    owners = {}
    for directory in directories:
        for claim in directory.glob(".claim-*"):
            fields = claim.name.removeprefix(".claim-").split("-", 1)
            if len(fields) != 2 or fields[0] not in METHODS | {"all"}:
                continue
            name = "all-" + fields[1]
            if name not in JOBS:
                continue
            try:
                host, pid, *_ = (claim / "owner").read_text().splitlines()
                if host == os.uname().nodename and pid.isdecimal():
                    owners.setdefault(int(pid), set()).add(name)
            except (OSError, ValueError):
                continue
    return owners


def job_from_environ(proc, jobs):
    try:
        entries = (proc / "environ").read_bytes().split(b"\0")
        for entry in entries:
            if not entry.startswith(b"DML_401K_JOBS="):
                continue
            path = Path(os.fsdecode(entry.split(b"=", 1)[1]))
            if path.parent != jobs or path.suffix != ".csv" or path.stem not in JOBS:
                return None
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            expected = list(map(str, JOBS[path.stem]))
            if rows == [["panel", "folds", "partition"], expected]:
                return path.stem
    except (OSError, csv.Error, ValueError):
        pass
    return None


def unclaimed_batch(proc):
    # Early R runs did not create claim files. Forked children inherit their
    # coordinator's command, which identifies a mixed batch but not a fold.
    try:
        argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if not any(a.startswith(b"--file=") and a.endswith(b"replicate_401k.R") for a in argv):
            return False
        args = argv[argv.index(b"--args") + 1 :]
        return len(args) == 5 and int(args[3]) > 1
    except (OSError, ValueError):
        return False


def fit_budget(group, baseline, peaks):
    # Five-fold LATE reached 2.67 GiB; other five-fold fits reached 2.15 GiB.
    if group == "unidentified-batch":
        return math.ceil(GIB * max(baseline, 3.0, 1.10 * max(peaks.values(), default=0)))
    floor = baseline
    if group.endswith("-k5"):
        floor = 3.0 if group == "LATE-k5" else 2.4
    return math.ceil(GIB * max(baseline, floor, 1.10 * peaks.get(group, 0)))


def worker_budgets(current, local, owners, jobs, baseline, peaks, proc_root=Path("/proc")):
    groups = {}
    for pid, (_, rss) in current[3].items():
        proc = proc_root / str(pid)
        candidates = set(owners.get(pid, ()))
        if pid in local:
            candidates.add(local[pid])
        else:
            one_row_job = job_from_environ(proc, jobs)
            if one_row_job:
                candidates.add(one_row_job)
        if candidates:
            # A PID may have just changed claims; reserve the largest plausible fit.
            group = max(map(fit_group, candidates), key=lambda g: fit_budget(g, baseline, peaks))
        else:
            group = "unidentified-batch" if unclaimed_batch(proc) else "unknown"
        groups[pid] = group
        try:
            status = (proc / "status").read_text()
            match = re.search(r"^VmHWM:\s+(\d+) kB$", status, re.M)
            if match:
                rss = max(rss, int(match[1]) * 1024)
        except OSError:
            pass
        peaks[group] = max(peaks.get(group, 0), rss / GIB)
    # A later worker's high-water can increase the allowance for every sibling.
    return {pid: fit_budget(group, baseline, peaks) for pid, group in groups.items()}


def choose_jobs(pending, slots, available_bytes, baseline, peaks):
    selected = []
    for name in pending:
        if len(selected) >= slots:
            break
        peak = fit_budget(fit_group(name), baseline, peaks)
        if peak <= available_bytes:
            selected.append(name)
            available_bytes -= peak
    return selected


def capacity(previous, current, args, worker_peaks=None, unseen=()):
    elapsed = max(current[0] - previous[0], 0.001)
    timebase = elapsed * os.sysconf("SC_CLK_TCK")
    total = max(0.0, (current[1] - previous[1]) / timebase)
    before, now = previous[3], current[3]
    audit = (
        sum(max(0, tick - before.get(pid, (tick,))[0]) for pid, (tick, _) in now.items()) / timebase
    )
    external = max(0.0, total - min(audit, len(now)))
    peak = args.worker_gib * GIB
    worker_peaks = worker_peaks or {}
    growth = sum(max(0, worker_peaks.get(pid, peak) - rss) for pid, (_, rss) in now.items())
    growth += sum(unseen)
    remaining = math.floor(current[2] - args.reserve_gib * GIB - growth)
    mem_slots = math.floor(remaining / peak)
    cpu_slots = math.floor(args.target - len(now) - len(unseen) - external)
    return max(0, min(cpu_slots, mem_slots, args.batch)), {
        "host_busy_cores": round(total, 2),
        "other_busy_cores": round(external, 2),
        "audit_r_processes": len(now),
        "available_gib": round(current[2] / GIB, 2),
        "largest_audit_worker_rss_gib": round(
            max((rss for _, rss in now.values()), default=0) / GIB, 3
        ),
        "reserved_worker_growth_gib": round(growth / GIB, 2),
        "cpu_slots": cpu_slots,
        "memory_slots": mem_slots,  # Upper bound in baseline-sized slots.
        "remaining_fit_budget_bytes": remaining,
    }


def worker_env(root):
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RCPP_PARALLEL_NUM_THREADS",
        "MC_CORES",
    ):
        env[key] = "1"
    env.update(
        R_LIBS_USER=os.pathsep.join(str(root / x) for x in ("hdm17", "rlib")), FIRST_PARTITION="1"
    )
    return env


def run(args):
    root = args.root.resolve()
    source = root / SOURCE
    checkpoints = root / "checkpoints"
    state = root / "dispatcher"
    jobs = state / "jobs"
    logs = state / "logs"
    evidence = state / "invalid"
    for directory in (checkpoints, jobs, logs, evidence):
        directory.mkdir(parents=True, exist_ok=True)
    lock = (state / "dispatcher.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another dispatcher owns this result directory.") from None
    if not source.is_dir() or not all((root / lib).is_dir() for lib in ("hdm17", "rlib")):
        raise SystemExit(f"Missing pinned source or R libraries in {root}")
    args.target = min(physical_cores(), args.target)
    reserve_file = args.reserve_file or root / "reserved-partitions.txt"
    reservations(reserve_file)  # Missing or malformed state must never release reservations.
    failures_file = state / "failures.json"
    failures = json.loads(failures_file.read_text()) if failures_file.exists() else {}
    status_file = state / "status.json"
    historic = (
        json.loads(status_file.read_text()).get("high_water", {}) if status_file.exists() else {}
    )
    active = {}
    peaks = historic.setdefault("worker_peak_rss_gib_by_fit", {})
    env = worker_env(root)
    previous = resources(set())
    time.sleep(min(2.0, args.poll))
    last_render = 0.0
    render_error = None
    render_failures = 0
    last_mirror = 0.0
    previous_reserved = None
    while True:
        now = time.monotonic()
        # Read the reservation file before the final mirror. Once a drained partition
        # is released, collect any newly published legacy fits before dispatching it.
        reserved = reservations(reserve_file)
        if (
            now - last_mirror >= args.mirror_interval
            or not last_mirror
            or reserved != previous_reserved
        ):
            mirror(args.legacy.resolve(), checkpoints)
            last_mirror = now
        previous_reserved = reserved
        completed = set()
        invalid = set()
        for path in checkpoints.glob("all-*.csv"):
            name = path.stem
            if name not in JOBS:
                continue
            if valid_joint(path, JOBS[name]):
                completed.add(name)
            else:
                if name in active and active[name].poll() is None:
                    continue
                # R publishes the entire file by rename. Preserve the invalid row
                # separately so its existence cannot short-circuit the next fit.
                path.replace(evidence / f"{name}-{time.time_ns()}.csv")
                failures[name] = failures.get(name, 0) + 1
                invalid.add(name)
                atomic_json(failures_file, failures)
        for name, proc in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del active[name]
            claim = checkpoints / f".claim-{name}"
            if name not in completed and name not in invalid and (rc != 0 or not claim.exists()):
                failures[name] = failures.get(name, 0) + 1
                atomic_json(failures_file, failures)
        claims = set()
        for claim in checkpoints.glob(".claim-all-*"):
            name = claim.name.removeprefix(".claim-")
            if name not in JOBS or not claim.is_dir():
                continue
            if name not in active and stale_local_claim(claim):
                try:
                    claim.rename(evidence / f"stale-{name}-{time.time_ns()}")
                except FileNotFoundError:
                    pass
            else:
                claims.add(name)
        running = set(active)
        blocked = {
            n for n, count in failures.items() if n not in completed and count >= args.max_failures
        }
        pending = [
            n
            for n, (_, _, r) in JOBS.items()
            if n not in completed | claims | running | blocked and r not in reserved
        ]
        local = {proc.pid: name for name, proc in active.items()}
        current = resources(local)
        owners = claim_jobs({checkpoints, args.legacy.resolve()})
        budgets = worker_budgets(current, local, owners, jobs, args.worker_gib, peaks)
        # New children can precede /proc visibility. Reserve their full peak and CPU.
        unseen = [
            fit_budget(fit_group(name), args.worker_gib, peaks)
            for pid, name in local.items()
            if pid not in current[3]
        ]
        slots, usage = capacity(previous, current, args, budgets, unseen)
        previous = current
        selected = choose_jobs(
            pending, slots, usage["remaining_fit_budget_bytes"], args.worker_gib, peaks
        )
        for name in selected:
            panel, folds, partition = JOBS[name]
            job_file = jobs / f"{name}.csv"
            job_file.write_text(f"panel,folds,partition\n{panel},{folds},{partition}\n")
            with (logs / f"{name}.log").open("ab") as log:
                log.write(
                    f"\nLaunching {name} at {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n".encode()
                )
                log.flush()
                active[name] = subprocess.Popen(
                    [
                        args.rscript,
                        str(HERE / "replicate_401k.R"),
                        str(source),
                        str(checkpoints),
                        "100",
                        "1",
                        "all",
                    ],
                    env={**env, "DML_401K_JOBS": str(job_file)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        finished = len(completed) == len(JOBS) and not active
        exhausted = (
            not active
            and not pending
            and not reserved
            and not (claims - completed)
            and bool(blocked)
        )
        if finished or exhausted or now - last_render >= args.render_interval:
            with (logs / "summary.log").open("ab") as log:
                result = subprocess.run(
                    [sys.executable, str(HERE / "summarize_401k.py"), str(checkpoints)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            render_error = result.returncode or None
            render_failures = render_failures + 1 if render_error else 0
            last_render = now
        for key in ("audit_r_processes", "largest_audit_worker_rss_gib", "host_busy_cores"):
            historic[key] = max(historic.get(key, 0), usage[key])
        historic["min_available_gib"] = min(
            historic.get("min_available_gib", float("inf")), usage["available_gib"]
        )
        historic["dispatcher_workers"] = max(historic.get("dispatcher_workers", 0), len(active))
        status = {
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "target_busy_cores": args.target,
            "valid_joint_jobs": len(completed),
            "total_joint_jobs": len(JOBS),
            "valid_finite_rows": 7 * len(completed),
            "reserved_partitions": sorted(reserved),
            "active": {n: p.pid for n, p in active.items()},
            "claimed": sorted(claims),
            "retry_exhausted": sorted(blocked),
            "failure_counts": failures,
            "pending_at_sample": len(pending),
            "launches_this_sample": len(selected),
            "resources": usage,
            "summary_exit_code": render_error or 0,
            "complete": finished and render_error is None,
            "high_water": historic,
            "stale_claims_archived": len(list(evidence.glob("stale-all-*"))),
        }
        atomic_json(status_file, status)
        if finished and not render_error:
            return 0
        if exhausted or (finished and render_failures >= args.max_failures):
            return 1
        time.sleep(args.poll)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="durable source, R libs and checkpoint parent",
    )
    parser.add_argument(
        "--reserve-file",
        type=Path,
        help="existing reloadable file of comma/newline-separated partition numbers; empty releases all (default ROOT/reserved-partitions.txt)",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("/tmp/dml-audit-1119/401k/checkpoints"),
        help="legacy checkpoint directory; only partition 1 joint CSVs are trusted",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="maximum host-wide core-equivalents; also capped at available physical cores",
    )
    parser.add_argument(
        "--reserve-gib", type=float, default=16, help="host memory headroom to preserve"
    )
    parser.add_argument(
        "--worker-gib",
        type=float,
        default=2.2,
        help="minimum projected peak in GiB per fit; five-fold and measured peaks can reserve more",
    )
    parser.add_argument(
        "--batch", type=int, default=5, help="maximum new single-thread workers per sample"
    )
    parser.add_argument(
        "--poll", type=float, default=10, help="seconds between resource and reservation samples"
    )
    parser.add_argument(
        "--render-interval", type=float, default=300, help="seconds between report updates"
    )
    parser.add_argument(
        "--mirror-interval", type=float, default=30, help="seconds between legacy cache scans"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="durable failure limit per job; exhausted jobs are listed in status.json",
    )
    parser.add_argument(
        "--rscript", default="Rscript", help="Rscript command or absolute executable"
    )
    args = parser.parse_args()
    if (
        min(args.target, args.batch, args.max_failures) < 1
        or args.reserve_gib < 0
        or min(args.worker_gib, args.poll, args.render_interval, args.mirror_interval) <= 0
    ):
        parser.error("budgets and intervals must be positive; reserve-gib may be zero")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
