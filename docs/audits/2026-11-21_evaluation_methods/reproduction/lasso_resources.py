"""Share the 401(k) dispatcher's RAM and host CPU budget with one-thread bootstrap refits."""

import argparse
import json
import math
import os
import signal
import tempfile
import time
from pathlib import Path

GIB = 1024**3
WORKER_GIB = 0.5
SHARED_GROWTH_GIB = 2.5


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Cannot read MemAvailable")


def dispatch(status):
    if status is None:
        return 0.0, max(1, len(os.sched_getaffinity(0)) - 8), True, False
    data = json.loads(status.read_text())
    if data["complete"]:
        return 0.0, data["target_busy_cores"], True, False
    # The R dispatcher writes its pre-launch budget together with post-launch PIDs.
    # Wait for its next sample before sharing RAM released by that discrepancy.
    settled = not data.get("launches_this_sample", 0)
    resources = data["resources"]
    return (
        resources["reserved_worker_growth_gib"],
        data["target_busy_cores"],
        settled and time.time() - status.stat().st_mtime < 90,
        True,
    )


def process_identity(pid):
    return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])


def registered_probe(cache):
    marker = cache / "serial-peak-active.json"
    if not marker.exists():
        return None
    data = json.loads(marker.read_text())
    pid = data["pid"]
    try:
        if data["proc_start_ticks"] != process_identity(pid):
            return None
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    peak_check = b"--measure-serial-peak" in cmd
    grouped_check = b"--governed-cache" in cmd and any(
        Path(os.fsdecode(arg)).name == "lasso_grouped_check.py" for arg in cmd
    )
    if not (peak_check or grouped_check) or os.fsencode(cache.resolve()) not in cmd:
        return None
    return pid


def cpu_and_children(pid, probe=None):
    cpu = [int(t) for t in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    host = sum(cpu) - cpu[3] - cpu[4]
    entries = {}
    parent = Path("/proc") / str(pid)
    cmd = (parent / "cmdline").read_bytes()
    children = [int(p) for p in (parent / "task" / str(pid) / "children").read_text().split()]
    for child in children + ([probe] if probe is not None else []):
        proc = Path("/proc") / str(child)
        try:
            if child != probe and (proc / "cmdline").read_bytes() != cmd:
                continue
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            hwm = next(
                (
                    line
                    for line in (proc / "status").read_text().splitlines()
                    if line.startswith("VmHWM:")
                ),
                None,
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (child != probe and int(stat[1]) != pid) or stat[0] == "Z" or hwm is None:
            continue
        entries[child] = (
            int(stat[11]) + int(stat[12]),
            int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / GIB,
            stat[0],
            int(hwm.split()[1]) / 1024**2,
        )
    return time.monotonic(), host, entries


def publish_request(status, pid, start_ticks, required):
    if status is None:
        return
    base = status.parent.parent if status.parent.name == "dispatcher" else status.parent
    path = base / "parallel-budget.json"
    data = {
        "pid": pid,
        "proc_start_ticks": int(start_ticks),
        "required_gib": min(SHARED_GROWTH_GIB, required),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(data, file)
        file.write("\n")
    Path(file.name).replace(path)


def select(
    previous, current, worker_cap, reserve, status, probe=None, previous_peak=0, priority=None
):
    elapsed = max(0.001, current[0] - previous[0]) * os.sysconf("SC_CLK_TCK")
    busy = max(0, (current[1] - previous[1]) / elapsed)
    own = (
        sum(
            max(0, tick - previous[2].get(pid, (tick,))[0])
            for pid, (tick, _, _, _) in current[2].items()
        )
        / elapsed
    )
    r_growth, target, fresh, sharing = dispatch(status)
    available = memory_available()
    remaining = available - r_growth - reserve
    # Padded observed Linux VmHWM covers native solver RSS. Current RSS is already
    # in MemAvailable, so the request and the admission charge only incremental growth.
    highest_rss = max((row[3] for row in current[2].values()), default=0)
    peak = max(WORKER_GIB, 1.5 * max(previous_peak, highest_rss))
    growths = [max(0.05, peak - row[1]) for row in current[2].values()]
    desired = (
        min(SHARED_GROWTH_GIB, sum(sorted(growths, reverse=True)[:worker_cap])) if sharing else 0.0
    )
    if sharing:
        remaining = min(remaining, desired)
    room = max(0, min(worker_cap, math.floor(target - max(0.0, busy - own)))) if fresh else 0
    selected = set()
    # Both R and Python current RSS are already reflected in
    # MemAvailable. Only the additional growth to a Python worker's allowance is charged.
    priority = priority or {}
    for pid, (_, rss, _state, _hwm) in sorted(
        current[2].items(),
        key=lambda e: (e[0] != probe, priority.get(e[0], 0), e[1][2] == "T", e[0]),
    ):
        growth = max(0.05, peak - rss)
        if len(selected) < room and remaining >= growth:
            selected.add(pid)
            remaining -= growth
    return selected, {
        "available_gib": round(available, 2),
        "r_future_growth_gib": r_growth,
        "system_reserve_gib": reserve,
        "worker_vmhwm_gib": round(highest_rss, 4),
        "worker_peak_allowance_gib": round(peak, 4),
        "requested_python_growth_gib": desired,
        "serial_peak_probe_in_pool": probe is not None,
        "host_cpu_target": target,
        "host_busy_cores": round(busy, 2),
        "lasso_busy_cores": round(own, 2),
        "status_fresh": fresh,
        "pool": len(current[2]),
        "selected": len(selected),
    }


def guard(pid, status, cache, cap, reserve):
    parent = Path("/proc") / str(pid)
    args = [os.fsdecode(a) for a in (parent / "cmdline").read_bytes().split(b"\0") if a]
    if not any(Path(a).name == "standalone_lasso.py" for a in args) or not any(
        Path(a).resolve() == cache.resolve() for a in args
    ):
        raise ValueError("Unexpected bootstrap pool parent or cache")
    identity = (parent / "stat").read_text().rsplit(")", 1)[1].split()[19]
    previous = cpu_and_children(pid)
    peak_seen = 0.0
    last_granted = {}
    priority = {}
    last_rotation = 0.0
    held = {}
    try:
        while parent.exists():
            time.sleep(3)
            try:
                if (parent / "stat").read_text().rsplit(")", 1)[1].split()[19] != identity:
                    break
                probe = registered_probe(cache)
                current = cpu_and_children(pid, probe)
            except (FileNotFoundError, ProcessLookupError):
                break
            peak_file = cache / "serial-peak.json"
            if peak_file.is_file():
                measured = json.loads(peak_file.read_text())
                if measured.get("measurement", "").startswith("/proc/self/status VmHWM"):
                    peak_seen = max(peak_seen, measured["full_process_peak_rss_kib"] / 1024**2)
            peak_seen = max([peak_seen, *(row[3] for row in current[2].values())])
            # Rotate leases every minute. A sleeping pool worker may have finished
            # its last task while an older task is still suspended in another PID.
            if current[0] - last_rotation >= 60:
                order = sorted(current[2], key=lambda child: (last_granted.get(child, 0), child))
                priority = {child: rank for rank, child in enumerate(order)}
                last_rotation = current[0]
            selected, row = select(
                previous, current, cap, reserve, status, probe, peak_seen, priority
            )
            last_granted.update(dict.fromkeys(selected, current[0]))
            if probe in current[2] and current[2][probe][2] == "T":
                held[probe] = str(process_identity(probe))
            previous = current
            if status is not None:
                publish_request(status, pid, identity, row["requested_python_growth_gib"])
            for child, (_, _, state, _) in current[2].items():
                try:
                    if child in selected and child in held:
                        os.kill(child, signal.SIGCONT)
                        held.pop(child, None)
                    elif child not in selected and state != "T":
                        os.kill(child, signal.SIGSTOP)
                        held[child] = (
                            (Path("/proc") / str(child) / "stat")
                            .read_text()
                            .rsplit(")", 1)[1]
                            .split()[19]
                        )
                except ProcessLookupError:
                    held.pop(child, None)
            row["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with (cache / "admission.jsonl").open("a") as log:
                log.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        for child, identity in held.items():
            try:
                actual = (
                    (Path("/proc") / str(child) / "stat").read_text().rsplit(")", 1)[1].split()[19]
                )
                if actual == identity:
                    os.kill(child, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                pass


def stop_guard(signum, frame):
    raise SystemExit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--reserve-gib", type=float, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop_guard)
    guard(args.pid, args.status, args.cache, args.cap, args.reserve_gib)
