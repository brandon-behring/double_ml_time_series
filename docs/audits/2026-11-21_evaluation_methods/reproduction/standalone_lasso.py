"""Checkpoint an exact outer-parallel replay of the snapshot's bootstrap and diagnostics."""

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
import resource
import runpy
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from lasso_resources import (
    WORKER_GIB,
    cpu_and_children,
    dispatch,
    memory_available,
    publish_request,
)

SEED = 42
COUNT = 1000
CHECK = 3
# The prior CLI's fixed bootstrap and the tracked 9,915-observation sample.
MODULE = Path("dml_ts/validation/lasso_diagnostic.py")
SCRIPT = Path("scripts/run_lasso_diagnostic.py")
FIXTURE = Path("test/fixtures/401k_data.csv")


def encode(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def save(path, content):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(content, f, indent=2, allow_nan=False, default=encode)
        f.write("\n")
    Path(f.name).replace(path)


def diagnostic(root):
    if sys.path[0] != str(root):
        sys.path.insert(0, str(root))
    from dml_ts.validation.lasso_diagnostic import LassoDiagnostic

    if Path(inspect.getfile(LassoDiagnostic)).resolve() != root / MODULE:
        raise ValueError("Expected the snapshotted LassoDiagnostic")
    return LassoDiagnostic(data_path=str(root / FIXTURE), random_state=SEED, verbose=False)


def stamp(warning_list):
    result = {}
    for entry in warning_list:
        name = entry.category.__name__
        result[name] = result.get(name, 0) + 1
    return result


def fitted_process_peak_kib():
    # Linux ru_maxrss can retain a large pre-exec launcher's RSS. VmHWM belongs to
    # this process image and includes native allocations between governor samples.
    return int(
        next(
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmHWM:")
        ).split()[1]
    )


def compute(root, start, states, count):
    obj = diagnostic(root)
    obj._rng.set_state(states)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = obj.analyze_bootstrap_distribution(n_bootstrap=count)
    # The original function may warn about the ancillary SD/Shapiro calculation for
    # a one-draw probe. The fitted estimate remains available and is checked below.
    return {
        "first_draw": start,
        "values": result.ate_estimates.tolist(),
        "warnings": stamp(caught),
        "peak_rss_kib": fitted_process_peak_kib(),
    }


def auxiliary(root, kind):
    obj = diagnostic(root)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "seeds":
            result = obj.analyze_seed_sensitivity(n_seeds=20)
        else:
            result = obj.analyze_hyperparameter_sensitivity(kind)
    return {
        "class": type(result).__name__,
        "fields": dataclasses.asdict(result),
        "warnings": stamp(caught),
    }


def original_summary(method):
    # Evaluate the original aggregation, with the same completed estimates in draw order.
    # The complete fit loop itself is executed by the original method in compute().
    body = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0].body
    (loop,) = [i for i, statement in enumerate(body) if isinstance(statement, ast.For)]
    function = ast.parse("def summary(self, ate_estimates, n_bootstrap): pass").body[0]
    function.body = body[loop + 1 :]
    if not isinstance(function.body[-1], ast.Return):
        raise ValueError("Unrecognized bootstrap aggregation")
    namespace = method.__globals__.copy()
    exec(
        compile(ast.fix_missing_locations(ast.Module([function], [])), str(MODULE), "exec"),
        namespace,
    )
    return namespace["summary"]


@contextlib.contextmanager
def guarded_pool(cache, status, capacity, reserve, invocation_dir):
    command = [
        sys.executable,
        str(Path(__file__).with_name("lasso_resources.py").resolve()),
        str(os.getpid()),
        str(cache),
        "--cap",
        str(capacity),
        "--reserve-gib",
        str(reserve),
    ]
    if status is not None:
        command.extend(["--status", str(status)])
    governor = subprocess.Popen(command, cwd=invocation_dir)
    try:
        with ProcessPoolExecutor(max_workers=capacity) as pool:
            yield pool, governor
    finally:
        governor.terminate()
        governor.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reserve-gib", type=float, default=16)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--measure-serial-peak",
        action="store_true",
        help="Measure three verified serial fits in the live governed pool",
    )
    parser.add_argument("--r-dispatch-status", type=Path, help="Live 401(k) dispatcher status JSON")
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_gib < 0:
        parser.error("Use a positive worker limit and nonnegative reserve")
    root = args.snapshot.resolve()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    invocation_dir = Path.cwd()
    status = args.r_dispatch_status.resolve() if args.r_dispatch_status else None
    colocated = cache.parent / "401k/dispatcher/status.json"
    if status is None and colocated.is_file():
        status = colocated
    if root == Path(__file__).resolve().parents[4]:
        parser.error(
            "Use a disposable copy because the original CLI writes into its results folder"
        )
    if args.measure_serial_peak:
        heartbeat = cache / "admission.jsonl"
        if not heartbeat.is_file() or time.time() - heartbeat.stat().st_mtime > 15:
            raise ValueError(
                "Start live pool admission before registering a serial peak measurement"
            )
        verification = json.loads((cache / "verification.json").read_text())
        if not verification["exact"] or verification["serial"] != verification["parallel"]:
            raise ValueError("Serial peak calibration requires the verified three-fit result")
        marker = cache / "serial-peak-active.json"
        pid = os.getpid()
        birth = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        launch_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(marker, {"pid": pid, "proc_start_ticks": birth})
        try:
            # The live pool governor resumes this process within its existing worker,
            # CPU and incremental memory limits, together with the production fits.
            os.kill(pid, signal.SIGSTOP)
            obj = diagnostic(root)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                measured = obj.analyze_bootstrap_distribution(n_bootstrap=CHECK)
            np.testing.assert_array_equal(measured.ate_estimates, verification["serial"])
            save(
                cache / "serial-peak.json",
                {
                    "rows": len(obj.preprocess_data()[0]),
                    "verified_serial_estimates": measured.ate_estimates,
                    "full_process_peak_rss_kib": fitted_process_peak_kib(),
                    "measurement": "/proc/self/status VmHWM before process exit",
                    "initial_process_rusage_maxrss_kib": launch_maxrss,
                    "final_process_rusage_maxrss_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "warnings": stamp(caught),
                },
            )
        finally:
            marker.unlink(missing_ok=True)
        return
    for flag in [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DML_N_JOBS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[flag] = "1"
    os.chdir(root)
    obj = diagnostic(root)
    n = len(obj.preprocess_data()[0])
    original = type(obj).analyze_bootstrap_distribution
    aggregate = original_summary(original)
    states, indices_hash = [], hashlib.sha256()
    for _ in range(COUNT):
        states.append(obj._rng.get_state())
        indices = np.asarray(obj._rng.choice(n, size=n, replace=True), dtype="<i8")
        indices_hash.update(indices.tobytes())
    final_state = obj._rng.get_state()
    manifest = {
        "sample_rows": n,
        "seed": SEED,
        "bootstrap_draws": COUNT,
        "bootstrap_indices_sha256": indices_hash.hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in [SCRIPT, MODULE, FIXTURE]
        },
    }
    from importlib.metadata import version

    manifest["versions"] = {p: version(p) for p in ["econml", "scikit-learn", "numpy", "scipy"]}
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Incompatible checkpoint fingerprint")
    save(manifest_path, manifest)
    # Cold workers have no RSS charged to MemAvailable yet. Reserve their full
    # allowance and respect the CPU target across R, Python and all other jobs.
    coordinator_start = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    while True:
        before = cpu_and_children(os.getpid())
        time.sleep(1)
        after = cpu_and_children(os.getpid())
        busy = (after[1] - before[1]) / ((after[0] - before[0]) * os.sysconf("SC_CLK_TCK"))
        r_growth, cpu_target, fresh, sharing = dispatch(status)
        if sharing:
            publish_request(status, os.getpid(), coordinator_start, args.workers * WORKER_GIB)
        ram_slots = int((memory_available() - args.reserve_gib - r_growth) // WORKER_GIB)
        capacity = min(args.workers, int(cpu_target - busy), ram_slots)
        if fresh and capacity > 0:
            break
        time.sleep(5)
    print(
        f"Host has {memory_available():.1f} GiB available, R workers reserve {r_growth:.1f} GiB "
        f"for growth; starting up to {capacity} one-thread workers",
        flush=True,
    )
    with guarded_pool(cache, status, capacity, args.reserve_gib, invocation_dir) as (
        pool,
        governor,
    ):
        probe_path = cache / "verification.json"
        if not probe_path.exists():
            # Compare one unmodified serial three-fit run with independently scheduled
            # single draws at the very same three MT19937 states, on the full fixture.
            direct = pool.submit(compute, root, 0, states[0], CHECK)
            single = [pool.submit(compute, root, i, states[i], 1) for i in range(CHECK)]
            direct_result = direct.result()
            serial = direct_result["values"]
            records = [future.result() for future in single]
            parallel = [row["values"][0] for row in records]
            np.testing.assert_array_equal(serial, parallel)
            # Check the same aggregation on the independently verified serial and parallel values.
            # The recomputed raw estimates were compared above before reuse as checkpoints.
            verified = aggregate(obj, np.asarray(parallel), CHECK)
            np.testing.assert_array_equal(
                [verified.ci_lower, verified.ci_upper], np.percentile(serial, [2.5, 97.5])
            )
            np.testing.assert_equal(verified.std_ate, np.std(serial, ddof=1))
            for row in records:
                save(cache / f"chunk-{row['first_draw']:04d}.json", row)
            save(
                probe_path,
                {
                    "serial": serial,
                    "serial_peak_rss_kib": direct_result["peak_rss_kib"],
                    "parallel_peak_rss_kib": [row["peak_rss_kib"] for row in records],
                    "parallel": parallel,
                    "exact": True,
                    "mean": verified.mean_ate,
                    "ci": [verified.ci_lower, verified.ci_upper],
                },
            )
            print(
                "Verified three original full-data serial fits against three parallel fits",
                flush=True,
            )
        proof = json.loads(probe_path.read_text())
        if not proof["exact"] or len(proof["serial"]) != CHECK:
            raise ValueError("Missing verification against the unmodified serial bootstrap")
        np.testing.assert_array_equal(proof["serial"], proof["parallel"])
        for i, number in enumerate(proof["serial"]):
            checkpoint = json.loads((cache / f"chunk-{i:04d}.json").read_text())
            np.testing.assert_array_equal(checkpoint["values"], [number])
        if args.probe_only:
            return
        skip = set()
        for file in cache.glob("chunk-*.json"):
            chunk = json.loads(file.read_text())
            start = chunk["first_draw"]
            values = chunk["values"]
            if not all(np.isfinite(values)) or any(
                i in skip or not 0 <= i < COUNT for i in range(start, start + len(values))
            ):
                raise ValueError(f"Invalid or overlapping checkpoint {file.name}")
            skip.update(range(start, start + len(values)))
        pending = []
        start = 0
        while start < COUNT:
            if start in skip:
                start += 1
                continue
            stop = start + 1
            while stop < min(COUNT, start + 4) and stop not in skip:
                stop += 1
            pending.append((start, stop - start))
            start = stop
        prepared = ["seeds", "cv_folds", "max_iter"]
        tasks = [
            (kind, None) for kind in prepared if not (cache / f"{kind}.json").exists()
        ] + pending
        active = {}
        while tasks or active:
            if governor.poll() is not None:
                raise RuntimeError(
                    f"Shared resource governor stopped with status {governor.returncode}"
                )
            # The live governor rechecks combined RAM and CPU and reversibly holds
            # any excess workers. Every admitted or held task retains its RNG state.
            while tasks and len(active) < capacity:
                key, size = tasks.pop(0)
                if isinstance(key, str):
                    future = pool.submit(auxiliary, root, key)
                    output = cache / f"{key}.json"
                else:
                    future = pool.submit(compute, root, key, states[key], size)
                    output = cache / f"chunk-{key:04d}.json"
                active[future] = output
            if not active:
                time.sleep(10)
                continue
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                file = active.pop(future)
                result = future.result()
                save(file, result)
                if "values" in result:
                    skip.update(
                        range(result["first_draw"], result["first_draw"] + len(result["values"]))
                    )
                print(f"{len(skip)}/{COUNT} complete; saved {file.name}", flush=True)
    all_values = np.full(COUNT, np.nan)
    warning_counts = {}
    for file in sorted(cache.glob("chunk-*.json")):
        row = json.loads(file.read_text())
        start = row["first_draw"]
        all_values[start : start + len(row["values"])] = row["values"]
        for name, count in row["warnings"].items():
            warning_counts[name] = warning_counts.get(name, 0) + count
    if not np.isfinite(all_values).all():
        raise ValueError("Missing checkpointed estimates")
    # The original CLI's three result-producing methods receive the same resulting
    # outputs. Its reporting, diagnostics, checks, root-cause logic, and file writer run unchanged.
    model_module = importlib.import_module(type(obj).__module__)

    def bootstrap(self, n_bootstrap=COUNT):
        if n_bootstrap != COUNT:
            raise ValueError("Unexpected bootstrap size")
        self._rng.set_state(final_state)
        return aggregate(self, all_values, n_bootstrap)

    def restore(kind):
        row = json.loads((cache / f"{kind}.json").read_text())
        fields = row["fields"]
        for name in ["ate_estimates", "parameter_values", "std_errors", "random_states"]:
            if name in fields:
                fields[name] = np.asarray(fields[name])
        return getattr(model_module, row["class"])(**fields)

    def hyper(self, parameter, values=None):
        if values is not None:
            raise ValueError("Extraction verified only for the original grid")
        return restore(parameter)

    def seed(self, n_seeds=20):
        if n_seeds != 20:
            raise ValueError("Unexpected seed count")
        return restore("seeds")

    from unittest.mock import patch

    with contextlib.ExitStack() as stack:
        for name, replacement in [
            ("analyze_bootstrap_distribution", bootstrap),
            ("analyze_hyperparameter_sensitivity", hyper),
            ("analyze_seed_sensitivity", seed),
        ]:
            stack.enter_context(patch.object(type(obj), name, replacement))
        report = runpy.run_path(str(root / SCRIPT), run_name="__main__")["results"]
    np.testing.assert_array_equal(report.bootstrap_diagnostic.ate_estimates, all_values)
    save(
        cache / "complete.json",
        {
            "completed": COUNT,
            "workers_cap": capacity,
            "ram_reserve_gib": args.reserve_gib,
            "warning_counts": warning_counts,
            "mean": report.bootstrap_diagnostic.mean_ate,
        },
    )


if __name__ == "__main__":
    main()
