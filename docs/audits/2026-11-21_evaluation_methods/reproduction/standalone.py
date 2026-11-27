"""Replay standalone studies in a disposable copy; keep original receipts untouched."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

OLD = Path("docs/audits/2026-11-19_time_series_dml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Isolated copy of the audited source")
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pilot-inputs", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--lasso-workers",
        type=int,
        default=12,
        help="Outer bootstrap workers (1 invokes the unmodified serial CLI)",
    )
    parser.add_argument(
        "--lasso-reserve-gib",
        type=float,
        default=16,
        help="Minimum host RAM headroom for new bootstrap workers",
    )
    parser.add_argument(
        "--grid-workers",
        type=int,
        default=4,
        help="Independent workers for the 81-setting Monte Carlo grid",
    )
    parser.add_argument(
        "--r-library",
        type=Path,
        help="R library containing sandwich for the original HeartSteps code",
    )
    parser.add_argument(
        "--r-dispatch-status",
        type=Path,
        help="Live 401(k) dispatcher status for shared CPU and RAM",
    )
    args = parser.parse_args()
    root, out, raw, cache = [
        p.resolve() for p in (args.snapshot, args.output, args.pilot_inputs, args.cache)
    ]
    if min(args.jobs, args.grid_workers, args.lasso_workers) < 1:
        parser.error("Worker limits must be positive")
    if root == Path(__file__).resolve().parents[4]:
        parser.error("Supply a disposable checkout to protect archived results")
    out.mkdir(parents=True, exist_ok=True)
    sim = out / "adversarial"
    sim.mkdir(exist_ok=True)
    py = args.python
    jobs = {}

    def add(name, *args):
        jobs[name] = list(map(str, args))

    for f in sorted((root / "examples").glob("*.py")):
        add(f.stem, py, f)
    for name in [
        "method_checks",
        "chapter3_baselines",
        "chapter10_pipeline",
        "rescore_bias_archive",
    ]:
        add(name, py, OLD / (name + ".py"))
    add(
        "chapter5",
        py,
        OLD / "chapter5_check.py",
        "--workers",
        3,
        "--draws",
        out / "chapter5-draws.csv",
    )
    add(
        "panel",
        py,
        "scripts/mc_panel_dml_table.py",
        "--workers",
        3,
        "--draws",
        out / "panel-draws.csv",
    )
    add("insurance", py, "scripts/mc_insurance_pricing.py", "--workers", 3, "--output", out)
    add(
        "retail",
        py,
        OLD / "retail_check.py",
        "--cache-dir",
        cache,
        "--workers",
        2,
        "--output-dir",
        out,
    )
    add(
        "stress",
        py,
        "scripts/validate_falsification.py",
        "--workers",
        3,
        "--check-controls",
        "--output",
        out,
    )
    ad = OLD / "adversarial"
    add(
        "adversarial-empirical",
        py,
        ad / "empirical.py",
        "--cache-dir",
        cache,
        "--output",
        out / "empirical.json",
    )
    add("adversarial-transport", py, ad / "transport.py", "--cache-dir", cache)
    add("adversarial-failures", py, ad / "failures.py", "--output", out / "failures.json")
    add("adversarial-simulation", py, ad / "simulation.py", "--workers", 4, "--output", sim)
    add(
        "adversarial-oracle",
        py,
        ad / "simulation.py",
        "--workers",
        4,
        "--output",
        sim,
        "--interaction-oracle",
    )
    add(
        "adversarial-crossfit",
        py,
        ad / "simulation_crossfit.py",
        "--workers",
        4,
        "--output",
        sim,
        "--primary",
        sim / "simulation-draws.csv.gz",
    )
    pilot = OLD / "pilots"
    add("bike", py, pilot / "bike.py", raw / "dml-audit-bike.zip")
    add(
        "heartsteps",
        "Rscript",
        pilot / "heartsteps.R",
        raw / "heartsteps_audit_suggestions.csv",
        raw / "heartsteps_audit_xgeepack.R",
    )
    src = raw / "dml-audit-synthdid-src"
    add(
        "california-reference",
        "Rscript",
        pilot / "california_reference.R",
        src / "data/california_prop99.csv",
        src,
    )
    add("california", py, pilot / "california.py", src / "data/california_prop99.csv")
    add(
        "iflex-reference",
        "Rscript",
        pilot / "iflex.R",
        raw / "iflex_audit.rdata",
        out / "iflex_daily.csv",
    )
    add("iflex-temporal", py, pilot / "iflex_temporal.py", out / "iflex_daily.csv")
    for name in [
        "run_401k_replication",
        "run_lasso_diagnostic",
        "test_issue_c2_binary_treatment",
        "test_issue_c3_monte_carlo_comparability",
        "test_issue_c5_401k_covariates",
        "verify_coverage_fix",
        "comprehensive_coverage_test",
        "comprehensive_method_comparison",
    ]:
        add(name, py, "scripts/" + name + ".py")
    if args.lasso_workers > 1:
        add(
            "run_lasso_diagnostic",
            py,
            Path(__file__).with_name("standalone_lasso.py").resolve(),
            root,
            out.parent / "standalone-lasso",
            "--workers",
            args.lasso_workers,
            "--reserve-gib",
            args.lasso_reserve_gib,
            *(
                ["--r-dispatch-status", args.r_dispatch_status.resolve()]
                if args.r_dispatch_status
                else []
            ),
        )
    # Each worker owns a full configuration, preserving the ordering of its 100 draws.
    add(
        "bias_validation_simulation",
        py,
        "-c",
        f"import scripts.bias_validation_simulation as m; m.N_JOBS={args.grid_workers}; m.main()",
    )
    env = os.environ.copy()
    env.update(
        dict.fromkeys(
            [
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "DML_N_JOBS",
                "BLIS_NUM_THREADS",
            ],
            "1",
        )
    )
    env.update(
        PYTHONPATH=str(root), MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "mpl"), PYTHONUNBUFFERED="1"
    )
    if args.r_library:
        env["R_LIBS_USER"] = os.pathsep.join(
            filter(None, [str(args.r_library.resolve()), env.get("R_LIBS_USER")])
        )
    receipts = out / "receipts"
    receipts.mkdir(exist_ok=True)

    def run(name):
        log = out / (name + ".log")
        began = time.time()
        with log.open("w") as fh:
            result = subprocess.run(
                jobs[name], stdout=fh, stderr=subprocess.STDOUT, cwd=root, env=env
            )
        record = {
            "study": name,
            "command": jobs[name],
            "exit": result.returncode,
            "seconds": round(time.time() - began, 2),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        }
        (receipts / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print(name, result.returncode, record["seconds"], flush=True)
        return result.returncode

    dependencies = {
        "adversarial-oracle": "adversarial-simulation",
        "adversarial-crossfit": "adversarial-simulation",
        "iflex-temporal": "iflex-reference",
    }
    selected = set(args.only or jobs)
    unknown = selected.difference(jobs)
    if unknown:
        parser.error(str(unknown))
    # A rerun of one downstream job may use the receipts and output in the same directory.
    finished = {}
    for name in selected:
        prerequisite = dependencies.get(name)
        if prerequisite and prerequisite not in selected:
            path = receipts / (prerequisite + ".json")
            if not path.is_file() or json.loads(path.read_text())["exit"]:
                parser.error(f"{name} requires a successful {prerequisite} run in {out}")
            finished[prerequisite] = 0

    pending = [name for name in jobs if name in selected]
    active = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        while pending or active:
            for name in pending.copy():
                prerequisite = dependencies.get(name)
                if prerequisite and prerequisite not in finished:
                    continue
                if prerequisite and finished[prerequisite]:
                    record = {"study": name, "exit": 1, "blocked_by": prerequisite}
                    (receipts / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                    finished[name] = 1
                elif len(active) < args.jobs:
                    active[pool.submit(run, name)] = name
                else:
                    break
                pending.remove(name)
            if active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    finished[active.pop(future)] = future.result()
    return int(any(finished[n] for n in selected))


if __name__ == "__main__":
    raise SystemExit(main())
