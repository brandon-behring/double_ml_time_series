"""Run independent, checksum-pinned empirical replications from their original inputs."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLICATIONS = {
    "italy": ("sequential/italy.py",),
    "water": ("taxes_water/replay.py", "water"),
    "turnout": ("turnout/replay.py",),
    "wolfers": ("panels/replay.py", "wolfers"),
    "beer": ("taxes_water/replay.py", "beer"),
    "dahl": ("panels/replay.py", "dahl"),
    "drink-less": ("sequential/drink_less.py",),
    "mybp": ("sequential/mybp.py",),
    "sicily": ("time_series/replay.py", "sicily"),
    "ramey": ("time_series/replay.py", "ramey"),
}
QUESTIONS = {
    "retail-timing": ("questions/retail.py",),
    "electricity-shifting": ("questions/energy.py",),
    "wage-employment": ("questions/minimum_wage.py",),
    "daycare-fines": ("questions/daycare.py",),
    "reminder-habituation": ("questions/reminders.py",),
}
FORWARD = {
    "reminder-cohorts": ("forward/reminders.py",),
    "gdp-vintages": ("forward/gdp.py",),
    "pennsylvania-bonus": ("forward/pennsylvania.py",),
    "canvassing-elections": ("forward/canvassing.py",),
    "progresa-schooling": ("forward/progresa.py",),
    "fertilizer-timing": ("forward/fertilizer.py",),
    "microcredit-sites": ("forward/microcredit.py",),
    "ban-the-box": ("forward/ban_box.py",),
    "retail-recommendations": ("forward/open_bandit.py",),
    "bracero-wages": ("forward/bracero.py",),
    "teacher-pay": ("forward/teacher_pay.py",),
    "weatherization": ("forward/weatherization.py",),
}
ROLLING = {
    "rolling-reminders": ("forward/reminder_rolling.py",),
    "rolling-calibration": ("forward/rolling_calibration.py",),
    "rolling-bootstrap": ("forward/reminder_bootstrap_calibration.py",),
    "rolling-controls": ("forward/rolling_controls.py",),
    "jtpa-holdout": ("forward/jtpa_holdout.py",),
}
PROSPECTIVE = {
    "shrinkage-replay": ("forward/shrinkage_replay.py",),
    "cricket-toss": ("forward/prospective_cricket.py",),
    "gss-forms": ("forward/prospective_gss.py",),
}
METHOD_VALIDATION = {
    "classical-econometrics": ("../classical/replay.py",),
    "classical-impulse": ("../classical/impulse.py",),
    "paired-time-splits": ("../paired/experiment.py",),
    "bangalore-pricing": ("../field/bangalore/replay.py",),
    "london-tariffs": ("../field/london/replay.py",),
}
STUDIES = {**REPLICATIONS, **QUESTIONS, **FORWARD, **ROLLING, **PROSPECTIVE, **METHOD_VALIDATION}
# Relative CPU work in the complete suite. Fitting the three learners on 4,000
# paths dominates the linear impulse fits and the meter-archive scan.
PARALLEL_WORK = {"paired-time-splits": 12, "classical-impulse": 1, "london-tariffs": 2}


def study_budgets(studies, budget):
    slots = dict.fromkeys(studies, 1)
    parallel = [study for study in studies if study in PARALLEL_WORK]
    if parallel:
        for _ in range(budget - len(studies)):
            study = max(parallel, key=lambda s: PARALLEL_WORK[s] / slots[s])
            slots[study] += 1
    return slots


def run_study(study, cache_dir, workers_per_study=1):
    script, *args = STUDIES[study]
    if study in PARALLEL_WORK:
        args.extend(["--workers", str(workers_per_study)])
    env = os.environ.copy()
    root = str(HERE.parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DML_N_JOBS",
    ):
        env[variable] = "1"
    return subprocess.run(
        [sys.executable, str(HERE / script), *args, "--cache-dir", str(cache_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "studies", nargs="*", metavar="STUDY", help="Select by name: " + ", ".join(STUDIES)
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.getenv("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache" / "dml-ts" / "external")
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Worker budget across studies and simulation paths (default 4)",
    )
    suites = {
        "replications": REPLICATIONS,
        "questions": QUESTIONS,
        "forward": FORWARD,
        "rolling": ROLLING,
        "prospective": PROSPECTIVE,
        "method-validation": METHOD_VALIDATION,
        "all": STUDIES,
    }
    parser.add_argument("--suite", choices=suites, help="Defaults to the original ten replications")
    args = parser.parse_args()
    if args.studies and args.suite:
        parser.error("select individual studies or one --suite")
    studies = list(dict.fromkeys(args.studies or suites[args.suite or "replications"]))
    unknown = set(studies).difference(STUDIES)
    if unknown:
        parser.error(f"unknown studies: {', '.join(sorted(unknown))}")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cores = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    workers = min(args.jobs, len(studies), cores)
    slots = study_budgets(studies, min(args.jobs, cores))
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_study, study, args.cache_dir.resolve(), slots[study]): study
            for study in studies
        }
        for future in as_completed(futures):
            result = future.result()
            study = futures[future]
            print(
                f"[{study}] {'PASS' if result.returncode == 0 else 'FAIL'}\n{result.stdout}",
                flush=True,
            )
            if result.returncode:
                failed.append(study)
    print(f"{len(studies) - len(failed)}/{len(studies)} studies passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
