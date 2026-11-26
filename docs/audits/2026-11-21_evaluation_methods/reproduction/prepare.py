"""Create a fresh working-tree snapshot and start the resource-aware historical refit."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHOR = "DMLonGitHub-b91cbf96c01eccd73367fbd6601ecdd7aa78403b"
OLD = Path("docs/audits/2026-11-19_time_series_dml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--r-assets", type=Path, default=ROOT / "results/audit_2026-11-19/401k")
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()
    run = args.run.resolve()
    assets = args.r_assets.resolve()
    for file in (AUTHOR, "hdm17", "rlib"):
        if not (assets / file).is_dir():
            parser.error(f"Missing pinned R input {assets / file}")
    run.mkdir(parents=True, exist_ok=False)
    snap = run / "snapshot"
    names = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
    )
    paths = sorted(set(names.decode().split("\0")) - {""})
    digest = hashlib.sha256()
    for path in paths:
        src = ROOT / path
        if not src.is_file():
            continue
        dest = snap / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        digest.update(path.encode() + b"\0" + hashlib.sha256(dest.read_bytes()).digest())
    (snap / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    fresh = run / "401k"
    fresh.mkdir()
    for k in (AUTHOR, "hdm17", "rlib"):
        (fresh / k).symlink_to(assets / k, target_is_directory=True)
    (fresh / "reserved-partitions.txt").write_text("")
    (fresh / "empty-legacy").mkdir()
    cmd = [
        str(snap / ".venv/bin/python"),
        str(snap / HERE.relative_to(ROOT) / "shared_dispatch.py"),
        str(snap / OLD / "dispatch_401k.py"),
        "--root",
        str(fresh),
        "--legacy",
        str(fresh / "empty-legacy"),
        "--target",
        str(args.target),
        "--reserve-gib",
        "16",
        "--poll",
        "5",
        "--render-interval",
        "300",
    ]
    with (fresh / "launch.log").open("ab") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=snap,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (run / "launch.json").write_text(
        json.dumps(
            {
                "snapshot": str(snap),
                "files": len(paths),
                "snapshot_digest": digest.hexdigest(),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
                .decode()
                .strip(),
                "cmd": cmd,
                "pid": proc.pid,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Started dispatcher {proc.pid} in {run}")


if __name__ == "__main__":
    main()
