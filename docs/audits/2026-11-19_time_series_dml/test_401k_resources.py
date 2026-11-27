import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dispatch_401k as d

LARGE = "all-LATE-k5-r001"
OTHER5 = "all-interactive-k5-r003"
SMALL = "all-LATE-k2-r002"
SMALL2 = "all-plinear-k2-r004"


class BudgetTests(unittest.TestCase):
    def test_minima_and_high_water(self):
        self.assertEqual(d.fit_budget("LATE-k5", 2.2, {}), 3 * d.GIB)
        self.assertEqual(d.fit_budget("LATE-k5", 3.5, {}), 3.5 * d.GIB)
        self.assertEqual(d.fit_budget("interactive-k5", 2.2, {}), math.ceil(2.4 * d.GIB))
        self.assertEqual(d.fit_budget("unknown", 2.2, {}), math.ceil(2.2 * d.GIB))
        self.assertEqual(
            d.fit_budget("LATE-k5", 2.2, {"LATE-k5": 3.2}), math.ceil(1.10 * 3.2 * d.GIB)
        )

    def test_unclaimed_multiworker_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = root / "9"
            proc.mkdir()
            argv = [
                b"R",
                b"--file=archive/replicate_401k.R",
                b"--args",
                b"/source",
                b"/output",
                b"1",
                b"6",
                b"all",
            ]
            (proc / "cmdline").write_bytes(b"\0".join(argv))
            sample = (0, 0, 50 * d.GIB, {9: (0, d.GIB)})
            self.assertEqual(d.worker_budgets(sample, {}, {}, root, 2.2, {}, root)[9], 3 * d.GIB)
            self.assertGreaterEqual(
                d.worker_budgets(sample, {}, {}, root, 2.2, {"LATE-k5": 3.5}, root)[9], 3.85 * d.GIB
            )
            argv[-2] = b"1"
            (proc / "cmdline").write_bytes(b"\0".join(argv))
            self.assertEqual(
                d.worker_budgets(sample, {}, {}, root, 2.2, {}, root)[9], math.ceil(2.2 * d.GIB)
            )

    def test_skip_then_debit_each_fit(self):
        pending = [LARGE, OTHER5, SMALL, SMALL2]
        self.assertEqual(d.choose_jobs(pending, 4, math.floor(2.21 * d.GIB), 2.2, {}), [SMALL])
        self.assertEqual(
            d.choose_jobs([LARGE, SMALL, SMALL2], 4, math.floor(5.0 * d.GIB), 2.2, {}), [LARGE]
        )
        self.assertEqual(
            d.choose_jobs(pending, 4, math.floor(5.41 * d.GIB), 2.2, {}), [LARGE, OTHER5]
        )
        self.assertEqual(d.choose_jobs(pending, 1, 100 * d.GIB, 2.2, {}), [LARGE])
        self.assertEqual(d.choose_jobs(pending, 0, 100 * d.GIB, 2.2, {}), [])

    def test_live_unseen_and_external_cpu(self):
        tick = os.sysconf("SC_CLK_TCK")
        args = SimpleNamespace(worker_gib=2.2, reserve_gib=16, target=7, batch=5)
        prev = (0, 0, 40 * d.GIB, {1: (0, 0), 2: (0, 0)})
        curr = (1, 3 * tick, 26 * d.GIB, {1: (tick, d.GIB), 2: (tick, d.GIB // 2)})
        budgets = {1: 3 * d.GIB, 2: math.ceil(2.2 * d.GIB)}
        slots, stats = d.capacity(prev, curr, args, budgets, unseen=[3 * d.GIB])
        # 10 GiB after system reserve; 2 + 1.7 GiB worker growth + 3 GiB unseen.
        self.assertAlmostEqual(stats["remaining_fit_budget_bytes"] / d.GIB, 3.3)
        self.assertEqual(stats["other_busy_cores"], 1)
        self.assertEqual(stats["cpu_slots"], 3)
        self.assertEqual(slots, 1)
        self.assertEqual(
            d.choose_jobs([LARGE, SMALL], slots, stats["remaining_fit_budget_bytes"], 2.2, {}),
            [LARGE],
        )
        self.assertEqual(
            d.capacity(prev, (1, 7 * tick, *curr[2:]), args, budgets, [3 * d.GIB])[0], 0
        )

    def test_pid_resolution_and_second_pass_on_high_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            jobs.mkdir()
            legacy = root / "old"
            legacy.mkdir()
            durable = root / "durable"
            durable.mkdir()
            for pid in [1, 2, 3, 4, 5, 6, 7, 8]:
                proc = root / str(pid)
                proc.mkdir()
                (proc / "status").write_text("VmHWM:\t1048576 kB\n")

            def job(pid, name, contents):
                file = jobs / (name + ".csv")
                file.write_text(contents)
                (root / str(pid) / "environ").write_bytes(
                    b"OTHER=filtered\0DML_401K_JOBS=" + os.fsencode(file) + b"\0"
                )

            job(2, LARGE, "panel,folds,partition\nLATE,5,1\n")
            job(4, OTHER5, "panel,folds,partition\ninteractive,5,3\nplinear,5,3\n")
            job(5, SMALL, "panel,folds,partition\nplinear,2,2\n")

            def claim(path, name, pid, host):
                c = path / (".claim-" + name)
                c.mkdir()
                (c / "owner").write_text(f"{host}\n{pid}\n")

            claim(legacy, "Forest-LATE-k5-r001", 1, os.uname().nodename)
            claim(durable, LARGE, 6, "remote-not-this-host")
            claim(durable, "bad-name", 7, os.uname().nodename)
            (root / "2" / "status").write_text("VmHWM: 4194304 kB\n")  # 4 GiB on core 2
            current = (1, 0, 100 * d.GIB, {i: (0, int(0.5 * d.GIB)) for i in range(1, 9)})
            # PID 3 is in active without a claim; PID 8 claims the wrong smaller job but
            # its canonical one-row input still identifies the larger fit.
            (root / "8" / "environ").write_bytes((root / "2" / "environ").read_bytes())
            claim(durable, SMALL2, 8, os.uname().nodename)
            owners = d.claim_jobs([legacy, durable])
            self.assertEqual(set(owners), {1, 8})
            peaks = {}
            out = d.worker_budgets(current, {3: OTHER5}, owners, jobs, 2.2, peaks, root)
            self.assertEqual(out[1], out[2])
            self.assertEqual(out[2], out[8])
            self.assertEqual(out[1], math.ceil(4.4 * d.GIB))
            self.assertEqual(out[3], math.ceil(2.4 * d.GIB))
            self.assertTrue(all(out[i] == math.ceil(2.2 * d.GIB) for i in [4, 5, 6, 7]))
            self.assertEqual(peaks["LATE-k5"], 4)
            # The peak propagates into new jobs and persists in the high_water map.
            self.assertEqual(d.choose_jobs([LARGE, SMALL], 2, 3.5 * d.GIB, 2.2, peaks), [SMALL])


if __name__ == "__main__":
    unittest.main()
