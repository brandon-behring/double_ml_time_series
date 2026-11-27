"""Regressions for sharing admission with active 401(k) fits."""

import json
import os
from unittest.mock import patch

import lasso_resources


def test_future_growth_and_combined_cpu_limit_the_live_pool():
    now = 10
    hz = os.sysconf("SC_CLK_TCK")
    then = dict.fromkeys(range(1, 19), (0, 0.25, "R", 0.25))
    current = dict.fromkeys(then, (now * hz, 0.25, "R", 0.25))
    with (
        patch.object(lasso_resources, "memory_available", return_value=37.25),
        patch.object(lasso_resources, "dispatch", return_value=(14.75, 50, True, True)),
    ):
        selected, _ = lasso_resources.select(
            (0, 0, then), (now, 51 * now * hz, current), 18, 18, None
        )
        assert len(selected) == 10  # 2.5 GiB shared growth limit / 0.25 GiB future growth.
        selected, _ = lasso_resources.select(
            (0, 0, then), (now, 66 * now * hz, current), 18, 18, None
        )
        assert len(selected) == 2  # 48 busy cores outside Python.


def test_r_new_launch_sample_is_held_until_the_growth_budget_includes_it(tmp_path):
    path = tmp_path / "status.json"
    status = {
        "complete": False,
        "target_busy_cores": 50,
        "launches_this_sample": 1,
        "resources": {"reserved_worker_growth_gib": 15},
    }
    path.write_text(json.dumps(status))
    assert lasso_resources.dispatch(path) == (15, 50, False, True)
    status["launches_this_sample"] = 0
    path.write_text(json.dumps(status))
    assert lasso_resources.dispatch(path) == (15, 50, True, True)


def test_rotated_lease_can_resume_an_older_unfinished_task():
    workers = {100: (0, 0.25, "S", 0.25), 101: (0, 0.25, "T", 0.25)}
    with (
        patch.object(lasso_resources, "memory_available", return_value=34),
        patch.object(lasso_resources, "dispatch", return_value=(17.7, 50, True, True)),
    ):
        selected, _ = lasso_resources.select(
            (0, 0, workers), (10, 0, workers), 18, 16, None, priority={101: 0, 100: 1}
        )
    assert selected == {101}
