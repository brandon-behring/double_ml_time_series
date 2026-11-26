"""Verify village assignment and independent Stata/Fortran checks of the original archive."""

import importlib
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

raw = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.progresa_raw"
)


@pytest.mark.tier1
def test_localities_preserve_assignment_across_reused_codes_and_select_eligible_children():
    rows = []
    for state, arm in [(1, 1), (2, 2)]:
        for local in (1, 2):
            for sex in (0, 1):
                for age in (12, 15):
                    rows.append(
                        [state, 1, local, 98, 1, age, sex, arm, int(state == 1 or age == 12)]
                    )
    data = pd.DataFrame(
        rows, columns=[*raw.LOCATION, "year", "pobre", "agefin", "sexr", "control", "inschoolr"]
    )
    corrupt = data.iloc[[0]].assign(pobre=0, control=2, inschoolr=0)
    result = raw.contrasts(pd.concat([data, corrupt], ignore_index=True))
    assert result["girls"]["effect_pp"] == pytest.approx(50)
    assert result["boys"]["villages"] == 4
    with pytest.raises(ValueError, match="one randomized"):
        raw.contrasts(pd.concat([data, corrupt.assign(pobre=1)], ignore_index=True))


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(
    os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in original archive replay"
)
def test_original_schooling_data_and_available_compiler(tmp_path):
    compiler = shutil.which("gfortran") is not None
    result = raw.run(Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path)), check_fortran=compiler)
    assert result["groups"]["girls"]["village_se_pp"] == pytest.approx(5.3305044356)
    assert result["groups"]["boys"]["village_se_pp"] == pytest.approx(4.9721903519)
    assert result["estimation_1998_controls"] == 3923
    if compiler:
        assert result["fortran"]["ascii_rows"] == 19500
