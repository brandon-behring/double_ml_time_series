"""panel_standard_error, _cluster_variance and group_codes (dml_ts/dml/_panel.py).

The cluster variances are checked against statsmodels' cluster-robust OLS covariance, and the
date HAC against statsmodels' HAC covariance of a mean, to 1e-12. With one regressor and no
intercept, statsmodels' CR1 factor G/(G-1) * (N-1)/(N-K) reduces to G/(G-1), the factor used here.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from dml_ts.dml._panel import _cluster_variance, group_codes, panel_standard_error

pytestmark = pytest.mark.tier1


def _regression(seed: int = 7, n_units: int = 12, n_dates: int = 9, drop: float = 0.15):
    """An unbalanced panel regression y = 2 x + e, with unit and date shocks in e."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_dates)
    date = np.tile(np.arange(n_dates), n_units)
    keep = rng.random(unit.size) > drop
    unit, date = unit[keep], date[keep]
    x = rng.normal(size=unit.size) + rng.normal(size=n_units)[unit]
    e = rng.normal(size=unit.size) + rng.normal(size=n_units)[unit] + rng.normal(size=n_dates)[date]
    y = 2.0 * x + e
    theta = x @ y / (x @ x)
    psi = x * (y - theta * x) / np.mean(x**2)
    return y, x, unit, date, psi


def _statsmodels_se(y, x, **cov):
    fit = sm.OLS(y, x[:, None]).fit(**cov)
    return float(np.sqrt(fit.cov_params()[0, 0]))


class TestAgainstStatsmodels:
    def test_unit_cluster(self):
        y, x, unit, date, psi = _regression()
        se, bandwidth = panel_standard_error(psi, unit, date, "unit_cluster")

        expected = _statsmodels_se(y, x, cov_type="cluster", cov_kwds={"groups": unit})
        assert se == pytest.approx(expected, rel=1e-12)
        assert bandwidth == 0

    def test_two_way_cluster(self):
        y, x, unit, date, psi = _regression()
        se, bandwidth = panel_standard_error(psi, unit, date, "two_way_cluster")

        groups = np.column_stack((unit, date))
        expected = _statsmodels_se(y, x, cov_type="cluster", cov_kwds={"groups": groups})
        assert se == pytest.approx(expected, rel=1e-12)
        assert bandwidth == 0

    def test_date_hac(self):
        _, _, unit, date, psi = _regression(n_dates=40)
        se, bandwidth = panel_standard_error(psi, unit, date, "date_hac")

        dates = np.unique(date)
        z = np.array([psi[date == d].sum() for d in dates]) * len(dates) / len(psi)
        fit = sm.OLS(z, np.ones_like(z)).fit(
            cov_type="HAC", cov_kwds={"maxlags": bandwidth, "use_correction": False}
        )
        assert bandwidth == int(len(z) ** (1 / 3)) > 0
        assert se == pytest.approx(float(np.sqrt(fit.cov_params()[0, 0])), rel=1e-12)

    def test_date_hac_follows_the_dates_sorted_order(self):
        _, _, unit, date, psi = _regression(n_dates=40)
        order = np.random.default_rng(3).permutation(len(psi))

        shuffled = panel_standard_error(psi[order], unit[order], date[order], "date_hac")
        labelled = panel_standard_error(
            psi, unit, pd.Timestamp("2020-01-01") + pd.to_timedelta(date, "D"), "date_hac"
        )

        assert shuffled == pytest.approx(
            panel_standard_error(psi, unit, date, "date_hac"), rel=1e-12
        )
        assert labelled == pytest.approx(
            panel_standard_error(psi, unit, date, "date_hac"), rel=1e-12
        )


class TestIdentifiersComparedAsGiven:
    def test_one_and_the_string_one_are_two_units(self):
        # np.asarray([1, "1", 2]) gives the strings "1", "1", "2": two units merged.
        codes = group_codes([1, "1", 2], 3, "unit")

        assert len(set(codes.tolist())) == 3

    def test_thirty_float_identifiers_are_thirty_units(self):
        labels = [1.0 + k / 100 for k in range(30)]

        assert len(set(group_codes(labels, 30, "unit").tolist())) == 30
        assert len(set(group_codes(np.array(labels), 30, "unit").tolist())) == 30

    def test_mixed_identifiers_give_the_standard_error_of_distinct_units(self):
        _, _, unit, date, psi = _regression(n_units=3)
        as_ints = panel_standard_error(psi, unit, date, "unit_cluster")
        mixed = np.array([1, "1", 2], dtype=object)[unit]

        merged = np.array(["1", "1", "2"])[unit]  # what np.asarray made of the identifiers

        # The codes come in another order, so the sums differ in the last bit only.
        assert panel_standard_error(psi, list(mixed), date, "unit_cluster")[0] == pytest.approx(
            as_ints[0], rel=1e-12
        )
        assert panel_standard_error(psi, merged, date, "unit_cluster")[0] != pytest.approx(
            as_ints[0], rel=1e-3
        )

    def test_equal_identifiers_are_one_unit(self):
        assert group_codes([1, 1.0, 2], 3, "unit").tolist() == [0, 0, 1]

    def test_codes_follow_the_sorted_identifiers_and_ignore_a_series_index(self):
        dates = pd.Series(
            pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]), index=[5, 4, 3]
        )

        assert group_codes(dates, 3, "date").tolist() == [2, 0, 1]
        assert group_codes(["b", "a", "b"], 3, "unit").tolist() == [1, 0, 1]

    def test_cluster_variance_does_not_depend_on_the_row_order(self):
        _, _, unit, date, psi = _regression()
        order = np.random.default_rng(1).permutation(len(psi))

        for method in ("unit_cluster", "two_way_cluster"):
            assert panel_standard_error(psi[order], unit[order], date[order], method)[
                0
            ] == pytest.approx(panel_standard_error(psi, unit, date, method)[0], rel=1e-12)


class TestRefusals:
    @pytest.mark.parametrize(
        ("labels", "message"),
        [
            ([1, None, 2], "Missing unit"),
            ([1, 2], "one identifier per observation"),
            (np.zeros((3, 2)), "one identifier per observation"),
            (5, "one identifier per observation"),
            ([1, [2], 3], "cannot be compared and sorted"),
        ],
    )
    def test_identifiers(self, labels, message):
        with pytest.raises(ValueError, match=message):
            group_codes(labels, 3, "unit")

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown panel inference method 'hac'"):
            panel_standard_error([1.0, 2.0], [0, 1], [0, 1], "hac")  # type: ignore[arg-type]

    @pytest.mark.parametrize("scores", [[1.0, np.nan, 2.0], [1.0, np.inf, 2.0], np.ones((3, 1))])
    def test_scores_that_are_not_a_finite_vector(self, scores):
        with pytest.raises(ValueError, match="scores must be a finite vector"):
            panel_standard_error(scores, [0, 1, 2], [0, 1, 2], "unit_cluster")

    def test_one_cluster(self):
        with pytest.raises(ValueError, match="at least 2 retained clusters"):
            panel_standard_error([1.0, 2.0, 3.0], ["a", "a", "a"], [0, 1, 2], "unit_cluster")

    def test_one_date(self):
        with pytest.raises(ValueError, match="at least 2 retained dates"):
            panel_standard_error([1.0, 2.0], [0, 1], [7, 7], "date_hac")

    @pytest.mark.parametrize("method", ["unit_cluster", "two_way_cluster", "date_hac"])
    def test_zero_scores(self, method):
        # A zero standard error is refused, not returned (Newey-West itself returns 0.0).
        with pytest.raises(ValueError, match="non-positive"):
            panel_standard_error(np.zeros(6), [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], method)

    def test_nonpositive_two_way_variance_is_reported(self):
        # Zero score sums in both dimensions and a positive cell variance: the
        # Cameron-Gelbach-Miller variance is negative. (Held test of the snapshot.)
        unit = np.array(["store A", "store A", "store B", "store B"], dtype=object)
        date = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="two_way_cluster variance"):
            panel_standard_error(np.array([1.0, -1.0, -1.0, 1.0]), unit, date, "two_way_cluster")

    def test_cluster_variance_formula(self):
        # Units {0, 0, 1}: sums 3 and 3, g = 2: 2 * (9 + 9) / 3**2 = 4.
        assert _cluster_variance(np.array([1.0, 2.0, 3.0]), np.array([[0], [0], [1]])) == 4.0
