"""Tests for src/drift.py (T11): the PSI helpers and the drift report, on synthetic data.

No MLflow and no TLC files: the report runs on a small synthetic zone x hour grid, with a stub
model that predicts the last observed hour (lag_1).
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_params
from src.drift import bin_edges, build_report, proportions, psi, psi_from_profile
from src.features import build_features

RNG = np.random.default_rng(0)


# ---------- PSI ----------

def test_psi_same_distribution_is_near_zero():
    ref = RNG.poisson(100, 5000)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, RNG.poisson(100, 5000)) < 0.1  # new sample, same law: stable


def test_psi_large_shift_is_drift():
    ref = RNG.poisson(100, 5000)
    assert psi(ref, ref * 3) > 0.25    # the lab's "amount x 3" incident
    assert psi(ref, ref * 0.05) > 0.25  # demand collapse


def test_psi_handles_repeated_values():
    """Many zeros make several deciles equal; the bins merge instead of breaking."""
    ref = np.concatenate([np.zeros(4000), RNG.poisson(50, 1000)])
    edges = bin_edges(ref)
    assert len(edges) < 12  # fewer than 10 bins after merging duplicate deciles
    assert np.isfinite(psi(ref, np.zeros(500)))


def test_psi_from_profile_matches_psi():
    """The API gauge (T10) computes the profile once and reuses it: same number."""
    ref, cur = RNG.poisson(100, 5000), RNG.poisson(80, 1000)
    edges = bin_edges(ref)
    assert psi_from_profile(edges, proportions(ref, edges), cur) == pytest.approx(psi(ref, cur))


def test_psi_ignores_nan_and_rejects_empty():
    ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    assert np.isfinite(psi(ref, [1.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        psi(ref, [np.nan])


# ---------- report ----------

class LastHourModel:
    """Stub model: predicts the count of the previous hour."""

    def predict(self, features):
        return features["lag_1"].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def feats():
    """3 zones, hourly, Jan to March 2020. Demand ~100 per hour, then collapses to ~10 in March."""
    hours = pd.date_range("2020-01-01", "2020-03-31 23:00", freq="h")
    frames = []
    for zone in (1, 2, 3):
        drift = hours >= "2020-03-01"
        frames.append(pd.DataFrame({
            "zone_id": zone,
            "pickup_hour": hours,
            "target": np.where(drift, RNG.poisson(10, len(hours)), RNG.poisson(100, len(hours))),
            "mean_fare": RNG.normal(12, 2, len(hours)),
            "mean_distance": RNG.normal(3, 0.5, len(hours)),
            "period": np.where(drift, "drift", "reference"),
        }))
    return build_features(pd.concat(frames, ignore_index=True), load_params()["features"]["lags"])


def test_report_on_a_reference_week_finds_no_drift(feats):
    report = build_report(feats, "2020-02-10", "2020-02-17", load_params(), LastHourModel(), "test")
    assert report["drift"] is False
    assert report["reasons"] == []
    assert all(f["status"] == "ok" for f in report["features"].values())


def test_report_on_the_collapse_finds_drift_on_every_layer(feats):
    report = build_report(feats, "2020-03-09", "2020-03-16", load_params(), LastHourModel(), "test")
    assert report["drift"] is True
    assert report["features"]["lag_1"]["status"] == "drift"         # input
    assert report["features"]["mean_fare"]["status"] == "ok"        # fares did not change
    assert report["prediction"]["status"] == "drift"                # output
    assert report["model_version"] == "test"
    assert report["performance"]["weekly"][0]["week_start"] == "2020-03-09"  # weeks start on Monday


def test_report_without_model_checks_inputs_only(feats):
    report = build_report(feats, "2020-03-09", "2020-03-16", load_params())
    assert "prediction" not in report and "performance" not in report
    assert report["drift"] is True


def test_report_rejects_an_empty_window(feats):
    with pytest.raises(ValueError):
        build_report(feats, "2021-01-01", "2021-01-08", load_params())
