"""Tests for src/data/validate.py (T4): a valid grid passes, each kind of broken data fails.

Synthetic data: 2 zones, January (reference) and February (drift) 2020. No TLC files needed.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_params
from src.data.validate import validate

ZONES = [13, 237]


@pytest.fixture(scope="module")
def params():
    p = copy.deepcopy(load_params())
    p["data"]["reference_months"] = ["2020-01"]
    p["data"]["drift_months"] = ["2020-02"]
    return p


@pytest.fixture
def demand():
    """A valid zone x hour grid, with the exact types preprocess.py writes."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2020-01-01", "2020-03-01", freq="h", inclusive="left").astype("datetime64[us]")
    df = pd.DataFrame({
        "zone_id": np.repeat(ZONES, len(hours)).astype("int16"),
        "pickup_hour": np.tile(hours, len(ZONES)),
    })
    df["target"] = rng.poisson(20, len(df)).astype("int32")
    df.loc[df.index % 50 == 0, "target"] = 0  # some empty hours
    empty = df["target"] == 0
    df["mean_fare"] = np.where(empty, np.nan, rng.uniform(5, 40, len(df)))
    df["mean_distance"] = np.where(empty, np.nan, rng.uniform(0.5, 10, len(df)))
    df["month"] = df["pickup_hour"].dt.strftime("%Y-%m").astype("str")
    df["period"] = np.where(df["month"] == "2020-01", "reference", "drift")
    df["period"] = df["period"].astype("str")
    return df


def failed_checks(report) -> str:
    return " | ".join(f"{f['column']}: {f['check']} {f['examples']}" for f in report["failures"])


def test_valid_grid_passes(demand, params):
    report = validate(demand, ZONES, params)
    assert report["valid"], failed_checks(report)
    assert report["rows"] == len(demand) and report["zones"] == 2


def test_negative_target_fails(demand, params):
    demand.loc[5, "target"] = -1
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "target" in failed_checks(report)


def test_missing_hour_fails(demand, params):
    """A gap in the grid would make every lag after it wrong."""
    report = validate(demand.drop(index=100), ZONES, params)
    assert not report["valid"]
    assert "hours" in failed_checks(report)


def test_duplicate_row_fails(demand, params):
    report = validate(pd.concat([demand, demand.iloc[[10]]], ignore_index=True), ZONES, params)
    assert not report["valid"]


def test_unknown_zone_fails(demand, params):
    demand["zone_id"] = demand["zone_id"].replace(13, 999).astype("int16")
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "zone_id" in failed_checks(report)


def test_broken_fare_column_fails(demand, params):
    """The incident 'broken fare column': fares in cents instead of dollars."""
    demand["mean_fare"] = demand["mean_fare"] * 100
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "mean_fare outside" in failed_checks(report)


def test_a_few_absurd_fares_are_tolerated(demand, params):
    """Real TLC files contain a handful of absurd fares; they must not block the pipeline."""
    demand.loc[demand["mean_fare"].notna().idxmax(), "mean_fare"] = 35_000.0
    assert validate(demand, ZONES, params)["valid"]


def test_fare_on_an_empty_hour_fails(demand, params):
    first_empty = demand.index[demand["target"] == 0][0]
    demand.loc[first_empty, "mean_fare"] = 10.0
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "empty exactly when target is 0" in failed_checks(report)


def test_wrong_period_fails(demand, params):
    demand.loc[0, "period"] = "drift"  # a January row labelled as drift
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "period" in failed_checks(report)


def test_extra_column_fails(demand, params):
    """strict=True: e.g. cbd_congestion_fee appearing in 2025 files must be noticed."""
    demand["cbd_congestion_fee"] = 0.75
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "cbd_congestion_fee" in failed_checks(report)


def test_wrong_type_fails(demand, params):
    demand["target"] = demand["target"].astype("float64")
    report = validate(demand, ZONES, params)
    assert not report["valid"]
    assert "target" in failed_checks(report)
