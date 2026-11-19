import numpy as np
import pandas as pd

from src.features import build_features, feature_columns, serving_features

LAGS = [1, 2, 3, 24, 168]


def make_grid(n_hours=400, zones=(10, 20)):
    hours = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.concat(
        [pd.DataFrame({"zone_id": z, "pickup_hour": hours, "target": rng.poisson(20, n_hours)}) for z in zones]
    )


def test_serving_matches_training_features():
    """The API must build exactly the same features as training."""
    df = make_grid()
    feats = build_features(df, LAGS)
    row = feats[(feats.zone_id == 20)].iloc[200]

    zone_series = df[df.zone_id == 20].sort_values("pickup_hour").reset_index(drop=True)
    pos = zone_series.index[zone_series.pickup_hour == row.pickup_hour][0]
    history = zone_series["target"].iloc[:pos].tolist()  # everything BEFORE the target hour

    served = serving_features(20, row.pickup_hour, history, LAGS)
    pd.testing.assert_series_equal(
        served.iloc[0].astype(float), row[feature_columns(LAGS)].astype(float), check_names=False
    )


def test_lags_are_per_zone():
    df = make_grid()
    feats = build_features(df, [1])
    z10 = df[df.zone_id == 10].sort_values("pickup_hour")["target"].to_numpy()
    f10 = feats[feats.zone_id == 10].sort_values("pickup_hour")
    assert (f10["lag_1"].to_numpy() == z10[:-1][-len(f10):]).all()


def test_gap_detected():
    df = make_grid().drop(index=5)  # remove one hour in one zone
    try:
        build_features(df, LAGS)
    except ValueError:
        return
    raise AssertionError("gap should raise")
