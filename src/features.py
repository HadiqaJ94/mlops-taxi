"""Feature engineering. Used by BOTH training and the API, so there is no train/serve skew.

Setup: a row at hour t has target = pickups in hour t, and lag_k = pickups in hour t-k.
So predicting "the next hour" = predicting the row whose lags end at the last observed hour.
"""
from collections.abc import Sequence

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CALENDAR_FEATURES = ["hour", "dayofweek", "is_weekend", "is_holiday"]
_HOLIDAYS = USFederalHolidayCalendar().holidays(start="2015-01-01", end="2035-12-31")


def lag_names(lags: Sequence[int]) -> list[str]:
    return [f"lag_{k}" for k in lags]


def feature_columns(lags: Sequence[int]) -> list[str]:
    """Order matters: this is the model's input schema."""
    return ["zone_id", *CALENDAR_FEATURES, *lag_names(lags)]


def calendar_features(timestamps) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    return pd.DataFrame(
        {
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "is_holiday": ts.normalize().isin(_HOLIDAYS).astype(int),
        }
    )


def build_features(df: pd.DataFrame, lags: Sequence[int], drop_na: bool = True) -> pd.DataFrame:
    """Training-time features from the zero-filled (zone, hour) table."""
    df = df.sort_values(["zone_id", "pickup_hour"]).reset_index(drop=True)

    # shift() works on rows, so every zone must have a gap-free hourly series
    gaps = df.groupby("zone_id")["pickup_hour"].diff().dropna()
    if not (gaps == pd.Timedelta("1h")).all():
        raise ValueError("Hourly grid has gaps: lags would be wrong. Zero-fill the grid first.")

    cal = calendar_features(df["pickup_hour"])
    for c in CALENDAR_FEATURES:
        df[c] = cal[c].to_numpy()

    grouped = df.groupby("zone_id")["target"]
    for k in lags:
        df[f"lag_{k}"] = grouped.shift(k)

    if drop_na:  # first max(lags) hours of each zone have no history
        df = df.dropna(subset=lag_names(lags)).reset_index(drop=True)
    return df


def serving_features(
    zone_id: int, target_hour, recent_counts: Sequence[float], lags: Sequence[int]
) -> pd.DataFrame:
    """One-row feature frame for the API.

    recent_counts: pickups per hour for this zone, oldest -> newest, the LAST value being
    the hour just before `target_hour`. Needs at least max(lags) values.
    """
    if len(recent_counts) < max(lags):
        raise ValueError(f"Need at least {max(lags)} hourly values, got {len(recent_counts)}")
    row = {"zone_id": zone_id, **calendar_features([target_hour]).iloc[0].to_dict()}
    for k in lags:
        row[f"lag_{k}"] = float(recent_counts[-k])
    return pd.DataFrame([row])[feature_columns(lags)]
