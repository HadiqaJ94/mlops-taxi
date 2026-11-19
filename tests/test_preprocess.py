import pandas as pd

from src.data.preprocess import aggregate_hourly, clean_trips, complete_grid


def raw():
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2020-01-01 00:10", "2020-01-01 00:50", "2020-01-01 02:05",
                                     "2009-01-01 00:00", "2020-02-01 00:00", None],
            "PULocationID": [5, 5, 5, 5, 5, 5],
            "fare_amount": [10.0, 12.0, 8.0, 5.0, 5.0, 5.0],
            "trip_distance": [1.0, 2.0, 1.5, 1.0, 1.0, 1.0],
        }
    )


def test_out_of_month_rows_are_dropped():
    clean, stats = clean_trips(raw(), "2020-01", 1, 263)
    assert len(clean) == 3
    assert stats["rows_raw"] == 6


def test_zero_hours_are_filled():
    clean, _ = clean_trips(raw(), "2020-01", 1, 263)
    full = complete_grid(aggregate_hourly(clean), [5], ["2020-01"])
    assert len(full) == 31 * 24  # every hour present
    by_hour = full.set_index("pickup_hour")["target"]
    assert by_hour["2020-01-01 00:00"] == 2
    assert by_hour["2020-01-01 01:00"] == 0  # the hour with no trip is an explicit 0
    assert by_hour["2020-01-01 02:00"] == 1
