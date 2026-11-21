"""Data validation (T4): check demand.parquet before anything trains on it.

DVC stage `validate`, between preprocess and train. If a rule fails, the stage exits with code 1,
`dvc repro` stops, and no model is trained on bad data. The report lists every failure (pandera
lazy mode), not only the first one.

Run:  uv run python -m src.data.validate      (or uv run dvc repro, which runs it before train)

Hard rules (the model or the lags would be wrong without them):
- exact columns and types; zone_id is one of the 50 zones in zones.json
- pickup_hour is on the hour; one row per (zone, hour); every zone has every hour (gap-free grid,
  because the lags are computed with shift() on rows)
- target >= 0; month matches pickup_hour; period is "reference" exactly for the reference months
- mean_fare and mean_distance are empty exactly when the hour had no trip
Tolerated noise (real TLC files have a few absurd fares): at most `max_outlier_share` of the
fares and distances outside their range. A broken column (x100, all negative) fails.
"""
import json
import logging
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from src.config import DEMAND_PATH, PROCESSED_DIR, load_params

log = logging.getLogger(__name__)

ZONES_PATH = PROCESSED_DIR / "zones.json"
REPORT_PATH = PROCESSED_DIR / "validation_report.json"


def expected_hours(months: list[str]) -> pd.DatetimeIndex:
    """Every hour of the given months, as preprocess.complete_grid builds them."""
    parts = [pd.date_range(f"{m}-01", pd.Timestamp(f"{m}-01") + pd.offsets.MonthBegin(1),
                           freq="h", inclusive="left") for m in months]
    return parts[0].append(parts[1:])


def outlier_share(s: pd.Series, low: float, high: float) -> float:
    s = s.dropna()
    return float(((s < low) | (s > high)).mean()) if len(s) else 0.0


def demand_schema(zones: list[int], params: dict) -> pa.DataFrameSchema:
    data, rules = params["data"], params["validate"]
    ref_months = data["reference_months"]
    months = ref_months + data["drift_months"]
    hours = expected_hours(months)
    max_share = rules["max_outlier_share"]
    fare_lo, fare_hi = rules["fare_range"]
    dist_lo, dist_hi = rules["distance_range"]

    def gap_free(df: pd.DataFrame) -> bool:
        """Every zone has exactly the expected hours: no gap, no extra hour."""
        counts = df.groupby("zone_id")["pickup_hour"].agg(["size", "min", "max"])
        return bool((counts["size"] == len(hours)).all()
                    and (counts["min"] == hours[0]).all() and (counts["max"] == hours[-1]).all())

    return pa.DataFrameSchema(
        {
            "zone_id": pa.Column("int16", pa.Check.isin(zones), description="TLC zone, one of zones.json"),
            "pickup_hour": pa.Column(
                "datetime64[us]",
                pa.Check(lambda s: s == s.dt.floor("h"), error="pickup_hour must be on the hour"),
            ),
            "target": pa.Column("int32", pa.Check.ge(0), description="pickups in that hour"),
            "mean_fare": pa.Column(float, nullable=True),
            "mean_distance": pa.Column(float, nullable=True),
            "month": pa.Column(str, pa.Check.isin(months)),
            "period": pa.Column(str, pa.Check.isin(["reference", "drift"])),
        },
        checks=[
            pa.Check(gap_free, error=f"each zone must have all {len(hours)} hours of {months[0]}..{months[-1]}"),
            pa.Check(lambda df: df["month"] == df["pickup_hour"].dt.strftime("%Y-%m"),
                     error="month must match pickup_hour"),
            pa.Check(lambda df: (df["period"] == "reference") == df["month"].isin(ref_months),
                     error=f"period must be 'reference' exactly for {ref_months}"),
            pa.Check(lambda df: df["mean_fare"].isna() == (df["target"] == 0),
                     error="mean_fare must be empty exactly when target is 0"),
            pa.Check(lambda df: df["mean_distance"].isna() == (df["target"] == 0),
                     error="mean_distance must be empty exactly when target is 0"),
            pa.Check(lambda df: outlier_share(df["mean_fare"], fare_lo, fare_hi) <= max_share,
                     error=f"more than {max_share:.0%} of mean_fare outside [{fare_lo}, {fare_hi}]"),
            pa.Check(lambda df: outlier_share(df["mean_distance"], dist_lo, dist_hi) <= max_share,
                     error=f"more than {max_share:.0%} of mean_distance outside [{dist_lo}, {dist_hi}]"),
        ],
        unique=["zone_id", "pickup_hour"],
        strict=True,   # an unexpected column is a failure too
        coerce=False,  # never silently fix the types: preprocess must produce them
        name="demand",
    )


def validate(df: pd.DataFrame, zones: list[int], params: dict) -> dict:
    """Validate and return a report. Pure: no files. report["valid"] is False on any failure."""
    rules = params["validate"]
    report = {
        "rows": len(df),
        "zones": int(df["zone_id"].nunique()) if "zone_id" in df else 0,
        "mean_fare_outlier_share": outlier_share(df.get("mean_fare", pd.Series(dtype=float)), *rules["fare_range"]),
        "mean_distance_outlier_share": outlier_share(df.get("mean_distance", pd.Series(dtype=float)),
                                                     *rules["distance_range"]),
    }
    try:
        demand_schema(zones, params).validate(df, lazy=True)
    except SchemaErrors as err:
        # one entry per failed rule: how many values broke it, and the first few of them
        report["valid"] = False
        report["failures"] = [
            {"column": None if pd.isna(col) else str(col), "check": str(check), "count": len(g),
             "examples": [str(v) for v in g["failure_case"].head(3)]}
            for (col, check), g in err.failure_cases.groupby(["column", "check"], dropna=False)
        ]
        return report
    report["valid"] = True
    report["failures"] = []
    return report


def main() -> int:
    params = load_params()
    zones = json.loads(ZONES_PATH.read_text())
    report = validate(pd.read_parquet(DEMAND_PATH), zones, params)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    if not report["valid"]:
        for f in report["failures"]:
            log.error("%s: %s (%d failing, e.g. %s)", f["column"] or "dataframe", f["check"], f["count"],
                      ", ".join(f["examples"]))
        log.error("demand.parquet is INVALID: training stopped. Details in %s", REPORT_PATH)
        return 1
    log.info("demand.parquet is valid: %d rows, %d zones", report["rows"], report["zones"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
