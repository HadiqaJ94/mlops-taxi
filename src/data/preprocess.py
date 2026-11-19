"""Raw TLC trips -> one row per (zone, hour), zero-filled, tagged reference/drift."""
import json
import logging

import numpy as np
import pandas as pd

from src.config import DEMAND_PATH, PROCESSED_DIR, RAW_DIR, load_params

log = logging.getLogger(__name__)

# fare/distance are kept as *monitoring-only* columns: they let you inject a
# "broken fare column" incident. They are NOT model features.
COLUMNS = ["tpep_pickup_datetime", "PULocationID", "fare_amount", "trip_distance"]


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01")
    return start, start + pd.offsets.MonthBegin(1)


def clean_trips(df: pd.DataFrame, month: str, min_zone: int, max_zone: int):
    """Clean one raw month. Returns (clean_df, stats) where stats counts dropped rows."""
    start, end = month_bounds(month)
    df = df.copy()
    stats = {"rows_raw": int(len(df))}

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")

    df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
    stats["rows_after_dropna"] = int(len(df))

    # TLC files contain stray timestamps (2008, 2009, previous/next month...)
    ts = df["tpep_pickup_datetime"]
    df = df[(ts >= start) & (ts < end)]
    stats["rows_in_month"] = int(len(df))

    df = df[df["PULocationID"].between(min_zone, max_zone)]
    stats["rows_valid_zone"] = int(len(df))

    df["zone_id"] = df["PULocationID"].astype("int64")
    return df, stats


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Sparse aggregation: only (zone, hour) pairs that had at least one trip."""
    df = df.copy()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    return (
        df.groupby(["zone_id", "pickup_hour"])
        .agg(
            target=("fare_amount", "size"),
            mean_fare=("fare_amount", "mean"),
            mean_distance=("trip_distance", "mean"),
        )
        .reset_index()
    )


def select_zones(agg: pd.DataFrame, reference_months: list[str], top_n: int) -> list[int]:
    """Busiest zones, chosen on the REFERENCE period only (no peeking at the drift)."""
    month = agg["pickup_hour"].dt.strftime("%Y-%m")
    ref = agg[month.isin(reference_months)]
    return sorted(int(z) for z in ref.groupby("zone_id")["target"].sum().nlargest(top_n).index)


def complete_grid(agg: pd.DataFrame, zones: list[int], months: list[str]) -> pd.DataFrame:
    """Full zone x hour grid. Hours with no trip become target=0 (not missing rows!)."""
    ranges = []
    for m in months:
        start, end = month_bounds(m)
        ranges.append(pd.date_range(start, end, freq="h", inclusive="left"))
    hours = ranges[0].append(ranges[1:])

    index = pd.MultiIndex.from_product([zones, hours], names=["zone_id", "pickup_hour"])
    agg = agg.copy()
    agg["zone_id"] = agg["zone_id"].astype("int64")
    out = agg.set_index(["zone_id", "pickup_hour"]).reindex(index)
    out["target"] = out["target"].fillna(0).astype("int32")
    return out.reset_index()


def main() -> None:
    p = load_params()["data"]
    ref_months, months = p["reference_months"], p["reference_months"] + p["drift_months"]

    parts, report = [], {"months": {}}
    for m in months:
        raw = pd.read_parquet(RAW_DIR / f"yellow_tripdata_{m}.parquet", columns=COLUMNS)
        clean, stats = clean_trips(raw, m, p["min_zone_id"], p["max_zone_id"])
        stats["rows_dropped_out_of_month"] = stats["rows_after_dropna"] - stats["rows_in_month"]
        report["months"][m] = stats
        parts.append(aggregate_hourly(clean))
        log.info("%s: %s", m, stats)

    agg = pd.concat(parts, ignore_index=True)
    zones = select_zones(agg, ref_months, p["top_zones"])
    full = complete_grid(agg[agg["zone_id"].isin(zones)], zones, months)

    full["month"] = full["pickup_hour"].dt.strftime("%Y-%m")
    full["period"] = np.where(full["month"].isin(ref_months), "reference", "drift")
    full["zone_id"] = full["zone_id"].astype("int16")
    full = full.sort_values(["zone_id", "pickup_hour"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_parquet(DEMAND_PATH, index=False)
    (PROCESSED_DIR / "zones.json").write_text(json.dumps(zones))
    report["n_zones"], report["n_rows"] = len(zones), int(len(full))
    (PROCESSED_DIR / "quality_report.json").write_text(json.dumps(report, indent=2))
    log.info("Saved %s (%d rows, %d zones)", DEMAND_PATH, len(full), len(zones))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
