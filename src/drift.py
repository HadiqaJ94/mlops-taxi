"""Drift detection (T11): is the current window still like the data the model was trained on?

Three checks, from the cheapest signal to the one that needs labels:
1. Input   PSI per feature, current window vs the reference period (what the model was trained on).
2. Output  PSI of the predictions, and their mean. Needs the model, not the labels.
3. Outcome WAPE per week on the current window, compared with the model's validation WAPE.
           Needs the labels, so it arrives last.

Run:  uv run python -m src.drift                                   # last 7 days of the data
      uv run python -m src.drift --start 2020-03-15 --end 2020-03-22
      uv run python -m src.drift --no-model                        # input check only, no MLflow

Writes reports/drift_report.json. Exit code 1 when drift is found, so a scheduled job (T17)
or retraining (T16) can react to it. Thresholds live in params.yaml under `drift`.

The PSI helpers (bin_edges, proportions, psi_from_profile) are shared with the API's /metrics
gauge (T10): compute the reference profile once at startup, then call psi_from_profile on the
sliding window of live inputs.
"""
import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEMAND_PATH, ROOT, load_params
from src.features import build_features, feature_columns

log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "reports" / "drift_report.json"
EPS = 1e-4  # stands in for an empty bin, so log(0) never happens


# ---------- PSI ----------

def bin_edges(reference, bins: int = 10) -> np.ndarray:
    """Decile edges of the reference values, open on both ends.

    Taxi counts have many repeated values (lots of zeros at night), so several deciles can be
    equal: duplicates are dropped and the data simply gets fewer, wider bins.
    """
    values = np.asarray(reference, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("reference has no values")
    inner = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def proportions(values, edges: np.ndarray) -> np.ndarray:
    """Share of the values in each bin. NaN values are ignored. Empty bins get EPS."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("no values to compare")
    # Bin i holds edges[i] < x <= edges[i + 1]: a value equal to a decile falls in the bin that
    # decile closes. This matters when many values sit exactly on a decile, like the zeros.
    idx = np.searchsorted(edges[1:-1], values, side="left")
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return np.clip(counts / counts.sum(), EPS, None)


def psi_from_profile(edges: np.ndarray, reference_props: np.ndarray, current) -> float:
    """PSI of `current` against a reference profile computed once with bin_edges/proportions."""
    cur = proportions(current, edges)
    return float(np.sum((cur - reference_props) * np.log(cur / reference_props)))


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index: 0 = same distribution. Rule of thumb (session 5):
    below 0.1 stable, 0.1 to 0.25 watch, above 0.25 act."""
    edges = bin_edges(reference, bins)
    return psi_from_profile(edges, proportions(reference, edges), current)


def status(value: float, watch: float, act: float) -> str:
    return "drift" if value >= act else "watch" if value >= watch else "ok"


# ---------- report ----------

def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.abs(y_true - np.asarray(y_pred, dtype=float)).sum() / max(y_true.sum(), 1.0))


def build_report(feats: pd.DataFrame, start, end, params: dict, model=None, model_version=None) -> dict:
    """Compare the window [start, end) with the reference period. Pure: no files, no MLflow.

    feats: output of build_features (one row per zone and hour, with lags and `period`).
    model: anything with .predict(X) on feature_columns, or None to skip checks 2 and 3.
    """
    cfg = params["drift"]
    watch, act = cfg["psi_watch"], cfg["psi_act"]
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    ref = feats[feats["period"] == "reference"]
    cur = feats[(feats["pickup_hour"] >= start) & (feats["pickup_hour"] < end)]
    if cur.empty:
        raise ValueError(f"no rows between {start} and {end}")

    reasons: list[str] = []
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": {"start": str(ref["pickup_hour"].min()), "end": str(ref["pickup_hour"].max()),
                      "rows": len(ref)},
        "current": {"start": str(start), "end": str(end), "rows": len(cur)},
        "thresholds": {"psi_watch": watch, "psi_act": act, "wape_ratio_max": cfg["wape_ratio_max"]},
        "model_version": model_version,
    }

    # 1. input: PSI per feature
    features = {}
    for name in cfg["features"]:
        value = psi(ref[name], cur[name], cfg["bins"])
        features[name] = {"psi": round(value, 4), "status": status(value, watch, act),
                          "reference_mean": round(float(ref[name].mean()), 3),
                          "current_mean": round(float(cur[name].mean()), 3)}
        if value >= act:
            reasons.append(f"input drift on {name}: PSI {value:.2f} >= {act}")
    report["features"] = features

    if model is not None:
        cols = feature_columns(params["features"]["lags"])
        ref_pred = np.clip(model.predict(ref[cols]), 0, None)
        cur_pred = np.clip(model.predict(cur[cols]), 0, None)

        # 2. output: PSI of the predictions
        value = psi(ref_pred, cur_pred, cfg["bins"])
        report["prediction"] = {"psi": round(value, 4), "status": status(value, watch, act),
                                "reference_mean": round(float(ref_pred.mean()), 3),
                                "current_mean": round(float(cur_pred.mean()), 3)}
        if value >= act:
            reasons.append(f"prediction drift: PSI {value:.2f} >= {act}")

        # 3. outcome: WAPE per week vs the model's validation WAPE (same window as train.py)
        cutoff = ref["pickup_hour"].max() - pd.Timedelta(days=params["train"]["val_days"])
        val = ref[ref["pickup_hour"] > cutoff]
        ref_wape = wape(val["target"], np.clip(model.predict(val[cols]), 0, None))
        limit = cfg["wape_ratio_max"] * ref_wape
        weeks = []
        scored = cur.assign(pred=cur_pred)
        for week, g in scored.groupby(pd.Grouper(key="pickup_hour", freq="W-MON", label="left", closed="left")):
            if g.empty:
                continue
            w = wape(g["target"], g["pred"])
            weeks.append({"week_start": str(week.date()), "rows": len(g), "wape": round(w, 4),
                          "status": "drift" if w > limit else "ok"})
            if w > limit:
                reasons.append(f"performance drop week of {week.date()}: WAPE {w:.3f} > {limit:.3f}")
        report["performance"] = {"reference_wape": round(ref_wape, 4), "limit": round(limit, 4),
                                 "weekly": weeks}

    report["drift"] = bool(reasons)
    report["reasons"] = reasons
    return report


def load_champion():
    """The model the API serves, and its version. Imported lazily: --no-model needs no MLflow."""
    import mlflow

    from src import registry

    client, name = registry.get_client(), registry.model_name()
    version = str(client.get_model_version_by_alias(name, registry.CHAMPION).version)
    return mlflow.sklearn.load_model(f"models:/{name}/{version}"), version


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drift report: current window vs reference period")
    ap.add_argument("--start", help="first hour of the current window (default: 7 days before --end)")
    ap.add_argument("--end", help="end of the current window, excluded (default: end of the data)")
    ap.add_argument("--no-model", action="store_true", help="input check only, no MLflow needed")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    params = load_params()
    feats = build_features(pd.read_parquet(DEMAND_PATH), params["features"]["lags"])
    end = pd.Timestamp(args.end) if args.end else feats["pickup_hour"].max() + pd.Timedelta(hours=1)
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=7)

    model, version = (None, None) if args.no_model else load_champion()
    report = build_report(feats, start, end, params, model, version)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    for reason in report["reasons"]:
        log.warning(reason)
    log.info("%s -> %s (drift=%s)", f"{start:%Y-%m-%d}..{end:%Y-%m-%d}", args.out, report["drift"])
    return 1 if report["drift"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
