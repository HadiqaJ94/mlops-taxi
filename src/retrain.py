"""Drift-triggered retraining (T16): on drift, retrain on recent weeks, register a candidate,
and let the training gate (src/gate.py, T6) decide whether it replaces the champion.

Windows, all ending before `as_of` (end excluded, params.yaml `retrain`):

    |<------------ train_weeks ------------>|<-- val_weeks -->|
    train_start                    train_end = val_start      val_end = as_of

No leakage: every training row is strictly before val_start, and a row's lags are earlier hours,
so nothing from the validation weeks reaches the model. The drift check, the candidate's metrics
and the gate all use the same validation weeks, so champion and candidate are compared on the
most recent data.

Run:  uv run python -m src.retrain                       # drift check on the last week, retrain if drift
      uv run python -m src.retrain --force               # retrain even without drift
      uv run python -m src.retrain --as-of 2020-03-23 --force   # incident demo (T19)
      uv run python -m src.retrain --dry-run --force     # train and decide in memory, register nothing

Writes reports/retrain_report.json (and drift_report.json, gate_report.json on the way).
Exit code 0 when no retrain is needed, and when the candidate is promoted or rejected; 1 on an error.
"""
import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from mlflow.exceptions import MlflowException

from src import drift, gate, registry
from src.config import DEMAND_PATH, ROOT, load_params
from src.features import build_features, feature_columns
from src.train import log_and_register, score, train_model, window

log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "reports" / "retrain_report.json"


def compute_windows(as_of, train_weeks: int, val_weeks: int, first_hour) -> dict:
    """Train and validation windows ending at as_of (excluded). Consecutive, never overlapping."""
    as_of = pd.Timestamp(as_of)
    if as_of != as_of.floor("h"):
        raise ValueError(f"as_of must be on the hour, got {as_of}")
    if train_weeks < 1 or val_weeks < 1:
        raise ValueError("train_weeks and val_weeks must be >= 1")
    val_start = as_of - pd.Timedelta(weeks=val_weeks)
    train_start = val_start - pd.Timedelta(weeks=train_weeks)
    if train_start < pd.Timestamp(first_hour):
        raise ValueError(f"training window starts {train_start}, before the first hour with full "
                         f"lags {first_hour}: use a later --as-of or fewer --train-weeks")
    return {"train_start": train_start, "train_end": val_start, "val_start": val_start, "val_end": as_of}


def check_drift(start, end, out: Path = drift.REPORT_PATH) -> dict:
    """Run src/drift.py on [start, end) and return its report (exit code 1 = drift, not an error).

    Without a champion (fresh registry) the model-based checks cannot run: fall back to inputs only.
    """
    args = ["--start", str(start), "--end", str(end), "--out", str(out)]
    try:
        drift.main(args)
    except MlflowException as exc:
        log.warning("no champion to check (%s): input drift only", exc.__class__.__name__)
        drift.main([*args, "--no-model"])
    return json.loads(Path(out).read_text())


def load_champion_model():
    """(version, model) of the current champion, or (None, None) if there is none."""
    client = registry.get_client()
    name = registry.model_name()
    version = registry.champion_version(client, name)
    return (None, None) if version is None else (str(version), gate.load_version(name, str(version)))


def summarize_drift(report: dict, path) -> dict:
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):  # repo-relative: the same in every clone and CI runner
        path = path.relative_to(ROOT).as_posix()
    return {
        "detected": bool(report["drift"]),
        "features_in_drift": [n for n, f in report.get("features", {}).items() if f["status"] == "drift"],
        "prediction_drift": report.get("prediction", {}).get("status") == "drift",
        "reasons": report.get("reasons", []),
        "report_path": str(path),
    }


def run_retrain(feats: pd.DataFrame, params: dict, as_of=None, force: bool = False, dry_run: bool = False,
                drift_report: Path | None = None) -> dict:
    """The whole retraining decision. Returns the report (written by main)."""
    rp, lags = params["retrain"], params["features"]["lags"]
    as_of = pd.Timestamp(as_of or rp["as_of"] or feats["pickup_hour"].max() + pd.Timedelta(hours=1))
    w = compute_windows(as_of, rp["train_weeks"], rp["val_weeks"], feats["pickup_hour"].min())
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "as_of": str(as_of),
        "windows": {"train": [str(w["train_start"]), str(w["train_end"])],
                    "validation": [str(w["val_start"]), str(w["val_end"])]},
        "force": force,
        "dry_run": dry_run,
    }

    # 1. trigger: drift on the validation weeks (or an existing drift report)
    if drift_report is not None:
        path, drift_json = drift_report, json.loads(Path(drift_report).read_text())
    else:
        path = drift.REPORT_PATH
        drift_json = check_drift(w["val_start"], w["val_end"], path)
    report["drift"] = summarize_drift(drift_json, path)
    if not report["drift"]["detected"] and not force:
        report.update(retrained=False, trigger=None, reason="no drift detected: no retrain needed")
        return report
    trigger = "drift" if report["drift"]["detected"] else "manual"
    report["trigger"] = trigger

    # 2. train on the recent weeks only, score on the validation weeks
    model = train_model(feats, w["train_start"], w["train_end"], params)
    val = window(feats, w["val_start"], w["val_end"])
    metrics = score(model, val, lags)
    report["retrained"] = True
    report["candidate"] = {"version": None, "run_id": None, "val_wape": round(metrics["val_wape"], 4),
                           "baseline_val_wape": round(metrics["baseline_val_wape"], 4),
                           "n_train": len(window(feats, w["train_start"], w["train_end"])), "n_val": len(val)}

    # 3. the gate decides, on the same validation weeks
    min_improvement = params["gate"]["min_improvement"]
    if dry_run:  # nothing registered: decide in memory against the current champion
        champion_version, champion = load_champion_model()
        models = {"candidate": model, **({"champion": champion} if champion is not None else {})}
        wapes, _ = gate.wapes_on_window(feats, w["val_start"], w["val_end"], lags, models)
        promote, reason = gate.decide(wapes["candidate"], wapes["baseline"], wapes.get("champion"), min_improvement)
        report["gate"] = {"decision": "promoted" if promote else "rejected", "promote": promote, "reason": reason,
                          "champion_version": champion_version, "dry_run": True,
                          "wape": {k: round(v, 4) for k, v in wapes.items()}}
        return report

    run_id, version = log_and_register(
        model, params, metrics, val[feature_columns(lags)].head(5),
        tags={"trigger": trigger, "training_period": "recent", "as_of": str(as_of),
              "train_window": f"{w['train_start']}..{w['train_end']}",
              "val_window": f"{w['val_start']}..{w['val_end']}",
              "drift_detected": str(report["drift"]["detected"]), "drift_report": str(path)},
        extra_params={"n_train": report["candidate"]["n_train"], "n_val": len(val)},
    )
    report["candidate"].update(version=version, run_id=run_id)
    g = gate.run(w["val_start"], w["val_end"], min_improvement, feats=feats)
    report["gate"] = {k: g.get(k) for k in ("decision", "promote", "reason", "champion_version", "wape")}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retrain on recent weeks when drift is found; the gate decides")
    ap.add_argument("--as-of", help="end of the recent windows, excluded (default: params.yaml, else end of data)")
    ap.add_argument("--train-weeks", type=int, help="override params.yaml retrain.train_weeks")
    ap.add_argument("--val-weeks", type=int, help="override params.yaml retrain.val_weeks")
    ap.add_argument("--force", action="store_true", help="retrain even if no drift is found")
    ap.add_argument("--dry-run", action="store_true", help="train and decide, register and promote nothing")
    ap.add_argument("--drift-report", type=Path, help="use this drift report instead of running src.drift")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    params = load_params()
    for key in ("train_weeks", "val_weeks"):
        if getattr(args, key) is not None:
            params["retrain"][key] = getattr(args, key)
    try:
        feats = build_features(pd.read_parquet(DEMAND_PATH), params["features"]["lags"])
        report = run_retrain(feats, params, args.as_of, args.force, args.dry_run, args.drift_report)
    except (MlflowException, ValueError, OSError) as exc:  # MLflow down, bad window, missing data
        log.error("retrain failed: %s: %s", type(exc).__name__, exc)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    outcome = report["gate"]["decision"] if report["retrained"] else "no retrain"
    if report["dry_run"] and report["retrained"]:
        outcome += " (dry run: nothing registered)"
    log.info("retrain: drift=%s trigger=%s -> %s (%s)", report["drift"]["detected"], report.get("trigger"),
             outcome, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
