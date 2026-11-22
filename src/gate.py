"""Training gate (T6): @candidate becomes @champion only if it beats the baseline AND the champion.

The candidate, the current champion and the "same hour last week" baseline are scored on exactly
the same rows: a FIXED validation window (params.yaml `gate`), features from src/features.py.
A fixed window makes WAPEs comparable from one run to the next; retraining (T16) passes its own
recent window with --val-start/--val-end so both models are judged on the same recent weeks.

Run:  uv run python -m src.gate                       # decide, promote if the candidate wins
      uv run python -m src.gate --dry-run             # decide only, touch nothing
      uv run python -m src.gate --val-start 2020-06-24 --val-end 2020-07-01

Writes reports/gate_report.json and tags the candidate version in MLflow (gate_decision,
gate_reason, gate_*_wape). Exit code 0 whether the candidate is promoted or rejected (a rejection
is a normal outcome); 1 on an error (no @candidate, MLflow unreachable, empty window).
"""
import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from mlflow.exceptions import MlflowException

from src import registry
from src.config import DEMAND_PATH, ROOT, load_params
from src.features import build_features, feature_columns
from src.train import wape, window

log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "reports" / "gate_report.json"


def decide(candidate_wape: float, baseline_wape: float, champion_wape: float | None,
           min_improvement: float) -> tuple[bool, str]:
    """Pure decision rule (no MLflow): returns (promote, reason).

    1. The candidate must beat the baseline (strictly lower WAPE), or it is useless.
    2. If a champion exists, the candidate's WAPE must be at most champion * (1 - min_improvement):
       min_improvement=0.01 means "at least 1% better", 0 means "better or equal".
    """
    if not 0 <= min_improvement < 1:
        raise ValueError(f"min_improvement must be in [0, 1), got {min_improvement}")
    c, b = candidate_wape, baseline_wape
    if c >= b:
        return False, f"candidate WAPE {c:.4f} does not beat the baseline {b:.4f}"
    if champion_wape is None:
        return True, f"no champion yet and candidate WAPE {c:.4f} beats the baseline {b:.4f}"
    needed = champion_wape * (1 - min_improvement)
    if c <= needed:
        return True, (f"candidate WAPE {c:.4f} beats the baseline {b:.4f} and the champion "
                      f"{champion_wape:.4f} (needed <= {needed:.4f}, min_improvement {min_improvement:g})")
    return False, (f"candidate WAPE {c:.4f} is not better than the champion {champion_wape:.4f} "
                   f"by min_improvement {min_improvement:g} (needed <= {needed:.4f})")


def wapes_on_window(feats: pd.DataFrame, start, end, lags, models: dict) -> tuple[dict, int]:
    """WAPE of each model and of the baseline (lag_168) on the same rows [start, end)."""
    val = window(feats, start, end)
    if val.empty:
        raise ValueError(f"no rows in the validation window {start}..{end}")
    cols = feature_columns(lags)
    out = {"baseline": wape(val["target"], val["lag_168"])}
    for role, model in models.items():
        out[role] = wape(val["target"], model.predict(val[cols]))
    return out, len(val)


def load_version(name: str, version: str):
    import mlflow

    return mlflow.sklearn.load_model(f"models:/{name}/{version}")


def run_gate(client, name: str, feats: pd.DataFrame, start, end, lags, min_improvement: float,
             dry_run: bool = False, load_model: Callable = load_version) -> dict:
    """Score candidate, champion and baseline on [start, end), decide, then apply (unless dry_run).

    `client` is an MlflowClient (or a fake in tests). Returns the report as a dict.
    """
    candidate = str(client.get_model_version_by_alias(name, registry.CANDIDATE).version)
    champion = registry.champion_version(client, name)
    champion = str(champion) if champion is not None else None
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": name,
        "candidate_version": candidate,
        "champion_version": champion,
        "window": {"start": str(pd.Timestamp(start)), "end": str(pd.Timestamp(end))},
        "min_improvement": min_improvement,
        "dry_run": dry_run,
    }

    if champion == candidate:  # e.g. the gate already ran on this candidate
        report.update(decision="skipped", promote=False, wape=None,
                      reason=f"candidate v{candidate} is already the champion: nothing to decide")
        return report

    models = {"candidate": load_model(name, candidate)}
    if champion is not None:
        models["champion"] = load_model(name, champion)
    wapes, rows = wapes_on_window(feats, start, end, lags, models)
    promote, reason = decide(wapes["candidate"], wapes["baseline"], wapes.get("champion"), min_improvement)
    report["window"]["rows"] = rows
    report["wape"] = {"candidate": round(wapes["candidate"], 4), "baseline": round(wapes["baseline"], 4),
                      "champion": round(wapes["champion"], 4) if "champion" in wapes else None}
    report.update(decision="promoted" if promote else "rejected", promote=promote, reason=reason)

    if dry_run:
        log.info("dry run: would have %s v%s", report["decision"], candidate)
        return report

    if promote:
        registry.promote(client, name, candidate)  # sets @champion and the `replaces` tag (T18)
    tags = {"gate_decision": report["decision"], "gate_reason": reason,
            "gate_window": f"{report['window']['start']}..{report['window']['end']}",
            "gate_candidate_wape": f"{wapes['candidate']:.4f}", "gate_baseline_wape": f"{wapes['baseline']:.4f}",
            "gate_champion_wape": f"{wapes['champion']:.4f}" if "champion" in wapes else "none",
            "gate_champion_version": champion or "none"}
    for key, value in tags.items():
        client.set_model_version_tag(name, candidate, key, value)
    return report


def run(val_start=None, val_end=None, min_improvement=None, dry_run=False, out: Path = REPORT_PATH,
        feats: pd.DataFrame | None = None) -> dict:
    """Load params, data and the registry, run the gate, write the report. Used by retraining (T16)."""
    params = load_params()
    g, lags = params["gate"], params["features"]["lags"]
    start, end = pd.Timestamp(val_start or g["val_start"]), pd.Timestamp(val_end or g["val_end"])
    min_improvement = g["min_improvement"] if min_improvement is None else min_improvement
    if feats is None:
        feats = build_features(pd.read_parquet(DEMAND_PATH), lags)

    client = registry.get_client()
    report = run_gate(client, registry.model_name(), feats, start, end, lags, min_improvement, dry_run)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    log.info("gate: %s v%s (%s) -> %s", report["decision"], report["candidate_version"], report["reason"], out)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Promote @candidate only if it beats the baseline and @champion")
    ap.add_argument("--val-start", help="first hour of the validation window (default: params.yaml gate)")
    ap.add_argument("--val-end", help="end of the validation window, excluded (default: params.yaml gate)")
    ap.add_argument("--min-improvement", type=float, help="relative WAPE gain needed over the champion")
    ap.add_argument("--dry-run", action="store_true", help="decide, but do not promote or tag anything")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)
    try:
        run(args.val_start, args.val_end, args.min_improvement, args.dry_run, args.out)
    # no @candidate or MLflow unreachable (MlflowException), empty window (ValueError),
    # no data file (OSError): a real failure, unlike a rejection
    except (MlflowException, ValueError, OSError) as exc:
        log.error("gate failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
