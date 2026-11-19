"""Paths and params shared by every script."""
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEMAND_PATH = PROCESSED_DIR / "demand.parquet"

load_dotenv(ROOT / ".env")  # secrets live in .env, never in Git


def load_params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text())
