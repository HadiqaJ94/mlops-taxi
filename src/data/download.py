"""Download the NYC TLC yellow taxi trip records (one Parquet file per month).

Source: NYC Taxi & Limousine Commission (TLC) Trip Record Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""
import logging
from pathlib import Path

import requests

from src.config import RAW_DIR, load_params

log = logging.getLogger(__name__)


def download_month(month: str, base_url: str, raw_dir: Path = RAW_DIR, timeout: int = 180) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"yellow_tripdata_{month}.parquet"
    dest = raw_dir / filename
    if dest.exists():
        log.info("Already downloaded: %s", filename)
        return dest

    # write to a temp file then rename: a failed download never leaves a corrupt file
    tmp = dest.with_suffix(".parquet.part")
    with requests.get(base_url + filename, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    log.info("Downloaded: %s", filename)
    return dest


def main() -> None:
    p = load_params()["data"]
    for month in p["reference_months"] + p["drift_months"]:
        download_month(month, p["base_url"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
