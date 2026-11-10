#!/usr/bin/env python3
"""Step 1b: download Binance USDT-M futures "metrics" (5-minute open interest, long/short ratios,
taker ratio) from the official archive for every pair in universe.csv.

    pip install requests pandas pyarrow
    python scripts/fetch_metrics.py

Source: https://data.binance.vision/data/futures/um/daily/metrics/<SYMBOL>/<SYMBOL>-metrics-<DATE>.zip
Output: data/raw/metrics/<PAIR>.parquet  (one file per pair, 5-minute rows, UTC)

Each pair is fetched from 2 months before it first enters the universe (warm-up for the liquidation
map) to the end of its last universe month. Days the archive does not have are skipped and counted.
Re-running only downloads pairs that have no output file yet.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
COLS = ["create_time", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def symbol_of(pair: str) -> str:
    return pair.replace("_USDT_USDT", "USDT")


def parse_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df = df[[c for c in COLS if c in df.columns]].copy()
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    return df


def fetch_day(session: requests.Session, symbol: str, day: pd.Timestamp) -> pd.DataFrame | None:
    url = f"{BASE}/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException:
            continue
        if r.status_code == 404:
            return None
        if r.ok:
            return parse_zip(r.content)
    raise RuntimeError(f"failed after 3 tries: {url}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    uni = pd.read_csv(a.data / "universe.csv", dtype={"month": str})
    out_dir = a.data / "raw" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    session = requests.Session()
    summary = []
    for pair, g in uni.groupby("pair"):
        out = out_dir / f"{pair}.parquet"
        if out.exists():
            continue
        start = pd.Period(g["month"].min(), "M").start_time - pd.DateOffset(months=2)
        end = min(pd.Period(g["month"].max(), "M").end_time.normalize(), today - pd.Timedelta("1D"))
        days = pd.date_range(start, end, freq="D")
        with ThreadPoolExecutor(a.workers) as ex:
            parts = list(ex.map(lambda d: fetch_day(session, symbol_of(pair), d), days))
        got = [p for p in parts if p is not None]
        if got:
            df = pd.concat(got, ignore_index=True).drop_duplicates("create_time").sort_values("create_time")
            df.to_parquet(out, index=False)
        first = f"{got[0]['create_time'].min():%Y-%m-%d}" if got else "-"
        summary.append({"pair": pair, "days_wanted": len(days), "days_found": len(got), "first_day": first})
        print(f"{pair:28s} {len(got):5d}/{len(days):5d} days, first {first}")
    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(a.data / "metrics_coverage.csv", index=False)
        print(f"\n{(s['days_found'] > 0).sum()}/{len(s)} pairs have metrics; "
              f"{s['days_found'].sum()}/{s['days_wanted'].sum()} days found. Wrote {a.data / 'metrics_coverage.csv'}")


if __name__ == "__main__":
    main()
