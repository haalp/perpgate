#!/usr/bin/env python3
"""Step 1 of the pipeline: download Binance USDT-M futures klines, build the monthly top-N
universe, and check the files we will backtest on.

Source: https://huggingface.co/datasets/rogerdehe/klines-binance (MIT, includes delisted pairs).

    pip install pandas pyarrow huggingface_hub
    python scripts/fetch_universe.py                 # top 50, 15m + 4h + 1d
    python scripts/fetch_universe.py --top 100
    python scripts/fetch_universe.py --no-download   # rebuild universe/report from files on disk

Outputs (under --data, default ./data):
    raw/futures/<tf>/<PAIR>.parquet   downloaded files, untouched
    universe.csv                      month, rank, pair, usd_volume_prev_month
    integrity.csv                     one row per (pair, timeframe): range, rows, gaps, bad bars

No look-ahead: the universe for month M is ranked on the USD volume of month M-1, so membership is
known before the month starts. Delisted pairs stay in for the months they qualified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = "rogerdehe/klines-binance"
SUFFIX = "_USDT_USDT"  # USDT-margined perpetual naming in the dataset
TF_FREQ = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "8h": "8h", "1d": "1D"}

# Not tradeable setups: stablecoins and wrapped/pegged assets barely move.
EXCLUDED_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "UST", "USTC", "EUR", "USD1", "PAXG", "XAUT", "WBTC"}


def base_of(pair: str) -> str:
    return pair[: -len(SUFFIX)] if pair.endswith(SUFFIX) else pair


def download(data: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", local_dir=data / "raw", allow_patterns=patterns)


def read_klines(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date")


def build_universe(data: Path, top: int, min_days: int) -> pd.DataFrame:
    """Rank pairs each month by the previous month's USD volume (close * base volume)."""
    rows = []
    for f in sorted((data / "raw" / "futures" / "1d").glob(f"*{SUFFIX}.parquet")):
        pair = f.stem
        if base_of(pair) in EXCLUDED_BASES:
            continue
        df = read_klines(f)
        if df.empty:
            continue
        usd = (df["close"] * df["volume"]).groupby(df["date"].dt.tz_localize(None).dt.to_period("M"))
        m = pd.DataFrame({"usd_volume": usd.sum(), "days": usd.size()})
        m["pair"] = pair
        rows.append(m.reset_index(names="vol_month"))
    if not rows:
        sys.exit("no 1d files found - run without --no-download first")
    vol = pd.concat(rows, ignore_index=True)
    # A pair needs a reasonably full month of history before it can be ranked (skips listing spikes).
    vol = vol[vol["days"] >= min_days].copy()
    vol["month"] = vol["vol_month"] + 1  # tradeable in the month after the one it was measured in
    vol["rank"] = vol.groupby("month")["usd_volume"].rank(method="first", ascending=False).astype(int)
    uni = vol[vol["rank"] <= top].sort_values(["month", "rank"])
    return uni.rename(columns={"usd_volume": "usd_volume_prev_month"})[["month", "rank", "pair", "usd_volume_prev_month"]]


def check_file(path: Path, tf: str) -> dict:
    df = read_klines(path)
    out = {"pair": path.stem, "tf": tf, "rows": len(df)}
    if df.empty:
        return out | {"status": "EMPTY"}
    step = pd.Timedelta(TF_FREQ[tf])
    d = df["date"]
    gaps = d.diff().dropna()
    gaps = gaps[gaps > step]
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    bad = ((h < o) | (h < c) | (l > o) | (l > c) | (h < l) | (c <= 0) | (v < 0) | df[["open", "high", "low", "close"]].isna().any(axis=1))
    out |= {
        "start": d.iloc[0].strftime("%Y-%m-%d"),
        "end": d.iloc[-1].strftime("%Y-%m-%d"),
        "duplicates": int(d.duplicated().sum()),
        "off_grid": int((d.dt.floor(TF_FREQ[tf]) != d).sum()),
        "gaps": len(gaps),
        "missing_bars": int((gaps / step - 1).sum()) if len(gaps) else 0,
        "longest_gap_days": round(gaps.max() / pd.Timedelta("1D"), 2) if len(gaps) else 0.0,
        "bad_ohlc_bars": int(bad.sum()),
        "zero_volume_bars": int((v == 0).sum()),
    }
    problems = [k for k in ("duplicates", "off_grid", "bad_ohlc_bars") if out[k]]
    if out["longest_gap_days"] > 3:
        problems.append("gap>3d")
    out["status"] = "CHECK:" + ",".join(problems) if problems else "OK"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--timeframes", default="15m,4h,1d")
    ap.add_argument("--min-days", type=int, default=20, help="days of history in a month before a pair can be ranked")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    tfs = [t for t in a.timeframes.split(",") if t]
    if unknown := [t for t in tfs if t not in TF_FREQ]:
        sys.exit(f"unknown timeframe(s): {unknown}")

    if not a.no_download:
        print("1/3 downloading all daily files (small) to rank pairs ...")
        download(a.data, [f"futures/1d/*{SUFFIX}.parquet", "coverage.csv"])

    uni = build_universe(a.data, a.top, a.min_days)
    a.data.mkdir(parents=True, exist_ok=True)
    uni.to_csv(a.data / "universe.csv", index=False)
    pairs = sorted(uni["pair"].unique())
    print(f"2/3 universe: {uni['month'].nunique()} months ({uni['month'].min()} .. {uni['month'].max()}), "
          f"{len(pairs)} distinct pairs ever in the top {a.top}")

    if not a.no_download:
        want = [f"futures/{tf}/{p}.parquet" for tf in tfs if tf != "1d" for p in pairs]
        want += [f"futures/1h/{p}-funding_rate.parquet" for p in pairs]  # sparse: present only for some pairs
        print(f"    downloading {len(want)} files for {', '.join(tfs)} + funding ...")
        download(a.data, want)

    rep = []
    for tf in tfs:
        for p in pairs:
            f = a.data / "raw" / "futures" / tf / f"{p}.parquet"
            rep.append(check_file(f, tf) if f.exists() else {"pair": p, "tf": tf, "rows": 0, "status": "MISSING"})
    rep = pd.DataFrame(rep)
    rep.to_csv(a.data / "integrity.csv", index=False)
    funding = sum((a.data / "raw" / "futures" / "1h" / f"{p}-funding_rate.parquet").exists() for p in pairs)
    print(f"3/3 integrity: {dict(rep['status'].str.split(':').str[0].value_counts())}; "
          f"funding files for {funding}/{len(pairs)} pairs")
    print(f"    wrote {a.data / 'universe.csv'} and {a.data / 'integrity.csv'}")


if __name__ == "__main__":
    main()
