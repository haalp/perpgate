#!/usr/bin/env python3
"""Read data/candidates.parquet and print result tables.

    python scripts/report.py                      # training period only (see PROTOCOL.md)
    python scripts/report.py --pairs BTC,ETH,SOL  # change the focus list
    python scripts/report.py --split validation   # only when PROTOCOL.md says the split may be opened

Columns: n = candidates, hit = share that touched +2R before the stop (a coin flip gives 0.333),
gross / net = mean result in R before / after fees and slippage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SPLITS = {"train": ("2020-01-01", "2025-01-01"), "validation": ("2025-01-01", "2026-01-01"),
          "test": ("2026-01-01", "2100-01-01")}
FOCUS = "LIT,HYPE,AAVE,BTC,ETH,SOL,FET"
AGG = dict(n=("r_net", "size"), hit=("label_2r", "mean"), gross=("r_gross", "mean"), net=("r_net", "mean"))


def table(c: pd.DataFrame, by, title: str) -> None:
    print(f"\n== {title}")
    print("(no candidates)" if c.empty else c.groupby(by, observed=True).agg(**AGG).round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", choices=list(SPLITS), default="train")
    ap.add_argument("--pairs", default=FOCUS, help="comma-separated base assets for the focus table")
    a = ap.parse_args()

    c = pd.read_parquet(a.data / "candidates.parquet")
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in SPLITS[a.split])
    c = c[(c["entry_time"] >= lo) & (c["entry_time"] < hi)].copy()
    if a.split != "train":
        print(f"NOTE: this is the {a.split} split. Looking at it before PROTOCOL.md allows it spends it.")
    print(f"{a.split}: {len(c)} candidates, {c['pair'].nunique()} pairs, "
          f"{c['entry_time'].min():%Y-%m-%d} .. {c['entry_time'].max():%Y-%m-%d}")
    c["year"] = c["entry_time"].dt.year
    c["side"] = c["side"].map({1: "long", -1: "short"})
    c["base"] = c["pair"].str.replace("_USDT_USDT", "", regex=False)

    table(c, ["setup", "side"], "setup x side")
    table(c, ["setup", "side", "year"], "setup x side x year (is any edge stable, or one year's doing?)")
    table(c[c["setup"] == "A"], ["level_type", "side"], "setup A by level type")

    focus = [b.strip().upper() for b in a.pairs.split(",") if b.strip()]
    c["group"] = c["base"].where(c["base"].isin(focus), "(rest of universe)")
    table(c, ["group", "setup"], "focus pairs vs the rest")
    absent = [b for b in focus if b not in set(c["base"])]
    if absent:
        print(f"no candidates in this split for: {', '.join(absent)} (not listed yet, or not in the top 50 then)")

    if c["mom_90d_rank"].notna().any():
        c["trend"] = pd.cut(c["mom_90d_rank"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True,
                            labels=["weakest 20%", "20-40%", "40-60%", "60-80%", "strongest 20%"])
        table(c.dropna(subset=["trend"]), ["setup", "trend"],
              "by 90-day trend strength in the trade's direction, ranked across all pairs at entry")
    c["stop"] = pd.cut(c["stop_pct"] * 100, [0.3, 0.6, 1.0, 1.5, 2.5, 5.0], include_lowest=True)
    table(c, ["stop"], "by stop distance in % (tight stops pay more of their R in costs)")


if __name__ == "__main__":
    main()
