#!/usr/bin/env python3
"""Step 2 of the pipeline: turn 15m klines into labelled candidate trades.

    python scripts/build_candidates.py            # reads ./data, writes ./data/candidates.parquet

Rules (mechanical version of the trade plan; every threshold is a named constant below):
  Trend    4h close vs 4h EMA50 and 1d close vs 1d EMA20 must agree. No trade against them.
  Setup A  1h close breaks a level, then a 15m bar retests the level and closes back above it.
           Levels: 1d / 4h / 1h range high, prior 48h high, yearly / monthly / weekly / daily open
           (Istanbul time; 1d and 4h candles stay on the exchange's UTC grid, as on the chart).
           A range = the last 20 bars of that timeframe span at most 4 ATR; its high/low are the levels.
  Setup B  6h impulse >= 2.5 ATR, then a pullback of <= 50%, then a 15m close above the impulse high.
  Entry    at the close of the 15m trigger bar. Stop below the retest/flag low. Shorts are mirrored.
  Label    label_2r = 1 if +2R is touched before the stop. If both fall in one bar, the stop wins.
  r_net    plan result in R after costs: half off at 2R, stop to entry, rest at 3R / entry / timeout.

Only closed bars are used, 1h/4h/1d are rebuilt from 15m, and a pair is only eligible in months where
it is in universe.csv. One open candidate per pair at a time. Funding is not modelled yet.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

M15 = pd.Timedelta("15min")
LOCAL_TZ = "Europe/Istanbul"   # day / week / month / year opens and the hour, dow features use this clock
DONCHIAN_H = 48          # 1h bars that define the "prior high" level
RANGE_BARS, RANGE_MAX_ATR = 20, 4.0   # a range: 20 bars of a timeframe spanning <= 4 ATR of that timeframe
# Level types for setup A, in priority order when several break on the same 1h bar.
LEVELS = ["rng_1d", "rng_4h", "rng_1h", "don48", "open_y", "open_m", "open_w", "open_d"]
SCAN_BARS = 48           # 15m bars (12h) allowed between setup and trigger
RETEST_ATR = 0.25        # retest must come within this many ATR of the level
FAIL_ATR = 0.5           # 15m close this far back through the level cancels setup A
IMPULSE_H, IMPULSE_ATR = 6, 2.5
FLAG_MIN_BARS, FLAG_MAX_RETRACE = 8, 0.5
STOP_BUFFER_ATR = 0.1
STOP_PCT_MIN, STOP_PCT_MAX = 0.003, 0.05   # skip stops tighter than 0.3% or wider than 5%
HORIZON_BARS = 672       # 7 days of 15m bars, then the trade is closed at market
TP1_R, TP2_R = 2.0, 3.0


def flip(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror prices so a short can be processed with the long logic."""
    return pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"], "close": -df["close"],
                         "volume": df["volume"]}, index=df.index)


def resample(b15: pd.DataFrame, rule: str, n: int) -> pd.DataFrame:
    g = b15.resample(rule, label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out[g["close"].count() == n]  # complete bars only


def htf_at(times: pd.Series, htf: pd.DataFrame, dur: str, span: int, name: str) -> pd.DataFrame:
    """Last *closed* higher-timeframe bar as of each time in `times`."""
    x = pd.DataFrame({"close_time": htf.index + pd.Timedelta(dur), f"c_{name}": htf["close"].to_numpy(),
                      f"ema_{name}": htf["close"].ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()})
    return pd.merge_asof(times.to_frame("t"), x, left_on="t", right_on="close_time").drop(columns=["t", "close_time"])


def atr_of(bars: pd.DataFrame) -> pd.Series:
    pc = bars["close"].shift()
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - pc).abs(), (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def range_at(times: pd.Series, bars: pd.DataFrame, dur: str, name: str) -> pd.DataFrame:
    """High and width (in ATR) of the last RANGE_BARS *closed* bars of a timeframe, as of each time."""
    hi, lo = bars["high"].rolling(RANGE_BARS).max(), bars["low"].rolling(RANGE_BARS).min()
    x = pd.DataFrame({"close_time": bars.index + pd.Timedelta(dur), f"rng_hi_{name}": hi.to_numpy(),
                      f"rng_w_{name}": ((hi - lo) / atr_of(bars)).to_numpy()})
    return pd.merge_asof(times.to_frame("t"), x, left_on="t", right_on="close_time").drop(columns=["t", "close_time"])


def period_opens(b15: pd.DataFrame) -> pd.DataFrame:
    """Daily / weekly (Monday) / monthly / yearly open on the LOCAL_TZ clock (00:00 Istanbul = 21:00 UTC).
    NaN when the period's first bar is missing."""
    idx = b15.index
    local = idx.tz_convert(LOCAL_TZ)
    day = local.floor("D")
    keys = {"d": day, "w": day - pd.to_timedelta(local.dayofweek, unit="D"),
            "m": day - pd.to_timedelta(local.day - 1, unit="D"), "y": day - pd.to_timedelta(local.dayofyear - 1, unit="D")}
    out = {}
    for k, key in keys.items():
        first_open = b15["open"].groupby(key).transform("first")
        first_time = pd.Series(idx, index=idx).groupby(key).transform("first")
        out[f"lvl_open_{k}"] = first_open.where(pd.DatetimeIndex(first_time).asi8 == key.asi8)
    return pd.DataFrame(out, index=idx)


def hourly_frame(b15: pd.DataFrame) -> pd.DataFrame:
    h = resample(b15, "1h", 4)
    h["atr"] = atr_of(h)
    h["lvl_don48"] = h["high"].rolling(DONCHIAN_H).max().shift()
    h["imp"] = h["close"] - h["close"].shift(IMPULSE_H)
    h["imp_high"] = h["high"].rolling(IMPULSE_H + 1).max()
    h["vol_rel"] = h["volume"] / h["volume"].rolling(20).mean().shift()
    ref = h["close"].abs()
    h["ret_24h"] = (h["close"] - h["close"].shift(24)) / ref.shift(24)
    h["ret_7d"] = (h["close"] - h["close"].shift(168)) / ref.shift(168)
    h["rv_24h"] = (h["close"].diff() / ref.shift()).rolling(24).std()
    h["bar_range_atr"] = (h["high"] - h["low"]) / h["atr"]
    h = h.join(period_opens(b15).resample("1h").last())
    t_open = pd.Series(h.index, index=h.index)
    t_close = pd.Series(h.index + pd.Timedelta("1h"), index=h.index)
    for dur, n, span, name in (("4h", 16, 50, "4h"), ("1D", 96, 20, "1d")):
        htf = resample(b15, dur, n)
        h[[f"c_{name}", f"ema_{name}"]] = htf_at(t_close, htf, dur, span, name).to_numpy()
        # ranges use bars closed before this 1h bar opened, so the breakout bar is never part of its own range
        h[[f"rng_hi_{name}", f"rng_w_{name}"]] = range_at(t_open, htf, dur, name).to_numpy()
    h[["rng_hi_1h", "rng_w_1h"]] = range_at(t_open, h, "1h", "1h").to_numpy()
    for name in ("1h", "4h", "1d"):
        h[f"lvl_rng_{name}"] = h[f"rng_hi_{name}"].where(h[f"rng_w_{name}"] <= RANGE_MAX_ATR)
    h["trend_up"] = (h["c_4h"] > h["ema_4h"]) & (h["c_1d"] > h["ema_1d"])
    return h


def label(lo, hi, cl, entry, stop, cost_pct):
    """Outcome of the plan from the bar after entry. Returns (label_2r, r_gross, r_net, bars_held)."""
    r = entry - stop
    first = lambda mask: int(mask.argmax()) if mask.any() else len(mask)
    i_stop, i_tp1 = first(lo <= stop), first(hi >= entry + TP1_R * r)
    n = len(lo)
    if i_stop <= i_tp1 and i_stop < n:
        lab, gross, held = 0, -1.0, i_stop
    elif i_tp1 < n:
        lo2, hi2 = lo[i_tp1 + 1:], hi[i_tp1 + 1:]
        i_be, i_tp2 = first(lo2 <= entry), first(hi2 >= entry + TP2_R * r)
        if i_be <= i_tp2 and i_be < len(lo2):
            rest, held = 0.0, i_tp1 + 1 + i_be
        elif i_tp2 < len(lo2):
            rest, held = TP2_R, i_tp1 + 1 + i_tp2
        else:
            rest, held = (cl[-1] - entry) / r, n - 1
        lab, gross = 1, 0.5 * TP1_R + 0.5 * rest
    else:
        lab, gross, held = 0, (cl[-1] - entry) / r, n - 1
    return lab, gross, gross - 2 * cost_pct * abs(entry) / r, held + 1


def candidates_for_pair(pair: str, b15: pd.DataFrame, months: set[str], cost_pct: float) -> list[dict]:
    ctx, events = {}, []
    for side in (1, -1):
        b = b15 if side == 1 else flip(b15)
        h = hourly_frame(b)
        ok = h["trend_up"] & h["atr"].notna()
        brk = pd.DataFrame({n: ok & (h["close"] > h[f"lvl_{n}"]) & (h["close"].shift() <= h[f"lvl_{n}"].shift())
                            for n in LEVELS})
        h["n_levels_broken"] = brk.sum(axis=1)
        imp = ok & (h["imp"] >= IMPULSE_ATR * h["atr"])
        for prio, n in enumerate(LEVELS):
            events += [(i, prio, side, "A", n) for i in np.flatnonzero(brk[n].to_numpy())]
        events += [(i, len(LEVELS), side, "B", "impulse") for i in np.flatnonzero(imp.to_numpy())]
        # plain arrays: the loop below runs once per event and must not touch pandas
        ctx[side] = ({c: b[c].to_numpy() for c in ("open", "high", "low", "close")},
                     {c: h[c].to_numpy() for c in h.columns})
    events.sort()   # by 1h bar, then level priority

    dates = b15.index.asi8                                 # 15m bar open times, ns since epoch (UTC)
    h_close = (h.index + pd.Timedelta("1h")).asi8          # 1h bar close times (same index for both sides)
    m15 = M15.value
    out, busy_until = [], -1
    for i, _, side, setup, level_type in events:
        if h_close[i] <= busy_until:
            continue
        k = int(np.searchsorted(dates, h_close[i]))
        if k + SCAN_BARS > len(dates) or dates[k + SCAN_BARS - 1] - dates[k] != (SCAN_BARS - 1) * m15:
            continue  # scan window runs off the data or over a gap
        B, H = ctx[side]
        w = slice(k, k + SCAN_BARS)
        atr = H["atr"][i]
        op, lo, cl = B["open"][w], B["low"][w], B["close"][w]
        low_so_far = np.minimum.accumulate(lo)
        if setup == "A":
            level = H[f"lvl_{level_type}"][i]
            dead = cl < level - FAIL_ATR * atr
            trig = (lo <= level + RETEST_ATR * atr) & (cl > level) & (cl > op)
        else:
            level = H["imp_high"][i]
            dead = low_so_far < H["close"][i] - FLAG_MAX_RETRACE * H["imp"][i]
            trig = (cl > level) & (np.arange(SCAN_BARS) >= FLAG_MIN_BARS)
        if not trig.any():
            continue
        j = int(trig.argmax())
        if dead[: j + 1].any():
            continue
        entry, stop = cl[j], low_so_far[j] - STOP_BUFFER_ATR * atr
        stop_pct = (entry - stop) / abs(entry)
        entry_time = pd.Timestamp(int(dates[k + j] + m15), tz="UTC")
        if not (STOP_PCT_MIN <= stop_pct <= STOP_PCT_MAX) or entry_time.strftime("%Y-%m") not in months:
            continue
        f = slice(k + j + 1, k + j + 1 + HORIZON_BARS)
        fd = dates[f]
        if len(fd) == 0:
            continue
        gap = np.flatnonzero(np.diff(fd) != m15)
        n = int(gap[0]) + 1 if len(gap) else len(fd)  # stop walking at the first gap
        lab, gross, net, held = label(B["low"][f][:n], B["high"][f][:n], B["close"][f][:n], entry, stop, cost_pct)
        busy_until = entry_time.value + held * m15
        local = entry_time.tz_convert(LOCAL_TZ)
        g = lambda c: H[c][i]
        out.append({
            "pair": pair, "entry_time": entry_time, "side": side, "setup": setup, "level_type": level_type,
            "entry": abs(entry), "stop": abs(stop), "stop_pct": stop_pct,
            "label_2r": lab, "r_gross": gross, "r_net": net, "bars_held": held,
            "exit_time": pd.Timestamp(int(busy_until), tz="UTC"),
            # features: all known at entry, all expressed in the direction of the trade
            "atr_pct": atr / abs(g("close")), "bars_to_trigger": j + 1,
            "dist_ema4h_atr": (g("c_4h") - g("ema_4h")) / atr, "dist_ema1d_pct": (g("c_1d") - g("ema_1d")) / abs(g("ema_1d")),
            "vol_rel": g("vol_rel"), "bar_range_atr": g("bar_range_atr"), "ret_24h": g("ret_24h"),
            "ret_7d": g("ret_7d"), "rv_24h": g("rv_24h"), "entry_ext_atr": (entry - level) / atr,
            "n_levels_broken": int(g("n_levels_broken")),
            **{f"dist_open_{x}_atr": (g("close") - g(f"lvl_open_{x}")) / atr for x in "dwmy"},
            **{f"rng_w_{x}": g(f"rng_w_{x}") for x in ("1h", "4h", "1d")},
            "hour": local.hour, "dow": local.dayofweek,
        })
    return out


def load_15m(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.as_unit("ns")  # the dataset stores milliseconds; keep one unit everywhere
    return df.drop_duplicates("date").set_index("date").sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--fee", type=float, default=0.0005, help="taker fee per side")
    ap.add_argument("--slippage", type=float, default=0.0002, help="assumed slippage per side")
    a = ap.parse_args()

    uni = pd.read_csv(a.data / "universe.csv", dtype={"month": str})
    rank = uni.set_index(["pair", "month"])["rank"]
    rows, missing, t0 = [], [], time.time()
    groups = list(uni.groupby("pair"))
    for n, (pair, g) in enumerate(groups, 1):
        f = a.data / "raw" / "futures" / "15m" / f"{pair}.parquet"
        if not f.exists():
            missing.append(pair)
            continue
        found = candidates_for_pair(pair, load_15m(f), set(g["month"]), a.fee + a.slippage)
        rows += found
        print(f"[{n:3d}/{len(groups)}] {pair:26s} {len(found):5d} candidates  ({time.time() - t0:5.0f}s)", flush=True)
    if missing:
        print(f"no 15m file for {len(missing)} pairs: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    if not rows:
        raise SystemExit("no candidates - check that the 15m files and universe.csv exist")
    c = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    c["universe_rank"] = [rank[(p, t.strftime("%Y-%m"))] for p, t in zip(c["pair"], c["entry_time"])]
    c.to_parquet(a.data / "candidates.parquet", index=False)

    print(f"{len(c)} candidates, {c['pair'].nunique()} pairs, {c['entry_time'].min():%Y-%m-%d} .. {c['entry_time'].max():%Y-%m-%d}")
    s = c.groupby(["setup", "level_type", "side"]).agg(n=("label_2r", "size"), hit_2r=("label_2r", "mean"), r_net=("r_net", "mean"),
                                         stop_pct=("stop_pct", "median"))
    print(s.round(3).to_string())
    print(f"all: hit_2r {c['label_2r'].mean():.3f}, mean r_net {c['r_net'].mean():+.3f} "
          f"(break-even hit rate for this plan is roughly 1/3 before costs)")


if __name__ == "__main__":
    main()
