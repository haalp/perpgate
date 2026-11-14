#!/usr/bin/env python3
"""Estimated liquidation map for one pair, rebuilt bar by bar from open interest and price.

This is a model, not exchange data: no venue publishes where positions get liquidated. The method is
the usual heatmap construction:
  * open interest rises in a bar  -> that many contracts were opened at the bar's close, one long and
    one short side each, spread over a fixed leverage mix; each slice gets a liquidation price;
  * open interest falls           -> every resting slice shrinks by the same proportion;
  * price trades through a level  -> the slices resting there are removed (they were liquidated).

The leverage mix is fixed up front and must not be tuned on backtest results.

    from liq_map import liq_features
    f = liq_features(bars15, oi)     # one row per 15m bar, known at that bar's close
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVERAGE_MIX = {5: 0.10, 10: 0.25, 20: 0.25, 25: 0.15, 50: 0.15, 100: 0.10}  # weights sum to 1
MAINT_MARGIN = 0.005      # rough maintenance margin, moves the liquidation price slightly toward entry
BIN = 0.0025              # price bucket width (0.25%, log scale)
NEAR, WIDE = 0.05, 0.10   # windows around price: "nearby" mass within 5%, peak search within 10%


def liq_features(bars: pd.DataFrame, oi: pd.Series) -> pd.DataFrame:
    """bars: 15m OHLC indexed by bar open time (UTC). oi: open interest in contracts, any finer or equal
    frequency, indexed by timestamp. Returns per-bar features that use data up to that bar's close only:

      liq_above_5pct / liq_below_5pct   short / long liquidation size within 5% of price, as a share of OI
      liq_peak_above_pct / _below_pct   distance (%) to the largest cluster within 10% above / below
      liq_imbalance                     (above - below) / (above + below) within 5%; +1 = all above
    """
    close_time = bars.index + pd.Timedelta("15min")
    oi = oi.dropna().sort_index()
    x = pd.merge_asof(pd.DataFrame({"t": pd.DatetimeIndex(close_time).as_unit("ns")}),
                      pd.DataFrame({"ts": pd.DatetimeIndex(oi.index).as_unit("ns"), "oi": oi.to_numpy()}),
                      left_on="t", right_on="ts")
    oi_bar = x["oi"].where(x["t"] - x["ts"] <= pd.Timedelta("1h"))  # no fresh OI -> no estimate

    step = np.log1p(BIN)
    lo_edge = np.log(bars["low"].min()) + np.log(1 - 1 / min(LEVERAGE_MIX)) - step
    lo_edge = np.floor(lo_edge / step) * step   # fixed grid, so buckets do not depend on how much data is loaded
    nb = int((np.log(bars["high"].max()) + np.log(1 + 1 / min(LEVERAGE_MIX)) - lo_edge) / step) + 3
    to_bin = lambda p: int((np.log(p) - lo_edge) / step)
    longs, shorts = np.zeros(nb), np.zeros(nb)   # resting long / short liquidation size per price bucket
    lev = np.array(list(LEVERAGE_MIX))
    w = np.array(list(LEVERAGE_MIX.values()))
    d_long = np.round(np.log(1 - 1 / lev + MAINT_MARGIN) / step).astype(int)    # bucket offsets from entry
    d_short = np.round(np.log(1 + 1 / lev - MAINT_MARGIN) / step).astype(int)
    k_near, k_wide = int(np.log1p(NEAR) / step), int(np.log1p(WIDE) / step)

    hi, lo, cl, o = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(), oi_bar.to_numpy()
    out = np.full((len(bars), 5), np.nan)
    prev = np.nan
    for i in range(len(bars)):
        if np.isnan(o[i]):
            longs[:], shorts[:], prev = 0.0, 0.0, np.nan   # OI gap: start the map again
            continue
        b = to_bin(cl[i])
        if not np.isnan(prev):
            d = o[i] - prev
            if d > 0:
                np.add.at(longs, b + d_long, d * w)
                np.add.at(shorts, b + d_short, d * w)
            elif d < 0 and prev > 0:
                longs *= o[i] / prev
                shorts *= o[i] / prev
        prev = o[i]
        longs[to_bin(lo[i]):] = 0.0          # long liquidations at or above the low were hit
        shorts[: to_bin(hi[i]) + 1] = 0.0    # short liquidations at or below the high were hit
        if o[i] <= 0:
            continue
        above, below = shorts[b + 1: b + 1 + k_near].sum(), longs[max(b - k_near, 0): b].sum()
        wa, wb = shorts[b + 1: b + 1 + k_wide], longs[max(b - k_wide, 0): b][::-1]
        out[i] = (above / o[i], below / o[i],
                  (wa.argmax() + 1) * BIN * 100 if wa.any() else np.nan,
                  (wb.argmax() + 1) * BIN * 100 if wb.any() else np.nan,
                  (above - below) / (above + below) if above + below > 0 else 0.0)
    return pd.DataFrame(out, index=bars.index, columns=["liq_above_5pct", "liq_below_5pct", "liq_peak_above_pct",
                                                        "liq_peak_below_pct", "liq_imbalance"])
