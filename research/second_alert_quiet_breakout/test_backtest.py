#!/usr/bin/env python3
"""Hand-checkable tests for backtest.py. Run: python3 test_backtest.py (or pytest)."""

import argparse

import numpy as np
import pandas as pd

import backtest as bt


def series_from(closes, highs=None, opens=None, lows=None, vol=500_000, start="2024-01-01"):
    c = np.asarray(closes, float)
    h = np.asarray(highs if highs is not None else c * 1.01, float)
    o = np.asarray(opens if opens is not None else c, float)
    l = np.asarray(lows if lows is not None else c * 0.99, float)
    df = pd.DataFrame({"date": pd.bdate_range(start, periods=len(c)), "open": o, "high": h,
                       "low": l, "close": c, "volume": vol})
    return bt.prepare(df, 14)


def params(**kw):
    base = dict(slots=5, k_atr=4.0, cap=63, cost=0.003, vol_floor=100_000, quiet_lookback=60,
                quiet_max=1.0, capital=500_000, seeds=50)
    base.update(kw)
    return argparse.Namespace(**base)


def test_wilder_atr_constant_range():
    s = series_from([100.0] * 40, highs=[101.0] * 40, lows=[99.0] * 40)
    assert np.isnan(s.atr[12]) and abs(s.atr[13] - 2.0) < 1e-12 and abs(s.atr[-1] - 2.0) < 1e-12


def test_trail_exit_on_close_below_ratchet():
    # flat 100 with range 2 -> ATR 2; rally to 120, then fall: stop = 120 - 4*ATR
    closes = [100.0] * 30 + [104, 108, 112, 116, 120] + [118, 115, 112, 110, 100]
    s = series_from(closes, highs=np.array(closes) + 1, lows=np.array(closes) - 1)
    ei = 29
    xi, px, why = bt.trade_path(s, ei, s.c[ei], 4.0, 63)
    peak = 120.0
    stops = [peak - 4 * s.atr[j] for j in range(34, len(closes))]
    first = next(j for j in range(35, len(closes)) if s.c[j] <= max(stops[: j - 34 + 1]))
    assert why == "trail" and xi == first and px == s.c[first]


def test_cap_exit_at_63_sessions():
    closes = 100 * np.cumprod(np.full(120, 1.002))       # steady grind up, never stops out
    s = series_from(closes)
    xi, _, why = bt.trade_path(s, 20, s.c[20], 4.0, 63)
    assert why == "cap" and xi - 20 == 63


def test_never_exits_on_entry_day_and_open_at_end():
    s = series_from([100.0] * 40)
    xi, px, why = bt.trade_path(s, 39, 100.0, 4.0, 63)
    assert why == "open" and xi == 39


def test_breakout_modes():
    closes = [100.0] * 30
    highs = [101.0] * 30
    opens = [100.0] * 30
    highs[26], closes[26] = 101.5, 100.5            # day 26 pokes above 101 but closes below it
    highs[27], opens[27], closes[27] = 106.0, 104.0, 105.0   # gap above the alert high
    s = series_from(closes, highs=highs, opens=opens)
    ai = 25                                        # alert-day high = 101
    stop_rule = bt.EntryRule("x", breakout="alert", window=10, fill="stop")
    close_rule = bt.EntryRule("x", breakout="alert", window=10, fill="close")
    assert bt.find_entry(s, ai, stop_rule) == (26, 101.0)       # buy-stop fills at the level
    assert bt.find_entry(s, ai, close_rule) == (27, 105.0)      # needs a close above 101
    assert bt.find_entry(s, ai, bt.EntryRule("x", breakout="alert", window=1)) == (26, 101.0)
    s2 = series_from(closes[:26] + [100.0] * 4, highs=[101.0] * 30)
    assert bt.find_entry(s2, ai, stop_rule) is None             # never breaks -> no trade


def test_gap_fill_uses_open():
    closes, highs, opens = [100.0] * 30, [101.0] * 30, [100.0] * 30
    highs[26], opens[26], closes[26] = 110.0, 108.0, 109.0
    s = series_from(closes, highs=highs, opens=opens)
    assert bt.find_entry(s, 25, bt.EntryRule("x", breakout="alert")) == (26, 108.0)


def test_pause_breakout_waits_for_rest():
    highs = [101.0] * 25 + [103, 105, 107, 106, 106.5, 106.8, 108, 110]
    closes = [h - 1 for h in highs]
    s = series_from(closes, highs=highs)
    rule = bt.EntryRule("x", breakout="pause", window=20, pause=3)
    # alert at 24 (high 101); 25-27 make new highs with no pause; 28-30 rest below 107; 31 breaks
    assert bt.find_entry(s, 24, rule) == (31, 107.0)


def test_second_alert_counts_distinct_sessions_and_rolls_holidays():
    s = series_from([100.0] * 50, start="2024-01-01")
    d = pd.DatetimeIndex(s.dates)
    monday = next(i for i in range(6, 20) if d[i].weekday() == 0)
    alerts = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA", "AAA"],
                           "date": [d[5], d[5], d[monday] - pd.Timedelta(days=1), d[30]]})
    # duplicate same-day alerts count once; a Sunday-stamped alert rolls to Monday
    sec = bt.second_alerts(alerts, {"AAA": s})
    assert len(sec) == 1
    row = sec.iloc[0]
    assert row.alert_i == monday and row.n_alert_sessions == 3


def test_slots_bind_and_ties_are_random():
    # 8 symbols all alert on the same day; 5 slots -> exactly 5 entries, varying by seed
    series, rows = {}, []
    for i in range(8):
        closes = 100 * np.cumprod(np.full(150, 1 + 0.0005 * (i + 1)))
        series[f"S{i}"] = series_from(closes)
        rows.append((f"S{i}", pd.Timestamp(series[f"S{i}"].dates[80]), 80, 2))
    sec = pd.DataFrame(rows, columns=["symbol", "alert_date", "alert_i", "n_alert_sessions"])
    p = params()
    cands, stats = bt.build_candidates(sec, series, bt.EntryRule("base"), p)
    assert stats["entered"] == 8
    cal = pd.DatetimeIndex(series["S0"].dates)
    book = bt.prepare_book(cands, series, cal, cal[0], cal[-1])
    runs = [bt.run_book(book, p, seed) for seed in range(30)]
    assert all(r["trades"] == 5 for r in runs)
    assert len({round(r["cagr"], 10) for r in runs}) > 1


def test_one_entry_per_symbol_and_cash_never_negative():
    s = series_from(100 * np.cumprod(np.full(200, 1.001)))
    c1 = bt.Candidate("AAA", None, pd.Timestamp(s.dates[50]), s.c[50], pd.Timestamp(s.dates[60]),
                      s.c[60], "trail", 10, 0.0, np.nan, 0, 0.0)
    c2 = bt.Candidate("AAA", None, pd.Timestamp(s.dates[70]), s.c[70], None, s.c[-1], "open",
                      0, 0.0, np.nan, 0, 0.0)
    cal = pd.DatetimeIndex(s.dates)
    book = bt.prepare_book([c1, c2], {"AAA": s}, cal, cal[0], cal[-1])
    assert bt.run_book(book, params(), 0)["trades"] == 1


def test_exit_frees_slot_same_day():
    s = {f"S{i}": series_from(np.full(100, 100.0)) for i in range(2)}
    cal = pd.DatetimeIndex(s["S0"].dates)
    a = bt.Candidate("S0", None, cal[10], 100.0, cal[20], 100.0, "trail", 10, 0.0, np.nan, 0, 0.0)
    b = bt.Candidate("S1", None, cal[20], 100.0, None, 100.0, "open", 0, 0.0, np.nan, 0, 0.0)
    book = bt.prepare_book([a, b], s, cal, cal[0], cal[-1])
    assert bt.run_book(book, params(slots=1), 0)["trades"] == 2


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
