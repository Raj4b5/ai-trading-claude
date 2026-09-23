#!/usr/bin/env python3
"""
Repeat-alert strategy: "quiet second alert + break of the previous high" entry test.

Question being tested
  When a stock went QUIET into its second Chartink multibagger alert, what happens
  to returns if, instead of buying the close of the alert session, we wait for the
  stock to BREAK THE PREVIOUS HIGH and make that breakout our entry?

Baseline (the current second-alert rules, unchanged)
  * signal   : the symbol's TRUE second alert day (3rd+ alerts never qualify)
  * liquidity: 20-session mean volume (incl. the alert day, min 10 bars) > 100,000
  * entry    : buy at the close of the alert session
  * exit     : close <= stop, stop = max(stop, peak_close - k * ATR14), peak and stop
               only ratchet up, checked on closes, never on the entry day;
               or 63 sessions held, whichever comes first
  * book     : 5 slots, size = min(equity / 5, cash), one entry per symbol ever,
               exits before entries, a skipped alert is gone (no queue),
               0.30% round-trip cost, same-day ties broken at random (N seeds)

Variants
  quiet    : ATR14(alert day) / ATR14(60 sessions earlier) < 1
             (volatility contracted: the stock went quiet into its 2nd alert)
  breakout : skip the alert-close buy; enter on the first session within W sessions
             after the alert that trades above a reference high
               ref = 'alert' -> the alert session's high
                     'hhN'   -> highest high of the N sessions ending on the alert day
                     'pause' -> the running post-alert high, broken only after the
                                stock has gone >= P sessions without a new high
             fill = 'stop'  -> buy-stop at the reference: max(open, ref)
                    'close' -> the session must CLOSE above ref; buy that close
             no break inside the window -> no trade

Eligibility (2nd alert + volume floor) is identical in every variant, so the
variants differ only in the quiet filter and the entry timing/price.

Input data
  --alerts : CSV/XLSX with columns symbol,date (every alert row, not just 2nd alerts;
             the script counts distinct alert sessions per symbol). Extra columns ignored.
             With --second-alerts, each row is already a symbol's second alert.
  --prices : a directory of <SYMBOL>.csv files, or one long CSV/parquet with a symbol
             column. Columns: date,open,high,low,close,volume. Prices must be
             split/bonus-adjusted (bhavcopy needs adjusting before use).

Usage
  python3 backtest.py --alerts alerts.csv --prices prices/ --start 2022-04-01 --end 2026-08-31
  python3 backtest.py --demo            # synthetic data, checks the machinery end to end
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- data

PRICE_COLS = ["date", "open", "high", "low", "close", "volume"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"price data missing columns {missing}")
    df = df[PRICE_COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return df


def load_prices(path: str, symbols: set[str] | None = None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if not fn.lower().endswith(".csv"):
                continue
            sym = os.path.splitext(fn)[0].upper()
            if symbols is not None and sym not in symbols:
                continue
            out[sym] = _normalise(pd.read_csv(os.path.join(path, fn)))
        return out
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    for sym, g in df.groupby("symbol"):
        if symbols is None or sym in symbols:
            out[sym] = _normalise(g)
    return out


def load_alerts(path: str, sheet: str | None = None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        a = pd.read_excel(path, sheet_name=sheet or 0)          # needs openpyxl
    else:
        a = pd.read_csv(path)
    a = a.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in a.columns})
    for want, alts in (("date", ("alert_date", "triggered_at", "datetime", "timestamp", "time")),
                       ("symbol", ("nsecode", "nse_code", "ticker", "tradingsymbol"))):
        if want not in a.columns:
            hit = next((c for c in alts if c in a.columns), None)
            if hit is None:
                raise ValueError(f"alerts file needs a '{want}' column, found {list(a.columns)}")
            a = a.rename(columns={hit: want})
    a["symbol"] = a["symbol"].astype(str).str.upper().str.strip()
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    return a[["symbol", "date"]].dropna()


# --------------------------------------------------------------------- indicators

def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ATR including the current bar; NaN until n true ranges exist."""
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    prev = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    atr = np.full(len(close), np.nan)
    if len(close) >= n:
        atr[n - 1] = tr[:n].mean()
        for i in range(n, len(close)):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


@dataclass
class Series:
    dates: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    vol20: np.ndarray


def prepare(df: pd.DataFrame, atr_n: int) -> Series:
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    vol20 = df["volume"].astype(float).rolling(20, min_periods=10).mean().to_numpy()
    return Series(df["date"].to_numpy(), df["open"].to_numpy(float), h, l, c,
                  wilder_atr(h, l, c, atr_n), vol20)


# ------------------------------------------------------------------------ signals

def given_second_alerts(alerts: pd.DataFrame, series: dict[str, Series]) -> pd.DataFrame:
    """Rows are already second alerts (one per symbol): just roll each to its session."""
    rows = []
    for r in alerts.sort_values("date").drop_duplicates("symbol").itertuples(index=False):
        s = series.get(r.symbol)
        if s is None:
            continue
        i = int(np.searchsorted(s.dates, np.datetime64(r.date, "ns")))
        if i < len(s.dates):
            rows.append((r.symbol, pd.Timestamp(s.dates[i]), i, np.nan))
    return pd.DataFrame(rows, columns=["symbol", "alert_date", "alert_i", "n_alert_sessions"])


def second_alerts(alerts: pd.DataFrame, series: dict[str, Series],
                  count_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per symbol: its TRUE 2nd alert, rolled to the next trading session."""
    a = alerts if count_from is None else alerts[alerts["date"] >= count_from]
    rows = []
    for sym, g in a.groupby("symbol"):
        s = series.get(sym)
        if s is None:
            continue
        # roll holiday-stamped alerts to the next session, then count distinct sessions
        idx = np.searchsorted(s.dates, g["date"].to_numpy(dtype="datetime64[ns]"), side="left")
        sessions = sorted(set(int(i) for i in idx if i < len(s.dates)))
        if len(sessions) >= 2:
            rows.append((sym, pd.Timestamp(s.dates[sessions[1]]), sessions[1], len(sessions)))
    return pd.DataFrame(rows, columns=["symbol", "alert_date", "alert_i", "n_alert_sessions"])


@dataclass(frozen=True)
class EntryRule:
    name: str
    quiet: bool = False
    breakout: str | None = None     # None | 'alert' | 'hh20' | 'hh60' | 'pause'
    window: int = 10
    fill: str = "stop"              # 'stop' | 'close'
    pause: int = 3                  # only for breakout='pause'


@dataclass
class Candidate:
    symbol: str
    alert_date: pd.Timestamp
    entry_date: pd.Timestamp
    fill: float
    exit_date: pd.Timestamp | None
    exit_px: float
    exit_reason: str
    held: int
    ret: float                      # net of the round-trip cost
    atr_ratio: float
    lag: int                        # sessions from alert to entry
    premium: float                  # entry fill / alert-session close - 1


def find_entry(s: Series, ai: int, rule: EntryRule) -> tuple[int, float] | None:
    n = len(s.c)
    if rule.breakout is None:
        return ai, s.c[ai]
    last = min(ai + rule.window, n - 1)
    if rule.breakout == "pause":
        hi, hi_i = s.h[ai], ai
        for j in range(ai + 1, last + 1):
            broke = s.h[j] > hi if rule.fill == "stop" else s.c[j] > hi
            if broke and j - hi_i - 1 >= rule.pause:
                return j, (max(s.o[j], hi) if rule.fill == "stop" else s.c[j])
            if s.h[j] > hi:
                hi, hi_i = s.h[j], j
        return None
    if rule.breakout == "alert":
        ref = s.h[ai]
    elif rule.breakout.startswith("hh"):
        look = int(rule.breakout[2:])
        ref = s.h[max(0, ai - look + 1): ai + 1].max()
    else:
        raise ValueError(rule.breakout)
    for j in range(ai + 1, last + 1):
        if rule.fill == "stop" and s.h[j] > ref:
            return j, max(s.o[j], ref)
        if rule.fill == "close" and s.c[j] > ref:
            return j, s.c[j]
    return None


def trade_path(s: Series, ei: int, fill: float, k_atr: float, cap: int) -> tuple[int, float, str]:
    """Close-based chandelier: peak = highest close since entry, stop only rises."""
    peak = max(fill, s.c[ei])
    stop = -np.inf
    for j in range(ei + 1, len(s.c)):
        peak = max(peak, s.c[j])
        if not np.isnan(s.atr[j]):
            stop = max(stop, peak - k_atr * s.atr[j])
        if s.c[j] <= stop:
            return j, s.c[j], "trail"
        if j - ei >= cap:
            return j, s.c[j], "cap"
    return len(s.c) - 1, s.c[-1], "open"


def build_candidates(sec: pd.DataFrame, series: dict[str, Series], rule: EntryRule,
                     p: argparse.Namespace) -> tuple[list[Candidate], dict]:
    cands, stats = [], defaultdict(int)
    for r in sec.itertuples(index=False):
        s, ai = series[r.symbol], r.alert_i
        if not (s.vol20[ai] > p.vol_floor):
            stats["fail_volume"] += 1
            continue
        stats["eligible"] += 1
        back = ai - p.quiet_lookback
        ratio = s.atr[ai] / s.atr[back] if back >= 0 and s.atr[back] > 0 else np.nan
        if rule.quiet and not ratio < p.quiet_max:
            stats["not_quiet" if not np.isnan(ratio) else "quiet_unscored"] += 1
            continue
        hit = find_entry(s, ai, rule)
        if hit is None:
            stats["no_breakout"] += 1
            continue
        ei, fill = hit
        xi, xpx, why = trade_path(s, ei, fill, p.k_atr, p.cap)
        ret = xpx * (1 - p.cost / 2) / (fill * (1 + p.cost / 2)) - 1
        cands.append(Candidate(r.symbol, r.alert_date, pd.Timestamp(s.dates[ei]), float(fill),
                               None if why == "open" else pd.Timestamp(s.dates[xi]), float(xpx),
                               why, xi - ei, float(ret), float(ratio), ei - ai,
                               float(fill / s.c[ai] - 1)))
        stats["entered"] += 1
    return cands, dict(stats)


# ---------------------------------------------------------------------- portfolio

@dataclass
class Book:
    live: list
    cal: pd.DatetimeIndex
    close: np.ndarray
    sym_row: list
    entry_day: dict
    exit_day: list


def prepare_book(cands: list[Candidate], series: dict[str, Series], cal: pd.DatetimeIndex,
                 start: pd.Timestamp, end: pd.Timestamp) -> Book:
    live = [c for c in cands if start <= c.entry_date <= end]
    wcal = cal[(cal >= start) & (cal <= end)]
    syms = sorted({c.symbol for c in live})
    row = {s: i for i, s in enumerate(syms)}
    close = np.full((len(syms), len(wcal)), np.nan)
    for s in syms:
        ser = pd.Series(series[s].c, index=pd.DatetimeIndex(series[s].dates))
        close[row[s]] = ser.reindex(ser.index.union(wcal)).ffill().reindex(wcal).to_numpy()
    day = {d: i for i, d in enumerate(wcal)}
    entry_day = defaultdict(list)
    for k, c in enumerate(live):
        entry_day[day[c.entry_date]].append(k)
    exit_day = [day[c.exit_date] if c.exit_date is not None and c.exit_date <= end else None for c in live]
    return Book(live, wcal, close, [row[c.symbol] for c in live], dict(entry_day), exit_day)


def run_book(b: Book, p: argparse.Namespace, seed: int) -> dict:
    if len(b.cal) < 2:
        return {"cagr": np.nan, "maxdd": np.nan, "trades": 0}
    rng = np.random.default_rng(seed)
    cash, pos, exits, traded, taken = p.capital, {}, defaultdict(list), set(), 0
    curve = np.empty(len(b.cal))
    for d in range(len(b.cal)):
        for k in exits.pop(d, ()):                       # exits run before entries
            cash += pos.pop(k) * b.live[k].exit_px * (1 - p.cost / 2)
        todays = b.entry_day.get(d)
        if todays:
            equity = cash + sum(q * b.close[b.sym_row[k], d] for k, q in pos.items())
            for j in rng.permutation(len(todays)):      # same-day ties: random order
                k = todays[j]
                c = b.live[k]
                if c.symbol in traded or len(pos) >= p.slots:
                    continue                             # skipped alerts are gone
                px = c.fill * (1 + p.cost / 2)
                q = math.floor(min(equity / p.slots, cash) / px)
                if q < 1:
                    continue
                cash -= q * px
                pos[k] = q
                traded.add(c.symbol)
                taken += 1
                if b.exit_day[k] is not None:
                    exits[b.exit_day[k]].append(k)
        curve[d] = cash + sum(q * b.close[b.sym_row[k], d] for k, q in pos.items())
    years = (b.cal[-1] - b.cal[0]).days / 365.25
    peak = np.maximum.accumulate(curve)
    return {"cagr": (curve[-1] / p.capital) ** (1 / years) - 1, "maxdd": float((curve / peak - 1).min()),
            "trades": taken}


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.2f}%"


def trade_stats(cands: list[Candidate]) -> dict:
    r = np.array([c.ret for c in cands], float)
    if not len(r):
        return {"n": 0, "mean": np.nan, "median": np.nan, "win": np.nan}
    return {"n": len(r), "mean": float(r.mean()), "median": float(np.median(r)), "win": float((r > 0).mean())}


def summarise(rule: EntryRule, cands: list[Candidate], stats: dict, series, cal, p,
              windows: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> dict:
    out = {"rule": asdict(rule), "funnel": stats, **trade_stats(cands),
           "median_lag": float(np.median([c.lag for c in cands])) if cands else np.nan,
           "median_premium": float(np.median([c.premium for c in cands])) if cands else np.nan,
           "exit_mix": {k: int(v) for k, v in pd.Series([c.exit_reason for c in cands]).value_counts().items()}}
    for label, a, b in windows:
        book = prepare_book(cands, series, cal, a, b)
        runs = [run_book(book, p, seed) for seed in range(p.seeds)]
        cg = np.array([x["cagr"] for x in runs], float)
        dd = np.array([x["maxdd"] for x in runs], float)
        out[label] = {"cagr_median": float(np.nanmedian(cg)), "cagr_p10": float(np.nanpercentile(cg, 10)),
                      "cagr_p90": float(np.nanpercentile(cg, 90)), "maxdd_median": float(np.nanmedian(dd)),
                      "trades_median": float(np.median([x["trades"] for x in runs])),
                      "per_trade": trade_stats(book.live)}
    return out


def paired(base: list[Candidate], other: list[Candidate], big: float = 0.30) -> dict:
    """Same symbols, different entry: what the variant changes trade by trade."""
    b = {c.symbol: c.ret for c in base}
    o = {c.symbol: c.ret for c in other}
    common = sorted(b.keys() & o.keys())
    diff = np.array([o[s] - b[s] for s in common], float)
    winners = [s for s, r in b.items() if r >= big]
    return {"common": len(common),
            "mean_diff": float(diff.mean()) if len(diff) else np.nan,
            "median_diff": float(np.median(diff)) if len(diff) else np.nan,
            "base_big_winners": len(winners),
            "big_winners_missed": sum(1 for s in winners if s not in o)}


def default_rules() -> list[EntryRule]:
    return [
        EntryRule("A0 baseline: buy 2nd-alert close"),
        EntryRule("A1 quiet only: buy 2nd-alert close", quiet=True),
        EntryRule("B0 all 2nd alerts, break alert-day high (10d)", breakout="alert"),
        EntryRule("B1 quiet + break alert-day high (10d, stop)", quiet=True, breakout="alert"),
        EntryRule("B2 quiet + close above alert-day high (10d)", quiet=True, breakout="alert", fill="close"),
        EntryRule("B3 quiet + break alert-day high (5d)", quiet=True, breakout="alert", window=5),
        EntryRule("B4 quiet + break alert-day high (20d)", quiet=True, breakout="alert", window=20),
        EntryRule("B5 quiet + break 20-session high (10d)", quiet=True, breakout="hh20"),
        EntryRule("B6 quiet + break 60-session base high (20d)", quiet=True, breakout="hh60", window=20),
        EntryRule("C0 all, pause >=3d then break post-alert high (20d)", breakout="pause", window=20),
        EntryRule("C1 quiet + pause >=3d then break post-alert high (20d)", quiet=True, breakout="pause",
                  window=20),
    ]


def report(results: list[dict], windows) -> str:
    win = lambda x: "n/a" if math.isnan(x) else f"{x * 100:.0f}%"
    lines = ["| variant | candidates | mean / trade | median | win | lag | entry vs alert close | "
             + " | ".join(f"CAGR {w[0]} (p10..p90) | maxDD {w[0]}" for w in windows) + " |",
             "|---|---|---|---|---|---|---|" + "---|---|" * len(windows)]
    for r in results:
        cells = [r["rule"]["name"], str(r["n"]), pct(r["mean"]), pct(r["median"]), win(r["win"]),
                 "n/a" if math.isnan(r["median_lag"]) else f"{r['median_lag']:.0f}d", pct(r["median_premium"])]
        for w in windows:
            x = r[w[0]]
            cells += [f"{pct(x['cagr_median'])} ({pct(x['cagr_p10'])}..{pct(x['cagr_p90'])})",
                      pct(x["maxdd_median"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Paired against A0 (same symbols, only the entry differs):", "",
              "| variant | symbols in both | mean diff / trade | median diff | A0 winners >= +30% missed |",
              "|---|---|---|---|---|"]
    for r in results[1:]:
        x = r["paired_vs_A0"]
        lines.append(f"| {r['rule']['name']} | {x['common']} | {pct(x['mean_diff'])} | {pct(x['median_diff'])} "
                     f"| {x['big_winners_missed']} of {x['base_big_winners']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- demo

def make_demo(dirpath: str, n_sym: int = 120, seed: int = 7) -> tuple[str, str]:
    """Random-walk prices with regime drift and random alerts: tests the machinery only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-06-01", "2026-08-31")
    os.makedirs(os.path.join(dirpath, "prices"), exist_ok=True)
    alerts = []
    for i in range(n_sym):
        sym = f"SYM{i:03d}"
        vol = rng.uniform(0.012, 0.035)
        r = rng.normal(0.0004, vol, len(dates))
        c = 100 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, vol / 3, len(dates)))
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, len(dates))))
        l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, len(dates))))
        v = rng.lognormal(np.log(rng.uniform(5e4, 2e6)), 0.5, len(dates)).round()
        pd.DataFrame({"date": dates, "open": o, "high": h, "low": l, "close": c, "volume": v}) \
            .to_csv(os.path.join(dirpath, "prices", f"{sym}.csv"), index=False)
        for d in rng.choice(dates[120:], size=rng.integers(0, 5), replace=False):
            alerts.append((sym, pd.Timestamp(d).date()))
    ap = os.path.join(dirpath, "alerts.csv")
    pd.DataFrame(alerts, columns=["symbol", "date"]).to_csv(ap, index=False)
    return ap, os.path.join(dirpath, "prices")


# ------------------------------------------------------------------- reconciliation

def check_known(path: str, prices: str, p: argparse.Namespace) -> int:
    """Replay trades the research engine reported and compare exits (same prices, same rules)."""
    known = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    raw = load_prices(prices, set(known["symbol"].str.upper()))
    bad = 0
    for r in known.itertuples(index=False):
        df = raw.get(r.symbol.upper())
        if df is None:
            print(f"{r.symbol:12s} no prices")
            bad += 1
            continue
        s = prepare(df, p.atr_n)
        ei = int(np.searchsorted(s.dates, np.datetime64(r.entry_date, "ns")))
        if ei >= len(s.dates) or pd.Timestamp(s.dates[ei]) != r.entry_date:
            print(f"{r.symbol:12s} entry date {r.entry_date.date()} not in price data")
            bad += 1
            continue
        xi, px, why = trade_path(s, ei, r.entry_px, r.k_atr, p.cap)
        got = pd.Timestamp(s.dates[xi])
        ok = got == r.exit_date and why == r.exit_reason and abs(px / r.exit_px - 1) < 0.005
        bad += not ok
        print(f"{r.symbol:12s} expected {r.exit_date.date()} {r.exit_px:>9.2f} {r.exit_reason:5s} | "
              f"got {got.date()} {px:>9.2f} {why:5s} {'ok' if ok else 'MISMATCH'}")
    print(f"{len(known) - bad} of {len(known)} reconcile")
    return 1 if bad else 0


# ---------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", help="csv or xlsx of every alert row (symbol, date)")
    ap.add_argument("--alerts-sheet", default=None, help="xlsx sheet name, e.g. 'Multibagger Alerts'")
    ap.add_argument("--prices")
    ap.add_argument("--start", default="2022-04-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--split", default="2024-07-01", help="regime split date for the early/late eras")
    ap.add_argument("--count-from", default=None, help="count 2nd alerts only from this date")
    ap.add_argument("--second-alerts", action="store_true",
                    help="--alerts rows are already second alerts (one per symbol), not every alert")
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--k-atr", type=float, default=5.5, help="trail multiple, e.g. 4 or 5.5")
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--cap", type=int, default=63)
    ap.add_argument("--cost", type=float, default=0.003, help="round trip")
    ap.add_argument("--vol-floor", type=float, default=100_000)
    ap.add_argument("--quiet-lookback", type=int, default=60)
    ap.add_argument("--quiet-max", type=float, default=1.0)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--check-known", metavar="CSV",
                    help="replay known engine trades (e.g. known_trades.csv) against --prices and exit")
    p = ap.parse_args(argv)

    if p.check_known:
        if not p.prices:
            ap.error("--check-known needs --prices")
        return check_known(p.check_known, p.prices, p)

    if p.demo:
        p.alerts, p.prices = make_demo(os.path.join(p.out, "demo_data"))
        p.seeds = min(p.seeds, 20)
    if not (p.alerts and p.prices):
        ap.error("--alerts and --prices are required (or --demo)")

    alerts = load_alerts(p.alerts, p.alerts_sheet)
    raw = load_prices(p.prices, set(alerts["symbol"]))
    series = {s: prepare(df, p.atr_n) for s, df in raw.items() if len(df) > 30}
    cal = pd.DatetimeIndex(np.unique(np.concatenate([s.dates for s in series.values()])))
    sec = (given_second_alerts(alerts, series) if p.second_alerts else
           second_alerts(alerts, series, pd.Timestamp(p.count_from) if p.count_from else None))
    start, end, split = pd.Timestamp(p.start), pd.Timestamp(p.end), pd.Timestamp(p.split)
    sec = sec[(sec["alert_date"] >= start) & (sec["alert_date"] <= end)].reset_index(drop=True)
    windows = [("full", start, end), ("early", start, split - pd.Timedelta(days=1)), ("late", split, end)]
    print(f"{len(alerts)} alert rows, {alerts['symbol'].nunique()} symbols, "
          f"{len(series)} with prices, {len(sec)} second alerts in window", file=sys.stderr)

    results, trades, base = [], [], None
    for rule in default_rules():
        cands, stats = build_candidates(sec, series, rule, p)
        res = summarise(rule, cands, stats, series, cal, p, windows)
        base = cands if base is None else base
        res["paired_vs_A0"] = paired(base, cands)
        results.append(res)
        trades += [{**asdict(c), "variant": rule.name} for c in cands]
        print(f"done: {rule.name}  ({len(cands)} candidate trades)", file=sys.stderr)

    os.makedirs(p.out, exist_ok=True)
    pd.DataFrame(trades).to_csv(os.path.join(p.out, "candidate_trades.csv"), index=False)
    with open(os.path.join(p.out, "summary.json"), "w") as f:
        json.dump({"params": {k: v for k, v in vars(p).items()}, "results": results}, f, indent=2, default=str)
    table = report(results, windows)
    with open(os.path.join(p.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
