#!/usr/bin/env python3
"""
Daily candles from Zerodha Kite Connect for the stocks in an alerts file, written the way
backtest.py reads prices: one <SYMBOL>.csv per stock with date,open,high,low,close,volume.

Needs
  * a Kite Connect app with historical-data access
  * network access to api.kite.trade
  * two environment variables, set in the environment's settings (never in code or chat):
      KITE_API_KEY       the app's API key
      KITE_ACCESS_TOKEN  a login session's access token; Kite expires these every morning

Usage
  python3 fetch_kite.py --alerts alerts.xlsx --alerts-sheet "Multibagger Alerts" --out data/prices
  python3 fetch_kite.py --symbols RELIANCE,TCS --from 2021-01-01 --out data/prices

By default only stocks with at least two distinct alert days are fetched (every stock that can
have a second alert), from 200 days before the first alert so the ATR and the quiet lookback
have history. Files already in --out are kept unless --refresh, so an interrupted run resumes.
Each stock gets a row in the report (default <out>_report.csv): the Kite instrument used, the
bars fetched, and any overnight gap big enough to be an unadjusted split or bonus. Kite adjusts
its candles for those; the report shows whether that held.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

BASE = "https://api.kite.trade"
SERIES = ("", "-BE", "-BZ", "-SM", "-ST")      # EQ first, then trade-to-trade and SME series
MAX_DAYS = 1900                                 # Kite serves at most 2000 days of daily candles a call


class KiteError(RuntimeError):
    """One request failed; the other stocks can still be fetched."""


class KiteAuthError(KiteError):
    """The token or the app's permissions are wrong, so every request will fail."""


def explain(path: str, r) -> str:
    try:
        body = r.json()
        kind, msg = body.get("error_type", ""), body.get("message", "")
    except ValueError:
        kind, msg = "", r.text[:200]
    if kind == "TokenException":
        return ("Kite rejected the access token (TokenException). Tokens expire every morning: log in "
                "again, update KITE_ACCESS_TOKEN in the environment's settings, and start a new session.")
    if kind == "PermissionException":
        return ("Kite refused this call for the app (PermissionException). Historical candles need a "
                "Kite Connect app with historical-data access.")
    return f"{path}: HTTP {r.status_code} {kind} {msg}".strip()


class Kite:
    """Minimal Kite Connect v3 client: the auth header, <= 3 requests a second, retries on 429/5xx."""

    def __init__(self, api_key: str, access_token: str, get=requests.get, sleep=time.sleep,
                 min_gap: float = 0.35, tries: int = 5):
        self.headers = {"X-Kite-Version": "3", "Authorization": f"token {api_key}:{access_token}"}
        self.get, self.sleep, self.min_gap, self.tries = get, sleep, min_gap, tries
        self.last = -1e9

    def call(self, path: str, params: dict | None = None):
        why = ""
        for attempt in range(self.tries):
            wait = self.min_gap - (time.monotonic() - self.last)
            if wait > 0:
                self.sleep(wait)
            self.last = time.monotonic()
            try:
                r = self.get(BASE + path, params=params, headers=self.headers, timeout=30)
            except requests.RequestException as e:
                why = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    return r
                if r.status_code == 403:
                    raise KiteAuthError(explain(path, r))
                if r.status_code != 429 and r.status_code < 500:
                    raise KiteError(explain(path, r))
                why = f"HTTP {r.status_code}"
            self.sleep(2 ** attempt)
        raise KiteError(f"{path}: gave up after {self.tries} tries ({why})")


def instruments(kite: Kite) -> pd.DataFrame:
    """Today's NSE cash-market instruments (indices and other segments dropped)."""
    df = pd.read_csv(io.StringIO(kite.call("/instruments/NSE").text))
    return df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]


def resolve(symbols, inst: pd.DataFrame) -> dict[str, tuple[int, str]]:
    """NSE symbol -> (instrument_token, tradingsymbol), trying the EQ series first."""
    by_ts = dict(zip(inst["tradingsymbol"].astype(str).str.upper(), inst["instrument_token"].astype(int)))
    out = {}
    for sym in symbols:
        ts = next((sym + sfx for sfx in SERIES if sym + sfx in by_ts), None)
        if ts is not None:
            out[sym] = (by_ts[ts], ts)
    return out


def chunks(start: pd.Timestamp, end: pd.Timestamp, days: int = MAX_DAYS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out, a = [], start
    while a <= end:
        b = min(a + pd.Timedelta(days=days - 1), end)
        out.append((a, b))
        a = b + pd.Timedelta(days=1)
    return out


def candles(kite: Kite, token: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for a, b in chunks(start, end):
        r = kite.call(f"/instruments/historical/{token}/day",
                      {"from": a.strftime("%Y-%m-%d 00:00:00"), "to": b.strftime("%Y-%m-%d 23:59:59")})
        rows += [c[:6] for c in r.json()["data"]["candles"]]
    df = pd.DataFrame(rows, columns=bt.PRICE_COLS)
    df["date"] = df["date"].astype(str).str[:10]           # "2024-01-02T00:00:00+0530" -> "2024-01-02"
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def split_like_gaps(df: pd.DataFrame, lo: float = 0.7, hi: float = 1.45) -> list[str]:
    """Days that open so far from the previous close that the move looks like an unadjusted
    split or bonus. NSE price bands stop most real moves at 20%."""
    ratio = df["open"] / df["close"].shift(1)
    return df.loc[(ratio < lo) | (ratio > hi), "date"].astype(str).tolist()


def alert_symbols(alerts: pd.DataFrame, all_symbols: bool = False) -> list[str]:
    days = alerts.groupby("symbol")["date"].nunique()
    return sorted(days.index if all_symbols else days[days >= 2].index)


def fetch(kite: Kite, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, out: str, report: str,
          refresh: bool = False) -> pd.DataFrame:
    os.makedirs(out, exist_ok=True)
    tokens = resolve(symbols, instruments(kite))
    rows = []
    for n, sym in enumerate(symbols, 1):
        path = os.path.join(out, f"{sym}.csv")
        row = {"symbol": sym, "status": "", "tradingsymbol": "", "instrument_token": "", "bars": 0,
               "first_date": "", "last_date": "", "split_like_gaps": ""}
        df = None
        if sym not in tokens:
            row["status"] = "not on Kite today (delisted, renamed or not NSE)"
        else:
            row["instrument_token"], row["tradingsymbol"] = tokens[sym]
            if os.path.exists(path) and not refresh:
                df, row["status"] = pd.read_csv(path), "kept"
            else:
                try:
                    df = candles(kite, tokens[sym][0], start, end)
                except KiteAuthError:
                    raise
                except KiteError as e:
                    row["status"] = f"error: {e}"
                else:
                    df.to_csv(path, index=False)
                    row["status"] = "fetched"
        if df is not None and len(df):
            row.update(bars=len(df), first_date=df["date"].iloc[0], last_date=df["date"].iloc[-1],
                       split_like_gaps=";".join(split_like_gaps(df)))
        rows.append(row)
        if n % 50 == 0 or n == len(symbols):
            print(f"{n}/{len(symbols)} stocks", file=sys.stderr)
    rep = pd.DataFrame(rows)
    rep.to_csv(report, index=False)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", help="csv or xlsx of every alert row (symbol, date), as for backtest.py")
    ap.add_argument("--alerts-sheet", default=None)
    ap.add_argument("--symbols", help="comma-separated NSE symbols, instead of --alerts")
    ap.add_argument("--all-symbols", action="store_true", help="every alerted stock, not only 2+ alert days")
    ap.add_argument("--from", dest="start", default=None, help="default: 200 days before the first alert")
    ap.add_argument("--to", dest="end", default=None, help="default: today")
    ap.add_argument("--out", default="data/prices")
    ap.add_argument("--report", default=None, help="default: <out>_report.csv")
    ap.add_argument("--refresh", action="store_true", help="download again even if the file exists")
    p = ap.parse_args(argv)
    key, token = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_ACCESS_TOKEN")
    if not (key and token):
        ap.error("KITE_API_KEY and KITE_ACCESS_TOKEN must be set in the environment's settings")
    if bool(p.alerts) == bool(p.symbols):
        ap.error("give either --alerts or --symbols")

    first = None
    if p.alerts:
        alerts = bt.load_alerts(p.alerts, p.alerts_sheet)
        symbols, first = alert_symbols(alerts, p.all_symbols), alerts["date"].min()
    else:
        symbols = sorted({s.strip().upper() for s in p.symbols.split(",") if s.strip()})
    start = (pd.Timestamp(p.start) if p.start else
             first - pd.Timedelta(days=200) if first is not None else pd.Timestamp("2021-01-01"))
    end = pd.Timestamp(p.end) if p.end else pd.Timestamp.today().normalize()
    report = p.report or p.out.rstrip("/\\") + "_report.csv"
    print(f"{len(symbols)} stocks, {start.date()} to {end.date()}", file=sys.stderr)
    try:
        rep = fetch(Kite(key, token), symbols, start, end, p.out, report, p.refresh)
    except KiteAuthError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 2
    status = rep["status"].str.split(":").str[0].value_counts()
    gaps = int((rep["split_like_gaps"] != "").sum())
    print(status.to_string(), file=sys.stderr)
    print(f"{gaps} stocks with a split-like overnight gap (see {report}); a handful is normal, "
          f"many means the candles are not adjusted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
