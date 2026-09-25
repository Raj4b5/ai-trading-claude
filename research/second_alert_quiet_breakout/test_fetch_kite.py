#!/usr/bin/env python3
"""Offline tests for fetch_kite.py against a fake Kite server. Run: python3 test_fetch_kite.py"""

import json
import os
import tempfile

import pandas as pd

import backtest as bt
import fetch_kite as fk

INSTRUMENTS = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,tick_size,lot_size,"
    "instrument_type,segment,exchange\n"
    "738561,2885,RELIANCE,RELIANCE INDUSTRIES,0,,0,0.05,1,EQ,NSE,NSE\n"
    "111,1,XYZ-BE,XYZ LTD,0,,0,0.05,1,EQ,NSE,NSE\n"
    "256265,1001,NIFTY 50,NIFTY 50,0,,0,0,0,EQ,INDICES,NSE\n"
    "222,2,M&M,MAHINDRA & MAHINDRA,0,,0,0.05,1,EQ,NSE,NSE\n")


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


def history(url, params):
    """Business-day candles for the requested range, with prices that encode the date."""
    days = pd.bdate_range(params["from"][:10], params["to"][:10])
    return Resp(200, {"status": "success", "data": {"candles": [
        [f"{d.date()}T00:00:00+0530", 100.0, 101.0, 99.0, 100.5, 5000] for d in days]}})


def fake(routes, calls=None, sleeps=None):
    def get(url, params=None, headers=None, timeout=None):
        if calls is not None:
            calls.append((url, params, headers))
        for key, resp in routes:
            if key in url:
                if isinstance(resp, list):
                    return resp.pop(0)
                return resp(url, params) if callable(resp) else resp
        raise AssertionError(f"unexpected url {url}")
    return fk.Kite("key", "tok", get=get, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_resolve_takes_eq_then_trade_to_trade_series():
    inst = fk.instruments(fake([("/instruments/NSE", Resp(200, INSTRUMENTS))]))
    got = fk.resolve(["RELIANCE", "XYZ", "NIFTY 50", "NOPE", "M&M"], inst)
    assert got == {"RELIANCE": (738561, "RELIANCE"), "XYZ": (111, "XYZ-BE"), "M&M": (222, "M&M")}


def test_chunks_cover_the_range_once():
    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2026-09-25")
    ch = fk.chunks(start, end)
    assert ch[0][0] == start and ch[-1][1] == end and len(ch) == 2
    assert all(b - a <= pd.Timedelta(days=fk.MAX_DAYS - 1) for a, b in ch)
    assert all(ch[i + 1][0] - ch[i][1] == pd.Timedelta(days=1) for i in range(len(ch) - 1))


def test_candles_join_chunks_and_send_the_auth_header():
    calls = []
    kite = fake([("/historical/", history)], calls)
    df = fk.candles(kite, 738561, pd.Timestamp("2019-01-01"), pd.Timestamp("2026-09-25"))
    assert len(calls) == 2 and list(df.columns) == bt.PRICE_COLS
    assert df["date"].tolist() == [str(d.date()) for d in pd.bdate_range("2019-01-01", "2026-09-25")]
    url, params, headers = calls[0]
    assert url.endswith("/instruments/historical/738561/day") and params["from"] == "2019-01-01 00:00:00"
    assert headers == {"X-Kite-Version": "3", "Authorization": "token key:tok"}


def test_split_like_gaps_flags_a_bonus_not_a_band_move():
    df = pd.DataFrame({"date": ["d1", "d2", "d3", "d4"], "open": [100.0, 81.0, 40.0, 41.0],
                       "close": [100.0, 80.0, 40.5, 41.0]})
    assert fk.split_like_gaps(df) == ["d3"]                  # 1:1 bonus halves the price; -19% is a move


def test_expired_token_stops_with_a_clear_message():
    body = {"status": "error", "error_type": "TokenException", "message": "Incorrect access_token."}
    kite = fake([("/instruments/NSE", Resp(403, body))])
    try:
        fk.instruments(kite)
    except fk.KiteAuthError as e:
        assert "KITE_ACCESS_TOKEN" in str(e)
    else:
        raise AssertionError("expected KiteAuthError")


def test_rate_limit_is_retried_with_backoff():
    sleeps = []
    kite = fake([("/instruments/NSE", [Resp(429, "Too many requests"), Resp(200, INSTRUMENTS)])], sleeps=sleeps)
    assert len(fk.instruments(kite)) == 3
    assert 1 in sleeps                                       # first backoff is 2 ** 0 seconds


def test_fetch_writes_files_backtest_reads_and_resumes():
    calls = []
    routes = [("/instruments/NSE", Resp(200, INSTRUMENTS)), ("/historical/", history)]
    with tempfile.TemporaryDirectory() as tmp:
        out, report = os.path.join(tmp, "prices"), os.path.join(tmp, "prices_report.csv")
        start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")
        rep = fk.fetch(fake(routes, calls), ["NOPE", "RELIANCE", "XYZ"], start, end, out, report)
        assert rep.set_index("symbol")["status"].str[:7].to_dict() == \
            {"NOPE": "not on ", "RELIANCE": "fetched", "XYZ": "fetched"}
        prices = bt.load_prices(out)
        assert sorted(prices) == ["RELIANCE", "XYZ"] and list(prices["XYZ"].columns) == bt.PRICE_COLS
        assert len(prices["RELIANCE"]) == len(pd.bdate_range(start, end))
        n = len(calls)
        rep = fk.fetch(fake(routes, calls), ["RELIANCE", "XYZ"], start, end, out, report)
        assert set(rep["status"]) == {"kept"} and len(calls) == n + 1     # only the instruments list again


def test_only_stocks_with_two_alert_days_are_fetched():
    alerts = pd.DataFrame({"symbol": ["A", "A", "B", "B", "C"],
                           "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-05", "2024-01-05",
                                                   "2024-03-01"])})
    assert fk.alert_symbols(alerts) == ["A"] and fk.alert_symbols(alerts, all_symbols=True) == ["A", "B", "C"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
