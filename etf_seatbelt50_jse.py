#!/usr/bin/env python3
# etf_seatbelt50_jse.py — Monthly 50/50 risk-on seatbelt for JSE ETFs
# Rule: If E1 (World) OR E2 (Top40) is above 10-month MA → invest 50/50 into E1 & E2.
#       Else (both below) → invest 100% into Bonds (STXGOV).
# Deps: pip install -U pandas numpy yfinance

import argparse
from dataclasses import dataclass
import pathlib
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

OUTDIR = pathlib.Path("etf_out_seatbelt50")
OUTDIR.mkdir(exist_ok=True)

EQUITY_1_CANDIDATES = ["STXWDM.JO", "SYGWD.JO", "STX500.JO"]  # World / SP500 feeders
EQUITY_2_CANDIDATES = ["STX40.JO"]                           # SA Top 40
BOND_CANDIDATES      = ["STXGOV.JO"]                         # SA GOVI bonds

SEATBELT_M = 10  # 10-month MA seatbelt

# ---------- Helpers ----------
def ensure_utc_index(obj):
    idx = obj.index
    if getattr(idx, "tz", None) is None:
        obj.index = pd.to_datetime(idx, utc=True)
    else:
        obj.index = idx.tz_convert("UTC")
    return obj

def first_available(tickers: List[str]) -> str:
    for t in tickers:
        try:
            if not yf.Ticker(t).history(period="5d").empty:
                return t
        except Exception:
            pass
    raise RuntimeError(f"No live data for any of: {tickers}")

def dedup_series_monthly(s: pd.Series) -> pd.Series:
    return s[~s.index.duplicated(keep="last")]

def download_monthly_close(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    df = ensure_utc_index(df)
    m = df["Close"].resample("ME").last().dropna()
    m.name = ticker
    m = m.astype(float)
    return dedup_series_monthly(m)

def get_scalar_or_nan(s: pd.Series, t) -> float:
    try:
        val = s.loc[t]
        if isinstance(val, (pd.Series, pd.DataFrame)):
            val = val.iloc[-1]
        return float(val)
    except Exception:
        return np.nan

@dataclass
class SignalRow:
    date: pd.Timestamp
    w_e1: float
    w_e2: float
    w_b: float
    regime: str  # RISK-ON or DEFENSIVE

# ---------- Core ----------
def compute_seatbelt50(e1: pd.Series, e2: pd.Series, b: pd.Series, seatbelt_m=SEATBELT_M) -> pd.DataFrame:
    common = e1.index.intersection(e2.index).intersection(b.index).sort_values()
    e1 = dedup_series_monthly(e1.loc[common])
    e2 = dedup_series_monthly(e2.loc[common])
    b  = dedup_series_monthly(b.loc[common])

    ma1 = e1.rolling(seatbelt_m).mean()
    ma2 = e2.rolling(seatbelt_m).mean()

    rows = []
    for t in common:
        p1 = get_scalar_or_nan(e1, t)
        p2 = get_scalar_or_nan(e2, t)
        m1 = get_scalar_or_nan(ma1, t)
        m2 = get_scalar_or_nan(ma2, t)

        # Risk-on if either equity is above its 10M MA (NaN-safe: if MAs not ready, default to risk-on)
        risk_on = (not np.isnan(p1) and not np.isnan(m1) and p1 >= m1) or \
                  (not np.isnan(p2) and not np.isnan(m2) and p2 >= m2) or \
                  (np.isnan(m1) and np.isnan(m2))  # early data: treat as risk-on

        if risk_on:
            rows.append(SignalRow(t, 0.5, 0.5, 0.0, "RISK-ON"))
        else:
            rows.append(SignalRow(t, 0.0, 0.0, 1.0, "DEFENSIVE"))

    out = pd.DataFrame([vars(r) for r in rows]).set_index("date")
    out.index = pd.to_datetime(out.index, utc=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Seatbelt 50/50 monthly signal for JSE ETFs")
    ap.add_argument("--start", type=int, default=2006)
    ap.add_argument("--contrib", type=float, default=1000.0)
    args = ap.parse_args()

    sym = {
        "E1": first_available(EQUITY_1_CANDIDATES),
        "E2": first_available(EQUITY_2_CANDIDATES),
        "B":  first_available(BOND_CANDIDATES),
    }
    start = f"{args.start}-01-01"

    e1 = download_monthly_close(sym["E1"], start)
    e2 = download_monthly_close(sym["E2"], start)
    b  = download_monthly_close(sym["B"],  start)

    sig = compute_seatbelt50(e1, e2, b, SEATBELT_M)

    # Current month view
    t = sig.index[-1]
    w1 = float(sig.iloc[-1]["w_e1"])
    w2 = float(sig.iloc[-1]["w_e2"])
    wb = float(sig.iloc[-1]["w_b"])
    regime = str(sig.iloc[-1]["regime"])
    r1 = args.contrib * w1
    r2 = args.contrib * w2
    rb = args.contrib * wb

    print("\n=== Seatbelt 50/50 — Current Month ===")
    print(f"Instruments: E1={sym['E1']} | E2={sym['E2']} | B={sym['B']}")
    print(f"As of {pd.to_datetime(t).date()}  [{regime}]")
    if wb == 0.0:
        print(f"Contribution split (R{args.contrib:,.0f}):  "
              f"{sym['E1']} → R{r1:,.0f}  |  {sym['E2']} → R{r2:,.0f}  |  {sym['B']} → R0")
    else:
        print(f"Contribution split (R{args.contrib:,.0f}):  "
              f"{sym['B']} → R{rb:,.0f}  |  {sym['E1']} → R0  |  {sym['E2']} → R0")

    # Save last 24 months for auditability
    renamed = sig.rename(columns={"w_e1": f"W_{sym['E1']}", "w_e2": f"W_{sym['E2']}", "w_b": f"W_{sym['B']}"})
    renamed.tail(24).to_csv(OUTDIR / "seatbelt50_last24.csv")
    print(f"\nSaved last 24 months → {OUTDIR.resolve()}/seatbelt50_last24.csv")

if __name__ == "__main__":
    main()
