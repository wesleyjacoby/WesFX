#!/usr/bin/env python3
# etf_seatbelt50_backtest_jse.py — Backtests the 50/50 risk-on seatbelt (JSE)
# Deps: pip install -U pandas numpy yfinance matplotlib

import argparse
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

OUTDIR = pathlib.Path("etf_out_seatbelt50_bt")
OUTDIR.mkdir(exist_ok=True)

EQUITY_1_CANDIDATES = ["STXWDM.JO", "SYGWD.JO", "STX500.JO"]
EQUITY_2_CANDIDATES = ["STX40.JO"]
BOND_CANDIDATES      = ["STXGOV.JO"]

SEATBELT_M = 10  # 10-month MA

# ------- helpers -------
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

def dl_monthly_close(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    df = ensure_utc_index(df)
    m = df["Close"].resample("ME").last().dropna()
    m.name = ticker
    m = m.astype(float)
    return dedup_series_monthly(m)

def scalar_or_nan(s: pd.Series, t) -> float:
    """Return float scalar at timestamp t; np.nan if missing/ambiguous."""
    try:
        v = s.loc[t]
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v = v.iloc[-1]
        return float(v)
    except Exception:
        return np.nan

def perf_from_equity(eq: pd.Series, freq="M") -> Dict[str, float]:
    eq = eq.dropna()
    if eq.empty:
        return dict(CAGR=0, MaxDD=0, Vol=0, Sharpe=0, Sortino=0, WorstYr=0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    base0 = eq.iloc[0] if eq.iloc[0] != 0 else 1.0
    cagr = (eq.iloc[-1] / base0) ** (1/years) - 1 if years > 0 else 0
    rets = eq.pct_change().dropna()
    ann = 12 if freq.upper().startswith("M") else 52
    vol = rets.std() * np.sqrt(ann)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(ann) if rets.std() > 0 else 0
    neg = rets[rets < 0]
    sortino = (rets.mean() / neg.std()) * np.sqrt(ann) if len(neg) and neg.std() > 0 else 0
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1
    maxdd = dd.min()
    yr = eq.resample("YE").last().pct_change().dropna()
    worst = yr.min() if not yr.empty else 0
    return dict(
        CAGR=round(cagr*100,2),
        MaxDD=round(maxdd*100,2),
        Vol=round(vol*100,2),
        Sharpe=round(float(sharpe),2),
        Sortino=round(float(sortino),2),
        WorstYr=round(float(worst)*100,2),
    )

# ------- core -------
def seatbelt50_signals(e1: pd.Series, e2: pd.Series, b: pd.Series, seatbelt_m=SEATBELT_M) -> pd.DataFrame:
    common = e1.index.intersection(e2.index).intersection(b.index).sort_values()
    e1 = dedup_series_monthly(e1.loc[common])
    e2 = dedup_series_monthly(e2.loc[common])
    b  = dedup_series_monthly(b.loc[common])

    ma1 = e1.rolling(seatbelt_m).mean()
    ma2 = e2.rolling(seatbelt_m).mean()

    rows = []
    for t in common:
        p1 = scalar_or_nan(e1, t)
        p2 = scalar_or_nan(e2, t)
        m1 = scalar_or_nan(ma1, t)
        m2 = scalar_or_nan(ma2, t)

        # Risk-on if either equity above 10M MA (NaN-safe: early months default to risk-on)
        risk_on = (not np.isnan(m1) and not np.isnan(p1) and p1 >= m1) or \
                  (not np.isnan(m2) and not np.isnan(p2) and p2 >= m2) or \
                  (np.isnan(m1) and np.isnan(m2))

        if risk_on:
            rows.append((t, 0.5, 0.5, 0.0, "RISK-ON"))
        else:
            rows.append((t, 0.0, 0.0, 1.0, "DEFENSIVE"))

    sig = pd.DataFrame(rows, columns=["date","w_e1","w_e2","w_b","regime"]).set_index("date")
    sig.index = pd.to_datetime(sig.index, utc=True)
    return sig

def sim_newmoney(sig: pd.DataFrame, e1: pd.Series, e2: pd.Series, b: pd.Series, contrib: float) -> pd.Series:
    idx = sig.index
    u1=u2=ub=0.0; nav=[]
    for t in idx:
        w1, w2, wb = float(sig.loc[t,"w_e1"]), float(sig.loc[t,"w_e2"]), float(sig.loc[t,"w_b"])
        if wb == 0.0:
            price1 = scalar_or_nan(e1, t)
            price2 = scalar_or_nan(e2, t)
            if not np.isnan(price1) and not np.isnan(price2):
                u1 += (contrib*0.5)/price1
                u2 += (contrib*0.5)/price2
        else:
            priceb = scalar_or_nan(b, t)
            if not np.isnan(priceb):
                ub += contrib/priceb
        nav.append(u1*scalar_or_nan(e1,t) + u2*scalar_or_nan(e2,t) + ub*scalar_or_nan(b,t))
    return pd.Series(nav, index=idx, name="Seatbelt50_NewMoney_NAV")

def sim_strict(sig: pd.DataFrame, e1: pd.Series, e2: pd.Series, b: pd.Series) -> pd.Series:
    idx = sig.index
    val = 1.0; out=[]
    prev = dict(w_e1=0.0, w_e2=0.0, w_b=1.0)
    for i in range(1, len(idx)):
        t0 = idx[i-1]; t1 = idx[i]
        p1_t0, p1_t1 = scalar_or_nan(e1,t0), scalar_or_nan(e1,t1)
        p2_t0, p2_t1 = scalar_or_nan(e2,t0), scalar_or_nan(e2,t1)
        pb_t0, pb_t1 = scalar_or_nan(b,t0),  scalar_or_nan(b,t1)
        r1 = (p1_t1/p1_t0 - 1.0) if (not np.isnan(p1_t0) and not np.isnan(p1_t1)) else 0.0
        r2 = (p2_t1/p2_t0 - 1.0) if (not np.isnan(p2_t0) and not np.isnan(p2_t1)) else 0.0
        rb = (pb_t1/pb_t0 - 1.0) if (not np.isnan(pb_t0) and not np.isnan(pb_t1)) else 0.0
        val *= (1 + prev["w_e1"]*r1 + prev["w_e2"]*r2 + prev["w_b"]*rb)
        out.append(val)
        prev = dict(
            w_e1=float(sig.loc[t1,"w_e1"]),
            w_e2=float(sig.loc[t1,"w_e2"]),
            w_b=float(sig.loc[t1,"w_b"])
        )
    return pd.Series(out, index=idx[1:], name="Seatbelt50_Strict_EQ")

def main():
    ap = argparse.ArgumentParser(description="Backtest 50/50 risk-on seatbelt (JSE)")
    ap.add_argument("--start", type=int, default=2006)
    ap.add_argument("--contrib", type=float, default=1000.0)
    args = ap.parse_args()

    sym = {
        "E1": first_available(EQUITY_1_CANDIDATES),
        "E2": first_available(EQUITY_2_CANDIDATES),
        "B":  first_available(BOND_CANDIDATES),
    }
    start = f"{args.start}-01-01"

    e1 = dl_monthly_close(sym["E1"], start)
    e2 = dl_monthly_close(sym["E2"], start)
    b  = dl_monthly_close(sym["B"],  start)
    common = e1.index.intersection(e2.index).intersection(b.index).sort_values()
    e1,e2,b = e1.loc[common], e2.loc[common], b.loc[common]

    sig = seatbelt50_signals(e1,e2,b, SEATBELT_M)

    eq_flow  = sim_newmoney(sig, e1, e2, b, args.contrib)
    eq_strict= sim_strict(sig, e1, e2, b)

    # Summary
    rows = []
    for name, ser in {
        "Seatbelt50_NewMoney_NAV": eq_flow,
        "Seatbelt50_Strict_EQ":    eq_strict,
    }.items():
        m = perf_from_equity(ser, "M")
        rows.append(dict(Strategy=name, **m))
    summary = pd.DataFrame(rows)

    print("\n=== Seatbelt 50/50 — Backtest (JSE) ===")
    print(f"Instruments: E1={sym['E1']} | E2={sym['E2']} | B={sym['B']}")
    print(summary.to_string(index=False))

    # Save + plot
    OUTDIR.mkdir(exist_ok=True)
    summary.to_csv(OUTDIR / "summary_seatbelt50.csv", index=False)
    curves = pd.DataFrame({"Seatbelt50_NewMoney_NAV": eq_flow, "Seatbelt50_Strict_EQ": eq_strict})
    curves.to_csv(OUTDIR / "curves_seatbelt50.csv")

    plt.figure(figsize=(10,6))
    (eq_flow / eq_flow.iloc[0]).plot(label="NewMoney (NAV)")
    (eq_strict / eq_strict.iloc[0]).plot(label="Strict (EQ)")
    plt.title(f"Seatbelt 50/50 — {sym['E1']} & {sym['E2']} / {sym['B']}")
    plt.ylabel("Normalized value")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "equity_seatbelt50.png", dpi=130)
    plt.close()

    print(f"Saved → {OUTDIR.resolve()}/summary_seatbelt50.csv, curves_seatbelt50.csv, equity_seatbelt50.png")

if __name__ == "__main__":
    main()
