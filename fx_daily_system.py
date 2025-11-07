# fx_daily_system.py
# Daily FX alerts with detailed Entry/SL/TP/Size and exit advisories.
# Keeps a tiny positions ledger:
#   - planned (enters at next day's open)
#   - open (monitored for TP/SL or RSI exit)
#
# Run:
#   python fx_daily_system.py                 # evening alerts (default)
#   MODE=tfsa python fx_daily_system.py       # month-end TFSA summary
#
# Deps: pip install pandas numpy yfinance python-dotenv requests

import os, pathlib, math, requests, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDCHF=X", "USDJPY=X"]  # majors
INTERVAL = "1d"
LOOKBACK_YEARS = 9  # ~2016+

# Broker spread in *points* (≈10 points = 1 pip). Tune per pair.
SPREAD_POINTS_PER_SYMBOL = {
    "EURUSD=X": 21,  # ≈2.1 pips
    "GBPUSD=X": 29,  # ≈2.9 pips
    "USDCHF=X": 24,  # ≈2.4 pips
    "USDJPY=X": 22,  # ≈2.2 pips (JPY pip size differs; handled below)
}
DEFAULT_SPREAD_POINTS = 25

# Strategy knobs (daily timeframe)
FAST_SMA = 50
SLOW_SMA = 200
EMA_PB   = 20
RSI_LEN  = 14
RSI_BUY  = 35
RSI_SELL = 65
ATR_LEN  = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
MIN_ATR_FRAC = 0.0005  # ignore ultra-quiet regimes

# Paper account & risk
ACCOUNT_START   = 30000.0  # R30k
RISK_PER_TRADE  = 0.01     # 1% risk per trade

# Telegram (env or GitHub Secrets)
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

# Timezone: Africa/Johannesburg (SAST = UTC+2)
TZ = timezone(timedelta(hours=2))

# Data/logs
DATA_DIR     = pathlib.Path("data_fx"); DATA_DIR.mkdir(exist_ok=True)
TRADES_CSV   = DATA_DIR / "trades.csv"     # closed trades
EQUITY_CSV   = DATA_DIR / "equity.csv"     # equity snapshots
POSITIONS_CSV= DATA_DIR / "positions.csv"  # planned/open positions

FAILED_SYMBOLS = []

# ─────────────────────────────────────────────────────────────────────────────
# Indicators & helpers
# ─────────────────────────────────────────────────────────────────────────────
def rsi(s: pd.Series, n=14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def pip_size(symbol: str) -> float:
    # 1 pip = 0.0001 for non-JPY; 0.01 for JPY pairs
    return 0.01 if "JPY" in symbol else 0.0001

def spread_fraction(symbol: str) -> float:
    # Convert points → pips → price fraction
    points = SPREAD_POINTS_PER_SYMBOL.get(symbol, DEFAULT_SPREAD_POINTS)
    pips = points / 10.0
    return pips * pip_size(symbol)

# ─────────────────────────────────────────────────────────────────────────────
# Robust data fetch with retries + fallback + column normalization
# ─────────────────────────────────────────────────────────────────────────────
def _flatten_to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            labels = {str(v).strip().lower() for v in df.columns.get_level_values(lvl)}
            if {"open","high","low","close","adj close"}.intersection(labels):
                cols = [str(v).strip() for v in df.columns.get_level_values(lvl)]
                df = df.copy(); df.columns = cols; break
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "open":   rename[c] = "Open"
        if cl == "high":   rename[c] = "High"
        if cl == "low":    rename[c] = "Low"
        if cl == "close":  rename[c] = "Close"
        if cl == "adj close": rename[c] = "Adj Close"
        if cl == "volume": rename[c] = "Volume"
    if rename:
        df = df.rename(columns=rename)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    required = {"Open","High","Low","Close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"missing columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open","High","Low","Close","Volume"]]

def fetch(symbol: str, years=LOOKBACK_YEARS, interval=INTERVAL) -> pd.DataFrame:
    start = (datetime.now(tz=TZ) - timedelta(days=365*years)).date().isoformat()
    last_err = None; df = pd.DataFrame()

    for attempt in range(5):
        try:
            df = yf.download(
                symbol, start=start, interval=interval,
                auto_adjust=True, progress=False, threads=False,
                group_by="column"
            )
            if df is not None and not df.empty:
                break
            last_err = RuntimeError("Empty dataframe from download()")
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    else:
        try:
            hist = yf.Ticker(symbol).history(period=f"{years}y", interval=interval, auto_adjust=True)
            if hist is not None and not hist.empty:
                df = hist
        except Exception as e:
            last_err = e

    if df is None or df.empty:
        FAILED_SYMBOLS.append(f"{symbol} ({last_err})")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    try:
        df = _flatten_to_ohlc(df)
    except Exception as e:
        FAILED_SYMBOLS.append(f"{symbol} ({e})")
        return pd.DataFrame()
    return df.sort_index()

# ─────────────────────────────────────────────────────────────────────────────
# Signals / sizing
# ─────────────────────────────────────────────────────────────────────────────
def compute(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["SMA_fast"] = sma(d["Close"], FAST_SMA)
    d["SMA_slow"] = sma(d["Close"], SLOW_SMA)
    d["EMA_pb"]   = ema(d["Close"], EMA_PB)
    d["RSI"]      = rsi(d["Close"], RSI_LEN)
    d["ATR"]      = atr(d, ATR_LEN)
    d["ATR_frac"] = d["ATR"] / d["Close"]
    d["Up"]       = d["SMA_fast"] > d["SMA_slow"]
    d["Dn"]       = d["SMA_fast"] < d["SMA_slow"]
    d["NearPB"]   = (d["Close"] - d["EMA_pb"]).abs() <= d["ATR"]
    d["OkVol"]    = d["ATR_frac"] >= MIN_ATR_FRAC
    return d.dropna().copy()

def latest_setup(d: pd.DataFrame):
    last = d.iloc[-1]
    long_ok  = bool(last["Up"] and last["NearPB"] and last["OkVol"] and last["RSI"] <= RSI_BUY)
    short_ok = bool(last["Dn"] and last["NearPB"] and last["OkVol"] and last["RSI"] >= RSI_SELL)
    return long_ok, short_ok

def size_fraction(entry: float, stop: float, equity: float, risk_frac=RISK_PER_TRADE) -> float:
    loss_frac = abs(stop - entry) / entry
    if loss_frac <= 0: return 0.0
    return float(min(risk_frac / loss_frac, 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# Ledger helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_positions() -> pd.DataFrame:
    if POSITIONS_CSV.exists():
        df = pd.read_csv(POSITIONS_CSV)
        if len(df):
            df["opened_dt"] = pd.to_datetime(df["opened_dt"], errors="coerce")
            df["planned_for"] = pd.to_datetime(df["planned_for"], errors="coerce")
        return df
    return pd.DataFrame(columns=[
        "symbol","side","status","entry","sl","tp","size_frac",
        "opened_dt","planned_for"
    ])

def save_positions(df: pd.DataFrame):
    df.to_csv(POSITIONS_CSV, index=False)

def append_trade(tr: dict):
    df_tr = pd.DataFrame([tr])
    if TRADES_CSV.exists():
        prev = pd.read_csv(TRADES_CSV)
        df_tr = pd.concat([prev, df_tr], ignore_index=True)
    df_tr.to_csv(TRADES_CSV, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Evening flow:
#  1) Promote planned→open at today's open price
#  2) For open: check TP/SL via today's High/Low; otherwise exit signal via RSI
#  3) Generate new planned orders for fresh setups with full Entry/SL/TP/Size
# ─────────────────────────────────────────────────────────────────────────────
def evening_process_and_message():
    equity = ACCOUNT_START  # paper equity reference for sizing math
    lines = []
    pos_df = load_positions()

    # Build per-symbol latest data dicts
    latest = {}
    for sym in SYMBOLS:
        raw = fetch(sym)
        if raw.empty:
            latest[sym] = {"status":"no-data"}
            continue
        d = compute(raw)
        if d.empty:
            latest[sym] = {"status":"no-ind"}
            continue
        latest[sym] = {
            "status":"ok",
            "df": d,
            "asof": d.index[-1].date().isoformat(),
            "open_today": d["Open"].iloc[-1],
            "high_today": d["High"].iloc[-1],
            "low_today":  d["Low"].iloc[-1],
            "close_today":d["Close"].iloc[-1],
            "atr_today":  d["ATR"].iloc[-1],
            "rsi_today":  d["RSI"].iloc[-1],
            "long_ok":  latest_setup(d)[0],
            "short_ok": latest_setup(d)[1],
            "cost": spread_fraction(sym),
        }

    # 1) Promote planned → open at today's open
    to_open_idx = []
    for i, row in pos_df[pos_df["status"]=="planned"].iterrows():
        sym = row["symbol"]
        info = latest.get(sym, {})
        if info.get("status")!="ok":  # can't open without data
            continue
        # Enter at today's open ± cost
        ent = info["open_today"] + (info["cost"] if row["side"]=="LONG" else -info["cost"])
        pos_df.at[i,"entry"] = float(ent)
        pos_df.at[i,"status"]="open"
        pos_df.at[i,"opened_dt"]= pd.to_datetime(datetime.now(tz=TZ).date())
        lines.append(f"{sym} — ✅ Opened {row['side']} at today's open ≈ {ent:.6f}")
        to_open_idx.append(i)
    if len(to_open_idx):
        save_positions(pos_df)

    # 2) For open positions: monitor TP/SL, else exit signal
    for i, row in pos_df[pos_df["status"]=="open"].iterrows():
        sym = row["symbol"]; info = latest.get(sym, {})
        if info.get("status")!="ok":
            lines.append(f"{sym} — ⚠️ Open {row['side']} but data unavailable today")
            continue
        side=row["side"]; entry=row["entry"]; sl=row["sl"]; tp=row["tp"]; sf=row["size_frac"]
        hi=info["high_today"]; lo=info["low_today"]; rsi=info["rsi_today"]; cost=info["cost"]

        closed = False
        reason = ""
        exit_px = None

        # Intraday touches: use High/Low versus levels
        if side=="LONG":
            if hi >= tp:
                exit_px = tp - cost; reason="TP"
            elif lo <= sl:
                exit_px = sl - cost; reason="SL"
            else:
                if rsi >= 50:
                    lines.append(f"{sym} — ⚠️ Exit signal (RSI≥50), close at next open. Hold for now. SL {sl:.6f}, TP {tp:.6f}")
                else:
                    lines.append(f"{sym} — Holding LONG. SL {sl:.6f}, TP {tp:.6f}")
        else:  # SHORT
            if lo <= tp:
                exit_px = tp + cost; reason="TP"
            elif hi >= sl:
                exit_px = sl + cost; reason="SL"
            else:
                if rsi <= 50:
                    lines.append(f"{sym} — ⚠️ Exit signal (RSI≤50), close at next open. Hold for now. SL {sl:.6f}, TP {tp:.6f}")
                else:
                    lines.append(f"{sym} — Holding SHORT. SL {sl:.6f}, TP {tp:.6f}")

        if exit_px is not None:
            gross = (exit_px - entry)/entry * (1 if side=="LONG" else -1)
            pnl_r = sf * gross * equity
            append_trade({
                "symbol": sym,
                "opened": row["opened_dt"].date().isoformat() if pd.notna(row["opened_dt"]) else "",
                "closed": datetime.now(tz=TZ).date().isoformat(),
                "side": side,
                "entry": round(float(entry),6),
                "exit":  round(float(exit_px),6),
                "size_frac": round(float(sf),4),
                "pnl_rands": round(float(pnl_r),2),
                "reason": reason
            })
            lines.append(f"{sym} — ✅ Closed {side} via {reason} at {exit_px:.6f} (paper PnL ~ R{pnl_r:.2f})")
            pos_df = pos_df.drop(index=i)
            save_positions(pos_df)

    # 3) Generate NEW planned orders for fresh setups (no existing open/planned)
    existing_syms = set(pos_df["symbol"])
    for sym in SYMBOLS:
        info = latest.get(sym, {})
        if info.get("status")!="ok":
            lines.append(f"{sym} — ❌ data unavailable (skipped)")
            continue

        d = info["df"]; asof = info["asof"]; cost = info["cost"]
        long_ok, short_ok = info["long_ok"], info["short_ok"]
        open_px = info["open_today"]; atr = info["atr_today"]

        has_any = sym in existing_syms
        if not has_any and (long_ok or short_ok):
            side = "LONG" if long_ok else "SHORT"
            if side=="LONG":
                planned_entry = open_px + cost
                sl = planned_entry - ATR_SL_MULT * atr
                tp = planned_entry + ATR_TP_MULT * atr
            else:
                planned_entry = open_px - cost
                sl = planned_entry + ATR_SL_MULT * atr
                tp = planned_entry - ATR_TP_MULT * atr
            sf = size_fraction(planned_entry, sl, ACCOUNT_START, RISK_PER_TRADE)

            # Record planned order for tomorrow
            plan_for = (d.index[-1] + timedelta(days=1)).date().isoformat()
            new_row = {
                "symbol": sym, "side": side, "status": "planned",
                "entry": float(planned_entry), "sl": float(sl), "tp": float(tp),
                "size_frac": float(sf),
                "opened_dt": "", "planned_for": plan_for
            }
            pos_df = pd.concat([pos_df, pd.DataFrame([new_row])], ignore_index=True)
            save_positions(pos_df)

            # Human-friendly message
            risk_pct = int(RISK_PER_TRADE*100)
            lines.append(
                f"{sym} — BUY setup" if side=="LONG" else f"{sym} — SELL setup"
            )
            lines.append(
                f"  Plan for next open • Entry {planned_entry:.6f} • SL {sl:.6f} • TP {tp:.6f} • Size ≈ {sf*100:.2f}% of equity (risk {risk_pct}%)"
            )
        elif not (long_ok or short_ok):
            lines.append(f"{sym} — No setup (as of {asof})")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# TFSA summary (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def run_tfsa_sweep_summary():
    if not TRADES_CSV.exists():
        send_telegram("ℹ️ TFSA sweep: no closed trades logged yet.")
        return
    df = pd.read_csv(TRADES_CSV)
    if "closed" not in df.columns or "pnl_rands" not in df.columns:
        send_telegram("ℹ️ TFSA sweep: log missing required columns.")
        return
    df["closed_dt"] = pd.to_datetime(df["closed"])
    now = datetime.now(tz=TZ)
    this_month = df[(df["closed_dt"].dt.year == now.year) & (df["closed_dt"].dt.month == now.month)]
    pnl_sum = float(this_month["pnl_rands"].sum()) if len(this_month) else 0.0
    month_str = now.strftime("%Y-%m")
    msg = (f"📥 TFSA sweep ({month_str})\n"
           f"Closed PnL this month: R{pnl_sum:,.2f}\n"
           f"Suggested transfer:   R{pnl_sum:,.2f}\n"
           f"(Trading float modeled at R{ACCOUNT_START:,.0f}, 1% risk per trade)")
    send_telegram(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        print("Telegram not configured; skipping send.")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=20,
        )
        if resp.status_code != 200:
            print("Telegram error:", resp.text)
    except Exception as e:
        print("Telegram send failed:", e)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = os.getenv("MODE", "alerts").lower()
    if mode == "alerts":
        print("Running WesFX evening alerts…")
        summary = evening_process_and_message()
        print("\n=== SUMMARY ===\n" + summary)
        ts = datetime.now(tz=TZ).strftime("%Y-%m-%d %H:%M")
        send_telegram(f"FX Daily ({ts})\n{summary}")
    elif mode == "tfsa":
        print("Running WesFX TFSA sweep summary…")
        run_tfsa_sweep_summary()
    else:
        print("Unknown MODE. Use MODE=alerts (default) or MODE=tfsa.")
