# fx_daily_system.py
# Evening FX alerts + month-end TFSA summary (paper trade log)
# Run:
#   python fx_daily_system.py                 # alerts mode (default)
#   MODE=tfsa python fx_daily_system.py       # TFSA sweep summary
#
# Deps: pip install -r requirements.txt

import os, pathlib, math, requests, calendar
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── load .env for local runs ─────────────────────────────────────────────────
load_dotenv()

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance:  pip install yfinance")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here if needed
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
DEFAULT_SPREAD_POINTS = 25  # fallback

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

# Month-end sweep suggestion base (used if needed)
BASE_CAP = float(os.getenv("BASE_CAP", "30000"))  # suggest using your trading float

# Timezone: Africa/Johannesburg (SAST = UTC+2)
TZ = timezone(timedelta(hours=2))

# Data/logs
DATA_DIR   = pathlib.Path("data_fx")
TRADES_CSV = DATA_DIR / "trades.csv"   # closed trades across runs
EQUITY_CSV = DATA_DIR / "equity.csv"   # equity snapshots
DATA_DIR.mkdir(exist_ok=True)

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

def fetch(symbol: str, years=LOOKBACK_YEARS, interval=INTERVAL) -> pd.DataFrame:
    start = (datetime.now(tz=TZ) - timedelta(days=365*years)).date().isoformat()
    df = yf.download(symbol, start=start, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df.index = pd.to_datetime(df.index)
    df.rename(columns={c: c.capitalize() for c in ["open","high","low","close","volume"]}, inplace=True)
    if "Volume" not in df.columns: df["Volume"] = 0
    return df

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
    # Choose position fraction of equity so loss to stop ≈ risk_frac * equity
    loss_frac = abs(stop - entry) / entry
    if loss_frac <= 0: return 0.0
    return float(min(risk_frac / loss_frac, 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# Backtest/paper execution (bar-by-bar)
# ─────────────────────────────────────────────────────────────────────────────
def backtest_portfolio(symbols=SYMBOLS):
    equity = ACCOUNT_START
    equity_curve = []
    trade_rows = []

    for sym in symbols:
        df = fetch(sym)
        d  = compute(df)
        cost = spread_fraction(sym)  # one-way price impact
        px_open_next = d["Open"].shift(-1)

        pos = 0
        entry = stop = take = np.nan
        size_frac_val = 0.0

        for i in range(len(d)-1):
            idx = d.index[i]; nxt = d.index[i+1]
            atr = d.at[idx, "ATR"]
            fill = px_open_next.iloc[i]
            if np.isnan(fill): continue

            # Exits
            if pos != 0 and not math.isnan(entry):
                hit_tp = (pos==1 and fill >= take) or (pos==-1 and fill <= take)
                hit_sl = (pos==1 and fill <= stop) or (pos==-1 and fill >= stop)
                exit_reason = None
                if hit_tp:
                    exit_px = take
                    exit_reason = "TP"
                elif hit_sl:
                    exit_px = stop
                    exit_reason = "SL"
                else:
                    if (pos==1 and d.at[idx,"RSI"]>=50) or (pos==-1 and d.at[idx,"RSI"]<=50):
                        exit_px = fill
                        exit_reason = "ExitSig"

                if exit_reason:
                    exit_adj = exit_px - cost if pos==1 else exit_px + cost
                    gross = (exit_adj - entry) / entry * pos
                    pnl_rands = size_frac_val * gross * equity
                    # equity change based on current equity
                    equity += pnl_rands

                    trade_rows.append({
                        "symbol": sym,
                        "opened": idx.date().isoformat(),
                        "closed": nxt.date().isoformat(),
                        "side": "LONG" if pos==1 else "SHORT",
                        "entry": round(float(entry), 6),
                        "exit":  round(float(exit_adj), 6),
                        "size_frac": round(size_frac_val, 4),
                        "pnl_rands": round(float(pnl_rands), 2),
                        "reason": exit_reason,
                    })
                    pos = 0; entry=stop=take=np.nan; size_frac_val=0.0

            # Entries
            if pos == 0:
                long_ok, short_ok = latest_setup(d.iloc[:i+1])
                if long_ok:
                    ent = fill + cost
                    sl  = ent - ATR_SL_MULT * atr
                    tp  = ent + ATR_TP_MULT * atr
                    sf  = size_fraction(ent, sl, equity, RISK_PER_TRADE)
                    if sf > 0:
                        pos, entry, stop, take, size_frac_val = 1, ent, sl, tp, sf
                elif short_ok:
                    ent = fill - cost
                    sl  = ent + ATR_SL_MULT * atr
                    tp  = ent - ATR_TP_MULT * atr
                    sf  = size_fraction(ent, sl, equity, RISK_PER_TRADE)
                    if sf > 0:
                        pos, entry, stop, take, size_frac_val = -1, ent, sl, tp, sf

            equity_curve.append({"date": nxt.date().isoformat(), "symbol": sym, "equity": round(equity, 2)})

    # Persist logs
    if trade_rows:
        df_tr = pd.DataFrame(trade_rows)
        if TRADES_CSV.exists():
            prev = pd.read_csv(TRADES_CSV)
            df_tr = pd.concat([prev, df_tr], ignore_index=True)
        df_tr.to_csv(TRADES_CSV, index=False)

    df_eq = pd.DataFrame(equity_curve)
    df_eq.to_csv(EQUITY_CSV, index=False)

    return equity

# ─────────────────────────────────────────────────────────────────────────────
# Alerts & summaries
# ─────────────────────────────────────────────────────────────────────────────
def daily_signal_summary() -> str:
    lines = []
    for sym in SYMBOLS:
        d = compute(fetch(sym))
        long_ok, short_ok = latest_setup(d)
        asof = d.index[-1].date().isoformat()
        if long_ok:
            lines.append(f"{sym} — BUY setup (as of {asof})")
        elif short_ok:
            lines.append(f"{sym} — SELL setup (as of {asof})")
        else:
            lines.append(f"{sym} — No setup (as of {asof})")
    return "\n".join(lines)

def send_telegram(msg: str):
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        print("Telegram not configured; skipping send.")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=15,
        )
        if resp.status_code != 200:
            print("Telegram error:", resp.text)
    except Exception as e:
        print("Telegram send failed:", e)

def run_tfsa_sweep_summary():
    # Summarize THIS MONTH's closed PnL across all symbols
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

    # Suggest sweep = max(min target, actual pnl) or just the actual PnL; we’ll keep it simple.
    suggestion = pnl_sum  # Rands

    month_str = now.strftime("%Y-%m")
    msg = (f"📥 TFSA sweep ({month_str})\n"
           f"Closed PnL this month: R{pnl_sum:,.2f}\n"
           f"Suggested transfer:   R{suggestion:,.2f}\n"
           f"(Trading float modeled at R{BASE_CAP:,.0f}, 1% risk per trade)")
    send_telegram(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = os.getenv("MODE", "alerts").lower()
    if mode == "alerts":
        print("Running WesFX evening alerts…")
        summary = daily_signal_summary()
        print("\n=== SIGNALS (for next session) ===\n" + summary)

        final_eq = backtest_portfolio(SYMBOLS)
        print(f"\nLogs written → {TRADES_CSV}, {EQUITY_CSV}")
        ts = datetime.now(tz=TZ).strftime("%Y-%m-%d %H:%M")
        send_telegram(f"FX Daily ({ts})\n{summary}")
    elif mode == "tfsa":
        print("Running WesFX TFSA sweep summary…")
        run_tfsa_sweep_summary()
    else:
        print("Unknown MODE. Use MODE=alerts (default) or MODE=tfsa.")
