# fx_4h_system.py
# 4-hour WesFX: trend (SMA/EMA), pullback to EMA, RSI gate, ATR sizing.
# Sends Telegram alerts with Entry/SL/TP/Size.
#
# Run:
#   python fx_4h_system.py                 # live 4h alerts
#   MODE=tfsa python fx_4h_system.py       # (optional) month-end summary (shares env)
#
# Deps: pandas numpy yfinance python-dotenv requests

import os, pathlib, math, requests, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDCHF=X", "USDJPY=X"]

# We’ll fetch 1-hour data for ~2 years and resample to 4-hour candles
INTRVL = "60m"
PERIOD = "730d"   # yfinance limit for hourly is generous enough for FX

# 4-hour tuned params
FAST_SMA = 34
SLOW_SMA = 100
EMA_PB   = 13
RSI_LEN  = 14
RSI_BUY  = 35
RSI_SELL = 65
ATR_LEN  = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
MIN_ATR_FRAC = 0.0004   # slightly lower than daily

# Spreads (points; ~10 points = 1 pip)
SPREAD_POINTS_PER_SYMBOL = {
    "EURUSD=X": 21,
    "GBPUSD=X": 29,
    "USDCHF=X": 24,
    "USDJPY=X": 22,
}
DEFAULT_SPREAD_POINTS = 25

# Paper account & risk
ACCOUNT_START   = 30000.0
RISK_PER_TRADE  = 0.01

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

# Timezone
TZ = timezone(timedelta(hours=2))  # Africa/Johannesburg

# Ledgers (separate folder from daily)
DATA_DIR     = pathlib.Path("data_fx_4h"); DATA_DIR.mkdir(exist_ok=True)
TRADES_CSV   = DATA_DIR / "trades.csv"
EQUITY_CSV   = DATA_DIR / "equity.csv"
POSITIONS_CSV= DATA_DIR / "positions.csv"

FAILED_SYMBOLS = []

# ── INDICATORS ───────────────────────────────────────────────────────────────
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
    return 0.01 if "JPY" in symbol else 0.0001

def spread_fraction(symbol: str) -> float:
    points = SPREAD_POINTS_PER_SYMBOL.get(symbol, DEFAULT_SPREAD_POINTS)
    pips = points / 10.0
    return pips * pip_size(symbol)

# ── FETCH & RESAMPLE ─────────────────────────────────────────────────────────
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
    for need in ["Open","High","Low","Close"]:
        if need not in df.columns:
            raise KeyError(f"missing columns: {need}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open","High","Low","Close","Volume"]]

def fetch_1h(symbol: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(5):
        try:
            df = yf.download(
                symbol, period=PERIOD, interval=INTRVL,
                auto_adjust=True, progress=False, threads=False,
                group_by="column"
            )
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                return _flatten_to_ohlc(df)
            last_err = RuntimeError("Empty dataframe")
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    FAILED_SYMBOLS.append(f"{symbol} ({last_err})")
    return pd.DataFrame()

def resample_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    # 4h bars aligned on UTC boundaries
    o = df1h["Open"].resample("4h").first()
    h = df1h["High"].resample("4h").max()
    l = df1h["Low"].resample("4h").min()
    c = df1h["Close"].resample("4h").last()
    v = df1h["Volume"].resample("4h").sum()
    df4 = pd.concat([o, h, l, c, v], axis=1)
    df4.columns = ["Open","High","Low","Close","Volume"]
    return df4.dropna()


def fetch_4h(symbol: str) -> pd.DataFrame:
    df1h = fetch_1h(symbol)
    if df1h.empty: return pd.DataFrame()
    df4 = resample_4h(df1h)
    return df4

# ── SIGNALS / SIZING ─────────────────────────────────────────────────────────
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

# ── LEDGER ───────────────────────────────────────────────────────────────────
def load_positions() -> pd.DataFrame:
    if POSITIONS_CSV.exists():
        df = pd.read_csv(POSITIONS_CSV)
        if len(df):
            df["opened_dt"]   = pd.to_datetime(df["opened_dt"], errors="coerce", utc=True)
            df["planned_for"] = pd.to_datetime(df["planned_for"], errors="coerce", utc=True)
        return df
    return pd.DataFrame(columns=[
        "symbol","side","status","entry","sl","tp","size_frac",
        "opened_dt","planned_for"
    ])

def save_positions(df: pd.DataFrame): df.to_csv(POSITIONS_CSV, index=False)

def append_trade(tr: dict):
    df_tr = pd.DataFrame([tr])
    if TRADES_CSV.exists():
        prev = pd.read_csv(TRADES_CSV)
        df_tr = pd.concat([prev, df_tr], ignore_index=True)
    df_tr.to_csv(TRADES_CSV, index=False)

# ── 4H FLOW ──────────────────────────────────────────────────────────────────
def next_bar_time(last_idx_utc: pd.Timestamp) -> pd.Timestamp:
    if not last_idx_utc.tzinfo:
        last_idx_utc = last_idx_utc.tz_localize("UTC")
    return (last_idx_utc + pd.Timedelta(hours=4)).replace(minute=0, second=0, microsecond=0)

def four_hour_process_and_message():
    equity = ACCOUNT_START
    lines = []
    pos_df = load_positions()

    # Build latest data
    latest = {}
    for sym in SYMBOLS:
        raw4 = fetch_4h(sym)
        if raw4.empty:
            latest[sym] = {"status":"no-data"}; continue
        d = compute(raw4)
        if d.empty:
            latest[sym] = {"status":"no-ind"}; continue
        last_ts = d.index[-1]  # UTC 4H bar end
        info = {
            "status":"ok",
            "df": d,
            "asof": last_ts.tz_convert("Africa/Johannesburg").strftime("%Y-%m-%d %H:%M"),
            "open_bar": d["Open"].iloc[-1],
            "high_bar": d["High"].iloc[-1],
            "low_bar":  d["Low"].iloc[-1],
            "close_bar":d["Close"].iloc[-1],
            "atr":      d["ATR"].iloc[-1],
            "rsi":      d["RSI"].iloc[-1],
            "long_ok":  latest_setup(d)[0],
            "short_ok": latest_setup(d)[1],
            "cost":     spread_fraction(sym),
            "next_bar_utc": next_bar_time(last_ts),
        }
        latest[sym] = info

    # 1) Promote planned → open when next_bar_utc reached/passed
    now_utc = datetime.now(timezone.utc)
    for i, row in load_positions()[load_positions()["status"]=="planned"].iterrows():
        sym = row["symbol"]; info = latest.get(sym, {})
        if info.get("status")!="ok": continue
        planned_for = row["planned_for"]
        if pd.isna(planned_for) or now_utc < planned_for:
            continue  # not time yet
        ent = info["open_bar"] + (info["cost"] if row["side"]=="LONG" else -info["cost"])
        pos_df.loc[i, ["entry","status","opened_dt"]] = [float(ent), "open", now_utc.isoformat()]
        lines.append(f"{sym} — ✅ Opened {row['side']} at bar open ≈ {ent:.6f}")
    if not pos_df.empty: save_positions(pos_df)

    # 2) Manage open positions: TP/SL on current bar; RSI exit instruction
    for i, row in pos_df[pos_df["status"]=="open"].iterrows():
        sym=row["symbol"]; info=latest.get(sym, {})
        if info.get("status")!="ok":
            lines.append(f"{sym} — ⚠️ Open {row['side']} but data unavailable"); continue
        side=row["side"]; entry=row["entry"]; sl=row["sl"]; tp=row["tp"]; sf=row["size_frac"]
        hi=info["high_bar"]; lo=info["low_bar"]; rsi=info["rsi"]; cost=info["cost"]

        exit_px = None; reason = None
        if side=="LONG":
            if hi >= tp:      exit_px, reason = tp - cost, "TP"
            elif lo <= sl:    exit_px, reason = sl - cost, "SL"
            else:
                if rsi >= 50:
                    lines.append(f"{sym} — ⚠️ Exit signal (RSI≥50), close at next 4H open. SL {sl:.6f}, TP {tp:.6f}")
                else:
                    lines.append(f"{sym} — Hold LONG. SL {sl:.6f}, TP {tp:.6f}")
        else:
            if lo <= tp:      exit_px, reason = tp + cost, "TP"
            elif hi >= sl:    exit_px, reason = sl + cost, "SL"
            else:
                if rsi <= 50:
                    lines.append(f"{sym} — ⚠️ Exit signal (RSI≤50), close at next 4H open. SL {sl:.6f}, TP {tp:.6f}")
                else:
                    lines.append(f"{sym} — Hold SHORT. SL {sl:.6f}, TP {tp:.6f}")

        if exit_px is not None:
            gross = (exit_px - entry)/entry * (1 if side=="LONG" else -1)
            pnl_r = sf * gross * equity
            append_trade({
                "symbol": sym,
                "opened": pd.to_datetime(row["opened_dt"]).date().isoformat() if pd.notna(row["opened_dt"]) else "",
                "closed": datetime.now(TZ).date().isoformat(),
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

    # 3) New planned orders
    existing = set(load_positions()["symbol"])
    for sym in SYMBOLS:
        info = latest.get(sym, {})
        if info.get("status")!="ok":
            lines.append(f"{sym} — ❌ data unavailable (skipped)"); continue

        d=info["df"]; asof=info["asof"]; cost=info["cost"]
        long_ok, short_ok = info["long_ok"], info["short_ok"]
        open_px = info["open_bar"]; atr = info["atr"]

        if (sym not in existing) and (long_ok or short_ok):
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

            plan_for = info["next_bar_utc"]  # enter at NEXT 4H bar open
            new_row = {
                "symbol": sym, "side": side, "status":"planned",
                "entry": float(planned_entry), "sl": float(sl), "tp": float(tp),
                "size_frac": float(sf),
                "opened_dt": "", "planned_for": plan_for.isoformat()
            }
            pos_df = pd.concat([pos_df, pd.DataFrame([new_row])], ignore_index=True)
            save_positions(pos_df)

            risk_pct = int(RISK_PER_TRADE*100)
            lines.append(
                f"{sym} — {'BUY' if side=='LONG' else 'SELL'} setup (as of {asof})\n"
                f"  Plan next 4H open • Entry {planned_entry:.6f} • SL {sl:.6f} • TP {tp:.6f} • Size ≈ {sf*100:.2f}% (risk {risk_pct}%)"
            )
        elif not (long_ok or short_ok):
            lines.append(f"{sym} — No setup (as of {asof})")

    return "\n".join(lines)

# ── TFSA (optional reuse) ────────────────────────────────────────────────────
def run_tfsa_sweep_summary():
    if not TRADES_CSV.exists():
        send_telegram("ℹ️ TFSA sweep 4H: no closed trades yet."); return
    df = pd.read_csv(TRADES_CSV)
    if "closed" not in df.columns or "pnl_rands" not in df.columns:
        send_telegram("ℹ️ TFSA sweep 4H: log missing columns."); return
    df["closed_dt"] = pd.to_datetime(df["closed"])
    now = datetime.now(TZ)
    this_month = df[(df["closed_dt"].dt.year==now.year)&(df["closed_dt"].dt.month==now.month)]
    pnl_sum = float(this_month["pnl_rands"].sum()) if len(this_month) else 0.0
    msg = (f"📥 TFSA sweep 4H ({now:%Y-%m})\n"
           f"Closed PnL this month: R{pnl_sum:,.2f}\n"
           f"Suggested transfer:   R{pnl_sum:,.2f}")
    send_telegram(msg)

# ── TELEGRAM ─────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        print("Telegram not configured; skipping send."); return
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

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = os.getenv("MODE","alerts").lower()
    if mode == "alerts":
        print("Running WesFX 4H alerts…")
        summary = four_hour_process_and_message()
        print("\n=== SUMMARY (4H) ===\n" + summary)
        ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        send_telegram(f"FX 4H ({ts})\n{summary}")
    elif mode == "tfsa":
        run_tfsa_sweep_summary()
    else:
        print("Unknown MODE. Use MODE=alerts (default) or MODE=tfsa.")
