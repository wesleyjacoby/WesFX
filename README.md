# WesFX — Daily FX Alerts + TFSA Sweep (Paper)

Mechanical daily system for EURUSD, GBPUSD, USDCHF, USDJPY:
trend (SMA50>200 / <) + pullback to EMA20 + RSI(14) gate + ATR filters,
1% risk per trade on a R30k model, realistic spread costs, Telegram alerts.

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env  # fill TG_BOT_TOKEN, TG_CHAT_ID
python fx_daily_system.py
```

- Evening run prints a signal summary and sends a Telegram DM.
- Logs:
  - `data_fx/trades.csv`  — closed paper trades
  - `data_fx/equity.csv`  — equity snapshots

**Month-end sweep summary**
```bash
MODE=tfsa python fx_daily_system.py
```

This sums this month’s closed PnL and DM’s a suggested TFSA transfer.

## GitHub Actions

Add repo **Secrets**:
- `TG_BOT_TOKEN`
- `TG_CHAT_ID`

Workflows:
- `fx-alerts.yml` → weekdays 18:30 SAST
- `tfsa-sweep.yml` → last day ~18:00 SAST

## Notes

- Spread config is in `SPREAD_POINTS_PER_SYMBOL` (points; ~10 points = 1 pip).
- Positions are paper-sized to ~1% risk using ATR-based stops.
- This is research/education only; not financial advice.
