#!/usr/bin/env bash
# One-shot post-fix verification (scheduled via systemd timer, see verify_sync setup).
# Checks that the 22:30 Paris scheduler run captured settled US daily closes and
# whether the Alpha Vantage news fetch succeeded or was rate-limited.
set -uo pipefail
cd /root/BeRich || exit 1

LOG="/root/BeRich/data/verify_sync_$(TZ=Europe/Paris date +%Y%m%d_%H%M).log"
exec >"$LOG" 2>&1

echo "=== BeRich sync verification — $(TZ=Europe/Paris date) (Paris) ==="
echo
echo "### Point 1 — OHLCV bars (AAPL): May 28 full volume + May 29 present?"
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/ohlcv/AAPL.parquet')
print(df.tail(5).to_string())
last = df.index[-1].strftime('%Y-%m-%d')
has_29 = '2026-05-29' in df.index.strftime('%Y-%m-%d').tolist()
try:
    vol28 = int(df.loc['2026-05-28', 'volume'])
    print(f'\n  -> 2026-05-28 volume = {vol28:,} (full bar if ~40-50M, partial if ~12M)')
except KeyError:
    print('\n  -> 2026-05-28 bar MISSING')
print(f'  -> 2026-05-29 bar present: {has_29}')
print(f'  -> last bar date: {last}')
"
echo
echo "### Point 2 — API freshness (/health)"
curl -s http://localhost:8000/health || echo 'API unreachable on localhost:8000'
echo
echo
echo "### Point 3 — Friday 22:30 scheduler run logs"
journalctl -u berich-scheduler --no-pager --since "2026-05-29 22:00" 2>/dev/null \
  | grep -iE "daily_paper|news|finbert|ingest|rate|alphavantage" | tail -40
echo
echo "=== end of report ==="
