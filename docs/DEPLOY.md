# Deployment — production server (Phase 4b)

Deploys BeRich as three systemd services behind Caddy with automatic HTTPS at
`https://frhb101828ds.ikexpress.com`. The reverse-proxy routes `/api/*` to the
FastAPI backend and everything else to the Next.js dashboard; a scheduler
daemon runs the daily refresh + signals + paper-book update at 17:30 ET on
weekdays without anyone having to be at a terminal.

## Architecture

```
                              Internet (443)
                                   │
                                   ▼
                          ┌────────────────┐
                          │  Caddy (TLS)   │   /etc/caddy/Caddyfile
                          └───────┬────────┘   automatic Let's Encrypt
              /api/*       ┌─────┴─────┐         everything else
                ▼          ▼           ▼
       ┌───────────────┐         ┌───────────────┐
       │ berich-api    │         │ berich-       │
       │ (FastAPI)     │         │ frontend      │
       │ 127.0.0.1:8000│         │ (Next.js)     │
       └───────┬───────┘         │ 127.0.0.1:3000│
               │                 └───────────────┘
               ▼
       ┌──────────────────────────────┐
       │ data/berich.duckdb           │
       │ data/ohlcv/*.parquet         │ ◀── ┌──────────────────────────┐
       │ data/models/                 │     │ berich-scheduler         │
       └──────────────────────────────┘     │ (APScheduler, daemon)    │
                                            │ daily 17:30 ET           │
                                            │  → refresh data          │
                                            │  → generate signals      │
                                            │  → open + walk paper book│
                                            │ weekly Sat → drift check │
                                            └──────────────────────────┘
```

Caddy is the only process exposed on `0.0.0.0`; the FastAPI and Next.js
processes both bind to `127.0.0.1` and are reachable only through the proxy
(belt-and-suspenders with UFW). No CDN.

## Security choices and trade-offs

- **Service user is `root` (not a dedicated `jerome` user).** The repo lives at
  `/root/BeRich` on this server; running services as the same owner avoids a
  user-creation + relocation step. Trade-off: if a process is compromised, the
  attacker gets root. This is acceptable for a single-user personal box but is
  not the layout to copy to a multi-tenant or higher-stakes deployment.
- **API key in `/etc/berich/env` (mode 600, owner root).** Loaded by systemd
  via `EnvironmentFile=`; never committed. The frontend reads its copy from
  the same file at unit start (interpolated into a systemd `Environment=`
  directive in `berich-frontend.service`). The token is the primary secret
  on this box.
- **Optional secrets in `/etc/berich/env`** (all opt-in, scheduler
  degrades gracefully when absent):
  - `ALPHAVANTAGE_KEY` — Phase 5b news fetcher
  - `NOTIFY_EMAIL`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS` — Phase polish
    email digest fired by the scheduler when new BUY signals open paper
    trades. Use a Gmail App Password (not your account password) if
    `SMTP_HOST=smtp.gmail.com`. Missing any of the four → email path
    silently skipped, the rest of the job still runs.
- **`/api/health` is intentionally exempt from the API key.** Caddy and any
  external uptime probe can ping it; the payload now includes the last
  refresh dates for OHLCV / news / signals plus the count of today's
  signals and currently open paper positions, so a dashboard or alerting
  rule can detect a stale scheduler at a glance.
- **UFW** allows only 22 / 80 / 443. Even if 3000 or 8000 were accidentally
  rebound to `0.0.0.0`, the firewall would still block them.

## Files installed

| Path | Owner | Mode | Purpose |
|---|---|---|---|
| `/etc/berich/env` | root:root | 600 | `BERICH_API_KEY=...` — the only secret |
| `/etc/systemd/system/berich-api.service` | root:root | 644 | uvicorn behind Caddy |
| `/etc/systemd/system/berich-frontend.service` | root:root | 644 | next build && next start |
| `/etc/systemd/system/berich-scheduler.service` | root:root | 644 | apscheduler daemon |
| `/etc/caddy/Caddyfile` | root:caddy | 644 | reverse-proxy + TLS |

None of these files live in the git repo (and never should). `/etc/berich/`
is also added to `.gitignore` defensively.

## Operations

### Status

```bash
systemctl status berich-api berich-frontend berich-scheduler caddy
```

All four should show `active (running)`. If any is failing, the logs tell you why:

```bash
journalctl -u berich-api -f
journalctl -u berich-frontend -f
journalctl -u berich-scheduler -f
journalctl -u caddy -f
```

### Restart after a code change

```bash
cd /root/BeRich
git pull
systemctl restart berich-api berich-scheduler   # picks up Python changes
systemctl restart berich-frontend               # `next build` runs in ExecStartPre
```

`berich-frontend` rebuilds the Next.js bundle on every start (see
`ExecStartPre=/usr/bin/npm run build`), so a restart is enough — no manual
`npm run build` step. The API/scheduler reload the Python source directly.

### Polish v2 (multi-asset + i18n + ticker drill-down)

Polish v2 added the `/api/signals/{ticker}/explain` and `/api/universes`
endpoints, the `universes:` block in `config/berich.yaml`, a homemade
i18n context (EN/FR, default FR, persisted in `localStorage`), per-ticker
detail pages (`/ticker/<TICKER>`) and a `/how-it-works` walkthrough. No
new service, no new secret, no new background job — just a backend +
frontend rebuild:

```bash
cd /root/BeRich
git pull
systemctl restart berich-api berich-frontend
```

The model is still trained on US stocks only; non-US-stocks universes
display a banner explicitly flagging their signals as experimental and
not validated. The scheduler now ingests every ticker in
`config.all_runtime_tickers()` (legacy `watchlist` ∪ all universes) so
the dashboard has price + signal history for every symbol you list,
not just the US ones.

If you change a systemd unit file, `systemctl daemon-reload` first.

### Rotate the API key

```bash
NEW=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
sudo sed -i "s|^BERICH_API_KEY=.*|BERICH_API_KEY=${NEW}|" /etc/berich/env
sudo sed -i "s|^Environment=\"NEXT_PUBLIC_API_KEY=.*|Environment=\"NEXT_PUBLIC_API_KEY=${NEW}\"|" \
    /etc/systemd/system/berich-frontend.service
sudo systemctl daemon-reload
sudo systemctl restart berich-api berich-frontend
echo "new token: ${NEW}"
```

The scheduler doesn't use the key — it talks directly to the database, not
the HTTP API — so it doesn't need a restart.

### Renew TLS

Caddy renews automatically (Let's Encrypt) on its own timer; nothing to do.

### Smoke tests

```bash
curl -s https://frhb101828ds.ikexpress.com/api/health
# → {"status":"ok"}

curl -s https://frhb101828ds.ikexpress.com/api/signals \
     -H "X-API-Key: $(sudo awk -F= '/^BERICH_API_KEY=/{print $2}' /etc/berich/env)"
# → JSON list of today's signals

curl -s -o /dev/null -w "%{http_code}\n" https://frhb101828ds.ikexpress.com/
# → 200 (Next.js dashboard)
```

## Configuration reference

### `/etc/caddy/Caddyfile`

```
frhb101828ds.ikexpress.com {
    encode gzip
    handle /api/* {
        reverse_proxy 127.0.0.1:8000
    }
    handle {
        reverse_proxy 127.0.0.1:3000
    }
}
```

### `/etc/systemd/system/berich-api.service`

```
[Unit]
Description=BeRich FastAPI backend (uvicorn behind Caddy)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/BeRich
EnvironmentFile=/etc/berich/env
ExecStart=/root/.local/bin/uv run berich serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### `/etc/systemd/system/berich-frontend.service`

```
[Unit]
Description=BeRich Next.js dashboard (next start behind Caddy)
After=network-online.target berich-api.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/BeRich/frontend
Environment="NEXT_PUBLIC_API_URL=https://frhb101828ds.ikexpress.com/api"
Environment="NEXT_PUBLIC_API_KEY=<copy of the token in /etc/berich/env>"
Environment="PORT=3000"
ExecStartPre=/usr/bin/npm run build
ExecStart=/usr/bin/npm start -- -H 127.0.0.1 -p 3000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### `/etc/systemd/system/berich-scheduler.service`

```
[Unit]
Description=BeRich daily scheduler (refresh+signals+paper, weekly drift)
After=network-online.target berich-api.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/BeRich
EnvironmentFile=/etc/berich/env
ExecStart=/root/.local/bin/uv run berich schedule
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Things to NOT do

- Don't commit `/etc/berich/env` or the API key. The repo's `.gitignore`
  blocks `/etc/berich/` defensively, but the directory is outside the repo
  anyway — the rule is just there as a tripwire.
- Don't bind the API or the frontend to `0.0.0.0` "for testing". UFW catches
  it, but the bind addresses in the unit files are the first line of defense.
- Don't disable `Restart=always` on these units; the failure mode is the
  service dying silently between dashboard checks.
- Don't `systemctl stop berich-api` without realizing `berich-frontend` was
  built against the API and the dashboard will show fetch errors until both
  are back up. Restart both.
