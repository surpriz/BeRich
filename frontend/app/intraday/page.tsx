"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  api,
  type IntradayPaperEquity,
  type IntradayPlannedOrder,
  type IntradayReplication,
  type PaperClosedTrade,
  type PaperPositions,
  type Signal,
} from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { fmtPrice } from "@/app/lib/format";
import { useLevel } from "@/app/lib/level";

/**
 * Intraday V2 (POC) — 1h crypto, single pair BTC/USDT, served from its own interval-dimensioned
 * models through the SAME promote() guard. A deliberately separate, honest surface:
 *   - Forecast orders (what the committed book WOULD open) — flagged as a forecast,
 *   - Executed (what the last hourly run actually did) — the copy-trading list,
 *   - Paper P&L vs the pair's own buy & hold.
 * Nothing here touches the frozen daily Brief/Wallet. Capital is paper-only.
 */
export default function IntradayPage() {
  const { locale } = useI18n();
  const fr = locale === "fr";
  const { level } = useLevel();
  const tier = "promoted";

  const [signals, setSignals] = useState<Signal[] | null>(null);
  const [forecast, setForecast] = useState<IntradayPlannedOrder[] | null>(null);
  const [executed, setExecuted] = useState<IntradayReplication | null>(null);
  const [equity, setEquity] = useState<IntradayPaperEquity | null>(null);
  const [positions, setPositions] = useState<PaperPositions | null>(null);
  const [closed, setClosed] = useState<PaperClosedTrade[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.intradaySignals().then(setSignals).catch((e) => setErr(String(e)));
    api.intradayBriefPlan().then(setForecast).catch(() => {});
    api.intradayReplication().then(setExecuted).catch(() => {});
    api.intradayPaperEquity(undefined, tier).then(setEquity).catch(() => {});
    api.intradayPaperPositions(undefined, tier).then(setPositions).catch(() => {});
    api.intradayPaperClosed(50, undefined, tier).then(setClosed).catch(() => {});
  }, []);

  const L = {
    title: fr ? "Intraday (POC) — crypto 1h" : "Intraday (POC) — 1h crypto",
    disclaimer: fr
      ? "POC expérimental : une seule paire (BTC/USDT), barres horaires, même garde-fou promote() que le swing. Ce n'est PAS une preuve d'edge — on mesure si un edge passe le garde-fou. Capital papier uniquement, aucun argent réel."
      : "Experimental POC: a single pair (BTC/USDT), hourly bars, the same promote() guard as the swing book. This is NOT an edge claim — it measures whether an edge clears the guard. Paper capital only, no real money.",
    signal: fr ? "Signal actuel" : "Current signal",
    forecast: fr ? "Ordres prévus (prévision)" : "Forecast orders (forecast)",
    forecastHint: fr
      ? "Recalculé en continu — ce que le livre engagé OUVRIRAIT. Seul le passage horaire exécute. Ne pas copier d'ici."
      : "Continuously recomputed — what the committed book WOULD open. Only the hourly run executes. Do not copy from here.",
    executed: fr ? "Exécuté (dernier passage horaire)" : "Executed (last hourly run)",
    executedHint: fr
      ? "Ce que le robot a réellement fait au dernier passage — la liste à copier."
      : "What the robot actually did at the last run — the list to copy.",
    pnl: fr ? "P&L papier (vs achat-conservation BTC)" : "Paper P&L (vs BTC buy & hold)",
    none: fr ? "Rien pour l'instant." : "Nothing yet.",
    back: fr ? "← Retour" : "← Back",
    openPos: fr ? "Positions ouvertes" : "Open positions",
    closedTrades: fr ? "Trades fermés" : "Closed trades",
    held: fr ? "h détenue(s)" : "h held",
  };

  const m = equity?.metrics;
  const capital = m?.capital ?? 0;
  const ret = m?.total_return_paper ?? 0;
  const benchRet = m?.total_return_bench ?? 0;
  const value = capital * (1 + ret);
  const eur = (n: number | null | undefined) =>
    n == null ? "—" : n.toLocaleString("fr-FR", { maximumFractionDigits: 0 }) + " €";
  const pct = (n: number | null | undefined) =>
    n == null ? "—" : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(2)}%`;

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <Link
        href="/brief"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
      >
        {L.back}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{L.title}</h1>

      <div className="mt-4 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface-2)] p-4 text-sm text-[var(--color-muted)]">
        {L.disclaimer}
      </div>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}

      {/* Current signal — shown at every level (the novice essential). */}
      <section className="mt-8">
        <h2 className="text-lg font-semibold">{L.signal}</h2>
        {signals && signals.length > 0 ? (
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {signals.map((s, i) => (
              <div key={i} className="card p-4">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{s.ticker}</span>
                  <span className="rounded bg-[var(--color-surface-2)] px-2 py-0.5 text-xs">
                    {s.signal}
                    {s.exit_strategy ? ` · ${s.exit_strategy}` : ""}
                  </span>
                </div>
                <div className="tabular mt-2 text-sm text-[var(--color-muted)]">
                  {fr ? "Proba" : "Proba"}: {((s.proba ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-3 text-[var(--color-muted)]">{L.none}</p>
        )}
      </section>

      {/* Forecast orders — clearly a forecast, never an instruction. */}
      <section className="mt-8">
        <h2 className="text-lg font-semibold">{L.forecast}</h2>
        <p className="mt-1 text-xs text-[var(--color-faint)]">{L.forecastHint}</p>
        {forecast && forecast.length > 0 ? (
          <ul className="mt-3 space-y-2">
            {forecast.map((o, i) => (
              <li key={i} className="card flex items-center justify-between p-3 text-sm">
                <span className="font-medium">
                  {o.ticker} · {o.direction} · {o.exit_strategy ?? "fixed"}
                </span>
                <span className="tabular text-[var(--color-muted)]">
                  {o.size_shares} @ {fmtPrice(o.entry)} · {eur(o.notional)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-[var(--color-muted)]">{L.none}</p>
        )}
      </section>

      {/* Executed — the copy list. Shown from standard up (novices stay on the forecast/signal). */}
      {level !== "simple" && (
        <section className="mt-8">
          <h2 className="text-lg font-semibold">{L.executed}</h2>
          <p className="mt-1 text-xs text-[var(--color-faint)]">{L.executedHint}</p>
          {executed && (executed.open.length > 0 || executed.close.length > 0) ? (
            <ul className="mt-3 space-y-2">
              {executed.open.map((o, i) => (
                <li key={`o${i}`} className="card flex items-center justify-between p-3 text-sm">
                  <span className="font-medium text-[var(--color-bull)]">
                    {fr ? "OUVERT" : "OPEN"} {o.ticker} · {o.direction}
                  </span>
                  <span className="tabular text-[var(--color-muted)]">
                    {o.size_shares} @ {fmtPrice(o.entry)}
                  </span>
                </li>
              ))}
              {executed.close.map((c, i) => (
                <li key={`c${i}`} className="card flex items-center justify-between p-3 text-sm">
                  <span className="font-medium text-[var(--color-bear)]">
                    {fr ? "FERMÉ" : "CLOSE"} {c.ticker} · {c.status}
                  </span>
                  <span className="tabular text-[var(--color-muted)]">
                    {c.pnl_pct != null ? pct(c.pnl_pct) : "—"}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-[var(--color-muted)]">{L.none}</p>
          )}
        </section>
      )}

      {/* Paper P&L — standard up. */}
      {level !== "simple" && m && (
        <section className="mt-8">
          <h2 className="text-lg font-semibold">{L.pnl}</h2>
          <div className="mt-3 grid gap-4 sm:grid-cols-3">
            <div className="card p-5">
              <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
                {fr ? "Valeur" : "Value"}
              </div>
              <div className="tabular mt-1 text-2xl font-bold">{eur(value)}</div>
            </div>
            <div className="card p-5">
              <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
                {fr ? "Rendement papier" : "Paper return"}
              </div>
              <div
                className="tabular mt-1 text-2xl font-bold"
                style={{ color: ret >= 0 ? "var(--color-bull)" : "var(--color-bear)" }}
              >
                {pct(ret)}
              </div>
            </div>
            <div className="card p-5">
              <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
                {fr ? "BTC achat-conservation" : "BTC buy & hold"}
              </div>
              <div className="tabular mt-1 text-2xl font-bold">
                {Number.isNaN(benchRet) ? "—" : pct(benchRet)}
              </div>
            </div>
          </div>
          <div className="mt-3 text-sm text-[var(--color-muted)]">
            {L.openPos}: {positions?.n ?? 0} · {L.closedTrades}: {m.n_closed} ·{" "}
            {fr ? "Réussite" : "Win rate"}: {(m.win_rate * 100).toFixed(0)}%
          </div>
        </section>
      )}

      {/* Expert: open positions + closed-trade detail. */}
      {level === "expert" && positions && positions.positions.length > 0 && (
        <section className="mt-8">
          <h2 className="text-lg font-semibold">{L.openPos}</h2>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[var(--color-faint)]">
                <tr>
                  <th className="py-1">{fr ? "Actif" : "Asset"}</th>
                  <th>{fr ? "Sens" : "Side"}</th>
                  <th>{fr ? "Entrée" : "Entry"}</th>
                  <th>{fr ? "Prix" : "Price"}</th>
                  <th>MTM</th>
                  <th>{L.held}</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {positions.positions.map((p, i) => (
                  <tr key={i} className="border-t border-[var(--color-line)]">
                    <td className="py-1">{p.ticker}</td>
                    <td>{p.direction}</td>
                    <td>{fmtPrice(p.entry)}</td>
                    <td>{fmtPrice(p.current_price)}</td>
                    <td style={{ color: (p.mtm_pct ?? 0) >= 0 ? "var(--color-bull)" : "var(--color-bear)" }}>
                      {pct(p.mtm_pct)}
                    </td>
                    <td>{p.days_held}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {level === "expert" && closed && closed.length > 0 && (
        <section className="mt-8">
          <h2 className="text-lg font-semibold">{L.closedTrades}</h2>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[var(--color-faint)]">
                <tr>
                  <th className="py-1">{fr ? "Actif" : "Asset"}</th>
                  <th>{fr ? "Statut" : "Status"}</th>
                  <th>{fr ? "Sortie" : "Exit"}</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {closed.map((c, i) => (
                  <tr key={i} className="border-t border-[var(--color-line)]">
                    <td className="py-1">{c.ticker}</td>
                    <td>{c.status}</td>
                    <td>{fmtPrice(c.exit_price)}</td>
                    <td style={{ color: c.pnl_pct >= 0 ? "var(--color-bull)" : "var(--color-bear)" }}>
                      {pct(c.pnl_pct)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
