"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api, type Signal, type SignalConfig } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { useStrategy } from "@/app/lib/strategy";

/**
 * "Brief du jour" — the squared, bias-free order sheet.
 *
 * One screen that turns the day's PROMOTED, actionable signals into concrete instructions:
 * per asset, the direction, how much to invest (€ and % of the portfolio), the entry, the
 * stop-loss and the take-profit. When nothing clears the bar, it says so — "stay flat" is a
 * valid, disciplined answer, not an empty page. Observe-tier (near-miss, paper-only) candidates
 * are listed separately and explicitly NOT to be funded.
 */

const ACTIONABLE = new Set(["LONG", "SHORT", "BUY"]);

const fmtEur = (n: number) =>
  n.toLocaleString("fr-FR", { maximumFractionDigits: 0 }) + " €";
const fmtPx = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 4 });

function dirOf(s: Signal): "long" | "short" {
  if (s.direction) return s.direction;
  return s.signal === "SHORT" ? "short" : "long";
}

function riskReward(s: Signal): number | null {
  const reward = Math.abs(s.take_profit - s.entry);
  const risk = Math.abs(s.entry - s.stop_loss);
  return risk > 0 ? reward / risk : null;
}

export default function BriefPage() {
  const { locale } = useI18n();
  const { strategy } = useStrategy();
  const [signals, setSignals] = useState<Signal[] | null>(null);
  const [cfg, setCfg] = useState<SignalConfig | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.signals().then(setSignals).catch((e) => setErr(String(e)));
    api.signalConfig().then(setCfg).catch(() => {});
  }, []);

  const fr = locale === "fr";
  const capital = cfg?.capital ?? 0;

  const { committed, observe, date } = useMemo(() => {
    const rows = (signals ?? []).filter(
      (s) => (s.exit_strategy ?? "fixed") === strategy && ACTIONABLE.has(s.signal) && s.acted !== false,
    );
    const sortKey = (s: Signal) => s.exp_return_net ?? s.proba ?? 0;
    const committed = rows
      .filter((s) => s.promoted && s.size_shares > 0)
      .sort((a, b) => sortKey(b) - sortKey(a));
    const observe = rows
      .filter((s) => !s.promoted && s.tier === "observe" && s.size_shares > 0)
      .sort((a, b) => sortKey(b) - sortKey(a));
    const date = rows[0]?.date ?? signals?.[0]?.date ?? null;
    return { committed, observe, date };
  }, [signals, strategy]);

  const totalNotional = committed.reduce((acc, s) => acc + s.notional, 0);
  const exposurePct = capital > 0 ? (totalNotional / capital) * 100 : 0;

  const L = {
    title: fr ? "Brief du jour" : "Daily brief",
    intro: fr
      ? "La feuille d'ordres carrée : ce que la stratégie dit de faire aujourd'hui, sans biais. Investis exactement ces montants, avec ces stops et ces objectifs — ou reste à plat si rien n'est listé."
      : "The squared order sheet: what the strategy says to do today, bias-free. Invest exactly these amounts with these stops and targets — or stay flat if nothing is listed.",
    orders: fr ? "ordres à capital engagé" : "committed-capital orders",
    exposure: fr ? "exposition totale" : "total exposure",
    invest: fr ? "Investir" : "Invest",
    ofPf: fr ? "du portefeuille" : "of portfolio",
    entry: fr ? "Entrée" : "Entry",
    stop: "Stop-loss",
    target: "Take-profit",
    exp: fr ? "Gain attendu (net)" : "Expected return (net)",
    empty: fr ? "Aucun ordre aujourd'hui." : "No orders today.",
    emptyHint: fr
      ? "La discipline dit : reste à plat. Aucun signal validé ne franchit la barre — ne force pas de trade."
      : "Discipline says: stay flat. No validated signal clears the bar — don't force a trade.",
    observeTitle: fr ? "À l'observation (sans argent)" : "On watch (no money)",
    observeHint: fr
      ? "Modèles « presque promus », suivis en direct sans capital. À NE PAS financer — pour information seulement."
      : "Near-miss models, tracked live without capital. Do NOT fund — for information only.",
    back: fr ? "← Retour" : "← Back",
  };

  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
      >
        {L.back}
      </Link>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h1 className="font-display text-4xl font-extrabold tracking-tight">{L.title}</h1>
        {date && <span className="tabular font-mono text-sm text-[var(--color-faint)]">{date}</span>}
      </div>
      <p className="mt-3 max-w-2xl text-base text-[var(--color-muted)]">{L.intro}</p>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}
      {!signals && !err && <p className="mt-6 text-[var(--color-muted)]">…</p>}

      {signals && (
        <>
          <div className="mt-8 flex flex-wrap gap-6 text-sm">
            <span>
              <span className="text-2xl font-bold text-[var(--color-fg)]">{committed.length}</span>{" "}
              <span className="text-[var(--color-muted)]">{L.orders}</span>
            </span>
            {capital > 0 && (
              <span>
                <span className="text-2xl font-bold text-[var(--color-fg)]">
                  {exposurePct.toFixed(0)}%
                </span>{" "}
                <span className="text-[var(--color-muted)]">
                  {L.exposure} ({fmtEur(totalNotional)})
                </span>
              </span>
            )}
          </div>

          {committed.length === 0 ? (
            <div className="card mt-6 p-8 text-center">
              <p className="font-display text-xl font-bold">{L.empty}</p>
              <p className="mx-auto mt-2 max-w-md text-sm text-[var(--color-muted)]">{L.emptyHint}</p>
            </div>
          ) : (
            <div className="mt-6 flex flex-col gap-3">
              {committed.map((s) => (
                <OrderCard key={`${s.ticker}-${s.exit_strategy}`} s={s} capital={capital} L={L} />
              ))}
            </div>
          )}

          {observe.length > 0 && (
            <section className="mt-10">
              <h2 className="font-display text-lg font-bold text-[var(--color-faint)]">
                {L.observeTitle}
              </h2>
              <p className="mb-3 text-xs text-[var(--color-faint)]">{L.observeHint}</p>
              <div className="flex flex-col gap-2 opacity-70">
                {observe.map((s) => (
                  <OrderCard
                    key={`obs-${s.ticker}-${s.exit_strategy}`}
                    s={s}
                    capital={capital}
                    L={L}
                    muted
                  />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </main>
  );
}

function OrderCard({
  s,
  capital,
  L,
  muted = false,
}: {
  s: Signal;
  capital: number;
  L: Record<string, string>;
  muted?: boolean;
}) {
  const dir = dirOf(s);
  const isLong = dir === "long";
  const color = isLong ? "var(--color-bull)" : "var(--color-bear)";
  const pct = capital > 0 ? (s.notional / capital) * 100 : null;
  const rr = riskReward(s);
  return (
    <div className="card p-4" style={{ borderLeft: `3px solid ${muted ? "var(--color-line)" : color}` }}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span
            className="rounded px-2 py-0.5 text-xs font-bold uppercase tracking-widest"
            style={{ color, backgroundColor: `${color}1a` }}
          >
            {isLong ? "LONG" : "SHORT"}
          </span>
          <Link
            href={`/ticker/${encodeURIComponent(s.ticker)}`}
            className="font-display text-lg font-bold hover:text-[var(--color-bull)]"
          >
            {s.ticker}
          </Link>
          {rr && (
            <span className="text-xs text-[var(--color-faint)]">R/R {rr.toFixed(1)}:1</span>
          )}
        </div>
        <div className="text-right">
          <div className="font-semibold">
            {L.invest} {fmtEur(s.notional)}
          </div>
          {pct != null && (
            <div className="text-xs text-[var(--color-muted)]">
              {pct.toFixed(1)}% {L.ofPf} · {s.size_shares} ×
            </div>
          )}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
        <Field label={L.entry} value={fmtPx(s.entry)} />
        <Field label={L.stop} value={fmtPx(s.stop_loss)} color="var(--color-bear)" />
        <Field label={L.target} value={fmtPx(s.take_profit)} color="var(--color-bull)" />
      </div>
      {s.exp_return_net != null && (
        <div className="mt-2 text-xs text-[var(--color-faint)]">
          {L.exp}: {(s.exp_return_net * 100).toFixed(2)}%
        </div>
      )}
    </div>
  );
}

function Field({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-[var(--color-faint)]">{label}</div>
      <div className="tabular mt-0.5 font-medium" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  );
}
