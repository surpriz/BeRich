"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  api,
  type PaperClosedTrade,
  type PaperEquity,
  type PaperPositions,
  type SignalConfig,
} from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { PageIntro } from "@/app/components/PageIntro";
import { PaperPanel } from "@/app/components/PaperPanel";

/**
 * "Mon portefeuille" — the paper-trading wallet, in one glance.
 *
 * Everything money-related lives here (moved off the Signals home): starting capital,
 * current value, gains/losses, the equity curve vs SPY, open positions and the full
 * closed-trade history — across ALL exit-strategy books (the robot trades the three books
 * simultaneously). The Committed/Observation toggle separates the real-capital track from
 * the no-capital watchlist.
 */

export default function WalletPage() {
  const { locale } = useI18n();
  const fr = locale === "fr";
  const [tier, setTier] = useState<"promoted" | "observe">("promoted");
  const [equity, setEquity] = useState<PaperEquity | null>(null);
  const [positions, setPositions] = useState<PaperPositions | null>(null);
  const [closed, setClosed] = useState<PaperClosedTrade[] | null>(null);
  const [cfg, setCfg] = useState<SignalConfig | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    // No strategy filter: the wallet is the WHOLE portfolio across the three books.
    api.paperEquity(undefined, tier).then(setEquity).catch((e) => setErr(String(e)));
    api.paperPositions(undefined, tier).then(setPositions).catch(() => {});
    api.paperClosed(50, undefined, tier).then(setClosed).catch(() => {});
    api.signalConfig().then(setCfg).catch(() => {});
  }, [tier]);

  const L = {
    title: fr ? "Mon portefeuille (papier)" : "My paper portfolio",
    intro: fr
      ? "Simulation honnête, tous livres confondus (Fixe + Suiveur + Suiveur+TP). Capital de départ, valeur actuelle, courbe d'évolution face au SPY, positions ouvertes et historique des trades fermés."
      : "Honest simulation, across all books (Fixed + Trailing + Trailing+TP). Starting capital, current value, equity curve vs SPY, open positions and the closed-trade history.",
    start: fr ? "Capital de départ" : "Starting capital",
    now: fr ? "Valeur actuelle" : "Current value",
    pnl: fr ? "Gains / pertes" : "Gains / losses",
    back: fr ? "← Retour" : "← Back",
  };

  const m = equity?.metrics;
  const capital = m?.capital ?? cfg?.capital ?? 0;
  const ret = m?.total_return_paper ?? 0;
  const value = capital * (1 + ret);
  const pnl = value - capital;
  const pnlColor = pnl >= 0 ? "var(--color-bull)" : "var(--color-bear)";
  const eur = (n: number) => n.toLocaleString("fr-FR", { maximumFractionDigits: 0 }) + " €";

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <Link
        href="/brief"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
      >
        {L.back}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{L.title}</h1>
      <div className="mt-4">
        <PageIntro page="wallet" />
      </div>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}

      {m && (
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <div className="card p-5">
            <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {L.start}
            </div>
            <div className="tabular mt-1 text-2xl font-bold">{eur(capital)}</div>
          </div>
          <div className="card p-5">
            <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {L.now}
            </div>
            <div className="tabular mt-1 text-2xl font-bold">{eur(value)}</div>
          </div>
          <div className="card p-5">
            <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {L.pnl}
            </div>
            <div className="tabular mt-1 text-2xl font-bold" style={{ color: pnlColor }}>
              {pnl >= 0 ? "+" : ""}
              {eur(pnl)}{" "}
              <span className="text-sm font-normal">
                ({ret >= 0 ? "+" : ""}
                {(ret * 100).toFixed(2)}%)
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="mt-8">
        {equity && positions && closed ? (
          <PaperPanel
            equity={equity}
            positions={positions.positions}
            closed={closed}
            cfg={cfg ?? undefined}
            tier={tier}
            onTierChange={setTier}
          />
        ) : (
          !err && <p className="text-[var(--color-muted)]">…</p>
        )}
      </div>
    </main>
  );
}
