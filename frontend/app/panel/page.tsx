"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  api,
  type AssetClass,
  type PaperClosedTrade,
  type PaperEquity,
  type PaperPositions,
  type Replication,
  type SignalConfig,
  type Universes,
} from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { PaperPanel } from "@/app/components/PaperPanel";

/**
 * "Panel diversifié" — the observe-tier book, surfaced as a first-class, varied panel.
 *
 * The committed book (promoted tier) is forex-heavy because its long gate must beat each
 * asset's buy & hold — a near-flat bar on forex, a steep one on trending US/FR stocks. The
 * observe tier holds the near-miss models that span ALL classes (US/FR/crypto/commodities/
 * forex), so it is the diversified panel the user wanted, running on its own separate 10k
 * simulated capital. It is exploratory by construction: lower-confidence models, no real
 * capital, kept clearly apart from the rigorous committed forward test.
 */

const CLASS_LABEL: Record<AssetClass, { fr: string; en: string }> = {
  us_stocks: { fr: "Actions US", en: "US stocks" },
  fr_stocks: { fr: "Actions FR", en: "FR stocks" },
  forex: { fr: "Forex", en: "Forex" },
  crypto: { fr: "Crypto", en: "Crypto" },
  commodities: { fr: "Matières 1ères", en: "Commodities" },
};

const CLASS_ORDER: AssetClass[] = ["us_stocks", "fr_stocks", "crypto", "commodities", "forex"];

const fmtPx = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 4 });

function bookLabel(strategy: string | null | undefined, fr: boolean): string {
  const s = strategy ?? "fixed";
  if (s === "trailing") return fr ? "Suiveur" : "Trailing";
  if (s === "trailing_tp") return fr ? "Suiveur+TP" : "Trailing+TP";
  return fr ? "Fixe" : "Fixed";
}

export default function PanelPage() {
  const { locale } = useI18n();
  const fr = locale === "fr";
  const [equity, setEquity] = useState<PaperEquity | null>(null);
  const [positions, setPositions] = useState<PaperPositions | null>(null);
  const [closed, setClosed] = useState<PaperClosedTrade[] | null>(null);
  const [cfg, setCfg] = useState<SignalConfig | null>(null);
  const [universes, setUniverses] = useState<Universes | null>(null);
  const [rep, setRep] = useState<Replication | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.paperEquity(undefined, "observe").then(setEquity).catch((e) => setErr(String(e)));
    api.paperPositions(undefined, "observe").then(setPositions).catch(() => {});
    api.paperClosed(50, undefined, "observe").then(setClosed).catch(() => {});
    api.signalConfig().then(setCfg).catch(() => {});
    api.universes().then(setUniverses).catch(() => {});
    api.replication("observe").then(setRep).catch(() => {});
  }, []);

  const classOf = (ticker: string): AssetClass | null => {
    if (!universes) return null;
    for (const c of CLASS_ORDER) {
      if (universes[c]?.includes(ticker)) return c;
    }
    return null;
  };

  // Count open positions per class — the visible proof the panel really is diversified.
  const byClass: Record<string, number> = {};
  for (const p of positions?.positions ?? []) {
    const c = classOf(p.ticker) ?? "other";
    byClass[c] = (byClass[c] ?? 0) + 1;
  }

  const m = equity?.metrics;
  const capital = m?.capital ?? cfg?.capital ?? 0;
  const ret = m?.total_return_paper ?? 0;
  const value = capital * (1 + ret);
  const pnl = value - capital;
  const pnlColor = pnl >= 0 ? "var(--color-bull)" : "var(--color-bear)";
  const eur = (n: number) => n.toLocaleString("fr-FR", { maximumFractionDigits: 0 }) + " €";

  const L = {
    title: fr ? "Panel diversifié" : "Diversified panel",
    intro: fr
      ? "Un second livre papier, sur capital simulé séparé (10 000 €), qui couvre TOUTES les classes — actions US/FR, crypto, matières premières, forex. Il fait tourner les modèles « observe » : presque promus, edge plus faible, suivis sans capital réel. Même discipline de gestion du risque que le Portefeuille (kill-switch de drawdown, plafonds d'exposition et de positions), appliquée à son propre capital."
      : "A second paper book, on its own separate simulated capital (10,000 €), spanning ALL classes — US/FR stocks, crypto, commodities, forex. It runs the 'observe' models: near-misses with a weaker edge, tracked with no real capital. Same risk discipline as the committed Wallet (drawdown kill-switch, exposure and position caps), applied to its own capital.",
    warn: fr
      ? "⚠️ Exploratoire, confiance moindre. Ces modèles n'ont pas passé le gate durci du livre engagé : à suivre par curiosité et pour la variété, PAS comme conseil validé. Le forward test rigoureux reste le livre « Portefeuille »."
      : "⚠️ Exploratory, lower confidence. These models did not clear the committed book's hardened gate: follow for variety and curiosity, NOT as validated advice. The rigorous forward test stays on the 'Wallet' book.",
    variety: fr ? "Variété (positions ouvertes par classe)" : "Variety (open positions by class)",
    start: fr ? "Capital de départ" : "Starting capital",
    now: fr ? "Valeur actuelle" : "Current value",
    pnl: fr ? "Gains / pertes" : "Gains / losses",
    copyTitle: fr ? "Reproduire ce panel" : "Replicate this panel",
    copyHint: fr
      ? "Ce que le robot a réellement exécuté sur ce livre au dernier passage (jamais une prévision)."
      : "What the robot actually executed on this book at the last run (never a forecast).",
    open: fr ? "🟢 Ouvertes" : "🟢 Opened",
    close: fr ? "🔴 Fermées" : "🔴 Closed",
    adjust: fr ? "🔧 Stops suiveurs à aligner" : "🔧 Trailing stops to align",
    nothing: fr ? "Rien au dernier passage." : "Nothing at the last run.",
    entry: fr ? "Entrée" : "Entry",
    stop: "Stop",
    target: fr ? "Objectif" : "Target",
    back: fr ? "← Portefeuille engagé" : "← Committed wallet",
  };

  const reason = (s: string) =>
    (
      ({
        closed_stop: fr ? "stop touché" : "stop hit",
        closed_target: fr ? "objectif atteint" : "target hit",
        closed_time: fr ? "échéance" : "time exit",
        closed_trail: fr ? "stop suiveur" : "trail stop",
      }) as Record<string, string>
    )[s] ?? s;

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <Link
        href="/wallet"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
      >
        {L.back}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{L.title}</h1>
      <p className="mt-3 max-w-3xl text-base text-[var(--color-muted)]">{L.intro}</p>

      <div className="mt-4 rounded-lg border border-[var(--color-warn,#e0a83d)]/40 bg-[var(--color-warn,#e0a83d)]/10 px-4 py-3 text-sm text-[var(--color-ink)]">
        {L.warn}
      </div>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}

      {m && (
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <Card label={L.start} value={eur(capital)} />
          <Card label={L.now} value={eur(value)} />
          <Card
            label={L.pnl}
            value={`${pnl >= 0 ? "+" : ""}${eur(pnl)}`}
            sub={`(${ret >= 0 ? "+" : ""}${(ret * 100).toFixed(2)}%)`}
            color={pnlColor}
          />
        </div>
      )}

      {positions && positions.positions.length > 0 && (
        <div className="mt-6">
          <div className="mb-2 text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
            {L.variety}
          </div>
          <div className="flex flex-wrap gap-2">
            {CLASS_ORDER.filter((c) => byClass[c]).map((c) => (
              <span
                key={c}
                className="rounded-full border border-[var(--color-line)] px-3 py-1 text-sm"
              >
                {fr ? CLASS_LABEL[c].fr : CLASS_LABEL[c].en}{" "}
                <span className="font-bold text-[var(--color-accent)]">{byClass[c]}</span>
              </span>
            ))}
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
            tier="observe"
          />
        ) : (
          !err && <p className="text-[var(--color-muted)]">…</p>
        )}
      </div>

      {rep && (rep.open.length > 0 || rep.close.length > 0 || rep.adjust.length > 0) && (
        <section className="mt-12">
          <h2 className="font-display text-xl font-bold">{L.copyTitle}</h2>
          <p className="mb-4 text-xs text-[var(--color-faint)]">{L.copyHint}</p>

          <Group title={L.open} n={rep.open.length} empty={L.nothing}>
            {rep.open.map((o) => (
              <div
                key={`o-${o.ticker}-${o.exit_strategy}`}
                className="card p-4"
                style={{
                  borderLeft: `3px solid ${o.direction === "long" ? "var(--color-bull)" : "var(--color-bear)"}`,
                }}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <Dir d={o.direction} />
                    <span className="font-display text-lg font-bold">{o.ticker}</span>
                    <Tag>{bookLabel(o.exit_strategy, fr)}</Tag>
                  </div>
                  <div className="tabular text-sm text-[var(--color-muted)]">{eur(o.notional)}</div>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                  <Field label={L.entry} value={fmtPx(o.entry)} />
                  <Field label={L.stop} value={fmtPx(o.stop)} color="var(--color-bear)" />
                  <Field label={L.target} value={fmtPx(o.target)} color="var(--color-bull)" />
                </div>
              </div>
            ))}
          </Group>

          <Group title={L.close} n={rep.close.length} empty={L.nothing}>
            {rep.close.map((c) => (
              <div key={`c-${c.ticker}-${c.exit_strategy}`} className="card flex flex-wrap items-center justify-between gap-2 p-4">
                <div className="flex items-center gap-3">
                  <Dir d={c.direction} />
                  <span className="font-display text-lg font-bold">{c.ticker}</span>
                  <Tag>{bookLabel(c.exit_strategy, fr)}</Tag>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-[var(--color-muted)]">{reason(c.status)}</span>
                  {c.pnl_pct != null && (
                    <span
                      className="tabular font-semibold"
                      style={{ color: c.pnl_pct >= 0 ? "var(--color-bull)" : "var(--color-bear)" }}
                    >
                      {c.pnl_pct >= 0 ? "+" : ""}
                      {(c.pnl_pct * 100).toFixed(2)}%
                    </span>
                  )}
                </div>
              </div>
            ))}
          </Group>

          <Group title={L.adjust} n={rep.adjust.length} empty={L.nothing}>
            {rep.adjust.map((a) => (
              <div key={`a-${a.ticker}-${a.exit_strategy}`} className="card flex flex-wrap items-center justify-between gap-2 p-4">
                <div className="flex items-center gap-3">
                  <Dir d={a.direction === "short" ? "short" : "long"} />
                  <span className="font-display text-lg font-bold">{a.ticker}</span>
                  <Tag>{bookLabel(a.exit_strategy, fr)}</Tag>
                </div>
                <div className="tabular text-sm">
                  {L.stop}:{" "}
                  <span className="font-semibold" style={{ color: "var(--color-bear)" }}>
                    {fmtPx(a.effective_stop)}
                  </span>
                </div>
              </div>
            ))}
          </Group>
        </section>
      )}
    </main>
  );
}

function Card({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="card p-5">
      <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">{label}</div>
      <div className="tabular mt-1 text-2xl font-bold" style={color ? { color } : undefined}>
        {value} {sub && <span className="text-sm font-normal">{sub}</span>}
      </div>
    </div>
  );
}

function Group({
  title,
  n,
  empty,
  children,
}: {
  title: string;
  n: number;
  empty: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-6">
      <h3 className="font-display text-sm font-bold">
        {title} <span className="text-[var(--color-faint)]">({n})</span>
      </h3>
      {n === 0 ? (
        <div className="card mt-2 p-4 text-sm text-[var(--color-muted)]">{empty}</div>
      ) : (
        <div className="mt-2 flex flex-col gap-3">{children}</div>
      )}
    </div>
  );
}

function Dir({ d }: { d: "long" | "short" }) {
  const color = d === "long" ? "var(--color-bull)" : "var(--color-bear)";
  return (
    <span
      className="rounded px-2 py-0.5 text-xs font-bold uppercase tracking-widest"
      style={{ color, backgroundColor: `${color}1a` }}
    >
      {d.toUpperCase()}
    </span>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded bg-white/[0.05] px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
      {children}
    </span>
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
