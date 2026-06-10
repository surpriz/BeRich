"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Replication } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { fmtPrice } from "@/app/lib/format";
import { PageIntro } from "@/app/components/PageIntro";

/**
 * "Réplication" — the morning copy-trading ritual, built from EXECUTIONS only.
 *
 * Three action lists from the robot's last daily run (facts, never forecasts):
 * OPEN what it opened, CLOSE what it closed, ALIGN the trailing stops it ratcheted.
 * A capital input rescales every amount from the 10k simulation base to the user's
 * real broker capital. During the forward test this page is a dry-run rehearsal;
 * later it is, word for word, the public copy-trading artifact.
 *
 * The body is exported as <ReplicationView> so the "Today" hub (Brief) can show it
 * behind a "what to copy this morning" tab without duplicating the logic.
 */

const CAPITAL_KEY = "berich.copy.capital";

function bookLabel(strategy: string | null | undefined, fr: boolean): string {
  const s = strategy ?? "fixed";
  if (s === "trailing") return fr ? "Suiveur" : "Trailing";
  if (s === "trailing_tp") return fr ? "Suiveur+TP" : "Trailing+TP";
  return fr ? "Fixe" : "Fixed";
}

const fmtPx = fmtPrice;

export default function CopyPage() {
  const { locale } = useI18n();
  const fr = locale === "fr";
  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <Link
        href="/brief"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
      >
        {fr ? "← Retour" : "← Back"}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">
        {fr ? "Réplication du robot" : "Replicate the robot"}
      </h1>
      <div className="mt-6">
        <PageIntro page="copy" />
      </div>
      <ReplicationView />
    </main>
  );
}

export function ReplicationView() {
  const { locale } = useI18n();
  const fr = locale === "fr";
  const [rep, setRep] = useState<Replication | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [capital, setCapital] = useState<number>(10_000);

  useEffect(() => {
    api.replication().then(setRep).catch((e) => setErr(String(e)));
    const stored = typeof window !== "undefined" ? window.localStorage.getItem(CAPITAL_KEY) : null;
    const v = stored ? Number(stored) : Number.NaN;
    if (Number.isFinite(v) && v > 0) setCapital(v);
  }, []);

  const setCap = (v: number) => {
    setCapital(v);
    if (typeof window !== "undefined" && Number.isFinite(v) && v > 0) {
      window.localStorage.setItem(CAPITAL_KEY, String(v));
    }
  };

  const base = rep?.capital_base ?? 10_000;
  const factor = capital > 0 ? capital / base : 1;
  const eur = (n: number) => n.toLocaleString("fr-FR", { maximumFractionDigits: 0 }) + " €";

  const L = {
    title: fr ? "Réplication du robot" : "Replicate the robot",
    intro: fr
      ? "Le rituel du matin pour copier le robot chez ton courtier : trois listes d'actions, construites UNIQUEMENT sur ce qu'il a réellement exécuté au dernier passage (jours ouvrés, 22h30 Paris) — jamais sur ses prévisions. Pendant le forward test, sers-t'en comme entraînement à blanc, sans argent."
      : "The morning ritual to copy the robot at your broker: three action lists built ONLY from what it actually executed at the last run (weekdays, 22:30 Paris) — never from its forecasts. During the forward test, use it as a dry-run rehearsal, no money.",
    capital: fr ? "Ton capital chez le courtier" : "Your broker capital",
    capitalHint: fr
      ? `Les montants ci-dessous sont mis à l'échelle : montant simulé × (ton capital ÷ ${eur(base)}).`
      : `Amounts below are rescaled: simulated amount × (your capital ÷ ${eur(base)}).`,
    openTitle: fr ? "🟢 OUVRIR — exécuté au dernier passage" : "🟢 OPEN — executed at the last run",
    openHint: fr
      ? "Ouvre la même position chez ton courtier, puis pose IMMÉDIATEMENT le stop et l'objectif indiqués."
      : "Open the same position at your broker, then IMMEDIATELY place the stop and target shown.",
    closeTitle: fr ? "🔴 FERMER — clôturé au dernier passage" : "🔴 CLOSE — closed at the last run",
    closeHint: fr
      ? "Si ton courtier n'a pas déjà déclenché ton stop/objectif tout seul, ferme cette position."
      : "If your broker's stop/target didn't already trigger on its own, close this position.",
    adjustTitle: fr ? "🔧 ALIGNER LES STOPS — positions suiveuses" : "🔧 ALIGN STOPS — trailing positions",
    adjustHint: fr
      ? "Pour chaque position suiveuse ouverte, vérifie que ton stop chez le courtier est au niveau indiqué (il se resserre jour après jour, jamais ne recule)."
      : "For each open trailing position, make sure your broker stop sits at the level shown (it ratchets tighter day after day, never loosens).",
    nothing: fr ? "Rien cette nuit." : "Nothing last night.",
    invest: fr ? "Montant" : "Amount",
    entry: fr ? "Entrée exécutée" : "Executed entry",
    stop: "Stop",
    target: fr ? "Objectif" : "Target",
    exit: fr ? "Sortie" : "Exit",
    reason: fr ? "Raison" : "Reason",
    stopNow: fr ? "Stop du jour" : "Today's stop",
    back: fr ? "← Retour" : "← Back",
    asOf: fr ? "Mis à jour" : "As of",
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
    <>
      <div className="card flex flex-wrap items-center gap-4 p-4">
        <label className="text-sm font-semibold" htmlFor="copy-capital">
          {L.capital}
        </label>
        <input
          id="copy-capital"
          type="number"
          min={100}
          step={100}
          value={capital}
          onChange={(e) => setCap(Number(e.target.value))}
          className="w-36 rounded-md border border-[var(--color-line)] bg-transparent px-3 py-1.5 text-right tabular"
        />
        <span className="text-xs text-[var(--color-faint)]">{L.capitalHint}</span>
      </div>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}
      {!rep && !err && <p className="mt-6 text-[var(--color-muted)]">…</p>}

      {rep && (
        <>
          <Section title={L.openTitle} hint={L.openHint} empty={L.nothing} n={rep.open.length}>
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
                    <Book s={o.exit_strategy} fr={fr} />
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">
                      {L.invest} {eur(o.notional * factor)}
                    </div>
                    <div className="text-xs text-[var(--color-muted)]">
                      ({eur(o.notional)} {fr ? "simulé" : "simulated"})
                    </div>
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                  <Field label={L.entry} value={fmtPx(o.entry)} />
                  <Field label={L.stop} value={fmtPx(o.stop)} color="var(--color-bear)" />
                  <Field label={L.target} value={fmtPx(o.target)} color="var(--color-bull)" />
                </div>
              </div>
            ))}
          </Section>

          <Section title={L.closeTitle} hint={L.closeHint} empty={L.nothing} n={rep.close.length}>
            {rep.close.map((c) => (
              <div key={`c-${c.ticker}-${c.exit_strategy}`} className="card p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <Dir d={c.direction} />
                    <span className="font-display text-lg font-bold">{c.ticker}</span>
                    <Book s={c.exit_strategy} fr={fr} />
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-[var(--color-muted)]">
                      {L.reason}: {reason(c.status)}
                    </span>
                    {c.exit_price != null && (
                      <span className="tabular">
                        {L.exit}: {fmtPx(c.exit_price)}
                      </span>
                    )}
                    {c.pnl_pct != null && (
                      <span
                        className="tabular font-semibold"
                        style={{
                          color: c.pnl_pct >= 0 ? "var(--color-bull)" : "var(--color-bear)",
                        }}
                      >
                        {c.pnl_pct >= 0 ? "+" : ""}
                        {(c.pnl_pct * 100).toFixed(2)}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </Section>

          <Section
            title={L.adjustTitle}
            hint={L.adjustHint}
            empty={L.nothing}
            n={rep.adjust.length}
          >
            {rep.adjust.map((a) => (
              <div key={`a-${a.ticker}-${a.exit_strategy}`} className="card p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <Dir d={a.direction === "short" ? "short" : "long"} />
                    <span className="font-display text-lg font-bold">{a.ticker}</span>
                    <Book s={a.exit_strategy} fr={fr} />
                  </div>
                  <div className="tabular text-sm">
                    {L.stopNow}:{" "}
                    <span className="font-semibold" style={{ color: "var(--color-bear)" }}>
                      {fmtPx(a.effective_stop)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </Section>

          <p className="mt-8 text-xs text-[var(--color-faint)]">
            {L.asOf}: {new Date(rep.as_of).toLocaleString(fr ? "fr-FR" : "en-US")}
          </p>
        </>
      )}
    </>
  );
}

function Section({
  title,
  hint,
  empty,
  n,
  children,
}: {
  title: string;
  hint: string;
  empty: string;
  n: number;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-8">
      <h2 className="font-display text-lg font-bold">
        {title} <span className="text-sm font-normal text-[var(--color-faint)]">({n})</span>
      </h2>
      <p className="mb-3 text-xs text-[var(--color-faint)]">{hint}</p>
      {n === 0 ? (
        <div className="card p-4 text-sm text-[var(--color-muted)]">{empty}</div>
      ) : (
        <div className="flex flex-col gap-3">{children}</div>
      )}
    </section>
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

function Book({ s, fr }: { s: string | null | undefined; fr: boolean }) {
  return (
    <span className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
      {bookLabel(s, fr)}
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
