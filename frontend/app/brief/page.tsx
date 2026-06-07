"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  api,
  type PaperPosition,
  type PlannedOrder,
  type Signal,
  type SignalConfig,
} from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";

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

// ``embedded`` renders the Brief inside another page (the Simple-level home) — same content,
// no back-link (it would loop back to itself).
// Next automatic execution: weekdays at 22:30 Europe/Paris (after the US close completes the
// daily bar). Shown on the forecast section so "upcoming orders" can never be mistaken for
// confirmed, act-on-it-now instructions.
function nextRunLabel(fr: boolean): string {
  const paris = new Date(new Date().toLocaleString("en-US", { timeZone: "Europe/Paris" }));
  const d = new Date(paris);
  const beforeRun = d.getHours() < 22 || (d.getHours() === 22 && d.getMinutes() < 30);
  if (!beforeRun) d.setDate(d.getDate() + 1);
  while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() + 1);
  const days = fr
    ? ["dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"]
    : ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const months = fr
    ? ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    : ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return fr
    ? `${days[d.getDay()]} ${d.getDate()} ${months[d.getMonth()]} à 22h30 (Paris)`
    : `${days[d.getDay()]}, ${months[d.getMonth()]} ${d.getDate()} at 22:30 (Paris)`;
}

// Compact per-book badge: the robot runs the three exit-strategy books at once, so every card
// says which book it belongs to.
function bookLabel(strategy: string | null | undefined, fr: boolean): string {
  const s = strategy ?? "fixed";
  if (s === "trailing") return fr ? "Suiveur" : "Trailing";
  if (s === "trailing_tp") return fr ? "Suiveur+TP" : "Trailing+TP";
  return fr ? "Fixe" : "Fixed";
}

export default function BriefPage({ embedded = false }: { embedded?: boolean }) {
  const { locale } = useI18n();
  const [signals, setSignals] = useState<Signal[] | null>(null);
  const [plan, setPlan] = useState<PlannedOrder[] | null>(null);
  const [cfg, setCfg] = useState<SignalConfig | null>(null);
  const [positions, setPositions] = useState<PaperPosition[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    // The Brief's new orders come from the PLAN (post-caps, portfolio-coherent), not raw signals.
    api.briefPlan().then(setPlan).catch((e) => setErr(String(e)));
    api.signals().then(setSignals).catch(() => {});
    api.signalConfig().then(setCfg).catch(() => {});
  }, []);

  // Open committed positions ("hold these") across ALL exit-strategy books: the robot trades the
  // three books simultaneously, so the Brief shows the WHOLE portfolio (with a per-book badge)
  // rather than the slice behind the strategy toggle — that slicing made positions "disappear".
  useEffect(() => {
    api
      .paperPositions(undefined, "promoted")
      .then((r) => setPositions(r.positions))
      .catch(() => setPositions([]));
  }, []);

  const fr = locale === "fr";
  const capital = cfg?.capital ?? 0;

  // New orders: the full portfolio-coherent plan (all books), biggest first.
  const committed = useMemo(
    () => [...(plan ?? [])].sort((a, b) => b.notional - a.notional),
    [plan],
  );
  // Observe (watch-only, no capital) stays informational from raw signals — all books too.
  const observe = useMemo(
    () =>
      (signals ?? []).filter(
        (s) =>
          ACTIONABLE.has(s.signal) &&
          s.acted !== false &&
          !s.promoted &&
          s.tier === "observe" &&
          s.size_shares > 0,
      ),
    [signals],
  );
  const date = committed[0]?.date_open ?? signals?.[0]?.date ?? null;

  const totalNotional = committed.reduce((acc, o) => acc + o.notional, 0);
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
    holdTitle: fr ? "Positions en cours — à conserver" : "Open positions — hold",
    holdHint: fr
      ? "Tout le portefeuille, tous livres confondus (badge = livre). Conserve jusqu'à la sortie (stop / objectif) — ne ferme pas « au feeling »."
      : "The whole portfolio, across all books (badge = book). Hold to the exit (stop / target) — don't close on a hunch.",
    holdEmpty: fr
      ? "Aucune position ouverte — le robot est entièrement à plat."
      : "No open positions — the robot is fully flat.",
    ordersTitle: fr ? "Prochains ordres — prévision" : "Upcoming orders — forecast",
    forecastWarn: fr
      ? `⚠️ PRÉVISION, pas une instruction. Recalculée en continu : l'exécution réelle a lieu automatiquement le ${nextRunLabel(true)}, aux prix de clôture de ce jour-là. Les prix affichés (dernière clôture connue) sont indicatifs — l'ordre peut changer ou disparaître d'ici là. Ne reproduis pas ces ordres manuellement : le robot s'en charge, et ce qui a été réellement exécuté apparaît dans « Positions en cours ».`
      : `⚠️ FORECAST, not an instruction. Recomputed continuously: the real execution happens automatically on ${nextRunLabel(false)}, at that day's closing prices. Prices shown (last known close) are indicative — an order can change or vanish by then. Do not replicate these orders manually: the robot handles it, and what was actually executed shows up under "Open positions".`,
    ordersNote: fr
      ? "Tous livres confondus (badge = livre), chacun avec son budget. Tailles déjà ajustées aux plafonds et aux positions ouvertes."
      : "Across all books (badge = book), each with its own budget. Sizes already scaled to the caps and to open positions.",
    current: fr ? "Cours" : "Price",
    mtm: fr ? "P&L latent" : "Unrealized P&L",
    held: fr ? "j" : "d",
    howTitle: fr ? "Comment lire cette page" : "How to read this page",
    back: fr ? "← Retour" : "← Back",
  };

  const HOW: { h: string; b: string }[] = fr
    ? [
        { h: "À quoi ça sert", b: "Une feuille d'ordres disciplinée : fais exactement ce qui est écrit, sans intuition. Tout est déjà calculé (sens, taille, stop, objectif)." },
        { h: "Positions en cours — à conserver", b: "Ce que tu détiens déjà. Règle : tenir jusqu'au stop ou à l'objectif. Le P&L latent fluctue — ce n'est PAS un déclencheur ; seuls le stop et l'objectif le sont." },
        { h: "Nouveaux ordres du jour", b: "Les GO à ouvrir. Investir X € (Y %) = la taille exacte, déjà gérée (~1 % de risque, ajustée aux plafonds). R/R x:1 = tu risques 1 pour viser x." },
        { h: "LONG vs SHORT", b: "LONG = pari à la hausse (stop sous l'entrée, objectif au-dessus). SHORT = pari à la baisse (stop au-dessus, objectif en-dessous)." },
        { h: "Le stop est sacré", b: "C'est ta perte maximale, fixée d'avance. Ne le déplace pas, ne ferme pas « au feeling ». Les pertes sont maîtrisées par le stop + le coupe-circuit global." },
        { h: "Vide = reste à plat", b: "Aucun ordre = ne rien faire aujourd'hui. L'inaction disciplinée est une décision valide." },
        { h: "Les molettes en haut", b: "Profil de risque = change la taille des ordres. Sélecteur de stratégie = quel livre (fixed / trailing) tu regardes." },
        { h: "La vérité", b: "L'edge est mince : il y aura des pertes, mais bornées. La rentabilité vient de petites pertes maîtrisées + laisser courir les gains, répété avec discipline. Outil de conseil : en réel, c'est toi qui passes les ordres à ces niveaux." },
      ]
    : [
        { h: "What it's for", b: "A disciplined order sheet: do exactly what's written, no hunches. Everything is pre-computed (side, size, stop, target)." },
        { h: "Open positions — hold", b: "What you already hold. Rule: hold to the stop or target. Unrealized P&L fluctuates — it is NOT a trigger; only the stop and target are." },
        { h: "New orders today", b: "The GOs to open. Invest X € (Y %) = the exact, risk-managed size (~1% risk, scaled to caps). R/R x:1 = risk 1 to aim for x." },
        { h: "LONG vs SHORT", b: "LONG = bet up (stop below entry, target above). SHORT = bet down (stop above, target below)." },
        { h: "The stop is sacred", b: "Your max loss, fixed in advance. Don't move it, don't close on a hunch. Losses are bounded by the stop + the global kill-switch." },
        { h: "Empty = stay flat", b: "No orders = do nothing today. Disciplined inaction is a valid decision." },
        { h: "The top toggles", b: "Risk profile = changes order sizes. Strategy selector = which book (fixed / trailing) you view." },
        { h: "The truth", b: "The edge is thin: there will be losses, but bounded. Profit comes from small managed losses + letting winners run, repeated with discipline. Advisory tool: in real life, you place the orders at these levels." },
      ];

  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      {!embedded && (
        <Link
          href="/"
          className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
        >
          {L.back}
        </Link>
      )}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h1 className="font-display text-4xl font-extrabold tracking-tight">{L.title}</h1>
        {date && <span className="tabular font-mono text-sm text-[var(--color-faint)]">{date}</span>}
      </div>
      <p className="mt-3 max-w-2xl text-base text-[var(--color-muted)]">{L.intro}</p>

      <details className="card mt-4 p-4">
        <summary className="cursor-pointer select-none text-sm font-semibold text-[var(--color-muted)] hover:text-[var(--color-fg)]">
          ⓘ {L.howTitle}
        </summary>
        <dl className="mt-3 flex flex-col gap-2.5">
          {HOW.map((x) => (
            <div key={x.h}>
              <dt className="text-sm font-semibold text-[var(--color-fg)]">{x.h}</dt>
              <dd className="text-sm text-[var(--color-muted)]">{x.b}</dd>
            </div>
          ))}
        </dl>
      </details>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}
      {!plan && !err && <p className="mt-6 text-[var(--color-muted)]">…</p>}

      {plan && (
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

          <section className="mt-8">
            <h2 className="font-display text-lg font-bold">
              {L.holdTitle}{" "}
              <span className="text-sm font-normal text-[var(--color-faint)]">
                ({positions.length})
              </span>
            </h2>
            <p className="mb-3 text-xs text-[var(--color-faint)]">{L.holdHint}</p>
            {positions.length === 0 ? (
              <div className="card p-5 text-center text-sm text-[var(--color-muted)]">
                {L.holdEmpty}
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {positions.map((p) => (
                  <HoldCard key={`hold-${p.ticker}-${p.exit_strategy}`} p={p} L={L} fr={fr} />
                ))}
              </div>
            )}
          </section>

          <h2 className="mt-10 font-display text-lg font-bold">{L.ordersTitle}</h2>
          {committed.length === 0 ? (
            <div className="card mt-3 p-8 text-center">
              <p className="font-display text-xl font-bold">{L.empty}</p>
              <p className="mx-auto mt-2 max-w-md text-sm text-[var(--color-muted)]">{L.emptyHint}</p>
            </div>
          ) : (
            <>
              <div className="mt-3 rounded-lg border border-[#e0a83d]/40 bg-[#e0a83d]/10 px-4 py-3 text-sm text-[var(--color-muted)]">
                {L.forecastWarn}
              </div>
              <p className="mt-2 text-xs text-[var(--color-faint)]">{L.ordersNote}</p>
              <div className="mt-3 flex flex-col gap-3">
                {committed.map((o) => (
                  <PlanCard
                    key={`${o.ticker}-${o.exit_strategy}`}
                    o={o}
                    capital={capital}
                    L={L}
                    fr={fr}
                  />
                ))}
              </div>
            </>
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

function PlanCard({
  o,
  capital,
  L,
  fr,
}: {
  o: PlannedOrder;
  capital: number;
  L: Record<string, string>;
  fr: boolean;
}) {
  const isLong = o.direction === "long";
  const color = isLong ? "var(--color-bull)" : "var(--color-bear)";
  const pct = capital > 0 ? (o.notional / capital) * 100 : null;
  const risk = Math.abs(o.entry - o.stop);
  const rr = risk > 0 ? Math.abs(o.target - o.entry) / risk : null;
  return (
    <div className="card p-4" style={{ borderLeft: `3px solid ${color}` }}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span
            className="rounded px-2 py-0.5 text-xs font-bold uppercase tracking-widest"
            style={{ color, backgroundColor: `${color}1a` }}
          >
            {isLong ? "LONG" : "SHORT"}
          </span>
          <Link
            href={`/ticker/${encodeURIComponent(o.ticker)}`}
            className="font-display text-lg font-bold hover:text-[var(--color-bull)]"
          >
            {o.ticker}
          </Link>
          <span className="rounded bg-white/[0.05] px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
            {bookLabel(o.exit_strategy, fr)}
          </span>
          {rr && <span className="text-xs text-[var(--color-faint)]">R/R {rr.toFixed(1)}:1</span>}
        </div>
        <div className="text-right">
          <div className="font-semibold">
            {L.invest} {fmtEur(o.notional)}
          </div>
          {pct != null && (
            <div className="text-xs text-[var(--color-muted)]">
              {pct.toFixed(1)}% {L.ofPf} · {o.size_shares} ×
            </div>
          )}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
        <Field label={L.entry} value={fmtPx(o.entry)} />
        <Field label={L.stop} value={fmtPx(o.stop)} color="var(--color-bear)" />
        <Field label={L.target} value={fmtPx(o.target)} color="var(--color-bull)" />
      </div>
    </div>
  );
}

function HoldCard({
  p,
  L,
  fr,
}: {
  p: PaperPosition;
  L: Record<string, string>;
  fr: boolean;
}) {
  const isLong = (p.direction ?? "long") === "long";
  const color = isLong ? "var(--color-bull)" : "var(--color-bear)";
  const pnlColor =
    p.mtm_pct > 0
      ? "var(--color-bull)"
      : p.mtm_pct < 0
        ? "var(--color-bear)"
        : "var(--color-muted)";
  // For a trailing trade the live ratcheting stop is the effective stop to watch.
  const stop = p.trail_stop ?? p.stop;
  return (
    <div className="card p-4" style={{ borderLeft: `3px solid ${color}` }}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span
            className="rounded px-2 py-0.5 text-xs font-bold uppercase tracking-widest"
            style={{ color, backgroundColor: `${color}1a` }}
          >
            {isLong ? "LONG" : "SHORT"}
          </span>
          <Link
            href={`/ticker/${encodeURIComponent(p.ticker)}`}
            className="font-display text-lg font-bold hover:text-[var(--color-bull)]"
          >
            {p.ticker}
          </Link>
          <span className="rounded bg-white/[0.05] px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-muted)]">
            {bookLabel(p.exit_strategy, fr)}
          </span>
          <span className="text-xs text-[var(--color-faint)]">
            {p.days_held} {L.held} · {p.size_shares} ×
          </span>
        </div>
        <div className="text-right">
          <div className="font-semibold" style={{ color: pnlColor }}>
            {p.mtm_pct >= 0 ? "+" : ""}
            {(p.mtm_pct * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-[var(--color-muted)]">{L.mtm}</div>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-3 text-sm">
        <Field label={L.entry} value={fmtPx(p.entry)} />
        <Field label={L.current} value={fmtPx(p.current_price)} />
        <Field label={L.stop} value={fmtPx(stop)} color="var(--color-bear)" />
        <Field label={L.target} value={fmtPx(p.target)} color="var(--color-bull)" />
      </div>
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
