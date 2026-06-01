import Link from "next/link";
import type { PaperEquity, Signal } from "@/app/lib/api";
import { useTranslate } from "../lib/i18n";

const fmt = (n: number) => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function verdict(signal: Signal["signal"]): { key: string; color: string } {
  if (signal === "BUY" || signal === "LONG") return { key: "light.buy", color: "var(--color-bull)" };
  if (signal === "SELL" || signal === "SHORT") return { key: "light.avoid", color: "var(--color-bear)" };
  return { key: "light.hold", color: "var(--color-neutral)" };
}

function Dots({ p }: { p: number }) {
  // Map probability distance from 0.5 into a 1–5 confidence scale.
  const level = Math.min(5, Math.max(1, Math.round(Math.abs(p - 0.5) * 12) + 1));
  return (
    <span className="inline-flex gap-0.5" aria-label={`confidence ${level}/5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className="inline-block h-1.5 w-1.5 rounded-full"
          style={{ background: i <= level ? "var(--color-fg)" : "var(--color-line)" }}
        />
      ))}
    </span>
  );
}

function IdeaCard({ s }: { s: Signal }) {
  const t = useTranslate();
  const v = verdict(s.signal);
  const p = s.proba_calibrated ?? s.proba;
  const actionable = s.signal === "BUY" || s.signal === "LONG";
  return (
    <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="card block p-4 transition-colors hover:bg-white/[0.03]">
      <div className="flex items-center justify-between">
        <span className="font-display text-lg font-bold">{s.ticker}</span>
        <span className="text-xs font-semibold" style={{ color: v.color }}>
          {t(v.key)}
        </span>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-[var(--color-muted)]">
        <span>{t("light.confidence")}</span>
        <Dots p={p} />
      </div>
      {actionable && (
        <dl className="mt-3 space-y-1 text-xs">
          <div className="flex justify-between">
            <dt className="text-[var(--color-faint)]">{t("light.entry")}</dt>
            <dd className="tabular">{fmt(s.entry)}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--color-faint)]">{t("light.stop")}</dt>
            <dd className="tabular text-[var(--color-bear)]/90">{fmt(s.stop_loss)}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--color-faint)]">{t("light.target")}</dt>
            <dd className="tabular text-[var(--color-bull)]/90">{fmt(s.take_profit)}</dd>
          </div>
        </dl>
      )}
    </Link>
  );
}

export function LightView({ signals, paper }: { signals: Signal[]; paper?: PaperEquity }) {
  const t = useTranslate();
  const ranked = [...signals].sort((a, b) => (b.proba_calibrated ?? b.proba) - (a.proba_calibrated ?? a.proba));
  const buys = ranked.filter((s) => s.signal === "BUY" || s.signal === "LONG");
  const shown = (buys.length ? buys : ranked).slice(0, 8);

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h2 className="font-display text-2xl font-bold">{t("light.title")}</h2>
        <p className="mt-1 text-sm text-[var(--color-muted)]">{t("light.subtitle")}</p>
      </section>

      {paper && (
        <div className="card flex flex-wrap items-baseline justify-between gap-2 p-4 text-sm">
          <span className="text-[var(--color-faint)]">{t("light.paper")}</span>
          <span className="tabular text-lg">
            <span style={{ color: paper.metrics.total_return_paper >= 0 ? "var(--color-bull)" : "var(--color-bear)" }}>
              {(paper.metrics.total_return_paper * 100).toFixed(1)}%
            </span>
            {Number.isFinite(paper.metrics.total_return_spy) && (
              <span className="ml-2 text-xs text-[var(--color-faint)]">
                ({(paper.metrics.total_return_spy * 100).toFixed(1)}% {t("light.vsMarket")})
              </span>
            )}
          </span>
        </div>
      )}

      <section>
        <h3 className="mb-3 text-xs uppercase tracking-widest text-[var(--color-faint)]">
          {t("light.topIdeas")}
        </h3>
        {shown.length ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {shown.map((s) => (
              <IdeaCard key={s.ticker} s={s} />
            ))}
          </div>
        ) : (
          <div className="card p-6 text-sm text-[var(--color-faint)]">{t("light.noIdeas")}</div>
        )}
      </section>

      <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-line)]/[0.03] p-4 text-xs text-[var(--color-muted)]">
        <div className="mb-1 font-semibold text-[var(--color-fg)]">{t("light.howRead")}</div>
        {t("light.howReadBody")}
      </div>
    </div>
  );
}
