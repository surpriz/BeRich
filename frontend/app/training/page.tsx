"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type TournamentCandidate, type TrainingStatus } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { Show } from "@/app/components/Show";
import { Info } from "@/app/components/Term";

const STATUS_STYLE: Record<TrainingStatus["status"], string> = {
  promoted: "text-[var(--color-bull)] border-[var(--color-bull)]",
  advisory_only: "text-[var(--color-muted)] border-[var(--color-line)]",
  never_trained: "text-[var(--color-faint)] border-[var(--color-line)]",
};

function fmt(n: number | undefined, digits = 3): string {
  return n === undefined || Number.isNaN(n) ? "—" : n.toFixed(digits);
}

const STRATEGY_LABEL: Record<string, string> = {
  fixed: "fixed",
  trailing: "trail",
  trailing_tp: "trail+TP",
};

const STRATEGY_MARK: Record<string, string> = {
  promoted: "✓",
  advisory_only: "·",
  never_trained: "–",
};

const STRATEGY_CHIP: Record<string, string> = {
  promoted: "text-[var(--color-bull)] border-[var(--color-bull)]/50",
  advisory_only: "text-[var(--color-muted)] border-[var(--color-line)]",
  never_trained: "text-[var(--color-faint)] border-[var(--color-line)]",
};

function when(iso: string | null): string {
  if (!iso) return "—";
  return iso.slice(0, 16).replace("T", " ");
}

type Grouped = { ticker: string; asset_class: string; long?: TrainingStatus; short?: TrainingStatus };

function group(rows: TrainingStatus[]): Grouped[] {
  const by = new Map<string, Grouped>();
  for (const r of rows) {
    const g = by.get(r.ticker) ?? { ticker: r.ticker, asset_class: r.asset_class };
    g[r.side] = r;
    by.set(r.ticker, g);
  }
  return [...by.values()];
}

function SideCell({ s, t }: { s: TrainingStatus | undefined; t: (k: string) => string }) {
  if (!s) return <span className="text-[var(--color-faint)]">—</span>;
  return (
    <div className="flex flex-col gap-1">
      <span className={`inline-block w-fit rounded border px-2 py-0.5 text-[11px] uppercase tracking-wider ${STATUS_STYLE[s.status]}`}>
        {t(`training.status.${s.status}`)}
      </span>
      {s.winner && <span className="text-xs text-[var(--color-muted)]">{s.winner}</span>}
      {s.horizon_days != null && s.status !== "never_trained" && (
        <span className="text-[11px] text-[var(--color-faint)]">
          {t("training.horizon")}: {s.horizon_days}
          {t("training.daysShort")}
        </span>
      )}
      <Show min="standard">
        {s.status !== "never_trained" && (
          <span className="tabular text-[11px] text-[var(--color-faint)]">
            AUC {fmt(s.metrics.auc)} · Sharpe {fmt(s.metrics.sharpe)}
            {s.metrics.benchmark_sharpe !== undefined && ` vs ${fmt(s.metrics.benchmark_sharpe)}`}
          </span>
        )}
        <span className="text-[11px] text-[var(--color-faint)]">
          {t("training.hpo")}:{" "}
          {s.hpo_trials > 0
            ? `${s.hpo_trials} ${t("training.trials")} (${t("training.acrossModels")})`
            : t("training.defaultParams")}
        </span>
        {s.strategies && s.strategies.some((st) => st.strategy !== "fixed") && (
          <div className="mt-0.5 flex flex-wrap items-center gap-1">
            <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
              {t("training.exitStrategies")}:
            </span>
            {s.strategies.map((st) => {
              const served = st.strategy === s.served_strategy;
              return (
                <span
                  key={st.strategy}
                  title={`${st.status}${served ? " · served" : ""}`}
                  className={`rounded border px-1.5 py-0.5 text-[10px] ${STRATEGY_CHIP[st.status]} ${served ? "ring-1 ring-[var(--color-bull)]/40" : ""}`}
                >
                  {STRATEGY_LABEL[st.strategy] ?? st.strategy} {STRATEGY_MARK[st.status]}
                </span>
              );
            })}
          </div>
        )}
      </Show>
    </div>
  );
}

function CandidatesTable({
  candidates,
  t,
}: {
  candidates: TournamentCandidate[];
  t: (k: string) => string;
}) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-[10px] uppercase tracking-widest text-[var(--color-faint)]">
          <th className="py-1 pr-3">{t("training.cand.model")}</th>
          <th className="py-1 pr-3">{t("training.cand.framework")}</th>
          <th className="py-1 pr-3 text-right">{t("training.cand.auc")}</th>
          <th className="py-1 pr-3 text-right">{t("training.cand.sharpe")}</th>
          <th className="py-1 pr-3 text-right">{t("training.cand.bench")}</th>
          <th className="py-1 pr-3 text-right">{t("training.cand.features")}</th>
          <th className="py-1 text-right">{t("training.cand.guard")}</th>
        </tr>
      </thead>
      <tbody>
        {candidates.map((c) => (
          <tr key={`${c.model_name}-${c.framework}`} className="border-t border-[var(--color-line)]/40">
            <td className="py-1.5 pr-3 text-[var(--color-muted)]">{c.model_name}</td>
            <td className="py-1.5 pr-3 text-[var(--color-faint)]">{c.framework}</td>
            <td className="tabular py-1.5 pr-3 text-right">{fmt(c.oos_auc)}</td>
            <td className="tabular py-1.5 pr-3 text-right">{fmt(c.strategy_sharpe)}</td>
            <td className="tabular py-1.5 pr-3 text-right text-[var(--color-faint)]">{fmt(c.benchmark_sharpe)}</td>
            <td className="tabular py-1.5 pr-3 text-right text-[var(--color-faint)]">{c.n_features}</td>
            <td className="py-1.5 text-right">
              <span className={c.beats_guard ? "text-[var(--color-bull)]" : "text-[var(--color-faint)]"}>
                {c.beats_guard ? "✓" : "—"}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function TrainingPage() {
  const { t } = useI18n();
  const [rows, setRows] = useState<TrainingStatus[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.training().then(setRows).catch((e) => setErr(String(e)));
  }, []);

  const grouped = rows ? group(rows) : [];
  const nAssets = grouped.length;
  const nSides = rows?.length ?? 0;
  const nPromoted = rows?.filter((r) => r.status === "promoted").length ?? 0;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <h1 className="font-display text-3xl font-bold">{t("training.title")}</h1>
      <p className="mt-2 max-w-2xl text-sm text-[var(--color-muted)]">{t("training.intro")}</p>

      {rows && (
        <div className="mt-4 flex gap-6 text-sm text-[var(--color-muted)]">
          <span>
            {nAssets} {t("training.assets")}
            <span className="text-[var(--color-faint)]">
              {" "}
              ({nSides} {t("training.sides")})
            </span>
          </span>
          <span className="text-[var(--color-bull)]">
            {nPromoted} {t("training.promotedCount")}
            <Info id="promoted" />
          </span>
        </div>
      )}

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}
      {!rows && !err && <p className="mt-6 text-[var(--color-muted)]">…</p>}

      {rows && (
        <div className="card mt-6 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-left text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
                <th className="px-4 py-3">{t("training.col.asset")}</th>
                <th className="px-4 py-3">{t("training.col.class")}</th>
                <th className="px-4 py-3">{t("training.col.long")}</th>
                <th className="px-4 py-3">{t("training.col.short")}</th>
                <th className="px-4 py-3">{t("training.col.lastTrained")}</th>
              </tr>
            </thead>
            <tbody>
              {grouped.map((g) => {
                const last = [g.long?.trained_at, g.short?.trained_at].filter(Boolean).sort().pop() ?? null;
                return (
                  <tr key={g.ticker} className="border-b border-[var(--color-line)]/50 align-top">
                    <td className="px-4 py-3">
                      <Link href={`/ticker/${encodeURIComponent(g.ticker)}`} className="font-medium hover:text-[var(--color-bull)]">
                        {g.ticker}
                      </Link>
                      {g.long?.status === "promoted" && g.short?.status === "promoted" && (
                        <span className="mt-1 block max-w-[12rem] text-[10px] leading-tight text-[var(--color-faint)]">
                          {t("training.bothSidesNote")}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs text-[var(--color-muted)]">{g.asset_class}</td>
                    <td className="px-4 py-3"><SideCell s={g.long} t={t} /></td>
                    <td className="px-4 py-3"><SideCell s={g.short} t={t} /></td>
                    <td className="tabular px-4 py-3 text-xs text-[var(--color-faint)]">{when(last)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {rows && (
        <Show min="expert">
          <section className="mt-10">
            <h2 className="font-display text-xl font-bold">
              {t("training.col.candidates")}
            </h2>
            <div className="mt-4 flex flex-col gap-4">
              {grouped
                .filter((g) => (g.long?.candidates.length ?? 0) + (g.short?.candidates.length ?? 0) > 0)
                .map((g) => (
                  <div key={g.ticker} className="card p-5">
                    <div className="mb-3 font-display text-base font-bold">{g.ticker}</div>
                    <div className="grid gap-6 lg:grid-cols-2">
                      {(["long", "short"] as const).map((side) => {
                        const cands = g[side]?.candidates ?? [];
                        if (cands.length === 0) return null;
                        return (
                          <div key={side}>
                            <div className="mb-1 text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
                              {t(`training.col.${side}`)}
                            </div>
                            <CandidatesTable candidates={cands} t={t} />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
            </div>
          </section>
        </Show>
      )}
    </main>
  );
}
