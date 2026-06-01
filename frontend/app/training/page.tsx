"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type TrainingStatus } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";

const STATUS_STYLE: Record<TrainingStatus["status"], string> = {
  promoted: "text-[var(--color-bull)] border-[var(--color-bull)]",
  advisory_only: "text-[var(--color-muted)] border-[var(--color-line)]",
  never_trained: "text-[var(--color-faint)] border-[var(--color-line)]",
};

function fmt(n: number | undefined, digits = 3): string {
  return n === undefined || Number.isNaN(n) ? "—" : n.toFixed(digits);
}

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
      {s.status !== "never_trained" && (
        <span className="tabular text-[11px] text-[var(--color-faint)]">
          AUC {fmt(s.metrics.auc)} · Sharpe {fmt(s.metrics.sharpe)}
          {s.metrics.benchmark_sharpe !== undefined && ` vs ${fmt(s.metrics.benchmark_sharpe)}`}
        </span>
      )}
      <span className="text-[11px] text-[var(--color-faint)]">
        {t("training.hpo")}: {s.hpo_trials > 0 ? `${s.hpo_trials} ${t("training.trials")}` : t("training.defaultParams")}
      </span>
    </div>
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
  const nPromoted = rows?.filter((r) => r.status === "promoted").length ?? 0;
  const nHpo = rows?.filter((r) => r.hpo_trials > 0).length ?? 0;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
      >
        ← {t("training.back")}
      </Link>
      <h1 className="font-display text-3xl font-bold">{t("training.title")}</h1>
      <p className="mt-2 max-w-2xl text-sm text-[var(--color-muted)]">{t("training.intro")}</p>

      {rows && (
        <div className="mt-4 flex gap-6 text-sm text-[var(--color-muted)]">
          <span>{rows.length} {t("training.entries")}</span>
          <span className="text-[var(--color-bull)]">{nPromoted} {t("training.promotedCount")}</span>
          <span>{nHpo} {t("training.hpoCount")}</span>
        </div>
      )}

      {nHpo === 0 && rows && (
        <div className="mt-4 rounded-md border border-[var(--color-line)] bg-white/[0.02] px-4 py-3 text-xs text-[var(--color-muted)]">
          {t("training.noHpoBanner")}
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
    </main>
  );
}
