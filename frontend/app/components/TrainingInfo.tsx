"use client";

import { useEffect, useState } from "react";
import { api, type TrainingStatus } from "@/app/lib/api";
import { useTranslate } from "@/app/lib/i18n";

// Compact "Training & HPO" panel for the ticker drill-down: last training, last HPO, trial
// count and promotion status, split long vs short. Past facts only — no scheduling claims, since
// the continuous retrainer loops rather than running a fixed per-asset cron.

function fmtAgo(iso: string | null | undefined, ago: string): { rel: string; exact: string } {
  if (!iso) return { rel: "—", exact: "" };
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return { rel: "—", exact: "" };
  const s = Math.max(0, (Date.now() - ms) / 1000);
  let dur: string;
  if (s < 60) dur = `${Math.round(s)}s`;
  else if (s < 3600) dur = `${Math.floor(s / 60)}m`;
  else if (s < 86400) dur = `${Math.floor(s / 3600)}h`;
  else dur = `${Math.floor(s / 86400)}j`;
  return { rel: `${ago} ${dur}`, exact: new Date(ms).toLocaleString() };
}

function StatusChip({ status }: { status: TrainingStatus["status"] }) {
  const t = useTranslate();
  const map: Record<TrainingStatus["status"], { label: string; cls: string }> = {
    promoted: { label: t("ticker.trainPromoted"), cls: "text-[var(--color-bull)]" },
    advisory_only: { label: t("ticker.trainAdvisory"), cls: "text-[var(--color-muted)]" },
    never_trained: { label: t("ticker.trainNever"), cls: "text-[var(--color-faint)]" },
  };
  const { label, cls } = map[status];
  return <span className={`tabular text-xs ${cls}`}>{label}</span>;
}

function SideColumn({ row }: { row: TrainingStatus | undefined }) {
  const t = useTranslate();
  const ago = t("ops.ago");
  if (!row) return <span className="text-[var(--color-faint)]">—</span>;
  const trained = fmtAgo(row.trained_at, ago);
  const hpo = fmtAgo(row.last_hpo_at, ago);
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      <span className="text-[var(--color-faint)]">{t("ticker.trainLast")}</span>
      <span className="tabular text-right" title={trained.exact}>
        {trained.rel}
      </span>
      <span className="text-[var(--color-faint)]">{t("ticker.hpoLast")}</span>
      <span className="tabular text-right" title={hpo.exact}>
        {hpo.rel}
      </span>
      <span className="text-[var(--color-faint)]">{t("ticker.hpoTrials")}</span>
      <span className="tabular text-right">{row.hpo_trials.toLocaleString()}</span>
      <span className="text-[var(--color-faint)]">{t("ticker.trainStatus")}</span>
      <span className="text-right">
        <StatusChip status={row.status} />
      </span>
    </div>
  );
}

export function TrainingInfo({ ticker }: { ticker: string }) {
  const t = useTranslate();
  const [rows, setRows] = useState<TrainingStatus[] | undefined>();

  useEffect(() => {
    let alive = true;
    api
      .trainingTicker(ticker)
      .then((r) => alive && setRows(r))
      .catch(() => alive && setRows([]));
    return () => {
      alive = false;
    };
  }, [ticker]);

  if (rows === undefined) {
    return <div className="card mb-8 h-32 animate-pulse p-5" />;
  }

  const long = rows.find((r) => r.side === "long");
  const short = rows.find((r) => r.side === "short");

  return (
    <section className="card mb-8 p-5">
      <h2 className="mb-4 font-display text-lg font-bold">{t("ticker.training")}</h2>
      {!long && !short ? (
        <p className="text-sm text-[var(--color-faint)]">{t("ticker.trainNone")}</p>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2">
          <div>
            <h3 className="mb-2 text-xs uppercase tracking-widest text-[var(--color-bull)]/80">
              {t("directionLong")}
            </h3>
            <SideColumn row={long} />
          </div>
          <div>
            <h3 className="mb-2 text-xs uppercase tracking-widest text-[var(--color-bear)]/80">
              {t("directionShort")}
            </h3>
            <SideColumn row={short} />
          </div>
        </div>
      )}
    </section>
  );
}
