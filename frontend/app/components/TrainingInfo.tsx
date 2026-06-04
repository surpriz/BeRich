"use client";

import { useEffect, useState } from "react";
import {
  api,
  type Signal,
  type SignalConfig,
  type TrainingStatus,
} from "@/app/lib/api";
import { useTranslate } from "@/app/lib/i18n";
import { ProbaGauge } from "@/app/components/bars";

// Compact "Training & HPO" panel for the ticker drill-down: last training, last HPO, trial
// count, promotion status and today's per-side signal, split long vs short. Training facts are
// past-only (no scheduling claims — the retrainer loops rather than running a fixed per-asset
// cron); the signal row shows whether each side actually fires today, so "promoted" (a quality
// verdict) isn't mistaken for "has an open position".

// Latest served signal for an exit strategy (history is date-descending, so first match wins).
function latestSignal(history: Signal[] | undefined, strategy: string | null | undefined) {
  const want = strategy ?? "fixed";
  return history?.find((s) => (s.exit_strategy ?? "fixed") === want);
}

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

// Today's signal for one side: its P(win) vs that side's threshold. Fires (colored direction)
// when the served strategy's probability clears the threshold, else faint "below threshold".
function SignalCell({
  proba,
  threshold,
  side,
}: {
  proba: number | null | undefined;
  threshold: number | null | undefined;
  side: "long" | "short";
}) {
  const t = useTranslate();
  if (proba == null || threshold == null) {
    return <span className="text-[var(--color-faint)]">—</span>;
  }
  const fires = proba >= threshold;
  const dirLabel = side === "long" ? t("directionLong") : t("directionShort");
  const textColor = side === "long" ? "text-[var(--color-bull)]" : "text-[var(--color-bear)]";
  const fillColor = side === "long" ? "var(--color-bull)" : "var(--color-bear)";
  return (
    <div className="flex flex-col items-end gap-1">
      <span className="tabular">
        {proba.toFixed(3)}{" "}
        {fires ? (
          <span className={textColor}>· {dirLabel}</span>
        ) : (
          <span className="text-[var(--color-faint)]">· {t("ticker.sigBelow")}</span>
        )}
      </span>
      <ProbaGauge proba={proba} threshold={threshold} color={fillColor} />
    </div>
  );
}

function SideColumn({
  row,
  history,
  cfg,
}: {
  row: TrainingStatus | undefined;
  history: Signal[] | undefined;
  cfg: SignalConfig | undefined;
}) {
  const t = useTranslate();
  const ago = t("ops.ago");
  if (!row) return <span className="text-[var(--color-faint)]">—</span>;
  const trained = fmtAgo(row.trained_at, ago);
  const hpo = fmtAgo(row.last_hpo_at, ago);
  const sig = latestSignal(history, row.served_strategy);
  const proba = row.side === "long" ? sig?.proba_long : sig?.proba_short;
  const threshold = row.side === "long" ? cfg?.buy_threshold : cfg?.short_threshold;
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      <span className="text-[var(--color-faint)]">{t("ticker.sigToday")}</span>
      <span className="text-right">
        <SignalCell proba={proba} threshold={threshold} side={row.side} />
      </span>
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

export function TrainingInfo({
  ticker,
  history,
  cfg,
}: {
  ticker: string;
  history: Signal[] | undefined;
  cfg: SignalConfig | undefined;
}) {
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
            <SideColumn row={long} history={history} cfg={cfg} />
          </div>
          <div>
            <h3 className="mb-2 text-xs uppercase tracking-widest text-[var(--color-bear)]/80">
              {t("directionShort")}
            </h3>
            <SideColumn row={short} history={history} cfg={cfg} />
          </div>
        </div>
      )}
    </section>
  );
}
