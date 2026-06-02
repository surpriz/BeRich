"use client";

import { useEffect, useMemo, useState } from "react";
import {
  api,
  PAPER_EXPORT_URL,
  type AssetClass,
  type Backtest,
  type DriftReport,
  type LongShortEquity,
  type LongShortLeg,
  type PaperCalibration,
  type PaperClosedTrade,
  type PaperEquity,
  type PaperPositions,
  type Signal,
  type SignalConfig,
  type Universes,
} from "@/app/lib/api";
import { SignalsTable } from "./components/SignalsTable";
import { LongShortPanel } from "./components/LongShortPanel";
import { BacktestPanel } from "./components/BacktestPanel";
import { CalibrationCard } from "./components/CalibrationCard";
import { DriftPanel } from "./components/DriftPanel";
import { PaperPanel } from "./components/PaperPanel";
import { UniverseTabs } from "./components/UniverseTabs";
import { Show } from "./components/Show";
import { useTranslate } from "./lib/i18n";
import { useLevel } from "./lib/level";

type State = {
  signals?: Signal[];
  drift?: DriftReport;
  backtest?: Backtest;
  paperEquity?: PaperEquity;
  paperPositions?: PaperPositions;
  paperClosed?: PaperClosedTrade[];
  paperCalibration?: PaperCalibration;
  universes?: Universes;
  longshortBasket?: LongShortLeg[];
  longshortEquity?: LongShortEquity;
  cfg?: SignalConfig;
  error?: string;
};

export default function Dashboard() {
  const [s, setS] = useState<State>({});
  const [activeUniverse, setActiveUniverse] = useState<AssetClass>("us_stocks");
  const t = useTranslate();
  const { level } = useLevel();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [
          signals,
          drift,
          backtest,
          paperEquity,
          paperPositions,
          paperClosed,
          paperCalibration,
          universes,
          longshortBasket,
          longshortEquity,
          cfg,
        ] = await Promise.all([
          api.signals(),
          api.drift(),
          api.backtest(),
          api.paperEquity(),
          api.paperPositions(),
          api.paperClosed(),
          api.paperCalibration(),
          api.universes().catch(() => undefined),
          api.longshortBasket().catch(() => [] as LongShortLeg[]),
          api.longshortEquity().catch(() => undefined),
          api.signalConfig().catch(() => undefined),
        ]);
        if (alive)
          setS({
            signals,
            drift,
            backtest,
            paperEquity,
            paperPositions,
            paperClosed,
            paperCalibration,
            universes,
            longshortBasket,
            longshortEquity,
            cfg,
          });
      } catch (e) {
        if (alive) setS({ error: e instanceof Error ? e.message : "request failed" });
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const filtered = useMemo(() => {
    if (!s.signals) return undefined;
    if (!s.universes) return s.signals;
    const allowed = new Set(s.universes[activeUniverse]);
    return s.signals.filter((sig) => allowed.has(sig.ticker));
  }, [s.signals, s.universes, activeUniverse]);

  const asOf = filtered?.[0]?.date ?? s.signals?.[0]?.date;

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-5xl font-extrabold tracking-tight">
            Be<span className="text-[var(--color-bull)]">Rich</span>
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted)]">{t("app.tagline")}</p>
          <p className="mt-2 max-w-xl text-xs text-[var(--color-faint)]">{t(`level.hint.${level}`)}</p>
        </div>
        {asOf && (
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {t("app.asOf")}
            </div>
            <div className="tabular text-lg">{asOf}</div>
          </div>
        )}
      </header>

      <div className="mb-3 rounded-lg border border-[var(--color-line)] bg-white/[0.02] px-4 py-3 text-sm">
        <div className="mb-1 font-semibold text-[var(--color-fg)]">{t("legend.title")}</div>
        <div className="flex flex-col gap-1.5">
          <div className="flex items-start gap-2">
            <span className="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-bull)] ring-1 ring-[var(--color-bull)]/40">
              {t("signal.validated")}
            </span>
            <span className="text-[var(--color-muted)]">{t("legend.validated")}</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-faint)] ring-1 ring-[var(--color-line)]">
              {t("signal.advisory")}
            </span>
            <span className="text-[var(--color-muted)]">{t("legend.advisory")}</span>
          </div>
        </div>
      </div>

      <div className="mb-8 rounded-lg border border-[var(--color-line)] bg-white/[0.02] px-4 py-3 text-sm text-[var(--color-muted)]">
        <span className="font-semibold text-[var(--color-fg)]">{t("philosophy.title")}</span>{" "}
        {t("philosophy.body")}
      </div>

      {s.error && (
        <div className="card p-6 text-[var(--color-bear)]">
          {t("error.api")} ({s.error}). {t("error.startServe")}{" "}
          <code className="tabular">berich serve</code>.
        </div>
      )}

      {!s.error && (
        <div className="flex flex-col gap-8">
          <section>
            <UniverseTabs
              universes={s.universes}
              active={activeUniverse}
              onChange={setActiveUniverse}
            />
            <h2 className="mb-3 font-display text-xl font-bold">
              {t("dashboard.todaySignals")}
            </h2>
            {filtered ? <SignalsTable signals={filtered} cfg={s.cfg} /> : <Skeleton h={280} />}
          </section>

          <Show min="standard">
            {s.paperEquity && s.paperPositions && s.paperClosed ? (
              <PaperPanel
                equity={s.paperEquity}
                positions={s.paperPositions.positions}
                closed={s.paperClosed}
              />
            ) : (
              <Skeleton h={600} />
            )}

            {s.backtest ? <BacktestPanel bt={s.backtest} /> : <Skeleton h={460} />}
          </Show>

          <Show min="expert">
            {s.longshortBasket !== undefined && (
              <LongShortPanel basket={s.longshortBasket} equity={s.longshortEquity} />
            )}

            <div className="grid gap-8 lg:grid-cols-2">
              <section>
                {s.paperCalibration ? (
                  <CalibrationCard calibration={s.paperCalibration} />
                ) : (
                  <Skeleton h={300} />
                )}
              </section>
              <section className="card flex flex-col justify-between p-5">
                <div>
                  <h3 className="font-display text-sm font-bold uppercase tracking-widest text-[var(--color-muted)]">
                    {t("dashboard.export")}
                  </h3>
                  <p className="mt-2 text-sm text-[var(--color-faint)]">
                    {t("dashboard.exportDesc")}
                  </p>
                </div>
                <a
                  href={PAPER_EXPORT_URL}
                  className="mt-4 inline-block self-start rounded-md border border-[var(--color-line)] bg-[var(--color-line)]/[0.04] px-4 py-2 text-sm text-[var(--color-bull)] hover:bg-[var(--color-line)]/[0.10]"
                >
                  {t("dashboard.downloadCsv")}
                </a>
              </section>
            </div>

            {s.drift ? <DriftPanel drift={s.drift} /> : <Skeleton h={460} />}
          </Show>
        </div>
      )}
    </main>
  );
}

function Skeleton({ h }: { h: number }) {
  return <div className="card animate-pulse" style={{ height: h }} />;
}
