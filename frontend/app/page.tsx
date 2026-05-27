"use client";

import { useEffect, useState } from "react";
import { api, type Backtest, type DriftReport, type Signal } from "@/app/lib/api";
import { SignalsTable } from "./components/SignalsTable";
import { BacktestPanel } from "./components/BacktestPanel";
import { DriftPanel } from "./components/DriftPanel";

type State = {
  signals?: Signal[];
  drift?: DriftReport;
  backtest?: Backtest;
  error?: string;
};

export default function Dashboard() {
  const [s, setS] = useState<State>({});

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [signals, drift, backtest] = await Promise.all([api.signals(), api.drift(), api.backtest()]);
        if (alive) setS({ signals, drift, backtest });
      } catch (e) {
        if (alive) setS({ error: e instanceof Error ? e.message : "request failed" });
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const asOf = s.signals?.[0]?.date;

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <header className="mb-10 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-5xl font-extrabold tracking-tight">
            Be<span className="text-[var(--color-bull)]">Rich</span>
          </h1>
          <p className="mt-1 text-sm text-[var(--color-muted)]">
            Swing-trade signals · trend-probability model · walk-forward validated
          </p>
        </div>
        {asOf && (
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">As of</div>
            <div className="tabular text-lg">{asOf}</div>
          </div>
        )}
      </header>

      {/* Honest disclaimer: signals are advisory and gated by the backtest verdict. */}
      <div className="mb-8 rounded-lg border border-[var(--color-neutral)]/30 bg-[var(--color-neutral)]/[0.06] px-4 py-3 text-sm text-[var(--color-neutral)]">
        Advisory only. Trust signals when the backtest beats buy &amp; hold — not before. Not financial advice.
      </div>

      {s.error && (
        <div className="card p-6 text-[var(--color-bear)]">
          Could not reach the API ({s.error}). Start it with <code className="tabular">berich serve</code>.
        </div>
      )}

      {!s.error && (
        <div className="flex flex-col gap-8">
          <section>
            <h2 className="mb-3 font-display text-xl font-bold">Today&apos;s signals</h2>
            {s.signals ? <SignalsTable signals={s.signals} /> : <Skeleton h={280} />}
          </section>

          <div className="grid gap-8 lg:grid-cols-5">
            <section className="lg:col-span-3">{s.backtest ? <BacktestPanel bt={s.backtest} /> : <Skeleton h={460} />}</section>
            <section className="lg:col-span-2">{s.drift ? <DriftPanel drift={s.drift} /> : <Skeleton h={460} />}</section>
          </div>
        </div>
      )}
    </main>
  );
}

function Skeleton({ h }: { h: number }) {
  return <div className="card animate-pulse" style={{ height: h }} />;
}
