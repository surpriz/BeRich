"use client";

import { useEffect, useRef } from "react";
import { AreaSeries, LineSeries, createChart, type IChartApi } from "lightweight-charts";
import type { PaperEquity } from "@/app/lib/api";

// Paper portfolio equity (lime area) vs same-capital SPY buy & hold (muted line).
// Honest visual: if the line is below the dashed benchmark, the model is losing.
export function PaperEquityChart({ equity }: { equity: PaperEquity }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || equity.dates.length === 0) return;
    const chart: IChartApi = createChart(ref.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#8a94a4",
        fontFamily: "var(--font-jetbrains)",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "rgba(20,24,31,0.06)" },
        horzLines: { color: "rgba(20,24,31,0.06)" },
      },
      rightPriceScale: { borderColor: "#e6e8ef" },
      timeScale: { borderColor: "#e6e8ef" },
      height: 280,
      autoSize: true,
    });

    const paper = chart.addSeries(AreaSeries, {
      lineColor: "#10936b",
      topColor: "rgba(16,147,107,0.22)",
      bottomColor: "rgba(16,147,107,0.02)",
      lineWidth: 2,
    });
    const spy = chart.addSeries(LineSeries, {
      color: "#5a6472",
      lineWidth: 1,
      lineStyle: 2,
    });

    paper.setData(equity.dates.map((time, i) => ({ time, value: equity.equity_paper[i] })));
    const spySeries = equity.dates
      .map((time, i) => ({ time, value: equity.equity_spy[i] }))
      .filter((p): p is { time: string; value: number } => p.value !== null);
    if (spySeries.length > 0) spy.setData(spySeries);
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [equity]);

  if (equity.dates.length === 0) {
    return (
      <div className="flex h-[280px] items-center justify-center text-sm text-[var(--color-faint)]">
        No paper trades yet. Run <code className="tabular">berich paper update</code>.
      </div>
    );
  }

  return <div ref={ref} className="h-[280px] w-full" />;
}
