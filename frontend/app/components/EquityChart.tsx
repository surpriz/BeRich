"use client";

import { useEffect, useRef } from "react";
import { AreaSeries, LineSeries, createChart, type IChartApi } from "lightweight-charts";
import type { Backtest } from "@/app/lib/api";

// Strategy (lime area) vs equal-weight buy & hold (muted line).
export function EquityChart({ equity }: { equity: Backtest["equity"] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
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
      height: 300,
      autoSize: true,
    });

    const strat = chart.addSeries(AreaSeries, {
      lineColor: "#10936b",
      topColor: "rgba(16,147,107,0.22)",
      bottomColor: "rgba(16,147,107,0.02)",
      lineWidth: 2,
    });
    const bench = chart.addSeries(LineSeries, {
      color: "#5a6472",
      lineWidth: 1,
      lineStyle: 2,
    });

    strat.setData(equity.dates.map((time, i) => ({ time, value: equity.strategy[i] })));
    bench.setData(equity.dates.map((time, i) => ({ time, value: equity.benchmark[i] })));
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [equity]);

  return <div ref={ref} className="h-[300px] w-full" />;
}
