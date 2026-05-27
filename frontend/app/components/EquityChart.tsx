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
        vertLines: { color: "rgba(35,44,55,0.4)" },
        horzLines: { color: "rgba(35,44,55,0.4)" },
      },
      rightPriceScale: { borderColor: "#232c37" },
      timeScale: { borderColor: "#232c37" },
      height: 300,
      autoSize: true,
    });

    const strat = chart.addSeries(AreaSeries, {
      lineColor: "#b6f24e",
      topColor: "rgba(182,242,78,0.28)",
      bottomColor: "rgba(182,242,78,0.01)",
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
