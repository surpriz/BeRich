"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import type { PriceBar, Signal } from "@/app/lib/api";

const SMA_BLUE = "#6aa8ff";
const SMA_VIOLET = "#b386ff";
const STOP_RED = "#ef4444";
const TARGET_GREEN = "#b6f24e";
const ENTRY_YELLOW = "#facc15";

function sma(values: number[], window: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(i >= window - 1 ? sum / window : null);
  }
  return out;
}

function rsi(closes: number[], window = 14): (number | null)[] {
  const out: (number | null)[] = new Array(closes.length).fill(null);
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    if (i <= window) {
      avgGain += gain;
      avgLoss += loss;
      if (i === window) {
        avgGain /= window;
        avgLoss /= window;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        out[i] = 100 - 100 / (1 + rs);
      }
    } else {
      avgGain = (avgGain * (window - 1) + gain) / window;
      avgLoss = (avgLoss * (window - 1) + loss) / window;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      out[i] = 100 - 100 / (1 + rs);
    }
  }
  return out;
}

export function TickerChart({
  bars,
  signals,
}: {
  bars: PriceBar[];
  signals: Signal[];
}) {
  const mainRef = useRef<HTMLDivElement>(null);
  const subRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mainRef.current || !subRef.current || bars.length === 0) return;

    const layout = {
      background: { color: "transparent" },
      textColor: "#8a94a4",
      fontFamily: "var(--font-jetbrains)",
      attributionLogo: false,
    };
    const grid = {
      vertLines: { color: "rgba(35,44,55,0.4)" },
      horzLines: { color: "rgba(35,44,55,0.4)" },
    };

    const chart: IChartApi = createChart(mainRef.current, {
      layout,
      grid,
      rightPriceScale: { borderColor: "#232c37" },
      timeScale: { borderColor: "#232c37", timeVisible: false },
      height: 380,
      autoSize: true,
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#b6f24e",
      downColor: "#ef4444",
      borderUpColor: "#b6f24e",
      borderDownColor: "#ef4444",
      wickUpColor: "#b6f24e",
      wickDownColor: "#ef4444",
    });
    candles.setData(
      bars.map((b) => ({
        time: b.date as Time,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    );

    const closes = bars.map((b) => b.close);
    const sma20 = sma(closes, 20);
    const sma50 = sma(closes, 50);
    const sma20Series = chart.addSeries(LineSeries, {
      color: SMA_BLUE,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    sma20Series.setData(
      bars
        .map((b, i) => ({ time: b.date as Time, value: sma20[i] }))
        .filter((p): p is { time: Time; value: number } => p.value !== null),
    );
    const sma50Series = chart.addSeries(LineSeries, {
      color: SMA_VIOLET,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    sma50Series.setData(
      bars
        .map((b, i) => ({ time: b.date as Time, value: sma50[i] }))
        .filter((p): p is { time: Time; value: number } => p.value !== null),
    );

    const barDates = new Set(bars.map((b) => b.date));
    const markers: SeriesMarker<Time>[] = signals
      .filter((s) => s.signal === "BUY" || s.signal === "SELL")
      .filter((s) => barDates.has(s.date))
      .map((s) => ({
        time: s.date as Time,
        position: s.signal === "BUY" ? "belowBar" : "aboveBar",
        color: s.signal === "BUY" ? "#b6f24e" : "#ef4444",
        shape: s.signal === "BUY" ? "arrowUp" : "arrowDown",
        text: s.signal,
      }));
    if (markers.length > 0) createSeriesMarkers(candles, markers);

    const latest = signals[0];
    if (latest && latest.signal === "BUY") {
      candles.createPriceLine({
        price: latest.entry,
        color: ENTRY_YELLOW,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "entry",
      });
      candles.createPriceLine({
        price: latest.stop_loss,
        color: STOP_RED,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "stop",
      });
      candles.createPriceLine({
        price: latest.take_profit,
        color: TARGET_GREEN,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "target",
      });
    }

    chart.timeScale().fitContent();

    const sub: IChartApi = createChart(subRef.current, {
      layout,
      grid,
      rightPriceScale: { borderColor: "#232c37" },
      timeScale: { borderColor: "#232c37" },
      height: 160,
      autoSize: true,
    });
    const volSeries = sub.addSeries(HistogramSeries, {
      color: "#5a6472",
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    sub.priceScale("vol").applyOptions({
      scaleMargins: { top: 0.05, bottom: 0.55 },
    });
    volSeries.setData(
      bars.map((b) => ({
        time: b.date as Time,
        value: b.volume,
        color: b.close >= b.open ? "rgba(182,242,78,0.45)" : "rgba(239,68,68,0.45)",
      })),
    );
    const rsiSeries = sub.addSeries(LineSeries, {
      color: "#facc15",
      lineWidth: 1,
      priceScaleId: "rsi",
      priceLineVisible: false,
    });
    sub.priceScale("rsi").applyOptions({
      scaleMargins: { top: 0.55, bottom: 0.05 },
    });
    const rsiValues = rsi(closes);
    rsiSeries.setData(
      bars
        .map((b, i) => ({ time: b.date as Time, value: rsiValues[i] }))
        .filter((p): p is { time: Time; value: number } => p.value !== null),
    );
    sub.timeScale().fitContent();

    const syncMain = chart
      .timeScale()
      .subscribeVisibleLogicalRangeChange((range) => {
        if (range) sub.timeScale().setVisibleLogicalRange(range);
      });
    const syncSub = sub.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) chart.timeScale().setVisibleLogicalRange(range);
    });

    return () => {
      void syncMain;
      void syncSub;
      chart.remove();
      sub.remove();
    };
  }, [bars, signals]);

  if (bars.length === 0) {
    return (
      <div className="flex h-[380px] items-center justify-center text-sm text-[var(--color-faint)]">
        No price history for this ticker yet.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div ref={mainRef} className="h-[380px] w-full" />
      <div className="flex items-center gap-4 px-1 text-xs text-[var(--color-faint)]">
        <span className="flex items-center gap-1">
          <span className="inline-block h-0.5 w-4" style={{ background: SMA_BLUE }} /> SMA20
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-0.5 w-4" style={{ background: SMA_VIOLET }} /> SMA50
        </span>
        <span className="flex items-center gap-1">
          <span style={{ color: TARGET_GREEN }}>▲</span> BUY
        </span>
        <span className="flex items-center gap-1">
          <span style={{ color: STOP_RED }}>▼</span> SELL
        </span>
      </div>
      <div ref={subRef} className="h-[160px] w-full" />
    </div>
  );
}
