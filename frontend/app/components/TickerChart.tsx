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
import { useI18n } from "@/app/lib/i18n";

const SMA_BLUE = "#2563eb";
const SMA_VIOLET = "#7c3aed";
const STOP_RED = "#e05252";
const TARGET_GREEN = "#10936b";
const ENTRY_YELLOW = "#b7791f";
const TREND_BLUE = "#4f46e5";
const TREND_BAND = "rgba(79,70,229,0.10)";

// LONG: legacy BUY also opens a long. SHORT is the only bearish call drawn on the chart.
function isLongSignal(sig: Signal["signal"]): boolean {
  return sig === "LONG" || sig === "BUY";
}

// Append `n` synthetic business-day timestamps after `last` (YYYY-MM-DD, skipping weekends).
function futureBusinessDays(last: string, n: number): string[] {
  const out: string[] = [];
  const d = new Date(`${last}T00:00:00Z`);
  while (out.length < n) {
    d.setUTCDate(d.getUTCDate() + 1);
    const dow = d.getUTCDay();
    if (dow === 0 || dow === 6) continue;
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
}

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

const TREND_HORIZON = 10;

// Forward trend overlay: project the last close over ~10 business days using the latest
// signal's quantiles. Center = ret_q50 (fall back to a flat line); band = ret_q10..ret_q90
// when present, else ±sigma_horizon around the projection. Draws nothing when there is no
// usable forward information. lightweight-charts needs strictly increasing time, so the
// overlay lives on its own line series over synthetic future business-day timestamps that
// start at the last real bar (shared point anchors the projection to the candles).
function addTrendOverlay(chart: IChartApi, bars: PriceBar[], latest: Signal | undefined) {
  if (!latest || bars.length === 0) return;
  const q50 = latest.ret_q50;
  const q10 = latest.ret_q10;
  const q90 = latest.ret_q90;
  const sigma = latest.sigma_horizon;
  const hasCenter = q50 != null;
  const hasBand = q10 != null && q90 != null;
  const hasSigma = sigma != null;
  if (!hasCenter && !hasBand && !hasSigma) return;

  const lastBar = bars[bars.length - 1];
  const lastClose = lastBar.close;
  const future = futureBusinessDays(lastBar.date, TREND_HORIZON);
  const times: string[] = [lastBar.date, ...future];

  // Returns are fractional; interpolate linearly from 0 at the anchor to the horizon return.
  const ramp = (ret: number) => times.map((time, i) => ({
    time: time as Time,
    value: lastClose * (1 + (ret * i) / (times.length - 1)),
  }));

  const centerRet = hasCenter ? (q50 as number) : 0;
  const center = chart.addSeries(LineSeries, {
    color: TREND_BLUE,
    lineWidth: 2,
    lineStyle: 2, // dashed
    priceLineVisible: false,
    lastValueVisible: false,
  });
  center.setData(ramp(centerRet));

  const upperRet = hasBand ? (q90 as number) : hasSigma ? centerRet + (sigma as number) : null;
  const lowerRet = hasBand ? (q10 as number) : hasSigma ? centerRet - (sigma as number) : null;
  if (upperRet != null && lowerRet != null) {
    const bandStyle = {
      color: TREND_BAND,
      lineWidth: 1 as const,
      lineStyle: 1, // dotted (faint band edges)
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    };
    chart.addSeries(LineSeries, bandStyle).setData(ramp(upperRet));
    chart.addSeries(LineSeries, bandStyle).setData(ramp(lowerRet));
  }
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
  const { t } = useI18n();
  const trendLabel = t("trendForecast");

  useEffect(() => {
    if (!mainRef.current || !subRef.current || bars.length === 0) return;

    const layout = {
      background: { color: "transparent" },
      textColor: "#8a94a4",
      fontFamily: "var(--font-jetbrains)",
      attributionLogo: false,
    };
    const grid = {
      vertLines: { color: "rgba(20,24,31,0.06)" },
      horzLines: { color: "rgba(20,24,31,0.06)" },
    };

    const chart: IChartApi = createChart(mainRef.current, {
      layout,
      grid,
      rightPriceScale: { borderColor: "#e6e8ef" },
      timeScale: { borderColor: "#e6e8ef", timeVisible: false },
      height: 380,
      autoSize: true,
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#10936b",
      downColor: "#e05252",
      borderUpColor: "#10936b",
      borderDownColor: "#e05252",
      wickUpColor: "#10936b",
      wickDownColor: "#e05252",
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
      .filter((s) => s.signal === "LONG" || s.signal === "BUY" || s.signal === "SHORT")
      .filter((s) => barDates.has(s.date))
      .map((s) => {
        const long = isLongSignal(s.signal);
        // LONG → green up-arrow below the bar; SHORT → red down-arrow above the bar.
        return {
          time: s.date as Time,
          position: long ? "belowBar" : "aboveBar",
          color: long ? TARGET_GREEN : STOP_RED,
          shape: long ? "arrowUp" : "arrowDown",
          text: s.signal,
        };
      });
    if (markers.length > 0) createSeriesMarkers(candles, markers);

    const latest = signals[0];
    // Draw entry / stop / target for the latest actionable call in either direction. Colors
    // stay semantic — green is always the profit target, red always the stop — so for a short
    // the green target line sits below entry and the red stop line above it.
    const actionable = latest && (isLongSignal(latest.signal) || latest.signal === "SHORT");
    if (latest && actionable) {
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

    addTrendOverlay(chart, bars, latest);

    chart.timeScale().fitContent();

    const sub: IChartApi = createChart(subRef.current, {
      layout,
      grid,
      rightPriceScale: { borderColor: "#e6e8ef" },
      timeScale: { borderColor: "#e6e8ef" },
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
        color: b.close >= b.open ? "rgba(16,147,107,0.5)" : "rgba(224,82,82,0.5)",
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
          <span style={{ color: TARGET_GREEN }}>▲</span> LONG
        </span>
        <span className="flex items-center gap-1">
          <span style={{ color: STOP_RED }}>▼</span> SHORT
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-0.5 w-4" style={{ background: TREND_BLUE }} /> {trendLabel}
        </span>
      </div>
      <div ref={subRef} className="h-[160px] w-full" />
    </div>
  );
}
