"use client";

import Link from "next/link";
import { useState } from "react";
import type { Signal } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { SignalBadge } from "./SignalBadge";
import { Show } from "./Show";
import { Info } from "./Term";

const DEFAULT_COST_BPS = 6;

/** Expected return as a % string, net of a user-supplied round-trip cost (bps). */
function ExpReturn({ s, costBps }: { s: Signal; costBps: number }) {
  if (s.exp_return_gross == null) return <span className="text-[var(--color-faint)]">—</span>;
  const gross = s.exp_return_gross;
  const net = gross - costBps / 1e4;
  const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
  const tone = (x: number) => (x > 0 ? "text-[var(--color-bull)]" : "text-[var(--color-bear)]");
  return (
    <span className="tabular text-xs">
      <span className={tone(gross)}>{pct(gross)}</span>
      <span className="text-[var(--color-faint)]"> / </span>
      <span className={tone(net)}>{pct(net)}</span>
    </span>
  );
}

const fmt = (n: number) => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function SideProbas({ s }: { s: Signal }) {
  const { t } = useI18n();
  if (s.proba_long == null && s.proba_short == null) return null;
  return (
    <Show min="expert">
      <div className="mt-1 flex gap-2 text-[10px] text-[var(--color-faint)]">
        {s.proba_long != null && (
          <span title={t("probaLong")}>L {s.proba_long.toFixed(2)}</span>
        )}
        {s.proba_short != null && (
          <span title={t("probaShort")}>S {s.proba_short.toFixed(2)}</span>
        )}
      </div>
    </Show>
  );
}

function ProbaBar({ p, calibrated }: { p: number; calibrated?: number | null }) {
  const shown = calibrated ?? p;
  const pct = Math.round(shown * 100);
  const color = shown >= 0.55 ? "var(--color-bull)" : shown <= 0.3 ? "var(--color-bear)" : "var(--color-neutral)";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-[var(--color-line)]">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="tabular text-xs text-[var(--color-muted)]" title={calibrated != null ? `raw ${p.toFixed(3)} → calibrated` : undefined}>
        {shown.toFixed(3)}
        {calibrated != null && <span className="ml-1 text-[10px] text-[var(--color-faint)]">cal</span>}
      </span>
    </div>
  );
}

const SLTP_LABEL: Record<string, string> = {
  vol_scaled: "vol",
  quantile: "quantile",
  atr_fixed: "ATR",
};

export function SignalsTable({ signals }: { signals: Signal[] }) {
  const { t } = useI18n();
  const defaultCost = signals.find((s) => s.cost_bps_roundtrip != null)?.cost_bps_roundtrip;
  const [costBps, setCostBps] = useState<number>(defaultCost ?? DEFAULT_COST_BPS);
  if (signals.length === 0) {
    return (
      <div className="card p-6 text-sm text-[var(--color-faint)]">
        No signals for this universe yet — run <code className="tabular">berich signals run</code>.
      </div>
    );
  }
  return (
    <div className="card overflow-hidden">
      <Show min="expert">
        <div className="flex items-center gap-2 border-b border-[var(--color-line)] px-5 py-2 text-xs text-[var(--color-muted)]">
          <span>{t("signal.yourFees")}</span>
          <input
            type="number"
            min={0}
            step={1}
            value={costBps}
            onChange={(e) => setCostBps(Math.max(0, Number(e.target.value) || 0))}
            className="tabular w-16 rounded border border-[var(--color-line)] bg-transparent px-2 py-0.5 text-right"
          />
          <span className="text-[var(--color-faint)]">{t("signal.bpsRoundtrip")}</span>
        </div>
      </Show>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-[var(--color-line)] text-left text-xs uppercase tracking-widest text-[var(--color-faint)]">
            <th className="px-5 py-3 font-medium">Ticker</th>
            <th className="px-3 py-3 font-medium">Signal</th>
            <th className="px-3 py-3 font-medium">
              P(win)
              <Info id="pwin" />
            </th>
            <th className="px-3 py-3 text-right font-medium">Entry</th>
            <th className="px-3 py-3 text-right font-medium">Stop</th>
            <th className="px-3 py-3 text-right font-medium">Target</th>
            <th className="px-3 py-3 font-medium">
              SL/TP
              <Info id="atr" />
            </th>
            <th className="px-5 py-3 text-right font-medium">Shares</th>
            <th className="px-5 py-3 text-right font-medium">
              {t("signal.expReturn")}
              <Info id="expreturn" />
            </th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr
              key={s.ticker}
              className="rise group border-b border-[var(--color-line)]/50 transition-colors last:border-0 hover:bg-white/[0.03]"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <td className="px-5 py-3 font-display text-base font-bold">
                <Link
                  href={`/ticker/${encodeURIComponent(s.ticker)}`}
                  className="block group-hover:text-[var(--color-bull)]"
                >
                  {s.ticker}
                </Link>
              </td>
              <td className="px-3 py-3">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  <SignalBadge signal={s.signal} />
                  <span
                    className={`mt-1 block text-[10px] uppercase tracking-wider ${
                      s.promoted ? "text-[var(--color-bull)]" : "text-[var(--color-faint)]"
                    }`}
                  >
                    {s.promoted ? t("signal.validated") : t("signal.advisory")}
                  </span>
                </Link>
              </td>
              <td className="px-3 py-3">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  <ProbaBar p={s.proba} calibrated={s.proba_calibrated} />
                  <SideProbas s={s} />
                </Link>
              </td>
              <td className="tabular px-3 py-3 text-right">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {fmt(s.entry)}
                </Link>
              </td>
              <td className="tabular px-3 py-3 text-right text-[var(--color-bear)]/80">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {fmt(s.stop_loss)}
                </Link>
              </td>
              <td className="tabular px-3 py-3 text-right text-[var(--color-bull)]/80">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {fmt(s.take_profit)}
                </Link>
              </td>
              <td className="px-3 py-3">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  <span className="text-xs text-[var(--color-muted)]">
                    {SLTP_LABEL[s.sltp_method ?? "atr_fixed"] ?? "ATR"}
                  </span>
                  {s.acted === false && (
                    <Show min="expert">
                      <span className="ml-1 text-[10px] text-[var(--color-bear)]" title="meta-label veto">
                        veto
                      </span>
                    </Show>
                  )}
                </Link>
              </td>
              <td className="tabular px-5 py-3 text-right font-medium">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {s.size_shares > 0 ? s.size_shares : <span className="text-[var(--color-faint)]">—</span>}
                </Link>
              </td>
              <td className="px-5 py-3 text-right">
                <ExpReturn s={s} costBps={costBps} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
