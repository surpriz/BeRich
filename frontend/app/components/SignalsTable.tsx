"use client";

import Link from "next/link";
import { useState } from "react";
import type { Signal, SignalConfig } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { useLevel } from "@/app/lib/level";
import { SignalBadge } from "./SignalBadge";
import { Show } from "./Show";
import { Info } from "./Term";

const DEFAULT_COST_BPS = 6;
const CLOSE_GAP = 0.05; // within 5 pts of threshold = "close to firing"

/** One-line plain-language reason for the row's signal (mirrors SignalAdvice, terse). */
function MiniAdvice({ s, cfg, t }: { s: Signal; cfg?: SignalConfig; t: (k: string) => string }) {
  if (!cfg) return null;
  const pctp = (x: number | null | undefined) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  // A non-validated model must never read as a buy/short instruction, however high P(win) is —
  // that contradiction ("Achat" next to "Non validé") was the #1 source of visitor confusion.
  if (s.signal === "LONG" || s.signal === "BUY") {
    if (!s.promoted)
      return <span className="text-[var(--color-faint)]">{t("advice.mini.longAdvisory")}</span>;
    return <span className="text-[var(--color-bull)]/80">{t("advice.mini.long")}</span>;
  }
  if (s.signal === "SHORT") {
    if (!s.promoted)
      return <span className="text-[var(--color-faint)]">{t("advice.mini.shortAdvisory")}</span>;
    return <span className="text-[var(--color-bear)]/80">{t("advice.mini.short")}</span>;
  }
  // NEUTRAL: show the two probas vs threshold + flag the side closest to firing.
  const longGap = s.proba_long == null ? Infinity : cfg.buy_threshold - s.proba_long;
  const shortGap =
    s.proba_short == null || !cfg.enable_short ? Infinity : cfg.short_threshold - s.proba_short;
  let tail = "";
  if (longGap <= CLOSE_GAP && longGap <= shortGap) tail = " · " + t("advice.mini.closeLong");
  else if (shortGap <= CLOSE_GAP) tail = " · " + t("advice.mini.closeShort");
  return (
    <span className="text-[var(--color-faint)]">
      {t("advice.mini.neutral")} ({t("advice.probUp")} {pctp(s.proba_long)} · {t("advice.probDown")}{" "}
      {pctp(s.proba_short)})
      {tail && <span className="text-[var(--color-neutral)]">{tail}</span>}
    </span>
  );
}

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
  trailing: "trail",
  trailing_tp: "trail+TP",
};

export function SignalsTable({ signals, cfg }: { signals: Signal[]; cfg?: SignalConfig }) {
  const { t } = useI18n();
  const { level } = useLevel();
  const plain = level === "simple"; // Discovery: show plain-language column headers
  const defaultCost = signals.find((s) => s.cost_bps_roundtrip != null)?.cost_bps_roundtrip;
  const [costBps, setCostBps] = useState<number>(defaultCost ?? DEFAULT_COST_BPS);
  // Validated assets first (stable sort) so the followable signals lead, and the exploratory
  // "not validated" ones sit visibly below rather than mixed in.
  const ordered = [...signals].sort((a, b) => Number(b.promoted) - Number(a.promoted));
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
              {plain ? t("plain.pwin") : "P(win)"}
              <Info id="pwin" />
            </th>
            <th className="px-3 py-3 text-right font-medium">
              {plain ? t("plain.entry") : "Entry"}
              <Info id="entry" />
            </th>
            <th className="px-3 py-3 text-right font-medium">
              {plain ? t("plain.sl") : "Stop"}
              <Info id="sl" />
            </th>
            <th className="px-3 py-3 text-right font-medium">
              {plain ? t("plain.tp") : "Target"}
              <Info id="tp" />
            </th>
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
          {ordered.map((s, i) => (
            <tr
              key={s.ticker}
              className="rise group border-b border-[var(--color-line)]/50 transition-colors last:border-0 hover:bg-[var(--color-surface-2)]"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <td className="px-5 py-3 font-display text-base font-bold">
                <Link
                  href={`/ticker/${encodeURIComponent(s.ticker)}`}
                  className="block group-hover:text-[var(--color-accent)]"
                >
                  {s.ticker}
                </Link>
              </td>
              <td className="px-3 py-3">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  <SignalBadge signal={s.signal} />
                  <span
                    title={s.promoted ? t("advice.promotedNote") : t("advice.advisoryNote")}
                    className={`mt-1 block cursor-help text-[10px] uppercase tracking-wider ${
                      s.promoted ? "text-[var(--color-bull)]" : "text-[var(--color-faint)]"
                    }`}
                  >
                    {s.promoted ? t("signal.validated") : t("signal.advisory")} ⓘ
                  </span>
                  <span className="mt-1 block max-w-[16rem] text-[10px] leading-tight">
                    <MiniAdvice s={s} cfg={cfg} t={t} />
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
