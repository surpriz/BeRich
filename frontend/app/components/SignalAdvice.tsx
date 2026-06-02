"use client";

import type { Signal, SignalConfig } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";

// Plain-language "why this result" advice for one signal, built from the per-side win
// probabilities and the decision thresholds. Explains LONG / SHORT / NEUTRAL and, when
// neutral, how close each side is to firing — so a promoted-but-neutral asset is understood,
// not mistaken for "broken". Fragments are translated; numbers are composed here (t() has no
// interpolation).
export function SignalAdvice({ signal, cfg }: { signal: Signal; cfg: SignalConfig | undefined }) {
  const { t } = useI18n();
  if (!cfg) return null;

  const pL = signal.proba_long ?? null;
  const pS = signal.proba_short ?? null;
  const buy = cfg.buy_threshold;
  const short = cfg.short_threshold;
  const pct = (x: number | null) => (x == null ? "—" : `${(x * 100).toFixed(0)}%`);

  let headline: string;
  let body: string;

  if (signal.signal === "LONG" || signal.signal === "BUY") {
    headline = t("advice.long.head");
    body = `${t("advice.long.body")} ${t("advice.thresholdNote")} ${pct(pL)} ≥ ${pct(buy)}.`;
  } else if (signal.signal === "SHORT") {
    headline = t("advice.short.head");
    body = `${t("advice.short.body")} ${t("advice.thresholdNote")} ${pct(pS)} ≥ ${pct(short)}.`;
  } else {
    // NEUTRAL — explain why nothing fired, and which side is closest to the threshold.
    headline = t("advice.neutral.head");
    const longGap = pL == null ? Infinity : buy - pL;
    const shortGap = pS == null || !cfg.enable_short ? Infinity : short - pS;
    const closeSide =
      longGap === Infinity && shortGap === Infinity
        ? null
        : longGap <= shortGap
          ? "long"
          : "short";
    const probLine = `${t("advice.probUp")} ${pct(pL)} · ${t("advice.probDown")} ${pct(pS)} (${t("advice.threshold")} ${pct(buy)}).`;
    let closeLine = "";
    if (closeSide === "long" && longGap <= 0.05) closeLine = " " + t("advice.closeLong");
    else if (closeSide === "short" && shortGap <= 0.05) closeLine = " " + t("advice.closeShort");
    body = `${t("advice.neutral.body")} ${probLine}${closeLine}`;
  }

  const promotedNote = signal.promoted ? t("advice.promotedNote") : t("advice.advisoryNote");

  return (
    <div className="rounded-md border border-[var(--color-line)] bg-white/[0.02] px-4 py-3 text-sm">
      <div className="font-medium text-[var(--color-fg)]">{headline}</div>
      <p className="mt-1 text-[var(--color-muted)]">{body}</p>
      <p className="mt-1.5 text-xs text-[var(--color-faint)]">{promotedNote}</p>
    </div>
  );
}
