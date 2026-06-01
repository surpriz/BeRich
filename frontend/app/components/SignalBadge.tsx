"use client";

import type { Signal } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";

const BULL = "text-[var(--color-bull)] border-[var(--color-bull)]/40 bg-[var(--color-bull)]/10";
const BEAR = "text-[var(--color-bear)] border-[var(--color-bear)]/40 bg-[var(--color-bear)]/10";
const NEUT = "text-[var(--color-neutral)] border-[var(--color-neutral)]/30 bg-[var(--color-neutral)]/8";

// LONG / legacy BUY → bull; SHORT → bear; NEUTRAL / legacy SELL → neutral.
const STYLES: Record<Signal["signal"], string> = {
  LONG: BULL,
  BUY: BULL,
  SHORT: BEAR,
  SELL: NEUT,
  NEUTRAL: NEUT,
};

const I18N_KEY: Record<Signal["signal"], string> = {
  LONG: "directionLong",
  BUY: "directionLong",
  SHORT: "directionShort",
  SELL: "directionNeutral",
  NEUTRAL: "directionNeutral",
};

export function SignalBadge({ signal }: { signal: Signal["signal"] }) {
  const { t } = useI18n();
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold tracking-wide ${STYLES[signal] ?? NEUT}`}
    >
      {t(I18N_KEY[signal] ?? "directionNeutral")}
    </span>
  );
}
