"use client";

import { useStrategy, STRATEGIES, type Strategy } from "@/app/lib/strategy";
import { useTranslate } from "@/app/lib/i18n";

const LABEL: Record<Strategy, string> = {
  fixed: "strategy.fixed",
  trailing: "strategy.trailing",
  trailing_tp: "strategy.trailing_tp",
};

export function StrategySwitch() {
  const { strategy, setStrategy } = useStrategy();
  const t = useTranslate();
  return (
    <div
      className="flex items-center gap-0.5 rounded-md border border-[var(--color-line)] p-0.5 text-xs"
      role="group"
      aria-label={t("strategy.label")}
      title={t("strategy.hint")}
    >
      {STRATEGIES.map((s) => (
        <button
          key={s}
          onClick={() => setStrategy(s)}
          aria-pressed={strategy === s}
          className={`rounded px-2.5 py-1 transition-colors ${
            strategy === s
              ? "bg-[var(--color-bull)]/10 text-[var(--color-bull)]"
              : "text-[var(--color-faint)] hover:text-[var(--color-ink)]"
          }`}
        >
          {t(LABEL[s])}
        </button>
      ))}
    </div>
  );
}
