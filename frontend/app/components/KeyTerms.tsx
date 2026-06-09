"use client";

import { useState } from "react";
import { useTranslate } from "@/app/lib/i18n";

// Compact, collapsible "vocabulary in 30 seconds" legend. The four term-pairs that, once
// understood, dissolve most novice confusion: confidence vs validation, forecast vs executed,
// and what the no-money Observation tier is. Persistent (unlike the first-run Onboarding modal)
// so it's there whenever a visitor needs it. Collapsed by default to stay out of the way.
const TERMS = ["pwin", "validation", "forecast", "observe"] as const;

export function KeyTerms() {
  const t = useTranslate();
  const [open, setOpen] = useState(false);

  return (
    <div className="mb-6 overflow-hidden rounded-2xl border border-[var(--color-line)] bg-[var(--color-surface-2)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-3 text-left"
      >
        <span
          aria-hidden="true"
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[var(--color-muted)] text-[11px] font-bold text-white"
        >
          A
        </span>
        <span className="flex-1 font-display text-sm font-semibold text-[var(--color-ink)]">
          {t("keyterms.title")}
        </span>
        <span className="text-xs text-[var(--color-muted)]">{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <dl className="grid gap-3 px-4 pb-4 sm:grid-cols-2">
          {TERMS.map((k) => (
            <div key={k}>
              <dt className="text-sm font-semibold text-[var(--color-fg)]">
                {t(`keyterms.${k}.term`)}
              </dt>
              <dd className="mt-0.5 text-xs leading-relaxed text-[var(--color-muted)]">
                {t(`keyterms.${k}.def`)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
