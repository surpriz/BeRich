"use client";

import { useEffect, useState } from "react";
import { useTranslate } from "@/app/lib/i18n";

// First-run guided intro — 4 short steps that teach the vocabulary and where to look.
// Shown once (localStorage "berich.onboarded"), and replayable from Help via the
// "berich:onboarding-replay" window event.
const STORAGE_KEY = "berich.onboarded";
const STEPS = ["s1", "s2", "s3", "s4"] as const;

export function Onboarding() {
  const t = useTranslate();
  const [open, setOpen] = useState(false);
  const [i, setI] = useState(0);

  useEffect(() => {
    if (window.localStorage.getItem(STORAGE_KEY) !== "1") setOpen(true);
    const replay = () => {
      setI(0);
      setOpen(true);
    };
    window.addEventListener("berich:onboarding-replay", replay);
    return () => window.removeEventListener("berich:onboarding-replay", replay);
  }, []);

  const close = () => {
    window.localStorage.setItem(STORAGE_KEY, "1");
    setOpen(false);
  };

  if (!open) return null;
  const step = STEPS[i];
  const last = i === STEPS.length - 1;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[var(--color-ink)]/40 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      onClick={close}
    >
      <div
        className="card w-full max-w-md p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
            {i + 1} / {STEPS.length}
          </span>
          <button
            type="button"
            onClick={close}
            className="text-sm text-[var(--color-muted)] hover:text-[var(--color-ink)]"
          >
            {t("onboarding.skip")} ✕
          </button>
        </div>

        <h2 className="mt-3 font-display text-2xl font-extrabold tracking-tight">
          {t(`onboarding.${step}.title`)}
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-muted)]">
          {t(`onboarding.${step}.body`)}
        </p>

        <div className="mt-5 flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((s, k) => (
              <span
                key={s}
                className={`h-1.5 w-6 rounded-full transition-colors ${
                  k === i ? "bg-[var(--color-accent)]" : "bg-[var(--color-line)]"
                }`}
              />
            ))}
          </div>
          <div className="flex gap-2">
            {i > 0 && (
              <button
                type="button"
                onClick={() => setI((v) => v - 1)}
                className="rounded-full border border-[var(--color-line)] px-4 py-1.5 text-sm text-[var(--color-muted)] hover:text-[var(--color-ink)]"
              >
                {t("onboarding.back")}
              </button>
            )}
            <button
              type="button"
              onClick={() => (last ? close() : setI((v) => v + 1))}
              className="rounded-full bg-[var(--color-accent)] px-5 py-1.5 text-sm font-semibold text-white"
            >
              {last ? t("onboarding.done") : t("onboarding.next")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
