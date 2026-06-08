"use client";

import { useEffect, useState } from "react";
import { useTranslate } from "@/app/lib/i18n";
import { useLevel } from "@/app/lib/level";

// Collapsible "what is this page?" banner. Plain-language explainer at the top of
// every main page — open by default for novices (Discovery / Investor), collapsed
// for Analyst. The open/closed choice is remembered per page in localStorage.
export function PageIntro({ page }: { page: string }) {
  const t = useTranslate();
  const { level } = useLevel();
  const storageKey = `berich.intro.${page}`;
  const defaultOpen = level !== "expert";
  const [open, setOpen] = useState(defaultOpen);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem(storageKey);
    if (stored === "0") setOpen(false);
    else if (stored === "1") setOpen(true);
    else setOpen(defaultOpen);
    setHydrated(true);
    // defaultOpen intentionally excluded: a stored choice wins over the level default.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    window.localStorage.setItem(storageKey, next ? "1" : "0");
  };

  const title = t(`intro.${page}.title`);
  const body = t(`intro.${page}.body`);

  return (
    <div
      className="mb-6 overflow-hidden rounded-2xl border border-[var(--color-accent)]/25 bg-[var(--color-accent)]/[0.05]"
      style={{ opacity: hydrated ? 1 : 0.999 }}
    >
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-3 text-left"
      >
        <span
          aria-hidden="true"
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[var(--color-accent)] text-[11px] font-bold text-white"
        >
          i
        </span>
        <span className="flex-1 font-display text-sm font-semibold text-[var(--color-ink)]">
          {open ? title : t("intro.whatis")}
        </span>
        <span className="text-xs text-[var(--color-muted)]">{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <p className="px-4 pb-4 pl-11 text-sm leading-relaxed text-[var(--color-muted)]">{body}</p>
      )}
    </div>
  );
}
