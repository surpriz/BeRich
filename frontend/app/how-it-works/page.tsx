"use client";

import Link from "next/link";
import { useTranslate } from "@/app/lib/i18n";

const SECTIONS = [
  "data",
  "features",
  "labels",
  "model",
  "signals",
  "honesty",
] as const;

function Pipeline() {
  const t = useTranslate();
  return (
    <div className="card my-8 overflow-x-auto p-6">
      <div className="flex min-w-max items-center gap-3 text-xs">
        {SECTIONS.slice(0, 5).map((s, i, arr) => (
          <div key={s} className="flex items-center gap-3">
            <div className="rounded-md border border-[var(--color-line)] bg-white/[0.02] px-3 py-2">
              <div className="font-display text-sm font-bold">{t(`how.section.${s}`)}</div>
            </div>
            {i < arr.length - 1 && <span className="text-[var(--color-faint)]">→</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HowItWorks() {
  const t = useTranslate();
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
      >
        ← {t("ticker.back")}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{t("how.title")}</h1>
      <p className="mt-3 text-base text-[var(--color-muted)]">{t("how.intro")}</p>

      <Pipeline />

      <div className="flex flex-col gap-6">
        {SECTIONS.map((s) => (
          <section key={s} className="card p-5">
            <h2 className="mb-2 font-display text-lg font-bold">{t(`how.section.${s}`)}</h2>
            <p className="text-sm text-[var(--color-muted)]">{t(`how.section.${s}Body`)}</p>
          </section>
        ))}
      </div>
    </main>
  );
}
