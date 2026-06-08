"use client";

import Link from "next/link";
import { useTranslate } from "@/app/lib/i18n";
import { PageIntro } from "@/app/components/PageIntro";

const STRATS = [
  { key: "fixed", accent: "var(--color-neutral)" },
  { key: "trailing", accent: "var(--color-bull)" },
  { key: "trailingtp", accent: "#e0a83d" },
] as const;

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-t border-[var(--color-line)]/50 pt-2">
      <div className="text-[10px] uppercase tracking-widest text-[var(--color-faint)]">{label}</div>
      <div className="mt-0.5 text-sm text-[var(--color-muted)]">{value}</div>
    </div>
  );
}

export default function StrategiesPage() {
  const t = useTranslate();
  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <Link
        href="/brief"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
      >
        ← {t("ticker.back")}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{t("strat.title")}</h1>
      <div className="mt-4">
        <PageIntro page="strategies" />
      </div>
      <p className="mt-3 max-w-2xl text-base text-[var(--color-muted)]">{t("strat.intro")}</p>

      <section className="card mt-8 p-5">
        <h2 className="mb-2 font-display text-lg font-bold">{t("strat.whatTitle")}</h2>
        <p className="text-sm text-[var(--color-muted)]">{t("strat.whatBody")}</p>
      </section>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        {STRATS.map(({ key, accent }) => (
          <section key={key} className="card flex flex-col p-5" style={{ borderTop: `3px solid ${accent}` }}>
            <h2 className="font-display text-xl font-bold" style={{ color: accent }}>
              {t(`strat.${key}.name`)}
            </h2>
            <p className="mt-2 text-sm font-medium text-[var(--color-fg)]">{t(`strat.${key}.philo`)}</p>
            <div className="mt-4 flex flex-col gap-3">
              <Row label={t("strat.howLabel")} value={t(`strat.${key}.how`)} />
              <Row label={t("strat.exampleLabel")} value={t(`strat.${key}.example`)} />
              <Row label={t("strat.proLabel")} value={t(`strat.${key}.pro`)} />
              <Row label={t("strat.conLabel")} value={t(`strat.${key}.con`)} />
              <Row label={t("strat.bestLabel")} value={t(`strat.${key}.best`)} />
            </div>
          </section>
        ))}
      </div>

      <section className="card mt-8 p-5">
        <h2 className="mb-2 font-display text-lg font-bold">{t("strat.paramsTitle")}</h2>
        <p className="text-sm text-[var(--color-muted)]">{t("strat.paramsBody")}</p>
      </section>

      <section className="card mt-6 p-5">
        <h2 className="mb-2 font-display text-lg font-bold">{t("strat.serveTitle")}</h2>
        <p className="text-sm text-[var(--color-muted)]">{t("strat.serveBody")}</p>
      </section>
    </main>
  );
}
