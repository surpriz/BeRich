"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Health } from "@/app/lib/api";
import { LanguageSelector, useTranslate } from "@/app/lib/i18n";

const STALE_DAYS = 2;

function isStale(iso: string | null): boolean {
  if (!iso) return true;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return true;
  const days = (Date.now() - d.getTime()) / 86_400_000;
  return days > STALE_DAYS;
}

export function HealthFooter() {
  const [h, setH] = useState<Health | null>(null);
  const t = useTranslate();

  useEffect(() => {
    let alive = true;
    api
      .health()
      .then((data) => alive && setH(data))
      .catch(() => alive && setH(null));
    return () => {
      alive = false;
    };
  }, []);

  const last = h?.ohlcv_last_refresh ?? null;
  const stale = isStale(last);

  return (
    <footer className="mt-12 border-t border-[var(--color-line)] py-4 text-xs text-[var(--color-faint)]">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6">
        <div className="flex items-center gap-3">
          <span>
            {t("footer.lastUpdate")}: <span className="tabular">{last ?? t("footer.never")}</span>
          </span>
          {stale && (
            <span className="rounded-md border border-[var(--color-neutral)]/40 bg-[var(--color-neutral)]/[0.08] px-2 py-0.5 text-[var(--color-neutral)]">
              {t("footer.stale")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <Link href="/ops" className="hover:text-[var(--color-bull)]">
            {t("footer.ops")}
          </Link>
          <Link href="/training" className="hover:text-[var(--color-bull)]">
            {t("footer.training")}
          </Link>
          <Link href="/under-the-hood" className="hover:text-[var(--color-bull)]">
            {t("footer.underHood")}
          </Link>
          <Link href="/how-it-works" className="hover:text-[var(--color-bull)]">
            {t("footer.howItWorks")}
          </Link>
          <LanguageSelector />
        </div>
      </div>
    </footer>
  );
}
