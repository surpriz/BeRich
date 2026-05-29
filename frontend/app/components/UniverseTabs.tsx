"use client";

import type { AssetClass, Universes } from "@/app/lib/api";
import { useTranslate } from "@/app/lib/i18n";

const ORDER: AssetClass[] = ["us_stocks", "fr_stocks", "forex", "crypto", "commodities"];

export function UniverseTabs({
  universes,
  active,
  onChange,
}: {
  universes: Universes | undefined;
  active: AssetClass;
  onChange: (a: AssetClass) => void;
}) {
  const t = useTranslate();
  const visible = ORDER.filter((k) => !universes || universes[k].length > 0 || k === "us_stocks");

  return (
    <div className="mb-6 flex flex-wrap items-center gap-1 border-b border-[var(--color-line)] text-sm">
      {visible.map((k) => {
        const on = k === active;
        return (
          <button
            key={k}
            onClick={() => onChange(k)}
            className={`-mb-px border-b-2 px-3 py-2 ${
              on
                ? "border-[var(--color-bull)] text-[var(--color-bull)]"
                : "border-transparent text-[var(--color-muted)] hover:text-[var(--color-text)]"
            }`}
          >
            {t(`universes.${k}`)}
          </button>
        );
      })}
    </div>
  );
}
