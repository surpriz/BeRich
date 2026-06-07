"use client";

import { useLevel, type Level } from "@/app/lib/level";
import { useTranslate } from "@/app/lib/i18n";

const LEVELS: Level[] = ["simple", "standard", "expert"];

export function LevelSwitch() {
  const { level, setLevel } = useLevel();
  const t = useTranslate();
  return (
    <div
      className="flex items-center gap-0.5 rounded-md border border-[var(--color-line)] p-0.5 text-xs"
      role="group"
      aria-label={t("level.label")}
    >
      {LEVELS.map((l) => (
        <button
          key={l}
          onClick={() => setLevel(l)}
          aria-pressed={level === l}
          className={`rounded px-2.5 py-1 transition-colors ${
            level === l
              ? "bg-[var(--color-bull)]/10 text-[var(--color-bull)]"
              : "text-[var(--color-faint)] hover:text-[var(--color-ink)]"
          }`}
        >
          {t(`level.${l}`)}
        </button>
      ))}
    </div>
  );
}
