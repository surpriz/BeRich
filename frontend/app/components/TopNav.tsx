"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { LanguageSelector, useTranslate } from "@/app/lib/i18n";
import { LevelSwitch } from "./LevelSwitch";
import { StrategySwitch } from "./StrategySwitch";

const TABS = [
  { href: "/", key: "nav.signals" },
  { href: "/strategies", key: "nav.strategies" },
  { href: "/training", key: "nav.training" },
  { href: "/ops", key: "nav.ops" },
  { href: "/how-it-works", key: "nav.help" },
];

export function TopNav() {
  const t = useTranslate();
  const path = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string) =>
    href === "/" ? path === "/" || path.startsWith("/ticker") : path.startsWith(href);

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--color-line)] bg-[var(--color-void)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <Link href="/" className="font-display text-xl font-extrabold tracking-tight">
          Be<span className="text-[var(--color-bull)]">Rich</span>
        </Link>

        <nav className="hidden items-center gap-1 text-sm md:flex">
          {TABS.map((tab) => (
            <Link
              key={tab.href}
              href={tab.href}
              className={`-mb-px border-b-2 px-3 py-2 transition-colors ${
                isActive(tab.href)
                  ? "border-[var(--color-bull)] text-[var(--color-bull)]"
                  : "border-transparent text-[var(--color-muted)] hover:text-[var(--color-ink)]"
              }`}
            >
              {t(tab.key)}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-3 md:flex">
          <StrategySwitch />
          <LevelSwitch />
          <LanguageSelector />
        </div>

        <button
          className="text-2xl leading-none text-[var(--color-muted)] md:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-label={t("nav.menu")}
        >
          ≡
        </button>
      </div>

      {open && (
        <div className="border-t border-[var(--color-line)] px-6 py-3 md:hidden">
          <nav className="flex flex-col gap-1 text-sm">
            {TABS.map((tab) => (
              <Link
                key={tab.href}
                href={tab.href}
                onClick={() => setOpen(false)}
                className={`py-1 ${
                  isActive(tab.href) ? "text-[var(--color-bull)]" : "text-[var(--color-muted)]"
                }`}
              >
                {t(tab.key)}
              </Link>
            ))}
          </nav>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <StrategySwitch />
            <LevelSwitch />
            <LanguageSelector />
          </div>
        </div>
      )}
    </header>
  );
}
