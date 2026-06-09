"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { LanguageSelector, useTranslate } from "@/app/lib/i18n";
import { LevelSwitch } from "./LevelSwitch";
import { RiskProfileSwitch } from "./RiskProfileSwitch";
import { StrategySwitch } from "./StrategySwitch";

// Everyday actions stay on the main bar; the technical pages live under "Backstage".
const PRIMARY = [
  { href: "/brief", key: "nav.today" },
  { href: "/wallet", key: "nav.wallet" },
  { href: "/", key: "nav.opportunities" },
];

const BACKSTAGE = [
  { href: "/intraday", key: "nav.intraday" },
  { href: "/strategies", key: "nav.strategies" },
  { href: "/training", key: "nav.training" },
  { href: "/ops", key: "nav.ops" },
  { href: "/changelog", key: "nav.changelog" },
  { href: "/how-it-works", key: "nav.help" },
];

export function TopNav() {
  const t = useTranslate();
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const isActive = (href: string) => {
    if (href === "/") return path === "/" || path.startsWith("/ticker");
    if (href === "/brief") return path.startsWith("/brief") || path.startsWith("/copy");
    return path.startsWith(href);
  };
  const backstageActive = BACKSTAGE.some((b) => path.startsWith(b.href));

  // Close the Backstage dropdown on outside click or Escape.
  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  const tabClass = (active: boolean) =>
    `-mb-px border-b-2 px-3 py-2 transition-colors ${
      active
        ? "border-[var(--color-accent)] text-[var(--color-accent)]"
        : "border-transparent text-[var(--color-muted)] hover:text-[var(--color-ink)]"
    }`;

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--color-line)] bg-[var(--color-bg)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <Link href="/brief" className="font-display text-xl font-extrabold tracking-tight">
          Be<span className="text-[var(--color-accent)]">Rich</span>
        </Link>

        <nav className="hidden items-center gap-1 text-sm md:flex">
          {PRIMARY.map((tab) => (
            <Link key={tab.href} href={tab.href} className={tabClass(isActive(tab.href))}>
              {t(tab.key)}
            </Link>
          ))}

          <div className="relative" ref={menuRef}>
            <button
              type="button"
              onClick={() => setMenuOpen((o) => !o)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              className={tabClass(backstageActive)}
            >
              {t("nav.backstage")} <span className="text-xs">▾</span>
            </button>
            {menuOpen && (
              <div
                role="menu"
                className="card absolute right-0 top-[calc(100%+8px)] z-50 min-w-44 overflow-hidden p-1"
              >
                {BACKSTAGE.map((tab) => (
                  <Link
                    key={tab.href}
                    href={tab.href}
                    role="menuitem"
                    onClick={() => setMenuOpen(false)}
                    className={`block rounded-md px-3 py-2 text-sm transition-colors ${
                      path.startsWith(tab.href)
                        ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                        : "text-[var(--color-muted)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-ink)]"
                    }`}
                  >
                    {t(tab.key)}
                  </Link>
                ))}
              </div>
            )}
          </div>
        </nav>

        <div className="hidden items-center gap-3 md:flex">
          <RiskProfileSwitch />
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
            {PRIMARY.map((tab) => (
              <Link
                key={tab.href}
                href={tab.href}
                onClick={() => setOpen(false)}
                className={`py-1 ${
                  isActive(tab.href) ? "text-[var(--color-accent)]" : "text-[var(--color-muted)]"
                }`}
              >
                {t(tab.key)}
              </Link>
            ))}
            <div className="mt-2 mb-1 text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {t("nav.backstage")}
            </div>
            {BACKSTAGE.map((tab) => (
              <Link
                key={tab.href}
                href={tab.href}
                onClick={() => setOpen(false)}
                className={`py-1 pl-3 ${
                  path.startsWith(tab.href)
                    ? "text-[var(--color-accent)]"
                    : "text-[var(--color-muted)]"
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
