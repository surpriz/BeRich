"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import en from "./en.json";
import fr from "./fr.json";

export type Locale = "en" | "fr";

const DICTIONARIES: Record<Locale, Record<string, string>> = { en, fr };

type Ctx = {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string) => string;
};

const I18nContext = createContext<Ctx | null>(null);

const STORAGE_KEY = "berich.locale";

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("fr");

  useEffect(() => {
    const stored = typeof window !== "undefined" ? window.localStorage.getItem(STORAGE_KEY) : null;
    if (stored === "en" || stored === "fr") setLocaleState(stored);
  }, []);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEY, l);
  }, []);

  const t = useCallback(
    (key: string) => DICTIONARIES[locale][key] ?? DICTIONARIES.en[key] ?? key,
    [locale],
  );

  const value = useMemo(() => ({ locale, setLocale, t }), [locale, setLocale, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): Ctx {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used inside <I18nProvider>");
  return ctx;
}

export function useTranslate(): (key: string) => string {
  return useI18n().t;
}

export function LanguageSelector() {
  const { locale, setLocale, t } = useI18n();
  return (
    <div className="flex items-center gap-1 text-xs">
      <span className="sr-only">{t("lang.label")}</span>
      <button
        onClick={() => setLocale("en")}
        className={`rounded px-2 py-1 ${
          locale === "en"
            ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
            : "text-[var(--color-faint)] hover:text-[var(--color-text)]"
        }`}
      >
        EN
      </button>
      <span className="text-[var(--color-faint)]">·</span>
      <button
        onClick={() => setLocale("fr")}
        className={`rounded px-2 py-1 ${
          locale === "fr"
            ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
            : "text-[var(--color-faint)] hover:text-[var(--color-text)]"
        }`}
      >
        FR
      </button>
    </div>
  );
}
