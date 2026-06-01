"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

// Global detail level — one switch that drives information density across every
// page. Mirrors the I18nProvider pattern (SSR-safe localStorage hydration).
export type Level = "standard" | "expert";

const ORDER: Level[] = ["standard", "expert"];
const STORAGE_KEY = "berich.level";

type Ctx = { level: Level; setLevel: (l: Level) => void };

const LevelContext = createContext<Ctx | null>(null);

export function atLeast(level: Level, min: Level): boolean {
  return ORDER.indexOf(level) >= ORDER.indexOf(min);
}

export function LevelProvider({ children }: { children: React.ReactNode }) {
  const [level, setLevelState] = useState<Level>("standard");

  useEffect(() => {
    const stored = typeof window !== "undefined" ? window.localStorage.getItem(STORAGE_KEY) : null;
    // "discovery" was removed — migrate any persisted value to "standard".
    if (stored === "expert") setLevelState("expert");
    else if (stored === "standard" || stored === "discovery") setLevelState("standard");
  }, []);

  const setLevel = useCallback((l: Level) => {
    setLevelState(l);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEY, l);
  }, []);

  const value = useMemo(() => ({ level, setLevel }), [level, setLevel]);
  return <LevelContext.Provider value={value}>{children}</LevelContext.Provider>;
}

export function useLevel(): Ctx {
  const ctx = useContext(LevelContext);
  if (!ctx) throw new Error("useLevel must be used inside <LevelProvider>");
  return ctx;
}
