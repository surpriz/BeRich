"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

// Global exit-strategy selector — one switch that picks the investor's philosophy (fixed TP/SL
// vs trailing stop) and drives which signals + paper book the whole dashboard shows. Mirrors the
// LevelProvider pattern (SSR-safe localStorage hydration).
export type Strategy = "fixed" | "trailing" | "trailing_tp";

export const STRATEGIES: Strategy[] = ["fixed", "trailing", "trailing_tp"];
const STORAGE_KEY = "berich.strategy";

type Ctx = { strategy: Strategy; setStrategy: (s: Strategy) => void };

const StrategyContext = createContext<Ctx | null>(null);

export function StrategyProvider({ children }: { children: React.ReactNode }) {
  const [strategy, setStrategyState] = useState<Strategy>("fixed");

  useEffect(() => {
    const stored = typeof window !== "undefined" ? window.localStorage.getItem(STORAGE_KEY) : null;
    if (stored === "fixed" || stored === "trailing" || stored === "trailing_tp") {
      setStrategyState(stored);
    }
  }, []);

  const setStrategy = useCallback((s: Strategy) => {
    setStrategyState(s);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEY, s);
  }, []);

  const value = useMemo(() => ({ strategy, setStrategy }), [strategy, setStrategy]);
  return <StrategyContext.Provider value={value}>{children}</StrategyContext.Provider>;
}

export function useStrategy(): Ctx {
  const ctx = useContext(StrategyContext);
  if (!ctx) throw new Error("useStrategy must be used inside <StrategyProvider>");
  return ctx;
}
