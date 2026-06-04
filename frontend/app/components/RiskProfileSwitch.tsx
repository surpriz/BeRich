"use client";

import { useEffect, useState } from "react";
import { api } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";

// Ordered low → high risk, so we can confirm before INCREASING exposure (the guardrail).
const PROFILES = ["prudent", "balanced", "offensive"] as const;
type Profile = (typeof PROFILES)[number];

const LABEL: Record<Profile, { fr: string; en: string }> = {
  prudent: { fr: "Prudent", en: "Cautious" },
  balanced: { fr: "Équilibré", en: "Balanced" },
  offensive: { fr: "Offensif", en: "Aggressive" },
};

const COLOR: Record<Profile, string> = {
  prudent: "var(--color-bull)",
  balanced: "#e0a83d",
  offensive: "var(--color-bear)",
};

export function RiskProfileSwitch() {
  const { locale } = useI18n();
  const [active, setActive] = useState<Profile | null>(null);
  const [busy, setBusy] = useState(false);
  const fr = locale === "fr";

  useEffect(() => {
    api
      .riskProfile()
      .then((r) => setActive((r.active as Profile) ?? "prudent"))
      .catch(() => {});
  }, []);

  async function choose(p: Profile) {
    if (busy || p === active) return;
    // Confirm only when moving to a HIGHER-risk posture — a deliberate-friction guardrail.
    if (active && PROFILES.indexOf(p) > PROFILES.indexOf(active)) {
      const msg = fr
        ? `Passer en profil « ${LABEL[p].fr} » augmente le risque par trade et desserre les garde-fous (drawdown, concentration). Continuer ?`
        : `Switching to "${LABEL[p].en}" raises per-trade risk and loosens the guardrails (drawdown, concentration). Continue?`;
      if (!window.confirm(msg)) return;
    }
    setBusy(true);
    try {
      const r = await api.setRiskProfile(p);
      setActive((r.active as Profile) ?? p);
    } catch {
      /* leave the previous selection on failure */
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="flex items-center gap-0.5 rounded-md border border-[var(--color-line)] p-0.5 text-xs"
      role="group"
      aria-label={fr ? "Profil de risque" : "Risk profile"}
      title={
        fr
          ? "Profil de risque : dose l'exposition (taille par trade, coupe-circuit, concentration). N'invente pas d'edge."
          : "Risk profile: scales exposure (per-trade size, kill-switch, concentration). Does not create edge."
      }
    >
      {PROFILES.map((p) => {
        const on = active === p;
        return (
          <button
            key={p}
            onClick={() => choose(p)}
            disabled={busy}
            aria-pressed={on}
            className="rounded px-2.5 py-1 transition-colors disabled:opacity-50"
            style={
              on
                ? { color: COLOR[p], backgroundColor: `${COLOR[p]}1a` }
                : { color: "var(--color-faint)" }
            }
          >
            {LABEL[p][fr ? "fr" : "en"]}
          </button>
        );
      })}
    </div>
  );
}
