// Model-freshness signal shared by /training, the ticker drill-down and /ops. Tiers track the
// continuous sweep's ~2-day steady re-fit cycle: green within 2 days of the last HPO, amber up to
// 5 days, red beyond — so an asset whose model was optimized too long ago is visible at a glance.
const DAY_MS = 86_400_000;
export const FRESH_MAX_MS = 2 * DAY_MS;
export const AGING_MAX_MS = 5 * DAY_MS;

export type FreshnessTier = "fresh" | "aging" | "stale" | "none";

function tierFromAge(ageMs: number): FreshnessTier {
  if (ageMs < FRESH_MAX_MS) return "fresh";
  if (ageMs < AGING_MAX_MS) return "aging";
  return "stale";
}

// Tier of a model given the ISO timestamp of its last optimization (or null if never optimized).
export function freshnessTier(iso: string | null | undefined, now: number = Date.now()): FreshnessTier {
  if (!iso) return "none";
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return "none";
  return tierFromAge(Math.max(0, now - ms));
}

// Tier from a pre-computed age in seconds (the /ops payload already carries the age, not the ISO).
export function freshnessTierFromSeconds(seconds: number | null | undefined): FreshnessTier {
  if (seconds == null) return "none";
  return tierFromAge(Math.max(0, seconds) * 1000);
}

const TIER_COLOR: Record<FreshnessTier, string> = {
  fresh: "var(--color-bull)",
  aging: "var(--color-neutral)",
  stale: "var(--color-bear)",
  none: "var(--color-faint)",
};

// CSS color value for a tier (use inline `style={{ color }}` or wrap in a Tailwind arbitrary class).
export function freshnessColor(tier: FreshnessTier): string {
  return TIER_COLOR[tier];
}

// "<ago> 2j" relative string + an exact local timestamp for the title/tooltip.
export function fmtAgo(
  iso: string | null | undefined,
  ago: string,
  now: number = Date.now(),
): { rel: string; exact: string } {
  if (!iso) return { rel: "—", exact: "" };
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return { rel: "—", exact: "" };
  const s = Math.max(0, (now - ms) / 1000);
  let dur: string;
  if (s < 60) dur = `${Math.round(s)}s`;
  else if (s < 3600) dur = `${Math.floor(s / 60)}m`;
  else if (s < 86400) dur = `${Math.floor(s / 3600)}h`;
  else dur = `${Math.floor(s / 86400)}j`;
  return { rel: `${ago} ${dur}`, exact: new Date(ms).toLocaleString() };
}
