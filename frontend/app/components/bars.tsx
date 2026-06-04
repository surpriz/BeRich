// Small presentational bar primitives shared across the dashboard. All take already-computed
// numbers and render a thin track — no data fetching, no business logic.

const clamp = (x: number) => Math.max(0, Math.min(100, x));

// A probability bar with a tick at the decision threshold: fill is the probability, the tick
// marks where the signal would start firing. Coloured when the probability clears the threshold.
export function ProbaGauge({
  proba,
  threshold,
  color,
}: {
  proba: number;
  threshold: number;
  color: string;
}) {
  const fires = proba >= threshold;
  return (
    <div className="relative w-full" title={`seuil ${threshold.toFixed(2)}`}>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
        <div
          className="h-full rounded-full transition-[width]"
          style={{
            width: `${clamp(proba * 100)}%`,
            background: fires ? color : "var(--color-faint)",
          }}
        />
      </div>
      <div
        className="absolute -top-0.5 h-2.5 w-px bg-[var(--color-ink)]/70"
        style={{ left: `${clamp(threshold * 100)}%` }}
      />
    </div>
  );
}

// Where the current price sits between stop (left) and target (right). The track is a faint
// red→green gradient; ticks mark the entry and the live price.
export function RangeBar({
  low,
  high,
  value,
  entry,
}: {
  low: number;
  high: number;
  value: number;
  entry: number;
}) {
  const span = high - low;
  if (!Number.isFinite(span) || span === 0) {
    return <span className="text-[var(--color-faint)]">—</span>;
  }
  const at = (x: number) => clamp(((x - low) / span) * 100);
  return (
    <div
      className="relative h-1.5 w-24 rounded-full"
      style={{
        background:
          "linear-gradient(90deg, color-mix(in srgb, var(--color-bear) 28%, transparent), color-mix(in srgb, var(--color-bull) 28%, transparent))",
      }}
    >
      <div
        className="absolute -top-0.5 h-2.5 w-px bg-[var(--color-faint)]"
        style={{ left: `${at(entry)}%` }}
        title="entrée"
      />
      <div
        className="absolute -top-1 h-3.5 w-1 -translate-x-1/2 rounded-sm bg-[var(--color-ink)]"
        style={{ left: `${at(value)}%` }}
        title="prix"
      />
    </div>
  );
}

// Used vs. budget (e.g. open exposure vs. the book cap). Turns amber past 80%, red past 100%.
export function BudgetBar({ used, budget }: { used: number; budget: number }) {
  const pct = budget > 0 ? (used / budget) * 100 : 0;
  const color =
    pct > 100
      ? "var(--color-bear)"
      : pct >= 80
        ? "var(--color-neutral)"
        : "var(--color-bull)";
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
      <div
        className="h-full rounded-full transition-[width]"
        style={{ width: `${clamp(pct)}%`, background: color }}
      />
    </div>
  );
}
