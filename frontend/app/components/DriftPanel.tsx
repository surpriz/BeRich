import type { DriftReport } from "@/app/lib/api";

export function DriftPanel({ drift }: { drift: DriftReport }) {
  return (
    <div className="card flex flex-col gap-4 p-5">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-xl font-bold">Feature drift</h2>
        <span
          className={`rounded-md border px-2.5 py-1 text-xs font-semibold ${
            drift.should_retrain
              ? "border-[var(--color-bear)]/40 bg-[var(--color-bear)]/10 text-[var(--color-bear)]"
              : "border-[var(--color-bull)]/40 bg-[var(--color-bull)]/10 text-[var(--color-bull)]"
          }`}
        >
          {drift.n_drifted}/{drift.n_features} drifted · {drift.should_retrain ? "retrain" : "stable"}
        </span>
      </div>

      <ul className="flex flex-col gap-2">
        {drift.features.slice(0, 8).map((f) => {
          const intensity = Math.min(1, f.psi / 0.5);
          return (
            <li key={f.feature} className="flex items-center gap-3">
              <span className="tabular w-36 truncate text-xs text-[var(--color-muted)]">{f.feature}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-line)]">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.max(4, intensity * 100)}%`,
                    background: f.drifted ? "var(--color-bear)" : "var(--color-faint)",
                  }}
                />
              </div>
              <span className="tabular w-12 text-right text-xs text-[var(--color-faint)]">{f.psi.toFixed(2)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
