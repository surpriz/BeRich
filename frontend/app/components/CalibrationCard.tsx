import type { PaperCalibration } from "@/app/lib/api";

// Classic reliability diagram rendered as a compact table + two side-by-side bars
// per bucket (predicted vs realized). The bars are scaled to 100 % so the gap
// is visually obvious — no charting library needed for this kind of comparison.

const fmtPct = (n: number) => `${(n * 100).toFixed(1)}%`;

function Bar({ value, color }: { value: number; color: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
      <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export function CalibrationCard({ calibration }: { calibration: PaperCalibration }) {
  const verdict = calibration.is_well_calibrated ? "well calibrated" : "off";
  const verdictColor = calibration.is_well_calibrated
    ? "text-[var(--color-bull)]"
    : "text-[var(--color-bear)]";

  if (calibration.n_with_proba === 0) {
    return (
      <div className="card p-5">
        <h3 className="font-display text-sm font-bold uppercase tracking-widest text-[var(--color-muted)]">
          Calibration
        </h3>
        <p className="mt-2 text-sm text-[var(--color-faint)]">
          No closed paper trades yet — calibration table will populate after the first wins/losses.
        </p>
      </div>
    );
  }

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-display text-sm font-bold uppercase tracking-widest text-[var(--color-muted)]">
          Calibration
        </h3>
        <span className="text-xs text-[var(--color-faint)]">
          {calibration.n_with_proba} of {calibration.n_trades_total} closed trades —
          <span className={`ml-1 font-bold ${verdictColor}`}>{verdict}</span>
        </span>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-widest text-[var(--color-faint)]">
            <th className="py-1 pr-2">Bucket</th>
            <th className="py-1 pr-2 text-right">n</th>
            <th className="py-1 pr-2">Predicted</th>
            <th className="py-1">Realized</th>
          </tr>
        </thead>
        <tbody>
          {calibration.buckets.map((b) => (
            <tr key={b.bucket} className="border-t border-[var(--color-line)]/40">
              <td className="tabular py-2 pr-2 text-[var(--color-muted)]">{b.bucket}</td>
              <td className="tabular py-2 pr-2 text-right text-[var(--color-muted)]">{b.n_trades}</td>
              <td className="py-2 pr-2">
                {b.n_trades > 0 ? (
                  <div className="flex items-center gap-2">
                    <span className="tabular w-12 text-right text-[var(--color-muted)]">
                      {fmtPct(b.mean_predicted)}
                    </span>
                    <Bar value={b.mean_predicted} color="#5a6472" />
                  </div>
                ) : (
                  <span className="text-[var(--color-faint)]">—</span>
                )}
              </td>
              <td className="py-2">
                {b.n_trades > 0 ? (
                  <div className="flex items-center gap-2">
                    <span className="tabular w-12 text-right text-[var(--color-muted)]">
                      {fmtPct(b.win_rate)}
                    </span>
                    <Bar
                      value={b.win_rate}
                      color={b.win_rate >= b.mean_predicted ? "#b6f24e" : "#ff5b6c"}
                    />
                  </div>
                ) : (
                  <span className="text-[var(--color-faint)]">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-[10px] text-[var(--color-faint)]">
        Lime = realized &gt;= predicted. Red = the model was over-confident in this bucket.
      </p>
    </div>
  );
}
