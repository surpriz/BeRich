import type { PaperPosition } from "@/app/lib/api";

/**
 * A trade whose barrier was already hit in the cached data but not yet booked — the daily close
 * job runs only weekday 22:30, so between the breach and the run the live MTM at the current price
 * badly misstates the outcome. The badge carries the REAL, capped exit P&L (`pending_exit_pct`),
 * e.g. a SOL short shows MTM −12% but the badge says "Stop hit · exit ≈ −3.6%".
 *
 * Shared by /wallet (PaperPanel) and /brief (HoldCard) so both surfaces tell the same truth.
 */
export function PendingExitBadge({
  exit,
  pct,
  fr,
}: {
  exit: NonNullable<PaperPosition["pending_exit"]>;
  pct: number | null | undefined;
  fr: boolean;
}) {
  const isStop = exit === "closed_stop" || exit === "closed_trail";
  const isTarget = exit === "closed_target";
  const color = isStop ? "var(--color-bear)" : isTarget ? "var(--color-bull)" : "var(--color-muted)";
  const icon = isStop ? "🛑" : isTarget ? "🎯" : "⏱";
  const what = isStop
    ? fr
      ? "Stop touché"
      : "Stop hit"
    : isTarget
      ? fr
        ? "Objectif atteint"
        : "Target hit"
      : fr
        ? "Échéance"
        : "Time exit";
  const outcome = pct == null ? "" : ` · ${fr ? "sortie" : "exit"} ≈ ${(pct * 100).toFixed(1)}%`;
  const title = fr
    ? "Barrière déjà franchie dans les données ; la position sera clôturée au prochain run quotidien (22h30). La perte/le gain réel est celui-ci, pas le MTM affiché."
    : "Barrier already hit in the data; the position will close at the next daily run (22:30). The real outcome is this one, not the live MTM.";
  return (
    <span
      title={title}
      className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
      style={{ color, backgroundColor: `${color}1a` }}
    >
      {icon} {what}
      {outcome}
    </span>
  );
}
