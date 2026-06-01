import type { Backtest } from "@/app/lib/api";
import { EquityChart } from "./EquityChart";
import { Info } from "./Term";

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const num = (n: number) => n.toFixed(2);

function Stat({
  label,
  strat,
  bench,
  format,
}: {
  label: React.ReactNode;
  strat: number;
  bench: number;
  format: (n: number) => string;
}) {
  const better = strat >= bench;
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">{label}</span>
      <span className={`tabular text-lg font-semibold ${better ? "text-[var(--color-bull)]" : "text-[var(--color-ink)]"}`}>
        {format(strat)}
      </span>
      <span className="tabular text-xs text-[var(--color-muted)]">bench {format(bench)}</span>
    </div>
  );
}

export function BacktestPanel({ bt }: { bt: Backtest }) {
  const s = bt.strategy;
  const b = bt.benchmark;
  return (
    <div className="card flex flex-col gap-5 p-5">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-xl font-bold">
          Walk-forward backtest
          <Info id="walk_forward" />
        </h2>
        <span
          className={`rounded-md border px-2.5 py-1 text-xs font-semibold ${
            bt.beats_buy_hold
              ? "border-[var(--color-bull)]/40 bg-[var(--color-bull)]/10 text-[var(--color-bull)]"
              : "border-[var(--color-bear)]/40 bg-[var(--color-bear)]/10 text-[var(--color-bear)]"
          }`}
        >
          {bt.beats_buy_hold ? "BEATS buy & hold" : "below buy & hold"} · AUC {bt.auc.toFixed(3)}
        </span>
      </div>

      <EquityChart equity={bt.equity} />

      <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
        <Stat
          label={
            <>
              Sharpe
              <Info id="sharpe" />
            </>
          }
          strat={s.sharpe}
          bench={b.sharpe}
          format={num}
        />
        <Stat label="Total return" strat={s.total_return} bench={b.total_return} format={pct} />
        <Stat label="Max drawdown" strat={s.max_drawdown} bench={b.max_drawdown} format={pct} />
        <Stat label="CAGR" strat={s.cagr} bench={b.cagr} format={pct} />
        <Stat label="Ann. vol" strat={s.ann_vol} bench={b.ann_vol} format={pct} />
        <Stat label="Win rate" strat={s.win_rate} bench={b.win_rate} format={pct} />
      </div>
    </div>
  );
}
