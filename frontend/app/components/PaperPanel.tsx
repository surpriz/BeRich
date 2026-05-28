import type { PaperClosedTrade, PaperEquity, PaperPosition } from "@/app/lib/api";
import { PaperEquityChart } from "./PaperEquityChart";

const fmt = (n: number) => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = (n: number) => `${(n * 100).toFixed(2)}%`;

function pnlColor(n: number): string {
  if (n > 0) return "text-[var(--color-bull)]";
  if (n < 0) return "text-[var(--color-bear)]";
  return "text-[var(--color-muted)]";
}

function statusLabel(status: string): string {
  // Map the DuckDB enum to a compact human label.
  return {
    closed_target: "target hit",
    closed_stop: "stop hit",
    closed_time: "time exit",
  }[status] ?? status;
}

export function PaperPanel({
  equity,
  positions,
  closed,
}: {
  equity: PaperEquity;
  positions: PaperPosition[];
  closed: PaperClosedTrade[];
}) {
  const m = equity.metrics;
  const paperReturn = m.total_return_paper;
  const spyReturn = m.total_return_spy;
  // ``total_return_spy`` is NaN when SPY isn't in the cache or no trades exist yet.
  const spyHas = Number.isFinite(spyReturn);
  const delta = spyHas ? paperReturn - spyReturn : 0;

  return (
    <section className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <h2 className="font-display text-xl font-bold">Paper trading</h2>
        <p className="text-xs text-[var(--color-faint)]">
          Simulation only. Same exit rule as the backtest; same disclaimers.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-4">
        <Stat label="Paper return" value={pct(paperReturn)} valueClass={pnlColor(paperReturn)} />
        <Stat
          label="SPY buy & hold"
          value={spyHas ? pct(spyReturn) : "—"}
          valueClass={spyHas ? pnlColor(spyReturn) : ""}
        />
        <Stat
          label="Paper − SPY"
          value={spyHas ? `${delta >= 0 ? "+" : ""}${pct(delta)}` : "—"}
          valueClass={spyHas ? pnlColor(delta) : ""}
        />
        <Stat label="Win rate" value={pct(m.win_rate)} sublabel={`${m.n_closed} closed`} />
      </div>

      <div className="card p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-display text-sm font-bold uppercase tracking-widest text-[var(--color-muted)]">
            Equity vs SPY (same capital)
          </h3>
          <span className="text-xs text-[var(--color-faint)]">
            start €{fmt(m.capital)} · max DD {pct(m.max_drawdown_paper)}
          </span>
        </div>
        <PaperEquityChart equity={equity} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <PositionsTable positions={positions} />
        <ClosedTable trades={closed} />
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  sublabel,
  valueClass,
}: {
  label: string;
  value: string;
  sublabel?: string;
  valueClass?: string;
}) {
  return (
    <div className="card p-4">
      <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">{label}</div>
      <div className={`tabular mt-1 text-2xl font-bold ${valueClass ?? ""}`}>{value}</div>
      {sublabel && <div className="text-xs text-[var(--color-faint)]">{sublabel}</div>}
    </div>
  );
}

function PositionsTable({ positions }: { positions: PaperPosition[] }) {
  return (
    <div className="card overflow-hidden">
      <div className="border-b border-[var(--color-line)] px-5 py-3 text-xs uppercase tracking-widest text-[var(--color-muted)]">
        Open positions ({positions.length})
      </div>
      {positions.length === 0 ? (
        <div className="px-5 py-6 text-sm text-[var(--color-faint)]">No open positions.</div>
      ) : (
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-left text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              <th className="px-5 py-2 font-medium">Ticker</th>
              <th className="px-3 py-2 text-right font-medium">Entry</th>
              <th className="px-3 py-2 text-right font-medium">Current</th>
              <th className="px-3 py-2 text-right font-medium">MTM</th>
              <th className="px-5 py-2 text-right font-medium">Days</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={`${p.date_open}-${p.ticker}`} className="border-b border-[var(--color-line)]/50 last:border-0">
                <td className="px-5 py-2 font-display text-sm font-bold">{p.ticker}</td>
                <td className="tabular px-3 py-2 text-right">{fmt(p.entry)}</td>
                <td className="tabular px-3 py-2 text-right">{fmt(p.current_price)}</td>
                <td className={`tabular px-3 py-2 text-right ${pnlColor(p.mtm_pct)}`}>
                  {pct(p.mtm_pct)}
                </td>
                <td className="tabular px-5 py-2 text-right text-[var(--color-muted)]">{p.days_held}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ClosedTable({ trades }: { trades: PaperClosedTrade[] }) {
  return (
    <div className="card overflow-hidden">
      <div className="border-b border-[var(--color-line)] px-5 py-3 text-xs uppercase tracking-widest text-[var(--color-muted)]">
        Recent closed trades ({trades.length})
      </div>
      {trades.length === 0 ? (
        <div className="px-5 py-6 text-sm text-[var(--color-faint)]">No closed trades yet.</div>
      ) : (
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-left text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              <th className="px-5 py-2 font-medium">Ticker</th>
              <th className="px-3 py-2 font-medium">Closed</th>
              <th className="px-3 py-2 font-medium">Reason</th>
              <th className="px-5 py-2 text-right font-medium">P&L</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={`${t.date_open}-${t.ticker}`} className="border-b border-[var(--color-line)]/50 last:border-0">
                <td className="px-5 py-2 font-display text-sm font-bold">{t.ticker}</td>
                <td className="tabular px-3 py-2 text-[var(--color-muted)]">{t.date_close}</td>
                <td className="px-3 py-2 text-xs text-[var(--color-muted)]">{statusLabel(t.status)}</td>
                <td className={`tabular px-5 py-2 text-right ${pnlColor(t.pnl_pct)}`}>{pct(t.pnl_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
