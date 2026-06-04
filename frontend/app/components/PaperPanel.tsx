import type { PaperClosedTrade, PaperEquity, PaperPosition, SignalConfig } from "@/app/lib/api";
import { PaperEquityChart } from "./PaperEquityChart";
import { BudgetBar, RangeBar } from "./bars";

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
    closed_trail: "trail stop",
  }[status] ?? status;
}

export function PaperPanel({
  equity,
  positions,
  closed,
  cfg,
}: {
  equity: PaperEquity;
  positions: PaperPosition[];
  closed: PaperClosedTrade[];
  cfg?: SignalConfig;
}) {
  const m = equity.metrics;
  const paperReturn = m.total_return_paper;
  const spyReturn = m.total_return_spy;
  // ``total_return_spy`` is NaN when SPY isn't in the cache or no trades exist yet.
  const spyHas = Number.isFinite(spyReturn);
  const delta = spyHas ? paperReturn - spyReturn : 0;
  // Open exposure measured at cost basis (entry × shares) vs. the book cap — mirrors the
  // server-side exposure-cap budget. positions are already the selected strategy's book.
  const usedExposure = positions.reduce((acc, p) => acc + p.entry * p.size_shares, 0);
  const bookBudget = m.capital * (cfg?.max_book_exposure_pct ?? 1);
  const exposurePct = bookBudget > 0 ? (usedExposure / bookBudget) * 100 : 0;

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

      <div className="card px-5 py-3">
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="uppercase tracking-widest text-[var(--color-muted)]">Book exposure</span>
          <span className="tabular text-[var(--color-faint)]">
            €{fmt(usedExposure)} / €{fmt(bookBudget)} · {exposurePct.toFixed(0)}%
          </span>
        </div>
        <BudgetBar used={usedExposure} budget={bookBudget} />
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

      <div className="flex flex-col gap-6">
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
        <div className="overflow-x-auto">
        <table className="w-full border-collapse whitespace-nowrap text-sm">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-left text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              <th className="px-5 py-2 font-medium">Ticker</th>
              <th className="px-3 py-2 font-medium">Opened</th>
              <th className="px-3 py-2 text-right font-medium">Entry</th>
              <th className="px-3 py-2 text-right font-medium">Current</th>
              <th className="px-3 py-2 text-right font-medium">Stop</th>
              <th className="px-3 py-2 text-center font-medium">Stop → Target</th>
              <th className="px-3 py-2 text-right font-medium">MTM</th>
              <th className="px-5 py-2 text-right font-medium">Days</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={`${p.date_open}-${p.ticker}`} className="border-b border-[var(--color-line)]/50 last:border-0">
                <td className="px-5 py-2 font-display text-sm font-bold">
                  <span className="flex items-center gap-2">
                    {p.ticker}
                    {p.direction === "short" ? (
                      <span className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[var(--color-bear)] ring-1 ring-[var(--color-bear)]/40">
                        short
                      </span>
                    ) : (
                      <span className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[var(--color-bull)] ring-1 ring-[var(--color-bull)]/40">
                        long
                      </span>
                    )}
                    {p.exit_strategy && p.exit_strategy !== "fixed" ? (
                      <span className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-[var(--color-muted)] ring-1 ring-[var(--color-line)]">
                        trail
                      </span>
                    ) : null}
                  </span>
                </td>
                <td className="tabular px-3 py-2 text-xs text-[var(--color-muted)]">{p.date_open.slice(0, 10)}</td>
                <td className="tabular px-3 py-2 text-right">{fmt(p.entry)}</td>
                <td className="tabular px-3 py-2 text-right">{fmt(p.current_price)}</td>
                <td className="tabular px-3 py-2 text-right text-[var(--color-muted)]">
                  {fmt(p.trail_stop ?? p.stop)}
                </td>
                <td className="px-3 py-2">
                  {p.target > 0 ? (
                    <div className="flex justify-center">
                      <RangeBar
                        low={p.trail_stop ?? p.stop}
                        high={p.target}
                        value={p.current_price}
                        entry={p.entry}
                      />
                    </div>
                  ) : (
                    <div className="text-center text-[var(--color-faint)]">—</div>
                  )}
                </td>
                <td className={`tabular px-3 py-2 text-right ${pnlColor(p.mtm_pct)}`}>
                  {pct(p.mtm_pct)}
                </td>
                <td className="tabular px-5 py-2 text-right text-[var(--color-muted)]">{p.days_held}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
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
        <div className="overflow-x-auto">
        <table className="w-full border-collapse whitespace-nowrap text-sm">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-left text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              <th className="px-5 py-2 font-medium">Ticker</th>
              <th className="px-3 py-2 font-medium">Opened</th>
              <th className="px-3 py-2 font-medium">Closed</th>
              <th className="px-3 py-2 font-medium">Reason</th>
              <th className="px-3 py-2 text-right font-medium">Entry</th>
              <th className="px-3 py-2 text-right font-medium">Exit</th>
              <th className="px-5 py-2 text-right font-medium">P&L</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={`${t.date_open}-${t.ticker}`} className="border-b border-[var(--color-line)]/50 last:border-0">
                <td className="px-5 py-2 font-display text-sm font-bold">{t.ticker}</td>
                <td className="tabular px-3 py-2 text-xs text-[var(--color-muted)]">{t.date_open.slice(0, 10)}</td>
                <td className="tabular px-3 py-2 text-xs text-[var(--color-muted)]">{t.date_close}</td>
                <td className="px-3 py-2 text-xs text-[var(--color-muted)]">{statusLabel(t.status)}</td>
                <td className="tabular px-3 py-2 text-right">{fmt(t.entry)}</td>
                <td className="tabular px-3 py-2 text-right">{fmt(t.exit_price)}</td>
                <td className={`tabular px-5 py-2 text-right ${pnlColor(t.pnl_pct)}`}>{pct(t.pnl_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </div>
  );
}
