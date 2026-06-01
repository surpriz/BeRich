import type { LongShortEquity, LongShortLeg } from "@/app/lib/api";
import { Info } from "./Term";

const fmtW = (n: number) => `${(n * 100).toFixed(2)}%`;

function Leg({ leg }: { leg: LongShortLeg }) {
  const isLong = leg.side === "LONG";
  const color = isLong ? "var(--color-bull)" : "var(--color-bear)";
  return (
    <tr className="border-b border-[var(--color-line)]/40 last:border-0">
      <td className="px-3 py-2 font-display font-bold">{leg.ticker}</td>
      <td className="px-3 py-2">
        <span className="text-xs font-semibold" style={{ color }}>
          {leg.side}
        </span>
      </td>
      <td className="tabular px-3 py-2 text-right">{fmtW(Math.abs(leg.weight))}</td>
      <td className="tabular px-3 py-2 text-right text-[var(--color-muted)]">{leg.score.toFixed(3)}</td>
    </tr>
  );
}

export function LongShortPanel({
  basket,
  equity,
}: {
  basket: LongShortLeg[];
  equity?: LongShortEquity;
}) {
  if (!basket || basket.length === 0) {
    return (
      <div className="card p-6 text-sm text-[var(--color-faint)]">
        No market-neutral basket yet — run <code className="tabular">berich longshort signals</code>.
      </div>
    );
  }
  const longs = basket.filter((l) => l.side === "LONG").sort((a, b) => b.weight - a.weight);
  const shorts = basket.filter((l) => l.side === "SHORT").sort((a, b) => a.weight - b.weight);
  const gross = basket.reduce((acc, l) => acc + Math.abs(l.weight), 0);

  return (
    <section className="card p-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
        <h3 className="font-display text-sm font-bold uppercase tracking-widest text-[var(--color-muted)]">
          Market-neutral long/short basket
          <Info id="market_neutral" />
        </h3>
        <div className="flex gap-4 text-xs text-[var(--color-faint)]">
          <span>gross {gross.toFixed(2)}</span>
          <span>{longs.length}L / {shorts.length}S</span>
          {equity && equity.n_baskets > 0 && equity.sharpe != null && (
            <span>paper Sharpe {equity.sharpe.toFixed(2)}</span>
          )}
        </div>
      </div>
      <p className="mb-4 text-xs text-[var(--color-faint)]">
        Dollar-neutral cross-sectional ranking (experimental — gated on a deflated-Sharpe
        <Info id="deflated_sharpe" /> significance test, not buy &amp; hold).
      </p>
      <div className="grid gap-6 md:grid-cols-2">
        {[
          { label: "Long", legs: longs },
          { label: "Short", legs: shorts },
        ].map(({ label, legs }) => (
          <div key={label}>
            <div className="mb-1 text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {label} leg
            </div>
            <table className="w-full border-collapse text-sm">
              <tbody>
                {legs.slice(0, 12).map((leg) => (
                  <Leg key={leg.ticker} leg={leg} />
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </section>
  );
}
