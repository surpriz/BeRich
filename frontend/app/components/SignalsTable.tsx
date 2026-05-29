import Link from "next/link";
import type { Signal } from "@/app/lib/api";
import { SignalBadge } from "./SignalBadge";

const fmt = (n: number) => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function ProbaBar({ p }: { p: number }) {
  const pct = Math.round(p * 100);
  const color = p >= 0.55 ? "var(--color-bull)" : p <= 0.3 ? "var(--color-bear)" : "var(--color-neutral)";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-[var(--color-line)]">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="tabular text-xs text-[var(--color-muted)]">{p.toFixed(3)}</span>
    </div>
  );
}

export function SignalsTable({ signals }: { signals: Signal[] }) {
  if (signals.length === 0) {
    return (
      <div className="card p-6 text-sm text-[var(--color-faint)]">
        No signals for this universe yet — run <code className="tabular">berich signals run</code>.
      </div>
    );
  }
  return (
    <div className="card overflow-hidden">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-[var(--color-line)] text-left text-xs uppercase tracking-widest text-[var(--color-faint)]">
            <th className="px-5 py-3 font-medium">Ticker</th>
            <th className="px-3 py-3 font-medium">Signal</th>
            <th className="px-3 py-3 font-medium">P(win)</th>
            <th className="px-3 py-3 text-right font-medium">Entry</th>
            <th className="px-3 py-3 text-right font-medium">Stop</th>
            <th className="px-3 py-3 text-right font-medium">Target</th>
            <th className="px-5 py-3 text-right font-medium">Shares</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr
              key={s.ticker}
              className="rise group border-b border-[var(--color-line)]/50 transition-colors last:border-0 hover:bg-white/[0.03]"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <td className="px-5 py-3 font-display text-base font-bold">
                <Link
                  href={`/ticker/${encodeURIComponent(s.ticker)}`}
                  className="block group-hover:text-[var(--color-bull)]"
                >
                  {s.ticker}
                </Link>
              </td>
              <td className="px-3 py-3">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  <SignalBadge signal={s.signal} />
                </Link>
              </td>
              <td className="px-3 py-3">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  <ProbaBar p={s.proba} />
                </Link>
              </td>
              <td className="tabular px-3 py-3 text-right">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {fmt(s.entry)}
                </Link>
              </td>
              <td className="tabular px-3 py-3 text-right text-[var(--color-bear)]/80">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {fmt(s.stop_loss)}
                </Link>
              </td>
              <td className="tabular px-3 py-3 text-right text-[var(--color-bull)]/80">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {fmt(s.take_profit)}
                </Link>
              </td>
              <td className="tabular px-5 py-3 text-right font-medium">
                <Link href={`/ticker/${encodeURIComponent(s.ticker)}`} className="block">
                  {s.size_shares > 0 ? s.size_shares : <span className="text-[var(--color-faint)]">—</span>}
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
