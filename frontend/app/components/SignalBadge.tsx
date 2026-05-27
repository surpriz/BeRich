import type { Signal } from "@/app/lib/api";

const STYLES: Record<Signal["signal"], string> = {
  BUY: "text-[var(--color-bull)] border-[var(--color-bull)]/40 bg-[var(--color-bull)]/10",
  SELL: "text-[var(--color-bear)] border-[var(--color-bear)]/40 bg-[var(--color-bear)]/10",
  NEUTRAL: "text-[var(--color-neutral)] border-[var(--color-neutral)]/30 bg-[var(--color-neutral)]/8",
};

export function SignalBadge({ signal }: { signal: Signal["signal"] }) {
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold tracking-wide ${STYLES[signal]}`}
    >
      {signal}
    </span>
  );
}
