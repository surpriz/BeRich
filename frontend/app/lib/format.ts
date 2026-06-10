// Adaptive price formatting shared across every page that shows an instrument price (entry, stop,
// target, current, exit). A fixed 2-decimal format collapsed low-priced assets — ADA-USD at ~0.16
// showed entry/current/stop all as "0.16"/"0.16"/"0.17", losing every meaningful digit. Decimals
// are chosen from the price magnitude so each asset keeps ~4 significant figures: forex pairs
// (EURUSD 1.0825), JPY crosses (157.23), large caps (250.43) and sub-cent crypto all stay legible.
export function fmtPrice(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  let decimals: number;
  if (abs === 0) {
    decimals = 2;
  } else if (abs >= 10) {
    decimals = 2; // 157.23 JPY cross, 250.43 stock, 65000 BTC
  } else if (abs >= 1) {
    decimals = 4; // 1.0825 EURUSD
  } else {
    // Sub-1: keep ~4 significant figures past the leading zeros (0.16 -> 0.1623, 0.0034 -> 0.003400).
    const leadingZeros = -Math.floor(Math.log10(abs)) - 1;
    decimals = Math.min(8, leadingZeros + 4);
  }
  return n.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
