// Typed client for the BeRich FastAPI backend.
// Base URL is configurable; defaults to the local serve address.

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type Signal = {
  date: string;
  ticker: string;
  signal: "BUY" | "SELL" | "NEUTRAL";
  proba: number;
  entry: number;
  stop_loss: number;
  take_profit: number;
  size_shares: number;
  notional: number;
};

export type FeatureDrift = {
  feature: string;
  psi: number;
  ks_pvalue: number;
  drifted: boolean;
};

export type DriftReport = {
  n_drifted: number;
  n_features: number;
  should_retrain: boolean;
  features: FeatureDrift[];
};

export type Metrics = {
  total_return: number;
  cagr: number;
  ann_vol: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  n_trades: number;
};

export type Backtest = {
  auc: number;
  strategy: Metrics;
  benchmark: Metrics;
  beats_buy_hold: boolean;
  equity: { dates: string[]; strategy: number[]; benchmark: number[] };
};

export type PaperPosition = {
  date_open: string;
  ticker: string;
  entry: number;
  stop: number;
  target: number;
  size_shares: number;
  current_price: number;
  days_held: number;
  mtm_pct: number;
  mtm_eur: number;
};

export type PaperPositions = {
  n: number;
  positions: PaperPosition[];
};

export type PaperMetrics = {
  n_open: number;
  n_closed: number;
  win_rate: number;
  total_return_paper: number;
  total_return_spy: number;
  max_drawdown_paper: number;
  capital: number;
};

export type PaperEquity = {
  dates: string[];
  equity_paper: number[];
  equity_spy: Array<number | null>;
  metrics: PaperMetrics;
};

export type PaperClosedTrade = {
  date_open: string;
  date_close: string;
  ticker: string;
  signal: string;
  entry: number;
  stop: number;
  target: number;
  size_shares: number;
  status: string;
  exit_price: number;
  pnl_pct: number;
  pnl_eur: number;
};

async function get<T>(path: string): Promise<T> {
  const key = process.env.NEXT_PUBLIC_API_KEY;
  const res = await fetch(`${API_BASE}${path}`, {
    headers: key ? { "X-API-Key": key } : undefined,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  signals: () => get<Signal[]>("/signals"),
  drift: () => get<DriftReport>("/drift"),
  backtest: (threshold = 0.5) => get<Backtest>(`/backtest?threshold=${threshold}`),
  paperPositions: () => get<PaperPositions>("/paper/positions"),
  paperEquity: () => get<PaperEquity>("/paper/equity"),
  paperClosed: (limit = 25) => get<PaperClosedTrade[]>(`/paper/closed-trades?limit=${limit}`),
};
