// Typed client for the BeRich FastAPI backend.
// Base URL is configurable; defaults to the local serve address. The API now
// lives under the /api prefix everywhere (see src/berich/api/app.py and
// docs/DEPLOY.md), so both prod and dev defaults must point at /api.

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000/api";

export type Signal = {
  date: string;
  ticker: string;
  // LONG/SHORT/NEUTRAL are the current strings; BUY/SELL are kept for legacy rows on disk.
  signal: "LONG" | "SHORT" | "NEUTRAL" | "BUY" | "SELL";
  proba: number;
  entry: number;
  stop_loss: number;
  take_profit: number;
  size_shares: number;
  notional: number;
  // Direction of the call + per-side calibrated P(win) (nullable: a side may have no model).
  direction?: "long" | "short" | null;
  proba_long?: number | null;
  proba_short?: number | null;
  // Enriched advice fields (nullable for back-compat / older rows).
  proba_calibrated?: number | null;
  meta_proba?: number | null;
  acted?: boolean | null;
  ret_q10?: number | null;
  ret_q50?: number | null;
  ret_q90?: number | null;
  sigma_horizon?: number | null;
  sltp_method?: string | null;
  // Exit strategy of the served model: "fixed" (TP/SL), "trailing" (ratcheting stop, no TP) or
  // "trailing_tp" (TP cap + ratchet). For trailing, stop_loss is the INITIAL stop; trail_atr is
  // the ratchet distance in ATRs once armed (trail_activation_atr).
  exit_strategy?: string | null;
  trail_atr?: number | null;
  trail_activation_atr?: number | null;
  // True only when the acted side's per-asset model passed the guard. False/absent = advisory
  // (optimized but not validated — served from its own model, but no edge claim).
  promoted?: boolean | null;
  // Expected return (fraction of entry) from the triple-barrier expectancy: gross = the model's
  // raw edge, net = gross − the round-trip cost the signal was scored with (cost_bps_roundtrip).
  // The UI recomputes net for any user cost as gross − userBps/1e4. Null on NEUTRAL.
  exp_return_gross?: number | null;
  exp_return_net?: number | null;
  cost_bps_roundtrip?: number | null;
};

export type LongShortLeg = {
  date: string;
  ticker: string;
  side: "LONG" | "SHORT";
  weight: number;
  score: number;
};

export type LongShortEquity = {
  n_baskets: number;
  sharpe?: number;
  total_return?: number;
  max_drawdown?: number;
  avg_gross?: number;
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
  direction?: "long" | "short";
  entry: number;
  stop: number;
  target: number;
  size_shares: number;
  current_price: number;
  days_held: number;
  mtm_pct: number;
  mtm_eur: number;
  // Exit strategy of the trade; for a trailing trade ``trail_stop`` is the live ratcheting stop.
  exit_strategy?: string | null;
  trail_stop?: number | null;
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

export type CalibrationBucket = {
  bucket: string;
  low: number;
  high: number;
  midpoint: number;
  mean_predicted: number;
  win_rate: number;
  n_trades: number;
};

export type PaperCalibration = {
  n_trades_total: number;
  n_with_proba: number;
  is_well_calibrated: boolean;
  buckets: CalibrationBucket[];
};

export type PriceBar = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type SignalExplain = {
  ticker: string;
  date: string;
  proba: number;
  direction?: "long" | "short" | null;
  proba_long?: number | null;
  proba_short?: number | null;
  base_value: number;
  top_features: { feature: string; contribution: number }[];
  recent_news: {
    title: string;
    time_published: string;
    source: string;
    url: string;
    finbert_score: number | null;
  }[];
};

export type Universes = {
  us_stocks: string[];
  fr_stocks: string[];
  forex: string[];
  crypto: string[];
  commodities: string[];
};

export type AssetClass = keyof Universes;

export type SignalConfig = {
  buy_threshold: number;
  short_threshold: number;
  enable_short: boolean;
  horizon_days: number;
  take_profit_atr: number;
  stop_loss_atr: number;
};

export type TournamentCandidate = {
  ticker: string;
  side: string;
  model_name: string;
  oos_auc: number;
  strategy_sharpe: number;
  benchmark_sharpe: number;
  beats_guard: boolean;
  framework: string;
  n_features: number;
};

export type OpsGpu = {
  index: number;
  name: string;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  temp_c: number;
};

export type OpsJob = { id: string; next_run: string | null };

export type OpsHpoRecent = {
  ticker: string;
  side: string;
  strategy?: string;
  status: string;
  trained_at: string | null;
  hpo_trials: number;
};

export type OpsSystem = {
  cpu_pct?: number | null;
  load1?: number;
  load5?: number;
  load15?: number;
  n_cpus?: number;
  load_ratio?: number;
  mem_used_pct?: number | null;
  mem_used_gb?: number;
  mem_total_gb?: number;
  disk_used_pct?: number | null;
  disk_used_gb?: number;
  disk_total_gb?: number;
};

export type OpsSweep = {
  running: boolean;
  current: string | null;
  last_activity: string | null;
  idle_seconds: number | null;
  avg_seconds: number | null;
  gave_up: number;
};

export type OpsLogLine = {
  time: string;
  message: string;
  level: "info" | "warning" | "error";
  source?: string;
};

export type OpsSnapshot = {
  gpus: OpsGpu[];
  system: OpsSystem;
  sweep: OpsSweep;
  scheduler: { unit: string; state: string; active_since: string | null };
  jobs: OpsJob[];
  hpo: {
    total: number;
    hpo_done: number;
    pending: number;
    promoted: number;
    advisory: number;
    recent: OpsHpoRecent[];
  };
  alerts: OpsLogLine[];
  logs: OpsLogLine[];
};

// One exit strategy's verdict for a (ticker, side): fixed / trailing / trailing_tp.
export type StrategyStatus = {
  strategy: string;
  status: "promoted" | "advisory_only" | "never_trained";
  winner: string | null;
  framework: string | null;
  trained_at: string | null;
  metrics: Record<string, number>;
  candidates: TournamentCandidate[];
  horizon_days?: number | null;
};

export type TrainingStatus = {
  ticker: string;
  asset_class: string;
  side: "long" | "short";
  // Headline = the SERVED strategy's verdict (see served_strategy).
  status: "promoted" | "advisory_only" | "never_trained";
  winner: string | null;
  framework: string | null;
  trained_at: string | null;
  metrics: Record<string, number>;
  candidates: TournamentCandidate[];
  hpo_trials: number;
  horizon_days?: number | null;
  // Which exit strategy currently serves this (ticker, side), and the full per-strategy slate.
  served_strategy?: string | null;
  strategies?: StrategyStatus[];
};

export type Health = {
  status: string;
  ohlcv_last_refresh: string | null;
  news_last_refresh: string | null;
  signals_last_date: string | null;
  n_signals_today: number;
  n_open_positions: number;
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
  signalHistory: (ticker: string) => get<Signal[]>(`/signals/${ticker}/history`),
  signalExplain: (ticker: string) => get<SignalExplain>(`/signals/${ticker}/explain`),
  prices: (ticker: string, days = 365) => get<PriceBar[]>(`/prices/${ticker}?days=${days}`),
  drift: () => get<DriftReport>("/drift"),
  backtest: (threshold = 0.5) => get<Backtest>(`/backtest?threshold=${threshold}`),
  paperPositions: (strategy?: string) =>
    get<PaperPositions>(`/paper/positions${strategy ? `?strategy=${strategy}` : ""}`),
  paperEquity: (strategy?: string) =>
    get<PaperEquity>(`/paper/equity${strategy ? `?strategy=${strategy}` : ""}`),
  paperClosed: (limit = 25, strategy?: string) =>
    get<PaperClosedTrade[]>(
      `/paper/closed-trades?limit=${limit}${strategy ? `&strategy=${strategy}` : ""}`,
    ),
  paperCalibration: () => get<PaperCalibration>("/paper/calibration"),
  universes: () => get<Universes>("/universes"),
  signalConfig: () => get<SignalConfig>("/config"),
  training: () => get<TrainingStatus[]>("/training"),
  ops: () => get<OpsSnapshot>("/ops"),
  longshortBasket: () => get<LongShortLeg[]>("/longshort/basket"),
  longshortEquity: () => get<LongShortEquity>("/longshort/equity"),
  health: () => get<Health>("/health"),
};

export const PAPER_EXPORT_URL = `${API_BASE}/paper/export.csv`;
