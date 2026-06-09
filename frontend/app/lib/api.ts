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
  // Trust tier of the acted side: "promoted" (committed book), "observe" (paper-only shadow), or
  // "advisory" (inspection only). Mirrors promoted; distinguishes observe from advisory.
  tier?: string | null;
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
  // Which exit-strategy book the trade belongs to ("fixed" | "trailing" | "trailing_tp").
  exit_strategy?: string | null;
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
  capital?: number;
  buy_threshold: number;
  short_threshold: number;
  enable_short: boolean;
  horizon_days: number;
  take_profit_atr: number;
  stop_loss_atr: number;
  max_ticker_exposure_pct?: number;
  max_book_exposure_pct?: number;
  max_class_exposure_pct?: number;
  drawdown_derisk_threshold?: number;
  drawdown_halt_threshold?: number;
  max_open_positions?: number;
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

// The morning copy-trading action list — built from EXECUTIONS (facts), never forecasts.
export type Replication = {
  as_of: string;
  capital_base: number;
  open: {
    ticker: string;
    direction: "long" | "short";
    exit_strategy?: string | null;
    entry: number;
    stop: number;
    target: number;
    size_shares: number;
    notional: number;
    date_open: string;
  }[];
  close: {
    ticker: string;
    direction: "long" | "short";
    exit_strategy?: string | null;
    status: string;
    exit_price: number | null;
    pnl_pct: number | null;
    date_close: string | null;
  }[];
  adjust: {
    ticker: string;
    direction: string;
    exit_strategy?: string | null;
    effective_stop: number;
    target: number;
  }[];
  closed_total: number;
};

// ----- Intraday V2 POC (1h crypto). Mirrors the daily shapes; equity benchmark is the pair (BTC),
// not SPY, and timestamps carry the hour. The daily types above are untouched.
export type IntradayPaperMetrics = {
  n_open: number;
  n_closed: number;
  win_rate: number;
  total_return_paper: number;
  total_return_bench: number;
  max_drawdown_paper: number;
  capital: number;
};

export type IntradayPaperEquity = {
  dates: string[];
  equity_paper: number[];
  equity_bench: Array<number | null>;
  metrics: IntradayPaperMetrics;
};

export type IntradayPlannedOrder = {
  ts_open: string;
  ticker: string;
  signal: "LONG" | "SHORT" | "BUY";
  direction: "long" | "short";
  entry: number;
  stop: number;
  target: number;
  size_shares: number;
  notional: number;
  exit_strategy?: string | null;
};

export type IntradayReplication = {
  as_of: string;
  capital_base: number;
  open: {
    ticker: string;
    direction: "long" | "short";
    exit_strategy?: string | null;
    entry: number;
    stop: number;
    target: number;
    size_shares: number;
    notional: number;
    ts_open: string;
  }[];
  close: {
    ticker: string;
    direction: "long" | "short";
    exit_strategy?: string | null;
    status: string;
    exit_price: number | null;
    pnl_pct: number | null;
    pnl_eur: number | null;
    ts_close: string | null;
  }[];
  adjust: {
    ticker: string;
    direction: string;
    exit_strategy?: string | null;
    effective_stop: number;
    target: number;
  }[];
  closed_total: number;
};

// One portfolio-coherent planned order for the Brief (post-caps; what the book would open today).
export type PlannedOrder = {
  date_open: string;
  ticker: string;
  signal: "LONG" | "SHORT" | "BUY";
  direction: "long" | "short";
  entry: number;
  stop: number;
  target: number;
  size_shares: number;
  notional: number;
  exit_strategy?: string | null;
};

// HPO sweep coverage at the (ticker × side × strategy) grain — feeds the /training & /ops bars.
export type HpoProgress = {
  total: number;
  hpo_done: number;
  deep_complete: number;
  deep_trials: number;
  pending: number;
  promoted: number;
  advisory: number;
};

// One-glance synthesis of the raw gauges: is the box under-/well-/over-utilized (or idle)?
export type OpsUtilization = {
  verdict: "idle" | "under" | "balanced" | "over";
  gpu_avg_pct: number | null;
  idle_gpus: number;
  n_gpus: number;
  cpu_ratio: number | null;
  reasons: string[];
};

export type OpsSnapshot = {
  gpus: OpsGpu[];
  system: OpsSystem;
  sweep: OpsSweep;
  scheduler: { unit: string; state: string; active_since: string | null };
  jobs: OpsJob[];
  hpo: HpoProgress & { recent: OpsHpoRecent[] };
  utilization: OpsUtilization;
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
  last_hpo_at?: string | null;
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
  last_hpo_at?: string | null;
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

// Build a query string from defined params only ("?a=1&b=2"); returns "" when none are set,
// so callers can append it unconditionally without producing a dangling "?".
function qs(params: Record<string, string | number | undefined>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

async function get<T>(path: string): Promise<T> {
  const key = process.env.NEXT_PUBLIC_API_KEY;
  const res = await fetch(`${API_BASE}${path}`, {
    headers: key ? { "X-API-Key": key } : undefined,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const key = process.env.NEXT_PUBLIC_API_KEY;
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(key ? { "X-API-Key": key } : {}) },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// Risk profile: a one-click posture (prudent / balanced / offensive) over the live sizing knobs.
export type RiskProfileInfo = {
  active: string;
  profiles: Record<string, Record<string, number>>;
};

export const api = {
  signals: () => get<Signal[]>("/signals"),
  signalHistory: (ticker: string) => get<Signal[]>(`/signals/${ticker}/history`),
  signalExplain: (ticker: string) => get<SignalExplain>(`/signals/${ticker}/explain`),
  prices: (ticker: string, days = 365) => get<PriceBar[]>(`/prices/${ticker}?days=${days}`),
  drift: () => get<DriftReport>("/drift"),
  backtest: (threshold = 0.5) => get<Backtest>(`/backtest?threshold=${threshold}`),
  // ``tier`` selects the committed-capital book ("promoted", default) or the observation shadow
  // book ("observe") — near-miss models tracked live without capital. The server defaults to
  // "promoted" when omitted, so existing callers keep their committed-book semantics.
  paperPositions: (strategy?: string, tier?: string) =>
    get<PaperPositions>(`/paper/positions${qs({ strategy, tier })}`),
  paperEquity: (strategy?: string, tier?: string) =>
    get<PaperEquity>(`/paper/equity${qs({ strategy, tier })}`),
  paperClosed: (limit = 25, strategy?: string, tier?: string) =>
    get<PaperClosedTrade[]>(`/paper/closed-trades${qs({ limit, strategy, tier })}`),
  paperCalibration: () => get<PaperCalibration>("/paper/calibration"),
  universes: () => get<Universes>("/universes"),
  signalConfig: () => get<SignalConfig>("/config"),
  training: () => get<TrainingStatus[]>("/training"),
  briefPlan: () => get<PlannedOrder[]>("/brief-plan"),
  replication: (tier?: string) => get<Replication>(`/replication${qs({ tier })}`),
  hpoProgress: () => get<HpoProgress>("/hpo-progress"),
  riskProfile: () => get<RiskProfileInfo>("/risk-profile"),
  setRiskProfile: (profile: string) => post<RiskProfileInfo>("/risk-profile", { profile }),
  trainingTicker: (ticker: string) =>
    get<TrainingStatus[]>(`/training/${encodeURIComponent(ticker)}`),
  ops: () => get<OpsSnapshot>("/ops"),
  longshortBasket: () => get<LongShortLeg[]>("/longshort/basket"),
  longshortEquity: () => get<LongShortEquity>("/longshort/equity"),
  health: () => get<Health>("/health"),
  // Intraday V2 POC (1h crypto) — a parallel surface; the daily endpoints above are unchanged.
  intradaySignals: () => get<Signal[]>("/intraday/signals"),
  intradayBriefPlan: () => get<IntradayPlannedOrder[]>("/intraday/brief-plan"),
  intradayReplication: () => get<IntradayReplication>("/intraday/replication"),
  intradayPaperPositions: (strategy?: string, tier?: string) =>
    get<PaperPositions>(`/intraday/paper/positions${qs({ strategy, tier })}`),
  intradayPaperEquity: (strategy?: string, tier?: string) =>
    get<IntradayPaperEquity>(`/intraday/paper/equity${qs({ strategy, tier })}`),
  intradayPaperClosed: (limit = 25, strategy?: string, tier?: string) =>
    get<PaperClosedTrade[]>(`/intraday/paper/closed-trades${qs({ limit, strategy, tier })}`),
};

export const PAPER_EXPORT_URL = `${API_BASE}/paper/export.csv`;
