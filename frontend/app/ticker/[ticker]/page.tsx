"use client";

import Link from "next/link";
import { useEffect, useState, use } from "react";
import {
  api,
  type AssetClass,
  type PriceBar,
  type Signal,
  type SignalConfig,
  type SignalExplain,
  type Universes,
} from "@/app/lib/api";
import { SignalBadge } from "@/app/components/SignalBadge";
import { SignalAdvice } from "@/app/components/SignalAdvice";
import { TickerChart } from "@/app/components/TickerChart";
import { TrainingInfo } from "@/app/components/TrainingInfo";
import { Show } from "@/app/components/Show";
import { Info } from "@/app/components/Term";
import { useTranslate } from "@/app/lib/i18n";
import { useStrategy } from "@/app/lib/strategy";

const ASSET_CLASS_LABEL: Record<AssetClass, string> = {
  us_stocks: "US Stock",
  fr_stocks: "FR Stock",
  forex: "Forex",
  crypto: "Crypto",
  commodities: "Commodity",
};

function classify(ticker: string, u: Universes | undefined): AssetClass | "unknown" {
  if (!u) return "unknown";
  for (const k of Object.keys(u) as AssetClass[]) {
    if (u[k].includes(ticker)) return k;
  }
  return "unknown";
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "bull" | "bear" }) {
  const color = tone === "bull" ? "text-[var(--color-bull)]" : tone === "bear" ? "text-[var(--color-bear)]" : "";
  return (
    <div className="card px-4 py-3">
      <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">{label}</div>
      <div className={`tabular mt-1 text-lg ${color}`}>{value}</div>
    </div>
  );
}

export default function TickerPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: raw } = use(params);
  const ticker = raw.toUpperCase();
  const t = useTranslate();
  const { strategy } = useStrategy();
  const [bars, setBars] = useState<PriceBar[] | undefined>();
  const [history, setHistory] = useState<Signal[] | undefined>();
  const [explain, setExplain] = useState<SignalExplain | null | undefined>();
  const [universes, setUniverses] = useState<Universes | undefined>();
  const [cfg, setCfg] = useState<SignalConfig | undefined>();
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [b, h, u, c] = await Promise.all([
          api.prices(ticker).catch(() => [] as PriceBar[]),
          api.signalHistory(ticker).catch(() => [] as Signal[]),
          api.universes().catch(() => undefined),
          api.signalConfig().catch(() => undefined),
        ]);
        if (!alive) return;
        setBars(b);
        setHistory(h);
        setUniverses(u);
        setCfg(c);
        try {
          const e = await api.signalExplain(ticker);
          if (alive) setExplain(e);
        } catch {
          if (alive) setExplain(null);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "request failed");
      }
    })();
    return () => {
      alive = false;
    };
  }, [ticker]);

  // One signal row per (date, exit strategy); show only the toggle's selected strategy so the
  // history reads one row per date (not three) and matches the rest of the dashboard.
  const shown = history?.filter((s) => (s.exit_strategy ?? "fixed") === strategy);
  const latest = shown?.[0];
  const klass = classify(ticker, universes);

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
      >
        ← {t("ticker.back")}
      </Link>

      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-4xl font-extrabold tracking-tight">{ticker}</h1>
          <div className="mt-1 flex items-center gap-3 text-sm text-[var(--color-muted)]">
            <span className="rounded-md border border-[var(--color-line)] px-2 py-0.5 text-xs uppercase tracking-widest">
              {klass === "unknown" ? "—" : ASSET_CLASS_LABEL[klass]}
            </span>
            {latest && (
              <>
                <SignalBadge signal={latest.signal} />
                {latest.direction && (
                  <span className="text-xs uppercase tracking-widest">
                    {latest.direction === "short" ? t("directionShort") : t("directionLong")}
                  </span>
                )}
                <Show min="standard">
                  <span
                    className="tabular"
                    title={
                      latest.proba_calibrated != null
                        ? `raw ${latest.proba.toFixed(3)} → calibrated`
                        : undefined
                    }
                  >
                    P(win) {(latest.proba_calibrated ?? latest.proba).toFixed(3)}
                    {latest.proba_calibrated != null && (
                      <span className="ml-1 text-[10px] text-[var(--color-faint)]">cal</span>
                    )}
                    <Info id="pwin" />
                  </span>
                </Show>
                <Show min="expert">
                  {latest.proba_long != null && (
                    <span className="tabular text-[var(--color-bull)]/80">
                      {t("probaLong")} {latest.proba_long.toFixed(3)}
                    </span>
                  )}
                  {latest.proba_short != null && (
                    <span className="tabular text-[var(--color-bear)]/80">
                      {t("probaShort")} {latest.proba_short.toFixed(3)}
                    </span>
                  )}
                </Show>
              </>
            )}
          </div>
        </div>
        {latest && (
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
              {t("ticker.asOf")}
            </div>
            <div className="tabular text-lg">{latest.date.slice(0, 10)}</div>
          </div>
        )}
      </header>

      {latest && (
        <div className="mb-6">
          <SignalAdvice signal={latest} cfg={cfg} />
        </div>
      )}

      {latest && (latest.signal === "LONG" || latest.signal === "SHORT" || latest.signal === "BUY") && (
        <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label={latest.direction === "short" ? t("shortEntry") : t("col.entry")} value={latest.entry.toFixed(2)} />
          <Stat label={latest.direction === "short" ? t("shortStop") : t("col.stop")} value={latest.stop_loss.toFixed(2)} tone="bear" />
          <Stat label={latest.direction === "short" ? t("shortTarget") : t("col.target")} value={latest.take_profit.toFixed(2)} tone="bull" />
          <Stat label={t("size")} value={`${latest.size_shares}`} />
        </div>
      )}

      {latest && (latest.ret_q10 != null || latest.sigma_horizon != null) && (
        <Show min="expert">
          <div className="mb-6 flex flex-wrap gap-6 rounded-lg border border-[var(--color-line)] bg-white/[0.02] px-4 py-3 text-xs text-[var(--color-muted)]">
            {latest.ret_q10 != null && latest.ret_q90 != null && (
              <span>
                <span className="text-[var(--color-faint)]">
                  {t("ticker.quantiles")}
                  <Info id="quantile" />
                </span>{" "}
                <span className="tabular">
                  {(latest.ret_q10 * 100).toFixed(1)}%
                  {latest.ret_q50 != null && ` · ${(latest.ret_q50 * 100).toFixed(1)}%`} ·{" "}
                  {(latest.ret_q90 * 100).toFixed(1)}%
                </span>
              </span>
            )}
            {latest.sigma_horizon != null && (
              <span>
                <span className="text-[var(--color-faint)]">{t("ticker.sigma")}</span>{" "}
                <span className="tabular">{(latest.sigma_horizon * 100).toFixed(2)}%</span>
              </span>
            )}
          </div>
        </Show>
      )}

      {error && (
        <div className="card p-6 text-[var(--color-bear)]">
          {t("error.api")} ({error})
        </div>
      )}

      <section className="card mb-8 p-4">
        {bars ? (
          <TickerChart bars={bars} signals={shown ?? []} />
        ) : (
          <div className="h-[540px] animate-pulse" />
        )}
      </section>

      <TrainingInfo ticker={ticker} history={history} cfg={cfg} />

      <Show min="standard">
      <section className="card mb-8 p-5">
        <h2 className="mb-3 font-display text-lg font-bold">{t("ticker.whySignal")}</h2>
        {explain === undefined && (
          <div className="h-24 animate-pulse rounded bg-white/[0.03]" />
        )}
        {explain === null && (
          <p className="text-sm text-[var(--color-faint)]">{t("ticker.noExplain")}</p>
        )}
        {explain && (
          <div className="grid gap-6 lg:grid-cols-2">
            <div>
              <h3 className="mb-2 text-xs uppercase tracking-widest text-[var(--color-faint)]">
                {t("ticker.topFeatures")}
                <Info id="shap" />
              </h3>
              <ul className="flex flex-col gap-1.5">
                {explain.top_features.map((f) => {
                  const positive = f.contribution >= 0;
                  return (
                    <li
                      key={f.feature}
                      className="flex items-center justify-between gap-3 border-b border-[var(--color-line)]/40 py-1 last:border-0"
                    >
                      <span className="tabular text-xs">{f.feature}</span>
                      <span
                        className={`tabular text-xs ${
                          positive ? "text-[var(--color-bull)]" : "text-[var(--color-bear)]"
                        }`}
                      >
                        {positive ? "+" : ""}
                        {f.contribution.toFixed(3)}
                      </span>
                    </li>
                  );
                })}
              </ul>
              <p className="mt-3 text-xs text-[var(--color-faint)]">
                {t("ticker.baseValue")}: <span className="tabular">{explain.base_value.toFixed(3)}</span>
              </p>
            </div>
            <Show min="expert">
            <div>
              <h3 className="mb-2 text-xs uppercase tracking-widest text-[var(--color-faint)]">
                {t("ticker.recentNews")}
              </h3>
              {explain.recent_news.length === 0 && (
                <p className="text-sm text-[var(--color-faint)]">{t("ticker.noNews")}</p>
              )}
              <ul className="flex flex-col gap-2">
                {explain.recent_news.map((n) => (
                  <li key={n.url || n.title} className="border-b border-[var(--color-line)]/40 pb-2 last:border-0">
                    <a
                      href={n.url || "#"}
                      target="_blank"
                      rel="noreferrer"
                      className="block text-sm text-[var(--color-text)] hover:text-[var(--color-bull)]"
                    >
                      {n.title}
                    </a>
                    <div className="mt-0.5 flex items-center gap-2 text-[11px] text-[var(--color-faint)]">
                      <span>{n.source}</span>
                      <span>·</span>
                      <span className="tabular">{n.time_published.slice(0, 10)}</span>
                      {n.finbert_score != null && (
                        <>
                          <span>·</span>
                          <span
                            className={
                              n.finbert_score > 0.1
                                ? "text-[var(--color-bull)]"
                                : n.finbert_score < -0.1
                                  ? "text-[var(--color-bear)]"
                                  : "text-[var(--color-neutral)]"
                            }
                          >
                            FinBERT {n.finbert_score >= 0 ? "+" : ""}
                            {n.finbert_score.toFixed(2)}
                          </span>
                        </>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            </Show>
          </div>
        )}
      </section>
      </Show>

      <section className="card p-5">
        <h2 className="mb-3 font-display text-lg font-bold">{t("ticker.history")}</h2>
        {!shown && <div className="h-24 animate-pulse rounded bg-white/[0.03]" />}
        {shown && shown.length === 0 && (
          <p className="text-sm text-[var(--color-faint)]">{t("ticker.noHistory")}</p>
        )}
        {shown && shown.length > 0 && (
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-[var(--color-line)] text-left text-xs uppercase tracking-widest text-[var(--color-faint)]">
                <th className="px-2 py-2 font-medium">{t("col.date")}</th>
                <th className="px-2 py-2 font-medium">{t("col.signal")}</th>
                <th className="px-2 py-2 font-medium">{t("direction")}</th>
                <Show min="standard">
                  <th className="px-2 py-2 font-medium">
                    P(win)
                    <Info id="pwin" />
                  </th>
                </Show>
                <th className="px-2 py-2 text-right font-medium">{t("col.entry")}</th>
                <th className="px-2 py-2 text-right font-medium">{t("col.stop")}</th>
                <th className="px-2 py-2 text-right font-medium">{t("col.target")}</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s) => (
                <tr
                  key={`${s.date}-${s.exit_strategy ?? "fixed"}`}
                  className="border-b border-[var(--color-line)]/40 last:border-0"
                >
                  <td className="tabular px-2 py-2">{s.date.slice(0, 10)}</td>
                  <td className="px-2 py-2">
                    <SignalBadge signal={s.signal} />
                  </td>
                  <td className="px-2 py-2 text-xs text-[var(--color-muted)]">
                    {s.direction === "short"
                      ? t("directionShort")
                      : s.direction === "long"
                        ? t("directionLong")
                        : "—"}
                  </td>
                  <Show min="standard">
                    <td className="tabular px-2 py-2">{s.proba.toFixed(3)}</td>
                  </Show>
                  <td className="tabular px-2 py-2 text-right">{s.entry.toFixed(2)}</td>
                  <td className="tabular px-2 py-2 text-right text-[var(--color-bear)]/80">
                    {s.stop_loss.toFixed(2)}
                  </td>
                  <td className="tabular px-2 py-2 text-right text-[var(--color-bull)]/80">
                    {s.take_profit.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
