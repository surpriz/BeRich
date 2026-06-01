"use client";

import { useEffect, useState } from "react";
import { api, type OpsSnapshot } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { Show } from "@/app/components/Show";
import { Info } from "@/app/components/Term";

const REFRESH_MS = 5000;

function relTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso).getTime();
  if (Number.isNaN(d)) return iso;
  const diff = d - Date.now();
  const abs = Math.abs(diff);
  const mins = Math.round(abs / 60000);
  const unit = mins < 60 ? `${mins} min` : `${Math.round(mins / 60)} h`;
  return diff >= 0 ? `dans ${unit}` : `il y a ${unit}`;
}

function GpuBar({ pct, tone }: { pct: number; tone: "util" | "mem" }) {
  const color = tone === "util" ? "var(--color-bull)" : "var(--color-neutral)";
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
      <div className="h-full rounded-full" style={{ width: `${Math.min(100, pct)}%`, background: color }} />
    </div>
  );
}

export default function OpsPage() {
  const { t } = useI18n();
  const [snap, setSnap] = useState<OpsSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [updated, setUpdated] = useState<string>("");

  useEffect(() => {
    let alive = true;
    const tick = () => {
      api
        .ops()
        .then((s) => {
          if (!alive) return;
          setSnap(s);
          setErr(null);
          setUpdated(new Date().toLocaleTimeString());
        })
        .catch((e) => alive && setErr(String(e)));
    };
    tick();
    const id = setInterval(tick, REFRESH_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const hpoPct = snap && snap.hpo.total > 0 ? Math.round((snap.hpo.hpo_done / snap.hpo.total) * 100) : 0;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <div className="flex items-baseline justify-between">
        <h1 className="font-display text-3xl font-bold">{t("ops.title")}</h1>
        <span className="text-xs text-[var(--color-faint)]">
          {updated ? `${t("ops.updated")} ${updated}` : "…"}
        </span>
      </div>
      <p className="mt-2 max-w-2xl text-sm text-[var(--color-muted)]">{t("ops.intro")}</p>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}
      {!snap && !err && <p className="mt-6 text-[var(--color-muted)]">…</p>}

      {snap && (
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          {/* GPUs */}
          <section className="card p-5">
            <h2 className="font-display text-lg font-bold">{t("ops.gpus")}</h2>
            {snap.gpus.length === 0 && (
              <p className="mt-2 text-sm text-[var(--color-faint)]">{t("ops.noGpu")}</p>
            )}
            <div className="mt-3 space-y-4">
              {snap.gpus.map((g) => (
                <div key={g.index}>
                  <div className="flex justify-between text-xs">
                    <span className="text-[var(--color-muted)]">GPU {g.index}</span>
                    <span className="tabular">{g.util_pct}% · {g.temp_c}°C</span>
                  </div>
                  <div className="mt-1"><GpuBar pct={g.util_pct} tone="util" /></div>
                  <Show min="expert">
                    <div className="mt-1 flex justify-between text-[11px] text-[var(--color-faint)]">
                      <span>{t("ops.mem")}</span>
                      <span className="tabular">
                        {(g.mem_used_mb / 1024).toFixed(1)} / {(g.mem_total_mb / 1024).toFixed(0)} GB
                      </span>
                    </div>
                    <div className="mt-0.5"><GpuBar pct={(g.mem_used_mb / g.mem_total_mb) * 100} tone="mem" /></div>
                  </Show>
                </div>
              ))}
            </div>
          </section>

          {/* HPO progress */}
          <section className="card p-5">
            <h2 className="font-display text-lg font-bold">
              {t("ops.hpo")}
              <Info id="hpo" />
            </h2>
            <div className="mt-3 text-sm">
              <div className="flex justify-between">
                <span className="text-[var(--color-muted)]">{t("ops.hpoDone")}</span>
                <span className="tabular">{snap.hpo.hpo_done} / {snap.hpo.total} ({hpoPct}%)</span>
              </div>
              <div className="mt-1"><GpuBar pct={hpoPct} tone="util" /></div>
              <div className="mt-3 flex gap-4 text-xs text-[var(--color-muted)]">
                <span className="text-[var(--color-bull)]">{snap.hpo.promoted} {t("ops.promoted")}</span>
                <Show min="expert">
                  <span>{snap.hpo.advisory} {t("ops.advisory")}</span>
                  <span>{snap.hpo.pending} {t("ops.pending")}</span>
                </Show>
              </div>
              {snap.hpo.recent.length > 0 && (
                <Show min="standard">
                <div className="mt-3 border-t border-[var(--color-line)]/50 pt-2">
                  <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">
                    {t("ops.recentTrained")}
                  </div>
                  {snap.hpo.recent.map((r) => (
                    <div key={`${r.ticker}-${r.side}`} className="mt-1 flex justify-between text-xs">
                      <span>{r.ticker}/{r.side}</span>
                      <span className={r.status === "promoted" ? "text-[var(--color-bull)]" : "text-[var(--color-faint)]"}>
                        {r.status} · {relTime(r.trained_at)}
                      </span>
                    </div>
                  ))}
                </div>
                </Show>
              )}
            </div>
          </section>

          {/* Scheduler jobs */}
          <Show min="standard">
          <section className="card p-5">
            <h2 className="font-display text-lg font-bold">{t("ops.jobs")}</h2>
            <p className="mt-1 text-xs text-[var(--color-faint)]">
              {t("ops.scheduler")}: <span className={snap.scheduler.state === "running" ? "text-[var(--color-bull)]" : "text-[var(--color-bear)]"}>{snap.scheduler.state}</span>
            </p>
            <div className="mt-3 space-y-1.5">
              {snap.jobs.map((j) => (
                <div key={j.id} className="flex justify-between text-xs">
                  <span className="text-[var(--color-muted)]">{j.id}</span>
                  <span className="tabular text-[var(--color-faint)]">{relTime(j.next_run)}</span>
                </div>
              ))}
            </div>
          </section>
          </Show>

          {/* Recent logs */}
          <Show min="expert">
          <section className="card p-5">
            <h2 className="font-display text-lg font-bold">{t("ops.logs")}</h2>
            <div className="mt-3 max-h-72 space-y-1 overflow-y-auto font-mono text-[11px] leading-relaxed">
              {snap.logs.map((l, i) => (
                <div key={i} className="flex gap-2">
                  <span className="shrink-0 text-[var(--color-faint)]">{l.time.slice(11, 19)}</span>
                  <span className="text-[var(--color-muted)]">{l.message}</span>
                </div>
              ))}
            </div>
          </section>
          </Show>
        </div>
      )}
    </main>
  );
}
