"use client";

import { useEffect, useState } from "react";
import { api, type OpsSnapshot } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";
import { Show } from "@/app/components/Show";
import { Info } from "@/app/components/Term";
import { PageIntro } from "@/app/components/PageIntro";

const REFRESH_MS = 5000;

const GOOD = "var(--color-bull)";
const WARN = "#e0a83d";
const BAD = "var(--color-bear)";

function relFromNow(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso).getTime();
  if (Number.isNaN(d)) return iso;
  const mins = Math.round(Math.abs(d - Date.now()) / 60000);
  const unit = mins < 60 ? `${mins} min` : `${Math.round(mins / 60)} h`;
  return d - Date.now() >= 0 ? `dans ${unit}` : unit;
}

function fmtDur(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}j ${Math.round((s % 86400) / 3600)}h`;
}

function fmtAgo(seconds: number | null): string {
  if (seconds == null) return "—";
  return fmtDur(seconds);
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
      <div className="h-full rounded-full" style={{ width: `${Math.min(100, Math.max(0, pct))}%`, background: color }} />
    </div>
  );
}

function band(pct: number | null | undefined, warn = 85, bad = 95): string {
  if (pct == null) return "var(--color-neutral)";
  if (pct >= bad) return BAD;
  if (pct >= warn) return WARN;
  return GOOD;
}

const UTIL_TONE: Record<string, string> = {
  balanced: GOOD,
  under: WARN,
  over: BAD,
  idle: "var(--color-neutral)",
};

function Metric({ label, value, pct, color }: { label: string; value: string; pct: number | null | undefined; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-[var(--color-muted)]">{label}</span>
        <span className="tabular">{value}</span>
      </div>
      <div className="mt-1"><Bar pct={pct ?? 0} color={color} /></div>
    </div>
  );
}

export default function OpsPage() {
  const { t } = useI18n();
  const [snap, setSnap] = useState<OpsSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [updatedAgo, setUpdatedAgo] = useState<number>(0);

  useEffect(() => {
    let alive = true;
    let last = Date.now();
    const tick = () => {
      api
        .ops()
        .then((s) => {
          if (!alive) return;
          setSnap(s);
          setErr(null);
          last = Date.now();
        })
        .catch((e) => alive && setErr(String(e)));
    };
    tick();
    const id = setInterval(tick, REFRESH_MS);
    const ago = setInterval(() => alive && setUpdatedAgo(Math.round((Date.now() - last) / 1000)), 1000);
    return () => {
      alive = false;
      clearInterval(id);
      clearInterval(ago);
    };
  }, []);

  const hpoPct = snap && snap.hpo.total > 0 ? Math.round((snap.hpo.hpo_done / snap.hpo.total) * 100) : 0;
  const errorCount = snap?.alerts.filter((a) => a.level === "error").length ?? 0;
  const warnCount = snap?.alerts.filter((a) => a.level === "warning").length ?? 0;
  const etaSeconds =
    snap && snap.sweep.avg_seconds && snap.hpo.pending > 0 ? snap.sweep.avg_seconds * snap.hpo.pending : null;

  // Overall health banner.
  let bannerTone = GOOD;
  let bannerText = t("ops.allGood");
  if (snap) {
    const stale = snap.sweep.idle_seconds != null && snap.sweep.idle_seconds > 600;
    if (errorCount > 0) {
      bannerTone = BAD;
      bannerText = `${errorCount} ${t("ops.alerts").toLowerCase()}`;
    } else if (snap.sweep.running && !stale) {
      bannerTone = GOOD;
      bannerText = t("ops.sweepHealthy");
    } else if (snap.sweep.running && stale) {
      bannerTone = WARN;
      bannerText = t("ops.sweepStale");
    } else if (snap.hpo.pending > 0) {
      bannerTone = WARN;
      bannerText = t("ops.sweepIdle");
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <div className="flex items-baseline justify-between">
        <h1 className="font-display text-3xl font-bold">{t("ops.title")}</h1>
        <span className="flex items-center gap-2 text-xs text-[var(--color-faint)]">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full" style={{ background: err ? BAD : GOOD }} />
          {t("ops.updated")} {updatedAgo}s {t("ops.ago")}
        </span>
      </div>
      <p className="mt-2 max-w-2xl text-sm text-[var(--color-muted)]">{t("ops.intro")}</p>
      <div className="mt-4">
        <PageIntro page="ops" />
      </div>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}
      {!snap && !err && <p className="mt-6 text-[var(--color-muted)]">…</p>}

      {snap && (
        <>
          {/* Health banner */}
          <div
            className="mt-6 flex items-center gap-3 rounded-lg border px-4 py-3 text-sm"
            style={{ borderColor: bannerTone, color: bannerTone, background: `color-mix(in srgb, ${bannerTone} 8%, transparent)` }}
          >
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: bannerTone }} />
            <span className="font-semibold">{bannerText}</span>
            {snap.sweep.running && snap.sweep.current && (
              <span className="text-[var(--color-muted)]">· {t("ops.current")}: <span className="tabular">{snap.sweep.current}</span></span>
            )}
            {warnCount > 0 && errorCount === 0 && (
              <span className="ml-auto text-[var(--color-faint)]">{warnCount} ⚠</span>
            )}
          </div>

          {/* Utilization verdict — translates the raw GPU/CPU gauges into one plain read. */}
          {(() => {
            const u = snap.utilization;
            const tone = UTIL_TONE[u.verdict] ?? "var(--color-neutral)";
            return (
              <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                <span className="text-[var(--color-muted)]">{t("ops.util.title")}</span>
                <span className="font-semibold" style={{ color: tone }}>
                  {t(`ops.util.${u.verdict}`)}
                </span>
                {u.n_gpus > 0 && u.gpu_avg_pct != null && (
                  <span className="tabular text-[var(--color-faint)]">
                    {t("ops.util.gpuAvg")} {u.gpu_avg_pct}%
                    {u.idle_gpus > 0 ? ` · ${u.idle_gpus}/${u.n_gpus} ${t("ops.util.idleGpu")}` : ""}
                  </span>
                )}
                {u.cpu_ratio != null && (
                  <span className="tabular text-[var(--color-faint)]">{t("ops.cpu")} ×{u.cpu_ratio}</span>
                )}
                <Show min="standard">
                  {u.reasons.length > 0 && (
                    <span className="text-[var(--color-muted)]">
                      — {u.reasons.map((r) => t(`ops.util.r.${r}`)).join(" · ")}
                    </span>
                  )}
                </Show>
              </div>
            );
          })()}

          <div className="mt-6 grid gap-6 md:grid-cols-2">
            {/* Sweep activity */}
            <section className="card p-5">
              <h2 className="font-display text-lg font-bold">{t("ops.sweep")}</h2>
              <div className="mt-3 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-[var(--color-muted)]">{t("ops.status")}</span>
                  <span style={{ color: snap.sweep.running ? GOOD : WARN }}>
                    {snap.sweep.running ? t("ops.sweepRunning") : t("ops.sweepStopped")}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-muted)]">{t("ops.current")}</span>
                  <span className="tabular">{snap.sweep.current ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-muted)]">{t("ops.lastActivity")}</span>
                  <span className="tabular" style={{ color: (snap.sweep.idle_seconds ?? 0) > 600 ? WARN : "inherit" }}>
                    {snap.sweep.idle_seconds != null ? `${t("ops.ago")} ${fmtAgo(snap.sweep.idle_seconds)}` : "—"}
                  </span>
                </div>
                <Show min="standard">
                  <div className="flex justify-between">
                    <span className="text-[var(--color-muted)]">{t("ops.perTriple")}</span>
                    <span className="tabular">{fmtDur(snap.sweep.avg_seconds)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[var(--color-muted)]">{t("ops.eta")}</span>
                    <span className="tabular">{fmtDur(etaSeconds)}</span>
                  </div>
                  {snap.sweep.gave_up > 0 && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-muted)]">{t("ops.gaveUp")}</span>
                      <span className="tabular" style={{ color: WARN }}>{snap.sweep.gave_up}</span>
                    </div>
                  )}
                </Show>
              </div>
            </section>

            {/* HPO progress */}
            <section className="card p-5">
              <h2 className="font-display text-lg font-bold">{t("ops.hpo")}<Info id="hpo" /></h2>
              <div className="mt-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-[var(--color-muted)]">{t("ops.hpoDone")}</span>
                  <span className="tabular">{snap.hpo.hpo_done} / {snap.hpo.total} ({hpoPct}%)</span>
                </div>
                <div className="mt-1"><Bar pct={hpoPct} color={GOOD} /></div>
                <div className="mt-3 flex flex-wrap gap-4 text-xs text-[var(--color-muted)]">
                  <span className="text-[var(--color-bull)]">{snap.hpo.promoted} {t("ops.promoted")}</span>
                  <span>{snap.hpo.deep_complete} {t("ops.deepComplete")}</span>
                  <span>{snap.hpo.advisory} {t("ops.advisory")}</span>
                  <span>{snap.hpo.pending} {t("ops.pending")}</span>
                </div>
                {snap.hpo.recent.length > 0 && (
                  <Show min="standard">
                    <div className="mt-3 border-t border-[var(--color-line)]/50 pt-2">
                      <div className="text-[11px] uppercase tracking-widest text-[var(--color-faint)]">{t("ops.recentTrained")}</div>
                      {snap.hpo.recent.map((r) => (
                        <div key={`${r.ticker}-${r.side}-${r.strategy ?? "fixed"}`} className="mt-1 flex justify-between text-xs">
                          <span>{r.ticker}/{r.side}{r.strategy && r.strategy !== "fixed" ? `/${r.strategy}` : ""}</span>
                          <span className={r.status === "promoted" ? "text-[var(--color-bull)]" : "text-[var(--color-faint)]"}>
                            {r.status} · {relFromNow(r.trained_at)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </Show>
                )}
              </div>
            </section>

            {/* System: CPU / RAM / disk */}
            <section className="card p-5">
              <h2 className="font-display text-lg font-bold">{t("ops.system")}</h2>
              <div className="mt-3 space-y-3">
                <Metric
                  label={`${t("ops.cpu")}${snap.system.n_cpus ? ` (${snap.system.n_cpus} ${t("ops.cores")})` : ""}`}
                  value={snap.system.cpu_pct != null ? `${snap.system.cpu_pct}%` : "—"}
                  pct={snap.system.cpu_pct}
                  color={band(snap.system.cpu_pct, 92, 98)}
                />
                <div className="flex justify-between text-[11px] text-[var(--color-faint)]">
                  <span>{t("ops.load")}</span>
                  <span className="tabular">{snap.system.load1 ?? "—"} · {snap.system.load5 ?? "—"} · {snap.system.load15 ?? "—"}</span>
                </div>
                <Metric
                  label={t("ops.ram")}
                  value={`${snap.system.mem_used_gb ?? "—"} / ${snap.system.mem_total_gb ?? "—"} GB`}
                  pct={snap.system.mem_used_pct}
                  color={band(snap.system.mem_used_pct)}
                />
                <Metric
                  label={t("ops.disk")}
                  value={`${snap.system.disk_used_gb ?? "—"} / ${snap.system.disk_total_gb ?? "—"} GB`}
                  pct={snap.system.disk_used_pct}
                  color={band(snap.system.disk_used_pct)}
                />
              </div>
            </section>

            {/* GPUs */}
            <section className="card p-5">
              <h2 className="font-display text-lg font-bold">{t("ops.gpus")}</h2>
              {snap.gpus.length === 0 && <p className="mt-2 text-sm text-[var(--color-faint)]">{t("ops.noGpu")}</p>}
              <div className="mt-3 space-y-4">
                {snap.gpus.map((g) => (
                  <div key={g.index}>
                    <div className="flex justify-between text-xs">
                      <span className="text-[var(--color-muted)]">GPU {g.index}</span>
                      <span className="tabular" style={{ color: g.temp_c >= 85 ? BAD : g.temp_c >= 75 ? WARN : "inherit" }}>
                        {g.util_pct}% · {g.temp_c}°C
                      </span>
                    </div>
                    <div className="mt-1"><Bar pct={g.util_pct} color={g.util_pct > 0 ? GOOD : "var(--color-line)"} /></div>
                    <Show min="expert">
                      <div className="mt-1 flex justify-between text-[11px] text-[var(--color-faint)]">
                        <span>{t("ops.mem")}</span>
                        <span className="tabular">{(g.mem_used_mb / 1024).toFixed(1)} / {(g.mem_total_mb / 1024).toFixed(0)} GB</span>
                      </div>
                      <div className="mt-0.5"><Bar pct={(g.mem_used_mb / g.mem_total_mb) * 100} color="var(--color-neutral)" /></div>
                    </Show>
                  </div>
                ))}
              </div>
            </section>

            {/* Scheduler jobs */}
            <Show min="standard">
              <section className="card p-5">
                <h2 className="font-display text-lg font-bold">{t("ops.jobs")}</h2>
                <p className="mt-1 text-xs text-[var(--color-faint)]">
                  {t("ops.scheduler")}:{" "}
                  {snap.scheduler.state === "running" ? (
                    <span className="text-[var(--color-bull)]">{snap.scheduler.state}</span>
                  ) : snap.sweep.running ? (
                    <span style={{ color: WARN }}>{snap.scheduler.state} · {t("ops.schedulerPaused")}</span>
                  ) : (
                    <span className="text-[var(--color-bear)]">{snap.scheduler.state}</span>
                  )}
                </p>
                <div className="mt-3 space-y-1.5">
                  {snap.jobs.map((j) => (
                    <div key={j.id} className="flex justify-between text-xs">
                      <span className="text-[var(--color-muted)]">{j.id}</span>
                      <span className="tabular text-[var(--color-faint)]">{relFromNow(j.next_run)}</span>
                    </div>
                  ))}
                </div>
              </section>
            </Show>

            {/* Alerts */}
            <section className="card p-5">
              <h2 className="font-display text-lg font-bold">{t("ops.alerts")}</h2>
              {snap.alerts.length === 0 ? (
                <p className="mt-2 flex items-center gap-2 text-sm text-[var(--color-bull)]">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: GOOD }} />
                  {t("ops.noAlerts")}
                </p>
              ) : (
                <div className="mt-3 space-y-1.5 font-mono text-[11px] leading-relaxed">
                  {snap.alerts.map((a, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="shrink-0" style={{ color: a.level === "error" ? BAD : WARN }}>{a.time.slice(11, 19)}</span>
                      <span className="text-[var(--color-muted)]">{a.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>

          {/* Recent logs */}
          <Show min="expert">
            <section className="card mt-6 p-5">
              <h2 className="font-display text-lg font-bold">{t("ops.logs")}</h2>
              <div className="mt-3 max-h-72 space-y-1 overflow-y-auto font-mono text-[11px] leading-relaxed">
                {snap.logs.map((l, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="shrink-0 text-[var(--color-faint)]">{l.time.slice(11, 19)}</span>
                    <span style={{ color: l.level === "error" ? BAD : l.level === "warning" ? WARN : "var(--color-muted)" }}>
                      {l.message}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </Show>
        </>
      )}
    </main>
  );
}
