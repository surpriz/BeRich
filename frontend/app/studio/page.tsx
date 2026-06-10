"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type VideoScript } from "@/app/lib/api";
import { useI18n } from "@/app/lib/i18n";

/**
 * /studio — the daily video script, ready to read on camera.
 *
 * Assembled server-side from the paper book's FACTS (executions, realized net P&L, open
 * positions, SHAP factors in plain French) — never the forecast. The script itself is in
 * French (the audience's language); only the page chrome is bilingual.
 */

export default function StudioPage() {
  const { locale } = useI18n();
  const fr = locale === "fr";
  const [data, setData] = useState<VideoScript | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api.videoScript().then(setData).catch((e) => setErr(String(e)));
  }, []);

  const L = {
    title: fr ? "Studio — le script du jour" : "Studio — today's script",
    intro: fr
      ? "Le texte prêt-à-dire de votre vidéo quotidienne, généré automatiquement depuis les faits du robot : exécutions réelles, P&L net de frais, le « pourquoi » de chaque trade en langage simple, la leçon de discipline et le disclaimer obligatoire. Jamais de prévision."
      : "The ready-to-read text for your daily video, auto-generated from the robot's facts: real executions, net P&L, the plain-language 'why' of each trade, the discipline lesson and the mandatory disclaimer. Never a forecast.",
    copy: fr ? "Copier le script" : "Copy script",
    copiedTxt: fr ? "Copié ✓" : "Copied ✓",
    back: fr ? "← Retour" : "← Back",
    loading: "…",
  };

  const copyAll = () => {
    if (!data) return;
    navigator.clipboard.writeText(data.script).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
      >
        {L.back}
      </Link>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h1 className="font-display text-4xl font-extrabold tracking-tight">{L.title}</h1>
        {data && (
          <span className="tabular font-mono text-sm text-[var(--color-faint)]">{data.date}</span>
        )}
      </div>
      <p className="mt-4 text-sm text-[var(--color-muted)]">{L.intro}</p>

      {err && <p className="mt-6 text-[var(--color-bear)]">{err}</p>}

      {data ? (
        <>
          <div className="mt-6 flex justify-end">
            <button
              type="button"
              onClick={copyAll}
              className="rounded-full bg-[var(--color-accent)] px-5 py-2 text-sm font-semibold text-white transition-opacity hover:opacity-90"
            >
              {copied ? L.copiedTxt : L.copy}
            </button>
          </div>
          <pre className="card mt-3 overflow-x-auto whitespace-pre-wrap p-6 font-sans text-sm leading-relaxed">
            {data.script}
          </pre>
        </>
      ) : (
        !err && <p className="mt-8 text-[var(--color-muted)]">{L.loading}</p>
      )}
    </main>
  );
}
