"use client";

import Link from "next/link";
import { useI18n } from "@/app/lib/i18n";

/**
 * Project journal / changelog — the step-by-step story of BeRich, newest first.
 *
 * Self-contained bilingual content (no i18n catalog keys) so the whole narrative lives in one
 * place and is easy to extend: append a new entry at the TOP of ENTRIES after each milestone.
 * Sources of truth this mirrors: git history, docs/RESULTS.md (the honest exploration log) and
 * CLAUDE.md (phase descriptions). Verdicts are reported honestly — most exploration phases did
 * NOT beat buy & hold, and that is stated plainly.
 */

type Tone = "ship" | "research" | "infra" | "risk";

type Entry = {
  date: string; // ISO or era label
  version?: string;
  tone: Tone;
  title: { fr: string; en: string };
  points: { fr: string; en: string }[];
  verdict?: { fr: string; en: string };
};

const TONE: Record<Tone, { label: { fr: string; en: string }; color: string }> = {
  ship: { label: { fr: "Livré", en: "Shipped" }, color: "var(--color-bull)" },
  research: { label: { fr: "Recherche", en: "Research" }, color: "#8b5cf6" },
  infra: { label: { fr: "Infra", en: "Infra" }, color: "#3b9eff" },
  risk: { label: { fr: "Risque", en: "Risk" }, color: "#e0a83d" },
};

const ENTRIES: Entry[] = [
  {
    date: "2026-06-04",
    version: "v0.6.0",
    tone: "risk",
    title: {
      fr: "Robustesse, rigueur de validation & money-management (audit trader 360°)",
      en: "Robustness, validation rigor & money-management (360° trader audit)",
    },
    points: [
      {
        fr: "Phase 1 — Fiabilité : Deflated Sharpe alimenté par le vrai nombre d'essais HPO (anti data-mining), plancher de 20 trades dans le guard, contrôle multiple-testing Benjamini-Hochberg sur tout le sweep (démotion automatique des promotions non significatives), slippage proportionnel au volume activé par défaut (coût honnête sur les actifs illiquides).",
        en: "Phase 1 — Reliability: Deflated Sharpe fed the real HPO trial count (anti data-mining), a 20-trade floor in the guard, sweep-wide Benjamini-Hochberg multiple-testing control (auto-demotes non-significant promotions), volume-proportional slippage on by default (honest cost on illiquid names).",
      },
      {
        fr: "Mode observation : un livre papier « shadow » suit en direct les modèles presque-promus, sans engager de capital — on continue d'accumuler des preuves avant de leur faire confiance.",
        en: "Observation mode: a paper-only shadow book tracks near-miss models live, with no capital engaged — forward evidence accrues before we trust them.",
      },
      {
        fr: "Phase 2 — Money-management : kill-switch de drawdown gradué (réduction à −10 %, arrêt à −20 %), cap de concentration par classe d'actif (40 %), plafond de positions concurrentes (15). Risque fixe à 1 % conservé.",
        en: "Phase 2 — Money-management: graduated drawdown kill-switch (de-risk at −10%, halt at −20%), per-asset-class concentration cap (40%), max concurrent positions (15). Fixed 1% risk kept.",
      },
      {
        fr: "Phase 3 — Prédiction : ensemble multi-modèles au service (opt-in), seuils de décision par actif tunés à l'entraînement, conditionnement du seuil par régime de volatilité (opt-in), surveillance hebdomadaire de dérive par actif avec alerte email.",
        en: "Phase 3 — Prediction: serve-time multi-model ensemble (opt-in), per-asset decision thresholds tuned at training, volatility-regime threshold conditioning (opt-in), weekly per-asset drift monitor with email alert.",
      },
      {
        fr: "335 tests verts, ruff + ty clean, frontend build OK.",
        en: "335 tests green, ruff + ty clean, frontend build OK.",
      },
    ],
    verdict: {
      fr: "Aucune revendication d'edge nouvelle. L'objectif : rendre les signaux dignes de confiance et le capital protégé, avec des pertes maîtrisées.",
      en: "No new edge claim. The goal: make the signals trustworthy and capital protected, with losses kept bounded.",
    },
  },
  {
    date: "2026-06-04",
    tone: "ship",
    title: {
      fr: "Univers porté à 100 actifs + tableau de bord enrichi",
      en: "Universe grown to 100 assets + richer dashboard",
    },
    points: [
      {
        fr: "Univers tradeable 50 → 100 actifs sur 5 classes (US, FR, forex, crypto, commodities) ; la file HPO priorise les nouveaux arrivants pour un cold-start rapide.",
        en: "Tradeable universe 50 → 100 assets across 5 classes (US, FR, forex, crypto, commodities); the HPO queue prioritizes new arrivals for a fast cold-start.",
      },
      {
        fr: "Pyramiding borné par des caps d'exposition par ticker et par livre ; jauges proba/seuil, bannière de couverture, barres de position & d'exposition ; tables paper en pleine largeur.",
        en: "Pyramiding bounded by per-ticker and per-book exposure caps; proba/threshold gauges, coverage banner, position & exposure bars; full-width paper tables.",
      },
      {
        fr: "Signal du jour par côté affiché dans le panneau Training & HPO de chaque ticker ; horodatage du dernier entraînement/HPO.",
        en: "Today's per-side signal shown in each ticker's Training & HPO panel; last training/HPO timestamps.",
      },
    ],
  },
  {
    date: "2026-06-03",
    tone: "infra",
    title: {
      fr: "Réentraîneur continu 24/7 + étude de faisabilité intraday",
      en: "Continuous 24/7 retrainer + intraday feasibility study",
    },
    points: [
      {
        fr: "Réentraîneur continu qui recharge la config à chaque cycle (nouveaux actifs pris en compte sans redéploiement) et priorise les actifs jamais cherchés.",
        en: "Continuous retrainer that reloads config each cycle (new assets picked up with no redeploy) and prioritizes never-searched assets.",
      },
      {
        fr: "Univers 29 → 50 actifs ; colonne open-date au paper, historique par stratégie sur la page ticker.",
        en: "Universe 29 → 50 assets; paper open-date column, per-strategy history on the ticker page.",
      },
      {
        fr: "Étude go/no-go de l'intraday crypto 1h (faisabilité documentée).",
        en: "Go/no-go feasibility study for intraday crypto 1h (documented).",
      },
    ],
  },
  {
    date: "2026-06-02",
    version: "Phase 13",
    tone: "ship",
    title: {
      fr: "Stratégies de sortie (trailing) + short autonome + monitoring",
      en: "Exit strategies (trailing) + autonomous shorts + monitoring",
    },
    points: [
      {
        fr: "Stratégies de sortie au-delà du TP/SL fixe : trailing (stop suiveur, sans TP) et trailing_tp (cap TP + stop suiveur), chacune entraînée + gatée comme un modèle parallèle, avec bascule sur le dashboard.",
        en: "Exit strategies beyond fixed TP/SL: trailing (ratcheting stop, no TP) and trailing_tp (TP cap + ratchet), each trained + gated as a parallel model, with a dashboard toggle.",
      },
      {
        fr: "Paper trading short-side autonome (ouverture + clôture) ; conseil en langage clair par signal (pourquoi long/short/neutre).",
        en: "Autonomous short-side paper trading (open + close); plain-language advice per signal (why long/short/neutral).",
      },
      {
        fr: "Vrai tableau de bord /ops (métriques système, activité du sweep, file HPO) ; sweep d'entraînement persistant via systemd.",
        en: "Real /ops dashboard (system metrics, sweep activity, HPO queue); persistent training sweep via systemd.",
      },
    ],
  },
  {
    date: "2026-06-01",
    version: "Phase 12",
    tone: "ship",
    title: {
      fr: "Tournoi par actif + shorts directionnels (architecture)",
      en: "Per-asset tournament + directional shorts (architecture)",
    },
    points: [
      {
        fr: "Entraînement/HPO passés du per-classe au per-actif : chaque actif a ses propres études Optuna par (ticker, modèle, côté) et un tournoi qui garde le gagnant walk-forward parmi LightGBM/LSTM/PatchTST/TFT.",
        en: "Training/HPO moved from per-class to per-asset: each asset gets its own Optuna studies per (ticker, model, side) and a tournament keeping the walk-forward winner among LightGBM/LSTM/PatchTST/TFT.",
      },
      {
        fr: "Label + backtest direction-aware → chaque actif peut porter un modèle long ET un short ; sizing en distance absolue plafonné au capital (sans levier).",
        en: "Direction-aware label + backtest → each asset can carry a long AND a short model; absolute-distance sizing capped at capital (no leverage).",
      },
      {
        fr: "Onglets /training (inventaire des modèles par actif) et /ops (statut live) ; alertes email sur crash de job ; backups rotatifs ; HPO par actif à 100 essais ; recherche d'horizon (5/10/15/20) ; rendement attendu par signal.",
        en: "/training (per-asset model inventory) and /ops (live status) tabs; email alerts on job crash; rotating backups; 100-trial per-asset HPO; horizon search (5/10/15/20); expected return per signal.",
      },
    ],
    verdict: {
      fr: "Architecture, pas une revendication d'edge : avec ~400 fenêtres labellisées par actif, la plupart restent « indicatifs ». 7 modèles promus au premier sweep (forex faibles + 2 actions FR + 1 short), à traiter comme candidats paper, non validés.",
      en: "Architecture, not an edge claim: with ~400 labeled windows/asset, most stay advisory. 7 models promoted on the first sweep (weak forex + 2 FR stocks + 1 short), to treat as paper candidates, not validated.",
    },
  },
  {
    date: "2026-05-31",
    tone: "infra",
    title: {
      fr: "Page « Under the hood » + seuil d'achat ajusté",
      en: "\"Under the hood\" page + tuned buy threshold",
    },
    points: [
      {
        fr: "Page architecture exhaustive avec diagramme de pipeline ; seuil d'achat 0.40 → 0.35 (atteignable sur l'échelle calibrée, qui se regroupe vers 0.32).",
        en: "Exhaustive architecture page with a pipeline diagram; buy threshold 0.40 → 0.35 (reachable on the calibrated scale, which clusters near 0.32).",
      },
    ],
  },
  {
    date: "v0.5.x",
    version: "Phases 10–11b",
    tone: "research",
    title: {
      fr: "Zoo de modèles GPU, long/short neutre, microstructure, fondamentaux",
      en: "GPU model zoo, market-neutral long/short, microstructure, fundamentals",
    },
    points: [
      {
        fr: "Phase 10 : zoo GPU (LSTM/PatchTST/TFT), ranker cross-sectionnel market-neutral, conseil adaptatif (SL/TP mis à l'échelle par une prévision de volatilité).",
        en: "Phase 10: GPU zoo (LSTM/PatchTST/TFT), market-neutral cross-sectional ranker, adaptive advice (SL/TP scaled by a volatility forecast).",
      },
      {
        fr: "Phase 11 : famille de features microstructure (CLV, gap d'ouverture, range, Parkinson, illiquidité d'Amihud, spread de Roll), toutes causales.",
        en: "Phase 11: microstructure feature family (CLV, opening gap, range, Parkinson, Amihud illiquidity, Roll spread), all causal.",
      },
      {
        fr: "Phase 11b : fondamentaux point-in-time — non concluant (profondeur de données insuffisante).",
        en: "Phase 11b: point-in-time fundamentals — inconclusive (insufficient data depth).",
      },
    ],
  },
  {
    date: "v0.4.0",
    version: "Phases 7–9",
    tone: "research",
    title: {
      fr: "PEAD, overlay de gestion du risque, reframe portefeuille",
      en: "PEAD, risk-management overlay, portfolio reframe",
    },
    points: [
      {
        fr: "Phase 7 : modèle événementiel PEAD (drift post-résultats) — AUC ≈ 0.535, Sharpe par trade ≈ 0.849, max DD ≈ −6.7 %. Le plus proche d'un signal utilisable.",
        en: "Phase 7: PEAD event model (post-earnings drift) — AUC ≈ 0.535, trade Sharpe ≈ 0.849, max DD ≈ −6.7%. The closest to a usable signal.",
      },
      {
        fr: "Phase 8 : overlay de gestion du risque (gates de régime/drawdown, sizing basé sur le risque).",
        en: "Phase 8: risk-management overlay (regime/drawdown gates, risk-based sizing).",
      },
      {
        fr: "Phase 9 : reframe portefeuille core-satellite (B&H + satellites).",
        en: "Phase 9: core-satellite portfolio reframe (B&H + satellites).",
      },
    ],
    verdict: {
      fr: "Tout blend B&H + PEAD/calendrier réduit le Sharpe vs B&H pur ; l'optimiseur walk-forward n'a jamais choisi PEAD. Reframe fermé.",
      en: "Every B&H + PEAD/calendar blend reduces Sharpe vs pure B&H; the walk-forward optimizer never picked PEAD. Reframe closed.",
    },
  },
  {
    date: "v0.2–v0.3",
    version: "Phases 5–6",
    tone: "research",
    title: {
      fr: "Résultats, sentiment d'actualité, expansion d'univers",
      en: "Earnings, news sentiment, universe expansion",
    },
    points: [
      {
        fr: "Phase 5a : features de surprise sur résultats (données gratuites, sans GPU).",
        en: "Phase 5a: earnings-surprise features (free data, no GPU).",
      },
      {
        fr: "Phase 5b : sentiment d'actualité via Alpha Vantage + scoring FinBERT (GPU).",
        en: "Phase 5b: news sentiment via Alpha Vantage + FinBERT scoring (GPU).",
      },
      {
        fr: "Phase 6 : expansion d'univers (mega / mid / small / all caps) ; baseline calendaire (turn-of-month, sans ML).",
        en: "Phase 6: universe expansion (mega / mid / small / all caps); calendar-only baseline (turn-of-month, no ML).",
      },
    ],
    verdict: {
      fr: "Même verdict que la Phase 3 : ces features dominent parfois l'importance mais n'élèvent pas l'AUC OOS et ne battent pas le buy & hold.",
      en: "Same verdict as Phase 3: these features sometimes dominate importance but don't lift OOS AUC and don't beat buy & hold.",
    },
  },
  {
    date: "v0.1",
    version: "Phases 1–3",
    tone: "research",
    title: {
      fr: "Fondations : données, features causales, label triple-barrière, le guard",
      en: "Foundations: data, causal features, triple-barrier label, the guard",
    },
    points: [
      {
        fr: "Pipeline de données yfinance (OHLCV + macro cross-asset), indicateurs techniques causaux, label triple-barrière direction-aware avec poids d'échantillon, splits walk-forward avec embargo.",
        en: "yfinance data pipeline (OHLCV + cross-asset macro), causal technical indicators, direction-aware triple-barrier label with sample weights, walk-forward splits with embargo.",
      },
      {
        fr: "Moteur de backtest triple-barrière avec frais + slippage réalistes ; baseline LightGBM ; calibration des probabilités (isotonic).",
        en: "Triple-barrier backtest engine with realistic fees + slippage; LightGBM baseline; probability calibration (isotonic).",
      },
      {
        fr: "La règle du guard : un modèle n'est promu que s'il bat son propre benchmark out-of-sample (long : bat le buy & hold ; short/neutre : Sharpe positif et significatif). Pas de lookahead, jamais.",
        en: "The guard rule: a model is promoted only if it beats its own benchmark out-of-sample (long: beats buy & hold; short/neutral: positive, significant Sharpe). No lookahead, ever.",
      },
      {
        fr: "Phase 3 : ingénierie de features, baseline LSTM (GPU), sweep de géométrie de label, features macro cross-asset (revertées).",
        en: "Phase 3: feature engineering, LSTM baseline (GPU), label-geometry sweep, cross-asset macro features (reverted).",
      },
    ],
    verdict: {
      fr: "Les features macro dominent l'importance LightGBM mais n'élèvent pas l'AUC. Cet espace est fermé : ne pas rouvrir les chasses aux features OHLCV/macro.",
      en: "Macro features dominate LightGBM importance but don't lift AUC. That space is settled: do not reopen OHLCV/macro feature hunts.",
    },
  },
];

function Badge({ tone, locale }: { tone: Tone; locale: "fr" | "en" }) {
  const t = TONE[tone];
  return (
    <span
      className="rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest"
      style={{ color: t.color, backgroundColor: `${t.color}1a` }}
    >
      {t.label[locale]}
    </span>
  );
}

export default function ChangelogPage() {
  const { locale } = useI18n();
  const back = locale === "fr" ? "← Retour" : "← Back";
  const title = locale === "fr" ? "Journal de bord" : "Project journal";
  const intro =
    locale === "fr"
      ? "L'histoire de BeRich, étape par étape — du plus récent au plus ancien. Les verdicts sont rapportés honnêtement : la plupart des phases d'exploration n'ont pas battu le buy & hold, et c'est dit clairement. Ce journal sert de changelog et de pense-bête pour la suite."
      : "The story of BeRich, step by step — newest first. Verdicts are reported honestly: most exploration phases did not beat buy & hold, and that's stated plainly. This journal doubles as a changelog and a memo for what's next.";
  const outcomeLabel = locale === "fr" ? "Verdict" : "Verdict";

  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
      >
        {back}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{title}</h1>
      <p className="mt-3 max-w-2xl text-base text-[var(--color-muted)]">{intro}</p>

      <div className="mt-10 flex flex-col gap-5">
        {ENTRIES.map((e, i) => (
          <section key={`${e.date}-${i}`} className="card p-5">
            <div className="flex flex-wrap items-center gap-3">
              <span className="tabular font-mono text-sm text-[var(--color-faint)]">{e.date}</span>
              {e.version ? (
                <span className="rounded bg-white/[0.04] px-2 py-0.5 text-xs font-semibold text-[var(--color-muted)]">
                  {e.version}
                </span>
              ) : null}
              <Badge tone={e.tone} locale={locale} />
            </div>
            <h2 className="mt-2 font-display text-xl font-bold">{e.title[locale]}</h2>
            <ul className="mt-3 flex flex-col gap-2">
              {e.points.map((p, j) => (
                <li key={j} className="flex gap-2 text-sm text-[var(--color-muted)]">
                  <span className="select-none text-[var(--color-bull)]">▸</span>
                  <span>{p[locale]}</span>
                </li>
              ))}
            </ul>
            {e.verdict ? (
              <p className="mt-3 border-l-2 border-[var(--color-line)] pl-3 text-sm italic text-[var(--color-faint)]">
                <span className="font-semibold not-italic text-[var(--color-muted)]">
                  {outcomeLabel} —{" "}
                </span>
                {e.verdict[locale]}
              </p>
            ) : null}
          </section>
        ))}
      </div>
    </main>
  );
}
