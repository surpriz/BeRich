"use client";

import Link from "next/link";
import { useI18n } from "@/app/lib/i18n";
import type { Locale } from "@/app/lib/i18n";

type Block =
  | { kind: "p"; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "table"; head: string[]; rows: string[][] };

type Section = { id: string; title: string; blocks: Block[] };

const CONTENT: Record<Locale, Section[]> = {
  fr: [
    {
      id: "overview",
      title: "1. Vue d'ensemble",
      blocks: [
        { kind: "p", text: "BeRich estime, pour chaque actif, une probabilité de tendance haussière à horizon de quelques jours (pas d'intraday) : la chance que la barrière haute (objectif) soit touchée avant la barrière basse (stop) dans les N prochains jours. Ce n'est PAS une prédiction de prix." },
        { kind: "p", text: "Règle d'or non négociable : un modèle n'est « promu » (servi en production) que s'il bat à la fois la baseline LightGBM ET le simple « acheter et garder », en walk-forward, hors échantillon, avec frais et slippage réalistes. Tant que ce n'est pas le cas, tout est affiché comme indicatif. À ce jour, aucun modèle ne franchit cette barre — c'est volontaire et honnête." },
      ],
    },
    {
      id: "data",
      title: "2. Acquisition des données",
      blocks: [
        { kind: "p", text: "Tout part de données quotidiennes (barres journalières), récupérées et mises en cache localement (Parquet)." },
        { kind: "ul", items: [
          "Prix OHLCV (ouverture, haut, bas, clôture, volume) via yfinance, depuis 2010, ajustés des splits/dividendes.",
          "Univers : ~274 actions US (10 méga-caps + ~113 mid + ~151 small), plus 8 actions FR, 5 paires forex, 3 cryptos, 3 matières premières.",
          "Calendrier de résultats (earnings) : dates passées et futures, surprises EPS.",
          "Actualités via Alpha Vantage, puis score de sentiment par FinBERT (voir §4).",
          "Fondamentaux trimestriels (CA, résultat net, actifs, capitaux propres, dette) — historique court (~5 ans chez yfinance), donc usage prospectif.",
        ] },
      ],
    },
    {
      id: "features",
      title: "3. Feature engineering",
      blocks: [
        { kind: "p", text: "À partir des prix, on calcule des indicateurs « causaux » (ils n'utilisent que le passé jusqu'à la date t — jamais le futur). Le jeu de base compte 22 features ; des familles optionnelles s'ajoutent." },
        { kind: "table", head: ["Famille", "Exemples", "Capte"], rows: [
          ["Base — momentum", "rendements 1/5j, momentum 10/20/60/120j", "tendance / élan"],
          ["Base — oscillateurs", "RSI 14, MACD (ligne/signal/histogramme)", "sur-achat / sur-vente"],
          ["Base — volatilité", "ATR%, volatilité réalisée 20j", "régime de risque"],
          ["Base — tendance/MM", "écart aux moyennes 20/50j, distance aux plus hauts/bas 60j", "position relative"],
          ["Base — volume / calendrier / régime", "z-score volume, saisonnalité, rendement+vol du marché (SPY)", "liquidité, effets de calendrier, contexte marché"],
          ["Microstructure (nouveau)", "CLV, gap d'ouverture, amplitude, Parkinson, illiquidité d'Amihud, spread de Roll", "liquidité, spread, position intraday"],
          ["Earnings (opt.)", "jours avant/après résultats, surprises", "pression événementielle"],
          ["News / FinBERT (opt.)", "sentiment moyen/extrême, divergence prix-sentiment", "tonalité des actualités"],
          ["Fondamentaux (opt., point-in-time)", "marge nette, ROE, dette/capitaux, croissance CA", "qualité / valorisation"],
        ] },
        { kind: "p", text: "Important : les scalers (normalisation) sont calibrés uniquement sur la portion d'entraînement, jamais sur le test — pour éviter toute fuite d'information. La recherche d'hyperparamètres (§8) choisit elle-même quelles familles garder ; le modèle US actif utilise actuellement momentum + calendrier + microstructure." },
      ],
    },
    {
      id: "finbert",
      title: "4. FinBERT (sentiment des actualités)",
      blocks: [
        { kind: "p", text: "FinBERT est un modèle de langage spécialisé en finance. Chaque actualité (titre + résumé) est classée en négatif / neutre / positif sur GPU. On en dérive un score de sentiment et des features (sentiment moyen sur 5 jours, intensité, divergence avec le prix). C'est le seul endroit où une IA de type « grand modèle de langage » intervient ; le reste repose sur des modèles tabulaires/séquentiels." },
      ],
    },
    {
      id: "labels",
      title: "5. Comment on définit « gagner » (labellisation)",
      blocks: [
        { kind: "p", text: "Méthode de la triple barrière : pour chaque jour, on ouvre un long hypothétique et on regarde les 10 jours suivants. Objectif = entrée + 2×ATR ; stop = entrée − 1×ATR ; barrière de temps = 10 jours." },
        { kind: "ul", items: [
          "Touche l'objectif en premier → label 1 (gagné).",
          "Touche le stop en premier → label 0 (perdu).",
          "Ni l'un ni l'autre en 10 jours → résolu par le signe du rendement final.",
        ] },
        { kind: "p", text: "Le modèle apprend P(gain) = probabilité de toucher l'objectif avant le stop. Comme le ratio gain/risque est 2:1, le seuil de rentabilité n'est pas 50 % mais ~33 % : au-dessus de 33 % de chances de gain, l'espérance est positive." },
      ],
    },
    {
      id: "models",
      title: "6. Les modèles",
      blocks: [
        { kind: "p", text: "Tous respectent une interface commune (entraînement / prédiction de probabilité), donc interchangeables." },
        { kind: "ul", items: [
          "LightGBM : arbres de décision boostés (CPU, rapide) — c'est la baseline et le modèle US actuellement servi.",
          "LSTM, PatchTST (transformeur), TFT : réseaux profonds sur GPU (pur PyTorch), lisant l'historique récent par séquences.",
          "Ensemble (stacking) : un méta-modèle combine les avis des modèles de base, sans fuite (validation croisée temporelle).",
          "Modèles dédiés par classe d'actif : un modèle propre à la crypto, au forex, aux matières premières, aux actions FR (avec son proxy de régime : Bitcoin pour la crypto, indice dollar pour le forex).",
          "Rankers long/short : versions « régression » pour classer les actifs entre eux (voir §11).",
        ] },
      ],
    },
    {
      id: "training",
      title: "7. Entraînement, validation et choix du modèle",
      blocks: [
        { kind: "p", text: "On utilise une validation « walk-forward » : on entraîne sur le passé, on prédit sur une fenêtre future jamais vue, on avance, et on répète. Un embargo (égal à l'horizon) sépare entraînement et test pour éviter tout chevauchement. C'est ainsi qu'on obtient des métriques honnêtes hors échantillon (AUC, Sharpe)." },
        { kind: "p", text: "Le choix du modèle passe par la « guard rule » : la promotion est refusée automatiquement si le candidat ne bat pas à la fois la baseline ET le buy & hold (pour le long-only), ou s'il n'affiche pas un Sharpe positif et statistiquement significatif (Sharpe dégonflé / DSR, pour le market-neutral). Impossible de servir un modèle pire sans forçage explicite." },
      ],
    },
    {
      id: "hpo",
      title: "8. Recherche d'hyperparamètres (HPO)",
      blocks: [
        { kind: "p", text: "Avec Optuna, on cherche automatiquement les meilleurs réglages — ET les meilleures features (chaque famille peut être activée/désactivée par la recherche). L'objectif n'est pas le Sharpe brut (trop sensible au biais de sélection) mais la VRAIE compétence de classement :" },
        { kind: "ul", items: [
          "Tâche US (binaire) : on maximise l'AUC out-of-sample (pouvoir de discrimination).",
          "Long/short : on maximise le rank-IC (corrélation de rang entre prédiction et réalité).",
        ] },
        { kind: "p", text: "Les essais sont suivis dans MLflow et partagés dans une base commune entre les 2 GPU. Les meilleurs paramètres trouvés alimentent automatiquement le ré-entraînement nocturne." },
      ],
    },
    {
      id: "calibration",
      title: "9. Calibration et méta-étiquetage",
      blocks: [
        { kind: "p", text: "Le modèle brut est souvent trop optimiste. La calibration (isotonique, ajustée hors échantillon) corrige la probabilité pour qu'elle colle à la réalité : si le modèle disait 0,63 mais ne gagnait en fait que 34 % du temps à ce niveau, la probabilité affichée devient 0,34. C'est cette probabilité honnête qui décide du signal." },
        { kind: "p", text: "Méta-étiquetage (optionnel) : un second modèle juge si le signal principal est fiable et peut rétrograder un ACHAT douteux en NEUTRE — un filtre de précision." },
      ],
    },
    {
      id: "cadence",
      title: "10. Cadence de ré-entraînement (automatique)",
      blocks: [
        { kind: "p", text: "Tout tourne seul, chaque jour ouvré (heure de Paris) :" },
        { kind: "table", head: ["Heure", "Tâche"], rows: [
          ["22:30", "Données du jour + signaux + portefeuille papier"],
          ["22:45", "Rafraîchissement de l'univers long/short (~274 actifs)"],
          ["23:00", "HPO nocturne léger (params + features)"],
          ["23:30", "Ré-entraînement du zoo US (utilise les meilleurs params HPO + guard)"],
          ["23:35", "Ré-entraînement des modèles dédiés par classe d'actif"],
          ["23:45", "Génération du panier long/short"],
          ["Samedi 08:00 / 12:00", "Contrôle de dérive (drift) / HPO approfondi sur GPU"],
        ] },
      ],
    },
    {
      id: "longshort",
      title: "11. Stratégie marché-neutre (long/short)",
      blocks: [
        { kind: "p", text: "Au lieu de parier « le marché monte », on classe ~274 actions chaque jour, on achète le meilleur décile et on vend à découvert le pire, à montants égaux (dollar-neutre). On ne gagne alors que sur l'écart entre bons et mauvais titres — le mouvement global du marché est neutralisé. Rééquilibrage tous les ~5 jours, pondération inverse-vol, gate sur le Sharpe dégonflé. À ce jour, le rank-IC est ≈ 0 (pas de compétence de tri exploitable) — donc aucun edge validé, affiché comme tel." },
      ],
    },
    {
      id: "signals",
      title: "12. Comment lire les signaux (achat / stop / objectif)",
      blocks: [
        { kind: "p", text: "Pour chaque actif, le tableau de bord montre une probabilité de gain calibrée P(win) et un verdict :" },
        { kind: "table", head: ["P(win) calibrée", "Signal", "Sens"], rows: [
          ["≥ 0,35", "ACHAT (BUY)", "nettement au-dessus du point mort (~0,33) → espérance positive"],
          ["entre 0,30 et 0,35", "NEUTRE", "pas d'edge clair — ni achat ni vente franc"],
          ["≤ 0,30", "VENTE (SELL)", "sous le point mort → espérance négative"],
        ] },
        { kind: "p", text: "Point crucial : NEUTRE ne veut PAS dire « vends ». Cela signifie « pas de signal franc ». Le seuil d'achat (0,35) tient compte du payoff 2:1 (point mort ~0,33)." },
        { kind: "ul", items: [
          "Prix d'entrée : la dernière clôture.",
          "Stop-loss (où couper la perte) : entrée − 1×ATR × facteur de volatilité. Le facteur élargit le stop quand le marché est agité, le resserre quand il est calme (SL/TP adaptatifs). Quand un modèle distributionnel est disponible, le stop peut aussi venir du quantile bas (q10) du rendement prévu.",
          "Take-profit (où prendre le bénéfice) : entrée + 2×ATR × facteur de volatilité (ou quantile haut q90).",
          "Taille de position : calculée pour ne risquer qu'1 % du capital jusqu'au stop.",
          "Bande q10–q90 : quand affichée, c'est l'éventail de rendements prévus (incertitude).",
        ] },
      ],
    },
    {
      id: "honesty",
      title: "13. Honnêteté & limites",
      blocks: [
        { kind: "p", text: "BeRich ne prétend pas avoir un edge. La meilleure discrimination US actuelle (AUC ≈ 0,536) reste faible (0,5 = hasard), et rien ne bat le buy & hold sur le Sharpe. La microstructure est le levier de features le plus prometteur à ce jour ; les fondamentaux ne sont pas validables faute d'historique. Les signaux sont des aides à la décision, PAS des conseils financiers. La bannière « indicatif / expérimental » est affichée partout, et la guard rule empêche toute promotion non méritée." },
      ],
    },
  ],
  en: [
    {
      id: "overview",
      title: "1. Overview",
      blocks: [
        { kind: "p", text: "BeRich estimates, per asset, a trend probability over a few-day horizon (no intraday): the chance the upper barrier (target) is hit before the lower one (stop) within N days. It does NOT predict a price." },
        { kind: "p", text: "Non-negotiable guard rule: a model is only promoted (served) if it beats both the LightGBM baseline AND simple buy & hold, walk-forward, out-of-sample, with realistic fees + slippage. Until then everything is shown as advisory. No model clears that bar today — on purpose, and surfaced honestly." },
      ],
    },
    {
      id: "data",
      title: "2. Data acquisition",
      blocks: [
        { kind: "p", text: "Everything starts from daily bars, fetched and cached locally (Parquet)." },
        { kind: "ul", items: [
          "OHLCV prices (open, high, low, close, volume) via yfinance, since 2010, split/dividend-adjusted.",
          "Universe: ~274 US stocks (10 mega + ~113 mid + ~151 small), plus 8 FR stocks, 5 FX pairs, 3 cryptos, 3 commodities.",
          "Earnings calendar: past and future dates, EPS surprises.",
          "News via Alpha Vantage, then sentiment-scored by FinBERT (see §4).",
          "Quarterly fundamentals (revenue, net income, assets, equity, debt) — short history (~5y on yfinance), so forward-looking use.",
        ] },
      ],
    },
    {
      id: "features",
      title: "3. Feature engineering",
      blocks: [
        { kind: "p", text: "From prices we compute causal indicators (only the past up to date t — never the future). The base set has 22 features; optional families add on." },
        { kind: "table", head: ["Family", "Examples", "Captures"], rows: [
          ["Base — momentum", "1/5d returns, 10/20/60/120d momentum", "trend / drift"],
          ["Base — oscillators", "RSI 14, MACD (line/signal/histogram)", "overbought / oversold"],
          ["Base — volatility", "ATR%, 20d realized vol", "risk regime"],
          ["Base — trend/MA", "gap to 20/50d averages, distance to 60d highs/lows", "relative position"],
          ["Base — volume / calendar / regime", "volume z-score, seasonality, market (SPY) return+vol", "liquidity, calendar effects, market context"],
          ["Microstructure (new)", "CLV, opening gap, range, Parkinson, Amihud illiquidity, Roll spread", "liquidity, spread, intraday position"],
          ["Earnings (opt.)", "days to/from earnings, surprises", "event pressure"],
          ["News / FinBERT (opt.)", "mean/extreme sentiment, price-sentiment divergence", "news tone"],
          ["Fundamentals (opt., point-in-time)", "net margin, ROE, debt/equity, revenue growth", "quality / valuation"],
        ] },
        { kind: "p", text: "Important: scalers are fit on the training portion only, never the test set — no information leakage. The hyperparameter search (§8) itself decides which families to keep; the active US model currently uses momentum + calendar + microstructure." },
      ],
    },
    {
      id: "finbert",
      title: "4. FinBERT (news sentiment)",
      blocks: [
        { kind: "p", text: "FinBERT is a finance-specialized language model. Each news item (title + summary) is classified as negative / neutral / positive on GPU. From this we derive a sentiment score and features (5-day mean sentiment, intensity, divergence from price). This is the only place a large-language-model-style AI is used; the rest is tabular/sequence models." },
      ],
    },
    {
      id: "labels",
      title: "5. How we define 'winning' (labeling)",
      blocks: [
        { kind: "p", text: "Triple-barrier method: each day, open a hypothetical long and watch the next 10 days. Target = entry + 2×ATR; stop = entry − 1×ATR; time barrier = 10 days." },
        { kind: "ul", items: [
          "Hits the target first → label 1 (win).",
          "Hits the stop first → label 0 (loss).",
          "Neither within 10 days → resolved by the sign of the final return.",
        ] },
        { kind: "p", text: "The model learns P(win) = probability of hitting the target before the stop. Because the reward:risk is 2:1, breakeven is not 50% but ~33%: above a 33% win chance, expectancy is positive." },
      ],
    },
    {
      id: "models",
      title: "6. The models",
      blocks: [
        { kind: "p", text: "All share a common interface (fit / predict-probability), so they're interchangeable." },
        { kind: "ul", items: [
          "LightGBM: gradient-boosted trees (CPU, fast) — the baseline and the currently-served US model.",
          "LSTM, PatchTST (transformer), TFT: GPU deep nets (pure PyTorch), reading recent history as sequences.",
          "Stacking ensemble: a meta-model combines base-model views, leak-free (time-series CV).",
          "Per-asset-class dedicated models: a model each for crypto, FX, commodities, FR stocks (with its own regime proxy: Bitcoin for crypto, dollar index for FX).",
          "Long/short rankers: regression variants that rank assets against each other (see §11).",
        ] },
      ],
    },
    {
      id: "training",
      title: "7. Training, validation and model selection",
      blocks: [
        { kind: "p", text: "We use walk-forward validation: train on the past, predict on a never-seen forward window, advance, repeat. An embargo (equal to the horizon) separates train and test to avoid overlap. This yields honest out-of-sample metrics (AUC, Sharpe)." },
        { kind: "p", text: "Model selection goes through the guard rule: promotion is automatically refused if the candidate does not beat both the baseline AND buy & hold (long-only), or lacks a positive, statistically significant Sharpe (deflated Sharpe / DSR, market-neutral). A worse model cannot be served without an explicit override." },
      ],
    },
    {
      id: "hpo",
      title: "8. Hyperparameter search (HPO)",
      blocks: [
        { kind: "p", text: "With Optuna we automatically search the best settings — AND the best features (each family can be toggled on/off by the search). The objective is not raw Sharpe (too prone to selection bias) but genuine ranking skill:" },
        { kind: "ul", items: [
          "US (binary) task: maximize out-of-sample AUC (discrimination power).",
          "Long/short: maximize rank-IC (rank correlation between prediction and reality).",
        ] },
        { kind: "p", text: "Trials are tracked in MLflow and shared in a common store across the 2 GPUs. The best params found automatically feed the nightly retrain." },
      ],
    },
    {
      id: "calibration",
      title: "9. Calibration and meta-labeling",
      blocks: [
        { kind: "p", text: "The raw model is often over-confident. Calibration (isotonic, fit out-of-sample) corrects the probability to match reality: if the model said 0.63 but only won 34% of the time at that level, the displayed probability becomes 0.34. This honest probability is what drives the signal." },
        { kind: "p", text: "Meta-labeling (optional): a second model judges whether the primary signal is reliable and can downgrade a doubtful BUY to NEUTRAL — a precision filter." },
      ],
    },
    {
      id: "cadence",
      title: "10. Retraining cadence (automatic)",
      blocks: [
        { kind: "p", text: "Everything runs on its own, every weekday (Paris time):" },
        { kind: "table", head: ["Time", "Job"], rows: [
          ["22:30", "Today's data + signals + paper book"],
          ["22:45", "Long/short universe refresh (~274 assets)"],
          ["23:00", "Light nightly HPO (params + features)"],
          ["23:30", "US zoo retrain (uses best HPO params + guard)"],
          ["23:35", "Per-asset-class dedicated models retrain"],
          ["23:45", "Long/short basket generation"],
          ["Sat 08:00 / 12:00", "Drift check / deep GPU HPO"],
        ] },
      ],
    },
    {
      id: "longshort",
      title: "11. Market-neutral (long/short) strategy",
      blocks: [
        { kind: "p", text: "Instead of betting 'the market goes up', we rank ~274 stocks each day, buy the top decile and short the bottom one in equal amounts (dollar-neutral). We then only earn on the spread between good and bad names — the overall market move is neutralized. Rebalanced every ~5 days, inverse-vol weighted, gated on the deflated Sharpe. So far rank-IC is ≈ 0 (no exploitable ranking skill) — no validated edge, shown as such." },
      ],
    },
    {
      id: "signals",
      title: "12. How to read the signals (buy / stop / target)",
      blocks: [
        { kind: "p", text: "For each asset the dashboard shows a calibrated win probability P(win) and a verdict:" },
        { kind: "table", head: ["Calibrated P(win)", "Signal", "Meaning"], rows: [
          ["≥ 0.35", "BUY", "clearly above breakeven (~0.33) → positive expectancy"],
          ["between 0.30 and 0.35", "NEUTRAL", "no clear edge — neither a buy nor a sell"],
          ["≤ 0.30", "SELL", "below breakeven → negative expectancy"],
        ] },
        { kind: "p", text: "Crucial: NEUTRAL does NOT mean 'sell'. It means 'no clear signal'. The buy threshold (0.35) accounts for the 2:1 payoff (breakeven ~0.33)." },
        { kind: "ul", items: [
          "Entry price: the last close.",
          "Stop-loss (where to cut the loss): entry − 1×ATR × volatility factor. The factor widens the stop when the market is turbulent, tightens it when calm (adaptive SL/TP). When a distributional model is available, the stop can also come from the low quantile (q10) of the predicted return.",
          "Take-profit (where to take the gain): entry + 2×ATR × volatility factor (or high quantile q90).",
          "Position size: sized to risk only 1% of capital to the stop.",
          "q10–q90 band: when shown, it's the predicted range of returns (uncertainty).",
        ] },
      ],
    },
    {
      id: "honesty",
      title: "13. Honesty & limits",
      blocks: [
        { kind: "p", text: "BeRich does not claim an edge. The best current US discrimination (AUC ≈ 0.536) is still weak (0.5 = chance), and nothing beats buy & hold on Sharpe. Microstructure is the most promising feature lever so far; fundamentals can't be validated for lack of history. Signals are decision aids, NOT financial advice. The 'advisory / experimental' banner is everywhere, and the guard rule prevents any undeserved promotion." },
      ],
    },
  ],
};

type Stage = { title: string; sub?: string; chips?: string[]; note?: string };

const DIAGRAM: Record<Locale, Stage[]> = {
  fr: [
    { title: "Données", chips: ["OHLCV (prix)", "Résultats", "News → FinBERT", "Fondamentaux"] },
    { title: "Feature engineering", sub: "22 de base + microstructure + familles optionnelles — toutes causales" },
    { title: "Labellisation triple-barrière", sub: "P(gain) — objectif +2×ATR / stop −1×ATR · point mort ~33 %" },
    {
      title: "Entraînement walk-forward — zoo de modèles",
      sub: "LightGBM · LSTM · PatchTST · TFT · ensemble · modèles par classe d'actif",
      note: "⟵ HPO (Optuna : objectif AUC / rank-IC + recherche de features) règle params et features",
    },
    { title: "Guard rule", sub: "bat la baseline ET le buy & hold ? → promotion · sinon on garde l'ancien" },
    { title: "Calibration + filtre méta", sub: "probabilité honnête + filtre de précision optionnel" },
    {
      title: "Signaux",
      sub: "stop-loss & take-profit adaptatifs à la volatilité · sizing à 1 % de risque",
      chips: ["ACHAT ≥ 0,35", "NEUTRE", "VENTE ≤ 0,30"],
    },
    { title: "Tableau de bord + portefeuille papier" },
  ],
  en: [
    { title: "Data", chips: ["OHLCV (prices)", "Earnings", "News → FinBERT", "Fundamentals"] },
    { title: "Feature engineering", sub: "base 22 + microstructure + optional families — all causal" },
    { title: "Triple-barrier labeling", sub: "P(win) — target +2×ATR / stop −1×ATR · breakeven ~33%" },
    {
      title: "Walk-forward training — model zoo",
      sub: "LightGBM · LSTM · PatchTST · TFT · ensemble · per-asset-class models",
      note: "⟵ HPO (Optuna: AUC / rank-IC objective + feature search) tunes params and features",
    },
    { title: "Guard rule", sub: "beats baseline AND buy & hold? → promote · else keep the previous one" },
    { title: "Calibration + meta filter", sub: "honest probability + optional precision filter" },
    {
      title: "Signals",
      sub: "volatility-adaptive stop-loss & take-profit · sized to 1% risk",
      chips: ["BUY ≥ 0.35", "NEUTRAL", "SELL ≤ 0.30"],
    },
    { title: "Dashboard + paper book" },
  ],
};

function PipelineDiagram({ locale }: { locale: Locale }) {
  const stages = DIAGRAM[locale];
  return (
    <div className="card my-8 p-6">
      <div className="flex flex-col items-center gap-0">
        {stages.map((s, i) => (
          <div key={s.title} className="flex w-full flex-col items-center">
            <div className="w-full max-w-xl rounded-lg border border-[var(--color-line)] bg-white/[0.02] px-4 py-3 text-center">
              <div className="font-display text-sm font-bold">{s.title}</div>
              {s.sub && <div className="mt-1 text-xs text-[var(--color-muted)]">{s.sub}</div>}
              {s.chips && (
                <div className="mt-2 flex flex-wrap justify-center gap-1.5">
                  {s.chips.map((c) => (
                    <span
                      key={c}
                      className="rounded border border-[var(--color-line)] bg-[var(--color-line)]/[0.06] px-2 py-0.5 text-[11px] text-[var(--color-muted)]"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              )}
              {s.note && (
                <div className="mt-1 text-[11px] italic text-[var(--color-faint)]">{s.note}</div>
              )}
            </div>
            {i < stages.length - 1 && (
              <span className="py-1 text-lg leading-none text-[var(--color-faint)]">↓</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function renderBlock(b: Block, i: number) {
  if (b.kind === "p") {
    return (
      <p key={i} className="mt-2 text-sm leading-relaxed text-[var(--color-muted)]">
        {b.text}
      </p>
    );
  }
  if (b.kind === "ul") {
    return (
      <ul key={i} className="mt-2 list-disc space-y-1 pl-5 text-sm text-[var(--color-muted)]">
        {b.items.map((it, j) => (
          <li key={j}>{it}</li>
        ))}
      </ul>
    );
  }
  return (
    <div key={i} className="mt-3 overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-[var(--color-line)] text-left text-[var(--color-faint)]">
            {b.head.map((h) => (
              <th key={h} className="px-3 py-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {b.rows.map((row, r) => (
            <tr key={r} className="border-b border-[var(--color-line)]/40 last:border-0">
              {row.map((cell, c) => (
                <td key={c} className="px-3 py-2 align-top text-[var(--color-muted)]">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function UnderTheHood() {
  const { locale, t } = useI18n();
  const sections = CONTENT[locale];
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-bull)]"
      >
        ← {t("hood.back")}
      </Link>
      <h1 className="font-display text-4xl font-extrabold tracking-tight">{t("hood.title")}</h1>
      <p className="mt-3 text-base text-[var(--color-muted)]">{t("hood.intro")}</p>

      <PipelineDiagram locale={locale} />

      <nav className="card my-8 p-5">
        <div className="mb-2 text-xs uppercase tracking-widest text-[var(--color-faint)]">
          {t("hood.toc")}
        </div>
        <ol className="grid gap-1 text-sm sm:grid-cols-2">
          {sections.map((s) => (
            <li key={s.id}>
              <a href={`#${s.id}`} className="text-[var(--color-muted)] hover:text-[var(--color-bull)]">
                {s.title}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <div className="flex flex-col gap-6">
        {sections.map((s) => (
          <section key={s.id} id={s.id} className="card scroll-mt-6 p-5">
            <h2 className="font-display text-lg font-bold">{s.title}</h2>
            {s.blocks.map(renderBlock)}
          </section>
        ))}
      </div>
    </main>
  );
}
