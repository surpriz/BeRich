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
    date: "2026-06-09",
    tone: "ship",
    title: {
      fr: "Panel diversifié : un 2e livre papier qui couvre toutes les classes (US/FR/crypto/matières/forex)",
      en: "Diversified panel: a 2nd paper book spanning all classes (US/FR/crypto/commodities/forex)",
    },
    points: [
      {
        fr: "Constat mesuré : le livre engagé est dominé par le forex car le gate des longs exige de battre le buy & hold de l'actif — barre quasi nulle sur le forex (Sharpe B&H +0,11), très haute sur les actions US (+0,72) et FR (+0,50). Une action en tendance se bat difficilement par un modèle quotidien ; le gate refuse à juste titre de prétendre le contraire.",
        en: "Measured finding: the committed book is forex-dominated because the long gate must beat the asset's buy & hold — a near-flat bar on forex (B&H Sharpe +0.11), a steep one on US (+0.72) and FR (+0.50) stocks. A daily model rarely beats a trending stock; the gate rightly refuses to pretend otherwise.",
      },
      {
        fr: "Nouvelle page « Panel diversifié » : surface le tier « observe » (modèles presque-promus, toutes classes) comme un vrai 2e livre, sur capital simulé séparé de 10 000 €. P&L, courbe, positions par classe (la variété rendue visible) et liste de réplication propre.",
        en: "New 'Diversified panel' page: surfaces the 'observe' tier (near-miss models, all classes) as a real 2nd book on its own separate 10,000 € simulated capital. P&L, equity curve, positions by class (variety made visible) and its own replication list.",
      },
      {
        fr: "Le livre engagé (forward test rigoureux des ~30 trades) reste totalement intact et séparé. Le panel est explicitement étiqueté « exploratoire, confiance moindre » : à suivre pour la variété, pas comme conseil validé.",
        en: "The committed book (rigorous ~30-trade forward test) stays fully intact and separate. The panel is explicitly labelled 'exploratory, lower confidence': follow it for variety, not as validated advice.",
      },
    ],
    verdict: {
      fr: "Variété immédiate sans attendre la révision du gate (planifiée après le forward test) et sans toucher au test rigoureux. Compromis assumé : plus de classes, edge plus faible.",
      en: "Immediate variety without waiting for the gate revision (planned after the forward test) and without touching the rigorous test. Accepted trade-off: more classes, weaker edge.",
    },
  },
  {
    date: "2026-06-09",
    tone: "infra",
    title: {
      fr: "Intraday V2 (POC) : signaux crypto 1h en direct, livre papier séparé",
      en: "Intraday V2 (POC): live 1-hour crypto signals, a separate paper book",
    },
    points: [
      {
        fr: "Nouveau sous-système intraday entièrement parallèle au swing journalier : barres horaires (1h) sur une seule paire, BTC/USDT, données profondes et gratuites depuis Binance (via ccxt). Le pipeline journalier déployé n'est pas modifié — cache, base de données et modèles intraday vivent dans leurs propres espaces (data/ohlcv_1h, berich_intraday.duckdb, .../<côté>/1h).",
        en: "A new intraday subsystem fully parallel to the daily swing book: hourly (1h) bars on a single pair, BTC/USDT, deep free history from Binance (via ccxt). The deployed daily pipeline is untouched — the intraday cache, database and models live in their own namespaces (data/ohlcv_1h, berich_intraday.duckdb, .../<side>/1h).",
      },
      {
        fr: "Même garde-fou promote() que le swing, sans dérogation : un modèle 1h n'est promu que s'il bat l'achat-conservation de BTC (long) ou affiche un Sharpe positif et significatif (short), avec ≥ 20 trades hors-échantillon, des frais Binance réalistes (~0,10 %/côté) et une annualisation honnête (bars_per_year = 8760 = 24×365, et non 252 — sinon le Sharpe serait gonflé ~6×).",
        en: "The same promote() guard as the swing book, no bypass: a 1h model is promoted only if it beats BTC buy-&-hold (long) or shows a positive, significant Sharpe (short), with ≥ 20 out-of-sample trades, realistic Binance fees (~0.10%/side) and honest annualization (bars_per_year = 8760 = 24×365, not 252 — which would inflate the Sharpe ~6×).",
      },
      {
        fr: "Un job horaire (24/7, crypto ne ferme jamais) rafraîchit les données, génère le signal 1h, ouvre les trades promus sous les mêmes plafonds de money-management et fait évoluer/ferme les positions. Nouvelle page /intraday : signal courant, ordres PRÉVUS (prévision, à ne pas copier), EXÉCUTÉ (ce que le robot a réellement fait), et P&L papier face à l'achat-conservation BTC — la même séparation prévision/exécuté que /brief et /copy.",
        en: "An hourly job (24/7 — crypto never closes) refreshes data, generates the 1h signal, opens promoted trades under the same money-management caps, and trails/closes positions. New /intraday page: current signal, FORECAST orders (a forecast, not to copy), EXECUTED (what the robot actually did), and paper P&L vs BTC buy-&-hold — the same forecast-vs-executed split as /brief and /copy.",
      },
    ],
    verdict: {
      fr: "POC — pas une preuve d'edge. On mesure si un edge intraday passe le même garde-fou après frais. Capital papier uniquement, aucun argent réel ; le swing journalier reste gelé et inchangé. Si l'edge 1h ne clear pas, il reste « advisory » et n'engage aucun capital.",
      en: "POC — not an edge claim. It measures whether an intraday edge clears the same guard after costs. Paper capital only, no real money; the daily swing book stays frozen and unchanged. If the 1h edge doesn't clear, it stays advisory and engages no capital.",
    },
  },
  {
    date: "2026-06-08",
    tone: "ship",
    title: {
      fr: "Email quotidien repensé : un vrai digest du matin, clair et bilingue",
      en: "Daily email reworked: a real morning digest, clear and bilingual",
    },
    points: [
      {
        fr: "L'email part désormais chaque jour de bourse, plus seulement les soirs où un nouvel ordre est ouvert. Vous recevez un point fiable même un jour calme. (Auparavant : aucun email les jours sans nouvel achat, d'où l'impression d'un envoi manquant.)",
        en: "The email now goes out every trading day, not only on evenings when a new order opens. You get a dependable briefing even on a quiet day. (Before: no email on days without a new buy — hence the feeling of a missing send.)",
      },
      {
        fr: "Contenu enrichi en quatre sections : Portefeuille (équité, P&L vs SPY, drawdown, positions), Ordres à copier (la liste exacte de ce qui a été exécuté, avec stops trailing à recopier), Activité du jour (clôtures et P&L réalisé), et À savoir (positions en MTM, alertes données périmées, nombre de modèles promus, rappel prévision vs exécuté).",
        en: "Four richer sections: Portfolio (equity, P&L vs SPY, drawdown, open positions), Orders to copy (the exact list of what was executed, with trailing stops to mirror), Today's activity (closes and realized P&L), and Good to know (MTM positions, stale-data alerts, promoted-model count, forecast-vs-executed reminder).",
      },
      {
        fr: "Mise en page soignée aux couleurs « Clarté » (accent indigo, gains émeraude / pertes corail) et entièrement bilingue FR/EN. L'ancien avertissement périmé « le modèle ne bat pas le buy & hold » est remplacé par la mention honnête « paper-trading en forward test, pas de capital réel ».",
        en: "Polished layout in the “Clarity” palette (indigo accent, emerald gains / coral losses), fully bilingual FR/EN. The stale “model does not beat buy & hold” disclaimer is replaced by the honest “paper-trading forward test, no real money”.",
      },
    ],
    verdict: {
      fr: "Changement de notification et d'affichage uniquement — aucune logique de trading, de gate ou de sizing modifiée. La liste « à copier » partage sa source avec la page Réplication. Le gel du forward test est respecté.",
      en: "Notification and presentation only — no trading, gate or sizing logic touched. The “orders to copy” list shares its source with the Replication page. The forward-test freeze is respected.",
    },
  },
  {
    date: "2026-06-08",
    tone: "ship",
    title: {
      fr: "Refonte de l'interface : thème clair « Clarté », menu allégé, pédagogie pour les novices",
      en: "Interface redesign: light “Clarity” theme, slimmer menu, novice-friendly guidance",
    },
    points: [
      {
        fr: "Nouveau design clair de type néobanque : fond papier, accent indigo, hausse en émeraude et baisse en corail, cartes arrondies à ombre douce, nouvelle police d'affichage (Schibsted Grotesk). Fini le terminal sombre au vert électrique. Graphiques repassés sur la palette claire.",
        en: "New light, neobank-style design: paper background, indigo accent, emerald gains and coral losses, soft-shadowed rounded cards, a fresh display font (Schibsted Grotesk). Gone is the dark terminal with electric lime. Charts retinted to the light palette.",
      },
      {
        fr: "Menu réorganisé en deux niveaux : la barre principale ne garde que le quotidien — Aujourd'hui, Portefeuille, Opportunités — et tout le technique (Stratégies, Entraînement, Machine, Journal, Aide) passe sous un menu déroulant « Coulisses ». Le Brief absorbe la Réplication via une bascule « Positions & prévisions / À répliquer ce matin ».",
        en: "Menu reorganized into two tiers: the main bar keeps only the everyday — Today, Portfolio, Opportunities — and all the technical pages (Strategies, Training, Machine, Journal, Help) move under a “Backstage” dropdown. The Brief absorbs Replication via a “Positions & forecast / Copy this morning” toggle.",
      },
      {
        fr: "Pédagogie pour les débutants : un bandeau pliable « C'est quoi cette page ? » en clair en haut de chaque page, les niveaux renommés Découverte / Investisseur / Analyste (jargon traduit en Découverte : P(win) → « Confiance »…), des infobulles sur les en-têtes techniques, et une courte visite guidée au premier passage (rejouable depuis l'Aide).",
        en: "Beginner guidance: a collapsible plain-language “What is this page?” banner atop every page, levels renamed Discovery / Investor / Analyst (jargon translated in Discovery: P(win) → “Confidence”…), tooltips on technical headers, and a short first-run guided tour (replayable from Help).",
      },
    ],
    verdict: {
      fr: "Changement purement cosmétique et pédagogique — aucune logique de trading, de gate ou de sizing modifiée. Le gel du forward test est respecté.",
      en: "Purely cosmetic and pedagogical — no trading, gate or sizing logic touched. The forward-test freeze is respected.",
    },
  },
  {
    date: "2026-06-07",
    tone: "infra",
    title: {
      fr: "Univers élargi : 100 → 125 actifs, surpondération là où l'edge survit",
      en: "Universe expanded: 100 → 125 assets, overweighted where the edge survives",
    },
    points: [
      {
        fr: "Forex +8 (EURAUD, EURCAD, CADJPY, CHFJPY, NZDJPY, GBPAUD, GBPCHF, EURNZD) — surpondéré exprès : les survivants FDR se concentrent sur le forex et les crosses JPY portent 3 livres promus chacun. Crypto +4 (XLM, ETC, NEAR, ALGO) et matières premières +3 (Brent, cacao, bétail) — les shorts y étaient promus 5/5.",
        en: "Forex +8 (EURAUD, EURCAD, CADJPY, CHFJPY, NZDJPY, GBPAUD, GBPCHF, EURNZD) — overweighted on purpose: FDR survivors cluster on forex and the JPY crosses carry 3 promoted books each. Crypto +4 (XLM, ETC, NEAR, ALGO) and commodities +3 (Brent, cocoa, live cattle) — shorts there were promoted 5/5.",
      },
      {
        fr: "US +6 (Eli Lilly, Texas Instruments, Qualcomm, Caterpillar, Goldman Sachs, IBM) et France +4 (Thales, Saint-Gobain, Orange, Legrand) — secteurs non couverts, liquides. Les 25 validés sur yfinance avant ajout : plus de 2 000 barres quotidiennes chacun, données fraîches.",
        en: "US +6 (Eli Lilly, Texas Instruments, Qualcomm, Caterpillar, Goldman Sachs, IBM) and France +4 (Thales, Saint-Gobain, Orange, Legrand) — uncovered, liquid sectors. All 25 validated on yfinance before adding: 2,000+ daily bars each, fresh data.",
      },
      {
        fr: "Le sweep continu (désormais 750 combos actif × sens × stratégie) traite les combos jamais explorés EN PREMIER : les 25 nouveaux reçoivent leur recherche profonde (100 essais HPO) avant tout ré-approfondissement des anciens. Redémarré pour effet immédiat — combo 1/750 : LLY/long/fixed. Verdicts attendus sous ~2-4 jours sur /training.",
        en: "The continuous sweep (now 750 ticker × side × strategy combos) processes never-searched combos FIRST: the 25 newcomers get their deep search (100 HPO trials) before any re-deepening of incumbents. Restarted for immediate effect — combo 1/750: LLY/long/fixed. Verdicts expected on /training within ~2-4 days.",
      },
      {
        fr: "Le forward test n'est pas pollué : le filet FDR au redémarrage a confirmé 46 promus → 0 démotion, le livre engagé ne bouge pas, et un nouvel actif ne peut entrer qu'en passant le même gate durci (DSR ≥ 0.95, ≥ 20 trades, FDR) que les autres.",
        en: "The forward test is not polluted: the FDR net at restart confirmed 46 promoted → 0 demotions, the committed book is untouched, and a newcomer can only enter by clearing the same hardened gate (DSR ≥ 0.95, ≥ 20 trades, FDR) as everyone else.",
      },
    ],
    verdict: {
      fr: "Plus de billets de loterie honnêtes, pas de promesse : la plupart des 25 resteront advisory-only — c'est le résultat attendu du gate, pas un échec. Ceux qui survivent rejoindront le livre observation puis, peut-être, le livre engagé.",
      en: "More honest lottery tickets, no promise: most of the 25 will stay advisory-only — that is the gate's expected outcome, not a failure. Survivors will join the observation book and then, maybe, the committed book.",
    },
  },
  {
    date: "2026-06-06",
    tone: "research",
    title: {
      fr: "Verdict du bake-off : PEAD et market-neutral enterrés — cap sur les 47 survivants",
      en: "Bake-off verdict: PEAD and market-neutral buried — focus on the 47 survivors",
    },
    points: [
      {
        fr: "Gate des longs durci (Sharpe significatif exigé, pas seulement « bat le B&H ») + FDR rendu réellement périodique : les 119 promus gonflés sont retombés à 47 survivants honnêtes (forex 26, shorts crypto 5/5 et matières premières 5/5), stables sous FDR.",
        en: "Long gate hardened (significant Sharpe required, not just beats-B&H) + FDR made truly periodic: the inflated 119 promotions fell to 47 honest survivors (forex 26, crypto shorts 5/5, commodity shorts 5/5), FDR-stable.",
      },
      {
        fr: "Portefeuille papier remis à zéro sur bases saines (seule la position validée par la nouvelle stratégie conservée) ; le Brief affiche désormais le plan réellement ouvrable (tailles post-plafonds) + les positions à conserver + un encart « Comment lire ».",
        en: "Paper portfolio reset to clean foundations (only the new-strategy-validated position kept); the Brief now shows the actually-openable plan (post-cap sizes) + held positions + a 'How to read' explainer.",
      },
      {
        fr: "Bake-off pré-enregistré sous le harnais durci — PEAD : 6 175 événements, AUC 0.53-0.55, Sharpe 0.83 (5 j) / 1.10 (20 j) MAIS sous son benchmark de fenêtre (1.01 / 1.31) → échec. Market-neutral : rank-IC ≈ 0 (t = 0.26), aucun pouvoir de classement → échec.",
        en: "Pre-registered bake-off under the hardened harness — PEAD: 6,175 events, AUC 0.53-0.55, Sharpe 0.83 (5d) / 1.10 (20d) BUT below its window benchmark (1.01 / 1.31) → fail. Market-neutral: rank-IC ≈ 0 (t = 0.26), zero ranking power → fail.",
      },
      {
        fr: "Alerte de dérive hebdo remplacée : les tests de distribution sur features journalières crient au loup structurellement (3 calibrages testés, 31/31 alertent) — l'email ne signale plus que les vrais problèmes de données (cache périmé, prix figés). La vraie protection : ré-entraînement continu + gate + FDR.",
        en: "Weekly drift alert replaced: distribution tests on daily features cry wolf structurally (3 calibrations tested, 31/31 alert) — the email now only flags real data problems (stale cache, frozen prices). The real protection: continuous retraining + gate + FDR.",
      },
    ],
    verdict: {
      fr: "Deux des trois pistes d'edge éliminées définitivement, sur règle pré-enregistrée. La direction restante : le forward test des 47 survivants (forex systématique + shorts crypto/matières premières) — décision de concentration à ~30 trades fermés par segment.",
      en: "Two of the three edge directions definitively eliminated, per the pre-registered rule. The remaining direction: the forward test of the 47 survivors (systematic forex + crypto/commodity shorts) — concentration decision at ~30 closed trades per segment.",
    },
  },
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
        className="mb-6 inline-flex items-center gap-2 text-sm text-[var(--color-muted)] hover:text-[var(--color-accent)]"
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
                <span className="rounded bg-[var(--color-surface-2)] px-2 py-0.5 text-xs font-semibold text-[var(--color-muted)]">
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
