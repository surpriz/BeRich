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
    date: "2026-06-10",
    tone: "infra",
    title: {
      fr: "/ops affiche l'espace disque restant",
      en: "/ops shows remaining disk space",
    },
    points: [
      {
        fr: "La carte « Système » de /ops affichait l'espace disque utilisé (ex. 95 / 1830 Go) mais pas la place encore libre. Une ligne « Espace libre » indique désormais directement les Go restants sur le disque de données (le volume où grandissent les modèles et la base d'optimisation Optuna), avec le même code couleur que la barre d'usage.",
        en: "The /ops 'System' card showed disk space used (e.g. 95 / 1830 GB) but not how much was still free. A new 'Free space' row now reports the remaining GB directly on the data disk (the volume where models and the Optuna optimization database grow), color-coded like the usage bar.",
      },
    ],
    verdict: {
      fr: "Affichage uniquement — aucun modèle, gate, sizing ni donnée de trading touché. La machine n'a qu'un seul disque de données utile (~1,8 To) ; c'est lui qui est suivi.",
      en: "Display only — no model, gate, sizing or trading data touched. The machine has a single meaningful data disk (~1.8 TB); that is the one tracked.",
    },
  },
  {
    date: "2026-06-10",
    tone: "ship",
    title: {
      fr: "Studio : le script vidéo quotidien, généré automatiquement depuis les faits",
      en: "Studio: the daily video script, auto-generated from the facts",
    },
    points: [
      {
        fr: "Nouvelle page /studio : chaque jour, un texte prêt-à-dire pour une vidéo — état du portefeuille (gains ET pertes, nets de frais), ce que le robot a réellement exécuté au dernier pointage, le « pourquoi » de chaque trade traduit en langage simple depuis les facteurs du modèle (contributions SHAP), les positions en cours, une leçon de discipline qui tourne chaque jour, et le disclaimer obligatoire. Bouton « Copier le script ».",
        en: "New /studio page: every day, a ready-to-read text for a video — portfolio state (gains AND losses, net of fees), what the robot actually executed at the last run, the 'why' of each trade translated to plain language from the model's factors (SHAP contributions), open positions, a rotating discipline lesson, and the mandatory disclaimer. 'Copy script' button.",
      },
      {
        fr: "Même contrat d'honnêteté que le reste de la plateforme : le script ne raconte que des FAITS (exécutions, P&L réalisé) — jamais les prévisions recalculées en continu — et rappelle systématiquement que le système est en test, sans argent réel, et que rien n'est un conseil en investissement.",
        en: "Same honesty contract as the rest of the platform: the script narrates FACTS only (executions, realized P&L) — never the continuously recomputed forecasts — and systematically states that the system is under test, with no real money, and that nothing is investment advice.",
      },
    ],
    verdict: {
      fr: "Outil de communication uniquement — aucun modèle, gate, sizing ni donnée de trading touché. Préparation du second objectif du projet (documenter publiquement le robot), dont la monétisation reste conditionnée au track record ET à la vérification du cadre réglementaire (AMF).",
      en: "Communication tool only — no model, gate, sizing or trading data touched. Groundwork for the project's second goal (publicly documenting the robot), whose monetization remains conditional on the track record AND on checking the regulatory framework (AMF).",
    },
  },
  {
    date: "2026-06-10",
    tone: "infra",
    title: {
      fr: "Machine mieux utilisée : GPU et CPU travaillent ensemble, recherches ciblées, essais sans espoir coupés tôt",
      en: "Better machine utilization: GPU and CPU work together, targeted searches, hopeless trials cut early",
    },
    points: [
      {
        fr: "Chevauchement CPU/GPU : jusqu'ici, pour chaque actif, la recherche LightGBM (CPU) tournait seule pendant que les 2 GPU restaient inactifs, puis les modèles profonds prenaient les GPU pendant que le CPU attendait. Désormais les deux phases tournent en même temps — les GPU démarrent immédiatement et LightGBM est cherché pendant qu'ils travaillent.",
        en: "CPU/GPU overlap: until now, for each asset, the LightGBM search (CPU) ran alone while both GPUs sat idle, then the deep models took the GPUs while the CPU waited. Both phases now run at the same time — the GPUs start immediately and LightGBM is searched while they work.",
      },
      {
        fr: "Recherches ciblées en semaine : sur un actif dont le champion est connu (ex. LightGBM gagne depuis des semaines), re-payer chaque nuit 3 entraînements profonds pour les perdants était le principal gaspillage GPU. En semaine, le rafraîchissement ne re-cherche plus que le champion + LightGBM (le challenger bon marché) ; le week-end, la compétition complète des 4 modèles est rouverte pour que personne ne soit enterré à tort. Réglable (`topup_winner_only`).",
        en: "Targeted weekday searches: on an asset whose champion is known (e.g. LightGBM has won for weeks), re-paying 3 deep-model trainings every night for the losers was the main GPU waste. On weekdays the refresh now re-searches only the champion + LightGBM (the cheap challenger); on weekends the full 4-model contest re-opens so nobody is wrongly buried. Configurable (`topup_winner_only`).",
      },
      {
        fr: "Élagage des essais sans espoir : pendant l'optimisation d'un modèle profond, chaque essai paie plusieurs entraînements complets (un par segment de validation). Le score partiel est maintenant rapporté après chaque segment et un essai manifestement sous la médiane est arrêté tôt (MedianPruner) — 30 à 50 % de GPU économisé sur les mauvais essais, sans toucher aux bons. Les essais élagués comptent toujours dans la correction anti-chance (Deflated Sharpe) du gate.",
        en: "Pruning hopeless trials: while optimizing a deep model, each trial pays several full trainings (one per validation segment). The partial score is now reported after each segment and a trial clearly below the median is stopped early (MedianPruner) — 30-50% of GPU saved on bad trials, without touching good ones. Pruned trials still count in the gate's anti-luck correction (Deflated Sharpe).",
      },
    ],
    verdict: {
      fr: "Ordonnancement et débit uniquement — les critères du gate, les labels, le sizing et le comptage des essais du Deflated Sharpe sont inchangés (le comptage couvre toujours le zoo complet, même quand la recherche est restreinte). Objectif : plus de combos rafraîchis par jour sur la même machine, donc des modèles plus frais.",
      en: "Scheduling and throughput only — gate criteria, labels, sizing and the Deflated Sharpe trial counting are unchanged (counting still covers the full zoo even when the search is narrowed). Goal: more combos refreshed per day on the same machine, hence fresher models.",
    },
  },
  {
    date: "2026-06-10",
    tone: "risk",
    title: {
      fr: "Audit complet : le paper book paie désormais les vrais coûts, le Brief dit tout, la réforme du gate est pré-enregistrée",
      en: "Full audit: the paper book now pays real costs, the Brief tells everything, the gate reform is pre-registered",
    },
    points: [
      {
        fr: "Honnêteté du forward test : chaque nouveau trade papier fige à l'ouverture son coût aller-retour estimé (commissions + slippage proportionnel au volume — le même modèle de coûts que le backtest qui a promu les modèles) et le déduit du P&L réalisé à la clôture. Les trades déjà ouverts ne sont pas réécrits. Le forward test ne peut plus être « moins cher » que la promesse backtestée.",
        en: "Forward-test honesty: every new paper trade freezes its estimated round-trip cost at open (commissions + volume-proportional slippage — the same cost model as the backtest that promoted the models) and deducts it from the realized P&L at close. Already-open trades are not rewritten. The forward test can no longer be 'cheaper' than the backtested promise.",
      },
      {
        fr: "Bug corrigé dans les statistiques par trade du backtest : le slippage était compté deux fois (déjà inclus dans les prix d'exécution, puis re-soustrait), et le slippage du dernier actif de la boucle s'appliquait à tous les trades multi-actifs. Le Sharpe du gate (calculé sur la série quotidienne) n'était pas affecté — la population promue ne change pas.",
        en: "Bug fixed in the backtest's per-trade statistics: slippage was double-counted (already in the fill prices, then subtracted again), and the last asset's slippage applied to every trade in multi-asset runs. The gate's Sharpe (computed on the daily series) was unaffected — the promoted population does not change.",
      },
      {
        fr: "Le Brief dit désormais tout sur chaque ordre prévu : validité (~N jours de bourse, l'horizon du modèle), P(gain) calibrée, gain attendu net et coût estimé en points de base — en plus du prix, du stop, de l'objectif et de la taille.",
        en: "The Brief now tells everything about each planned order: shelf life (~N trading days, the model's horizon), calibrated P(win), net expected return and estimated cost in basis points — on top of price, stop, target and size.",
      },
      {
        fr: "Alerte de concentration devise : trois paires JPY ouvertes = un seul pari corrélé que les plafonds par classe ne voient pas. Le portefeuille et le digest quotidien signalent désormais toute devise portant ≥ 50 % du capital (information seulement — aucun sizing modifié pendant le gel).",
        en: "Currency-concentration warning: three open JPY crosses = one correlated bet the per-class caps can't see. The wallet and the daily digest now flag any currency carrying ≥ 50% of capital (information only — no sizing changed during the freeze).",
      },
      {
        fr: "Infra durcie (gratuit) : sauvegarde atomique (.tmp + rename) avec copie off-site optionnelle via rclone, watchdog systemd indépendant qui alerte par email si le run de 22:30 n'a pas eu lieu, et contre-vérification hebdomadaire des clôtures yfinance face à Stooq (source indépendante gratuite).",
        en: "Infra hardened (free): atomic backup (.tmp + rename) with optional off-site copy via rclone, an independent systemd watchdog that emails when the 22:30 run did not happen, and a weekly cross-check of yfinance closes against Stooq (free independent source).",
      },
      {
        fr: "Réforme du gate PRÉ-ENREGISTRÉE (docs/GATE_REFORM.md), à appliquer uniquement à la fin du forward test : le gate devra backtester la politique réellement servie (proba calibrée + seuil par actif — aujourd'hui il teste la proba brute au seuil global), fin du triple usage des OOF, MIN_TRADES 20→50, AUC_FLOOR 0,52, gaps remplis à l'open, réforme du gate long, caps par devise. Les critères sont figés maintenant pour qu'aucun choix ne soit fait après lecture des résultats.",
        en: "Gate reform PRE-REGISTERED (docs/GATE_REFORM.md), to be applied only when the forward test ends: the gate must backtest the actually-served policy (calibrated proba + per-asset threshold — today it tests raw proba at the global threshold), end of the triple OOF reuse, MIN_TRADES 20→50, AUC_FLOOR 0.52, gap-aware stop fills, long-gate reform, per-currency caps. The criteria are frozen now so no choice is made after seeing the results.",
      },
    ],
    verdict: {
      fr: "Verdict de l'audit : le socle méthodologique (causalité, walk-forward, Deflated Sharpe, FDR) est solide ; les failles trouvées sont des raffinements + 2 bugs réels, corrigés sans toucher à la population promue. Le gel du forward test est respecté : tout changement de validation attend les ~30 trades fermés.",
      en: "Audit verdict: the methodological core (causality, walk-forward, Deflated Sharpe, FDR) is sound; the findings are refinements + 2 real bugs, fixed without touching the promoted population. The forward-test freeze is honored: every validation change waits for the ~30 closed trades.",
    },
  },
  {
    date: "2026-06-10",
    tone: "infra",
    title: {
      fr: "Correctif : une position sans prix ne casse plus les pages (wallet, brief, panel)",
      en: "Fix: a position with no price no longer breaks pages (wallet, brief, panel)",
    },
    points: [
      {
        fr: "Quand le prix courant d'une position fraîchement ouverte n'a pas pu être récupéré au dernier pointage (ex. un creux de données sur le ticker), ses champs marché (prix courant, P&L) arrivaient à « null ». Le formatage de ces valeurs plantait alors le rendu de toute la page — wallet, brief et panel affichaient « This page couldn't load ».",
        en: "When a freshly opened position's current price couldn't be fetched at the last mark-to-market (e.g. a data gap on the ticker), its market fields (current price, P&L) arrived as null. Formatting those values then crashed the whole page render — wallet, brief and panel showed 'This page couldn't load'.",
      },
      {
        fr: "Les formateurs et les colonnes concernées sont désormais résilients : une donnée manquante affiche « — » au lieu de faire planter la page. Les types ont aussi été corrigés pour refléter que ces champs peuvent manquer, afin que le compilateur force la prise en compte du cas à l'avenir.",
        en: "The formatters and affected columns are now resilient: a missing value renders '—' instead of crashing the page. The types were also fixed to reflect that these fields can be absent, so the compiler enforces handling that case going forward.",
      },
    ],
    verdict: {
      fr: "Correctif d'affichage uniquement — aucun modèle, gate, sizing ni donnée de trading touché. La valeur manquante est transitoire (repointée au prochain run quotidien) ; en attendant, les pages restent accessibles.",
      en: "Display fix only — no model, gate, sizing or trading data touched. The missing value is transient (re-marked at the next daily run); meanwhile the pages stay accessible.",
    },
  },
  {
    date: "2026-06-10",
    tone: "infra",
    title: {
      fr: "Entraînement entrelacé : les modèles existants ne vieillissent plus pendant l'ajout d'actifs",
      en: "Interleaved training: existing models no longer go stale while new assets are onboarded",
    },
    points: [
      {
        fr: "Constat : après l'ajout de 30 actifs, le sweep traitait d'abord TOUS les nouveaux combos (198 premières optimisations profondes à la suite) avant de rafraîchir quoi que ce soit. Pendant ce backlog (~20h), les modèles existants restaient en pause et le « plus ancien » grimpait — d'où le 1j 22h observé sur /ops.",
        en: "Symptom: after adding 30 assets, the sweep processed ALL new combos first (198 deep first-time optimizations back-to-back) before refreshing anything. During that backlog (~20h) existing models stayed paused and the 'oldest' climbed — hence the 1d 22h seen on /ops.",
      },
      {
        fr: "Correctif : les nouveaux actifs (sans modèle) gardent la priorité, mais désormais un modèle existant est intercalé — rafraîchi du plus ancien d'abord — tous les 4 nouveaux. Le plus vieux modèle est donc re-touché en continu au lieu d'attendre la fin du backlog. Le ratio est réglable (`sweep_interleave_every`, défaut 4 ; 0 = désactivé).",
        en: "Fix: new assets (no model) keep priority, but one existing model is now spliced in — refreshed oldest-first — after every 4 new ones. The oldest model is therefore touched continuously instead of waiting for the backlog to drain. The ratio is configurable (`sweep_interleave_every`, default 4; 0 = off).",
      },
      {
        fr: "POC intraday mis en pause en parallèle (`enabled: false`) : les 2 GPU repassent à 100% sur le swing, donc le backlog de cold-start se vide plus vite. Le sous-système intraday est conservé intact et réactivable d'un seul réglage.",
        en: "Intraday POC paused in parallel (`enabled: false`): both GPUs go back to 100% on swing, so the cold-start backlog drains faster. The intraday subsystem is kept intact and re-enablable with a single flag.",
      },
    ],
    verdict: {
      fr: "Ordre de la file d'entraînement uniquement — aucun label, gate, sizing, seuil ni modèle promu touché ; le gel du forward test est respecté. Objectif tenu : aucun actif n'est laissé trop longtemps sans ré-optimisation, même pendant l'onboarding d'un gros lot.",
      en: "Training-queue order only — no label, gate, sizing, threshold or promoted model touched; the forward-test freeze is respected. Goal met: no asset is left un-reoptimized for too long, even while a large batch is onboarded.",
    },
  },
  {
    date: "2026-06-09",
    tone: "infra",
    title: {
      fr: "Fraîcheur visible : âge de dernière optimisation par actif (vert / ambre / rouge)",
      en: "Freshness made visible: per-asset last-optimization age (green / amber / red)",
    },
    points: [
      {
        fr: "Chaque actif affiche désormais depuis combien de temps son modèle a été optimisé pour la dernière fois (date du dernier réglage des hyper-paramètres), codé couleur : vert si < 2 jours, ambre jusqu'à 5 jours, rouge au-delà. Visible sur trois pages : la colonne « Optimisé » du tableau /training, le panneau « Entraînement & HPO » du détail de chaque actif, et /ops.",
        en: "Every asset now shows how long ago its model was last optimized (the last hyper-parameter tuning), color-coded: green if < 2 days, amber up to 5 days, red beyond. Surfaced on three pages: the 'Optimized' column of the /training table, each asset's 'Training & HPO' drill-down panel, and /ops.",
      },
      {
        fr: "Sur /training, l'âge retenu par actif est celui du côté le plus ancien (long ou court) — un côté à jour ne peut pas masquer un côté en retard. Les seuils de couleur suivent le rythme réel du sweep (cycle de rafraîchissement ~2 jours), donc le rouge signale vraiment un actif qui décroche.",
        en: "On /training, the per-asset age is the oldest of the two sides (long or short) — an up-to-date side can't mask a lagging one. The color thresholds follow the sweep's real cadence (~2-day refresh cycle), so red genuinely flags an asset falling behind.",
      },
      {
        fr: "Sur /ops, nouvelle ligne « Plus ancien modèle » : l'âge du modèle le plus ancien parmi les actifs actuellement suivis par le sweep, calculé en direct depuis la base d'optimisation. Le calcul est borné aux combos en rotation (les anciennes études orphelines des sous-systèmes retirés — modèle global, ranker market-neutral — sont ignorées), donc /ops s'accorde exactement avec /training. C'est la jauge de saturation de la machine — si ce chiffre dérive vers plusieurs jours, c'est le signal qu'il faut arrêter d'ajouter des actifs.",
        en: "On /ops, a new 'Oldest model' row: the age of the oldest model among the assets the sweep currently tracks, computed live from the optimization database. It's scoped to the combos in rotation (orphan studies from retired subsystems — the global model, the market-neutral ranker — are ignored), so /ops reconciles exactly with /training. It's the machine-saturation gauge — if this drifts toward several days, it's the signal to stop adding assets.",
      },
    ],
    verdict: {
      fr: "Affichage uniquement (la donnée existait déjà dans l'API) — aucun label, gate, sizing, seuil ni modèle promu touché ; le gel du forward test est respecté. Tu vois maintenant d'un coup d'œil quels actifs restent frais et lesquels décrochent.",
      en: "Display only (the data already existed in the API) — no label, gate, sizing, threshold or promoted model touched; the forward-test freeze is respected. You can now see at a glance which assets stay fresh and which fall behind.",
    },
  },
  {
    date: "2026-06-09",
    tone: "infra",
    title: {
      fr: "Ré-entraînement par ancienneté : aucun actif laissé périmer",
      en: "Staleness-first retraining: no asset left to go stale",
    },
    points: [
      {
        fr: "Le sweep continu re-entraînait les actifs déjà optimisés dans un ordre figé (l'ordre du fichier de configuration). Au moindre redémarrage du service, il repartait du haut de la liste — les actifs en fin de liste pouvaient donc être ré-optimisés bien moins souvent que les premiers, et rester sur des données plus anciennes.",
        en: "The continuous sweep re-trained already-optimized assets in a fixed order (config-file order). On any service restart it resumed from the top of the list — so assets at the tail could be re-optimized far less often than those at the head, and sit on older data.",
      },
      {
        fr: "Nouvel ordonnancement : les actifs sans modèle (nouveaux arrivants) passent toujours en premier — un actif sans modèle est la priorité absolue — puis tous les autres sont ré-entraînés du plus ancien au plus récent, en lisant la vraie date de dernière optimisation dans la base Optuna. L'actif dont le modèle est le plus vieux est donc toujours le prochain re-fit. Cela borne l'ancienneté maximale et résiste aux redémarrages : le sweep reprend sur l'actif réellement le plus ancien, jamais en re-faisant la tête de liste pendant que la queue patiente.",
        en: "New ordering: assets with no model yet (newcomers) always go first — an asset with no model is the top priority — then every other asset is re-trained oldest-first, reading the real last-optimization time from the Optuna database. The asset whose model is oldest is always the next re-fit. This bounds worst-case staleness and survives restarts: the sweep resumes on the genuinely oldest asset, never redoing the head of the list while the tail waits.",
      },
      {
        fr: "Qualité par actif inchangée : chaque ré-entraînement garde le tournoi complet (tous les modèles : LightGBM, LSTM, PatchTST, TFT) avec optimisation des hyper-paramètres, sur les dernières données fraîches. On ne coupe ni modèles ni HPO — on garantit seulement que personne n'est oublié. La barre /ops affiche désormais l'ancienneté du plus vieux modèle en file au début de chaque cycle.",
        en: "Per-asset quality unchanged: every re-train keeps the full tournament (all models: LightGBM, LSTM, PatchTST, TFT) with hyper-parameter optimization, on the freshest data. No models or HPO are cut — we only guarantee no one is forgotten. The /ops log now shows the oldest queued model's age at the start of each cycle.",
      },
    ],
    verdict: {
      fr: "Infrastructure uniquement (ordre de la file d'entraînement) — aucun label, gate, sizing, seuil ni modèle promu touché ; le gel du forward test est respecté. Chaque actif reste entraîné et optimisé sur données fraîches, au plus vite, sans qu'aucun ne prenne du retard.",
      en: "Infrastructure only (training-queue order) — no label, gate, sizing, threshold or promoted model touched; the forward-test freeze is respected. Every asset stays trained and optimized on fresh data, as fast as possible, with none falling behind.",
    },
  },
  {
    date: "2026-06-09",
    tone: "infra",
    title: {
      fr: "Univers élargi : 150 → 180 actifs (+30), capacité machine validée",
      en: "Universe expanded: 150 → 180 assets (+30), machine capacity validated",
    },
    points: [
      {
        fr: "30 actifs ajoutés sur les cinq classes : US +8 (General Electric, RTX, Lockheed Martin, Deere, AT&T, Lowe's, Union Pacific, Citigroup → 53), France +6 (AXA, Teleperformance, Eurofins, Sodexo, STMicroelectronics, Stellantis → 37), forex +5 (EUR/SEK, EUR/NOK, USD/HKD, USD/SGD, USD/PLN → 34), crypto +6 (Zcash, Dash, Neo, Decentraland, Enjin, Maker → 29) et matières premières +5 (riz, avoine, tourteau de soja, lait, aluminium → 27).",
        en: "30 assets added across all five classes: US +8 (General Electric, RTX, Lockheed Martin, Deere, AT&T, Lowe's, Union Pacific, Citigroup → 53), France +6 (AXA, Teleperformance, Eurofins, Sodexo, STMicroelectronics, Stellantis → 37), forex +5 (EUR/SEK, EUR/NOK, USD/HKD, USD/SGD, USD/PLN → 34), crypto +6 (Zcash, Dash, Neo, Decentraland, Enjin, Maker → 29) and commodities +5 (rough rice, oats, soybean meal, milk, aluminium → 27).",
      },
      {
        fr: "Choix guidés par la capacité réelle du serveur, mesurée avant d'ajouter : ~8 min par combo en recherche profonde (100 essais), ~2,5 min en ré-entraînement léger. À 180 actifs (1 080 combos), chaque modèle reste rafraîchi sur données fraîches en moins de 2 jours — très en deçà du plafond où le refresh dépasserait la semaine (~450-670 actifs). On privilégie l'historique profond et la diversité sectorielle (défense, rail, assurance, semi-conducteurs, nouveaux blocs de devises, aluminium).",
        en: "Picks driven by the server's real capacity, measured before adding: ~8 min per combo on deep search (100 trials), ~2.5 min on a light re-fit. At 180 assets (1,080 combos) every model still gets refreshed on fresh data in under 2 days — well below the ceiling where refresh would exceed a week (~450-670 assets). Priority on deep history and sector diversity (defense, rail, insurance, semiconductors, new currency blocs, aluminium).",
      },
      {
        fr: "Les 30 validés sur yfinance avant ajout : plus de 1 000 barres quotidiennes chacun (la plupart > 3 000), données fraîches au 9 juin. Le sweep continu passe sur 1 080 combos et traite les combos jamais explorés EN PREMIER — les 30 nouveaux reçoivent leur recherche profonde avant tout ré-approfondissement des anciens. Le gel du forward test est respecté : un nouvel actif ne peut entrer dans le livre engagé qu'en passant le même gate durci ; capital papier uniquement, gate / sizing / seuils inchangés.",
        en: "All 30 validated on yfinance before adding: 1,000+ daily bars each (most > 3,000), fresh data as of June 9. The continuous sweep now spans 1,080 combos and processes never-searched combos FIRST — the 30 newcomers get their deep search before any re-deepening of incumbents. The forward-test freeze is respected: a newcomer can only enter the committed book by clearing the same hardened gate; paper capital only, gate / sizing / thresholds unchanged.",
      },
    ],
    verdict: {
      fr: "Univers plus large et plus diversifié, calibré sur la capacité mesurée du serveur pour que chaque actif reste entraîné en temps utile. Seul l'univers grandit — la logique de gate, de sizing et de seuils ne bouge pas.",
      en: "A broader, more diversified universe, calibrated to the server's measured capacity so every asset stays trained in time. Only the universe grows — the gate, sizing and threshold logic stays put.",
    },
  },
  {
    date: "2026-06-09",
    tone: "ship",
    title: {
      fr: "Clarté visiteur : accueil + Brief plus lisibles dès la première visite",
      en: "Visitor clarity: home + Brief easier to grasp on first visit",
    },
    points: [
      {
        fr: "Confusion n°1 corrigée : un actif « Non validé » avec une confiance (P(win)) élevée affichait quand même « Achat — hausse probable ». Le mini-conseil est désormais nuancé pour ces modèles (« Piste à la hausse — modèle non validé, pas d'achat »), pour ne jamais lire comme un ordre. Confiance du jour et fiabilité prouvée du modèle sont deux choses distinctes.",
        en: "Top confusion fixed: a 'Not validated' asset with high confidence (P(win)) still showed 'Buy — up-move likely'. The mini-advice is now hedged for those models ('Upside lead — model not validated, no buy'), so it never reads as an order. Today's confidence and the model's proven reliability are two separate things.",
      },
      {
        fr: "Vocabulaire unifié : « Indicatif » devient « Non validé » partout (plus clair sur ce que ça veut dire), avec une infobulle d'explication au survol du badge et les actifs validés remontés en haut du tableau.",
        en: "Unified vocabulary: 'Advisory' becomes 'Not validated' everywhere (clearer about what it means), with an explanatory tooltip on hover and validated assets sorted to the top of the table.",
      },
      {
        fr: "Brief : la section « Prochains ordres » est renommée « Ce que le robot prévoit de faire ce soir — vous n'avez rien à faire » pour qu'on ne la prenne plus pour une liste d'actions. La section « À l'observation » est explicitement reliée aux modèles « Non validés » de l'accueil. Un aiguilleur d'une ligne indique quel onglet suivre selon qu'on copie le robot ou qu'on veut juste comprendre.",
        en: "Brief: the 'Upcoming orders' section is renamed 'What the robot plans to do tonight — nothing for you to do' so it's no longer mistaken for an action list. The 'On watch' section is explicitly linked to the home page's 'Not validated' models. A one-line router says which tab to follow depending on whether you copy the robot or just want to understand.",
      },
      {
        fr: "Nouvelle légende « Le vocabulaire en 30 secondes » (P(win), Validé/Non validé, Prévision/Exécuté, Observation) sur le Brief, et étape d'accueil renforcée pour bien séparer confiance et validation.",
        en: "New 'The vocabulary in 30 seconds' legend (P(win), Validated/Not validated, Forecast/Executed, Observation) on the Brief, plus a strengthened onboarding step that cleanly separates confidence from validation.",
      },
    ],
    verdict: {
      fr: "Interface et formulations uniquement — aucun modèle, gate, sizing ni seuil touché ; le gel du forward test est respecté.",
      en: "UI and wording only — no model, gate, sizing or threshold touched; the forward-test freeze is respected.",
    },
  },
  {
    date: "2026-06-09",
    tone: "infra",
    title: {
      fr: "État machine : verdict d'utilisation en un coup d'œil sur /ops",
      en: "Machine status: one-glance utilization verdict on /ops",
    },
    points: [
      {
        fr: "La page /ops affichait six jauges brutes (deux cartes GPU + CPU) sans dire ce qu'elles signifient ensemble. Nouvelle ligne de synthèse : « Utilisation : sous-utilisée / bien utilisée / saturée / au repos », calculée côté serveur (donc testée) à partir des chiffres déjà collectés.",
        en: "The /ops page showed six raw gauges (two GPU cards + CPU) without saying what they mean together. New synthesis line: 'Utilization: under-utilized / well-utilized / saturated / at rest', computed server-side (so it's tested) from the numbers already collected.",
      },
      {
        fr: "Le signal clé sur une machine bi-GPU est l'asymétrie : une carte à fond pendant que sa jumelle est à 0 % alors que le sweep tourne, c'est du débit perdu, pas une saturation saine — le sweep entraîne un combo à la fois. Le verdict explicite cette cause, ainsi qu'un CPU peu sollicité (charge1 ÷ cœurs).",
        en: "The key signal on a dual-GPU box is asymmetry: one card pegged while its twin sits at 0% during an active sweep is wasted throughput, not healthy saturation — the sweep trains one combo at a time. The verdict spells out that cause, plus a mostly-idle CPU (load1 ÷ cores).",
      },
    ],
    verdict: {
      fr: "Observabilité uniquement — aucune logique de trading, de gate, de sizing ni de seuils touchée ; le gel du forward test est respecté.",
      en: "Observability only — no trading, gate, sizing or threshold logic touched; the forward-test freeze is respected.",
    },
  },
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
      {
        fr: "Même discipline de gestion du risque que le Portefeuille appliquée au Panel : kill-switch de drawdown gradué (réduction à −10 %, arrêt à −20 %), plafond de positions et caps d'exposition — mais calculés sur le propre capital et le propre drawdown du Panel, indépendamment du livre engagé.",
        en: "Same risk discipline as the committed Wallet applied to the Panel: graduated drawdown kill-switch (de-risk at −10%, halt at −20%), position cap and exposure caps — computed on the Panel's own capital and own drawdown, independently of the committed book.",
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
      fr: "Univers élargi : 125 → 150 actifs, +5 par classe",
      en: "Universe expanded: 125 → 150 assets, +5 per class",
    },
    points: [
      {
        fr: "25 actifs ajoutés, répartis sur les cinq classes : US +5 (Procter & Gamble, Cisco, Wells Fargo, Boeing, Nike → 45), France +5 (Renault, Carrefour, Publicis, Bouygues, Bureau Veritas → 31), forex +5 (AUDCAD, AUDNZD, CADCHF, GBPCAD, NZDCAD → 29), crypto +5 (Monero, Tezos, EOS, Aave, Filecoin → 23) et matières premières +5 (essence RBOB, porc maigre, bovin d'engraissement, huile de soja, jus d'orange → 22).",
        en: "25 assets added, spread across all five classes: US +5 (Procter & Gamble, Cisco, Wells Fargo, Boeing, Nike → 45), France +5 (Renault, Carrefour, Publicis, Bouygues, Bureau Veritas → 31), forex +5 (AUDCAD, AUDNZD, CADCHF, GBPCAD, NZDCAD → 29), crypto +5 (Monero, Tezos, EOS, Aave, Filecoin → 23) and commodities +5 (RBOB gasoline, lean hogs, feeder cattle, soybean oil, orange juice → 22).",
      },
      {
        fr: "Les 25 validés sur yfinance avant ajout : plus de 1 000 barres quotidiennes chacun (la plupart > 4 000), données fraîches au 8-9 juin. Le sweep continu passe désormais sur 900 combos (actif × sens × stratégie) et traite les combos jamais explorés EN PREMIER — les 25 nouveaux reçoivent leur recherche profonde (100 essais HPO) avant tout ré-approfondissement des anciens.",
        en: "All 25 validated on yfinance before adding: 1,000+ daily bars each (most > 4,000), fresh data as of June 8-9. The continuous sweep now spans 900 combos (ticker × side × strategy) and processes never-searched combos FIRST — the 25 newcomers get their deep search (100 HPO trials) before any re-deepening of incumbents.",
      },
      {
        fr: "Le forward test n'est pas pollué : un nouvel actif ne peut entrer dans le livre engagé qu'en passant le même gate durci (DSR ≥ 0.95, ≥ 20 trades hors-échantillon, FDR) que les autres. La plupart resteront advisory-only — c'est le résultat attendu du gate.",
        en: "The forward test is not polluted: a newcomer can only enter the committed book by clearing the same hardened gate (DSR ≥ 0.95, ≥ 20 out-of-sample trades, FDR) as everyone else. Most will stay advisory-only — the gate's expected outcome.",
      },
    ],
    verdict: {
      fr: "Plus de candidats honnêtes pour le forward test, sans promesse : capital papier uniquement, livre engagé inchangé. Le gel du forward test est respecté — seul l'univers s'élargit, la logique de gate, de sizing et de seuils ne bouge pas.",
      en: "More honest forward-test candidates, no promise: paper capital only, committed book unchanged. The forward-test freeze is respected — only the universe grows; the gate, sizing and threshold logic stays put.",
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
