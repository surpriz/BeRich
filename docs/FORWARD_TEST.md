# Forward test — état des lieux et procédure au déclencheur

**Document de référence unique.** Rédigé le 2026-06-10. Si des semaines ou des mois passent
avant la prochaine session : lire ce document, puis `docs/GATE_REFORM.md` (la réforme
pré-enregistrée), puis `docs/RESULTS.md` (l'historique complet des phases). Tout ce qui doit
se passer au terme du forward test est écrit ici et dans GATE_REFORM.md — rien n'est dans la
tête de personne.

## Où on en est (2026-06-10)

- **46-47 modèles promus** (forex-lourd, crypto/commodity shorts, quelques longs US/FR) sur
  ~180 actifs / 1 080 combos (ticker × side × stratégie de sortie). Ce sont des *candidats*,
  pas un edge prouvé — voir RESULTS.md, Phase 14.
- **Livre engagé (tier promoted, 10 k€ papier)** : 7 positions ouvertes, **2 fermées / ~30**.
- **Livre d'observation (tier observe, sans capital)** : 10 ouvertes, 6 fermées.
- Rythme attendu : 15 positions max en rotation, horizons 10-20 j → ~15-30 clôtures/mois →
  déclencheur atteint vers **juillet-août 2026**.
- Le système est **GELÉ** : correctifs de bugs, UI et observabilité uniquement. Aucun
  changement de labels, gate, sizing, seuils ou population promue avant le déclencheur.

### Changements notables faits pendant le gel (audit du 2026-06-10, commits `04bbe5d`, `0d51b7a`+)

- Le paper book paie désormais les **vrais coûts** (frais + slippage vol-proportionnel, le
  même modèle de coûts que le gate) — figés à l'ouverture, déduits à la clôture. Les trades
  ouverts avant le 2026-06-10 clôturent à coût zéro (historique non réécrit).
- Bug corrigé : double comptage du slippage dans les stats par trade du backtest (le Sharpe
  du gate n'était pas affecté ; population promue inchangée).
- Brief enrichi : validité de l'ordre (horizon), P(gain) calibrée, espérance nette, coût bps.
- Alerte de **concentration devise** (/wallet + digest) — observabilité seulement, ne bloque
  rien pendant le gel. Constat immédiat : GBP portait 89 % du capital sur 2 positions.
- Infra : backup atomique + off-site rclone (opt-in `BERICH_BACKUP_REMOTE`), watchdog systemd
  du run de 22:30, contre-vérification hebdo yfinance↔Stooq.
- Machine : chevauchement CPU/GPU dans le sweep, top-ups ciblés (gagnant + lgbm en semaine,
  contest complet le week-end, `zoo.topup_winner_only`), pruning Optuna par fold des modèles
  profonds. Ordonnancement uniquement — gate inchangé (le comptage DSR couvre toujours le zoo
  complet).

## Le déclencheur

**~30 trades fermés sur le livre engagé** (tier `promoted`). Vérification à tout moment :

```bash
uv run python scripts/apply_gate_reform.py    # affiche N fermés / 30 et refuse tant que non atteint
```

## Procédure au déclencheur (pré-enregistrée — ne pas improviser)

1. `uv run python scripts/apply_gate_reform.py` → table de décision : **espérance nette par
   segment (classe d'actif × sens)**.
2. **Règle de décision** (archivée dans RESULTS.md) : garder les segments à espérance nette
   positive, couper les autres. Pas de re-litigation trade par trade.
3. Appliquer la **réforme du gate** — les 8 points de `docs/GATE_REFORM.md`, dont les plus
   importants : le gate doit backtester la politique réellement servie (proba calibrée +
   seuil par actif), fin du triple usage des OOF, MIN_TRADES 20→50, AUC_FLOOR 0,52, stops
   remplis à l'open sur gap, réforme du gate long, caps durs par devise.
4. Laisser le sweep re-gater les 1 080 combos + FDR → nouvelle population promue.
5. **Nouveau forward test propre** sur la nouvelle population (compteur remis à zéro).
6. Critère de succès du nouveau test (figé dans GATE_REFORM.md) : espérance nette ≥ 0 et
   fraction de segments positifs en hausse. Sinon : pivot vers les pistes pré-enregistrées
   (meta-labeling, regime conditioning, microstructure) — **pas de troisième re-réglage du
   gate sur les mêmes données**.

### Si les résultats sont bons
Concentration du capital papier sur les segments gagnants → 3-6 mois de track record public
vérifiable → alors seulement, question de l'argent réel (petit) et/ou de la monétisation du
conseil (vérifier le cadre AMF/CIF avant toute monétisation — voir conversation du
2026-06-10 : recommandation d'investissement = activité réglementée en France).

### Si les résultats sont mauvais
Conclusion honnête : l'edge candidat n'était pas réel. Couper, appliquer quand même la
réforme du gate (plus stricte), pivoter la recherche. Pas d'argent réel.

## Money management — état des lieux (audit du 2026-06-10)

**Verdict : solide et volontairement minimal — le bon choix pendant la validation.**
Optimiser le money management avant de savoir si l'edge existe revient à régler les roues
d'une voiture sans moteur : le MM ne crée pas d'espérance positive, il décide de la survie
et de la vitesse de croissance une fois l'espérance acquise.

### Ce qui est en place (niveau professionnel de base)
- Risque fixe **1 % du capital par trade**, taille dérivée de la distance au stop (chaque
  trade risque le même montant au stop — c'est déjà un sizing ajusté à la volatilité).
- **Stops ATR fixés à l'avance**, direction-aware, tie-break conservateur (stop avant target
  dans le même bar). Trois stratégies de sortie dont trailing (laisser courir les gains).
- **Kill-switch gradué** : derisk ×0,5 au-delà de −10 % de drawdown, halte totale à −20 %.
- **Plafonds d'exposition** : par actif, par livre, par classe (40 %), 15 positions max.
- **Capital par niveau de confiance** : seul le tier promoted engage du capital.
- Profils de risque prudent/équilibré/offensif.

### Les écarts vs un desk professionnel (tous PRÉVUS, tous post-déclencheur)
1. **Corrélation ignorée entre positions** — 3 crosses JPY = un seul pari. Alerte en place
   depuis le 2026-06-10 ; le **cap dur par devise** est l'item 7 de GATE_REFORM.md.
2. **Pas de vol targeting au niveau portefeuille** — le code existe (`risk/sizing.py`),
   volontairement OFF. À activer après validation de l'edge, jamais avant.
3. **Risque de gap** — le stop borne la perte théorique, mais un gap overnight peut la
   dépasser ; le backtest le suppose rempli à la barrière. Fix = item 5 de GATE_REFORM.md.
   (Sans levier, la perte reste bornée par le notional — le risque est réel mais contenu.)
4. **Kelly / sizing croissant** — OFF à raison : un Kelly sur un edge mal estimé détruit le
   capital plus vite que tout. À considérer en *fractional Kelly* uniquement après un edge
   démontré en forward.

**Ordre d'activation post-déclencheur (si edge confirmé)** : caps devise → vol targeting →
(éventuellement) fractional Kelly. Un seul changement à la fois, chacun mesuré.

## Garde-fous permanents (rappel)

- Pas de feature hunt OHLCV/macro, pas de PEAD, pas de cross-sectional daily (RESULTS.md).
- Honnêteté de reporting : les pertes s'affichent comme les gains ; rien n'est caché.
- Toute modification visible → entrée datée bilingue sur /changelog.
