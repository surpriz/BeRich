# Gate Reform — pré-enregistrement (rédigé le 2026-06-10, GELÉ jusqu'à la fin du forward test)

Ce document pré-enregistre les changements de validation décidés lors de l'audit du
2026-06-10, **avant** de connaître le résultat du forward test (~30 trades fermés sur le
livre committed, règle archivée dans `docs/RESULTS.md`). Comme le bake-off de juin 2026 :
on fige les critères maintenant pour qu'aucun choix ne soit fait après avoir vu les
résultats. **Rien ici ne s'exécute avant la fin du forward test** — appliquer ces
changements plus tôt changerait la population promue en cours d'évaluation.

Déclencheur : ~30 trades fermés sur le livre committed (vérifier via `/wallet` ou
`SELECT count(*) FROM paper_trades WHERE tier='promoted' AND status LIKE 'closed%'`).

Exécution : `uv run python scripts/apply_gate_reform.py` (dry-run par défaut ; `--apply`
modifie le code/les constantes puis laisse le sweep continu re-gater + FDR). Après
application, le forward test repart de zéro sur la nouvelle population.

## Changements pré-enregistrés (dans l'ordre d'importance)

### 1. Le gate doit backtester la politique réellement servie

Constat (audit 2026-06-10) : `train_candidate` backteste à **proba brute ≥ seuil global
0.35** (`tournament.py` — `BacktestConfig(entry_threshold=config.signals.buy_threshold)`
sur `oof.frame["proba"]`), alors que le serving trade à **proba calibrée ≥
decision_threshold tuné par actif**. Le Sharpe/DSR qui franchit le gate n'est donc pas
celui de la politique jouée par le paper book.

Changement : dans `train_candidate`, calibrer d'abord les OOF, puis backtester au
`decision_threshold` retenu (ou au seuil global si aucun n'est tuné) sur la proba
calibrée — gate et serving deviennent la même politique, par construction.

### 2. Fin du triple usage des OOF (calibration + tuning du seuil + évaluation du gate)

Constat : les mêmes OOF servent à fitter le calibrateur, chercher le seuil optimal
(`optimal_decision_threshold`) et calculer le Sharpe/AUC du gate. Le seuil est donc
choisi sur les données qui l'évaluent (biais de sélection optimiste).

Changement : split temporel des OOF — premiers 2/3 pour calibration + tuning du seuil,
dernier 1/3 (jamais vu par le tuning) pour le backtest du gate. Si le dernier tiers
produit < MIN_TRADES, l'actif est advisory (pas de promotion sur données insuffisantes).

### 3. MIN_TRADES : 20 → 50

Constat : 20 trades OOS suffisent à une lucky streak pour franchir un DSR ≥ 0.95.
Changement : `registry.MIN_TRADES = 50`. Conséquence assumée : moins d'actifs promus,
plus fiables. (Si la population tombe < 10 modèles, retenir 30 comme repli — décidé
maintenant, pas après lecture des résultats.)

### 4. AUC_FLOOR : 0.5 → 0.52

Constat : `auc > 0.5` n'exclut que le hasard parfait. Changement :
`tournament.AUC_FLOOR = 0.52` — cohérent avec les niveaux observés sur les phases
"signal réel" (h=3-5 : 0.55-0.58 ; PEAD : 0.53).

### 5. Gaps : remplissage du stop au pire de (barrière, open du bar)

Constat : `_resolve_exit` / `_resolve_exit_trailing` (engine.py) supposent un fill au
prix exact de la barrière même quand le bar ouvre au-delà (gap overnight) — optimiste,
surtout sur crypto/commodities. Changement : si `open` du bar de déclenchement est déjà
au-delà du stop, le fill est `open` (pire), pas la barrière ; idem pour le paper
(`_resolve_paper_exit`, `_resolve_trailing_exit`). Le target reste rempli à la barrière
(un gap favorable au-delà du TP remplit au mieux à l'open — symétrie honnête : utiliser
l'open là aussi).

### 6. Réforme du gate long (biais beats-buy-hold, déjà notée dans RESULTS.md)

Constat : `beats_buy_hold` promeut des longs forex (baseline plate) et bloque des longs
US/FR (baseline en tendance 0.7+ Sharpe) — le gate mesure la baseline, pas l'edge.
Changement pré-enregistré : le long doit battre le B&H **sur métrique ajustée du risque
ET sur drawdown** (Sharpe > B&H Sharpe OU Calmar > B&H Calmar), tout en gardant le
plancher de significativité (DSR ≥ 0.95, p < 0.05) désormais commun aux deux côtés.
La décision finale Sharpe-vs-Calmar-vs-sleeve sera tranchée à l'application, mais
l'espace des options est figé ici — pas d'invention post-résultats.

### 7. Caps corrélation-aware (devise / facteur)

Constat : 3 crosses JPY ouverts = un seul pari que les caps par classe ne voient pas
(l'alerte d'observabilité ajoutée le 2026-06-10 le montre, sans bloquer).
Changement : transformer l'alerte en cap dur — exposition par devise ≤
`max_class_exposure_pct` (même budget qu'une classe), appliquée dans
`_apply_exposure_caps` via un `class_of` enrichi (devise pour le forex).

### 8. Pyramidage borné (ne pas renforcer une position en perte)

Constat (forward test, 2026-06-16, à 12 trades) : le livre a ouvert **3 shorts Visa, 2 ADA,
2 SOL** — il a empilé dans le même sens sur le même actif, et ces ajouts portaient sur les
segments perdants (crypto-short, us-short). Le pyramidage est autorisé et borné par les caps
d'exposition, mais ajouter à une position **déjà dans le rouge** a concentré les dégâts au lieu
de les diluer. Aucun cap actuel ne distingue « première entrée » de « renforcement d'un perdant ».
Changement pré-enregistré (espace figé, décision finale à l'application) :
- interdire l'ouverture d'un nouveau lot sur un (ticker, sens) dont une position ouverte est
  actuellement sous l'eau (MTM < 0), **ou** plafond par actif plus serré pour les renforts ;
- à appliquer dans `signals/paper._plan_book` / `_apply_exposure_caps` (lecture du MTM des
  positions ouvertes du même (ticker, sens) avant d'autoriser un nouveau lot).
À cadrer avec l'item 7 (caps corrélation-aware) : même famille de problème — la concentration
non vue par les caps par classe.

### 9. Mineurs

- Embargo `horizon + 2` pour les exit modes trailing (l'activation du ratchet décale
  l'information de sortie au-delà de l'horizon nominal).
- `optimal_decision_threshold(min_count=max(20, 0.05 * n_oof))` — un seuil tuné sur
  20 points parmi 250 est du bruit.
- Documenter `cost_bps` (frais + slippage vol-proportionnel) dans les métadonnées de
  modèle pour que le gate et le brief citent le même chiffre.

## Ce qui n'est PAS dans cette réforme (et pourquoi)

- Pas de nouvelle feature OHLCV/macro, pas de PEAD, pas de cross-sectional daily — les
  trois guardrails de `docs/RESULTS.md` restent en vigueur.
- Pas de Kelly / vol-targeting : le money management reste minimal tant qu'aucun edge
  net n'est démontré en forward.
- `meta_labeling` et `regime_conditioning` restent off — ils auront leur propre
  validation pré-enregistrée s'ils sont retenus comme piste (cf. RESULTS.md, prochaine
  itération).

## Critère de succès (figé maintenant)

La réforme est jugée réussie si, sur le forward test suivant (même règle des ~30 trades
committed) : l'expectancy nette par trade des promus est ≥ 0 ET la fraction de segments
(classe × side) à expectancy positive est supérieure à celle du forward test actuel.
Sinon : retour à l'inventaire des pistes (microstructure, horizons courts) — pas de
re-tuning du gate une troisième fois sur les mêmes données.
