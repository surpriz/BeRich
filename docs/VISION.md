# Vision — un robot polyvalent ET spécialisé, multi-actifs

Document de cap (non technique). Capture l'intention long terme exprimée par le propriétaire,
ce qui est déjà fait, ce qui est facile vs un vrai chantier, et le principe directeur qui
empêche la « polyvalence » de devenir une fuite en avant. À lire avec `docs/FORWARD_TEST.md`
(l'étape en cours) et `docs/RESULTS.md` (ce qui a été essayé et enterré).

## L'intention

À terme : un robot capable de couvrir un **maximum de classes d'actifs** — actions (US, FR, et
plus tard IT, DE, UK…), crypto, devises, matières premières, et d'autres instruments (ETF,
futures) — pour dire quoi acheter/vendre, quand, combien, avec quel stop et quel objectif. Le
tout **polyvalent** (un seul moteur, des garde-fous communs) ET **spécialisé** (un modèle propre
par actif). L'idée : un outil qu'on puisse suivre avec discipline pour surfer les tendances sur
quelques jours à quelques semaines, sans biais humain.

## Ce qui est DÉJÀ fait (l'intention est en grande partie l'architecture actuelle)

Le robot est **déjà multi-actifs**. État au 2026-06-13 : **180 actifs sur 5 classes** —
actions US (53), actions FR (37), forex (34), crypto (29), matières premières (27).

Le pipeline est **agnostique à la classe** par construction :
- un seul moteur (features causales, labels triple-barrière, walk-forward, gate, money
  management) tourne sur les 5 classes ;
- un **modèle propre par (actif × sens × stratégie de sortie)** — le modèle d'AAPL n'est pas
  celui de BTC (HPO par actif, seuil de décision par actif) ;
- un **proxy de régime par classe** (SPY pour les actions, indice dollar pour le forex, BTC pour
  la crypto) ;
- un **modèle de coûts par classe** (slippage proportionnel au volume pour les actions ; forfait
  pour la crypto) et des **plafonds d'exposition par classe**.

C'est exactement l'architecture « spécialisé ET polyvalent » souhaitée — et c'est aussi celle des
fonds quantitatifs multi-stratégies. Ce n'est pas naïf : c'est correct.

**Pourquoi ça *semble* focalisé sur une classe** : seuls ~46 des 180 actifs ont passé la
validation, concentrés en forex/crypto-short. Les actions longues n'ont presque pas battu le
buy-and-hold, donc elles n'ont **pas** été promues. La réduction du champ est le **verdict honnête
du gate**, pas une limitation de design.

## Le principe directeur (non négociable)

> **L'architecture doit être polyvalente ; le droit de trader une classe se MÉRITE au gate.**

Ajouter des actifs « pour être complet » a un coût caché : plus de combos à garder frais sur les
mêmes GPU, une pénalité de tests multiples (FDR) plus dure, de la fraîcheur diluée — sans plus
d'edge garanti. La polyvalence est une **capacité**, pas un objectif en soi. On élargit l'univers
quand c'est utile ; on ne trade que ce qui survit à la validation.

## Feuille de route d'extension (par difficulté)

- **Facile (config + validation données)** : actions IT/DE/UK, **ETF**. Le pipeline les traite
  déjà comme des « actions » ; il suffit de valider la donnée yfinance (>1000 barres, fraîche) et
  le sweep continu les entraîne automatiquement. SPY, déjà utilisé comme proxy, *est* un ETF.
- **Moyen (1-2 h/classe)** : vérifier le modèle de coûts et le proxy de régime de chaque nouvelle
  classe ou place de marché (calendriers de cotation différents).
- **Vrai chantier d'ingénierie : les futures.** Les *données* de futures sont déjà ingérées
  (GC=F, CL=F sont des contrats à terme), mais le livre papier les traite comme du **comptant** :
  ni roll (changement de contrat), ni marge, ni levier ne sont modélisés. Trader des futures
  « proprement » est un développement à part entière.

## Money management polyvalent

- La partie **risque par trade** (1 %, dimensionné par l'ATR) est **déjà agnostique à la classe** :
  l'ATR normalise la volatilité, donc 1 % de risque sur BTC = 1 % sur EUR/USD malgré des échelles
  de prix radicalement différentes.
- Le **frontière** est le risque au **niveau du portefeuille** multi-actifs : reconnaître que
  « short EURGBP + short GBPJPY » est un seul pari GBP, ou « short BTC + ETH + SOL » un seul pari
  crypto. Alerte de concentration déjà posée (2026-06-10) ; le cap dur par devise/facteur et le
  vol-targeting sont **pré-enregistrés pour après le forward test** (`docs/GATE_REFORM.md`).
- Levier/marge diffèrent par instrument (futures, forex sont à levier ; actions au comptant ne le
  sont pas). Le livre papier est **cash** aujourd'hui — à modéliser le jour où on trade des
  instruments à levier pour de vrai.

## Horizon temporel : swing vs intraday (réflexion stratégique)

Question ouverte : serait-il plus facile d'avoir un robot fiable en **intraday** (15 min / 1 h)
plutôt qu'en **swing** (jours à semaines) ? Verdict actuel, étayé par l'évidence : **non, le swing
est le bon terrain pour un acteur de notre taille.**

- **Les coûts dominent en intraday.** Plus la fréquence est haute, plus on paie frais+spread+
  slippage par trade, alors que l'edge par trade rétrécit. Un swing peut bouger de plusieurs %
  (les ~6-50 bps de coût sont négligeables) ; un trade de 15 min visant 0,2 % est tué par
  0,1 % de spread. C'est LA raison pour laquelle l'intraday retail perd systématiquement.
- **La concurrence est la plus dure du marché.** L'intraday est l'arène des firmes HFT
  (microseconde, co-location, infrastructures à millions). Plus l'horizon est court, plus on joue
  contre les acteurs les mieux financés sur *leur* terrain. À l'horizon de quelques jours, ces
  acteurs ne sont pas intéressés.
- **Plus de données ≠ plus d'edge.** 100× plus de barres de bruit microstructurel restent du
  bruit ; on obtiendrait juste une estimation plus *confiante* d'un edge nul.
- **La plateforme l'a déjà testé.** Le POC intraday (crypto 1 h) a été construit, lancé, puis
  **mis en pause** faute d'edge promouvable. C'est un *résultat*, pas une supposition.
- **Le « retard des fondamentaux » n'est pas le sujet.** Le swing actuel ne repose **pas** sur des
  fondamentaux : earnings (Phase 5a), news/FinBERT (Phase 5b) et fondamentaux (Phase 11b) ont
  tous été essayés et largement abandonnés. L'edge visé est **technique/momentum**, pas
  fondamental — donc le risque d'interprétation tardive ne s'applique pas vraiment.
- **Ce que dit la science** : le *time-series momentum* (l'effet de suivi de tendance, horizon
  semaines-mois) est l'anomalie la plus robuste et la plus répliquée sur un siècle et toutes les
  classes d'actifs. Il n'existe pas d'équivalent intraday durable et accessible au retail —
  l'intraday est précisément là où les edges se font arbitrer le plus vite.

**Nuance utile** : le projet a trouvé que l'horizon court *du swing* (3-5 jours) donne le meilleur
pouvoir prédictif (AUC ~0,58). La marge de progrès est là — resserrer le swing — pas descendre en
intraday. L'intraday reste réactivable d'un réglage si le contexte change (capital, infrastructure,
edge démontré), mais ce n'est pas la priorité.

## Garde-fous (rappel)

- Pas de feature hunt OHLCV/macro, pas de PEAD, pas de cross-sectional daily (`docs/RESULTS.md`).
- Honnêteté : on n'affiche jamais un actif/une classe comme « validé » sans que le gate l'ait
  promu ; les pertes s'affichent comme les gains.
- Le système est GELÉ pendant le forward test : élargir l'univers ou changer l'horizon attend le
  verdict des ~30 trades.
