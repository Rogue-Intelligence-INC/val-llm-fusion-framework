# Framework de fusion Val-LLM

[中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | **Français**

`valhalla-server` — un framework qui décompose n'importe quel LLM par
profondeur de couche, l'adapte via le système Fate, fusionne librement
les modalités texte/audio/image, réinjecte le résultat dans les poids
et le sert sous forme de sessions structurées.

## Positionnement

Valhalla est un **substrat de mémoire longue durée natif Fate**. Il ne
fait ni recherche ni génération. L'expérience converge passivement en
blocs (bundles) ; la saisie se fait par intensité de champ
(énergie · √complexité) ; c'est la propre équation de cluster de Fate
qui juge « répondre / se taire » ; l'apprentissage a lieu *après* le
jugement, tout au long de la durée d'utilisation. Ce qu'il vend : zéro
fabrication, une discipline du silence et une stationnarité du
comportement — pas la précision des réponses. Ce n'est **ni** une API
de mémoire pour chatbot, **ni** un remplacement de RAG, et il ne
concourt pas au classement LoCoMo (comparaisons à questions identiques
et limites : `docs/VALHALLA_COMPETITIVE_COMPARISON_20261113.md`).

## Fonctionnement

1. **Décomposition** — un LM est divisé par profondeur en quatre
   quadrants (Q0→Hub, Q1→Tile, Q2→Stem, Q3→Hub) et adapté via Fate.
2. **Fusion** — les ports texte/audio/image d'un même échantillon
   pulsent en entrelacé dans une seule session Triad ; ni isolation,
   ni routage prédéfini.
3. **Mémoire** — les signaux entrent par l'admission passive de Fate
   (`receive_passive`) ; la convergence ferme tuiles et cellules
   souches ; le matériau se bundle lorsque `(x−1)²` satisfait
   l'équation propre du bloc.
4. **Jugement** — la décision fog (emit / abstain / defer) appartient
   à l'équation Fate vivante du moteur ; la passerelle doit l'exécuter.
   La tête de décodage Transformer ne peut que reformuler les preuves
   libérées par Fate et ne peut jamais annuler une abstention.
5. **Réinjection (baking)** — la structure est réécrite dans un modèle
   de base pour produire un nouveau checkpoint (l'original est
   intact), avec carte de modèle et registre haché.

## Preuves mesurées (réexécutions locales fraîches, 2026-11-15)

- Extreme targeted, 806 questions : **12/12**, jetons fabriqués **0**
  (identique sur l'ancien moteur et le moteur nouvellement synchronisé).
- Passerelle multimodale, bras Fate-native : **6/9**, fabriqués **0**.
- Limites locales en 312 tours : aiguilles cœur **89/90**, fabrication
  **0/312**, questions distractrices répondues 3/120 (le silence est
  le défaut).
- Invariance de fenêtre : réponses cœur identiques entre bras de
  fenêtre, fabrication 0 ; un bras de 356 lignes s'ingère désormais
  en ~1,2 s (optimisation par étapes : 462 s → 3,8 s sur le bras le
  plus lourd).
- Échelle : comportement invariant à l'échelle mesuré sur un corpus
  de 14,95 M de jetons (voir `docs/COMPREHENSIVE_SUMMARY_20261114.md`).

Reproduction : `.\reproduce.ps1 --quick` (batterie complète ~15 min) ;
voir `engine/README.md` pour la batterie du moteur local.

## Organisation du dépôt

- `valhalla_server/` — la passerelle (arbitrage Fate d'abord, registre
  des têtes de décodage, pipeline modelops, provisionnement VALENC1).
- `engine/` — sources Rust synchronisées du moteur ValhallaBase/Fate
  (source de vérité : l'espace de travail Valhalla ; manifeste de
  synchronisation inclus).
- `docs/` — synthèses mesurées, comparaison concurrentielle, rapports
  d'état.
- `tests/` — sondes et suites de limites ; `tests/reproduce_all.py`
  est le pilote de reproduction en un clic.
- `runtime/` — moteurs au repos, chiffrés (VALENC1) ; `data/` —
  registre des têtes de décodage et coffre des modèles.

## Limites (énoncées clairement)

- Les comparaisons à questions identiques face aux systèmes de type
  RAG/Mem0 sont documentées, y compris là où Valhalla perd ; nous ne
  revendiquons aucune victoire LoCoMo.
- L'émission proactive n'est pas revendiquée.
- Échec connu à ce jour : sous question à fausse prémisse, une donnée
  poison sous forme d'instruction peut encore être émise (mesuré ;
  en analyse, non masqué par un correctif).

Contact : licensing@rogue-intelligence.com
