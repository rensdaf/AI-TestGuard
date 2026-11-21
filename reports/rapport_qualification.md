# Rapport de qualification — modèle de détection de fraude bancaire

> **Statut des chiffres.** Les sections marquées **[À COMPLÉTER]** attendent l'exécution sur le vrai jeu Kaggle
> (`python scripts/run_experiments.py` avec `data/creditcard.csv`, qui remplit `reports/resultats.md`).
> Les chiffres « démo synthétique » proviennent du jeu synthétique de `app/synthetic.py` : ils illustrent les
> mécanismes de test et ne prouvent **aucune** performance réelle.

## 1. Contexte et périmètre
Une banque souhaite savoir si un classifieur de fraude par carte peut passer en phase pilote. Les FP bloquent des
paiements légitimes (coût client) ; les FN laissent passer des fraudes (perte financière). Périmètre : données,
prétraitement, modèle, prédiction, sauvegarde/rechargement (voir `plan_de_test.md`). Hors périmètre : API, déploiement,
données bancaires récentes. Jeu : 284 807 transactions, 492 fraudes (0,172 %), 2 jours de données (septembre 2013, Europe).

## 2. Plan de test et traçabilité
17 exigences (T01–T17) reliées à leurs tests et à leurs risques dans `plan_de_test.md`. Les critères d'acceptation
ont été fixés **avant** l'évaluation finale et réglés sur train/validation.

## 3. Architecture et choix méthodologiques
Chargement → `validate_data` (contrat strict) → partition stratifiée 60/20/20 (graine 42) → `Pipeline`
(`RobustScaler` sur `Time` et `Amount` uniquement, V1–V28 déjà issues d'une ACP) + modèle → seuil choisi sur la
validation (précision maximale sous rappel ≥ 80 %) → `joblib` → rechargement → évaluation finale sur le test.
- `class_weight="balanced"` pour compenser le déséquilibre ; `RobustScaler` car `Amount` est très asymétrique.
- Le test n'est utilisé ni pour le modèle, ni pour les hyperparamètres, ni pour le seuil.
- Test anti-fuite : `scaler.center_` doit égaler la médiane du train (et différer de celle du test).

## 4. Tests unitaires et d'intégration
Suite : `test_preprocessing`, `test_prediction`, `test_integration`, `test_robustness`, `test_properties`,
`test_regression`, `test_performance`. Les tests vérifient des comportements (rejets typés, valeurs de métriques sur un
cas calculé à la main, égalité des probabilités avant/après rechargement à 1e-12), pas seulement l'absence d'exception.
**[À COMPLÉTER]** : coller la sortie de `pytest --cov=app` (nombre de tests, % de couverture) et l'URL du run CI.

## 5. Performances du modèle
### 5.1 Pourquoi l'accuracy trompe
Prédire « normale » partout donne 284 315 / 284 807 = **99,83 %** d'accuracy avec un rappel de **0 %**. L'accuracy
est donc rapportée mais jamais utilisée seule ; on s'appuie sur rappel, précision, PR-AUC et FPR/FNR.

### 5.2 Résultats sur le jeu de test (seuil choisi sur la validation)
**[À COMPLÉTER]** avec `reports/resultats.md` (Kaggle) :

| Modèle | Seuil | Précision | Rappel | F1 | ROC-AUC | PR-AUC | FPR | FNR | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Régression logistique | | | | | | | | | | | | |
| Arbre de décision | | | | | | | | | | | | |

Figures : `reports/figures/roc_pr.png`, `reports/figures/confusion.png`.

### 5.3 Grille d'interprétation
- **FN** (fraudes manquées) : perte directe ; mesurés par le FNR = 1 − rappel. Le seuil est abaissé pour les réduire, au prix de FP.
- **FP** : coût d'expérience client et de traitement ; mesurés par le FPR et la précision. Avec ~284 000 transactions normales, un FPR de 0,1 % représente ~284 alertes erronées par 284 000 opérations.
- **ROC-AUC** est optimiste en présence d'une classe rare ; **PR-AUC** est la métrique de comparaison principale.
- Comparer train vs test pour le surapprentissage (arbre plus exposé que la régression logistique).
- Démo synthétique : l'arbre affiche PR-AUC 0,27 (train) / 0,21 (test) contre 0,98 / 0,95 pour la régression logistique, et une précision de 23 % au même rappel : illustration d'un modèle plus complexe mais moins fiable.

## 6. Robustesse et non-régression
### 6.1 Politique et scénarios
| Scénario | Politique | Mesure |
|---|---|---|
| NaN / inf / type incorrect / colonne absente / cible invalide | **Rejeter** (exception explicite) | 100 % rejetés (testé) |
| Colonne supplémentaire | **Ignorer** (règle documentée) | testé |
| Amount extrême (×100, 1e9) | **Traiter**, probabilité valide | Δproba, changements de classe |
| Bruit multiplicatif 5 % sur `Amount`, gaussien σ=0,1 sur V1–V28 | **Traiter**, mesurer la sensibilité | Δproba, flips, Δrappel/précision/FPR |
| Décalage de distribution | **Traiter + alerte** (taux d'alertes) | Δtaux d'alertes, Δrappel |

Les perturbations sont distinctes pour `Amount` (bruit relatif, réaliste pour un montant) et V1–V28 (composantes
anonymisées, bruit absolu faible, amplitude documentée). On mesure la sensibilité, on n'exige pas d'invariance : les
changements de classe doivent se concentrer près du seuil (testé).
Démo synthétique : bruit 5 % sur `Amount` → 0 changement ; décalage +1,5 sur V1–V14 → rappel −72 points et taux d'alertes
divisé par ~10 : **un décalage de distribution est invisible pour les rejets mais dégrade fortement la détection**, d'où le besoin de monitoring (§9).
**[À COMPLÉTER]** : tableau de `reports/resultats.md` sur Kaggle.

### 6.2 Non-régression
Baseline figée `reports/baseline.json` : version du code, hyperparamètres, versions des bibliothèques, graine, seuil, description du
jeu, métriques de validation. Tolérances : PR-AUC −0,03, rappel −0,05, FPR +0,01. La baseline n'est **jamais** recalculée par la CI ;
`test_degraded_model_is_detected` vérifie qu'un modèle dégradé (arbre de profondeur 1) est bien rejeté.
**Décision (rapport comparatif)** : accepter / rejeter / investiguer selon la position vs tolérances. **[À COMPLÉTER]** pour la version proposée.

## 7. Performance et sécurité opérationnelle
Protocole : 300 prédictions isolées après échauffement, lot de 1000, `time.perf_counter`, un processus, machine et OS à noter ;
latence d'une prédiction **isolée** (hors HTTP), à distinguer de celle d'une requête API (HTTP + validation). Seuils proposés :
P95 isolé < 50 ms, modèle < 500 Ko, justifiés par un SLA d'autorisation de paiement de l'ordre de la centaine de millisecondes (à confirmer).
Démo (conteneur) : P95 ≈ 4,5 ms, modèle ≈ 3,5 Ko. **[À COMPLÉTER]** avec les mesures sur votre machine.
Sécurité : rejet des types/champs invalides (testé), absence de secrets dans le dépôt (test + secret scanning GitHub),
pas de journalisation des valeurs de transaction.

## 8. Biais et limites d'équité
Le jeu ne contient ni âge, sexe, localisation ni client : **aucune conclusion d'équité n'est possible**, et il serait
incorrect d'inférer des attributs démographiques à partir de V1–V28. Données nécessaires : segment client, pays/zone de
commerçant, canal (en ligne / présentiel), type de carte, plage d'âge si légalement utilisable. Analyses à mener ensuite : rappel,
FPR et précision par groupe avec intervalles de confiance (bootstrap), via Fairlearn ; attention aux petits effectifs de fraude par groupe.

## 9. Surveillance post-déploiement
| Élément | Métrique | Disponibilité |
|---|---|---|
| Data drift | PSI / KS par variable, distribution de `Amount`, de la probabilité | immédiate |
| Taux d'alertes | part de transactions signalées vs référence | immédiate |
| API | disponibilité, latence P95/P99, taux d'erreurs de validation | immédiate |
| FP/FN, rappel, précision | à partir des labels | retardée (rétrofacturation, enquêtes, jours à semaines) |
| Concept drift | dégradation de PR-AUC/rappel sur labels récents | retardée |

Data drift : la distribution des entrées change. Concept drift : la relation entrée→fraude change (nouveaux modes opératoires), possiblement sans drift visible des entrées.
Alertes simulées : franchissement d'un seuil sur le taux d'alertes ou le PSI → investigation ; chute de rappel sur labels mûris → ré-entraînement.
Prometheus/Grafana en option production.

## 10. Anomalies
| ID | Description | Gravité | Statut |
|---|---|---|---|
| A1 | Un décalage de distribution est accepté sans rejet et dégrade fortement le rappel | Majeure | Ouverte — mitigée par monitoring (§9) |
| A2 | Probabilités de la régression logistique saturées (`class_weight="balanced"`) → seuil proche de 1 | Mineure | Ouverte — envisager calibration (sur validation) |
| A3 | Jeu Kaggle daté (2013) et limité à 2 jours | Majeure | Ouverte — limite de représentativité |
| A4 | Couverture/CI à confirmer sur le dépôt réel | Mineure | **[À COMPLÉTER]** |

## 11. Limites et risques résiduels
Jeu pédagogique ancien, variables anonymisées (pas d'explicabilité métier), 492 fraudes seulement → forte variance des
estimations de rappel (intervalles de confiance à fournir), partition aléatoire et non temporelle (risque d'optimisme), pas d'analyse de coût
monétaire, aucune donnée d'équité, perfs mesurées hors API.

## 12. Décision et réponse à la question de synthèse
**Décision [À COMPLÉTER selon vos résultats]** : *aptitude à poursuivre vers un pilote* si, sur Kaggle, tous les critères
du plan (§4) sont remplis sans avoir touché au test ; *investiguer* si un critère est limite ; *rejeter* sinon.
Réussir les tests techniques ≠ être autorisé à déployer : une validation métier, opérationnelle, sécuritaire et réglementaire (RGPD, DSP2) reste indispensable.

**Conditions pour un pilote** : critères d'acceptation validés par le métier avant évaluation, rappel et FPR dans les bornes avec intervalle
de confiance, pas de surapprentissage notable, non-régression verte, rejets d'entrées invalides à 100 %, latence conforme, mode « shadow » (le
modèle score sans bloquer) avec revue humaine des alertes.
**Preuves supplémentaires avant production** : validation sur des données bancaires récentes avec découpage **temporel** ;
analyse du coût FP/FN en euros pour fixer le seuil ; audit d'équité sur données réelles ; tests de charge en environnement cible ;
supervision (drift, alertes, labels retardés) et plan de ré-entraînement/rollback ; revue sécurité et conformité ; calibration et stabilité du seuil.
