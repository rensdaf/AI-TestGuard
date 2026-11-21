# Plan de test — AI-TestGuard (détection de fraude par carte bancaire)

## 1. Périmètre
| Composant | Fonction / module | Inclus |
|---|---|---|
| Données | `validate_data`, contrat de schéma | oui |
| Prétraitement | `preprocess_data`, `split_data`, `RobustScaler` (Time, Amount) dans un `Pipeline` | oui |
| Modèle | `train_model` (régression logistique, arbre), `select_threshold` | oui |
| Prédiction | `predict_fraud` (probabilité + décision au seuil) | oui |
| Intégration | sauvegarde/rechargement `joblib`, chaîne complète | oui |
| API HTTP, déploiement, MLOps | — | hors périmètre (prototype) ; latence API à mesurer si déployée |

## 2. Matrice d'exigences et de traçabilité
| ID | Exigence | Type | Risque maîtrisé | Résultat attendu | Test(s) |
|---|---|---|---|---|---|
| T01 | Colonnes nécessaires présentes | Données | Prédiction faussée/plantage silencieux | Rejet explicite si colonne absente | `test_missing_column_rejected` |
| T02 | Types conformes au contrat | Unitaire | Conversions implicites erronées | `TypeError` si non numérique | `test_wrong_type_rejected` |
| T03 | Aucune valeur non finie en sortie du prétraitement | Unitaire | Propagation de NaN/inf dans le modèle | Aucun NaN/inf | `test_non_finite_rejected`, `test_no_non_finite_after_preprocessing` |
| T04 | Probabilité valide | Fonctionnel ML | Scores inexploitables | 0 ≤ p ≤ 1, finie | `test_probabilities_are_valid`, `test_properties` |
| T05 | Évaluation sur données indépendantes | ML | Optimisme, fuite de données | Test jamais utilisé pour le réglage | `test_split_*`, `test_scaler_fitted_on_train_only` |
| T06 | Résultats reproductibles | Non-régression | Résultats non rejouables | Écart nul à graine fixe | `test_reproducibility_same_seed`, `test_split_reproducible_with_seed` |
| T07 | Tests exécutés sur GitHub | CI | Régression non détectée | Workflow vert ou échec signalé | `tests.yml` |
| T08 | Cible ∈ {0,1} | Données | Étiquettes corrompues | Rejet | `test_invalid_target_rejected` |
| T09 | Montant `Amount` ≥ 0 | Données | Incohérence métier, valeurs aberrantes | Rejet | `test_negative_amount_rejected` |
| T10 | Pipeline identique après sauvegarde/rechargement | Intégration | Modèle déployé ≠ modèle validé | Écart ≤ 1e-12 | `test_end_to_end_save_reload` |
| T11 | Partition stratifiée, disjointe, complète | Données | Test sans fraude, biais d'échantillonnage | Proportion de fraudes conservée (±0,2 pt) | `test_split_is_disjoint_stratified_and_complete` |
| T12 | Métriques fiables sur classes déséquilibrées | ML | Illusion d'accuracy | Valeurs exactes sur cas connu ; accuracy ≠ qualité | `test_evaluate_model_known_values`, `test_accuracy_paradox` |
| T13 | Seuil choisi sur la validation avec contrainte de rappel | ML | Fuite via le seuil | Rappel validation ≥ cible | `test_select_threshold_respects_min_recall` |
| T14 | Robustesse aux entrées perturbées / extrêmes | Robustesse | Sorties instables ou invalides | Sensibilité bornée, rejets explicites | `test_robustness.py` |
| T15 | Non-régression vs baseline figée | Non-régression | Dégradation silencieuse | PR-AUC −0,03 / rappel −0,05 / FPR +0,01 max | `test_regression.py` |
| T16 | Latence et taille de modèle | Performance | Non-respect des contraintes opérationnelles | P95 isolé < 50 ms ; modèle < 500 Ko | `test_performance.py` |
| T17 | Pas de secret dans le dépôt | Sécurité | Fuite d'identifiants | Aucun motif sensible | `test_no_secret_in_repo` (+ secret scanning GitHub) |

Politique d'entrée invalide (contrat) : **rejet explicite** (NaN, inf, type, schéma, cible, montant négatif, vide) ;
colonnes supplémentaires **ignorées** (règle documentée) ; décalage de distribution : **traité + alerte** (suivi du taux d'alertes).

## 3. Outils et preuves
pytest, pytest-cov (`coverage.xml`), Hypothesis, Ruff, scikit-learn, joblib, psutil, `time.perf_counter`, GitHub Actions.
Preuves : sortie pytest, `coverage.xml` (artefact CI), `reports/metrics.json`, `reports/resultats.md`, `reports/figures/`, `reports/baseline.json`, URLs des runs CI (réussi + échec volontaire).

## 4. Critères d'acceptation / rejet (fixés a priori, réglés sur train/validation)
Ces valeurs sont des **propositions** à faire valider par le métier avant de regarder le test.

| Critère | Acceptation pour pilote | Rejet |
|---|---|---|
| Tests unitaires/intégration | 100 % passés, couverture `app/` ≥ 80 % | tout échec |
| Rappel (validation, au seuil retenu) | ≥ 75 % | < 60 % |
| Taux de faux positifs | ≤ 0,1 % | > 0,5 % |
| PR-AUC (validation) | ≥ 0,70 | < 0,50 |
| Écart rappel train − test | ≤ 10 points | > 20 points |
| Robustesse | aucune entrée invalide acceptée ; Δrappel ≤ 5 pts sous bruit réaliste | entrée invalide acceptée |
| Non-régression | dans les tolérances de la baseline | hors tolérance → investigation |
| Latence P95 (prédiction isolée) | < 50 ms (à ajuster au SLA) | > 200 ms |

Entre acceptation et rejet : **investiguer**.
