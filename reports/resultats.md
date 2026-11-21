# Résultats automatiques

Source des données : **Kaggle creditcard.csv**

## Comparaison des modèles (jeu de test, seuil choisi sur la validation)

| Modèle | seuil | accuracy | precision | recall | f1 | roc_auc | pr_auc | fpr | fnr | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.988 | 0.9985 | 0.5419 | 0.8571 | 0.6640 | 0.9725 | 0.7191 | 0.0012 | 0.1429 | 84 | 71 | 14 | 56793 |
| tree | 0.911 | 0.9927 | 0.1735 | 0.8673 | 0.2891 | 0.9172 | 0.6890 | 0.0071 | 0.1327 | 85 | 405 | 13 | 56459 |

## Surapprentissage (train vs test : rappel / PR-AUC)

| Modèle | rappel train | rappel test | PR-AUC train | PR-AUC test |
|---|---|---|---|---|
| logreg | 0.841 | 0.857 | 0.774 | 0.719 |
| tree | 0.925 | 0.867 | 0.758 | 0.689 |

## Robustesse (logreg)

| Scénario | Δproba moy. | changements de classe | Δrappel | Δprécision | ΔFPR | taux d'alertes |
|---|---|---|---|---|---|---|
| Bruit multiplicatif 5 % sur Amount | 0.0025 | 0.0246% | +0.000 | -0.033 | +0.00018 | 0.2897% |
| Bruit gaussien σ=0,1 sur V1..V28 | 0.0159 | 0.0211% | +0.000 | -0.027 | +0.00014 | 0.2862% |
| Amount × 100 (valeurs extrêmes) | 0.7422 | 70.6173% | +0.082 | -0.540 | +0.70725 | 70.8894% |
| Décalage +1,5 sur V1..V14 (changement de distribution) | 0.0623 | 0.0667% | -0.020 | +0.159 | -0.00063 | 0.2054% |

## Entrées invalides

| Scénario | Comportement |
|---|---|
| NaN dans V5 | rejeté : ValueError |
| Inf dans Amount | rejeté : ValueError |
| Colonne V7 absente | rejeté : ValueError |
| Type texte dans V2 | rejeté : TypeError |

## Performance

| Mesure | Valeur |
|---|---|
| p50_ms | 1.887 |
| p95_ms | 2.437 |
| throughput_per_s_batch1000 | 496179.424 |
| model_size_kb | 3.467 |
| rss_mb | 352.678 |
| protocole | 300 prédictions isolées après échauffement, lot de 1000, time.perf_counter, 1 processus |