# AI-TestGuard — qualification d'un modèle de détection de fraude

Stratégie de test reproductible pour un pipeline ML (Credit Card Fraud, Kaggle) : tests unitaires, intégration,
robustesse, non-régression, performance, CI GitHub Actions.

## Démarrage
```bash
pip install -r requirements.txt
# optionnel : placer data/creditcard.csv (voir data/README.md)
pytest --cov=app                          # tests rapides (jeu synthétique, graine 42)
ruff check .                              # style
PYTHONPATH=. python scripts/run_experiments.py   # modèles, robustesse, perf -> reports/
```

## Structure
| Dossier | Rôle |
|---|---|
| `app/` | validation/prétraitement, entraînement, prédiction, évaluation |
| `tests/` | unitaires, intégration, robustesse, non-régression, performance |
| `scripts/make_baseline.py` | crée **une fois** `reports/baseline.json` (jamais recalculée en CI) |
| `scripts/run_experiments.py` | comparaison logreg/arbre, figures, robustesse, latence |
| `reports/` | plan de test, rapport de qualification, résultats |

## Décisions méthodologiques clés
- Partition stratifiée train/val/test **avant** toute transformation apprise ; scaler dans un `Pipeline`.
- Seuil choisi sur la **validation** (précision max sous rappel ≥ 80 %), test réservé à l'évaluation finale.
- Contrat de données : les entrées invalides sont **rejetées** (aucune correction silencieuse).

## Preuves CI (à compléter)
- Exécution réussie : _URL du run_
- Échec volontaire détecté (branche `demo-echec`, ex. `assert False` ou baisse du seuil de couverture) : _URL du run_
