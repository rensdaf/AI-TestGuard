# Données

Le jeu **Credit Card Fraud Detection** (Kaggle, ULB Machine Learning Group) n'est **pas versionné** (150 Mo, conditions de la source).

1. Télécharger `creditcard.csv` : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Le placer dans `data/creditcard.csv` (ignoré par Git via `.gitignore`).

Schéma : `Time`, `V1`..`V28` (composantes ACP anonymisées), `Amount`, `Class` (0 normale, 1 fraude).
284 807 transactions dont 492 fraudes (0,172 %).

Sans ce fichier, les tests et la CI utilisent `app/synthetic.py` (jeu synthétique au même schéma) : utile pour
vérifier le code, **jamais** pour conclure sur les performances.
