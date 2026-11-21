"""Jeu synthétique (même schéma que Kaggle) pour tests rapides, CI et démonstration.
Ne remplace PAS le vrai jeu de données pour les résultats du rapport."""
import numpy as np
import pandas as pd


def make_synthetic_df(n=20000, fraud_rate=0.01, seed=42):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    V = rng.normal(size=(n, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 9, 13, 16]] = [-2.0, -2.0, 1.5, -1.8, -2.0, -1.5]
    V[y == 1] += shift
    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.lognormal(3.5, 1.2, n)
    df["Class"] = y
    return df
