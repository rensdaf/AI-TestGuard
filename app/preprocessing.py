"""Contrôle du schéma, validation et préparation des données (contrat de données)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]


def validate_data(df, require_target=True):
    """Contrat : on REJETTE (ValueError/TypeError) toute donnée invalide, aucune correction silencieuse.

    Les colonnes supplémentaires sont tolérées et ignorées par preprocess_data.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame pandas")
    if df.empty:
        raise ValueError("DataFrame vide")
    if df.columns.duplicated().any():
        raise ValueError("Colonnes dupliquées")
    required = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    for col in required:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            raise TypeError(f"Type non numérique pour la colonne {col} : {dtype}")
    values = df[required].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Valeurs manquantes (NaN) détectées")
    if np.isinf(values).any():
        raise ValueError("Valeurs infinies détectées")
    if (df["Amount"] < 0).any():
        raise ValueError("Montant (Amount) négatif")
    if require_target and not df[TARGET].isin([0, 1]).all():
        raise ValueError("La cible doit valoir 0 ou 1")
    return df


def preprocess_data(df):
    """Retourne (X, y) : sélection/ordonnancement des colonnes, sans transformation apprise.

    Les transformations apprises (mise à l'échelle) vivent dans le Pipeline sklearn
    et sont donc ajustées uniquement sur l'entraînement.
    """
    validate_data(df)
    X = df[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    return X, y


def split_data(df, seed=42, val_size=0.2, test_size=0.2):
    """Partition stratifiée train / validation / test, AVANT toute transformation apprise."""
    X, y = preprocess_data(df)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
