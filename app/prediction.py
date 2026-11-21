"""Prédiction : validation de l'entrée puis probabilité + décision au seuil."""
from app.preprocessing import FEATURE_COLUMNS, validate_data


def predict_fraud(model, X, threshold=0.5):
    """Retourne (probabilités de fraude, étiquettes 0/1). Rejette les entrées invalides."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Le seuil doit être dans [0, 1]")
    validate_data(X, require_target=False)
    proba = model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    return proba, (proba >= threshold).astype(int)
