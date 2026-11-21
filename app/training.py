"""Construction, entraînement, sélection du seuil, sauvegarde du pipeline."""
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

SCALED_COLUMNS = ["Time", "Amount"]  # les V1..V28 sont déjà issues d'une ACP


def build_pipeline(model_name="logreg", seed=42, **params):
    pre = ColumnTransformer(
        [("scale", RobustScaler(), SCALED_COLUMNS)],
        remainder="passthrough", verbose_feature_names_out=False)
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=seed, **params)
    elif model_name == "tree":
        defaults = dict(max_depth=5, min_samples_leaf=20)
        defaults.update(params)
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **defaults)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
    return Pipeline([("prep", pre), ("clf", clf)])


def train_model(X, y, model_name="logreg", seed=42, **params):
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X et y doivent être non vides et de même longueur")
    if len(set(np.asarray(y).tolist())) < 2:
        raise ValueError("La cible doit contenir les deux classes")
    return build_pipeline(model_name, seed, **params).fit(X, y)


def select_threshold(y_val, proba_val, min_recall=0.80):
    """Seuil choisi sur la VALIDATION : précision maximale sous contrainte de rappel minimal.

    Repli (aucun seuil n'atteint le rappel visé) : maximum du F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= min_recall
    if ok.any():
        idx = np.where(ok)[0][np.argmax(precision[ok])]
    else:
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        idx = int(np.argmax(f1))
    return float(thresholds[idx])


def save_pipeline(model, threshold, path, metadata=None):
    joblib.dump({"pipeline": model, "threshold": float(threshold),
                 "metadata": metadata or {}}, path)


def load_pipeline(path):
    return joblib.load(path)
