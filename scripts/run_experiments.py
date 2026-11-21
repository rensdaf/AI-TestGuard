"""Expériences complètes : comparaison de modèles, robustesse, performance.

Usage :  python scripts/run_experiments.py            (data/creditcard.csv si présent, sinon synthétique)
Sorties : reports/metrics.json, reports/resultats.md, reports/figures/*.png
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from app.evaluation import evaluate_model  # noqa: E402
from app.prediction import predict_fraud  # noqa: E402
from app.preprocessing import split_data  # noqa: E402
from app.synthetic import make_synthetic_df  # noqa: E402
from app.training import save_pipeline, select_threshold, train_model  # noqa: E402

SEED, MIN_RECALL = 42, 0.80
OUT = Path("reports")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

csv = Path("data/creditcard.csv")
if csv.exists():
    df, source = pd.read_csv(csv), "Kaggle creditcard.csv"
else:
    df, source = make_synthetic_df(), "SYNTHÉTIQUE (démonstration, pas de résultats réels)"
print("Source :", source)

X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, seed=SEED)
results, models = {}, {}
for name in ["logreg", "tree"]:
    model = train_model(X_tr, y_tr, name, seed=SEED)
    thr = select_threshold(y_va, model.predict_proba(X_va)[:, 1], MIN_RECALL)  # choix sur VALIDATION
    models[name] = (model, thr)
    results[name] = {"threshold": thr}
    for split, X, y in [("train", X_tr, y_tr), ("validation", X_va, y_va), ("test", X_te, y_te)]:
        p, lab = predict_fraud(model, X, thr)
        results[name][split] = evaluate_model(y, lab, p)

# --- Figures (évaluation finale sur le test) ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, (model, thr) in models.items():
    p = model.predict_proba(X_te)[:, 1]
    RocCurveDisplay.from_predictions(y_te, p, name=name, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y_te, p, name=name, ax=ax[1])
ax[0].set_title("Courbes ROC (test)")
ax[1].set_title("Courbes précision-rappel (test)")
fig.tight_layout()
fig.savefig(FIG / "roc_pr.png", dpi=130)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, (name, (model, thr)) in zip(ax, models.items()):
    _, lab = predict_fraud(model, X_te, thr)
    ConfusionMatrixDisplay.from_predictions(y_te, lab, display_labels=["Normale", "Fraude"], ax=a,
                                            colorbar=False)
    a.set_title(f"{name} (seuil={thr:.3f})")
fig.tight_layout()
fig.savefig(FIG / "confusion.png", dpi=130)

# --- Robustesse (modèle logreg) ---
model, thr = models["logreg"]
p0, l0 = predict_fraud(model, X_te, thr)
base = evaluate_model(y_te, l0, p0)
v_cols = [f"V{i}" for i in range(1, 29)]
rng = np.random.default_rng(SEED)


def scenario(name, fn):
    Xp = fn(X_te.copy())
    p, lab = predict_fraud(model, Xp, thr)
    m = evaluate_model(y_te, lab, p)
    return {"scenario": name, "mean_abs_dproba": float(np.abs(p - p0).mean()),
            "class_flip_rate": float((lab != l0).mean()), "d_recall": m["recall"] - base["recall"],
            "d_precision": m["precision"] - base["precision"], "d_fpr": m["fpr"] - base["fpr"],
            "alert_rate": float(lab.mean())}


def bump(cols, delta):
    def f(X):
        X[cols] = X[cols] + delta
        return X
    return f


def noise_amount(X):
    X["Amount"] = X["Amount"] * (1 + rng.normal(0, 0.05, len(X)))
    return X


def noise_v(X):
    X[v_cols] = X[v_cols] + rng.normal(0, 0.1, (len(X), 28))
    return X


def big_amount(X):
    X["Amount"] = X["Amount"] * 100
    return X


rob = [scenario("Bruit multiplicatif 5 % sur Amount", noise_amount),
       scenario("Bruit gaussien σ=0,1 sur V1..V28", noise_v),
       scenario("Amount × 100 (valeurs extrêmes)", big_amount),
       scenario("Décalage +1,5 sur V1..V14 (changement de distribution)", bump(v_cols[:14], 1.5))]
invalid = {}
for name, fn in [("NaN dans V5", lambda X: X.assign(V5=np.where(np.arange(len(X)) == 0, np.nan, X["V5"]))),
                 ("Inf dans Amount", lambda X: X.assign(Amount=np.where(np.arange(len(X)) == 0, np.inf, X["Amount"]))),
                 ("Colonne V7 absente", lambda X: X.drop(columns=["V7"])),
                 ("Type texte dans V2", lambda X: X.astype({"V2": object}).assign(V2="abc"))]:
    try:
        predict_fraud(model, fn(X_te.head(50).copy()), thr)
        invalid[name] = "NON REJETÉ (anomalie)"
    except (ValueError, TypeError) as e:
        invalid[name] = f"rejeté : {type(e).__name__}"

# --- Performance ---
row = X_te.head(1)
predict_fraud(model, row, thr)
lat = []
for _ in range(300):
    t0 = time.perf_counter()
    predict_fraud(model, row, thr)
    lat.append((time.perf_counter() - t0) * 1000)
batch = X_te.head(1000)
t0 = time.perf_counter()
predict_fraud(model, batch, thr)
dt = time.perf_counter() - t0
Path("models").mkdir(exist_ok=True)
save_pipeline(model, thr, "models/pipeline.joblib", {"seed": SEED})
perf = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
        "throughput_per_s_batch1000": len(batch) / dt,
        "model_size_kb": Path("models/pipeline.joblib").stat().st_size / 1024,
        "rss_mb": psutil.Process().memory_info().rss / 1e6,
        "protocole": "300 prédictions isolées après échauffement, lot de 1000, time.perf_counter, 1 processus"}

(OUT / "metrics.json").write_text(json.dumps(
    {"source": source, "seed": SEED, "min_recall_target": MIN_RECALL, "models": results,
     "robustness": rob, "invalid_inputs": invalid, "performance": perf}, indent=2))

keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "tp", "fp", "fn", "tn"]
md = [f"# Résultats automatiques\n\nSource des données : **{source}**\n",
      "## Comparaison des modèles (jeu de test, seuil choisi sur la validation)\n",
      "| Modèle | seuil | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
for n, r in results.items():
    t = r["test"]
    md.append(f"| {n} | {r['threshold']:.3f} | " + " | ".join(f"{t[k]:.4f}" if isinstance(t[k], float) else str(t[k]) for k in keys) + " |")
md += ["\n## Surapprentissage (train vs test : rappel / PR-AUC)\n", "| Modèle | rappel train | rappel test | PR-AUC train | PR-AUC test |", "|---|---|---|---|---|"]
for n, r in results.items():
    md.append(f"| {n} | {r['train']['recall']:.3f} | {r['test']['recall']:.3f} | {r['train']['pr_auc']:.3f} | {r['test']['pr_auc']:.3f} |")
md += ["\n## Robustesse (logreg)\n", "| Scénario | Δproba moy. | changements de classe | Δrappel | Δprécision | ΔFPR | taux d'alertes |", "|---|---|---|---|---|---|---|"]
for r in rob:
    md.append(f"| {r['scenario']} | {r['mean_abs_dproba']:.4f} | {r['class_flip_rate']:.4%} | {r['d_recall']:+.3f} | {r['d_precision']:+.3f} | {r['d_fpr']:+.5f} | {r['alert_rate']:.4%} |")
md += ["\n## Entrées invalides\n", "| Scénario | Comportement |", "|---|---|"] + [f"| {k} | {v} |" for k, v in invalid.items()]
md += ["\n## Performance\n", "| Mesure | Valeur |", "|---|---|"] + [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in perf.items()]
(OUT / "resultats.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
