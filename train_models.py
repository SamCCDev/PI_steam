"""
==============================================================================
  TRAIN MODELS — Entrenamiento reproducible de todos los modelos (Dashboard v2)
==============================================================================
  Lee output/dataset_ml.csv (+ diccionario) y entrena, evalúa y serializa:

    models/lr.joblib                Regresión Logística multinomial (base, interpretable)
    models/svm.joblib               SVM kernel RBF (no lineal)
    models/mlp.joblib               Perceptrón multicapa (ReLU)
    models/owners_regressor.joblib  Regresión de owners (log) -> umbrales ajustables
    models/similar.joblib           Preprocesador + NearestNeighbors + metadatos
    models/feature_schema.json      Roles, defaults, categorías, umbrales, prior global
    reports/metrics.json            AUC/F1/accuracy por modelo + balance
    reports/confusion_<modelo>.json Matriz de confusión 3x3 por modelo
    output/model_web.json           Export del LR para el fallback estático (v1)

  Corre IGUAL en la laptop (`python train_models.py`) y en un notebook Databricks
  (ver notebooks/08_train_all.py). Es la fuente única de los artefactos del backend.

  Metodología: split estratificado 80/20 para reportar métricas honestas; luego
  se reentrena cada modelo sobre el 100% de los datos para servir en producción.
==============================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             confusion_matrix, r2_score, mean_absolute_error)
from sklearn.base import clone
import joblib

OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {0: "Flop", 1: "Rentable", 2: "Hit"}
FLOP_MAX_OWNERS = 200000
HIT_MIN_OWNERS = 1000000
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(OUTPUT_DIR / "dataset_ml.csv", sep=";")
    dic = pd.read_csv(OUTPUT_DIR / "dataset_ml_dictionary.csv", sep=";")
    use = dic.set_index("column")["use_as_feature"]
    num_feats = [c for c in dic[dic.role.isin(["feature_numeric", "feature_derived"])].column
                 if c in df.columns and use[c]]
    bool_feats = [c for c in dic[dic.role == "feature_tag"].column if c in df.columns and use[c]]
    cat_feats = [c for c in dic[dic.role == "feature_categorical"].column if c in df.columns and use[c]]
    return df, num_feats, bool_feats, cat_feats


def make_preprocessor(num_feats, bool_feats, cat_feats):
    return ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("bool", "passthrough", bool_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])


def clf_metrics(model, X_te, y_te):
    proba = model.predict_proba(X_te)
    pred = model.predict(X_te)
    return {
        "auc_ovr_macro": round(float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro")), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "confusion": confusion_matrix(y_te, pred).tolist(),
    }


def main():
    df, num_feats, bool_feats, cat_feats = load_data()
    feat_cols = num_feats + bool_feats + cat_feats
    X = df[feat_cols].copy()
    y = df["label"].astype(int)
    print(f"Datos: {len(df):,} juegos | features: {len(feat_cols)} "
          f"({len(num_feats)} num, {len(bool_feats)} bool, {len(cat_feats)} cat)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    classifiers = {
        "lr": LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        "svm": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             max_iter=600, early_stopping=True, random_state=RANDOM_STATE),
    }

    metrics = {"_meta": {
        "n_train": int(len(df)), "features": len(feat_cols),
        "class_balance": df["label_name"].value_counts(normalize=True).round(3).to_dict(),
        "thresholds": {"flop_max": FLOP_MAX_OWNERS, "hit_min": HIT_MIN_OWNERS},
    }}

    for name, clf in classifiers.items():
        pre = make_preprocessor(num_feats, bool_feats, cat_feats)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        print(f"\n[{name}] entrenando (split 80/20)...")
        pipe.fit(X_tr, y_tr)
        m = clf_metrics(pipe, X_te, y_te)
        metrics[name] = {k: v for k, v in m.items() if k != "confusion"}
        (REPORTS_DIR / f"confusion_{name}.json").write_text(
            json.dumps({"labels": ["Flop", "Rentable", "Hit"], "matrix": m["confusion"]}, indent=2),
            encoding="utf-8")
        print(f"    AUC(ovr-macro)={m['auc_ovr_macro']}  F1-macro={m['f1_macro']}  acc={m['accuracy']}")
        # Reentrenar sobre el 100% para servir
        pre_full = make_preprocessor(num_feats, bool_feats, cat_feats)
        pipe_full = Pipeline([("pre", pre_full), ("clf", clf.__class__(**clf.get_params()))])
        pipe_full.fit(X, y)
        joblib.dump(pipe_full, MODELS_DIR / f"{name}.joblib")

    # ── Modelos POST-lanzamiento (mismas features + señales tempranas de recepción) ─
    # No usan volumen crudo de reseñas (sería proxy de owners); sí recepción/engagement.
    POST_EXTRA = [c for c in ["ccu", "rating_porcentaje", "metacritic_score",
                              "ts_positive_ratio", "ts_avg_playtime_hrs", "ts_months_active"]
                  if c in df.columns]
    num_post = num_feats + POST_EXTRA
    Xp = df[num_post + bool_feats + cat_feats].copy()
    Xp_tr, Xp_te = Xp.loc[X_tr.index], Xp.loc[X_te.index]
    print(f"\n[post] señales tempranas: {POST_EXTRA}")
    metrics["post"] = {}
    for name, clf in classifiers.items():
        pipe = Pipeline([("pre", make_preprocessor(num_post, bool_feats, cat_feats)), ("clf", clone(clf))])
        pipe.fit(Xp_tr, y_tr)
        m = clf_metrics(pipe, Xp_te, y_te)
        metrics["post"][name] = {k: v for k, v in m.items() if k != "confusion"}
        print(f"    post-{name}: AUC={m['auc_ovr_macro']}  F1={m['f1_macro']}  acc={m['accuracy']}")
        pf = Pipeline([("pre", make_preprocessor(num_post, bool_feats, cat_feats)), ("clf", clone(clf))])
        pf.fit(Xp, y)
        joblib.dump(pf, MODELS_DIR / f"post_{name}.joblib")

    # ── Regresión de owners (log) -> habilita umbrales ajustables ──────────
    print("\n[owners_regressor] entrenando...")
    y_owners = np.log1p(df["owners_lower_bound"].astype(float))
    yo_tr = np.log1p(df.loc[X_tr.index, "owners_lower_bound"].astype(float))
    yo_te = np.log1p(df.loc[X_te.index, "owners_lower_bound"].astype(float))
    reg = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)),
                    ("reg", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
    reg.fit(X_tr, yo_tr)
    pred_log = reg.predict(X_te)
    metrics["owners_regressor"] = {
        "r2_log": round(float(r2_score(yo_te, pred_log)), 4),
        "mae_owners": int(mean_absolute_error(np.expm1(yo_te), np.expm1(pred_log))),
    }
    print(f"    R2(log)={metrics['owners_regressor']['r2_log']}  "
          f"MAE(owners)={metrics['owners_regressor']['mae_owners']:,}")
    reg_full = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)),
                         ("reg", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
    reg_full.fit(X, y_owners)
    joblib.dump(reg_full, MODELS_DIR / "owners_regressor.joblib")

    # ── Juegos del mismo camino: preprocesador + NearestNeighbors ──────────
    print("\n[similar] ajustando NearestNeighbors...")
    pre_nn = make_preprocessor(num_feats, bool_feats, cat_feats)
    Xt = pre_nn.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=12, metric="cosine").fit(Xt)
    meta_cols = ["appid", "name", "developer", "label", "label_name", "owners_lower_bound", "price"]
    sim_meta = df[meta_cols].reset_index(drop=True)
    joblib.dump({"preprocessor": pre_nn, "nn": nn, "meta": sim_meta,
                 "feat_cols": feat_cols}, MODELS_DIR / "similar.joblib")

    # ── Esquema de features para el backend (defaults, categorías, umbrales) ─
    dev_prior_global = round(float((df["label"] >= 1).mean()), 4)
    schema = {
        "classes": ["Flop", "Rentable", "Hit"],
        "thresholds": {"flop_max": FLOP_MAX_OWNERS, "hit_min": HIT_MIN_OWNERS},
        "numeric": {c: {"median": float(df[c].median()),
                        "min": float(df[c].min()), "max": float(df[c].max())}
                    for c in num_feats},
        "boolean": bool_feats,
        "categorical": {c: {"categories": sorted(df[c].astype(str).unique().tolist()),
                            "default": str(df[c].mode().iloc[0])} for c in cat_feats},
        "dev_prior_global": dev_prior_global,
        "feat_cols": feat_cols,
        "models": ["lr", "svm", "mlp"],
        "post_extra": POST_EXTRA,
        "post_feat_cols": num_post + bool_feats + cat_feats,
        "post_numeric": {c: {"median": float(df[c].median()), "min": float(df[c].min()),
                             "max": float(df[c].max())} for c in POST_EXTRA},
    }
    (MODELS_DIR / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Export del LR para el fallback estático (no toca el HTML) ───────────
    export_lr_web(df, num_feats, bool_feats, cat_feats)

    print("\n" + "=" * 64)
    print("  ENTRENAMIENTO COMPLETO")
    print("=" * 64)
    for k in ["lr", "svm", "mlp"]:
        print(f"  {k:4s}  AUC={metrics[k]['auc_ovr_macro']}  F1={metrics[k]['f1_macro']}  acc={metrics[k]['accuracy']}")
    print(f"  owners_regressor  R2(log)={metrics['owners_regressor']['r2_log']}")
    print(f"  artefactos -> {MODELS_DIR}/  |  métricas -> {REPORTS_DIR}/")
    print("=" * 64)


def export_lr_web(df, num_feats, bool_feats, cat_feats):
    """Replica export_model_web.py con el set de features v2 (para el fallback estático)."""
    pre = make_preprocessor(num_feats, bool_feats, cat_feats)
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    X = df[num_feats + bool_feats + cat_feats]
    model.fit(X, df["label"].astype(int))
    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    classes = [CLASS_NAMES[c] for c in clf.classes_]
    coef = {cls: {names[fi]: float(clf.coef_[ci, fi]) for fi in range(len(names))}
            for ci, cls in enumerate(classes)}
    intercept = {cls: float(clf.intercept_[ci]) for ci, cls in enumerate(classes)}
    scaler = model.named_steps["pre"].named_transformers_["num"]
    numeric = {c: {"mean": float(scaler.mean_[i]), "scale": float(scaler.scale_[i]),
                   "median": float(df[c].median())} for i, c in enumerate(num_feats)}
    categorical = {c: {"categories": [str(v) for v in sorted(df[c].astype(str).unique())],
                       "default": str(df[c].mode().iloc[0])} for c in cat_feats}
    payload = {"classes": classes, "intercept": intercept, "coef": coef,
               "numeric": numeric, "boolean": bool_feats, "categorical": categorical,
               "n_train": int(len(df))}
    (OUTPUT_DIR / "model_web.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
