"""
==============================================================================
  TRAIN MODELS — Fuente ÚNICA de entrenamiento (local y Databricks)
==============================================================================
  Un solo archivo entrena los tres modelos comparados. Detecta dónde corre:

    • LOCAL       lee output/dataset_ml.csv  →  guarda models/*.joblib + reports/
    • DATABRICKS  lee la tabla Unity Catalog `dataset_ml` (vía Spark) → guarda las
                  métricas y los coeficientes como tablas UC (evidencia reproducible)

  La selección de features y la configuración de los modelos son IDÉNTICAS en
  ambos entornos, así que el resultado es el mismo: los .joblib que sirve el
  backend reproducen exactamente lo que se ve en Databricks. (Free Edition no
  permite exportar archivos del serverless, por eso los .joblib se generan en
  local; el notebook de Databricks importa este mismo módulo — ver notebooks/08.)

  Artefactos (modo local):
    models/lr.joblib · svm.joblib · mlp.joblib   LR multinomial · SVM-RBF · MLP
    models/owners_regressor.joblib               regresión de owners -> umbrales
    models/similar.joblib                        preprocesador + NearestNeighbors
    models/feature_schema.json                   roles, defaults, categorías, umbrales
    reports/metrics.json · confusion_*.json      métricas y matrices por modelo
    output/model_web.json                        coeficientes LR (panel analítico)

  Metodología: split estratificado 80/20 para métricas honestas; cada modelo se
  reentrena sobre el 100% de los datos para servir en producción.
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
import joblib

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CLASS_NAMES = {0: "Flop", 1: "Rentable", 2: "Hit"}
FLOP_MAX_OWNERS = 200000
HIT_MIN_OWNERS = 1000000
RANDOM_STATE = 42

# Tabla Unity Catalog (solo Databricks). Ajustar catálogo/esquema si no son los de Free Edition.
UC_DATASET = "workspace.default.dataset_ml"
UC_METRICS = "workspace.default.model_metrics"
UC_COEF = "workspace.default.model_lr_coefficients"


# ── Carga del dataset según el entorno ────────────────────────────────────
def load_dataset():
    """Devuelve (df, entorno). LOCAL si existe el CSV; DATABRICKS si hay un Spark activo."""
    csv = OUTPUT_DIR / "dataset_ml.csv"
    if csv.exists():
        return pd.read_csv(csv, sep=";"), "local"
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is not None:
            return spark.table(UC_DATASET).toPandas(), "databricks"
    except Exception:
        pass
    raise FileNotFoundError(
        "No encontré output/dataset_ml.csv (local) ni una sesión Spark con la tabla "
        f"{UC_DATASET} (Databricks). Genera el dataset con build_dataset.py primero.")


# ── Selección de features (idéntica en local y Databricks) ────────────────
def select_features(df):
    """Reglas deterministas sobre el propio DataFrame (no dependen del diccionario),
    para que local y Databricks elijan EXACTAMENTE las mismas columnas.
    Excluye identificadores, objetivo y todo lo post-lanzamiento (anti-fuga)."""
    id_cols = ["appid", "name", "developer", "publisher", "release_date", "label", "label_name"]
    target = ["owners_lower_bound", "ccu"]
    postlaunch = ["positive", "negative", "rating_porcentaje", "metacritic_score"]
    temporal = ["release_year", "release_month"]            # se usan para análisis, no como feature
    cat_feats = [c for c in ["dev_experience", "controller_support", "pub_experience",
                             "price_tier", "release_quarter"] if c in df.columns]
    bool_feats = [c for c in df.columns if c.startswith(("genre_", "cat_", "tag_", "platform_", "is_"))]
    df[bool_feats] = df[bool_feats].fillna(0).astype(int)

    excluded = set(id_cols + target + postlaunch + temporal + cat_feats + bool_feats)
    excluded |= {c for c in df.columns if c.startswith("ts_")}     # agregados post-lanzamiento
    num_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]

    def near_constant(s):
        return s.nunique(dropna=False) < 2 or s.value_counts(normalize=True, dropna=False).iloc[0] >= 0.999
    bool_feats = [c for c in bool_feats if not near_constant(df[c])]
    num_feats = [c for c in num_feats if not near_constant(df[c])]
    return num_feats, bool_feats, cat_feats


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
        "confusion": confusion_matrix(y_te, pred, labels=[0, 1, 2]).tolist(),
    }


def build_classifiers():
    """Los tres modelos comparados, con configuración fija (sin búsqueda de hiperparámetros)
    para que el resultado sea determinista y reproducible en cualquier entorno."""
    return {
        "lr": LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        "svm": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             max_iter=600, early_stopping=True, random_state=RANDOM_STATE),
    }


# ── Entrenamiento (núcleo compartido) ─────────────────────────────────────
def run(df, env):
    num_feats, bool_feats, cat_feats = select_features(df)
    feat_cols = num_feats + bool_feats + cat_feats
    X = df[feat_cols].copy()
    y = df["label"].astype(int)
    print(f"[{env}] {len(df):,} juegos | features: {len(feat_cols)} "
          f"({len(num_feats)} num, {len(bool_feats)} bool, {len(cat_feats)} cat)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    metrics = {"_meta": {
        "n_train": int(len(df)), "features": len(feat_cols),
        "class_balance": df["label_name"].value_counts(normalize=True).round(3).to_dict(),
        "thresholds": {"flop_max": FLOP_MAX_OWNERS, "hit_min": HIT_MIN_OWNERS},
    }}
    confusions = {}

    for name, clf in build_classifiers().items():
        pipe = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)), ("clf", clf)])
        print(f"\n[{name}] entrenando (split 80/20)...")
        pipe.fit(X_tr, y_tr)
        m = clf_metrics(pipe, X_te, y_te)
        metrics[name] = {k: v for k, v in m.items() if k != "confusion"}
        confusions[name] = {"labels": ["Flop", "Rentable", "Hit"], "matrix": m["confusion"]}
        print(f"    AUC(ovr-macro)={m['auc_ovr_macro']}  F1-macro={m['f1_macro']}  acc={m['accuracy']}")
        if env == "local":
            full = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)),
                             ("clf", clf.__class__(**clf.get_params()))])
            full.fit(X, y)
            joblib.dump(full, MODELS_DIR / f"{name}.joblib")

    # ── Regresión de owners (log) -> umbrales ajustables ──
    print("\n[owners_regressor] entrenando...")
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

    coef_rows = _lr_coefficients(df, num_feats, bool_feats, cat_feats)

    if env == "local":
        _save_local(df, num_feats, bool_feats, cat_feats, X, y,
                    metrics, confusions, coef_rows)
    else:
        _save_databricks(metrics, coef_rows)

    print("\n" + "=" * 64)
    print("  ENTRENAMIENTO COMPLETO")
    print("=" * 64)
    for k in ["lr", "svm", "mlp"]:
        print(f"  {k:4s}  AUC={metrics[k]['auc_ovr_macro']}  F1={metrics[k]['f1_macro']}  acc={metrics[k]['accuracy']}")
    print(f"  owners_regressor  R2(log)={metrics['owners_regressor']['r2_log']}")
    return metrics


def _lr_coefficients(df, num_feats, bool_feats, cat_feats):
    """Coeficientes del LR por clase (interpretabilidad). Devuelve lista de filas."""
    model = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))])
    model.fit(df[num_feats + bool_feats + cat_feats], df["label"].astype(int))
    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    rows = []
    for ci, cls in enumerate(clf.classes_):
        for fi, fname in enumerate(names):
            coef = float(clf.coef_[ci, fi])
            rows.append({"clase": CLASS_NAMES.get(int(cls), str(cls)), "feature": fname,
                         "coef_logodds": coef, "odds_ratio": float(np.exp(coef))})
    return rows


def _save_local(df, num_feats, bool_feats, cat_feats, X, y, metrics, confusions, coef_rows):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    feat_cols = num_feats + bool_feats + cat_feats

    # owners_regressor sobre el 100%
    reg_full = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)),
                         ("reg", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
    reg_full.fit(X, np.log1p(df["owners_lower_bound"].astype(float)))
    joblib.dump(reg_full, MODELS_DIR / "owners_regressor.joblib")

    # juegos del mismo camino
    print("[similar] ajustando NearestNeighbors...")
    pre_nn = make_preprocessor(num_feats, bool_feats, cat_feats)
    Xt = pre_nn.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=12, metric="cosine").fit(Xt)
    meta_cols = ["appid", "name", "developer", "label", "label_name", "owners_lower_bound", "price"]
    joblib.dump({"preprocessor": pre_nn, "nn": nn, "meta": df[meta_cols].reset_index(drop=True),
                 "feat_cols": feat_cols}, MODELS_DIR / "similar.joblib")

    # esquema para el backend
    schema = {
        "classes": ["Flop", "Rentable", "Hit"],
        "thresholds": {"flop_max": FLOP_MAX_OWNERS, "hit_min": HIT_MIN_OWNERS},
        "numeric": {c: {"median": float(df[c].median()),
                        "min": float(df[c].min()), "max": float(df[c].max())} for c in num_feats},
        "boolean": bool_feats,
        "categorical": {c: {"categories": sorted(df[c].astype(str).unique().tolist()),
                            "default": str(df[c].mode().iloc[0])} for c in cat_feats},
        "dev_prior_global": round(float((df["label"] >= 1).mean()), 4),
        "feat_cols": feat_cols,
        "models": ["lr", "svm", "mlp"],
    }
    (MODELS_DIR / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, cm in confusions.items():
        (REPORTS_DIR / f"confusion_{name}.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")

    # model_web.json para el panel analítico (importancia de variables)
    _export_model_web(df, num_feats, bool_feats, cat_feats, coef_rows)
    print(f"artefactos -> {MODELS_DIR}/  |  métricas -> {REPORTS_DIR}/")


def _export_model_web(df, num_feats, bool_feats, cat_feats, coef_rows):
    scaler = make_preprocessor(num_feats, bool_feats, cat_feats)
    scaler.fit(df[num_feats + bool_feats + cat_feats])
    num_tr = scaler.named_transformers_["num"]
    coef = {}
    intercept = {}
    for r in coef_rows:
        coef.setdefault(r["clase"], {})[r["feature"]] = r["coef_logodds"]
    # intercepto: recalculado aparte (rápido, mismo modelo)
    model = Pipeline([("pre", make_preprocessor(num_feats, bool_feats, cat_feats)),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))])
    model.fit(df[num_feats + bool_feats + cat_feats], df["label"].astype(int))
    clf = model.named_steps["clf"]
    intercept = {CLASS_NAMES[int(c)]: float(clf.intercept_[i]) for i, c in enumerate(clf.classes_)}
    numeric = {c: {"mean": float(num_tr.mean_[i]), "scale": float(num_tr.scale_[i]),
                   "median": float(df[c].median())} for i, c in enumerate(num_feats)}
    categorical = {c: {"categories": [str(v) for v in sorted(df[c].astype(str).unique())],
                       "default": str(df[c].mode().iloc[0])} for c in cat_feats}
    payload = {"classes": list(coef.keys()), "intercept": intercept, "coef": coef,
               "numeric": numeric, "boolean": bool_feats, "categorical": categorical,
               "n_train": int(len(df))}
    (OUTPUT_DIR / "model_web.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _save_databricks(metrics, coef_rows):
    """Guarda métricas y coeficientes como tablas Unity Catalog (evidencia reproducible)."""
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    rows = [{"modelo": k, **{kk: vv for kk, vv in metrics[k].items()}} for k in ["lr", "svm", "mlp"]]
    (spark.createDataFrame(pd.DataFrame(rows))
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(UC_METRICS))
    (spark.createDataFrame(pd.DataFrame(coef_rows))
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(UC_COEF))
    print(f"métricas -> {UC_METRICS}  |  coeficientes -> {UC_COEF}")


def main():
    df, env = load_dataset()
    run(df, env)


if __name__ == "__main__":
    main()
