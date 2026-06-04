# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Entrenamiento consolidado v2 (LR · SVM · MLP) en Databricks
# MAGIC
# MAGIC Versión Databricks de `train_models.py` (la fuente local de los artefactos del
# MAGIC backend). Entrena y **compara los tres modelos** sobre el dataset v2 ampliado y
# MAGIC guarda métricas y coeficientes como tablas de Unity Catalog.
# MAGIC
# MAGIC **Antes de correr:** sube el `output/dataset_ml.csv` **v2** (7.817 juegos, con las
# MAGIC features nuevas: `num_tags`, `pub_experience`, `price_tier`, `release_quarter`,
# MAGIC `is_early_access`, `dev_success_prior`, etc.) a una tabla de Unity Catalog llamada
# MAGIC `dataset_ml` (Catalog UI → Create Table → CSV, separador `;`).
# MAGIC
# MAGIC **Por qué scikit-learn y no PySpark ML:** Free Edition es **serverless (Spark
# MAGIC Connect)** y bloquea la MLlib clásica (Py4JSecurityException). El dataset cabe en
# MAGIC el driver, así que se entrena con scikit-learn sobre `toPandas()`.
# MAGIC
# MAGIC **Nota sobre artefactos:** los `.joblib` que sirve el backend salen de
# MAGIC `train_models.py` corrido en local (Free Edition no permite descargar archivos del
# MAGIC serverless). Este notebook reproduce el entrenamiento y deja métricas/coeficientes
# MAGIC como evidencia y para el dashboard estático.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "default"
DATASET_TABLE = f"{CATALOG}.{SCHEMA}.dataset_ml"
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.model_metrics_v2"
COEF_TABLE    = f"{CATALOG}.{SCHEMA}.model_lr_coefficients_v2"
SEED = 42

# COMMAND ----------

import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             confusion_matrix, classification_report, r2_score)

# Free Edition activa el autolog de MLflow y, en multiclase, calcula roc_auc sin
# multi_class='ovr' -> ValueError. Lo desactivamos. Tampoco usamos start_run()
# (falla en serverless con CONFIG_NOT_AVAILABLE spark.mlflow.modelRegistryUri).
try:
    import mlflow
    mlflow.autolog(disable=True)
except Exception:
    pass

pdf = spark.table(DATASET_TABLE).toPandas()
print(f"dataset_ml: {pdf.shape[0]:,} juegos × {pdf.shape[1]} columnas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Selección de features (inline, sin depender de otras tablas)
# MAGIC Mismas reglas que `build_dataset_v2.py`: se excluyen identificadores, objetivo y
# MAGIC señales POST-lanzamiento (anti-fuga). Se descartan columnas casi-constantes.

# COMMAND ----------

ID_COLS   = ["appid", "name", "developer", "publisher", "release_date", "label", "label_name"]
TARGET    = ["owners_lower_bound", "ccu"]
POSTLAUNCH = ["positive", "negative", "rating_porcentaje", "metacritic_score"]
TEMPORAL  = ["release_year", "release_month"]          # análisis, no features
CAT_FEATS = [c for c in ["dev_experience", "controller_support", "pub_experience",
                         "price_tier", "release_quarter"] if c in pdf.columns]
BOOL_PREFIXES = ("genre_", "cat_", "tag_", "platform_", "is_")

# Asegurar tipos: booleanas/tags a int; numéricas a float
bool_feats = [c for c in pdf.columns if c.startswith(BOOL_PREFIXES)]
pdf[bool_feats] = pdf[bool_feats].fillna(0).astype(int)

excluded = set(ID_COLS + TARGET + POSTLAUNCH + TEMPORAL + CAT_FEATS + bool_feats)
excluded |= {c for c in pdf.columns if c.startswith("ts_")}     # agregados post-lanzamiento
num_feats = [c for c in pdf.select_dtypes(include=[np.number]).columns if c not in excluded]

# Descartar casi-constantes (una categoría domina >= 99.9% o varianza nula)
def near_constant(s):
    return s.nunique(dropna=False) < 2 or s.value_counts(normalize=True, dropna=False).iloc[0] >= 0.999

bool_feats = [c for c in bool_feats if not near_constant(pdf[c])]
num_feats  = [c for c in num_feats if not near_constant(pdf[c])]

print(f"features -> {len(num_feats)} num · {len(bool_feats)} bool · {len(CAT_FEATS)} cat")
X = pdf[num_feats + bool_feats + CAT_FEATS].copy()
y = pdf["label"].astype(int)
balance = pdf["label_name"].value_counts(normalize=True).round(3).to_dict()
print("balance Flop/Rentable/Hit:", balance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split estratificado + preprocesador compartido

# COMMAND ----------

def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("bool", "passthrough", bool_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train={len(X_tr):,}  test={len(X_te):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Entrenar y comparar los tres modelos
# MAGIC LR (con `GridSearchCV` de 3 folds para control de sobreajuste), SVM-RBF y MLP.

# COMMAND ----------

CLASS_NAMES = ["Flop", "Rentable", "Hit"]

def evaluate(model, name):
    proba = model.predict_proba(X_te)
    pred = model.predict(X_te)
    auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
    f1 = f1_score(y_te, pred, average="macro")
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])
    print(f"\n[{name}]  AUC(ovr-macro)={auc:.4f}  F1(macro)={f1:.4f}  Accuracy={acc:.4f}")
    print("Matriz de confusión (filas=real, cols=predicho):")
    print(f"{'':>12}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:>12}" + "".join(f"{cm[i, j]:>10}" for j in range(3)))
    return {"modelo": name, "auc_ovr_macro": round(float(auc), 4),
            "f1_macro": round(float(f1), 4), "accuracy": round(float(acc), 4)}

results = []

# 3.1 Regresión logística con GridSearchCV
lr_grid = GridSearchCV(
    Pipeline([("pre", make_pre()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))]),
    param_grid={"clf__C": [0.1, 1.0, 10.0]}, scoring="roc_auc_ovr", cv=3, n_jobs=-1)
lr_grid.fit(X_tr, y_tr)
print(f"LR mejor C={lr_grid.best_params_['clf__C']}  AUC CV(ovr)={lr_grid.best_score_:.4f}")
lr_best = lr_grid.best_estimator_
results.append(evaluate(lr_best, "lr"))

# 3.2 SVM kernel RBF
svm = Pipeline([("pre", make_pre()),
                ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                            class_weight="balanced", random_state=SEED))])
svm.fit(X_tr, y_tr)
results.append(evaluate(svm, "svm"))

# 3.3 Perceptrón multicapa (ReLU)
mlp = Pipeline([("pre", make_pre()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                      max_iter=600, early_stopping=True, random_state=SEED))])
mlp.fit(X_tr, y_tr)
results.append(evaluate(mlp, "mlp"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tabla comparativa + guardar métricas en Unity Catalog

# COMMAND ----------

metrics_df = pd.DataFrame(results).sort_values("auc_ovr_macro", ascending=False)
print("Comparación de modelos (test):")
display(spark.createDataFrame(metrics_df))

(spark.createDataFrame(metrics_df)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(METRICS_TABLE))
print(f"Métricas guardadas en {METRICS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Coeficientes interpretables del LR (por clase) → tabla UC
# MAGIC Una fila por (clase, feature). `odds_ratio > 1` empuja hacia esa clase.

# COMMAND ----------

clf = lr_best.named_steps["clf"]
names = [n.split("__", 1)[-1] for n in lr_best.named_steps["pre"].get_feature_names_out()]
classes = list(clf.classes_)
CLS = {0: "Flop", 1: "Rentable", 2: "Hit"}
rows = []
for ci, cls in enumerate(classes):
    for fi, fname in enumerate(names):
        rows.append({"clase": CLS.get(cls, str(cls)), "feature": fname,
                     "coef_logodds": float(clf.coef_[ci, fi]),
                     "odds_ratio": float(np.exp(clf.coef_[ci, fi]))})
coef_df = pd.DataFrame(rows)
coef_df["abs_impact"] = coef_df["coef_logodds"].abs()

print("Top factores hacia 'Hit':")
display(spark.createDataFrame(
    coef_df[coef_df.clase == "Hit"].sort_values("abs_impact", ascending=False).head(12)
    [["feature", "coef_logodds", "odds_ratio"]]))

(spark.createDataFrame(coef_df)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(COEF_TABLE))
print(f"Coeficientes guardados en {COEF_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Regresión de owners (log) — habilita umbrales ajustables
# MAGIC Predice `log1p(owners)`; aplicar cortes sobre la predicción reclasifica
# MAGIC Flop/Rentable/Hit sin reentrenar.

# COMMAND ----------

y_log_tr = np.log1p(pdf.loc[X_tr.index, "owners_lower_bound"].astype(float))
y_log_te = np.log1p(pdf.loc[X_te.index, "owners_lower_bound"].astype(float))
reg = Pipeline([("pre", make_pre()), ("reg", HistGradientBoostingRegressor(random_state=SEED))])
reg.fit(X_tr, y_log_tr)
r2 = r2_score(y_log_te, reg.predict(X_te))
print(f"owners_regressor  R²(log)={r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Notas
# MAGIC - Los `.joblib` que consume el backend (`app/`) se generan con `train_models.py` en
# MAGIC   **local** (Free Edition no permite descargar archivos del serverless).
# MAGIC - Este notebook deja en Unity Catalog: `model_metrics_v2` (comparación) y
# MAGIC   `model_lr_coefficients_v2` (coeficientes por clase) como evidencia reproducible.
# MAGIC - Restricciones respetadas de Free Edition serverless: sin MLflow `start_run`,
# MAGIC   sin `.cache()`, sin DBFS, MLlib clásica evitada (todo scikit-learn sobre pandas).
