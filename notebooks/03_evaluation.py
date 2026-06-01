# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Evaluación y Validación Cruzada (scikit-learn)
# MAGIC
# MAGIC Evalúa en test (AUC, F1, matriz de confusión) y hace `GridSearchCV` (3 folds)
# MAGIC para control de sobreajuste (Doc. Técnico §4.4). Lee `features_silver` de UC.
# MAGIC
# MAGIC scikit-learn en el driver (Free Edition serverless bloquea la MLlib clásica).

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "default"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.features_silver"
CATALOG_TABLE = f"{CATALOG}.{SCHEMA}.feature_columns"
SEED = 42

# COMMAND ----------

import numpy as np, pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report

# Free Edition activa el autologging de MLflow por defecto, y al hacer .fit() de un
# modelo MULTICLASE intenta calcular roc_auc sin multi_class='ovr' -> ValueError.
# Lo desactivamos porque hacemos logging manual de las métricas más abajo.
mlflow.autolog(disable=True)

pdf = spark.table(SILVER_TABLE).toPandas()
cat = spark.table(CATALOG_TABLE).toPandas()
num_feats  = cat[cat.kind == "numeric"]["feature"].tolist()
bool_feats = cat[cat.kind == "boolean"]["feature"].tolist()
cat_feats  = [c for c in cat[cat.kind == "categorical"]["feature"].tolist() if c in pdf.columns]

X = pdf[num_feats + bool_feats + cat_feats].copy()
y = pdf["label"].astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Split estratificado + pipeline

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"train={len(X_train):,}  test={len(X_test):,}  positivos train={y_train.mean():.1%}")

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("bool", "passthrough", bool_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])
pipe = Pipeline([("pre", pre),
                 ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. GridSearchCV (3 folds) — control de sobreajuste

# COMMAND ----------

grid = GridSearchCV(
    pipe,
    param_grid={"clf__C": [0.1, 1.0, 10.0]},
    scoring="roc_auc_ovr", cv=3, n_jobs=-1)   # OVR para multiclase
grid.fit(X_train, y_train)
best = grid.best_estimator_
print(f"Mejor C: {grid.best_params_['clf__C']}  |  AUC CV (ovr): {grid.best_score_:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Métricas multiclase en test + registro en MLflow

# COMMAND ----------

proba = best.predict_proba(X_test)            # [n, 3]
pred = best.predict(X_test)
auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
f1 = f1_score(y_test, pred, average="macro")
acc = accuracy_score(y_test, pred)

with mlflow.start_run(run_name="lr_cv_multiclase"):
    mlflow.log_param("best_C", grid.best_params_["clf__C"])
    mlflow.log_metrics({"auc_ovr_macro": float(auc), "f1_macro": float(f1), "accuracy": float(acc)})

print(f"AUC(ovr-macro)={auc:.4f}  F1(macro)={f1:.4f}  Accuracy={acc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Matriz de confusión 3×3 + reporte por clase

# COMMAND ----------

CLASS_NAMES = ["Flop", "Rentable", "Hit"]
cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
print("Matriz de confusión (filas=real, columnas=predicho):")
print(f"{'':>12}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"{n:>12}" + "".join(f"{cm[i, j]:>10}" for j in range(3)))
print()
print(classification_report(y_test, pred, labels=[0, 1, 2], target_names=CLASS_NAMES))
