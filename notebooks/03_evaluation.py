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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report

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
    param_grid={"clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"]},
    scoring="roc_auc", cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print(f"Mejor C: {grid.best_params_['clf__C']}  |  AUC CV: {grid.best_score_:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Métricas en test + registro en MLflow

# COMMAND ----------

import mlflow

proba = best.predict_proba(X_test)[:, 1]
pred = best.predict(X_test)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, pred)
acc = accuracy_score(y_test, pred)

with mlflow.start_run(run_name="lr_cv_sklearn"):
    mlflow.log_param("best_C", grid.best_params_["clf__C"])
    mlflow.log_metrics({"auc": float(auc), "f1": float(f1), "accuracy": float(acc)})

print(f"AUC={auc:.4f}  F1={f1:.4f}  Accuracy={acc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Matriz de confusión (clave con clases desbalanceadas)

# COMMAND ----------

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
print(f"            Pred 0   Pred 1")
print(f"  Real 0    {tn:>6}   {fp:>6}")
print(f"  Real 1    {fn:>6}   {tp:>6}")
print()
print(classification_report(y_test, pred, target_names=["Fracaso", "Éxito"]))
