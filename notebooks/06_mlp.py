# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Perceptrón Multicapa (MLP)
# MAGIC
# MAGIC Fase de pulido (Documento Técnico §4.2.3). Red neuronal densa para capturar
# MAGIC interacciones no lineales (precio × tags × experiencia del estudio).
# MAGIC
# MAGIC Con scikit-learn (`MLPClassifier`) **sí** hay activación ReLU/Tanh — la
# MAGIC limitación (solo Sigmoide) era de PySpark. Modela en pandas (serverless).

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
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report

pdf = spark.table(SILVER_TABLE).toPandas()
cat = spark.table(CATALOG_TABLE).toPandas()
num_feats  = cat[cat.kind == "numeric"]["feature"].tolist()
bool_feats = cat[cat.kind == "boolean"]["feature"].tolist()
cat_feats  = [c for c in cat[cat.kind == "categorical"]["feature"].tolist() if c in pdf.columns]

X = pdf[num_feats + bool_feats + cat_feats].copy()
y = pdf["label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pipeline MLP (capas densas + ReLU)
# MAGIC Arquitectura `[input, 64, 32, 1]`. El escalado es necesario para que converja.
# MAGIC `MLPClassifier` no tiene `class_weight`; usamos `early_stopping` para regularizar
# MAGIC y reportamos métricas robustas al desbalance (AUC/F1, no solo accuracy).

# COMMAND ----------

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("bool", "passthrough", bool_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])
mlp = Pipeline([("pre", pre),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                      solver="adam", alpha=1e-3, max_iter=300,
                                      early_stopping=True, random_state=SEED))])
mlp.fit(X_train, y_train)
print(f"MLP entrenado. Capas: {mlp.named_steps['clf'].hidden_layer_sizes}, "
      f"iteraciones: {mlp.named_steps['clf'].n_iter_}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluación + MLflow

# COMMAND ----------

import mlflow

proba = mlp.predict_proba(X_test)[:, 1]
pred = mlp.predict(X_test)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, pred)
acc = accuracy_score(y_test, pred)

with mlflow.start_run(run_name="mlp_relu"):
    mlflow.log_params({"hidden_layers": "64,32", "activation": "relu", "alpha": 1e-3})
    mlflow.log_metrics({"auc": float(auc), "f1": float(f1), "accuracy": float(acc)})

print(f"MLP -> AUC={auc:.4f}  F1={f1:.4f}  Accuracy={acc:.4f}\n")
cm = confusion_matrix(y_test, pred); tn, fp, fn, tp = cm.ravel()
print(f"            Pred 0   Pred 1")
print(f"  Real 0    {tn:>6}   {fp:>6}")
print(f"  Real 1    {fn:>6}   {tp:>6}\n")
print(classification_report(y_test, pred, target_names=["Fracaso", "Éxito"]))
