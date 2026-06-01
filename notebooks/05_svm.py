# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Support Vector Machine (kernel RBF)
# MAGIC
# MAGIC Fase de pulido (Documento Técnico §4.1.2). SVM con kernel **RBF** para capturar
# MAGIC fronteras no lineales en la alta dimensionalidad de la matriz de tags.
# MAGIC
# MAGIC Con scikit-learn (`SVC(kernel="rbf")`) **sí** hay kernel radial — la limitación
# MAGIC era de PySpark `LinearSVC`. Modela sobre los datos en pandas (serverless).

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
from sklearn.svm import SVC
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
# MAGIC ## 1. Pipeline SVM-RBF
# MAGIC El escalado es **crítico** para SVM (kernel sensible a la escala). `probability=True`
# MAGIC habilita `predict_proba` para el AUC. `class_weight="balanced"` por el desbalance.

# COMMAND ----------

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("bool", "passthrough", bool_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])
svm = Pipeline([("pre", pre),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                            class_weight="balanced", probability=True, random_state=SEED))])
svm.fit(X_train, y_train)
print("SVM-RBF entrenado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluación + MLflow

# COMMAND ----------

import mlflow

CLASS_NAMES = ["Flop", "Rentable", "Hit"]
proba = svm.predict_proba(X_test)            # [n, 3]
pred = svm.predict(X_test)
auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
f1 = f1_score(y_test, pred, average="macro")
acc = accuracy_score(y_test, pred)

with mlflow.start_run(run_name="svm_rbf"):
    mlflow.log_params({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
    mlflow.log_metrics({"auc_ovr_macro": float(auc), "f1_macro": float(f1), "accuracy": float(acc)})

print(f"SVM-RBF -> AUC(ovr-macro)={auc:.4f}  F1(macro)={f1:.4f}  Accuracy={acc:.4f}\n")
cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
print(f"{'':>12}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
for i, n in enumerate(CLASS_NAMES):
    print(f"{n:>12}" + "".join(f"{cm[i, j]:>10}" for j in range(3)))
print()
print(classification_report(y_test, pred, labels=[0, 1, 2], target_names=CLASS_NAMES))

# COMMAND ----------

# MAGIC %md
# MAGIC **Nota:** SVM-RBF no da coeficientes interpretables (frontera no lineal). Para
# MAGIC interpretación usar el notebook 02 (Regresión Logística). Comparar AUC contra LR.
