# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Baseline: Regresión Logística (scikit-learn)
# MAGIC
# MAGIC Modelo base interpretable (Documento Técnico §4.1).
# MAGIC
# MAGIC **Por qué scikit-learn y no PySpark ML:** Free Edition usa **serverless (Spark
# MAGIC Connect)**, que **bloquea la MLlib clásica** de PySpark (`StringIndexer`,
# MAGIC `LogisticRegression`, etc. no están whitelisted en Py4J). Como el dataset es
# MAGIC pequeño (~3k filas), lo traemos a pandas y modelamos con scikit-learn en el
# MAGIC driver (consumo previsto en el Doc. Técnico §4.3).
# MAGIC
# MAGIC Lee `features_silver` de Unity Catalog y guarda los coeficientes (log-odds /
# MAGIC odds ratio) como tabla UC para el dashboard.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "default"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.features_silver"
CATALOG_TABLE = f"{CATALOG}.{SCHEMA}.feature_columns"
COEF_TABLE    = f"{CATALOG}.{SCHEMA}.model_lr_coefficients"

# COMMAND ----------

import numpy as np, pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Free Edition activa el autologging de MLflow por defecto y, al hacer .fit() de un
# modelo MULTICLASE, intenta calcular roc_auc sin multi_class='ovr' -> ValueError.
# Lo desactivamos antes de entrenar (este notebook no necesita MLflow).
mlflow.autolog(disable=True)

# Traer a pandas (dataset pequeño -> cabe en el driver)
pdf = spark.table(SILVER_TABLE).toPandas()
cat = spark.table(CATALOG_TABLE).toPandas()

num_feats  = cat[cat.kind == "numeric"]["feature"].tolist()
bool_feats = cat[cat.kind == "boolean"]["feature"].tolist()
cat_feats  = [c for c in cat[cat.kind == "categorical"]["feature"].tolist() if c in pdf.columns]

X = pdf[num_feats + bool_feats + cat_feats].copy()
y = pdf["label"].astype(int)
print(f"X={X.shape}  positivos={y.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pipeline sklearn
# MAGIC Numéricas -> StandardScaler. Categóricas -> OneHotEncoder. Booleanas tal cual.
# MAGIC `class_weight="balanced"` mitiga el desbalance de clases.

# COMMAND ----------

def build_pipeline(num_feats, bool_feats, cat_feats, C=1.0):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("bool", "passthrough", bool_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=C)
    return Pipeline([("pre", pre), ("clf", clf)])

model = build_pipeline(num_feats, bool_feats, cat_feats)
model.fit(X, y)
print("Modelo entrenado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Coeficientes interpretables por clase (multinomial)
# MAGIC Doc. Técnico §4.1 (Softmax). En multiclase hay un set de coeficientes por clase
# MAGIC (Flop/Rentable/Hit). `odds_ratio > 1` empuja hacia esa clase; `< 1` la aleja.
# MAGIC Formato largo: una fila por (feature, clase).

# COMMAND ----------

CLASS_NAMES = {0: "Flop", 1: "Rentable", 2: "Hit"}

feature_names = model.named_steps["pre"].get_feature_names_out()
clean = [n.split("__", 1)[-1] for n in feature_names]   # quitar prefijos num__/bool__/cat__
coef_matrix = model.named_steps["clf"].coef_            # shape [n_clases, n_features]
intercepts = model.named_steps["clf"].intercept_
classes = list(model.named_steps["clf"].classes_)

rows = []
for ci, cls in enumerate(classes):
    for fi, fname in enumerate(clean):
        rows.append({"clase": CLASS_NAMES.get(cls, str(cls)),
                     "feature": fname,
                     "coef_logodds": float(coef_matrix[ci, fi]),
                     "odds_ratio": float(np.exp(coef_matrix[ci, fi])),
                     "intercept_clase": float(intercepts[ci])})
coef_df = pd.DataFrame(rows)
coef_df["abs_impact"] = coef_df["coef_logodds"].abs()

# Top factores que empujan hacia "Hit" (la clase más valiosa)
print("Top factores hacia 'Hit':")
hit = coef_df[coef_df["clase"] == "Hit"].sort_values("abs_impact", ascending=False).head(12)
display(spark.createDataFrame(hit[["feature", "coef_logodds", "odds_ratio"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Guardar coeficientes (todas las clases) como tabla UC (para el dashboard)

# COMMAND ----------

(spark.createDataFrame(coef_df)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(COEF_TABLE))

print(f"Coeficientes (3 clases) guardados en {COEF_TABLE}. Continuar en 03_evaluation.")
