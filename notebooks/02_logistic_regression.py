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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
# MAGIC ## 2. Coeficientes interpretables (log-odds y odds ratio)
# MAGIC Doc. Técnico §4.1: cuánto aporta cada feature a la probabilidad de éxito.
# MAGIC `odds_ratio > 1` favorece el éxito; `< 1` lo penaliza.

# COMMAND ----------

feature_names = model.named_steps["pre"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
intercept = float(model.named_steps["clf"].intercept_[0])

# Limpiar prefijos del ColumnTransformer (num__, bool__, cat__) para legibilidad
clean = [n.split("__", 1)[-1] for n in feature_names]

coef_df = pd.DataFrame({"feature": clean, "coef_logodds": coefs})
coef_df["odds_ratio"] = np.exp(coef_df["coef_logodds"])
coef_df["abs_impact"] = coef_df["coef_logodds"].abs()
coef_df = coef_df.sort_values("abs_impact", ascending=False).reset_index(drop=True)

print(f"Intercepto (log-odds base): {intercept:.4f}")
display(spark.createDataFrame(coef_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Guardar coeficientes como tabla UC (para el dashboard)

# COMMAND ----------

coef_out = coef_df.copy()
coef_out["intercept"] = intercept
(spark.createDataFrame(coef_out)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(COEF_TABLE))

print(f"Coeficientes guardados en {COEF_TABLE}. Continuar en 03_evaluation.")
