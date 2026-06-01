# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Simulador Comercial (scikit-learn)
# MAGIC
# MAGIC Simulador pre-lanzamiento (Doc. Técnico §5). El usuario fija precio, experiencia
# MAGIC del estudio y mecánicas (tags); devuelve probabilidad de éxito y recomendaciones.
# MAGIC
# MAGIC Modela con scikit-learn en el driver (Free Edition serverless bloquea la MLlib
# MAGIC clásica). Entrena en segundos desde `features_silver`; sin artefactos de archivo.

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

pdf = spark.table(SILVER_TABLE).toPandas()
cat = spark.table(CATALOG_TABLE).toPandas()
num_feats  = cat[cat.kind == "numeric"]["feature"].tolist()
bool_feats = cat[cat.kind == "boolean"]["feature"].tolist()
cat_feats  = [c for c in cat[cat.kind == "categorical"]["feature"].tolist() if c in pdf.columns]
FEATS = num_feats + bool_feats + cat_feats

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Entrenar el modelo (una vez por sesión)

# COMMAND ----------

X = pdf[FEATS].copy()
y = pdf["label"].astype(int)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("bool", "passthrough", bool_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])
model = Pipeline([("pre", pre),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
model.fit(X, y)
print("Modelo entrenado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fila base (medianas/modas) + tags candidatos

# COMMAND ----------

base = {}
for c in num_feats:
    base[c] = float(pdf[c].median())
for c in bool_feats:
    base[c] = 0
for c in cat_feats:
    base[c] = pdf[c].mode().iloc[0]

# Tags candidatos: mayor |coeficiente| si existe la tabla del notebook 02
try:
    coef = spark.table(COEF_TABLE).toPandas()
    top_tags = [f for f in coef.sort_values("abs_impact", ascending=False)["feature"] if f in bool_feats][:12]
except Exception:
    top_tags = bool_feats[:12]
print("Tags en el simulador:", top_tags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Widgets de entrada (aparecen arriba del notebook)

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("price", "19.99", "Precio (USD)")
if "dev_experience" in cat_feats:
    dbutils.widgets.dropdown("dev_experience", "Novato", ["Novato", "Establecido", "AAA"], "Experiencia estudio")
for t in top_tags:
    label = t.replace("tag_", "").replace("genre_", "").replace("cat_", "").replace("_", " ").title()
    dbutils.widgets.dropdown(t, "No", ["No", "Sí"], label[:30])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Predicción (cambiar widgets y re-ejecutar esta celda)

# COMMAND ----------

def predict(overrides):
    row = dict(base); row.update(overrides)
    Xrow = pd.DataFrame([row])[FEATS]
    return float(model.predict_proba(Xrow)[0, 1])

user = {"price": float(dbutils.widgets.get("price"))}
if "dev_experience" in cat_feats:
    user["dev_experience"] = dbutils.widgets.get("dev_experience")
for t in top_tags:
    user[t] = 1 if dbutils.widgets.get(t) == "Sí" else 0

prob = predict(user)
print("=" * 45)
print(f"  PROBABILIDAD DE ÉXITO COMERCIAL: {prob*100:.1f}%")
print(f"  (éxito = más de 20.000 dueños estimados)")
print("=" * 45)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Motor de recomendaciones (Doc. Técnico §5)
# MAGIC Prueba cambios uno a uno y reporta el delta de probabilidad.

# COMMAND ----------

recs = []
for delta in (-5, +5):
    new_price = max(0.0, user["price"] + delta)
    recs.append((f"Cambiar precio a ${new_price:.2f}", (predict({**user, "price": new_price}) - prob) * 100))
for t in top_tags:
    accion = "Añadir" if user[t] == 0 else "Quitar"
    nombre = t.replace("tag_", "").replace("genre_", "").replace("cat_", "").replace("_", " ").title()
    recs.append((f"{accion} '{nombre}'", (predict({**user, t: 1 - user[t]}) - prob) * 100))
if "dev_experience" in cat_feats and user.get("dev_experience") != "AAA":
    nivel = "Establecido" if user.get("dev_experience") == "Novato" else "AAA"
    recs.append((f"Si el estudio fuera '{nivel}'", (predict({**user, "dev_experience": nivel}) - prob) * 100))

recs.sort(key=lambda x: x[1], reverse=True)
print("RECOMENDACIONES (impacto en la probabilidad de éxito):\n")
for accion, d in recs:
    flag = "🟢" if d > 0.5 else ("🔴" if d < -0.5 else "⚪")
    print(f"  {flag} {accion:<40} {'+' if d>=0 else ''}{d:.1f} pts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Contexto: distribución de éxito en estudios similares

# COMMAND ----------

if "dev_experience" in cat_feats:
    sub = pdf[pdf["dev_experience"] == user.get("dev_experience", "Novato")]
    display(spark.createDataFrame(sub.groupby("label").size().reset_index(name="count")))
