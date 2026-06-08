# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Simulador Comercial (multiclase Flop/Rentable/Hit)
# MAGIC
# MAGIC Simulador pre-lanzamiento (Doc. Técnico §5). El usuario fija precio, experiencia
# MAGIC del estudio y mecánicas (tags); el sistema devuelve la **probabilidad de cada
# MAGIC clase comercial** (Flop / Rentable / Hit) y un motor de recomendaciones.
# MAGIC
# MAGIC Modela con scikit-learn en el driver (Free Edition serverless bloquea la MLlib
# MAGIC clásica). Entrena en segundos desde `features_silver`; sin artefactos de archivo.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "default"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.features_silver"
CATALOG_TABLE = f"{CATALOG}.{SCHEMA}.feature_columns"

CLASS_NAMES = {0: "Flop", 1: "Rentable", 2: "Hit"}

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
# MAGIC ## 1. Entrenar el modelo multinomial (una vez por sesión)

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
classes = list(model.named_steps["clf"].classes_)
print(f"Modelo entrenado. Clases: {[CLASS_NAMES[c] for c in classes]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fila base (medianas/modas) + tags candidatos
# MAGIC Solo se ofrecen como palanca los tags con **prevalencia 5–95%** (evita
# MAGIC recomendaciones degeneradas sobre columnas casi-constantes, p.ej. platform_windows).

# COMMAND ----------

base = {c: float(pdf[c].median()) for c in num_feats}
for c in bool_feats:
    base[c] = 0
for c in cat_feats:
    base[c] = pdf[c].mode().iloc[0]

prevalence = {c: pdf[c].mean() for c in bool_feats}
variable_tags = [c for c in bool_feats if 0.05 <= prevalence[c] <= 0.95]

# Priorizar por impacto hacia "Hit" si existe la tabla de coeficientes del notebook 02
try:
    coef = spark.table(f"{CATALOG}.{SCHEMA}.model_lr_coefficients").toPandas()
    hit = coef[coef["clase"] == "Hit"].copy()
    hit["abs_impact"] = hit["coef_logodds"].abs()
    ranked = [f for f in hit.sort_values("abs_impact", ascending=False)["feature"] if f in variable_tags]
    top_tags = ranked[:12]
except Exception:
    top_tags = variable_tags[:12]
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
# MAGIC Devuelve P(Flop), P(Rentable), P(Hit) y un **score de viabilidad** = 1 − P(Flop).

# COMMAND ----------

def predict_probs(overrides):
    """Devuelve dict {clase: prob} para la configuración dada."""
    row = dict(base); row.update(overrides)
    Xrow = pd.DataFrame([row])[FEATS]
    p = model.predict_proba(Xrow)[0]
    return {CLASS_NAMES[c]: float(p[i]) for i, c in enumerate(classes)}

def viability(overrides):
    """Probabilidad de NO ser un flop comercial = P(Rentable) + P(Hit)."""
    p = predict_probs(overrides)
    return p.get("Rentable", 0.0) + p.get("Hit", 0.0)

user = {"price": float(dbutils.widgets.get("price"))}
if "dev_experience" in cat_feats:
    user["dev_experience"] = dbutils.widgets.get("dev_experience")
for t in top_tags:
    user[t] = 1 if dbutils.widgets.get(t) == "Sí" else 0

probs = predict_probs(user)
viab = viability(user)
pronostico = max(probs, key=probs.get)
print("=" * 50)
print(f"  PRONÓSTICO COMERCIAL: {pronostico}")
print(f"  Viabilidad (no-flop): {viab*100:.1f}%")
print("-" * 50)
for cls in ["Flop", "Rentable", "Hit"]:
    barra = "█" * int(probs.get(cls, 0) * 30)
    print(f"  {cls:<9} {probs.get(cls,0)*100:5.1f}%  {barra}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Motor de recomendaciones (Doc. Técnico §5)
# MAGIC Prueba cambios uno a uno y reporta el delta en **P(Hit)** — el salto al tier
# MAGIC alto es donde hay margen de mejora (la viabilidad ya suele estar saturada).

# COMMAND ----------

def hit_prob(overrides):
    return predict_probs(overrides).get("Hit", 0.0)

p_hit = hit_prob(user)
recs = []
for delta in (-5, +5):
    new_price = max(0.0, user["price"] + delta)
    recs.append((f"Cambiar precio a ${new_price:.2f}", (hit_prob({**user, "price": new_price}) - p_hit) * 100))
for t in top_tags:
    accion = "Añadir" if user[t] == 0 else "Quitar"
    nombre = t.replace("tag_", "").replace("genre_", "").replace("cat_", "").replace("_", " ").title()
    recs.append((f"{accion} '{nombre}'", (hit_prob({**user, t: 1 - user[t]}) - p_hit) * 100))
if "dev_experience" in cat_feats and user.get("dev_experience") != "AAA":
    nivel = "Establecido" if user.get("dev_experience") == "Novato" else "AAA"
    recs.append((f"Si el estudio fuera '{nivel}'", (hit_prob({**user, "dev_experience": nivel}) - p_hit) * 100))

recs.sort(key=lambda x: x[1], reverse=True)
print(f"P(Hit) actual: {p_hit*100:.1f}%")
print("RECOMENDACIONES (impacto en la probabilidad de ser un 'Hit'):\n")
for accion, d in recs:
    flag = "🟢" if d > 0.5 else ("🔴" if d < -0.5 else "⚪")
    print(f"  {flag} {accion:<40} {'+' if d>=0 else ''}{d:.1f} pts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Contexto: distribución de clases en estudios similares

# COMMAND ----------

if "dev_experience" in cat_feats:
    sub = pdf[pdf["dev_experience"] == user.get("dev_experience", "Novato")].copy()
    sub["clase"] = sub["label"].map(CLASS_NAMES)
    display(spark.createDataFrame(sub.groupby("clase").size().reset_index(name="count")))
