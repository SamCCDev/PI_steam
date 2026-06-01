# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Modelo Bayesiano Jerárquico (shrinkage por desarrollador)
# MAGIC
# MAGIC Fase de pulido (Documento Técnico §4.2.4). Los juegos de un mismo estudio NO son
# MAGIC independientes. Este notebook estima la probabilidad de éxito **por developer**
# MAGIC aplicando *shrinkage*: los estudios con pocos juegos (novatos) se encogen hacia
# MAGIC la media global del mercado, mitigando predicciones sobreoptimistas.
# MAGIC
# MAGIC **Implementación:** Empirical Bayes con prior conjugada Beta-Binomial (partial
# MAGIC pooling). Es la versión ligera y sin dependencias del modelo jerárquico; la
# MAGIC versión completa con MCMC (PyMC) queda como extensión (lenta en serverless).

# COMMAND ----------

CATALOG = "workspace"
SCHEMA  = "default"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.dataset_ml"          # tiene 'developer' y 'label'
PRIOR_TABLE  = f"{CATALOG}.{SCHEMA}.dev_success_prior"   # salida

# COMMAND ----------

import numpy as np, pandas as pd

pdf = spark.table(SOURCE_TABLE).toPandas()
df = pdf[(pdf["owners_lower_bound"] > 0)].copy() if "owners_lower_bound" in pdf.columns else pdf.copy()
df["developer"] = df["developer"].fillna("").replace("", "Desconocido")
# Outcome binario para el shrinkage: 'viable' = comercialmente exitoso (Rentable o Hit, label>=1)
df["viable"] = (df["label"] >= 1).astype(int)
print(f"Juegos: {len(df):,}  |  developers únicos: {df['developer'].nunique():,}")
print(f"Tasa global de viabilidad (Rentable+Hit): {df['viable'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Estimar la prior global (Beta) por método de momentos
# MAGIC Usamos los developers con ≥2 juegos para estimar la distribución de tasas de
# MAGIC éxito del mercado y derivar la fuerza de la prior (pseudo-conteos α₀, β₀).

# COMMAND ----------

grp = df.groupby("developer")["viable"].agg(n="count", k="sum")
multi = grp[grp["n"] >= 2].copy()
rates = (multi["k"] / multi["n"])

global_mean = df["viable"].mean()
m = float(rates.mean())
v = float(rates.var(ddof=1))
# Método de momentos Beta: kappa = m(1-m)/v - 1  (fuerza de la prior en pseudo-juegos)
kappa = max((m * (1 - m) / v) - 1, 1.0) if v > 0 else 50.0
alpha0 = m * kappa
beta0 = (1 - m) * kappa
print(f"Media global de éxito: {global_mean:.3f}")
print(f"Prior Beta -> α₀={alpha0:.2f}  β₀={beta0:.2f}  (fuerza κ={kappa:.1f} pseudo-juegos)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Posterior por developer (shrinkage)
# MAGIC `p_shrunk = (k + α₀) / (n + α₀ + β₀)`. Con `n` pequeño domina la prior (se encoge
# MAGIC a la media global); con `n` grande domina la evidencia propia del estudio.

# COMMAND ----------

dev = df.groupby("developer")["viable"].agg(n="count", k="sum").reset_index()
dev["p_raw"] = dev["k"] / dev["n"]
dev["p_shrunk"] = (dev["k"] + alpha0) / (dev["n"] + alpha0 + beta0)
dev["shrinkage"] = (dev["p_raw"] - dev["p_shrunk"]).abs().round(3)
dev = dev.sort_values("n", ascending=False)

# Ejemplos: estudios grandes (la evidencia manda) vs novatos (la prior manda)
print("Estudios establecidos (n alto -> poco shrinkage):")
print(dev.head(5)[["developer", "n", "p_raw", "p_shrunk"]].to_string(index=False))
print("\nEstudios novatos (n=1 -> fuerte shrinkage hacia la media global):")
print(dev[dev["n"] == 1].head(5)[["developer", "n", "p_raw", "p_shrunk"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Guardar la prior por developer (la usa el dashboard como ajuste)

# COMMAND ----------

(spark.createDataFrame(dev)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PRIOR_TABLE))
print(f"Prior por developer guardada en {PRIOR_TABLE}.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Lectura del resultado:** un estudio novato con 1 solo éxito tiene `p_raw=1.0`
# MAGIC (sobreoptimista), pero `p_shrunk` lo baja hacia la media del mercado — exactamente
# MAGIC el comportamiento que pide el Doc. Técnico §4.2.4 para no sobreestimar a estudios
# MAGIC sin historial. El dashboard puede combinar esta prior con la predicción del modelo.
