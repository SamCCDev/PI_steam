# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Entrenamiento de los modelos en Databricks
# MAGIC
# MAGIC Este notebook **no duplica** la lógica de entrenamiento: importa y ejecuta el
# MAGIC mismo `scripts/train_models.py` que se corre en local. Así el resultado es
# MAGIC idéntico y existe una única fuente de verdad.
# MAGIC
# MAGIC **Antes de correr:** sube `output/dataset_ml.csv` a una tabla de Unity Catalog
# MAGIC llamada `dataset_ml` (Catalog UI → Create Table → CSV, separador `;`). El script
# MAGIC detecta que está en Databricks (no hay CSV local pero sí una sesión Spark con la
# MAGIC tabla) y guarda las métricas y los coeficientes como tablas UC.
# MAGIC
# MAGIC **Por qué scikit-learn y no PySpark ML:** Free Edition es serverless (Spark
# MAGIC Connect) y bloquea la MLlib clásica; el dataset cabe en el driver y se entrena con
# MAGIC scikit-learn sobre `toPandas()`.
# MAGIC
# MAGIC **Artefactos:** los `.joblib` que sirve el backend se generan en **local**
# MAGIC (Free Edition no permite descargar archivos del serverless). Este notebook deja la
# MAGIC evidencia como tablas Unity Catalog: `model_metrics` y `model_lr_coefficients`.

# COMMAND ----------

# Añade scripts/ al path. Ajusta la ruta si el repo está en otra ubicación del workspace.
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "scripts")))

import train_models

# COMMAND ----------

# El script detecta Databricks por la sesión Spark activa y la tabla Unity Catalog.
# (Para apuntar a otro catálogo/esquema, edita UC_DATASET/UC_METRICS/UC_COEF en train_models.py.)
df, env = train_models.load_dataset()
print("Entorno detectado:", env)        # -> "databricks"
metrics = train_models.run(df, env)
metrics
