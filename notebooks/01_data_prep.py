# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Preparación de Datos (Unity Catalog)
# MAGIC
# MAGIC Lee el master `dataset_ml` (generado por `build_dataset.py` y subido como tabla),
# MAGIC selecciona las features PRE-lanzamiento de forma schema-driven, descarta columnas
# MAGIC casi-constantes y guarda las tablas de trabajo en Unity Catalog.
# MAGIC
# MAGIC **Free Edition:** el DBFS público (`/FileStore`) está deshabilitado. Por eso NO se
# MAGIC usan rutas de archivo: la entrada y las salidas son **tablas gestionadas de UC**.
# MAGIC
# MAGIC ### Cómo subir el dataset (una vez)
# MAGIC 1. `build_dataset.py` genera `output/dataset_ml.csv`.
# MAGIC 2. En Databricks: **Catalog → (tu schema) → Create → Table → Upload file**.
# MAGIC 3. Sube `dataset_ml.csv` y nómbrala `dataset_ml` (separador `;`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración (ajustar si tu catálogo/schema difieren)

# COMMAND ----------

CATALOG = "workspace"      # En Free Edition suele ser 'workspace'
SCHEMA  = "default"        # Schema donde subiste la tabla

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.dataset_ml"            # tabla creada al subir el CSV
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.features_silver"       # salida: dataset limpio
CATALOG_TABLE = f"{CATALOG}.{SCHEMA}.feature_columns"      # salida: catálogo de features

# Etiqueta multiclase (Doc. Técnico §4.1, Softmax): 0=Flop, 1=Rentable, 2=Hit.
# Cortes en owners_lower_bound (solo se usan si la tabla no trajera 'label' ya calculada).
FLOP_MAX_OWNERS = 200000
HIT_MIN_OWNERS  = 1000000

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

df = spark.table(SOURCE_TABLE)
print(f"{SOURCE_TABLE}: {df.count():,} filas, {len(df.columns)} columnas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Garantizar la etiqueta multiclase `label` (0=Flop, 1=Rentable, 2=Hit)
# MAGIC `build_dataset.py` ya la crea; este bloque la reconstruye solo si falta.

# COMMAND ----------

if "label" not in df.columns:
    df = df.withColumn(
        "label",
        F.when(F.col("owners_lower_bound") >= HIT_MIN_OWNERS, F.lit(2))
         .when(F.col("owners_lower_bound") >= FLOP_MAX_OWNERS, F.lit(1))
         .otherwise(F.lit(0)).cast(IntegerType()))

# Distribución de clases (0=Flop, 1=Rentable, 2=Hit)
display(df.groupBy("label").count().orderBy("label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Selección schema-driven de features (mismos roles que `build_dataset.py`)
# MAGIC Excluye identificadores, objetivo y todo lo POST-lanzamiento (fuga de datos).

# COMMAND ----------

ID_COLS = {"appid", "name", "developer", "publisher", "release_date"}
TARGET_COLS = {"owners_lower_bound", "ccu", "label"}
# Outcome conocido solo TRAS el lanzamiento (fuga de datos para el simulador pre-lanzamiento)
POSTLAUNCH_COLS = {"positive", "negative", "rating_porcentaje", "metacritic_score"}
BOOL_PREFIXES = ("genre_", "cat_", "tag_", "platform_", "is_")
CATEGORICAL_COLS = [c for c in ("controller_support", "dev_experience") if c in df.columns]

EXCLUDE = ID_COLS | TARGET_COLS | POSTLAUNCH_COLS
numeric_types = ("int", "bigint", "double", "float", "decimal")

bool_feats, num_feats = [], []
for name, dtype in df.dtypes:
    if name in EXCLUDE or name in CATEGORICAL_COLS or name.startswith("ts_"):
        continue
    if name.startswith(BOOL_PREFIXES):
        bool_feats.append(name)
    elif dtype in numeric_types:
        num_feats.append(name)

print(f"Candidatas -> numéricas={len(num_feats)} booleanas={len(bool_feats)} categóricas={len(CATEGORICAL_COLS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Descartar columnas casi-constantes (baja varianza; cubre `hub_followers`)

# COMMAND ----------

# Nota: en serverless NO se permite .cache()/persist. Con ~3k filas no hace falta.
n = df.count()
dropped = []
MAX_DOMINANT = 0.999

# Booleanas (0/1): la media basta -> ratio dominante = max(p, 1-p). UNA sola pasada.
# Alias por índice (mi/di) para no romper con nombres raros como 'tag_point_&_click'.
if bool_feats:
    exprs = []
    for i, c in enumerate(bool_feats):
        exprs.append(F.avg(F.col(c).cast("double")).alias(f"m{i}"))
        exprs.append(F.countDistinct(F.col(c)).alias(f"d{i}"))
    row = df.agg(*exprs).first()
    for i, c in enumerate(bool_feats):
        p = row[f"m{i}"] or 0.0
        if row[f"d{i}"] < 2 or max(p, 1 - p) >= MAX_DOMINANT:
            dropped.append(c)

# Numéricas: son pocas (~7), el loop es barato.
for c in num_feats:
    if df.select(c).distinct().count() < 2:
        dropped.append(c); continue
    top = df.groupBy(c).count().orderBy(F.desc("count")).first()["count"]
    if top / n >= MAX_DOMINANT:
        dropped.append(c)

bool_feats = [c for c in bool_feats if c not in dropped]
num_feats = [c for c in num_feats if c not in dropped]

# Booleanas: asegurar int 0/1 sin nulos
for c in bool_feats:
    df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0)).cast(IntegerType()))

print(f"Numéricas ({len(num_feats)}): {num_feats}")
print(f"Booleanas/tags: {len(bool_feats)} columnas")
print(f"Categóricas: {CATEGORICAL_COLS}")
print(f"Descartadas por baja varianza ({len(dropped)}): {dropped}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guardar tablas de trabajo en Unity Catalog

# COMMAND ----------

keep = (["appid", "name", "dev_experience", "controller_support", "owners_lower_bound", "label"]
        + num_feats + bool_feats)
keep = [c for c in dict.fromkeys(keep) if c in df.columns]

silver = df.select(*keep)
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SILVER_TABLE)
print(f"{SILVER_TABLE}: {silver.count():,} filas, {len(silver.columns)} columnas")

# COMMAND ----------

rows = ([(c, "numeric") for c in num_feats]
        + [(c, "boolean") for c in bool_feats]
        + [(c, "categorical") for c in CATEGORICAL_COLS])
feat_catalog = spark.createDataFrame(rows, ["feature", "kind"])
feat_catalog.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CATALOG_TABLE)
display(feat_catalog)

# COMMAND ----------

# MAGIC %md
# MAGIC **Listo.** Continuar en `02_logistic_regression`. (Las tablas viven en Unity Catalog;
# MAGIC no se usa ninguna ruta DBFS.)
