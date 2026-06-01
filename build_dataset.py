"""
==============================================================================
  BUILD DATASET — Consolida los CSV relacionales en un master ML-ready
==============================================================================
  Entrada (output/):  games_metadata.csv | games_tags.csv |
                       games_text.csv | games_timeseries.csv  (sep=';')
  Salida  (output/):  dataset_ml.csv            -> master 1 fila por juego
                      dataset_ml_dictionary.csv -> diccionario de columnas

  Decisiones (acordadas con el equipo):
    - Filtra owners_lower_bound > 0 (Doc. Técnico §4.3: descartar filas sin
      datos de ventas; en este dataset 0 = sin datos de SteamSpy).
    - Etiqueta: label = owners_lower_bound > 20.000  (Doc. Técnico §4.1).
    - Marca las columnas POST-lanzamiento (positive, negative, ccu, rating,
      agregados ts_*) como NO-features para el simulador PRE-lanzamiento, pero
      las conserva en el dataset para análisis.
    - Schema-driven: detecta tags/géneros por prefijo; no hay listas fijas.

  Reproducible: re-ejecutar tras actualizar output/*.csv regenera el master.
==============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")
SEP = ";"

SUCCESS_THRESHOLD = 20000
PRICE_CAP_USD = 200.0
DROP_ZERO_OWNERS = True

# Columnas POST-lanzamiento / identificadores: NO usar como features pre-lanzamiento.
ID_COLS = ["appid", "name", "developer", "publisher", "release_date"]
TARGET_COLS = ["owners_lower_bound", "ccu"]
# Outcome conocido solo TRAS el lanzamiento -> fuga de datos para un simulador pre-lanzamiento.
# metacritic_score = nota de crítica especializada, publicada en/después del lanzamiento.
POSTLAUNCH_COLS = ["positive", "negative", "rating_porcentaje", "metacritic_score"]
BOOL_PREFIXES = ("genre_", "cat_", "tag_", "platform_", "is_")


def load_csv(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta steam_etl.py primero.")
    return pd.read_csv(path, sep=SEP)


def derive_dev_experience(meta: pd.DataFrame) -> pd.DataFrame:
    """Experiencia del estudio contando appids previos (Doc. Técnico §6.4)."""
    counts = meta.groupby("developer")["appid"].transform("count")
    meta = meta.copy()
    meta["dev_game_count"] = counts.fillna(1).astype(int)
    meta["dev_experience"] = np.select(
        [meta["dev_game_count"] >= 10, meta["dev_game_count"] >= 3],
        ["AAA", "Establecido"],
        default="Novato",
    )
    return meta


def aggregate_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Colapsa el historial mensual a un vector por juego (señales POST-lanzamiento)."""
    if ts.empty:
        return pd.DataFrame(columns=["appid"])
    g = ts.groupby("appid")
    agg = pd.DataFrame({
        "ts_total_reviews":   g["review_count"].sum(),
        "ts_positive":        g["positive_count"].sum(),
        "ts_negative":        g["negative_count"].sum(),
        "ts_steam_purchase":  g["steam_purchase_count"].sum(),
        "ts_early_access":    g["early_access_count"].sum(),
        "ts_months_active":   g["review_count"].apply(lambda s: int((s > 0).sum())),
        "ts_avg_playtime_hrs": g["avg_playtime_at_review_hrs"].mean().round(2),
    }).reset_index()
    total = agg["ts_positive"] + agg["ts_negative"]
    agg["ts_positive_ratio"] = np.where(total > 0, (agg["ts_positive"] / total).round(3), 0.0)
    return agg


def build():
    print("[1/6] Cargando CSV relacionales...")
    meta = load_csv("games_metadata.csv")
    tags = load_csv("games_tags.csv")
    try:
        text = load_csv("games_text.csv")
    except FileNotFoundError:
        text = None
    try:
        ts = load_csv("games_timeseries.csv")
    except FileNotFoundError:
        ts = pd.DataFrame(columns=["appid"])

    # Dedup defensivo por appid
    meta = meta.drop_duplicates(subset=["appid"], keep="first")
    tags = tags.drop_duplicates(subset=["appid"], keep="first")
    print(f"    metadata={len(meta):,}  tags={len(tags):,}  timeseries={len(ts):,} filas")

    print("[2/6] Derivando dev_experience y agregando series de tiempo...")
    meta = derive_dev_experience(meta)
    ts_agg = aggregate_timeseries(ts)

    # Feature pre-lanzamiento barata: longitud del pitch de tienda
    if text is not None and "short_description" in text.columns:
        text = text.drop_duplicates(subset=["appid"], keep="first")
        text["short_desc_len"] = text["short_description"].fillna("").astype(str).str.len()
        meta = meta.merge(text[["appid", "short_desc_len"]], on="appid", how="left")
        meta["short_desc_len"] = meta["short_desc_len"].fillna(0).astype(int)

    print("[3/6] Uniendo en master por appid...")
    df = meta.merge(tags, on="appid", how="inner")
    df = df.merge(ts_agg, on="appid", how="left")
    # Juegos sin filas de timeseries -> agregados en 0
    ts_cols = [c for c in df.columns if c.startswith("ts_")]
    df[ts_cols] = df[ts_cols].fillna(0)

    print("[4/6] Filtrando, etiquetando y limpiando...")
    if DROP_ZERO_OWNERS:
        before = len(df)
        df = df[df["owners_lower_bound"] > 0].copy()
        print(f"    owners>0: {before:,} -> {len(df):,} filas (descartadas {before-len(df):,} sin datos de ventas)")

    df["price"] = df["price"].clip(upper=PRICE_CAP_USD)
    df["label"] = (df["owners_lower_bound"] > SUCCESS_THRESHOLD).astype(int)

    # Tipos estrictos: booleanas/tags -> int 0/1; numéricas -> float; resto se infiere
    bool_cols = [c for c in df.columns if c.startswith(BOOL_PREFIXES)]
    df[bool_cols] = df[bool_cols].fillna(0).astype(int)

    # Imputar numéricas restantes por mediana
    num_candidates = (df.select_dtypes(include=[np.number]).columns.tolist())
    for c in num_candidates:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    # Strings/identificadores nulos (developer, publisher, release_date) -> "" para
    # dejar el master sin nulos. No afecta al modelado (no son features).
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("")

    print("[5/6] Clasificando columnas y construyendo diccionario...")
    def classify(col):
        if col in ID_COLS:                        return "id"
        if col == "label":                        return "target_label"
        if col in TARGET_COLS:                    return "target_raw"
        if col in POSTLAUNCH_COLS:                return "outcome_postlaunch"
        if col.startswith("ts_"):                 return "timeseries_postlaunch"
        if col.startswith(BOOL_PREFIXES):         return "feature_tag"
        if col in ("dev_experience", "controller_support"): return "feature_categorical"
        if col in ("dev_game_count",):            return "feature_derived"
        return "feature_numeric"

    near_const = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        top_ratio = df[c].value_counts(normalize=True, dropna=False).iloc[0]
        if df[c].nunique(dropna=False) < 2 or top_ratio >= 0.999:
            near_const.append(c)

    dictionary = pd.DataFrame({
        "column": df.columns,
        "role": [classify(c) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=False)) for c in df.columns],
        "pct_nonzero": [round(float((df[c] != 0).mean()) * 100, 1)
                        if pd.api.types.is_numeric_dtype(df[c]) else None
                        for c in df.columns],
        "near_constant": [c in near_const for c in df.columns],
        "use_as_feature": [classify(c).startswith("feature_") and c not in near_const
                           for c in df.columns],
    })

    print("[6/6] Guardando...")
    master_path = OUTPUT_DIR / "dataset_ml.csv"
    dict_path = OUTPUT_DIR / "dataset_ml_dictionary.csv"
    df.to_csv(master_path, index=False, encoding="utf-8", sep=SEP)
    dictionary.to_csv(dict_path, index=False, encoding="utf-8", sep=SEP)

    # ── Reporte de calidad ────────────────────────────────────────────────
    feats = dictionary[dictionary["use_as_feature"]]["column"].tolist()
    balance = df["label"].value_counts(normalize=True).round(3).to_dict()
    print("\n" + "=" * 60)
    print("  DATASET ML CONSTRUIDO")
    print("=" * 60)
    print(f"  Filas (juegos):           {len(df):,}")
    print(f"  Columnas totales:         {len(df.columns)}")
    print(f"  Features usables:         {len(feats)}  "
          f"({sum(c.startswith('tag_') or c.startswith('genre_') or c.startswith('cat_') for c in feats)} tags/géneros/cats)")
    print(f"  Balance label (éxito>20k):{balance}")
    print(f"  Columnas casi-constantes: {len(near_const)} -> {near_const}")
    print(f"  Nulos en el master:       {int(df.isnull().sum().sum())}")
    print(f"\n  -> {master_path}")
    print(f"  -> {dict_path}")
    print("=" * 60)
    print("  Roles: feature_* = entradas del modelo PRE-lanzamiento.")
    print("         outcome_postlaunch / timeseries_postlaunch = NO usar como")
    print("         features (fuga de datos); útiles solo para análisis.")
    print("=" * 60)


if __name__ == "__main__":
    build()
