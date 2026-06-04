"""
==============================================================================
  BUILD DATASET v2 — Master ML-ready con features ampliadas (Dashboard v2)
==============================================================================
  Superset de build_dataset.py. Mantiene las mismas entradas/salidas y la
  etiqueta multiclase, y AÑADE las features nuevas acordadas para la v2:

    - num_genres / num_tags / num_categories  (riqueza de catálogo del juego)
    - release_quarter (categórica, estacionalidad) + release_year/month (análisis)
    - pub_game_count + pub_experience          (tamaño/experiencia del publisher)
    - price_tier                               (F2P / budget / mid / premium)
    - is_early_access                          (bool)
    - dev_success_prior                        (Empirical Bayes LOO por developer,
                                                sin fuga: excluye el propio juego)

  Salida (output/):
    dataset_ml.csv             -> master 1 fila por juego (sobrescribe)
    dataset_ml_dictionary.csv  -> diccionario con columna nueva `etapa` (pre/post/meta)

  Anti-fuga: el predictor PRE-lanzamiento excluye positive/negative/ccu/rating/
  metacritic y agregados ts_*. Esas viven en el dataset solo para el track
  POST-lanzamiento y el análisis.

  Reproducible: re-ejecutar tras actualizar output/*.csv regenera el master.
==============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")
SEP = ";"

# Clasificación multiclase del éxito comercial (Softmax). Cortes en buckets reales de SteamSpy:
#   Flop (0) < 200k  |  Rentable (1) 200k–1M  |  Hit (2) >= 1M
FLOP_MAX_OWNERS = 200000
HIT_MIN_OWNERS  = 1000000
LABEL_NAMES = {0: "Flop", 1: "Rentable", 2: "Hit"}

PRICE_CAP_USD = 200.0
DROP_ZERO_OWNERS = True          # núcleo con ventas estimadas por SteamSpy
PRIOR_STRENGTH_K = 12.0          # pseudo-conteo para el shrinkage del prior por developer

ID_COLS = ["appid", "name", "developer", "publisher", "release_date"]
TARGET_COLS = ["owners_lower_bound", "ccu"]
POSTLAUNCH_COLS = ["positive", "negative", "rating_porcentaje", "metacritic_score"]
BOOL_PREFIXES = ("genre_", "cat_", "tag_", "platform_", "is_")

# Features categóricas (OHE en el modelo). Se leen por nombre, no por prefijo.
CATEGORICAL_FEATS = ["dev_experience", "controller_support",
                     "pub_experience", "price_tier", "release_quarter"]
# Columnas temporales que NO son features (un juego futuro no tiene año/mes conocidos
# de forma comparable); se conservan para análisis del panel.
TEMPORAL_ANALYSIS = ["release_year", "release_month"]


def load_csv(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta steam_etl.py primero.")
    return pd.read_csv(path, sep=SEP)


def derive_experience(df: pd.DataFrame, who: str) -> pd.DataFrame:
    """Tier de experiencia (Novato/Establecido/AAA) contando juegos previos del
    developer o publisher. who in {'developer','publisher'}."""
    prefix = "dev" if who == "developer" else "pub"
    counts = df.groupby(who)["appid"].transform("count")
    df[f"{prefix}_game_count"] = counts.fillna(1).astype(int)
    df[f"{prefix}_experience"] = np.select(
        [df[f"{prefix}_game_count"] >= 10, df[f"{prefix}_game_count"] >= 3],
        ["AAA", "Establecido"], default="Novato")
    return df


def derive_release_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae año, mes y trimestre de release_date (tolerante a formatos sucios)."""
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = dt.dt.year
    df["release_month"] = dt.dt.month
    q = dt.dt.quarter
    df["release_quarter"] = q.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    # Faltantes -> marcador propio para no inventar estacionalidad
    df["release_year"] = df["release_year"].fillna(0).astype(int)
    df["release_month"] = df["release_month"].fillna(0).astype(int)
    df["release_quarter"] = df["release_quarter"].fillna("Desconocido")
    return df


def derive_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Tramo de precio comercial."""
    is_free = df["is_free"].astype(str).str.lower().isin(["true", "1", "1.0"]) \
        if "is_free" in df.columns else (df["price"] <= 0)
    price = df["price"].fillna(0)
    df["price_tier"] = np.select(
        [is_free | (price <= 0), price < 10, price < 20],
        ["F2P", "Budget", "Mid"], default="Premium")
    return df


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


def derive_dev_success_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical Bayes (Beta-Binomial) leave-one-out: probabilidad de que un juego
    del MISMO developer sea 'viable' (Rentable o Hit), excluyendo el juego actual
    para no filtrar su propia etiqueta. Devs sin otros juegos -> prior global.
    """
    viable = (df["label"] >= 1).astype(int)
    p0 = float(viable.mean())
    a0, b0 = p0 * PRIOR_STRENGTH_K, (1 - p0) * PRIOR_STRENGTH_K
    dev_n = df.groupby("developer")["label"].transform("count")
    dev_s = df.assign(_v=viable).groupby("developer")["_v"].transform("sum")
    loo_s = dev_s - viable          # éxitos del developer sin contar este juego
    loo_n = dev_n - 1               # juegos del developer sin contar este
    df["dev_success_prior"] = ((loo_s + a0) / (loo_n + a0 + b0)).round(4)
    df.attrs["dev_prior_global"] = round(p0, 4)
    return df


def build():
    print("[1/7] Cargando CSV relacionales...")
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

    meta = meta.drop_duplicates(subset=["appid"], keep="first")
    tags = tags.drop_duplicates(subset=["appid"], keep="first")
    print(f"    metadata={len(meta):,}  tags={len(tags):,}  timeseries={len(ts):,} filas")

    print("[2/7] Derivando experiencia (dev+pub), fechas y tramo de precio...")
    meta = derive_experience(meta, "developer")
    meta = derive_experience(meta, "publisher")
    meta = derive_release_parts(meta)
    meta = derive_price_tier(meta)
    ts_agg = aggregate_timeseries(ts)

    if text is not None and "short_description" in text.columns:
        text = text.drop_duplicates(subset=["appid"], keep="first")
        text["short_desc_len"] = text["short_description"].fillna("").astype(str).str.len()
        meta = meta.merge(text[["appid", "short_desc_len"]], on="appid", how="left")
        meta["short_desc_len"] = meta["short_desc_len"].fillna(0).astype(int)

    print("[3/7] Uniendo en master por appid...")
    df = meta.merge(tags, on="appid", how="inner")
    df = df.merge(ts_agg, on="appid", how="left")
    ts_cols = [c for c in df.columns if c.startswith("ts_")]
    df[ts_cols] = df[ts_cols].fillna(0)

    print("[4/7] Derivando riqueza de catálogo y early access...")
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    df[genre_cols + tag_cols + cat_cols] = df[genre_cols + tag_cols + cat_cols].fillna(0).astype(int)
    df["num_genres"] = df[genre_cols].sum(axis=1)
    df["num_tags"] = df[tag_cols].sum(axis=1)
    df["num_categories"] = df[cat_cols].sum(axis=1)
    ea_src = [c for c in ["genre_early_access", "tag_early_access"] if c in df.columns]
    df["is_early_access"] = (df[ea_src].sum(axis=1) > 0).astype(int) if ea_src else 0

    print("[5/7] Filtrando, etiquetando y prior por developer...")
    if DROP_ZERO_OWNERS:
        before = len(df)
        df = df[df["owners_lower_bound"] > 0].copy()
        print(f"    owners>0: {before:,} -> {len(df):,} filas (descartadas {before-len(df):,} sin datos de ventas)")

    df["price"] = df["price"].clip(upper=PRICE_CAP_USD)
    df["label"] = np.select(
        [df["owners_lower_bound"] >= HIT_MIN_OWNERS,
         df["owners_lower_bound"] >= FLOP_MAX_OWNERS],
        [2, 1], default=0).astype(int)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df = derive_dev_success_prior(df)

    print("[6/7] Tipos, imputación y diccionario...")
    bool_cols = [c for c in df.columns if c.startswith(BOOL_PREFIXES)]
    df[bool_cols] = df[bool_cols].fillna(0).astype(int)
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("")

    def classify(col):
        if col in ID_COLS:                          return "id"
        if col in ("label", "label_name"):          return "target_label"
        if col in TARGET_COLS:                      return "target_raw"
        if col in POSTLAUNCH_COLS:                  return "outcome_postlaunch"
        if col.startswith("ts_"):                   return "timeseries_postlaunch"
        if col in TEMPORAL_ANALYSIS:                return "analysis_temporal"
        if col.startswith(BOOL_PREFIXES):           return "feature_tag"
        if col in CATEGORICAL_FEATS:                return "feature_categorical"
        if col in ("dev_game_count", "pub_game_count", "dev_success_prior",
                   "num_genres", "num_tags", "num_categories"):
            return "feature_derived"
        return "feature_numeric"

    def stage(role):
        if role.startswith("feature_"):                       return "pre"
        if role in ("outcome_postlaunch", "timeseries_postlaunch"): return "post"
        return "meta"

    near_const = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        top_ratio = df[c].value_counts(normalize=True, dropna=False).iloc[0]
        if df[c].nunique(dropna=False) < 2 or top_ratio >= 0.999:
            near_const.append(c)

    roles = [classify(c) for c in df.columns]
    dictionary = pd.DataFrame({
        "column": df.columns,
        "role": roles,
        "etapa": [stage(r) for r in roles],
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=False)) for c in df.columns],
        "pct_nonzero": [round(float((df[c] != 0).mean()) * 100, 1)
                        if pd.api.types.is_numeric_dtype(df[c]) else None
                        for c in df.columns],
        "near_constant": [c in near_const for c in df.columns],
        "use_as_feature": [classify(c).startswith("feature_") and c not in near_const
                           for c in df.columns],
    })

    print("[7/7] Guardando...")
    master_path = OUTPUT_DIR / "dataset_ml.csv"
    dict_path = OUTPUT_DIR / "dataset_ml_dictionary.csv"
    df.to_csv(master_path, index=False, encoding="utf-8", sep=SEP)
    dictionary.to_csv(dict_path, index=False, encoding="utf-8", sep=SEP)

    feats = dictionary[dictionary["use_as_feature"]]["column"].tolist()
    new_feats = [c for c in ["num_genres", "num_tags", "num_categories", "pub_game_count",
                             "pub_experience", "price_tier", "release_quarter",
                             "is_early_access", "dev_success_prior"] if c in feats]
    balance = df["label_name"].value_counts(normalize=True).round(3).to_dict()
    print("\n" + "=" * 64)
    print("  DATASET ML v2 CONSTRUIDO")
    print("=" * 64)
    print(f"  Filas (juegos):           {len(df):,}")
    print(f"  Columnas totales:         {len(df.columns)}")
    print(f"  Features usables:         {len(feats)}")
    print(f"  Features NUEVAS activas:  {new_feats}")
    print(f"  Clases (Flop/Rentable/Hit): {balance}")
    print(f"  Prior global viable:      {df.attrs.get('dev_prior_global')}")
    print(f"  Casi-constantes:          {len(near_const)}")
    print(f"  Nulos en el master:       {int(df.isnull().sum().sum())}")
    print(f"\n  -> {master_path}")
    print(f"  -> {dict_path}")
    print("=" * 64)


if __name__ == "__main__":
    build()
