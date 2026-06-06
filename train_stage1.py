"""
==============================================================================
  TRAIN STAGE 1 — Modelo de TRACCIÓN (embudo, etapa 1 de 2)
==============================================================================
  Pregunta: ¿el juego logrará tracción comercial (ventas estimables por SteamSpy)?
  Target binario `tiene_ventas = owners_lower_bound > 0` sobre TODOS los juegos
  (~33k), no solo los que ya venden. Reutiliza la ingeniería de features de
  build_dataset_v2.py (sin dev_success_prior, que es propio de la etapa 2).

  El simulador combina:  P(no-tracción) = 1 - P(vende)
                         P(Flop|Rent|Hit) = P(vende) · etapa2(Flop|Rent|Hit)

  Salidas:
    output/dataset_stage1.csv   master de todos los juegos + tiene_ventas
    models/stage1.joblib        Pipeline de clasificación binaria (LR calibrada)
    models/stage1_schema.json   feat_cols + defaults para el backend
    reports/stage1.json         métricas (AUC, F1, accuracy, matriz)
==============================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import joblib

import build_dataset_v2 as bd   # reutiliza los helpers de derivación

OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)
SEED = 42

CAT_FEATS = ["dev_experience", "controller_support", "pub_experience", "price_tier", "release_quarter"]
BOOL_PREFIXES = ("genre_", "cat_", "tag_", "platform_", "is_")
NUM_FEATS = ["price", "supported_languages", "total_achievements", "total_dlcs", "min_ram_gb",
             "short_desc_len", "dev_game_count", "pub_game_count", "num_genres", "num_tags", "num_categories"]


def build_all_games() -> pd.DataFrame:
    """Master de TODOS los juegos con las features pre-lanzamiento + tiene_ventas."""
    meta = bd.load_csv("games_metadata.csv").drop_duplicates("appid")
    tags = bd.load_csv("games_tags.csv").drop_duplicates("appid")
    try:
        text = bd.load_csv("games_text.csv").drop_duplicates("appid")
    except FileNotFoundError:
        text = None

    meta = bd.derive_experience(meta, "developer")
    meta = bd.derive_experience(meta, "publisher")
    meta = bd.derive_release_parts(meta)
    meta = bd.derive_price_tier(meta)
    if text is not None and "short_description" in text.columns:
        text["short_desc_len"] = text["short_description"].fillna("").astype(str).str.len()
        meta = meta.merge(text[["appid", "short_desc_len"]], on="appid", how="left")
        meta["short_desc_len"] = meta["short_desc_len"].fillna(0).astype(int)

    df = meta.merge(tags, on="appid", how="inner")
    genre = [c for c in df.columns if c.startswith("genre_")]
    tagc = [c for c in df.columns if c.startswith("tag_")]
    catc = [c for c in df.columns if c.startswith("cat_")]
    df[genre + tagc + catc] = df[genre + tagc + catc].fillna(0).astype(int)
    df["num_genres"] = df[genre].sum(axis=1)
    df["num_tags"] = df[tagc].sum(axis=1)
    df["num_categories"] = df[catc].sum(axis=1)
    ea = [c for c in ["genre_early_access", "tag_early_access"] if c in df.columns]
    df["is_early_access"] = (df[ea].sum(axis=1) > 0).astype(int) if ea else 0

    df["price"] = df["price"].clip(upper=200.0)
    df["tiene_ventas"] = (df["owners_lower_bound"] > 0).astype(int)

    bool_cols = [c for c in df.columns if c.startswith(BOOL_PREFIXES)]
    df[bool_cols] = df[bool_cols].fillna(0).astype(int)
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("")
    return df


def main():
    df = build_all_games()
    # FILTRO de metadata completa — clave para eliminar el sesgo de recolección.
    # Los juegos owners==0 venían del catálogo de Steam con campos sin poblar (sin fecha,
    # sin plataforma, sin descripción); el modelo aprendía "metadata faltante" en vez de
    # calidad. Al exigir metadata completa, la etapa de tracción mide señales reales.
    before = len(df)
    df = df[(df["release_quarter"] != "Desconocido") & (df["platform_windows"] == 1)
            & ((df["num_genres"] + df["num_tags"]) >= 1) & (df["short_desc_len"] > 0)].reset_index(drop=True)
    print(f"Filtro metadata completa: {before:,} -> {len(df):,} juegos (elimina el artefacto de recolección)")

    bool_feats = [c for c in df.columns if c.startswith(BOOL_PREFIXES)
                  and df[c].nunique() > 1 and df[c].value_counts(normalize=True).iloc[0] < 0.999]
    num_feats = [c for c in NUM_FEATS if c in df.columns]
    cat_feats = [c for c in CAT_FEATS if c in df.columns]
    feat_cols = num_feats + bool_feats + cat_feats

    bal = df["tiene_ventas"].value_counts(normalize=True).round(3).to_dict()
    print(f"Juegos: {len(df):,} | tiene_ventas: {bal} | features: {len(feat_cols)}")

    # Guardar master (para Databricks / reproducibilidad)
    keep = ["appid", "name", "owners_lower_bound", "tiene_ventas"] + feat_cols
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    df[keep].to_csv(OUTPUT_DIR / "dataset_stage1.csv", index=False, sep=";", encoding="utf-8")

    X = df[feat_cols]; y = df["tiene_ventas"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    def make_pre():
        return ColumnTransformer([
            ("num", StandardScaler(), num_feats),
            ("bool", "passthrough", bool_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ])

    pipe = Pipeline([("pre", make_pre()),
                     ("clf", LogisticRegression(max_iter=2000))])  # sin balanceo: probabilidades calibradas para el embudo
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = pipe.predict(X_te)
    metrics = {
        "target": "tiene_ventas (owners>0)",
        "n_total": int(len(df)),
        "balance": bal,
        "auc": round(float(roc_auc_score(y_te, proba)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "confusion": confusion_matrix(y_te, pred).tolist(),
    }
    print(f"Stage 1 (tracción): AUC={metrics['auc']}  F1={metrics['f1']}  acc={metrics['accuracy']}")

    # Reentrenar sobre el 100% y serializar
    full = Pipeline([("pre", make_pre()),
                     ("clf", LogisticRegression(max_iter=2000))])  # sin balanceo: probabilidades calibradas para el embudo
    full.fit(X, y)
    joblib.dump(full, MODELS_DIR / "stage1.joblib")

    schema = {
        "target": "tiene_ventas",
        "feat_cols": feat_cols,
        "numeric": {c: {"median": float(df[c].median())} for c in num_feats},
        "boolean": bool_feats,
        "categorical": {c: {"categories": sorted(df[c].astype(str).unique().tolist()),
                            "default": str(df[c].mode().iloc[0])} for c in cat_feats},
    }
    (MODELS_DIR / "stage1_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "stage1.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> models/stage1.joblib · models/stage1_schema.json · reports/stage1.json · output/dataset_stage1.csv")


if __name__ == "__main__":
    main()
