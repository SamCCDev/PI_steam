"""
==============================================================================
  EXPORT MODEL WEB — Exporta la Regresión Logística multinomial a JSON
==============================================================================
  Entrena el modelo sobre output/dataset_ml.csv (mismo pipeline que el notebook
  02/04) y exporta sus parámetros a output/model_web.json para que el dashboard
  estático `steampredict_dashboard_comercial.html` calcule la predicción en el
  navegador (softmax de 3 clases: Flop / Rentable / Hit), sin backend.

  Reproducible: re-ejecutar tras regenerar el dataset actualiza el JSON.
  Tras correrlo, `embed_model_in_html.py` inyecta el JSON dentro del HTML.
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

OUTPUT_DIR = Path("output")
CLASS_NAMES = {0: "Flop", 1: "Rentable", 2: "Hit"}


def main():
    df = pd.read_csv(OUTPUT_DIR / "dataset_ml.csv", sep=";")
    dic = pd.read_csv(OUTPUT_DIR / "dataset_ml_dictionary.csv", sep=";")
    use = dic.set_index("column")["use_as_feature"]

    num_feats = [c for c in dic[dic.role.isin(["feature_numeric", "feature_derived"])].column
                 if c in df.columns and use[c]]
    bool_feats = [c for c in dic[dic.role == "feature_tag"].column if use[c]]
    cat_feats = [c for c in ["dev_experience", "controller_support"] if c in df.columns]

    X = df[num_feats + bool_feats + cat_feats].copy()
    y = df["label"].astype(int)

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("bool", "passthrough", bool_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
    model.fit(X, y)

    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    classes = [CLASS_NAMES[c] for c in clf.classes_]

    # Coeficientes por clase: {clase: {feature_output_name: coef}}
    coef = {}
    intercept = {}
    for ci, cls in enumerate(classes):
        coef[cls] = {names[fi]: float(clf.coef_[ci, fi]) for fi in range(len(names))}
        intercept[cls] = float(clf.intercept_[ci])

    # Parámetros del StandardScaler + mediana (valor base para features sin input en la UI)
    scaler = model.named_steps["pre"].named_transformers_["num"]
    numeric = {}
    for i, c in enumerate(num_feats):
        numeric[c] = {"mean": float(scaler.mean_[i]),
                      "scale": float(scaler.scale_[i]),
                      "median": float(df[c].median())}

    # Categorías OHE + categoría base (moda)
    categorical = {}
    for c in cat_feats:
        categorical[c] = {"categories": [str(v) for v in sorted(df[c].dropna().unique())],
                          "default": str(df[c].mode().iloc[0])}

    payload = {
        "classes": classes,
        "intercept": intercept,
        "coef": coef,
        "numeric": numeric,
        "boolean": bool_feats,
        "categorical": categorical,
        "n_train": int(len(df)),
    }

    out = OUTPUT_DIR / "model_web.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ── Sanidad: replicar predict_proba manualmente y comparar con sklearn ──
    row = X.iloc[[0]]
    sk = model.predict_proba(row)[0]
    vals = {}
    for i, c in enumerate(num_feats):
        vals[c] = (float(row[c].iloc[0]) - numeric[c]["mean"]) / numeric[c]["scale"]
    for c in bool_feats:
        vals[c] = float(row[c].iloc[0])
    for c in cat_feats:
        for cat in categorical[c]["categories"]:
            vals[f"{c}_{cat}"] = 1.0 if str(row[c].iloc[0]) == cat else 0.0
    logits = []
    for cls in classes:
        s = intercept[cls] + sum(coef[cls].get(k, 0.0) * v for k, v in vals.items())
        logits.append(s)
    e = np.exp(np.array(logits) - max(logits))
    manual = e / e.sum()

    print(f"Modelo exportado a {out}")
    print(f"Clases: {classes}  |  features: {len(names)}  |  train: {len(df):,}")
    print(f"Sanidad (fila 0)  sklearn={np.round(sk,4)}  manual={np.round(manual,4)}")
    print(f"Max diff: {np.abs(sk - manual).max():.2e}  (debe ser ~0)")


if __name__ == "__main__":
    main()
