"""
==============================================================================
  STATS — Agregados del dataset para el panel analítico (estilo Power BI)
==============================================================================
  Calcula una sola vez (al importar) los datos que alimentan los gráficos
  ECharts del frontend. Reutiliza el DataFrame ya cargado en inference.py.
==============================================================================
"""

import json
from pathlib import Path

from app import inference as inf

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "output"

DF = inf.DF


def _market():
    vc = DF["label_name"].value_counts()
    return [{"clase": c, "n": int(vc.get(c, 0))} for c in ["Flop", "Rentable", "Hit"]]


def _price_hist():
    bins = [0, 5, 10, 15, 20, 30, 40, 60, 1e9]
    labels = ["F2P/<5", "5–10", "10–15", "15–20", "20–30", "30–40", "40–60", "60+"]
    price = DF["price"].clip(lower=0)
    counts = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sel = (price >= lo) & (price < hi) if i == 0 else (price >= lo) & (price < hi)
        counts.append(int(sel.sum()))
    return {"labels": labels, "counts": counts}


def _owners_by_genre():
    genres = [c for c in DF.columns if c.startswith("genre_")]
    rows = []
    for g in genres:
        sel = DF[DF[g] == 1]
        if len(sel) < 30:
            continue
        rows.append({"genero": inf.pretty_label(g), "n": int(len(sel)),
                     "owners_mediana": int(sel["owners_lower_bound"].median()),
                     "pct_hit": round(float((sel["label"] == 2).mean()) * 100, 1)})
    rows.sort(key=lambda r: -r["owners_mediana"])
    return rows[:12]


def _lr_importance():
    """Top features que empujan hacia/afuera de Hit, según los coeficientes del LR."""
    p = OUTPUT_DIR / "model_web.json"
    if not p.exists():
        return {"positivos": [], "negativos": []}
    coef = json.loads(p.read_text(encoding="utf-8"))["coef"]["Hit"]
    # Se omite la categoría 'Desconocido' (fecha faltante): es un artefacto, no una decisión de diseño.
    items = sorted(((k, v) for k, v in coef.items() if "Desconocido" not in k), key=lambda kv: kv[1])
    def clean(name):
        return inf.pretty_label(name) if any(name.startswith(x) for x in
               ("genre_", "cat_", "tag_", "platform_", "is_")) else name.replace("_", " ")
    neg = [{"feature": clean(k), "coef": round(v, 3)} for k, v in items[:8]]
    pos = [{"feature": clean(k), "coef": round(v, 3)} for k, v in items[-8:][::-1]]
    return {"positivos": pos, "negativos": neg}


def _metrics():
    p = REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _confusion():
    out = {}
    for name in ("lr", "svm", "mlp"):
        p = REPORTS_DIR / f"confusion_{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _hit_by_quarter():
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sel = DF[DF["release_quarter"] == q]
        if len(sel) < 20:
            continue
        rows.append({"q": q, "n": int(len(sel)),
                     "pct_hit": round(float((sel["label"] == 2).mean()) * 100, 1),
                     "pct_norentable": round(float((sel["label"] >= 1).mean()) * 100, 1)})
    return rows


def _scatter(n=500):
    sample = DF.sample(min(n, len(DF)), random_state=7)
    pts = []
    for _, r in sample.iterrows():
        pts.append([round(float(r["price"]), 2),
                    int(r["owners_lower_bound"]),
                    str(r["label_name"])])
    return pts


# Se calcula una vez al importar el módulo
PAYLOAD = {
    "market": _market(),
    "price_hist": _price_hist(),
    "owners_by_genre": _owners_by_genre(),
    "lr_importance": _lr_importance(),
    "metrics": _metrics(),
    "confusion": _confusion(),
    "scatter": _scatter(),
    "hit_by_quarter": _hit_by_quarter(),
    "n_total": int(len(DF)),
}


def get_stats():
    return PAYLOAD
