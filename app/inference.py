"""
==============================================================================
  INFERENCE — Carga de modelos y lógica de predicción/recomendación/similares
==============================================================================
  Sin dependencias externas más allá de las ya usadas para entrenar
  (scikit-learn / pandas / numpy / joblib). Lo consume app/server.py.

  Funciones públicas:
    get_config()                 -> esquema + grupos de features para armar el form
    predict(features, model)     -> probabilidades por modelo + incertidumbre + owners
    recommend(features, K, model)-> mejor paquete de cambios (beam search) para subir P(Hit)
    similar(features, k)         -> juegos reales del "mismo camino" (NearestNeighbors)
    search_games(q)              -> buscar juegos reales por nombre (modo validación)
    get_game(appid)              -> features de un juego real para cargarlo en el simulador
==============================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

CLASSES = ["Flop", "Rentable", "Hit"]
HIT_IDX = 2

# ── Carga única de artefactos ─────────────────────────────────────────────
SCHEMA = json.loads((MODELS_DIR / "feature_schema.json").read_text(encoding="utf-8"))
FEAT_COLS = SCHEMA["feat_cols"]
NUM_FEATS = list(SCHEMA["numeric"].keys())
BOOL_FEATS = SCHEMA["boolean"]
CAT_FEATS = list(SCHEMA["categorical"].keys())
THRESHOLDS = SCHEMA["thresholds"]

MODELS = {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in SCHEMA["models"]}
OWNERS_REG = joblib.load(MODELS_DIR / "owners_regressor.joblib")
SIMILAR = joblib.load(MODELS_DIR / "similar.joblib")

# ── Modelos post-lanzamiento (señales tempranas de recepción) ─────────────
POST_EXTRA = SCHEMA.get("post_extra", [])
POST_FEAT_COLS = SCHEMA.get("post_feat_cols", [])
POST_NUMERIC = SCHEMA.get("post_numeric", {})
POST_MODELS = {}
for _n in SCHEMA["models"]:
    _p = MODELS_DIR / f"post_{_n}.joblib"
    if _p.exists():
        POST_MODELS[_n] = joblib.load(_p)
POST_LABELS = {"ccu": "Jugadores concurrentes (pico)", "rating_porcentaje": "% reseñas positivas",
               "metacritic_score": "Nota Metacritic", "ts_positive_ratio": "Proporción positiva",
               "ts_avg_playtime_hrs": "Horas jugadas (prom.)", "ts_months_active": "Meses activo"}

# Nota: se probó un modelo en 2 etapas (tracción → clase) con `train_stage1.py`, pero la
# etapa de tracción resultó confundida por la metodología de recolección (owners>0 vino del
# crawl de SteamSpy y owners==0 del catálogo, con metadata poblada de forma distinta). Por eso
# NO se integra al producto; queda documentado como hallazgo en la documentación del estudio.

DF = pd.read_csv(OUTPUT_DIR / "dataset_ml.csv", sep=";")
PREVALENCE = {c: float(DF[c].mean()) for c in BOOL_FEATS}

# Booleanas que el recomendador puede togglear: prevalencia 5–95% y no plataformas/free.
_BLOCK = {"is_free"}
MUTABLE_BOOLS = [c for c in BOOL_FEATS
                 if 0.05 <= PREVALENCE[c] <= 0.95
                 and not c.startswith("platform_") and c not in _BLOCK]


def pretty_label(col: str) -> str:
    """tag_open_world -> 'Open World'; genre_rpg -> 'RPG'."""
    for p in ("genre_", "cat_", "tag_", "platform_", "is_"):
        if col.startswith(p):
            col = col[len(p):]
            break
    txt = col.replace("_", " ").strip()
    small = {"rpg", "fps", "pvp", "vr", "2d", "3d", "moba", "co op"}
    return " ".join(w.upper() if w.lower() in small else w.capitalize() for w in txt.split())


# Etiquetas legibles de las features no booleanas (para la explicabilidad)
NICE_LABELS = {
    "price": "Precio", "supported_languages": "Idiomas soportados",
    "total_achievements": "Logros", "total_dlcs": "DLCs", "min_ram_gb": "RAM mínima",
    "short_desc_len": "Longitud de la descripción", "dev_game_count": "Juegos previos del estudio",
    "pub_game_count": "Juegos previos del publisher", "dev_experience": "Experiencia del estudio",
    "pub_experience": "Experiencia del publisher", "controller_support": "Soporte de control",
    "release_quarter": "Trimestre de lanzamiento", "dev_success_prior": "Historial del estudio",
}
# Features derivadas: son consecuencia de otras, no decisiones; se omiten de la explicación.
_DERIVED_SKIP = {"num_genres", "num_tags", "num_categories", "price_tier"}


def feat_label(col: str) -> str:
    if col.startswith(("genre_", "cat_", "tag_", "platform_", "is_")):
        return pretty_label(col)
    return NICE_LABELS.get(col, col)


# ── Construcción del vector de entrada ────────────────────────────────────
def _defaults() -> dict:
    d = {c: SCHEMA["numeric"][c]["median"] for c in NUM_FEATS}
    d.update({c: 0 for c in BOOL_FEATS})
    d.update({c: SCHEMA["categorical"][c]["default"] for c in CAT_FEATS})
    return d


def normalize_state(features: dict) -> dict:
    """Mezcla los inputs del usuario con los defaults y re-deriva las features
    dependientes (num_tags, is_early_access, price_tier) para mantener coherencia."""
    s = _defaults()
    if features:
        for k, v in features.items():
            if k in s:
                s[k] = v
    # Re-derivar riqueza de catálogo desde los flags activos
    s["num_genres"] = sum(int(s.get(c, 0)) for c in BOOL_FEATS if c.startswith("genre_"))
    s["num_tags"] = sum(int(s.get(c, 0)) for c in BOOL_FEATS if c.startswith("tag_"))
    s["num_categories"] = sum(int(s.get(c, 0)) for c in BOOL_FEATS if c.startswith("cat_"))
    if "is_early_access" in s:
        s["is_early_access"] = int(bool(s.get("genre_early_access", 0)) or bool(s.get("tag_early_access", 0)))
    # price_tier coherente con el precio si el usuario movió el precio pero no el tier
    price = float(s.get("price", 0) or 0)
    s["price_tier"] = ("F2P" if price <= 0 else "Budget" if price < 10 else "Mid" if price < 20 else "Premium")
    return s


def _row(state: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: state.get(c, 0) for c in FEAT_COLS}])[FEAT_COLS]


def _post_state(features: dict) -> dict:
    """Estado pre-lanzamiento + las señales tempranas de recepción (default = mediana)."""
    s = normalize_state(features)
    for c in POST_EXTRA:
        s[c] = features.get(c, POST_NUMERIC.get(c, {}).get("median", 0))
    return s


def _post_row(state: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: state.get(c, 0) for c in POST_FEAT_COLS}])[POST_FEAT_COLS]


def _probs(state: dict, model: str) -> np.ndarray:
    return MODELS[model].predict_proba(_row(state))[0]


def _phit(state: dict, model: str) -> float:
    return float(_probs(state, model)[HIT_IDX])


def _explain(state: dict, model: str, target_idx: int = HIT_IDX, top: int = 6) -> list:
    """Contribución de cada feature a P(clase objetivo): cuánto cambia la probabilidad
    si esa feature se lleva a su valor base (mediana / 0 / categoría por defecto).
    Modelo-agnóstico: vale para LR, SVM y MLP por igual."""
    if model not in MODELS:
        model = "mlp" if "mlp" in MODELS else list(MODELS)[0]
    defaults = _defaults()
    base_p = float(MODELS[model].predict_proba(_row(state))[0][target_idx])
    out = []
    for col in FEAT_COLS:
        if col in _DERIVED_SKIP:
            continue
        if state.get(col) == defaults.get(col):
            continue
        reset = normalize_state({**state, col: defaults.get(col)})
        p = float(MODELS[model].predict_proba(_row(reset))[0][target_idx])
        delta = round(base_p - p, 4)
        if abs(delta) < 0.005:
            continue
        out.append({"factor": feat_label(col), "delta": delta})
    out.sort(key=lambda x: -abs(x["delta"]))
    return out[:top]


# ── API: predicción multi-modelo + incertidumbre ──────────────────────────
def predict(features: dict, model: str = "todos", mode: str = "pre") -> dict:
    pre_state = normalize_state(features)
    use_post = mode == "post" and bool(POST_MODELS)
    if use_post:
        mdl, row = POST_MODELS, _post_row(_post_state(features))
    else:
        mdl, row = MODELS, _row(pre_state)

    out, argmax_classes = {}, []
    for name in mdl:
        p = mdl[name].predict_proba(row)[0]
        out[name] = {"probs": {CLASSES[i]: round(float(p[i]), 4) for i in range(3)},
                     "clase": CLASSES[int(np.argmax(p))]}
        argmax_classes.append(out[name]["clase"])

    # owners y explicación se mantienen sobre las features pre-lanzamiento (decisiones de diseño)
    owners_est = max(0, int(np.expm1(OWNERS_REG.predict(_row(pre_state))[0])))
    ref_pre = "mlp" if "mlp" in MODELS else list(MODELS)[0]

    ref = "mlp" if "mlp" in out else list(out)[0]
    ref_sorted = sorted(out[ref]["probs"].values(), reverse=True)
    margen = round(ref_sorted[0] - ref_sorted[1], 4)
    desacuerdo = len(set(argmax_classes)) > 1
    nivel = "alta" if (desacuerdo or margen < 0.12) else "media" if margen < 0.30 else "baja"

    return {
        "modo": "post" if use_post else "pre",
        "modelos": out,
        "owners_estimados": owners_est,
        "clase_por_owners": _class_from_owners(owners_est),
        "incertidumbre": {"margen_top2": margen, "desacuerdo": desacuerdo,
                          "nivel": nivel, "modelo_ref": ref},
        "explicacion": {"target": "Hit", "modelo": ref_pre,
                        "factores": _explain(pre_state, ref_pre, HIT_IDX)},
    }


def _class_from_owners(owners: float, flop_max=None, hit_min=None) -> str:
    flop_max = flop_max or THRESHOLDS["flop_max"]
    hit_min = hit_min or THRESHOLDS["hit_min"]
    if owners >= hit_min:
        return "Hit"
    if owners >= flop_max:
        return "Rentable"
    return "Flop"


# ── API: recomendador combinatorio (beam search) ──────────────────────────
def _candidate_moves(state: dict):
    moves = []
    price = float(state.get("price", 0) or 0)
    for dp in (-5, -3, 3, 5):
        np_ = round(min(60.0, max(0.0, price + dp)), 2)
        if abs(np_ - price) > 1e-6:
            moves.append(("price", np_, f"Ajustar precio a ${np_:.2f}"))
    langs = int(state.get("supported_languages", 0) or 0)
    for dl in (4, 8):
        moves.append(("supported_languages", langs + dl, f"Soportar +{dl} idiomas (total {langs+dl})"))
    if state.get("controller_support") != "full":
        moves.append(("controller_support", "full", "Soporte completo de control"))
    if state.get("release_quarter") != "Q4":
        moves.append(("release_quarter", "Q4", "Lanzar en Q4 (temporada alta)"))
    for b in MUTABLE_BOOLS:
        cur = int(state.get(b, 0))
        verb = "Quitar" if cur else "Añadir"
        moves.append((b, 0 if cur else 1, f"{verb} '{pretty_label(b)}'"))
    return moves


def _apply(state: dict, feat: str, val) -> dict:
    ns = dict(state)
    ns[feat] = val
    return normalize_state(ns)


def _state_key(state: dict) -> tuple:
    return tuple(state.get(c) for c in FEAT_COLS)


def recommend(features: dict, K: int = 3, model: str = "mlp", beam: int = 4) -> dict:
    if model not in MODELS:
        model = "mlp" if "mlp" in MODELS else list(MODELS)[0]
    base = normalize_state(features)
    base_phit = _phit(base, model)

    # Cambios individuales (interpretabilidad)
    singles = []
    for feat, val, desc in _candidate_moves(base):
        d = _phit(_apply(base, feat, val), model) - base_phit
        singles.append({"desc": desc, "delta": round(d, 4)})
    singles.sort(key=lambda x: -x["delta"])

    # Beam search del mejor paquete
    beam_states = [(base_phit, base, [])]
    best = (base_phit, base, [])
    seen = {_state_key(base)}
    for _ in range(K):
        cands = []
        for _, state, changes in beam_states:
            used = {c["feature"] for c in changes}
            for feat, val, desc in _candidate_moves(state):
                if feat in used:
                    continue
                ns = _apply(state, feat, val)
                key = _state_key(ns)
                if key in seen:
                    continue
                seen.add(key)
                ph = _phit(ns, model)
                cands.append((ph, ns, changes + [{"feature": feat, "desc": desc, "phit": round(ph, 4)}]))
        if not cands:
            break
        cands.sort(key=lambda x: -x[0])
        beam_states = cands[:beam]
        if beam_states[0][0] > best[0]:
            best = beam_states[0]

    return {
        "modelo": model,
        "phit_antes": round(base_phit, 4),
        "phit_despues": round(best[0], 4),
        "delta": round(best[0] - base_phit, 4),
        "cambios": best[2],
        "cambios_individuales": singles[:6],
    }


# ── API: juegos del mismo camino ──────────────────────────────────────────
def similar(features: dict, k: int = 8) -> dict:
    state = normalize_state(features)
    Xt = SIMILAR["preprocessor"].transform(_row(state))
    k = min(int(k), 12)
    dist, idx = SIMILAR["nn"].kneighbors(Xt, n_neighbors=k)
    meta = SIMILAR["meta"]
    juegos = []
    for d, i in zip(dist[0], idx[0]):
        r = meta.iloc[int(i)]
        juegos.append({
            "appid": int(r["appid"]), "name": str(r["name"]),
            "developer": str(r["developer"]), "clase": str(r["label_name"]),
            "owners": int(r["owners_lower_bound"]), "price": float(r["price"]),
            "similitud": round(1 - float(d), 3),
        })
    return {"juegos": juegos}


# ── API: búsqueda de juegos reales (modo validación) ──────────────────────
def search_games(q: str, limit: int = 20) -> dict:
    if not q or len(q) < 2:
        return {"juegos": []}
    m = DF[DF["name"].astype(str).str.contains(q, case=False, na=False, regex=False)].head(limit)
    return {"juegos": [{"appid": int(row["appid"]), "name": str(row["name"]),
                        "clase": str(row["label_name"]), "owners": int(row["owners_lower_bound"])}
                       for _, row in m.iterrows()]}


def get_game(appid: int) -> dict:
    g = DF[DF["appid"] == int(appid)]
    if g.empty:
        return {"error": "no encontrado"}
    row = g.iloc[0]
    feats = {c: (float(row[c]) if isinstance(row[c], (int, float, np.number)) else str(row[c]))
             for c in FEAT_COLS}
    return {"appid": int(appid), "name": str(row["name"]),
            "clase_real": str(row["label_name"]), "owners_real": int(row["owners_lower_bound"]),
            "features": feats}


# ── API: config para construir el formulario en el frontend ───────────────
def get_config() -> dict:
    groups = {"genre": [], "cat": [], "tag": [], "platform": []}
    for c in BOOL_FEATS:
        for p in groups:
            if c.startswith(p + "_"):
                groups[p].append({"col": c, "label": pretty_label(c),
                                  "prevalencia": round(PREVALENCE[c], 3),
                                  "mutable": c in MUTABLE_BOOLS})
                break
    numeric = {c: {**SCHEMA["numeric"][c], "label": pretty_label(c)} for c in NUM_FEATS}
    return {
        "classes": CLASSES,
        "thresholds": THRESHOLDS,
        "models": list(MODELS.keys()),
        "numeric": numeric,
        "categorical": SCHEMA["categorical"],
        "groups": groups,
        "dev_prior_global": SCHEMA.get("dev_prior_global"),
        "post": {
            "enabled": bool(POST_MODELS),
            "extra": POST_EXTRA,
            "numeric": {c: {**POST_NUMERIC.get(c, {}), "label": POST_LABELS.get(c, c)}
                        for c in POST_EXTRA},
        },
    }
