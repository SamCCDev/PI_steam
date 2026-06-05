# Predictor de Éxito Comercial de Videojuegos en Steam

Sistema de Machine Learning que estima, **antes del lanzamiento**, en qué categoría comercial caerá un
videojuego de PC: **Flop** (<200k propietarios), **Rentable** (200k–1M) o **Hit** (≥1M). Incluye un ETL de
datos de Steam/SteamSpy, un pipeline de entrenamiento reproducible y un **dashboard interactivo con backend
local** (estética Steam) que compara tres modelos, explica sus predicciones y recomienda mejoras.

> **Dos versiones conviven en el repo:**
> - **v2 (actual):** dashboard con backend (`app/` + `web/`), 3 modelos comparables, 7.817 juegos. Es lo principal.
> - **v1 (la prueba):** dashboard estático de una sola página con el modelo embebido (`steampredict_dashboard_comercial.html`). Se conserva como contexto y *fallback* offline.
>
> El plan/bitácora vivo está en [`Plan v2 - Dashboard Comercial Customizable.md`](Plan%20v2%20-%20Dashboard%20Comercial%20Customizable.md) y la documentación académica en [`Documentacion del Estudio - Predictor v2.md`](Documentacion%20del%20Estudio%20-%20Predictor%20v2.md).

---

## Inicio rápido (dashboard v2)

Requiere Python con `scikit-learn`, `pandas`, `numpy`, `joblib` (ya usados para entrenar). El backend usa solo
la librería estándar, **no necesita instalar nada extra**.

```bash
git clone https://github.com/SamCCDev/PI_steam.git
cd PI_steam
python app/server.py
```

Abre **http://127.0.0.1:8000**. Los modelos entrenados (`models/*.joblib`) ya vienen en el repo, así que el
dashboard funciona sin reentrenar y **sin conexión a internet**.

El dashboard tiene 5 vistas: **Simulador** (predicción + incertidumbre), **Comparar modelos** (LR/SVM/MLP lado
a lado), **Juegos del mismo camino** (vecinos reales), **Recomendaciones** (mejor paquete de cambios para subir
P(Hit)) y **Panel analítico** (gráficos del mercado).

---

## Flujo de datos (de extremo a extremo)

```
1. ETL          steam_etl.py        → output/*.csv            (4 tablas crudas de Steam/SteamSpy)
2. Dataset      build_dataset_v2.py → output/dataset_ml.csv   (1 fila/juego, 92 features, sin nulos)
3. Entreno      train_models.py     → models/*.joblib         (LR, SVM-RBF, MLP, regresor de owners, NN)
                                       reports/metrics.json    (AUC/F1/accuracy + matrices)
4. Backend      app/server.py       → carga los .joblib y expone /api/* (predict, recommend, similar, stats)
5. Frontend     web/                → consume /api/* y dibuja las 5 vistas
```

El mismo entrenamiento corre en **Databricks** (`notebooks/08_train_all.py`) para la parte de escala/académica.

---

## Estructura del proyecto

```text
├── steam_etl.py                  # ETL concurrente (Steam Storefront + SteamSpy + Reviews)
├── build_dataset_v2.py           # Consolida los 4 CSV en el master ML-ready (features v2)
├── train_models.py               # Entrena y serializa todos los modelos (fuente de los .joblib)
├── export_model_web.py           # Exporta la LR a JSON para el dashboard estático v1
├── embed_model_in_html.py        # Inyecta ese JSON en el HTML v1
├── app/                          # Backend (Python stdlib, sin dependencias)
│   ├── server.py                 #   http.server + router de endpoints
│   ├── inference.py              #   carga de modelos, predict, recommend, similar
│   └── stats.py                  #   agregados para el panel analítico
├── web/                          # Frontend SPA (estética Steam)
│   ├── index.html  css/  js/     #   app.js, charts.js (ECharts), api.js
│   └── vendor/ -> ../vendor/     #   tailwind.js, lucide.js, echarts.min.js (offline)
├── models/                       # *.joblib + feature_schema.json (artefactos del backend)
├── reports/                      # metrics.json, confusion_*.json
├── notebooks/                    # Databricks: 01–07 (v1) + 08_train_all (v2 consolidado)
├── ingenieria_datos/             # Apartado del curso: anonimización SHA-256 + Spark RDD
├── output/                       # Datasets (CSV, separador ';')
└── steampredict_dashboard_comercial.html   # Dashboard estático v1 (la prueba)
```

---

## Reproducir el pipeline

```bash
python steam_etl.py --sample 2000 --source steamspy   # (opcional) traer más juegos
python build_dataset_v2.py                            # regenerar el master con features v2
python train_models.py                                # reentrenar todos los modelos -> models/ y reports/
python app/server.py                                  # levantar el dashboard
```

`build_dataset_v2.py` es *schema-driven* (detecta tags/géneros por prefijo y descarta columnas
casi-constantes) e idempotente: re-ejecutar tras actualizar los CSV regenera todo.

### Variables y prevención de fuga de datos

El conjunto tiene **92 features** (12 numéricas, 75 binarias, 5 categóricas). El predictor es **pre-lanzamiento**,
así que se **excluyen** las variables que solo se conocen tras lanzar: reseñas (`positive`/`negative`),
valoración, Metacritic, jugadores concurrentes (`ccu`) y los agregados `ts_*`. El diccionario
`output/dataset_ml_dictionary.csv` marca el rol y la `etapa` (pre/post/meta) de cada columna.

Features nuevas de la v2: conteos de catálogo (`num_tags`, `num_genres`, `num_categories`), `release_quarter`,
`pub_experience`/`pub_game_count`, `price_tier`, `is_early_access` y `dev_success_prior` (prior bayesiano por
estudio calculado *leave-one-out* para no filtrar la propia etiqueta).

---

## Modelos y resultados

Tres clasificadores con el mismo preprocesamiento (estandarización + one-hot + paso directo de binarias,
`class_weight="balanced"`), más una regresión de owners que habilita umbrales ajustables.

| Modelo | AUC (OVR-macro) | F1 (macro) | Accuracy |
| :--- | :---: | :---: | :---: |
| Regresión Logística | 0.884 | 0.692 | 0.697 |
| SVM (RBF) | 0.881 | 0.720 | 0.739 |
| **MLP (ReLU)** | **0.888** | **0.740** | **0.772** |

Medido en test (split 80/20 estratificado sobre 7.817 juegos con `owners>0`). Balance de clases
27% Flop / 54% Rentable / 18% Hit. El MLP es el modelo de referencia del dashboard; la LR se conserva por
interpretabilidad. Detalle y matrices de confusión en la documentación del estudio.

---

## Entrenamiento en Databricks (Free Edition)

`notebooks/08_train_all.py` reproduce `train_models.py` en Databricks: lee la tabla `dataset_ml` de Unity
Catalog, entrena LR (con `GridSearchCV`), SVM-RBF y MLP, y guarda métricas y coeficientes como tablas UC
(`model_metrics_v2`, `model_lr_coefficients_v2`).

**Configuración (una vez):**
1. Generar el master local: `python build_dataset_v2.py`.
2. Subir `output/dataset_ml.csv` como tabla **`dataset_ml`** (Catalog → Create Table → CSV, separador `;`).
3. Importar el notebook y ejecutarlo. Ajustar `CATALOG`/`SCHEMA` si no son `workspace`/`default`.

**Restricciones de Free Edition (serverless) respetadas:** MLlib clásica bloqueada → se usa scikit-learn sobre
`toPandas()`; DBFS público deshabilitado → datos como tablas Unity Catalog; sin `.cache()`; sin MLflow
`start_run`. Por eso los `.joblib` que sirve el backend se generan en **local** con `train_models.py` (Free
Edition no permite descargar archivos del serverless salvo vía un Volume de UC).

---

## Apartado de Ingeniería de Datos (`ingenieria_datos/`)

Aplica dos técnicas del curso a los datos de Steam: **anonimización SHA-256** de estudio/distribuidora
(`steam_anonimizar.py`) y **análisis con Spark RDD** sobre los 32.966 registros (`steam_rdd_analisis.py`, para
Databricks; con verificación local en pandas). Ver `ingenieria_datos/README.md`.

---

## Dashboard estático v1 (la prueba)

[`steampredict_dashboard_comercial.html`](steampredict_dashboard_comercial.html) corre en el navegador **sin
servidor** (modelo LR embebido, softmax replicado en JavaScript). Publicado en GitHub Pages:
**https://samccdev.github.io/PI_steam/**. 100% offline (Tailwind/lucide vendoreados). Regenerar tras reentrenar:

```bash
python build_dataset_v2.py && python export_model_web.py && python embed_model_in_html.py
```

---

## ETL — uso de `steam_etl.py`

Motor de extracción concurrente (multihilo + limitador de tasa para evitar HTTP 429).

```bash
python steam_etl.py                       # extracción estándar
python steam_etl.py --sample 2000 --source steamspy   # juegos con ventas reales (SteamSpy)
python steam_etl.py --validate 730        # diagnóstico de un appid sin escribir archivos
```

Produce 4 CSV relacionales en `output/` (separador `;`): `games_metadata`, `games_tags`, `games_text`,
`games_timeseries`. Configura tu `STEAM_API_KEY` en un `.env` (ver `.env.example`) para consultas ampliadas.
