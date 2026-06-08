# Predictor de Éxito Comercial de Videojuegos en Steam

### ▶ Demo en vivo: **https://steampredict.onrender.com**

> Alojado en el plan gratuito de Render. Si nadie lo visitó en los últimos ~15 minutos, el primer acceso
> tarda **30–60 s** en "despertar" el servidor; el dashboard muestra una pantalla de carga mientras tanto.

Sistema de Machine Learning que estima, **antes del lanzamiento**, en qué categoría comercial caerá un
videojuego de PC: **Flop** (<200k propietarios), **Rentable** (200k–1M) o **Hit** (≥1M). Todo el modelado es
estrictamente **pre-lanzamiento**: usa solo variables conocibles antes de publicar el juego (precio, géneros,
tags, idiomas, experiencia del estudio…), nunca reseñas ni métricas posteriores a la venta.

---

## Índice

- [Inicio rápido (local)](#inicio-rápido-local)
- [El dashboard](#el-dashboard)
- [Modelos y resultados](#modelos-y-resultados)
- [Datos](#datos)
- [Reproducir el pipeline](#reproducir-el-pipeline)
- [Entrenamiento en Databricks](#entrenamiento-en-databricks)
- [Ingeniería de datos](#apartado-de-ingeniería-de-datos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Documentación](#documentación)

---

## Inicio rápido (local)

Requiere Python con las dependencias de `requirements.txt` (versiones exactas: los `.joblib` del repo se
serializaron con scikit-learn 1.8.0). El backend usa solo la librería estándar.

```bash
git clone https://github.com/SamCCDev/PI_steam.git
cd PI_steam
pip install -r requirements.txt
python app/server.py
```

Abre **http://127.0.0.1:8000**. Los modelos entrenados (`models/*.joblib`) ya vienen en el repo, así que el
dashboard funciona sin reentrenar.

---

## El dashboard

Cinco vistas con estética Steam:

| Vista | Qué hace |
|---|---|
| **Simulador** | Configura un juego hipotético → embudo de 2 etapas, probabilidades por clase, owners estimados, incertidumbre |
| **Comparar modelos** | LR / SVM / MLP lado a lado sobre la misma configuración + métricas de evaluación |
| **Mismo camino** | Juegos reales con perfil de features cercano al tuyo (vecinos más próximos) |
| **Recomendaciones** | Mejor paquete de cambios accionables para subir P(Hit) (búsqueda voraz) |
| **Panel analítico** | 7 gráficos del mercado con datos vivos del backend, **reactivos a la simulación**: marcan tu rango de precio, tus géneros, tu trimestre y ubican "Tu juego" en el scatter precio/owners |

Cada gráfica y tarjeta lleva un icono **?** con la explicación de qué muestra y cómo leerla. También hay un
buscador de juegos reales (modo validación: predicción vs resultado real) y una terminal con comandos
(`help`, `set`, `find`, `predict`…).

---

## Modelos y resultados

Tres clasificadores con el mismo preprocesamiento (estandarización + one-hot + paso directo de binarias,
`class_weight="balanced"`), más una regresión de owners que habilita umbrales ajustables.

| Modelo | AUC (OVR-macro) | F1 (macro) | Accuracy |
| :--- | :---: | :---: | :---: |
| Regresión Logística | 0.884 | 0.692 | 0.697 |
| SVM (RBF) | 0.881 | 0.719 | 0.739 |
| **MLP (ReLU)** | **0.894** | **0.751** | **0.783** |

Medido en test (split 80/20 estratificado sobre 7.817 juegos con `owners>0`). Balance de clases
27% Flop / 54% Rentable / 18% Hit. El MLP es el modelo de referencia del dashboard; la LR se conserva por
interpretabilidad. Detalle, matrices de confusión y una guía de conceptos están en la documentación del estudio.

### Embudo de dos etapas

El simulador combina dos modelos: **Etapa 1** estima P(el juego logra tracción comercial) sobre 17.565 juegos
con metadata completa (AUC 0.90), y **Etapa 2** predice Flop/Rentable/Hit condicionado a esa tracción. El
resultado son cuatro probabilidades (sin tracción / Flop / Rentable / Hit). Entrenar la etapa 1 sobre datos sin
filtrar daba un AUC engañoso (0.96) por un sesgo de recolección; el detalle está en la documentación (§5.5).

---

## Datos

### Flujo de extremo a extremo

```
1. ETL          scripts/steam_etl.py       → output/*.csv          (4 tablas crudas de Steam/SteamSpy)
2. Dataset      scripts/build_dataset.py   → output/dataset_ml.csv (1 fila/juego, 92 features, sin nulos)
3. Entreno      scripts/train_models.py    → models/*.joblib       (LR, SVM-RBF, MLP, owners, NN) + reports/
                scripts/train_stage1.py    → models/stage1.joblib  (etapa 1 del embudo: tracción)
4. Backend      app/server.py              → carga los .joblib y expone /api/* (predict, recommend, …)
5. Frontend     web/index.html             → consume /api/* y dibuja las 5 vistas
```

Embudo de datos: **32.966** juegos recolectados → **17.565** con metadata completa (universo de la etapa 1)
→ **7.817** con ventas estimables por SteamSpy (entrenamiento del clasificador).

### Variables y prevención de fuga de datos

El conjunto tiene **92 features** (12 numéricas, 75 binarias, 5 categóricas). El predictor es **pre-lanzamiento**,
así que se **excluyen** del modelado las variables que solo se conocen tras lanzar: reseñas
(`positive`/`negative`), valoración, Metacritic, jugadores concurrentes (`ccu`) y los agregados `ts_*`. El
diccionario `output/dataset_ml_dictionary.csv` marca el rol y la `etapa` de cada columna.

Features de catálogo: conteos (`num_tags`, `num_genres`, `num_categories`), `release_quarter`,
`pub_experience`/`pub_game_count`, `price_tier`, `is_early_access` y `dev_success_prior` (prior bayesiano por
estudio calculado *leave-one-out* para no filtrar la propia etiqueta).

### Calidad de datos: precios de moneda regional

El scrape original capturó 46 precios en moneda regional o de otra edición (Cyberpunk 2077 a $199,
Red Dead Redemption 2 a $53.990). Se re-consultaron esos 46 appids contra la Steam Storefront API
forzando región US (`cc=us`) y se reescribió `games_metadata.csv`; los precios altos legítimos
(RPG Maker, juegos de precio-broma a $200) se conservaron. Tras la reparación se regeneró el dataset
y se reentrenaron los modelos. La corrección ya está aplicada en los CSV versionados.

---

## Entorno

Todo el pipeline local (ETL, dataset, entrenamiento y backend) corre con un único entorno.
Las versiones exactas están en `requirements.txt` (las mismas que usa Render).

```bash
# Opción A — micromamba / conda / mamba (entorno aislado, recomendado)
micromamba create -f environment.yml
micromamba activate steampredict

# Opción B — venv + pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir el pipeline

```bash
python scripts/steam_etl.py --sample 2000 --source steamspy   # (opcional) traer más juegos
python scripts/build_dataset.py                               # regenerar el master con todas las features
python scripts/train_models.py                                # modelos LR/SVM/MLP + owners + NN
python scripts/train_stage1.py                                # etapa 1 del embudo: tracción (filtrado)
python app/server.py                                          # levantar el dashboard
```

`build_dataset.py` es *schema-driven* (detecta tags/géneros por prefijo y descarta columnas
casi-constantes) e idempotente: re-ejecutar tras actualizar los CSV regenera todo.

### ETL — uso de `steam_etl.py`

Motor de extracción concurrente (multihilo + limitador de tasa para evitar HTTP 429).

```bash
python scripts/steam_etl.py                       # extracción estándar
python scripts/steam_etl.py --sample 2000 --source steamspy   # juegos con ventas reales (SteamSpy)
python scripts/steam_etl.py --validate 730        # diagnóstico de un appid sin escribir archivos
```

Produce 4 CSV relacionales en `output/` (separador `;`): `games_metadata`, `games_tags`, `games_text`,
`games_timeseries`. Configura tu `STEAM_API_KEY` en un `.env` (ver `.env.example`) para consultas ampliadas.

---

## Entrenamiento en Databricks

El entrenamiento tiene **una sola fuente**: `scripts/train_models.py`. El mismo archivo detecta dónde corre —
en local lee `output/dataset_ml.csv` y guarda los `.joblib`; en Databricks lee la tabla de Unity Catalog
`dataset_ml` y guarda las métricas y coeficientes como tablas (`model_metrics`, `model_lr_coefficients`). El
notebook `notebooks/08_train_all.py` no duplica nada: solo importa y ejecuta ese archivo, de modo que el
resultado es idéntico al local.

**Configuración (una vez):**
1. Generar el master local: `python scripts/build_dataset.py`.
2. Subir `output/dataset_ml.csv` como tabla **`dataset_ml`** (Catalog → Create Table → CSV, separador `;`).
3. Importar el notebook 08 y ejecutarlo (ajustar `UC_DATASET`/`UC_METRICS`/`UC_COEF` en `train_models.py` si
   el catálogo/esquema no son `workspace`/`default`).

**Restricciones de Free Edition (serverless) respetadas:** MLlib clásica bloqueada → se usa scikit-learn sobre
`toPandas()`; DBFS público deshabilitado → datos como tablas Unity Catalog; sin `.cache()`; sin MLflow
`start_run`. Por eso los `.joblib` que sirve el backend se generan en **local** (Free Edition no permite
descargar archivos del serverless salvo vía un Volume de UC).

---

## Apartado de Ingeniería de Datos

`ingenieria_datos/` aplica dos técnicas del curso a los datos de Steam: **anonimización SHA-256** de
estudio/distribuidora (`steam_anonimizar.py`) y **análisis con Spark RDD** sobre los 32.966 registros
(`steam_rdd_analisis.py`, para Databricks; con verificación local en pandas). Ver `ingenieria_datos/README.md`.

---

## Estructura del proyecto

```text
├── README.md                     # esta guía
├── requirements.txt              # versiones exactas (los .joblib requieren scikit-learn 1.8.0)
├── environment.yml               # entorno micromamba/conda (reusa requirements.txt)
├── render.yaml                   # blueprint del despliegue en Render (plan free)
├── scripts/                      # pipeline de datos y entrenamiento (correr desde la raíz)
│   ├── steam_etl.py              #   ETL concurrente (Steam Storefront + SteamSpy + Reviews)
│   ├── build_dataset.py          #   consolida los 4 CSV en el master ML-ready
│   ├── train_models.py           #   fuente única de entrenamiento (local y Databricks)
│   └── train_stage1.py           #   etapa 1 del embudo: modelo de tracción (datos filtrados)
├── app/                          # backend (Python stdlib, sin dependencias)
│   ├── server.py                 #   http.server + router de endpoints
│   ├── inference.py              #   carga de modelos, predict, recommend, similar, embudo
│   └── stats.py                  #   agregados para el panel analítico
├── web/index.html                # frontend completo (una sola página, ECharts, estética Steam)
├── vendor/                       # tailwind.js, lucide.js, echarts.min.js (offline)
├── docs/                         # informe del estudio + documento técnico (+ bitácora interna)
├── models/                       # *.joblib + feature_schema.json + stage1.joblib
├── reports/                      # metrics.json, confusion_*.json, stage1.json
├── notebooks/                    # Databricks: 01–08 (08 = invoca train_models.py)
├── ingenieria_datos/             # apartado del curso: anonimización SHA-256 + Spark RDD
└── output/                       # datasets (CSV, separador ';')
```

---

## Documentación

- [`docs/Documentacion del Estudio - Predictor.md`](docs/Documentacion%20del%20Estudio%20-%20Predictor.md) — informe metodológico y de resultados (incluye la guía de conceptos del dashboard).
- [`docs/Documento Tecnico - Predictor de Videojuegos.md`](docs/Documento%20Tecnico%20-%20Predictor%20de%20Videojuegos.md) — especificación técnica (objetivo, arquitectura, diccionario de datos).
