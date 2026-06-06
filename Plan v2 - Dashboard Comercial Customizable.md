# Plan v2 — Dashboard Comercial Customizable (SteamPredict)

> **Documento vivo / bitácora.** Sirve para dos cosas: (1) fijar el diseño de la versión 2 antes de
> escribir código y (2) registrar el avance. Cada fase tiene checkboxes; se marcan a medida que se completan.
> La sección [§13 Bitácora de avance](#13-bitácora-de-avance) lleva el registro cronológico.

**Fecha de creación:** 2026-06-03 · **Deadline de presentación:** lunes 2026-06-08 (tarde) · **Ventana de trabajo:** fin de semana 6–7 jun.
**Autor del proyecto:** SamDev · **Estado global:** `EN CURSO` — Fase 0 (datos + entrenamiento) completada y verificada el 2026-06-03.

**Relación con v1:** esta versión **no reemplaza** `steampredict_dashboard_comercial.html`. La v1 (estática, modelo embebido en JSON)
queda como *prueba de concepto*, fallback offline y contexto. La v2 es una aplicación nueva con backend.

---

## 0. Resumen ejecutivo (qué es la v2 en una frase)

Una aplicación local (backend en Python stdlib + frontend estética Steam) donde el usuario simula un videojuego,
**compara tres modelos** (LR / SVM-RBF / MLP) sobre el mismo input, ve la **incertidumbre** de la predicción,
recibe un **paquete de recomendaciones** para subir su probabilidad de Hit, se compara contra **juegos reales del
mismo camino**, y explora un **panel analítico tipo Power BI**. Todo reproducible y 100% offline.

---

## 1. Decisiones de diseño (las 20 preguntas, consolidadas)

| # | Tema | Decisión tomada |
|---|------|-----------------|
| 1 | Selección de datos | Objetivo ~7k con `owners>0` + extra con ≥1 review (incluye gratuitos/demos) para mayor aleatoriedad. Descarga de ~2k en curso. |
| 2 | Umbrales Flop/Rentable/Hit | Se mantienen fijos (200k / 1M). **Ajustables** vía slider gracias a un modelo de regresión de owners (ver §4.4). |
| 3 | Features nuevas | `num_tags` (≥15 tags considerados), idiomas soportados, tamaño de estudio (dev + publisher), y otras (ver §3.1). |
| 4 | Post-lanzamiento | Sí — track separado que usa señales tempranas (reviews/rating/ccu). |
| 5 | "Customizable" | (a) ajustar inputs del juego **y** (b) elegir/comparar modelos. |
| 6 | Comparación de modelos | Predicciones distintas lado a lado (no un solo "mejor"). |
| 7 | Incertidumbre | Sí — se muestra explícitamente (ver §4.5). |
| 8 | Recomendador | Combinatorio: propone el mejor *paquete* de cambios, no cambios sueltos (ver §6). |
| 9 | Estética | Estilo Steam, mejora considerable del dashboard, varias páginas, gráficos estilo Power BI. |
| 10 | Páginas | Simulador · Comparación de modelos · Juegos similares · Análisis & Recomendaciones · Panel analítico. |
| 11 | Gráficos Power BI | Sección/panel aparte con visualizaciones densas (ver §5.5). |
| 12 | Modos | Modo presentación (guiado) + modo quick (carga rápida de datos). |
| 13 | Juegos reales | Sí, si la complejidad lo permite — elegir un juego del dataset y ver predicción vs. realidad. |
| 14 | Idioma | Español únicamente. |
| 15 | Técnica | Backend (corre en la laptop, reproducible). Entrenamiento idealmente en Databricks. |
| 16 | Reentrenar | Sí, desde cero, aprovechando lo aprendido. Reusar recursos existentes (no reinventar). |
| 17 | v1 vs v2 | v2 aparte; v1 como contexto. |
| 18 | Prioridad | Riqueza visual primero; documentar todo al final para el estudio. |
| 19 | Rúbrica | Sin requisito duro. Posible ángulo: datos de panel, ingeniería de datos (anonimización, Spark). |
| 20 | Deadline | Lunes por la tarde; disponible todo el fin de semana. |

---

## 2. Veredicto de viabilidad

Leyenda: 🟢 viable en el plazo · 🟡 viable con riesgo / recorte · 🔴 fuera de alcance este finde.

| Capacidad | Estado | Nota |
|-----------|:---:|------|
| Reentreno con datos mejor seleccionados | 🟢 | Cabe en memoria; reentreno en segundos. Build contra dataset actual, *swap* al terminar el ETL. |
| Comparación LR/SVM/MLP lado a lado | 🟢 | Requiere backend: SVM-RBF y MLP no se replican en JS como el LR de v1. |
| Incertidumbre | 🟢 | `predict_proba`, margen top-2 y desacuerdo entre modelos. |
| Recomendador combinatorio | 🟢 | Greedy/beam en backend con los modelos reales. |
| Juegos del mismo camino | 🟢 | `NearestNeighbors`/cosine sobre el vector estandarizado. |
| Umbrales ajustables | 🟢 | Regresión de owners → reclasificación client-side sin reentrenar. |
| Variante post-lanzamiento | 🟡 | Segundo track; depende del tiempo. Bien rotulado para evitar confusión con el pre-lanzamiento. |
| Estética Steam + Power BI | 🟢 | ECharts vendorizado + paleta Steam en CSS. |
| NLP de la descripción | 🟡 | Da señal real, pero es lo más caro. *Stretch* del domingo. |
| Entrenar en Databricks | 🟢 | Un único `train_models.py` corre igual en laptop y en notebook Databricks (evita sacar `.joblib` del serverless). |
| Datos de panel / Spark / anonimización | 🟡 | Para la documentación del estudio, no para el producto. |

---

## 3. Capa de datos

### 3.1 Features nuevas a derivar (sobre las ~82 actuales)

| Feature | Tipo | Fuente / derivación | Pre/Post |
|---------|------|---------------------|:--------:|
| `num_tags` | int | conteo de `tag_*` activos por juego | pre |
| `num_genres` | int | conteo de `genre_*` | pre |
| `release_month` | cat (1–12) | de `release_date` | pre |
| `release_quarter` | cat (Q1–Q4) | de `release_date` | pre |
| `pub_game_count` | int | nº de juegos por `publisher` (análogo a `dev_game_count`) | pre |
| `pub_experience` | cat | tier de publisher (Novato/Establecido/AAA) | pre |
| `price_tier` | cat | F2P / budget <$10 / mid $10–20 / premium >$20 | pre |
| `is_early_access` | bool | de `genre_early_access` / `tag_early_access` | pre |
| `dev_success_prior` | float | prior bayesiano por developer, **leave-one-out** (no fuga) | pre |
| *(stretch)* `desc_*` | varios | longitud, keywords, sentimiento léxico de `short_description` | pre |

Ya existentes que se exponen como inputs: `supported_languages`, `price`, `total_achievements`, `total_dlcs`,
`min_ram_gb`, `controller_support`, `dev_experience`, plataformas, géneros, categorías, 48 tags.

**Anti-fuga (recordatorio):** el predictor pre-lanzamiento **excluye** `positive`, `negative`, `rating_porcentaje`,
`metacritic_score`, `ccu`, `owners_lower_bound` y todos los `ts_*`. Esas columnas solo viven en el track post-lanzamiento.

### 3.2 Script `build_dataset_v2.py`

- Extiende `build_dataset.py`. Mantiene los 4 CSV relacionales como entrada.
- Añade las features de §3.1 con roles en el diccionario (`use_as_feature`, `pre_launch`, `dtype`).
- Genera un **diccionario versionado** `output/dataset_ml_dictionary.csv` con la columna `etapa` (pre/post/id/label).
- Idempotente y reejecutable cuando llegue la descarga de ~2k juegos.

### 3.3 Selección / saneo (punto 1)

- Conserva `owners>0` como núcleo (~5.9k → meta 7k).
- Añade un estrato con `owners==0` pero con reviews/tags presentes, marcado con `tiene_ventas=False`, para que el
  modelo vea negativos más variados (gratuitos, demos). Se controla la proporción para no desbalancear.
- Anonimización opcional para el estudio: `developer`/`publisher` → hash estable (`dev_id`, `pub_id`).

---

## 4. Capa de modelado

### 4.1 Modelos entrenados

| Modelo | Rol | Notas |
|--------|-----|-------|
| Logistic Regression (multinomial) | base interpretable | exporta coeficientes → insights y fallback estático |
| SVM (kernel RBF, `probability=True`) | no lineal | `StandardScaler` obligatorio |
| MLP (`MLPClassifier`, ReLU) | no lineal | 2 capas ocultas (p.ej. 64→32) |
| **Owners regressor** | habilita umbrales | regresión sobre `log1p(owners)`; clasificación = umbral aplicado a la predicción |
| Empirical Bayes (Beta-Binomial por dev) | prior / feature | leave-one-out para no filtrar |
| NearestNeighbors | juegos similares | sobre el vector preprocesado |

### 4.2 Preprocesamiento (compartido)

`ColumnTransformer`: `StandardScaler` (numéricas) + `OneHotEncoder(handle_unknown="ignore")` (categóricas) +
passthrough (booleanas). Se serializa **junto** con cada modelo (un solo `Pipeline` por modelo) para que el backend
no tenga que reconstruirlo.

### 4.3 Validación y métricas

- `train_test_split` estratificado 80/20 + `GridSearchCV` (5-fold) por modelo.
- Métricas: AUC OVR-macro, F1-macro, accuracy, matriz de confusión 3×3, classification report.
- Todo se vuelca a `reports/metrics.json` y `reports/confusion_<modelo>.json` para alimentar el panel analítico.

### 4.4 Umbrales ajustables (punto 2)

El owners regressor predice `owners_estimados`. La clasificación Flop/Rentable/Hit es entonces una función de dos
cortes (`t_flop`, `t_hit`) **aplicada en el cliente**: mover los sliders reclasifica al instante sin reentrenar.
Los modelos de clasificación (LR/SVM/MLP) usan los cortes fijos por defecto; el regresor da la flexibilidad.

### 4.5 Incertidumbre (punto 7)

Tres señales, combinadas en un indicador visual:
1. **Probabilidad de la clase ganadora** (`max predict_proba`).
2. **Margen top-2** (diferencia entre 1ª y 2ª clase): margen pequeño → "el modelo duda".
3. **Desacuerdo entre modelos**: si LR/SVM/MLP no coinciden en la clase, se rotula como predicción inestable.

### 4.6 Track post-lanzamiento (punto 4)

Modelo gemelo que **sí** usa señales tempranas (`positive`, `negative`, `rating_porcentaje`, `ccu`, primeros `ts_*`).
Se entrena aparte y se expone como un toggle "ya lancé / aún no" en el Simulador. Nunca se mezcla con el pre-lanzamiento.

### 4.7 Script `train_models.py` (reproducible)

- Un único script que corre **igual en la laptop y en un notebook Databricks** (`08_train_all.py`).
- Lee el dataset (CSV local o tabla Unity Catalog), entrena todo, y guarda:
  - `models/*.joblib` (cada Pipeline) — consumidos por el backend.
  - `reports/metrics.json`, `reports/confusion_*.json`.
  - `output/model_web.json` (fallback estático del LR, compatible con v1).
- Evita la fricción de sacar artefactos del serverless Free: si se entrena en Databricks, basta correr el mismo
  script localmente para regenerar los `.joblib`.

---

## 5. Frontend (estética Steam, varias páginas)

### 5.1 Stack visual

- **Tailwind** (vendorizado, ya disponible) para layout.
- **Lucide** (vendorizado) para iconos.
- **ECharts** (a vendorizar) como motor de gráficos Power BI — offline, ~100 kB gzip, pensado para dashboards densos.
- Paleta Steam en CSS variables: fondo `#171a21`, superficie `#1b2838`, acento `#66c0f4`, secundario `#2a475e`,
  éxito `#4c6b22`, texto `#c7d5e0`.

### 5.2 Página — Simulador (input)

- Controles: precio, idiomas, achievements, DLCs, RAM, controller support, experiencia dev/publisher, mes de
  lanzamiento, early access, y matriz de tags/géneros/categorías.
- Salida: P(Flop/Rentable/Hit) del modelo seleccionado + indicador de incertidumbre (§4.5).
- Toggle "ya lancé / aún no" → cambia al track post-lanzamiento.

### 5.3 Página — Comparación de modelos

- Mismo input → tres columnas (LR / SVM / MLP) con sus probabilidades y clase ganadora.
- Resalta coincidencia/desacuerdo. Mini-tabla de métricas de cada modelo (AUC/F1) para contextualizar.

### 5.4 Página — Juegos del mismo camino

- `NearestNeighbors` devuelve los k juegos reales más parecidos al simulado.
- Tabla/cards con su resultado real (Flop/Rentable/Hit, owners) → "a juegos así les fue de esta forma".
- Modo validación: elegir un juego real del dataset y ver predicción del modelo vs. su clase verdadera (punto 13).

### 5.5 Página — Panel analítico (Power BI)

Visualizaciones con ECharts a partir de `/api/stats` y `reports/`:
- Distribución del mercado (Flop/Rentable/Hit), histograma de precios, owners por género.
- Importancia de variables (coeficientes LR / permutación).
- Matrices de confusión por modelo, curvas/medidas de desempeño.
- Posición del juego simulado dentro del mercado (scatter/quadrant).

### 5.6 Página — Análisis & Recomendaciones

- Render del paquete de recomendaciones (§6) con el delta de P(Hit) y un "antes/después".

### 5.7 Modos (punto 12)

- **Modo presentación:** flujo guiado paso a paso, ejemplos precargados, narrativa para la exposición.
- **Modo quick:** formulario compacto con defaults sensatos para llenar datos rápido ante público.

---

## 6. Motor de recomendaciones combinatorio (punto 8)

Algoritmo (en backend, usando el modelo seleccionado):

```
entrada: vector x del juego, presupuesto de cambios K
candidatos: precio (±$5 por paso), togglear tags con prevalencia 5–95%,
            +idiomas, controller support, mes de lanzamiento
búsqueda: beam search (ancho B) sobre combinaciones de hasta K cambios
objetivo: maximizar P(Hit) (o P(no-Flop)) respetando restricciones realistas
salida: mejor paquete { cambios[], P(Hit)_antes, P(Hit)_después, delta }
```

Restricciones para evitar recomendaciones absurdas: no togglear features degeneradas (p.ej. `platform_windows`),
no proponer precios fuera de rango por tier, no más de K cambios. Se reporta también el top-3 de cambios individuales
para interpretabilidad.

---

## 7. Backend (Python stdlib — `http.server`, sin dependencias)

### 7.1 Endpoints

| Método | Ruta | Entrada | Salida |
|--------|------|---------|--------|
| POST | `/api/predict` | features del juego + `modelo?` | probabilidades por modelo + incertidumbre |
| POST | `/api/recommend` | features + `K` | paquete de cambios + deltas |
| POST | `/api/similar` | features + `k` | juegos reales vecinos con su resultado |
| GET | `/api/stats` | — | agregados para los gráficos del panel |
| GET | `/api/games?q=` | query | búsqueda de juegos reales (modo validación) |
| GET | `/api/models` | — | métricas y metadatos de cada modelo |
| GET | `/` | — | sirve el frontend |

### 7.2 Contrato de `/api/predict` (borrador)

```json
// request
{ "modelo": "todos", "features": { "price": 19.99, "supported_languages": 8,
  "dev_experience": "Establecido", "release_month": 11, "tags": ["tag_rpg","tag_open_world"], ... } }

// response
{ "modelos": {
    "lr":  { "probs": {"Flop":0.18,"Rentable":0.49,"Hit":0.33}, "clase":"Rentable" },
    "svm": { "probs": {...}, "clase":"Hit" },
    "mlp": { "probs": {...}, "clase":"Rentable" } },
  "incertidumbre": { "margen_top2": 0.16, "desacuerdo": true, "nivel": "media" },
  "owners_estimados": 410000 }
```

### 7.3 Detalles

- Carga de `models/*.joblib` en el `startup` (una sola vez).
- Validación de entrada manual + defaults desde `models/feature_schema.json` (medianas/modas del entrenamiento).
- Sirve el frontend estático desde la misma app → un solo `python app/server.py` y abrir `localhost:8000`.
- Sin llamadas externas → compatible con la red Fortinet.

---

## 8. Estructura de carpetas propuesta

```
codes/
├── app/                      # backend stdlib http.server (nuevo)
│   ├── server.py             #   arranque + router de endpoints
│   ├── inference.py          #   carga de modelos, predict, recommend, similar
│   └── stats.py              #   agregados para el panel
├── web/                      # frontend v2 (nuevo)
│   ├── index.html
│   ├── css/  js/  vendor/    #   tailwind.js, lucide.js, echarts.js
├── models/                   # *.joblib (nuevo, generado)
├── reports/                  # metrics.json, confusion_*.json (nuevo, generado)
├── build_dataset_v2.py       # dataset con features nuevas (nuevo)
├── train_models.py           # entrenamiento reproducible (nuevo)
├── notebooks/08_train_all.py # mismo entrenamiento en Databricks (nuevo)
└── (intactos) steam_etl.py, build_dataset.py, export_model_web.py,
              embed_model_in_html.py, steampredict_dashboard_comercial.html  # v1
```

Nada de v1 se modifica ni se borra.

---

## 9. Qué reusamos (no reinventar la rueda)

| Recurso | Uso | Enlace |
|---------|-----|--------|
| ECharts | gráficos Power BI offline (mejor que Plotly por peso, que Chart.js por capacidad) | https://echarts.apache.org |
| Paleta oficial Steam | CSS variables | https://colorswall.com/palette/193 |
| Patrón FastAPI + sklearn | servir varios `.joblib` | https://machinelearningmastery.com/train-serve-and-deploy-a-scikit-learn-model-with-fastapi/ |
| `cosine_similarity` / `NearestNeighbors` | juegos similares | https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html |
| Ideas de features (NLP descripción, precio, longevidad) | feature engineering | https://github.com/tgrauenhorst/Predicting_Video_Game_Success · https://github.com/dgraves4/steam-indie-success |
| Artefactos v1 | fallback estático + export del LR | `export_model_web.py`, `embed_model_in_html.py` |

---

## 10. Plan por fases (checklist / bitácora)

### Fase 0 — Datos + entrenamiento reproducible  ✅ COMPLETA (2026-06-03)
- [x] `build_dataset_v2.py`: features nuevas (§3.1) + diccionario versionado con `etapa`. → 7.817 juegos, 92 features.
- [ ] Estrato `owners==0` con reviews para negativos variados (§3.3). *(pendiente — ver decisión §12.6)*
- [x] `train_models.py`: LR, SVM, MLP, owners-regressor, NN + `dev_success_prior` (Bayes LOO) → `models/*.joblib`.
- [x] `reports/metrics.json` + matrices de confusión (`reports/confusion_*.json`).
- [x] `output/model_web.json` regenerado (no se re-embebe en el HTML → v1 intacta).
- [x] `models/feature_schema.json` (defaults/categorías/umbrales para el backend).
- [x] Versión Databricks `notebooks/08_train_all.py` (entrenamiento consolidado v2 en UC; lógica validada contra el CSV local: mismas 92 features y métricas).

**Resultados test (split 80/20):** LR AUC 0.884 / F1 0.69 · SVM AUC 0.881 / F1 0.72 · **MLP AUC 0.888 / F1 0.74 / acc 0.77** · owners-reg R²(log) 0.63. Balance Flop/Rentable/Hit = 27/54/18.

### Fase 1 — Backend  ✅ COMPLETA (2026-06-03)
- [x] `app/server.py` (http.server) + `app/inference.py` + `app/stats.py`, carga de modelos al arranque.
- [x] `/api/predict` (multi-modelo + incertidumbre: margen top-2 + desacuerdo + owners estimados).
- [x] `/api/recommend` (beam search, K configurable, paquete + cambios individuales).
- [x] `/api/similar` (NearestNeighbors coseno con metadatos del juego real).
- [x] `/api/stats` + `/api/games?q=` + `/api/game/<appid>` + `/api/config`.
- [x] Validación manual + defaults desde `feature_schema.json`. Sirve estáticos de `web/` y `/vendor/`.
- [x] Verificado por HTTP (arranque, GET y POST). Búsqueda OK (`portal` → Portal/Portal 2/Portal Knights).

### Fase 2 — Frontend base (estética Steam)  ✅ COMPLETA (2026-06-03, falta QA visual)
- [x] Layout SPA + paleta Steam (`web/index.html`, `web/css/app.css`) + ECharts vendorizado.
- [x] Página Simulador con incertidumbre + selector de modelo + KPIs.
- [x] Página Comparación de modelos (barras agrupadas + tabla con AUC/F1/acc).

### Fase 3 — Páginas ricas  ✅ COMPLETA (2026-06-03, falta QA visual)
- [x] Página Juegos del mismo camino + modo validación (cargar juego real → predicción vs realidad).
- [x] Página Análisis & Recomendaciones (paquete beam search + cambios individuales).
- [x] Página Panel analítico: donut de mercado, histograma de precios, owners por género, importancia LR, scatter precio/owners, matriz de confusión, tabla de desempeño.

### Fase 4 — Modos + pulido
- [ ] Modo presentación (guiado, ejemplos precargados).
- [ ] Modo quick.
- [ ] Fallback estático verificado.

### Fase 5 — Documentación para el estudio  🟡 EN CURSO
- [x] Reporte de métricas, matrices, metodología → `Documentacion del Estudio - Predictor v2.md` (humanizado).
- [x] Ángulo ingeniería de datos: anonimización, datos de panel (serie mensual), Spark/Databricks.
- [ ] Actualizar `README.md`.
- [x] Commits bajo SamDev (`198a233`, `c1e7554`, `1f341b1`, `2f2ebb4`). Push pendiente de tu visto bueno.

### Pendiente cuando termine el ETL
- [ ] Reejecutar `build_dataset_v2.py` con los ~2k nuevos juegos.
- [ ] Reentrenar (`train_models.py`) y regenerar artefactos.

---

## 11. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| ETL no termina a tiempo (Fortinet/red) | Construir todo contra el dataset actual (5.9k); *swap* es solo reejecutar 2 scripts. |
| SVM/MLP no exportables a JS | Backend local (decidido). Fallback estático LR si el backend falla en la presentación. |
| Sacar `.joblib` del serverless Free | `train_models.py` corre local → no se depende de descargar de Databricks. |
| Umbrales requieren reentrenar | Owners-regressor → reclasificación en cliente. |
| NLP consume el finde | Marcado como *stretch*; se omite si peligra el deadline. |
| Recomendaciones absurdas | Filtro de prevalencia 5–95% + restricciones de rango. |
| Sobreajuste por desbalance | `class_weight="balanced"` + métricas macro + CV. |

---

## 12. Decisiones (confirmadas 2026-06-03)

1. **Entrenamiento:** ✅ `train_models.py` único (corre en laptop y en Databricks). Hecho.
2. **NLP de descripción:** ✅ *stretch* — se intenta en el finde sin arriesgar el deadline.
3. **Post-lanzamiento:** ✅ toggle "ya lancé / aún no" dentro del Simulador (no página propia).
4. **Carpetas:** ✅ `app/` + `web/` + `models/` + `reports/`, bien estructurado, v1 intacta.
5. **Backend:** ✅ Python stdlib `http.server` (no FastAPI) — por Python 3.14 (wheels de pydantic-core inciertos) + reproducibilidad cero-dependencias para la presentación.
6. **Estrato `owners==0`:** ⏳ pendiente — evaluar tras ver el balance (Hit cayó a 18% con los datos nuevos). Riesgo: etiquetar como Flop juegos que solo carecen de estimación de SteamSpy. Se decide antes de un reentrenamiento final.

---

## 13. Bitácora de avance

> Registro cronológico. Se añade una línea por sesión/hito. Formato: `fecha — qué se hizo — artefacto`.

- 2026-06-03 — Inventario del repo v1 + investigación de viabilidad (ECharts, FastAPI, repos reusables). Plan v2 redactado. — `Plan v2 - Dashboard Comercial Customizable.md`
- 2026-06-03 — Git: `git pull` de `971ee1b` (+1961 juegos exitosos). Conteos reales: 32.966 totales / **7.824 con owners>0**. — `output/*.csv`
- 2026-06-03 — Decisiones confirmadas (§12): backend stdlib, post-lanzamiento como toggle, carpetas `app/web/models/reports`, `train_models.py` único. ECharts vendorizado (`vendor/echarts.min.js`, 1007 KB).
- 2026-06-03 — Apartado de ingeniería de datos creado y verificado: anonimización SHA-256 (developer/publisher) + análisis Spark RDD para Databricks + verificación local con pandas. Archivos de referencia del curso movidos a `referencia/`. — `ingenieria_datos/`
- 2026-06-03 — **Fase 0 COMPLETA**: `build_dataset_v2.py` (7.817 juegos, 92 features, 9 nuevas, 0 nulos) + `train_models.py` (LR/SVM/MLP/owners-reg/NN + Bayes LOO). Test: MLP AUC 0.888 / F1 0.74 / acc 0.77. Artefactos cargan y predicen OK. — `models/`, `reports/`
- 2026-06-03 — **Fase 1 COMPLETA**: backend stdlib (`app/server.py` + `inference.py` + `stats.py`). Endpoints predict/recommend/similar/stats/games/config verificados por HTTP. Recomendador beam search sube P(Hit) de 0.19→0.65 en una prueba RPG; similares devuelve juegos coherentes. — `app/`
- 2026-06-03 — Commit `198a233` (28 archivos) bajo SamDev: Fase 0 + Fase 1 + apartado ingeniería de datos.
- 2026-06-03 — **Fases 2-3 COMPLETAS** (código): frontend SPA estética Steam con las 5 pestañas (Simulador, Comparar modelos, Juegos similares, Recomendaciones, Panel analítico) + modos Presentación/Quick + búsqueda y carga de juegos reales (modo validación). Estáticos servidos OK por el backend. **Pendiente: QA visual del usuario.** — `web/`
- 2026-06-03 — Frontend validado por el usuario ("bastante bien"). Fix Tailwind v4 + terminal de actividad + transiciones + extras (KPI ingreso, gráfico por trimestre) commiteado en `1f341b1`.
- 2026-06-03 — `notebooks/08_train_all.py`: entrenamiento consolidado v2 en Databricks (LR+GridSearchCV, SVM-RBF, MLP, owners-reg) que guarda `model_metrics_v2` y `model_lr_coefficients_v2` en Unity Catalog. Lógica validada localmente (mismas 92 features y métricas que `train_models.py`). — `notebooks/`
- 2026-06-03 — Documentación del estudio (`Documentacion del Estudio - Predictor v2.md`), README v2, y push de todos los commits a GitHub (bajo SamDev).
- 2026-06-03 — **Explicabilidad por predicción**: `/api/predict` devuelve los factores que más mueven P(Hit) (contribución por perturbación, modelo-agnóstica); tarjeta nueva en el Simulador.
- 2026-06-03 — **Modelo post-lanzamiento**: 3 modelos (`post_*.joblib`) que añaden señales de recepción (ccu, rating, metacritic, playtime). MLP 0.888 → **0.907 AUC**. Toggle "ya lancé / aún no" en el Simulador. Doc del estudio §5.5 actualizada.
- 2026-06-03 — **Rediseño del frontend integrado** (commit `2e8de58`): nuevo `web/index.html` (estética Steam, terminal flotante estilo macOS, fuentes, animaciones) cableado al backend real (predict/recommend/similar/stats/games/game). Verificado headless con Playwright: las 5 vistas renderizan con datos reales, 0 errores JS. SPA anterior (`web/js`, `web/css`) eliminado.
- 2026-06-03 — Notebooks **01-07 alineados a v2**: el 01 añade las categóricas nuevas (pub_experience/price_tier/release_quarter) y excluye las temporales → produce 12 num / 75 bool / 5 cat = 92 features (validado localmente); 02-07 las toman por ser schema-driven.
- _(nota: el nuevo diseño aún no incluye la tarjeta de explicabilidad ni el toggle post-lanzamiento; quedan para integrar en el rediseño si se desea)_
- 2026-06-03 — **Experimento modelo en 2 etapas** (`train_stage1.py`): etapa de tracción (owners>0 sobre 33k) AUC 0.96, pero **confundida por artefactos de recolección** (platform_windows coef ≈ −9.3, fechas faltantes; SteamSpy vs catálogo poblaron metadata distinto). Descartado del producto (colapsaba a ~2% tracción). Documentado como hallazgo metodológico en el informe del estudio (§5.6). Backend revertido a una sola etapa.

---

## 14. Documentación final para el estudio (entregable §5)

Al cerrar, se redacta (con `/humanizer` para el texto formal):
- Metodología: selección de datos, features, anti-fuga, validación, métricas.
- Comparación de algoritmos (LR/SVM/MLP) con tablas y matrices.
- Ingeniería de datos: anonimización de developer/publisher, datos de panel (serie mensual juego×mes), rol de Spark/Databricks.
- Limitaciones y trabajo futuro (NLP, series de tiempo, más juegos).
