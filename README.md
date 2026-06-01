# Predictor de Videojuegos - Pipeline de Extracción (Steam ETL)

Este proyecto consiste en un robusto pipeline de extracción, transformación y carga (ETL) desarrollado en Python para recopilar datos históricos de videojuegos de la plataforma Steam (a través de la **Steam Storefront API**, **SteamSpy API** y **Steam Reviews API**). 

El objetivo del dataset generado es alimentar modelos de Machine Learning (como Regresión Logística, SVM, MLP y Modelos Bayesianos Jerárquicos) para analizar y predecir el éxito comercial de un videojuego basado en sus características teóricas y de diseño.

---

## 🚀 Guía de Clonación y Configuración del Entorno (Paso a Paso)

Sigue estos pasos para replicar exactamente el entorno y ejecutar los scripts.

### 1. Clonar el Repositorio
Abre tu terminal y ejecuta el siguiente comando para clonar este repositorio en tu máquina local:
```bash
git clone https://github.com/SamCCDev/PI_steam.git
cd PI_steam
```

### 2. Crear y Activar el Entorno Virtual
Se recomienda utilizar un entorno virtual de Python (`venv`) para aislar las dependencias del proyecto.

* **En Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **En Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```

* **En Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### 3. Instalar las Dependencias
Una vez activado el entorno virtual, instala las dependencias requeridas ejecutando:
```bash
pip install -r requirements.txt
```
*(Nota: El script principal `steam_etl.py` también cuenta con auto-instalación dinámica de dependencias si detecta que faltan al ejecutarse).*

### 4. Configurar Variables de Entorno (Clave de API de Steam)
Para realizar consultas ilimitadas y utilizar las funciones avanzadas del pipeline, debes configurar tu propia clave de API de Steam.
1. Copia el archivo de plantilla `.env.example` y renombralo como `.env`:
   ```bash
   cp .env.example .env
   ```
2. Abre el archivo `.env` recién creado en un editor de texto y asigna tu API Key:
   ```text
   STEAM_API_KEY=TuClaveDeApiAqui
   ```
*(Nota: El archivo `.env` está configurado en `.gitignore` para que nunca se suba al repositorio y permanezca seguro en tu máquina).*

---

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```text
├── steam_etl.py                           # Script principal del pipeline ETL
├── build_dataset.py                       # Consolida los CSV relacionales en un master ML-ready
├── merge_and_update.py                    # Script de consolidación y fusión de datos históricos
├── complete_timeseries.py                 # Script para completar la serie temporal de registros nulos
├── requirements.txt                       # Archivo de dependencias del entorno
├── .env.example                           # Plantilla de configuración de variables de entorno
├── .gitignore                             # Reglas para excluir archivos locales y temporales
├── Documento Tecnico - Predictor...md     # Documentación técnica, metodológica y diccionario de datos
├── Plan de Accion - Predictor...md        # Plan de acción por fases (MVP + limitaciones técnicas)
├── notebooks/                             # Notebooks de Databricks (entrenamiento PySpark ML)
│   ├── 01_data_prep.py                    # Carga, limpieza, etiqueta y features (schema-driven)
│   ├── 02_logistic_regression.py          # Baseline interpretable + coeficientes
│   ├── 03_evaluation.py                   # AUC, F1, CrossValidation, MLflow
│   └── 04_dashboard.py                    # Simulador comercial + motor de recomendaciones
└── output/                                # Directorio de datasets resultantes (delimitados por ';')
    ├── games_metadata.csv                 # 1:1 Metadata de videojuegos (precios, ventas estimadas, CCU)
    ├── games_tags.csv                     # One-Hot Encoding de géneros, categorías y etiquetas
    ├── games_text.csv                     # Textos limpios para análisis de NLP (descripción)
    └── games_timeseries.csv               # Registros mensuales de tracción y retención de usuarios
```

---

## 🛠️ Instrucciones de Uso de los Scripts

### 1. Ingesta y Extracción de Datos (`steam_etl.py`)
Este es el motor de extracción concurrente. Está diseñado con multihilo cooperativo (5 hilos) y limitadores de tasa coordinados para evitar bloqueos por rate-limit (errores 429).

* **Extracción Estándar (1,000 juegos aleatorios):**
  ```bash
  python steam_etl.py
  ```

* **Extracción de Prueba (Muestra pequeña de ejemplo):**
  ```bash
  python steam_etl.py --sample 10
  ```

* **Diagnóstico de un Juego Específico (ej. CS2 - appid 730):**
  Solo imprime la información parseada en consola sin escribir nada a los archivos del dataset.
  ```bash
  python steam_etl.py --validate 730
  ```

### 2. Consolidación de Históricos (`merge_and_update.py`)
Si cuentas con backups previos o particiones descargadas en subcarpetas de respaldo, este script fusiona los datasets de forma inteligente (removiendo duplicados y priorizando los datos más recientes) y descarga dueños mínimos ausentes desde SteamSpy.
```bash
python merge_and_update.py
```

### 3. Completado de Series Temporales (`complete_timeseries.py`)
Garantiza que todos los juegos presentes en la metadata principal tengan representación en el dataset de series temporales (`games_timeseries.csv`). Aquellos que no tienen reseñas registradas o son pre-lanzamientos reciben un registro base inicial por defecto con valores en cero para facilitar la ingesta en modelos secuenciales (RNN / LSTM).
```bash
python complete_timeseries.py
```

---

## 📊 Resumen del Diccionario de Datos (`output/`)

* **`games_metadata.csv`**: Clave principal `appid`. Registra variables comerciales críticas como `price`, `owners_lower_bound` (variable objetivo de ventas), `ccu` (jugadores concurrentes), `rating_porcentaje` y proxy de wishlists (`hub_followers`).
* **`games_tags.csv`**: Matriz booleana (1/0) de géneros (Action, RPG...), categorías (Multi-player, Steam Cloud...) y las 50 etiquetas de usuario más relevantes para Machine Learning.
* **`games_text.csv`**: Textos planos de descripción corta y detallada listos para pipelines de NLP.
* **`games_timeseries.csv`**: Historial mensual desde Enero de 2024 que captura evolución de reseñas (`review_count`), sentimiento, volumen de compra directa en Steam, y retención (`avg_playtime_at_review_hrs`).

---

## 🧩 Dataset Consolidado ML-Ready (`build_dataset.py`)

Los 4 CSV son **relacionales** (1:N). Para modelar se necesita una sola tabla 1:1 por juego. `build_dataset.py` une todo y produce el master listo para entrenar:

```bash
python build_dataset.py
```

**Genera:**

* **`output/dataset_ml.csv`**: una fila por juego (`appid`), une metadata + tags + agregados de series de tiempo, deriva `dev_experience` y `dev_game_count`, filtra `owners_lower_bound > 0`, etiqueta `label = owners > 20.000` y limpia tipos (sin nulos en features).
* **`output/dataset_ml_dictionary.csv`**: diccionario de cada columna con su `role` y si es usable como feature.

**Roles de columna (clave para evitar fuga de datos):**

| Rol | Uso | Ejemplos |
| :--- | :--- | :--- |
| `feature_numeric` / `feature_tag` / `feature_categorical` / `feature_derived` | **Entradas del modelo PRE-lanzamiento** | `price`, `tag_*`, `dev_experience`, `short_desc_len` |
| `target_label` / `target_raw` | Variable objetivo | `label`, `owners_lower_bound`, `ccu` |
| `outcome_postlaunch` | **NO usar como feature** (se conoce tras lanzar) | `positive`, `negative`, `rating_porcentaje` |
| `timeseries_postlaunch` | **NO usar como feature** (post-lanzamiento) | `ts_total_reviews`, `ts_avg_playtime_hrs` |

> Re-ejecutar tras actualizar `output/*.csv` regenera el master automáticamente (schema-driven: detecta tags/géneros por prefijo). `hub_followers` y otras columnas casi-constantes se marcan y excluyen solas.

Para Databricks puedes subir directamente `dataset_ml.csv` (1 archivo) en lugar de los 4 CSV relacionales.

---

## 🧠 Entrenamiento en Databricks Free Edition

El entrenamiento se realiza en notebooks de Databricks usando PySpark ML. El plan completo, incluyendo limitaciones técnicas y orden sugerido por días, está en [`Plan de Accion - Predictor de Videojuegos.md`](Plan%20de%20Accion%20-%20Predictor%20de%20Videojuegos.md).

### Pasos de configuración (una sola vez)

> **Importante:** Databricks Free Edition usa **Unity Catalog** y tiene el **DBFS público (`/FileStore`) deshabilitado**. Por eso NO se suben los CSV a una ruta de archivo: se cargan como **tabla gestionada** y los notebooks persisten todo como tablas de Unity Catalog (sin rutas DBFS).

1. **Crear cuenta** en [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. Generar el master local: `python build_dataset.py` → produce `output/dataset_ml.csv`.
3. **Subir el dataset como tabla:** en la UI, `Catalog → (tu schema, p.ej. workspace.default) → Create → Table → Upload file`. Sube `dataset_ml.csv`, separador `;`, y nómbrala **`dataset_ml`**.
4. **Importar los notebooks** vía Git folder (Repos) o subiéndolos a tu workspace.
5. Si tu catálogo/schema no son `workspace`/`default`, ajusta las constantes `CATALOG` y `SCHEMA` en la primera celda de cada notebook.

### Estructura de notebooks (MVP funcional → pulido)

| # | Notebook | Propósito |
| :--- | :--- | :--- |
| 01 | `01_data_prep` | Lee la tabla `dataset_ml`, selecciona features (schema-driven), guarda tablas UC `features_silver` y `feature_columns`. |
| 02 | `02_logistic_regression` | Baseline (`owners_lower_bound > 20.000`) + coeficientes (odds ratio) en tabla UC. |
| 03 | `03_evaluation` | AUC, F1, matriz de confusión, CrossValidator (3 folds), tracking MLflow. |
| 04 | `04_dashboard` | Simulador con `dbutils.widgets` + motor de recomendaciones (reentrena en memoria). |
| 05+ | `05_svm`, `06_mlp`, `07_bayesian` | Fase de pulido posterior al MVP. |

> El modelo **no se persiste como archivo** (eso requeriría DBFS/Volume): los datos viajan entre notebooks como tablas UC y el dashboard reentrena la Regresión Logística en segundos.

### Criterio de MVP terminado

Los notebooks 01-04 ejecutan end-to-end sin errores y producen **AUC > 0.65** en el conjunto de test. La precisión se mejora en la fase de pulido.

### Limitaciones a tener en cuenta

* **Serverless (Spark Connect) bloquea la MLlib clásica de PySpark.** `StringIndexer`, `LogisticRegression` y demás estimadores de `pyspark.ml` lanzan `Py4JSecurityException: not whitelisted`. Por eso el modelado se hace con **scikit-learn** sobre los datos en pandas (`spark.table(...).toPandas()`), válido por el tamaño del dataset (~3k filas) y previsto en el Doc. Técnico §4.3.
* **DBFS público deshabilitado** en Free Edition → datos vía tablas Unity Catalog (`saveAsTable` / `spark.table`), no rutas `/FileStore`.
* **Sin `.cache()`/persist** en serverless (`PERSIST TABLE is not supported`).
* **Sin Jobs Scheduler** en Free Edition — el ETL se mantiene local.
* Para la fase de pulido: SVM RBF y MLP se harán con **scikit-learn** (`SVC(kernel="rbf")`, `MLPClassifier` con ReLU/Tanh) — más capaz que la MLlib para este tamaño y sin las restricciones de serverless.
