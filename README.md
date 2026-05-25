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
├── merge_and_update.py                    # Script de consolidación y fusión de datos históricos
├── complete_timeseries.py                 # Script para completar la serie temporal de registros nulos
├── requirements.txt                       # Archivo de dependencias del entorno
├── .env.example                           # Plantilla de configuración de variables de entorno
├── .gitignore                             # Reglas para excluir archivos locales y temporales
├── Documento Tecnico - Predictor...md     # Documentación técnica, metodológica y diccionario de datos
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
