# Análisis predictivo del éxito comercial de videojuegos en Steam

**Documento técnico y de arquitectura de datos.**

Documentación completa del proyecto: qué problema resuelve, de dónde salen los datos, cómo se
extraen y se preparan, qué variables entran al modelo y por qué, cómo se entrenan y comparan los
modelos, qué resultados se obtuvieron y cómo se reproduce todo. Las cifras provienen de la
ejecución de los scripts del repositorio (`scripts/steam_etl.py`, `scripts/build_dataset.py`,
`scripts/train_models.py` y `scripts/train_stage1.py`) y de los artefactos que generan en
`models/` y `reports/`.

---

## 1. Objetivo principal

El objetivo central del proyecto es construir un modelo predictivo basado en aprendizaje
automático capaz de identificar, analizar y ponderar las variables que determinan el éxito
comercial de un videojuego en el mercado de PC (plataforma Steam).

El sistema deja de lado la intuición empírica para basarse en datos históricos, y funciona como un
"filtro estratégico comercial" que permite a desarrolladores, estudios e inversores evaluar la
viabilidad de un proyecto durante la fase de conceptualización o pre-lanzamiento, cuando todavía no
hay ventas que observar y casi todas las decisiones de diseño, precio y marketing siguen abiertas.

## 2. Justificación y relevancia comercial

La industria del desarrollo de videojuegos se caracteriza por su alto riesgo de inversión y la
volatilidad de las preferencias del consumidor. Un fracaso comercial puede llevar a la quiebra a un
estudio entero. El proyecto busca tres cosas:

1. **Reducir la incertidumbre del mercado**, cimentando las decisiones de diseño y marketing en
   datos históricos en lugar de corazonadas.
2. **Mitigar riesgos y mejorar el retorno de la inversión**, permitiendo asignar presupuestos con
   más criterio y ajustar mecánicas, precio o estrategia antes de incurrir en gastos irreversibles.
3. **Ofrecer una herramienta consultiva**: un simulador donde el creador introduce los parámetros
   de su juego y obtiene una predicción de éxito, una estimación de propietarios y un paquete de
   recomendaciones.

El resultado visible no es el código del modelo, sino un producto interactivo: un dashboard que
recibe las especificaciones teóricas de un juego (precio, experiencia del estudio, géneros,
mecánicas) y devuelve la probabilidad de cada categoría comercial, junto con recomendaciones y una
comparación con juegos reales de perfil parecido.

### 2.1 Definición de la variable objetivo

Las categorías comerciales se definen por el número de propietarios que estima SteamSpy
(`owners_lower_bound`):

| Categoría | Propietarios (owners) | Interpretación |
|-----------|-----------------------|----------------|
| Flop | menos de 200.000 | ventas comercialmente bajas |
| Rentable | 200.000 a 1.000.000 | recupera costos y deja margen |
| Hit | 1.000.000 o más | éxito comercial claro |

Los dos cortes (200.000 y 1.000.000) son configurables en el dashboard gracias a la regresión de
propietarios (ver 5.3).

---

## 3. Origen y extracción de los datos

### 3.1 Fuentes

Los datos provienen de dos servicios públicos:

- **Steam Storefront API**: metadatos del juego (nombre, desarrollador, distribuidora, fecha,
  precio, idiomas, logros, DLC, plataformas, requisitos, géneros y categorías).
- **SteamSpy API**: estimación de ventas (owners), jugadores concurrentes y etiquetas de usuario.

A ellas se suma el endpoint público de reseñas de Steam, del que se deriva la serie mensual de
actividad.

### 3.2 Cómo funciona el ETL (`steam_etl.py`)

El script de extracción descarga la información en cuatro CSV relacionales unidos por la clave
`appid`. Está pensado para sostener miles de juegos sin que Steam bloquee la IP, así que incorpora
varias capas de optimización de red:

- **Concurrencia controlada.** Un `ThreadPoolExecutor` de 5 hilos procesa juegos en paralelo y,
  dentro de cada juego, un mini-pool de 2 hilos extrae SteamSpy y reseñas a la vez.
- **Sesiones HTTP persistentes.** Cada hilo mantiene su propia `requests.Session` con pool de
  conexiones, de modo que reutiliza el canal TCP/TLS y ahorra el coste de reabrirlo en cada llamada.
- **Limitadores de tasa por API.** Cada servicio tiene su propio ritmo calibrado con margen de
  seguridad: Steam Storefront a ~0,63 req/s (el límite real ronda 200 peticiones cada 5 minutos),
  SteamSpy a 1 req/s y reseñas a 5 req/s.
- **Caché masiva de SteamSpy.** Al inicio se descarga el catálogo completo de SteamSpy en memoria,
  lo que evita una llamada individual por juego.
- **Auto-pausa ante el error 429.** Si Steam responde "demasiadas peticiones", todos los hilos se
  detienen 60 segundos para no arriesgar un bloqueo prolongado.

El script ofrece tres modos de uso:

```bash
python scripts/steam_etl.py                 # extracción estándar
python scripts/steam_etl.py --sample 2000   # procesa solo N juegos (pruebas)
python scripts/steam_etl.py --validate 730  # inspecciona un appid sin escribir archivos
```

El modo `--quality-filter` resuelve el principal problema del catálogo aleatorio de Steam: cerca
del 90 % de los appids corresponden a juegos sin owners registrados, lo que produce un dataset
mayoritariamente vacío. Con el filtro, primero se descarga el catálogo de SteamSpy (que ya trae
owners y reseñas) y solo se consultan al Storefront los juegos que superan unos umbrales mínimos de
propietarios y reseñas, configurables con `--min-owners` y `--min-reviews`.

### 3.3 Estructura relacional

| Archivo | Cardinalidad | Contenido |
|---------|--------------|-----------|
| `games_metadata.csv` | 1 fila por juego | precios, ventas estimadas, desarrollador, plataformas, logros, DLC, RAM, idiomas |
| `games_tags.csv` | 1 fila por juego | matriz booleana de géneros, categorías de Steam y etiquetas de usuario |
| `games_text.csv` | 1 fila por juego | descripción corta y completa (de aquí sale la longitud de texto) |
| `games_timeseries.csv` | N filas por juego | serie mensual de actividad de reseñas desde enero de 2024 |

`build_dataset.py` consolida los cuatro archivos en un único maestro con una fila por juego. Es
schema-driven (detecta géneros y etiquetas por su prefijo de columna), idempotente (re-ejecutarlo
tras actualizar los CSV regenera todo) y, además del maestro, produce un diccionario de datos
versionado con el rol y la etapa de cada columna.

### 3.4 Selección de juegos

Del catálogo descargado (32.966 juegos) se conservan **7.817** que tienen estimación de ventas en
SteamSpy. Los demás no necesariamente venden poco: SteamSpy simplemente no publica estimación para
ellos, de modo que su número de propietarios figura como cero. Etiquetarlos como Flop introduciría
ruido, así que quedan fuera del entrenamiento del clasificador de tres clases. La distribución
resultante de la variable objetivo es **27,4 % Flop, 54,2 % Rentable y 18,4 % Hit**.

---

## 4. Variables del modelo

El maestro contiene 124 columnas. De ellas, **92 se usan como variables predictoras**: 12
numéricas, 5 categóricas y 75 binarias. Las 32 restantes son identificadores, la propia etiqueta o
columnas excluidas a propósito (ver 4.2).

### 4.1 Por qué estas variables y no otras

La selección sigue cuatro criterios:

- **Solo información pre-lanzamiento.** El simulador predice antes de publicar, así que únicamente
  entran variables que un estudio conoce mientras todavía diseña el juego: precio, idiomas,
  géneros, mecánicas, experiencia del estudio. Todo lo que solo existe después del lanzamiento se
  excluye (ver 4.2).
- **Señal sobre ruido.** Durante la construcción del dataset se descartan columnas casi constantes,
  porque una variable que vale lo mismo en casi todos los juegos no aporta capacidad de
  discriminación y solo añade dimensiones. Por eso quedaron fuera, por ejemplo, `tag_pixel_art`,
  `tag_roguelike`, `tag_permadeath`, `tag_soulslike` o `cat_local_co_op`: aparecían en muy pocos
  juegos del conjunto.
- **Variables que el negocio entiende.** Se priorizaron factores sobre los que un desarrollador
  puede decidir (precio, número de idiomas, soporte de control, mecánicas), porque la herramienta
  es consultiva: no sirve de nada recomendar algo que no se puede cambiar.
- **Sin proxies que resultaron vacíos.** El plan inicial contemplaba usar los seguidores del hub de
  la comunidad (`hub_followers`) como sustituto de las listas de deseos. En la práctica el campo
  quedó mal mapeado en el ETL, con varianza casi nula, así que se descartó. Las integraciones
  externas que se barajaron al principio (Kickstarter, HowLongToBeat) no se implementaron y no
  forman parte del modelo.

### 4.2 Prevención de fuga de datos

Como el simulador es pre-lanzamiento, se excluyen todas las variables que solo se conocen una vez
que el juego salió a la venta. Permanecen en el maestro con su etapa marcada como `post` en el
diccionario, pero ningún modelo las consume:

| Variable | Por qué se excluye |
|----------|--------------------|
| `positive`, `negative` | reseñas acumuladas tras el lanzamiento |
| `rating_porcentaje` | porcentaje de valoración positiva (post-venta) |
| `metacritic_score` | nota de prensa publicada tras salir |
| `ccu` | jugadores concurrentes (solo existe con el juego en el mercado) |
| `ts_*` (12 columnas) | agregados mensuales de la serie de reseñas |

### 4.3 Variables numéricas (12)

| Variable | Origen | Descripción | Mediana / rango |
|----------|--------|-------------|-----------------|
| `price` | Steam | Precio de venta en USD (0 si es gratuito) | 4,99 · 0–149,99 |
| `supported_languages` | Steam | Cantidad de idiomas soportados | 6 · 0–103 |
| `total_achievements` | Steam | Logros desbloqueables | 22 · 0–5000 |
| `total_dlcs` | Steam | Expansiones de pago | 0 · 0–1071 |
| `min_ram_gb` | Steam (regex) | RAM mínima extraída del texto de requisitos | 2 · 0–512 |
| `dev_game_count` | derivado | Juegos previos del estudio | 1 · 1–29 |
| `pub_game_count` | derivado | Juegos previos de la distribuidora | 1 · 1–57 |
| `short_desc_len` | Steam (texto) | Longitud de la descripción de tienda | 230 · 0–342 |
| `num_genres` | derivado | Géneros declarados | 3 · 0–10 |
| `num_tags` | derivado | Etiquetas de usuario consideradas activas | 5 · 0–14 |
| `num_categories` | derivado | Categorías de Steam declaradas | 4 · 0–11 |
| `dev_success_prior` | derivado | Prior bayesiano del estudio (leave-one-out) | 0,73 · 0,48–0,92 |

### 4.4 Variables categóricas (5)

| Variable | Origen | Categorías |
|----------|--------|-----------|
| `dev_experience` | derivado | Novato / Establecido / AAA (según juegos previos del estudio) |
| `pub_experience` | derivado | Novato / Establecido / AAA (según juegos previos de la distribuidora) |
| `controller_support` | Steam | full / none |
| `price_tier` | derivado | F2P / Budget / Mid / Premium |
| `release_quarter` | derivado | Q1 / Q2 / Q3 / Q4 (o Desconocido) |

### 4.5 Variables binarias (75)

**Plataformas y estado (5):** `platform_windows`, `platform_mac`, `platform_linux`, `is_free`,
`is_early_access`.

**Géneros (13):** acción, aventura, casual, indie, multijugador masivo, RPG, carreras, simulación,
deportes, estrategia, violento, gore y acceso anticipado (`genre_action`, `genre_adventure`,
`genre_casual`, `genre_indie`, `genre_massively_multiplayer`, `genre_rpg`, `genre_racing`,
`genre_simulation`, `genre_sports`, `genre_strategy`, `genre_violent`, `genre_gore`,
`genre_early_access`).

**Categorías de Steam (13):** un jugador, multijugador, cooperativo, cooperativo en línea, soporte
total y parcial de mando, logros, nube, workshop, compras dentro de la app, cartas de intercambio,
juego remoto conjunto y pantalla dividida (`cat_single_player`, `cat_multi_player`, `cat_co_op`,
`cat_online_co_op`, `cat_full_controller_support`, `cat_partial_controller_support`,
`cat_steam_achievements`, `cat_steam_cloud`, `cat_steam_workshop`, `cat_in_app_purchases`,
`cat_steam_trading_cards`, `cat_remote_play_together`, `cat_shared_split_screen_co_op`).

**Etiquetas de usuario (44):** capturan el *core loop* y el estilo del juego tal como lo clasifican
los jugadores. Incluyen mecánicas (`tag_open_world`, `tag_crafting`, `tag_survival`,
`tag_turn_based`, `tag_sandbox`, `tag_management`, `tag_stealth`), perspectivas y modos
(`tag_first_person`, `tag_fps`, `tag_co_op`, `tag_multiplayer`, `tag_pvp`, `tag_battle_royale`,
`tag_local_co_op`, `tag_moba`), géneros finos (`tag_rpg`, `tag_action_rpg`, `tag_strategy`,
`tag_platformer`, `tag_metroidvania`, `tag_tower_defense`, `tag_city_builder`, `tag_simulation`,
`tag_sports`, `tag_racing`, `tag_puzzle`, `tag_point_&_click`, `tag_visual_novel`), tono y estilo
visual (`tag_horror`, `tag_story_rich`, `tag_dark`, `tag_atmospheric`, `tag_relaxing`,
`tag_difficult`, `tag_anime`, `tag_2d`, `tag_3d`, `tag_realistic`, `tag_cartoon`) y etiquetas de
mercado (`tag_indie`, `tag_casual`, `tag_early_access`, `tag_free_to_play`, `tag_vr`).

### 4.6 Ingeniería de variables

Sobre las columnas directas de la API se derivaron las que el modelo necesita y la API no entrega:

- **Conteos de catálogo** (`num_genres`, `num_tags`, `num_categories`): miden cuán definido está el
  juego, partiendo de las matrices booleanas.
- **Tamaño y experiencia del estudio y la distribuidora** (`dev_game_count`, `pub_game_count`,
  `dev_experience`, `pub_experience`): contando cuántos appids previos tiene cada nombre.
- **Trimestre de lanzamiento** (`release_quarter`): para capturar la estacionalidad comercial.
- **Tramo de precio** (`price_tier`) y **acceso anticipado** (`is_early_access`).
- **Prior bayesiano del estudio** (`dev_success_prior`): estima la probabilidad de que un juego del
  mismo estudio sea Rentable o Hit, con un esquema Beta-Binomial que contrae hacia la media global
  del mercado (0,73). Se calcula dejando fuera el propio juego (leave-one-out), de modo que no
  filtra su propia etiqueta; para un estudio sin historial, el valor cae a la media global.

---

## 5. Modelado

### 5.1 Cómo funciona el entrenamiento

La lógica de entrenamiento vive en un único archivo, `scripts/train_models.py`, que es la **única
fuente de verdad**. El mismo archivo detecta dónde corre:

- **En local** lee `output/dataset_ml.csv`, entrena y guarda los modelos serializados en `models/`,
  las métricas en `reports/` y los coeficientes del modelo lineal en `output/model_web.json`.
- **En Databricks** lee la tabla `dataset_ml` de Unity Catalog y guarda las métricas y los
  coeficientes como tablas gestionadas.

El preprocesamiento es común a los tres clasificadores y se serializa junto a cada modelo en un solo
`Pipeline`, de modo que el backend no tiene que reconstruirlo: las numéricas se estandarizan (media
0, desviación 1), las categóricas se codifican one-hot y las binarias pasan directo. Los tres usan
`class_weight="balanced"` para compensar el desbalance de clases.

Se entrenan tres clasificadores y tres modelos auxiliares:

1. **Regresión logística multinomial (softmax).** El modelo base e interpretable; sus coeficientes
   indican qué factores empujan hacia cada categoría y alimentan tanto el gráfico de importancia de
   variables como el motor de recomendaciones.
2. **Máquina de vectores de soporte con kernel RBF.** Captura fronteras no lineales en el espacio de
   alta dimensión que produce el one-hot de las etiquetas.
3. **Perceptrón multicapa.** Red densa con dos capas ocultas (64 y 32 neuronas) y activación ReLU.
   Modela interacciones complejas entre variables y resultó el de mejor desempeño.

Auxiliares:

- **Regresión de propietarios** (`HistGradientBoostingRegressor` sobre el logaritmo de los owners).
  No clasifica: predice una cantidad. Aplicar un corte sobre esa predicción permite reclasificar
  Flop/Rentable/Hit con umbrales movibles sin reentrenar, que es como el dashboard ofrece umbrales
  ajustables.
- **Vecinos más cercanos** (NearestNeighbors, distancia coseno, k=12) sobre el vector preprocesado,
  que alimenta la vista de juegos reales parecidos.
- **Prior bayesiano por estudio**, descrito en 4.6.

La validación usa una partición estratificada 80/20 (1.564 juegos de prueba). El flujo didáctico de
notebooks añade además `GridSearchCV` para el control de sobreajuste del baseline logístico. Las
métricas reportadas son AUC OVR-macro, F1-macro, exactitud y la matriz de confusión 3×3, todas
medidas sobre el conjunto de prueba que los modelos no vieron al entrenar.

### 5.2 El embudo comercial de dos etapas y por qué existe

El clasificador de tres clases solo aprendió de juegos que efectivamente vendieron. Aplicarlo
directo a cualquier configuración asumiría que todo juego logra ventas, algo que el propio dataset
desmiente. Por eso el simulador encadena dos preguntas:

1. **¿Logrará tracción comercial?** Es decir, ¿llegará SteamSpy a estimarle ventas? Es un problema
   binario, entrenado con **17.565 juegos de metadata completa**, donde el 73 % nunca registra
   ventas estimables (`train_stage1.py`).
2. **Si vende, ¿cuánto?** Ahí entra el clasificador de tres clases, entrenado con los 7.817 juegos
   con ventas.

Las cuatro franjas del embudo salen de multiplicar probabilidades (por ejemplo, P(Hit total) =
P(tracción) × P(Hit | vende)) y siempre suman 100 %: sin tracción, Flop, Rentable y Hit.

### 5.3 Los notebooks (versión didáctica para Databricks)

`scripts/train_models.py` produce los modelos de producción de una sola vez. Los notebooks existen
con otro propósito: recorrer el proceso paso a paso, una técnica por cuaderno, para poder explicar y
defender cada decisión por separado. Se ejecutan en Databricks y se comunican entre sí a través de
**tablas de Unity Catalog** (no de archivos), porque Free Edition tiene el almacenamiento público de
archivos deshabilitado. El flujo de datos entre ellos es el siguiente:

```
dataset_ml ──01──▶ features_silver ──▶ 02,03,04,06,07
           └──────────────────────────▶ 05,08
02 ─▶ model_lr_coefficients ─▶ 07
```

| Notebook | Lee | Produce | Qué hace y por qué |
|----------|-----|---------|--------------------|
| `01_data_prep` | `dataset_ml` | `features_silver`, `feature_columns` | Selecciona, de forma schema-driven, las 92 columnas pre-lanzamiento, descarta las casi constantes y deja una tabla limpia lista para modelar. Es la base de la que beben los demás. |
| `02_logistic_regression` | `features_silver` | `model_lr_coefficients` | Entrena el baseline. Lo importante aquí no es la exactitud sino los **coeficientes**: cuánto empuja cada variable hacia cada clase (log-odds), que es lo que hace al modelo interpretable y alimenta el gráfico de importancia. |
| `03_svm` | `features_silver` | — | Máquina de vectores de soporte con kernel RBF. Sirve para ver cuánto gana un modelo no lineal frente al baseline en el espacio de alta dimensión que crean los tags. |
| `04_mlp` | `features_silver` | — | Perceptrón multicapa (64→32, ReLU). El modelo más flexible; captura interacciones que los anteriores no ven (por ejemplo, que un precio alto penaliza en un indie 2D pero no en un RPG de mundo abierto). |
| `05_bayesian_shrinkage` | `dataset_ml` | `dev_success_prior` | Calcula el prior por estudio con Empirical Bayes (Beta-Binomial). Los juegos de un mismo estudio no son independientes; el *shrinkage* encoge la estimación de los estudios con pocos juegos hacia la media global, evitando conclusiones sobreoptimistas a partir de uno o dos títulos. |
| `06_evaluation` | `features_silver` | — | Mide a los modelos sobre el conjunto de prueba (AUC, F1, matriz de confusión) y aplica `GridSearchCV` de 3 particiones para controlar el sobreajuste. Es el cuaderno que compara y decide cuál es el modelo de referencia. |
| `07_dashboard` | `features_silver`, `model_lr_coefficients` | — | Un simulador multiclase dentro del propio notebook, antecedente del dashboard web. Toma una configuración de juego y devuelve las probabilidades y un primer motor de recomendaciones. |
| `08_train_all` | `dataset_ml` | `model_metrics`, `model_lr_coefficients` | Entrenamiento consolidado de producción. **No duplica lógica**: importa y ejecuta `scripts/train_models.py`, el mismo archivo que se corre en local, de modo que el resultado es idéntico por construcción. |

La separación tiene una ventaja pedagógica: cada cuaderno se puede leer y correr de forma aislada
(siempre que `01` ya haya dejado `features_silver`), y cada uno responde a una sección del marco
teórico (baseline lineal, modelos no lineales, jerárquico bayesiano, evaluación). El notebook `08`
es el puente con producción: lo que el resto explica en piezas, él lo deja entrenado de una sola vez
con la misma lógica del backend.

**Restricciones de Free Edition que condicionan el diseño.** Como el entorno es serverless, la MLlib
clásica de PySpark está bloqueada; por eso se modela con scikit-learn sobre `toPandas()` (el conjunto
cabe en memoria del driver). Y como no se pueden descargar archivos del serverless, los `.joblib` que
sirve el backend se generan en local: Databricks deja la evidencia como tablas (`model_metrics`,
`model_lr_coefficients`), no como artefactos descargables.

---

## 6. Resultados

### 6.1 Comparación de modelos (conjunto de prueba, 1.564 juegos)

| Modelo | AUC (OVR-macro) | F1-macro | Exactitud |
|--------|:---:|:---:|:---:|
| Regresión logística | 0,884 | 0,692 | 0,697 |
| SVM (RBF) | 0,881 | 0,719 | 0,739 |
| Perceptrón multicapa | **0,894** | **0,751** | **0,783** |

Los tres modelos discriminan de forma parecida en AUC (alrededor de 0,88). El perceptrón obtiene la
mejor exactitud y F1, así que es el modelo de referencia del dashboard. La regresión logística rinde
algo menos pero aporta interpretabilidad, y por eso se conserva para explicar qué variables
influyen. La diferencia de AUC entre el mejor y el peor es de tres milésimas, de modo que la
elección entre ellos pesa más por interpretabilidad y exactitud que por capacidad de ranking.

### 6.2 Matriz de confusión del perceptrón

Filas: categoría real. Columnas: categoría predicha.

| real \ predicho | Flop | Rentable | Hit |
|---|:---:|:---:|:---:|
| **Flop** | 290 | 125 | 14 |
| **Rentable** | 55 | 756 | 36 |
| **Hit** | 18 | 92 | 178 |

La sensibilidad por categoría es 68 % en Flop (290/429), 89 % en Rentable (756/847) y 62 % en Hit
(178/288). Casi toda la confusión ocurre entre categorías vecinas: un Flop se confunde con Rentable
mucho más que con Hit, y lo mismo pasa entre Hit y Rentable. Eso es razonable porque la variable es
ordinal; el modelo rara vez salta de un extremo al otro (solo 14 Flop reales se predijeron como Hit,
y 18 Hit como Flop). La categoría Hit es la más difícil, lo que concuerda con que es la minoritaria y
la que depende de factores externos al juego (marketing, comunidad, momento de mercado) que el
modelo no observa.

### 6.3 Regresión de propietarios

El modelo de regresión alcanza un R² de 0,63 sobre el logaritmo de los propietarios, con un error
absoluto medio de unos 470.000 propietarios. Sirve bien para ordenar juegos y para alimentar los
umbrales ajustables, aunque su error en escala absoluta es alto, algo esperable dado que SteamSpy
reporta rangos amplios.

### 6.4 Etapa de tracción (embudo)

La etapa de tracción alcanza un AUC de 0,896 y una exactitud de 0,86 sobre los 17.565 juegos de
metadata completa. Un proyecto bien configurado obtiene alrededor de 51 % de probabilidad de
tracción frente a un 4 % de uno pobre.

### 6.5 Un sesgo de recolección detectado y corregido

La etapa de tracción se exploró primero sobre los 32.959 juegos del catálogo completo y alcanzó un
AUC de 0,96, aparentemente excelente. El análisis de los coeficientes reveló el problema: el
predictor más fuerte era `platform_windows` (peso ≈ −9,3), seguido de la ausencia de fecha de
lanzamiento. Esas variables no miden la calidad del juego sino la forma en que se recolectaron los
datos. El subconjunto con ventas provino del crawl de SteamSpy, con metadata completa; el subconjunto
sin ventas, del catálogo de Steam, con campos sin poblar. El modelo separaba ambos mundos por la
presencia de metadata, no por el diseño del juego. Al quitar las dos variables más sospechosas el
AUC apenas bajó a 0,94 y otras del mismo tipo ocuparon su lugar, lo que confirmó que la separación se
apoyaba en artefactos de recolección.

La solución fue filtrar a juegos con metadata completa (fecha de lanzamiento real, plataforma
declarada, al menos un género o etiqueta y descripción): 17.565 juegos. Sobre ese subconjunto el
artefacto desaparece, `platform_windows` deja de dominar y la etapa de tracción baja a un AUC de
0,896 apoyado en señales reales del juego. El filtro también es coherente con el dominio: casi todos
los juegos de Steam son de Windows, así que un `platform_windows` en cero delataba metadata faltante,
no un juego sin esa plataforma. La lección quedó registrada: un AUC alto puede esconder un sesgo de
recolección; auditar los coeficientes lo reveló y filtrar por completitud de metadata recuperó un
modelo válido sin descargar más datos.

### 6.6 Calidad de datos: precios en moneda regional

Durante las pruebas se detectó que varios títulos muy conocidos figuraban con precios imposibles:
Cyberpunk 2077 a 199 dólares, ELDEN RING a 249 o Red Dead Redemption 2 a 53.990. El ETL había
capturado, para 46 juegos, el precio en moneda regional o el de una edición especial vigente en el
momento del scrape. La corrección re-consultó únicamente esos 46 appids contra la Steam Storefront
API forzando la región estadounidense (`cc=us`) y reescribió el precio en la tabla de metadatos. Los
precios altos legítimos (software como RPG Maker o juegos cuyo precio elevado es deliberado) se
mantuvieron intactos. Tras regenerar el dataset y reentrenar, el perceptrón mejoró, señal de que el
ruido de precios estaba degradando una de las variables con mayor peso. La corrección ya está
aplicada en los CSV del repositorio. Quedan dos lecciones: el precio de un catálogo internacional
debe extraerse fijando la región desde el inicio del ETL, y una validación temprana con casos
conocidos habría detectado el problema antes del primer entrenamiento.

### 6.7 Hallazgos descriptivos

- El cuarto trimestre concentra la mayor proporción de Hit (22,9 %), por encima de los otros tres
  (alrededor de 20,7–20,8 %). Coincide con la temporada de fin de año.
- Los géneros con mayor mediana de propietarios son los multijugador masivo y, en general, los que
  combinan componente social.
- Entre los factores que la regresión logística asocia a la categoría Hit aparecen el número de
  etiquetas, el soporte multijugador y la experiencia previa del estudio.

---

## 7. Ingeniería de datos

El proyecto incorpora tres técnicas vistas en la asignatura de Ingeniería de Datos, aplicadas al
conjunto de Steam (carpeta `ingenieria_datos/`).

- **Anonimización.** Los nombres de estudio y distribuidora se reemplazan por un hash SHA-256 de 16
  caracteres (`dev_id`, `pub_id`). El procedimiento recorre los 32.966 registros (25.248 estudios y
  23.420 distribuidoras únicas) y permite publicar el análisis sin exponer la identidad de las
  empresas.
- **Procesamiento con Spark.** Un análisis con RDD sobre Databricks recorre los 32.966 registros y
  obtiene conteos y agregaciones (estudios únicos, distribución de géneros). Demuestra que el flujo
  escala más allá de lo que cabe en una sola máquina. Una verificación local en pandas reproduce los
  mismos números sin necesidad de instalar Spark.
- **Datos de panel.** La serie mensual de reseñas (una observación por juego y mes) tiene estructura
  de panel. En este estudio se usa solo de forma agregada para describir el conjunto; explotar su
  dimensión temporal queda como línea futura.

---

## 8. El producto

El resultado visible es un dashboard con dos modos de uso: local (sin conexión, útil para la
exposición) y en línea, desplegado en el plan gratuito de Render en
<https://steampredict.onrender.com>. El backend está hecho con la librería estándar de Python, sin
dependencias externas: carga los modelos serializados y expone una API; el frontend, con estética
inspirada en Steam, es una sola página que consume esa API. El plan gratuito de Render duerme tras
quince minutos sin tráfico, así que el primer acceso puede tardar entre treinta y sesenta segundos en
despertar el servidor.

El dashboard ofrece cinco vistas:

1. **Simulador.** El usuario configura el juego y obtiene el embudo de dos etapas, la clasificación
   de consenso, la probabilidad de cada clase, los owners estimados, el ingreso bruto aproximado y un
   indicador de incertidumbre.
2. **Comparación de modelos.** El mismo input pasa por los tres clasificadores en paralelo, con sus
   probabilidades y una tabla de métricas (AUC, F1, exactitud).
3. **Juegos del mismo camino.** Vecinos reales del juego simulado, con su categoría real, owners y
   precio, y la portada tomada del CDN de Steam.
4. **Recomendaciones.** Un motor de búsqueda greedy que propone el mejor paquete de cambios para
   subir la probabilidad de Hit, junto con el efecto aislado de cada ajuste.
5. **Panel analítico.** Siete gráficos sobre el mercado que, además de alimentarse de los agregados
   reales del backend, reaccionan a la simulación en curso: el histograma de precios marca el rango
   del precio configurado, el gráfico de géneros resalta los géneros activos, el de estacionalidad
   señala el trimestre elegido y el diagrama de dispersión sitúa al juego simulado entre juegos
   reales identificables por nombre.

Cada gráfico y cada tarjeta incluye un icono de ayuda con una explicación de qué muestra, cómo se
calcula y cómo leerlo, pensada para que el panel se entienda sin un presentador al lado.

---

## 9. Guía de conceptos del dashboard

Esta sección explica, en lenguaje llano, qué significa cada elemento que muestra el dashboard y por
qué se construyó así. Está pensada como apoyo para la defensa del trabajo.

### 9.1 Por qué tres modelos y por qué se entrenaron así

Los tres clasificadores comparten exactamente el mismo preprocesamiento, y eso es deliberado: si
cada modelo viera los datos de forma distinta, la comparación entre ellos no diría nada. Las
variables numéricas se estandarizan porque la SVM y el perceptrón son sensibles a la escala: sin
estandarizar, una variable en dólares y otra que cuenta etiquetas competirían en unidades
incomparables. Las categóricas se convierten a one-hot y las binarias pasan directo.

El parámetro `class_weight="balanced"` responde al desbalance de clases. Sin él, un modelo perezoso
aprendería a responder "Rentable" casi siempre y aun así acertaría la mitad de las veces. Con los
pesos balanceados, equivocarse en un Hit (la clase minoritaria) cuesta más que equivocarse en un
Rentable, y el modelo se ve obligado a aprender las tres categorías.

La elección de los algoritmos también tiene lógica: la regresión logística es lineal e interpretable;
la SVM con kernel RBF captura fronteras no lineales; y el perceptrón modela interacciones complejas
entre variables y resultó el de mejor desempeño. Comparar un modelo simple contra dos
progresivamente más flexibles permite ver cuánta señal adicional aporta la complejidad.

### 9.2 Probabilidad de no-Flop

Es la suma P(Rentable) + P(Hit): la probabilidad de que el juego al menos recupere la inversión. Se
muestra como indicador propio porque, para decidir si lanzar, a un estudio le suele importar más "no
perder dinero" que "ser un éxito masivo". Una configuración con 15 % de Hit pero 80 % de no-Flop es
una apuesta razonable; una con 25 % de Hit pero 50 % de no-Flop es una moneda al aire.

### 9.3 Incertidumbre

Una probabilidad sin su nivel de confianza invita a malas decisiones, así que el dashboard la
acompaña de dos señales. El margen top-2 es la diferencia entre las dos clases más probables: si la
predicción es 45 % Rentable contra 43 % Flop, el modelo en realidad está dudando, aunque "gane"
Rentable. El desacuerdo entre modelos compara la clase ganadora de los tres algoritmos: como cada uno
mira los datos con una lente distinta, que los tres coincidan es evidencia de una señal robusta, y
que discrepen delata un caso ambiguo. El nivel mostrado (baja, media o alta) combina ambas señales.

### 9.4 Métricas de evaluación

- **Exactitud.** Porcentaje de aciertos. Es la métrica más intuitiva pero la más engañosa con clases
  desbalanceadas: responder siempre "Rentable" daría 54 % de exactitud sin haber aprendido nada. Por
  eso nunca se reporta sola.
- **F1-macro.** Para cada clase combina precisión y recall en una media armónica, y luego promedia
  las tres clases sin ponderar por tamaño. Eso obliga a rendir bien también en la clase minoritaria
  (Hit): descuidarla hunde el F1-macro aunque la exactitud global se mantenga.
- **AUC (OVR-macro).** Mide capacidad de ordenamiento, independiente del umbral de decisión; 0,5
  equivale al azar y 1,0 a separación perfecta. Es la métrica que delató el sesgo de recolección de
  la sección 6.5: un AUC demasiado bueno merece auditoría, no celebración.
- **Matriz de confusión.** Muestra dónde se equivoca el modelo, no solo cuánto. Aquí los errores se
  concentran entre clases vecinas, coherente con una variable ordinal.

### 9.5 Owners estimados e ingreso bruto

El número de propietarios sale de la regresión de gradient boosting entrenada sobre el logaritmo de
los owners; se usa logaritmo porque la variable abarca varios órdenes de magnitud y en escala cruda
los pocos gigantes dominarían el ajuste. El ingreso bruto es una aproximación deliberadamente simple:
owners × precio × 70 % (la comisión de Steam ronda el 30 %), sin descontar rebajas ni precios
regionales. Debe leerse como cota de referencia, no como proyección financiera.

---

## 10. Limitaciones y trabajo futuro

- El número de propietarios es una cota inferior estimada por SteamSpy, no la cifra exacta de ventas.
  Los umbrales heredan esa imprecisión.
- La cobertura se limita a juegos con estimación de ventas, lo que sesga la muestra hacia títulos con
  cierta visibilidad.
- El modelo no usa todavía el texto de la descripción ni las imágenes de la ficha, que la literatura
  señala como predictivos.
- El indicador de seguidores del hub (proxy de listas de deseos) quedó mal mapeado en el ETL y se
  descartó por varianza casi nula.
- Las líneas siguientes son la extracción de señales de texto de la descripción, el uso del panel
  mensual completo para un modelo de evolución temporal y la ampliación del conjunto a más juegos.

---

## 11. Reproducir el proyecto

### 11.1 Entorno

Todo el pipeline local (ETL, dataset, entrenamiento y backend) corre con un único entorno. Las
versiones exactas están en `requirements.txt`; los modelos serializados requieren scikit-learn 1.8.0
para cargarse.

```bash
# Opción A: micromamba / conda (entorno aislado, recomendado)
micromamba create -f environment.yml
micromamba activate steampredict

# Opción B: venv + pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 11.2 Pipeline de datos y modelos

```bash
python scripts/steam_etl.py --sample 2000 --source steamspy   # (opcional) traer más juegos
python scripts/build_dataset.py                               # consolidar el master con las 92 features
python scripts/train_models.py                                # LR, SVM, MLP, regresión de owners y vecinos
python scripts/train_stage1.py                                # etapa 1 del embudo (tracción, datos filtrados)
python app/server.py                                          # levantar el dashboard en localhost:8000
```

### 11.3 Entrenamiento en Databricks

El entrenamiento tiene una sola fuente: `scripts/train_models.py`. En local guarda los `.joblib`; en
Databricks lee la tabla `dataset_ml` de Unity Catalog y guarda las métricas y los coeficientes como
tablas. El notebook `notebooks/08_train_all.py` solo importa y ejecuta ese archivo. Para la versión
didáctica, los notebooks 01 a 07 recorren el proceso paso a paso en el orden descrito en la sección
5.3.
