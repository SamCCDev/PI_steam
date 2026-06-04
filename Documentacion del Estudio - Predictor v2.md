# Predictor de éxito comercial de videojuegos en Steam — Documentación metodológica (v2)

Documento de respaldo del estudio. Describe cómo se construyó el conjunto de datos, qué variables se
usaron, cómo se entrenaron y compararon los modelos, y qué resultados se obtuvieron. Las cifras provienen
de la ejecución reproducible de `build_dataset_v2.py` y `train_models.py` (replicada en Databricks por el
notebook `notebooks/08_train_all.py`).

---

## 1. Objeto del estudio

El sistema estima en qué categoría comercial caerá un videojuego de PC a partir de información disponible
**antes de su lanzamiento**. Las categorías se definen por el número de propietarios que reporta SteamSpy:

| Categoría | Propietarios (owners) | Interpretación |
|-----------|-----------------------|----------------|
| Flop | menos de 200.000 | ventas comercialmente bajas |
| Rentable | 200.000 a 1.000.000 | recupera costos y deja margen |
| Hit | 1.000.000 o más | éxito comercial claro |

El producto es un simulador: el usuario describe un juego hipotético (precio, género, etiquetas, idiomas,
experiencia del estudio, etc.) y obtiene la probabilidad de cada categoría según tres modelos distintos.

## 2. Construcción del conjunto de datos

Los datos se obtienen de dos servicios públicos: la Steam Storefront API (metadatos, géneros, categorías,
precios) y la API de SteamSpy (estimación de ventas y etiquetas de usuario). El script `steam_etl.py` los
descarga en cuatro CSV relacionales (metadatos, etiquetas, texto y serie temporal de reseñas) y
`build_dataset_v2.py` los consolida en un único maestro con una fila por juego.

Del catálogo descargado (32.966 juegos) se conservan **7.817** que tienen estimación de ventas en SteamSpy.
Los restantes no es que vendan poco: SteamSpy no publica estimación para ellos, de modo que su número de
propietarios figura como cero. Etiquetarlos como Flop introduciría ruido, así que se excluyen del
entrenamiento. La distribución resultante de la variable objetivo es 27,4 % Flop, 54,2 % Rentable y 18,4 % Hit.

## 3. Variables predictoras

El modelo usa 92 variables: 12 numéricas, 75 binarias y 5 categóricas.

- **Numéricas:** precio, número de idiomas soportados, logros, DLCs, RAM mínima, longitud de la descripción
  de tienda, cantidad de juegos previos del estudio y de la distribuidora, riqueza de catálogo del juego
  (cuántos géneros, etiquetas y categorías declara) y un indicador bayesiano del estudio (ver abajo).
- **Binarias:** géneros, categorías de Steam, etiquetas de usuario (tags) y plataformas.
- **Categóricas:** experiencia del estudio y de la distribuidora (Novato / Establecido / AAA según el número
  de juegos previos), soporte de control, tramo de precio y trimestre de lanzamiento.

### 3.1 Ingeniería de variables (v2)

Sobre la versión inicial se agregaron variables que el equipo consideró relevantes para el negocio: el conteo
de etiquetas y géneros, el trimestre de lanzamiento (estacionalidad), el tamaño de la distribuidora, el tramo
de precio y la condición de acceso anticipado.

El indicador bayesiano del estudio (`dev_success_prior`) merece una nota. Estima la probabilidad de que un
juego del mismo estudio sea Rentable o Hit, usando un esquema Beta-Binomial con contracción hacia la media
global. Se calcula dejando fuera el propio juego (leave-one-out), de modo que no filtra su propia etiqueta.
Para un estudio sin juegos previos, el valor se reduce a la media global.

### 3.2 Prevención de fuga de datos

El simulador es pre-lanzamiento, así que se excluyen todas las variables que solo se conocen una vez que el
juego salió a la venta: reseñas positivas y negativas, porcentaje de valoración, nota de Metacritic, jugadores
concurrentes y los agregados mensuales de la serie temporal. Esas columnas permanecen en el conjunto, pero
quedan reservadas para un eventual modelo post-lanzamiento.

## 4. Modelado

El entrenamiento se hace con scikit-learn. La razón es práctica: Databricks Free Edition corre en modo
serverless (Spark Connect) y bloquea la MLlib clásica de PySpark. Como el conjunto cabe en memoria, se trae a
pandas y se modela en el driver. El mismo código corre en local (`train_models.py`) y en Databricks (notebook
08), lo que mantiene la reproducibilidad.

El preprocesamiento es común a los tres modelos: estandarización de las numéricas, codificación one-hot de las
categóricas y paso directo de las binarias. Todos usan `class_weight="balanced"` para compensar el desbalance.

Se entrenaron tres algoritmos:

1. Regresión logística multinomial (softmax). Es el modelo base e interpretable; sus coeficientes indican qué
   factores empujan hacia cada categoría.
2. Máquina de vectores de soporte con kernel RBF, para capturar relaciones no lineales.
3. Perceptrón multicapa (dos capas ocultas, activación ReLU).

Se añadió un modelo de regresión sobre el logaritmo de los propietarios. No clasifica; predice una cantidad.
Aplicar un corte sobre esa predicción permite reclasificar Flop/Rentable/Hit con umbrales movibles sin
reentrenar, que es como el dashboard ofrece umbrales ajustables.

La validación usa una partición estratificada 80/20. La regresión logística se ajustó además con
`GridSearchCV` de 3 particiones para controlar el sobreajuste. Las métricas reportadas son AUC OVR-macro,
F1-macro, exactitud y la matriz de confusión 3×3, todas sobre el conjunto de prueba.

## 5. Resultados

### 5.1 Comparación de modelos (conjunto de prueba, 1.564 juegos)

| Modelo | AUC (OVR-macro) | F1-macro | Exactitud |
|--------|:---:|:---:|:---:|
| Regresión logística | 0,884 | 0,692 | 0,697 |
| SVM (RBF) | 0,881 | 0,720 | 0,739 |
| Perceptrón multicapa | **0,888** | **0,740** | **0,772** |

Los tres modelos discriminan de forma parecida en AUC (alrededor de 0,88). El perceptrón obtiene la mejor
exactitud y F1, así que es el modelo de referencia del dashboard. La regresión logística rinde algo menos pero
aporta interpretabilidad, y por eso se conserva para explicar qué variables influyen. La diferencia de AUC
entre el mejor y el peor es de tres milésimas, de modo que la elección entre ellos pesa más por
interpretabilidad y exactitud que por capacidad de ranking.

### 5.2 Matriz de confusión del perceptrón

Filas: categoría real. Columnas: categoría predicha.

| real \ predicho | Flop | Rentable | Hit |
|---|:---:|:---:|:---:|
| **Flop** | 296 | 121 | 12 |
| **Rentable** | 65 | 737 | 45 |
| **Hit** | 21 | 93 | 174 |

La sensibilidad por categoría es 69 % en Flop (296/429), 87 % en Rentable (737/847) y 60 % en Hit (174/288).
Casi toda la confusión ocurre entre categorías vecinas: un Flop se confunde con Rentable mucho más que con Hit,
y lo mismo pasa entre Hit y Rentable. Eso es razonable porque la variable es ordinal; el modelo rara vez salta
de un extremo al otro (solo 12 Flop reales fueron predichos como Hit, y 21 Hit como Flop). La categoría Hit es
la más difícil, lo que concuerda con que es la minoritaria y la que depende de factores externos al juego
(marketing, comunidad, momento de mercado) que el modelo no observa.

### 5.3 Regresión de propietarios

El modelo de regresión alcanza un R² de 0,63 sobre el logaritmo de los propietarios, con un error absoluto
medio de unos 480.000 propietarios. Sirve bien para ordenar juegos y para alimentar los umbrales ajustables,
aunque su error en escala absoluta es alto, algo esperable dado que SteamSpy reporta rangos amplios.

### 5.4 Hallazgos descriptivos

- El cuarto trimestre concentra la mayor proporción de Hit (22,9 %), por encima de los otros tres (alrededor
  de 20,7–20,8 %). Coincide con la temporada de fin de año.
- Los géneros con mayor mediana de propietarios son los multijugador masivo y, en general, los que combinan
  componente social.
- Entre los factores que la regresión logística asocia a la categoría Hit aparecen el número de etiquetas, el
  soporte multijugador y la experiencia previa del estudio.

## 6. Ingeniería de datos

El proyecto incorpora tres técnicas vistas en la asignatura de Ingeniería de Datos, aplicadas al conjunto de
Steam (carpeta `ingenieria_datos/`).

- **Anonimización.** Los nombres de estudio y distribuidora se reemplazan por un hash SHA-256 de 16 caracteres
  (`dev_id`, `pub_id`). El procedimiento reproduce el del ejercicio de clase sobre datos de aduanas y permite
  publicar el análisis sin exponer la identidad de las empresas.
- **Procesamiento con Spark.** Un análisis con RDD sobre Databricks recorre los 32.966 registros y obtiene
  conteos y agregaciones (25.248 estudios únicos, distribución de géneros). Demuestra que el flujo escala más
  allá de lo que cabe en una sola máquina.
- **Datos de panel.** La serie mensual de reseñas (una observación por juego y mes) tiene estructura de panel.
  Se reserva para el seguimiento post-lanzamiento, donde la evolución temprana de las reseñas sirve para
  refinar la predicción.

El entrenamiento es reproducible en dos entornos: en local con `train_models.py` y en Databricks con el
notebook 08, que deja las métricas y los coeficientes como tablas de Unity Catalog.

## 7. Producto

El resultado visible es un panel local. El backend (Python estándar, sin dependencias externas) carga los
modelos y expone una API; el frontend, con estética inspirada en Steam, ofrece cinco vistas: simulador,
comparación de los tres modelos, juegos reales de perfil parecido, recomendaciones para mejorar la
probabilidad de Hit y un panel analítico con gráficos del mercado. Funciona sin conexión, lo que permite
mostrarlo en la exposición sin depender de la red.

## 8. Limitaciones y trabajo futuro

- El número de propietarios es una cota inferior estimada por SteamSpy, no la cifra exacta de ventas. Los
  umbrales heredan esa imprecisión.
- La cobertura se limita a juegos con estimación de ventas, lo que sesga la muestra hacia títulos con cierta
  visibilidad.
- El modelo no usa todavía el texto de la descripción ni las imágenes de la ficha, que la literatura señala
  como predictivos.
- El indicador de seguidores del hub (proxy de listas de deseos) quedó mal mapeado en el ETL y se descarta por
  varianza casi nula.
- Las líneas siguientes son: una variante post-lanzamiento que aproveche el panel mensual de reseñas, la
  extracción de señales de texto de la descripción y la ampliación del conjunto a más juegos.
