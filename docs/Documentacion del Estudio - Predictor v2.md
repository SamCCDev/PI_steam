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
concurrentes y los agregados mensuales de la serie temporal. Esas columnas permanecen en el conjunto con su
etapa marcada en el diccionario de datos, pero ningún modelo las consume.

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
| SVM (RBF) | 0,881 | 0,719 | 0,739 |
| Perceptrón multicapa | **0,890** | **0,741** | **0,769** |

Los tres modelos discriminan de forma parecida en AUC (alrededor de 0,88). El perceptrón obtiene la mejor
exactitud y F1, así que es el modelo de referencia del dashboard. La regresión logística rinde algo menos pero
aporta interpretabilidad, y por eso se conserva para explicar qué variables influyen. La diferencia de AUC
entre el mejor y el peor es de tres milésimas, de modo que la elección entre ellos pesa más por
interpretabilidad y exactitud que por capacidad de ranking.

### 5.2 Matriz de confusión del perceptrón

Filas: categoría real. Columnas: categoría predicha.

| real \ predicho | Flop | Rentable | Hit |
|---|:---:|:---:|:---:|
| **Flop** | 299 | 119 | 11 |
| **Rentable** | 71 | 721 | 55 |
| **Hit** | 24 | 81 | 183 |

La sensibilidad por categoría es 70 % en Flop (299/429), 85 % en Rentable (721/847) y 64 % en Hit (183/288).
Casi toda la confusión ocurre entre categorías vecinas: un Flop se confunde con Rentable mucho más que con Hit,
y lo mismo pasa entre Hit y Rentable. Eso es razonable porque la variable es ordinal; el modelo rara vez salta
de un extremo al otro (solo 11 Flop reales fueron predichos como Hit, y 24 Hit como Flop). La categoría Hit es
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

### 5.5 Modelo en dos etapas (embudo) y un sesgo de recolección corregido

Se exploró un enfoque de embudo en dos etapas: una primera etapa que estimara si el juego logra tracción
comercial (que SteamSpy reporte propietarios) sobre los 32.959 juegos del catálogo, y una segunda condicionada
a esa tracción que predijera Flop/Rentable/Hit. La intención era aprovechar todos los datos, no solo los 7.817
con ventas estimadas.

La etapa de tracción alcanzó un AUC de 0,96, aparentemente excelente. El análisis de sus coeficientes reveló el
problema: el predictor más fuerte era `platform_windows` (peso ≈ −9,3), seguido de la ausencia de fecha de
lanzamiento. Esas variables no miden la calidad del juego sino la **forma en que se recolectaron los datos**: el
subconjunto con ventas provino del crawl de SteamSpy y el subconjunto sin ventas del catálogo de Steam, y cada
fuente pobló de manera distinta campos como plataformas, fecha, cartas de intercambio, DLCs o requisitos de RAM.
Al quitar las dos variables más sospechosas el AUC apenas bajó a 0,94 y otras del mismo tipo ocuparon su lugar,
lo que confirma que la separación se apoya en artefactos de recolección y no en señales comerciales reales.

La causa era de fondo: los juegos sin tracción provenían del catálogo de Steam con campos sin poblar (sin fecha,
sin plataforma, sin descripción), mientras que los juegos con tracción venían del crawl de SteamSpy con metadata
completa. El modelo separaba ambos mundos por la presencia de metadata, no por el diseño del juego. En una
configuración típica esto hacía colapsar la tracción a ~2 %.

La solución fue **filtrar a juegos con metadata completa** (fecha de lanzamiento real, plataforma declarada, al
menos un género o etiqueta y descripción): 17.565 juegos. Sobre ese subconjunto el artefacto desaparece —
`platform_windows` deja de dominar y la etapa de tracción baja a un AUC de 0,89, ahora apoyado en señales reales
del juego. Las probabilidades se vuelven sensatas: un proyecto bien configurado obtiene ~51 % de probabilidad de
tracción frente a ~4 % de uno pobre. El filtro también es coherente con el dominio: casi todos los juegos de
Steam son de Windows, así que un `platform_windows` en cero delataba metadata faltante, no un juego sin esa
plataforma.

Con esa corrección el embudo se incorpora al producto: el simulador muestra P(tracción) y, combinada con la
etapa 2, los cuatro resultados (sin tracción / Flop / Rentable / Hit). La lección metodológica queda registrada:
un AUC alto puede esconder un sesgo de recolección; auditar los coeficientes lo reveló, y filtrar por completitud
de metadata recuperó un modelo válido sin descargar más datos.

### 5.6 Calidad de datos: reparación de precios en moneda regional

Durante las pruebas del simulador se detectó que varios títulos muy conocidos figuraban con precios
imposibles: Cyberpunk 2077 a 199 dólares, ELDEN RING a 249 o Red Dead Redemption 2 a 53.990. La revisión del
origen mostró que el ETL había capturado, para 46 juegos, el precio en moneda regional o el de una edición
especial vigente en el momento del scrape, y que el constructor del dataset los recortaba después al tope de
200 dólares. El error afectaba sobre todo a títulos AAA, justamente los más visibles al validar el modelo.

La reparación fue quirúrgica: un script (`scripts/fix_price_outliers.py`) re-consultó únicamente esos 46
appids contra la Steam Storefront API forzando la región estadounidense (`cc=us`) y reescribió el precio en
la tabla de metadatos, conservando un respaldo del archivo original. Los precios altos legítimos —software
como RPG Maker o juegos cuyo precio elevado es deliberado, como "This Game Costs 200 Dollars"— se mantuvieron
intactos. Tras la corrección se regeneró el dataset completo y se reentrenaron los cinco modelos; el
perceptrón mejoró ligeramente (AUC de 0,888 a 0,890 y sensibilidad de Hit de 60 % a 64 %), señal de que el
ruido de precios estaba degradando una de las variables con mayor peso.

Quedan dos lecciones para la sección de limitaciones. Primero, el precio de un catálogo internacional debe
extraerse fijando explícitamente la región desde el inicio del ETL. Segundo, una validación temprana con
casos conocidos (¿cuánto cuesta Cyberpunk?) habría detectado el problema antes del primer entrenamiento;
incorporamos esa verificación a la rutina de pruebas del dashboard.

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
  En este estudio se usa solo de forma agregada para describir el conjunto; explotar su dimensión temporal
  queda como línea futura (sección 9).

El entrenamiento es reproducible en dos entornos: en local con `train_models.py` y en Databricks con el
notebook 08, que deja las métricas y los coeficientes como tablas de Unity Catalog.

## 7. Producto

El resultado visible es un dashboard con dos modos de uso: local (sin conexión, útil para la exposición) y
**en línea**, desplegado en el plan gratuito de Render en <https://steampredict.onrender.com>. El backend
(Python estándar, sin dependencias externas) carga los modelos y expone una API; el frontend, con estética
inspirada en Steam, ofrece cinco vistas: simulador, comparación de los tres modelos, juegos reales de perfil
parecido, recomendaciones para mejorar la probabilidad de Hit y un panel analítico con siete gráficos del
mercado.

El panel analítico no es estático: además de alimentarse de los agregados reales del backend, reacciona a la
simulación en curso. El histograma de precios marca el rango donde cae el precio configurado, el gráfico de
géneros resalta los géneros activos, el de estacionalidad señala el trimestre elegido y el diagrama de
dispersión precio-propietarios sitúa al juego simulado (con su clase y probabilidad de Hit) entre 1.200
juegos reales identificables por nombre. Cada gráfico y cada tarjeta del simulador incluye un icono de ayuda
con una explicación de qué muestra, cómo se calcula y cómo leerlo, pensada para que el panel se entienda sin
necesidad de un presentador al lado.

## 8. Guía de conceptos del dashboard (apoyo para el estudio)

Esta sección explica, en lenguaje llano, qué significa cada elemento que muestra el dashboard y por qué se
construyó así. Está pensada como material de estudio para la defensa.

### 8.1 Por qué tres modelos y por qué se entrenaron así

Los tres clasificadores comparten exactamente el mismo preprocesamiento, y eso es deliberado: si cada modelo
viera los datos de forma distinta, la comparación entre ellos no diría nada. Las variables numéricas se
estandarizan (media 0, desviación 1) porque la SVM y el perceptrón son sensibles a la escala: sin
estandarizar, una variable en dólares y otra que cuenta tags competirían en unidades incomparables. Las
categóricas se convierten a one-hot y las binarias pasan directo.

El parámetro `class_weight="balanced"` responde al desbalance de clases (54 % Rentable, 27 % Flop, 18 % Hit).
Sin él, un modelo perezoso aprendería a responder "Rentable" casi siempre y aun así acertaría la mitad de las
veces. Con los pesos balanceados, equivocarse en un Hit —la clase minoritaria— cuesta más que equivocarse en
un Rentable, y el modelo se ve obligado a aprender las tres categorías.

La partición 80/20 es estratificada: el 20 % de prueba conserva la misma proporción de clases que el total.
Todas las métricas reportadas se miden únicamente sobre ese 20 % que los modelos jamás vieron durante el
entrenamiento; medir sobre los datos de entrenamiento solo demostraría memoria, no capacidad de generalizar.

La elección de los tres algoritmos también tiene lógica: la regresión logística es lineal e interpretable
(sus coeficientes alimentan el gráfico de importancia de variables y el motor de recomendaciones); la SVM con
kernel RBF captura fronteras no lineales; y el perceptrón multicapa modela interacciones complejas entre
variables y resultó el de mejor desempeño. Comparar un modelo simple contra dos progresivamente más flexibles
permite ver cuánta señal adicional aporta la complejidad.

### 8.2 El embudo comercial de dos etapas

El simulador responde dos preguntas encadenadas. La primera: ¿este juego logrará siquiera una tracción
comercial medible? Es un problema binario, entrenado con 17.565 juegos de metadata completa, donde el 73 %
nunca registra ventas estimables. La segunda: suponiendo que vende, ¿cuánto? Ahí entra el clasificador de
tres clases, entrenado solo con los 7.817 juegos que sí vendieron.

La separación importa porque el clasificador aprendió únicamente de juegos que entraron al mercado; aplicarlo
directo a cualquier configuración asumiría que todo juego vende algo, lo cual el propio dataset desmiente.
Las cuatro franjas del embudo salen de multiplicar probabilidades —por ejemplo, P(Hit total) = P(tracción) ×
P(Hit | vende)— y siempre suman 100 %: sin tracción, Flop, Rentable y Hit.

### 8.3 Probabilidad de no-Flop

Es la suma P(Rentable) + P(Hit): la probabilidad de que el juego al menos recupere la inversión. Se muestra
como indicador propio porque, para decidir si lanzar, a un estudio le suele importar más "no perder dinero"
que "ser un éxito masivo". Una configuración con 15 % de Hit pero 80 % de no-Flop es una apuesta razonable;
una con 25 % de Hit pero 50 % de no-Flop es una moneda al aire.

### 8.4 Incertidumbre

Una probabilidad sin su nivel de confianza invita a malas decisiones, así que el dashboard la acompaña de dos
señales. El margen top-2 es la diferencia entre las dos clases más probables: si la predicción es 45 %
Rentable contra 43 % Flop, el modelo en realidad está dudando, aunque "gane" Rentable. El desacuerdo entre
modelos compara la clase ganadora de los tres algoritmos: como cada uno mira los datos con una lente distinta
(lineal, kernel, red neuronal), que los tres coincidan es evidencia de una señal robusta, y que discrepen
delata un caso ambiguo. El nivel mostrado (baja/media/alta) combina ambas señales.

### 8.5 Métricas de evaluación

- **Accuracy (exactitud).** Porcentaje de aciertos. Es la métrica más intuitiva pero la más engañosa con
  clases desbalanceadas: responder siempre "Rentable" daría 54 % de exactitud sin haber aprendido nada. Por
  eso nunca se reporta sola.
- **F1-macro.** Para cada clase combina precisión (de lo que predije como X, cuánto era X) y recall (de los X
  reales, cuántos encontré) en una media armónica, y luego promedia las tres clases sin ponderar por tamaño.
  Eso obliga a rendir bien también en la clase minoritaria (Hit): descuidarla hunde el F1-macro aunque la
  exactitud global se mantenga.
- **AUC (OVR-macro).** Mide capacidad de ordenamiento: la probabilidad de que el modelo asigne mayor puntaje
  a un caso positivo que a uno negativo, promediada tratando cada clase contra el resto. Es independiente del
  umbral de decisión; 0,5 equivale al azar y 1,0 a separación perfecta. Es la métrica que delató el sesgo de
  recolección de la sección 5.5: un AUC "demasiado bueno" merece auditoría, no celebración.
- **Matriz de confusión.** Muestra dónde se equivoca el modelo, no solo cuánto. En este problema los errores
  se concentran entre clases vecinas (Flop↔Rentable, Rentable↔Hit), lo que es coherente con una variable
  ordinal: el modelo casi nunca confunde los extremos.

### 8.6 Owners estimados e ingreso bruto

El número de propietarios sale de una regresión independiente (gradient boosting) entrenada sobre el
logaritmo de los owners — se usa logaritmo porque la variable abarca varios órdenes de magnitud (de miles a
decenas de millones) y en escala cruda los pocos gigantes dominarían el ajuste. Esta estimación alimenta el
punto "Tu juego" del panel analítico y permite reclasificar con umbrales ajustables. El ingreso bruto es una
aproximación deliberadamente simple: owners × precio × 70 % (la comisión de Steam es ~30 %), sin descontar
rebajas ni precios regionales; debe leerse como cota de referencia, no como proyección financiera.

## 9. Limitaciones y trabajo futuro

- El número de propietarios es una cota inferior estimada por SteamSpy, no la cifra exacta de ventas. Los
  umbrales heredan esa imprecisión.
- La cobertura se limita a juegos con estimación de ventas, lo que sesga la muestra hacia títulos con cierta
  visibilidad.
- El modelo no usa todavía el texto de la descripción ni las imágenes de la ficha, que la literatura señala
  como predictivos.
- El indicador de seguidores del hub (proxy de listas de deseos) quedó mal mapeado en el ETL y se descarta por
  varianza casi nula.
- Las líneas siguientes son: la extracción de señales de texto de la descripción (NLP), el uso del panel
  mensual completo para un modelo de evolución temporal, y la ampliación del conjunto a más juegos.
