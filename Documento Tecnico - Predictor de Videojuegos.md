# **Proyecto: Análisis Predictivo y Machine Learning en la Industria de los Videojuegos**

**Documento Técnico y de Arquitectura de Datos**  
*Última actualización: 24 de Mayo de 2026, 08:05 AM (Local)*

## **1\. Objetivo Principal**

El objetivo central de este proyecto de investigación y desarrollo es construir un modelo predictivo basado en Machine Learning capaz de identificar, analizar y ponderar las variables que determinan el éxito comercial de un videojuego en el mercado de PC (plataforma Steam).

El sistema dejará de lado la intuición empírica para basarse estrictamente en datos históricos, creando un "Filtro Estratégico Comercial" que permita a desarrolladores, estudios e inversores evaluar la viabilidad de sus proyectos durante la fase de conceptualización o pre-lanzamiento.

## **2\. Justificación y Relevancia Comercial**

La industria del desarrollo de videojuegos se caracteriza por su alto riesgo de inversión y la volatilidad de las preferencias del consumidor. Un fracaso comercial puede llevar a la quiebra a estudios enteros.

Este proyecto busca:

1. **Reducir la incertidumbre del mercado:** Cimentando las decisiones de diseño y marketing en datos históricos tangibles.  
2. **Mitigar riesgos y maximizar el ROI:** Permitiendo a los inversores asignar presupuestos de manera inteligente y a los desarrolladores pivotar o ajustar mecánicas, precios o estrategias de marketing antes de incurrir en gastos irreversibles.  
3. **Proveer una herramienta consultiva:** Ofreciendo un simulador (Dashboard) donde los creadores introduzcan los parámetros de su juego y obtengan una predicción de éxito y recomendaciones estratégicas.

## **3\. Arquitectura y Viabilidad de los Datos**

Para entrenar los algoritmos de Machine Learning (Clasificación, Regresión y NLP), es imperativo construir un dataset robusto. El análisis de viabilidad de las fuentes de datos arroja la siguiente estructura:

### **3.1. Datos Disponibles a través de APIs de Steam y SteamSpy**

La extracción principal se realizará mediante la Steam Storefront API y la API de SteamSpy, obteniendo variables directas como:

* **Identificadores:** appid, name, developer, publisher.  
* **Ventas y Retención:** owners\_lower\_bound (rango estimado de dueños), ccu (pico de jugadores concurrentes).  
* **Recepción:** positive y negative reviews, rating\_porcentaje.  
* **Comercial y Técnico:** price (precio base), is\_free, supported\_languages, total\_dlcs, total\_achievements, soporte de SO (platform\_windows, mac, linux), controller\_support.

### **3.2. Ingeniería de Características (Feature Engineering)**

Para capturar la complejidad del diseño de un videojuego, se procesarán datos semi-estructurados:

* **Clasificación Temática:** genres y categories (Ej: Single-player, Online PvP, In-App Purchases).  
* **Mecánicas y Estilo Visual (User Tags):** Se utilizarán los "User Tags" de Steam para crear variables booleanas (1/0) que capturen el *Core Loop* y el arte del juego. Ejemplos de variables a extraer: *Pixel Graphics, First-Person, Open World, Crafting, Permadeath, Story Rich*.  
* **Texto Libre (NLP):** Extracción de la descripción completa del juego y de las reseñas de usuarios para entrenar modelos de análisis de sentimiento e identificar palabras clave recurrentes en juegos exitosos.

### **3.3. Resolución de Obstáculos de Datos (El Proxy de Tracción)**

Al diseñar el modelo, se identificaron métricas cruciales que no están disponibles públicamente. Se implementarán las siguientes soluciones (Proxys):

* **Problema:** Las *Wishlists* (Listas de deseados) son privadas y constituyen el predictor más fuerte de ventas iniciales.  
* **Solución (Proxy):** Se utilizará el número de **Seguidores del Hub de la Comunidad** (dato público) como variable sustituta. La industria valida una correlación fuerte donde *Wishlists ≈ Seguidores x 7 a 10*.  
* **Integración de Datos Externos:** \* *Duración del Juego:* Requiere cruce de datos (*web scraping*) con bases de datos externas como *HowLongToBeat.com*.  
  * *Éxito de Crowdfunding:* Requiere cruce con datasets públicos de *Kickstarter*.  
  * *Requisitos de Hardware:* Requiere expresiones regulares (Regex) para estructurar el texto libre de los requisitos mínimos publicados en Steam.

## **4\. Metodología de Machine Learning**

El desarrollo del modelo seguirá el pipeline estándar de la ciencia de datos. Cabe destacar que la reducción de los ~15,000 registros de metadata a ~6,500 registros útiles en series de tiempo es un comportamiento estándar y esperado en el proceso de extracción (ETL). Los valores nulos o ausentes en las series de tiempo suelen corresponder a juegos en fase de pre-lanzamiento ("Coming Soon"), títulos retirados de la tienda o proyectos sin la masa crítica mínima de jugadores para generar datos estadísticamente significativos. Un dataset estructurado de este volumen es un bloque sólido y perfectamente funcional para entrenar los modelos de Machine Learning.

La estrategia de modelado se dividirá en fases iterativas, comenzando por establecer un baseline hasta llegar a arquitecturas más complejas:

### **4.1. Fase 1: Modelos Base y Fronteras de Decisión**
Antes de implementar redes neuronales, es mandatorio establecer una línea base de rendimiento. El objetivo inicial debe plantearse como un problema de clasificación (ej. definir si un juego supera las 20,000 unidades vendidas: 1 o 0).

1. **Regresión Logística (Modelo Base):**
   * **Aplicación:** Utiliza una función Sigmoide para la clasificación binaria (Éxito vs. Fracaso comercial) o Softmax si se divide la variable objetivo en múltiples categorías (Flop, Rentable, Hit).
   * **Ventaja:** Permite interpretar directamente los coeficientes. Podrás cuantificar exactamente cuánto aumenta la probabilidad de éxito si se añade la etiqueta de "Multijugador" o si se traduce a 5 idiomas adicionales.

2. **Support Vector Machine (SVM):**
   * **Aplicación:** Dado que el dataset contendrá una alta dimensionalidad producto del One-Hot Encoding de decenas de tags de Steam (acción, permadeath, crafteo, etc.), un SVM es altamente efectivo.
   * **Ventaja:** Al utilizar un kernel radial (RBF), el SVM encontrará el hiperplano óptimo para separar los juegos exitosos de los fallidos en un espacio multidimensional, maximizando el margen entre los vectores de soporte (los juegos más difíciles de clasificar).

### **4.2. Fase 2: Redes Neuronales y Manejo de Incertidumbre**
Una vez validado el rendimiento base, se avanza a modelos capaces de capturar interacciones complejas.

3. **Perceptrón Multicapa (MLP):**
   * **Aplicación:** Una red neuronal artificial densa (Feedforward). La capa de entrada recibirá el vector numérico (precio, especificaciones) y las booleanas de los tags. Las capas ocultas utilizarán funciones de activación no lineales (como ReLU o Tanh).
   * **Ventaja:** Detectará patrones cruzados que los modelos lineales ignoran (ej. un precio de $40 USD puede ser un factor de riesgo en un juego Indie 2D, pero positivo o neutro en un RPG de Mundo Abierto).

4. **Modelos Bayesianos Jerárquicos:**
   * **Aplicación:** Ideal para estructurar dependencias. Los juegos desarrollados por un mismo estudio o publicados por el mismo Publisher no son eventos estadísticamente independientes.
   * **Ventaja:** Permite agrupar los datos por developer. Si el modelo evalúa el nuevo juego de un estudio novato, la distribución de probabilidad se ajustará ("encogimiento" o *shrinkage*) hacia la media global del mercado, mitigando el sobreajuste y entregando una predicción basada en la incertidumbre real.

*(Nota técnica: Aunque las Redes Neuronales Convolucionales - CNN - son extremadamente potentes, están diseñadas para datos espaciales y mallas. No deben aplicarse a este dataset tabular numérico, a menos que en una fase futura se decida descargar los píxeles de las imágenes promocionales de la tienda para su análisis).*

### **4.3. Requisitos Pre-Entrenamiento (Enfoque Data-Ready)**
Para que el dataset sea consumible por Scikit-Learn o arquitecturas en PyTorch/TensorFlow, se ejecutará el siguiente flujo de limpieza estricto:

* **Limpieza de Nulos:** Imputar por la mediana (para variables numéricas como el precio) o eliminar filas donde las variables objetivo (owners_lower_bound, ccu) estén vacías.
* **Codificación:** Aplicar One-Hot Encoding estricto a las listas de géneros y categorías, resultando en columnas tipo int o bool (1/0).
* **Escalado:** Aplicar StandardScaler o MinMaxScaler a variables de alta varianza (price, total_achievements, hub_followers). Una red neuronal MLP no convergerá eficientemente si el precio oscila entre 0 y 60 mientras los seguidores oscilan entre 0 y 1,000,000.
* **Alineación Temporal de Series de Tiempo (Reindexación en Pandas):** El pipeline de extracción (ETL) almacena una única fila con fecha `2024-01` y valores en `0` para aquellos juegos que no registran actividad de reseñas (evitando inflar innecesariamente el tamaño del archivo en disco con registros vacíos redundantes). 
  * *Recomendación para el modelado:* Durante la preparación de datos para entrenamiento, se aconseja expandir la secuencia temporal de cada juego generando todos los meses faltantes entre su fecha de lanzamiento (o la fecha de inicio del estudio) y la actualidad usando `df.reindex(...)` en Pandas, imputando los meses vacíos con `.fillna(0)`. Esto proporciona secuencias regulares homogéneas para modelos de series de tiempo (como LSTMs o RNNs) manteniendo la base de datos optimizada en disco.

### **4.4. Proceso de Validación**
Pruebas estrictas de *cross-validation* para evitar el *overfitting* (sobreajuste), asegurando que el modelo generalice correctamente frente a proyectos nunca antes vistos.

## **5\. Implementación Final: Propuesta de Valor para el Cliente**

El resultado final no será el código del modelo, sino un producto interactivo.

**El Input del Cliente:**

Un estudio de desarrollo introducirá las especificaciones teóricas de su juego: Presupuesto/Precio de venta, nivel de experiencia del estudio, géneros, estilo artístico, mecánicas clave (mundo abierto, cooperativo, etc.) y su nivel actual de tracción pre-lanzamiento (Seguidores en Steam, campañas de Kickstarter).

**El Output del Sistema (El Producto):**

El modelo procesará estos datos y entregará un reporte dinámico en un *Dashboard* que incluirá:

* Probabilidad porcentual de éxito comercial.  
* Estimación de ventas del Día 1 y estimación de Wishlists basadas en tracción actual.  
* **Insights Estratégicos:** Feedback consultivo generado dinámicamente. Por ejemplo: *"Su diseño incluye 'Mundo Abierto', pero su estudio está clasificado como 'Novato'. El algoritmo penaliza esta combinación por riesgo de alcance desmesurado. Si cambia el enfoque a 'Niveles Lineales' y baja el precio en $5, la probabilidad de éxito aumenta un 18%."*

## **6\. Diccionario de Datos Completo (Data Dictionary)**

Para asegurar la correcta ingestión y procesamiento de datos por parte de modelos de Machine Learning y procesos automatizados, se detalla el listado de variables a recolectar.

**Total estimado de variables base:** \~32 variables estructurales \+ Matriz de N variables booleanas (One-Hot Encoding para Tags).

### **6.1. Estructura de Archivos (CSVs)**

Debido a la diferente cardinalidad de los datos (1 juego tiene N reseñas o N tags), los datos se estructurarán de forma relacional en múltiples archivos CSV unidos por la llave principal appid.

* games\_metadata.csv: Contiene la información 1:1 de cada juego (precios, ventas, desarrollador).  
* games\_tags.csv: Contiene la matriz booleana de etiquetas de diseño y mecánicas.  
* games\_text.csv: Contiene textos largos para NLP (descripciones y resúmenes de reviews).  
* games\_external.csv: Contiene métricas obtenidas de plataformas ajenas a Steam.
* games\_timeseries.csv: Contiene métricas mensuales de actividad y retención (desde enero de 2024), extraídas de las reseñas de Steam.

### **6.2. Variables Principales (games\_metadata.csv)**

| Variable | Tipo de Dato | Origen | Descripción |
| :---- | :---- | :---- | :---- |
| appid | Integer (PK) | Steam API | Identificador único del juego. (Llave Primaria). |
| name | String | Steam API | Nombre oficial del juego. |
| developer | String | Steam API | Estudio creador del juego. |
| publisher | String | Steam API | Empresa publicadora. |
| release\_date | Date | Steam API | Fecha de lanzamiento oficial. |
| price | Float | Steam API | Precio de venta en USD (0.0 si es Free-to-Play). |
| is\_free | Boolean | Steam API | Indica si el juego es modelo Free-to-Play. |
| owners\_lower\_bound | Integer | SteamSpy API | **(Variable Objetivo)** Límite inferior estimado de ventas/dueños. |
| ccu | Integer | SteamSpy API | **(Variable Objetivo)** Pico de jugadores concurrentes. |
| rating\_porcentaje | Float | Steam API | **(Variable Objetivo)** Porcentaje de reseñas positivas (0.0 a 100.0). |
| positive | Integer | Steam API | Total de reseñas positivas históricas. |
| negative | Integer | Steam API | Total de reseñas negativas históricas. |
| hub\_followers | Integer | Steam Web | **(Proxy de Wishlists)** Cantidad de seguidores en el Hub de la comunidad. |
| total\_achievements | Integer | Steam API | Cantidad de logros desbloqueables disponibles. |
| total\_dlcs | Integer | Steam API | Cantidad de expansiones de pago disponibles. |
| supported\_languages | Integer / List | Steam API | Cantidad total de idiomas soportados (o lista separada por comas). |
| controller\_support | Categorical | Steam API | Nivel de soporte para mando (full, partial, none). |
| platform\_windows | Boolean | Steam API | Compatibilidad con OS Windows. |
| platform\_mac | Boolean | Steam API | Compatibilidad con OS macOS. |
| platform\_linux | Boolean | Steam API | Compatibilidad con OS Linux / SteamOS. |
| min\_ram\_gb | Float | Steam API (Regex) | Requisito mínimo de memoria RAM extraído del texto de requisitos. |

### **6.3. Variables de Diseño y Mecánicas (games\_tags.csv)**

*Nota: Estas variables se generarán aplicando One-Hot Encoding a las listas de 'Genres', 'Categories' y 'User Tags'.*

| Variable | Tipo de Dato | Origen | Descripción |
| :---- | :---- | :---- | :---- |
| appid | Integer (FK) | Steam API | Llave foránea. |
| genre\_action | Boolean | Steam API | Pertenece al género Acción (y así iterativamente: genre\_rpg, genre\_strategy...). |
| cat\_multiplayer | Boolean | Steam API | Contiene modo multijugador online. |
| cat\_in\_app\_purch | Boolean | Steam API | Contiene microtransacciones internas. |
| tag\_pixel\_art | Boolean | User Tags | Clasificado por usuarios con el estilo visual "Pixel Art". |
| tag\_open\_world | Boolean | User Tags | Clasificado por usuarios con la mecánica "Open World". |
| tag\_permadeath | Boolean | User Tags | Clasificado por usuarios con la mecánica "Permadeath". |
| tag\_story\_rich | Boolean | User Tags | Clasificado por usuarios como "Story Rich" (Alta carga narrativa). |
| *(... \+ N Tags)* | Boolean | User Tags | Se seleccionarán los Top 50-100 Tags más relevantes estadísticamente. |

### **6.4. Variables Externas (games\_external.csv)**

| Variable | Tipo de Dato | Origen | Descripción |
| :---- | :---- | :---- | :---- |
| appid | Integer (FK) | Steam API | Llave foránea. |
| dev\_experience | Categorical | Derivado | Experiencia calculada del estudio (Novato, Establecido, AAA) contando appids previos. |
| kickstarter\_success | Boolean | Kickstarter | Indica si el juego fue financiado exitosamente por crowdfunding. |
| kickstarter\_usd | Float | Kickstarter | Monto recaudado en Kickstarter (si aplica). |
| est\_length\_hours | Float | HowLongToBeat | Tiempo medio estimado para completar la campaña principal. |

### **6.5. Variables para NLP (games\_text.csv)**

| Variable | Tipo de Dato | Origen | Descripción |
| :---- | :---- | :---- | :---- |
| appid | Integer (FK) | Steam API | Llave foránea. |
| short\_description | Text | Steam API | El pitch corto (1-2 párrafos) mostrado en la tienda. |
| about\_the\_game | Text | Steam API | El HTML/Texto completo que detalla las características del juego. |
| reviews\_text\_sample | Text | Steam Reviews | Concatenación de una muestra aleatoria representativa de reseñas de usuarios. |

### **6.6. Variables de Series de Tiempo (games\_timeseries.csv)**

Este archivo contiene el historial de interacción mensual de los usuarios desde Enero de 2024 en adelante.

| Variable | Tipo de Dato | Origen | Descripción |
| :---- | :---- | :---- | :---- |
| appid | Integer (FK) | Steam API | Llave foránea. |
| month | String | Steam Reviews | Mes al que corresponden las métricas (Formato `YYYY-MM`). |
| review\_count | Integer | Steam Reviews | Cantidad total de reseñas publicadas en el mes (Proxy de actividad/ventas). |
| positive\_count | Integer | Steam Reviews | Cantidad de reseñas positivas en el mes (Proxy de sentimiento/recepción). |
| negative\_count | Integer | Steam Reviews | Cantidad de reseñas negativas en el mes. |
| steam\_purchase\_count | Integer | Steam Reviews | Cantidad de usuarios que compraron el juego directamente en Steam (vs claves externas). |
| early\_access\_count | Integer | Steam Reviews | Cantidad de reseñas publicadas mientras el juego estaba en acceso anticipado. |
| avg\_playtime\_at\_review\_hrs | Float | Steam Reviews | Promedio de horas jugadas al momento de emitir la reseña en el mes (Proxy de retención). |

## **7\. Ejecución del Pipeline (Comandos)**

El script de extracción de datos (`steam_etl.py`) está diseñado para instalar sus dependencias de forma automática y puede ejecutarse desde la terminal utilizando los siguientes comandos:

* **Extracción Principal (Por Defecto):**
  Ejecuta el pipeline completo y procesa una muestra predeterminada de 1000 juegos.
  ```bash
  python steam_etl.py
  ```

* **Modo de Prueba (Sample):**
  Procesa únicamente el número de juegos especificado, ideal para pruebas rápidas.
  ```bash
  python steam_etl.py --sample 10
  ```

* **Modo de Diagnóstico (Validate):**
  Solo inspecciona y extrae los datos de un juego en específico (ej. 730 para CS2) y los imprime en consola sin generar ni escribir en los archivos CSV. Ideal para revisar qué extrae la API.
  ```bash
  python steam_etl.py --validate 730
  ```

### **7.1. Optimización y Control de Concurrencia (Producción)**
Para acelerar la extracción de 5,000 juegos y evitar el bloqueo por parte de Steam (errores `429 Too Many Requests`), el script incorpora un motor de red concurrente de alto rendimiento:
* **Multithreading Cooperativo:** Uso de `ThreadPoolExecutor` con 5 hilos de ejecución concurrente para procesar descargas en paralelo.
* **Controladores de Tasa Globales (`GlobalRateLimiter`):** Una cola coordinada entre hilos que garantiza un intervalo mínimo de 1.8 segundos entre peticiones al Storefront de Steam, evitando colisiones de IP.
* **Filtros de Carga (Payload Optimization):** El script pasa el parámetro `filters=basic,price_overview,genres,categories,achievements` para eliminar un 90% del tamaño de la respuesta (imágenes y videos HD), reduciendo el ancho de banda y latencia.
* **Emulación de Navegador (User-Agent Chrome):** Peticiones firmadas con headers reales para evitar bloqueos automatizados contra clientes scripts por defecto.
* **Auto-Pausa en 429:** En caso de detectar un error 429, todos los hilos se pausan inmediatamente de forma síncrona por 60 segundos por intento para evitar que la IP sea bloqueada de forma prolongada.
