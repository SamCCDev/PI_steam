# Apartado de Ingeniería de Datos I

Este apartado integra cuatro técnicas vistas en la materia **Ingeniería de Datos I**
sobre los datos reales del proyecto (predicción de éxito comercial de videojuegos
en Steam):

1. **Anonimización de datos sensibles con hash SHA-256.**
2. **Análisis con Spark RDD** (conteos y agregaciones).
3. **Calidad de datos con PySpark** (perfilado de nulos y dominios categóricos).
4. **RDD vs DataFrame** (las mismas métricas por ambas APIs, comparadas).

Los ejercicios originales del curso operaban sobre archivos `xlsx`/`csv` de
importaciones de aduanas de Bolivia. **Esos archivos no están en este repositorio**
(dependen de `data/*` que no se versiona), por lo que aquí se aplican las
**mismas técnicas** a nuestros propios datos de Steam. Así el apartado funciona
como evidencia reproducible para el curso y, de paso, alimenta la documentación
del proyecto principal: la anonimización de `developer`/`publisher` permite
estudiar la distribución de estudios y editoras sin exponer sus nombres, y el
perfilado de calidad cuantifica los huecos de metadata que motivaron el embudo
de dos etapas del modelo.

## Estructura

```
ingenieria_datos/
├── README.md                       # este archivo
├── steam_verificar_entorno.py      # (0) verificación del entorno y de los datos
├── steam_anonimizar.py             # (1) anonimización SHA-256 sobre Steam
├── steam_rdd_analisis.py           # (2) análisis Spark RDD (Databricks / pyspark)
├── steam_rdd_analisis_local.py     # (2') verificación local sin Spark (pandas)
├── steam_calidad_datos.py          # (3) perfilado de calidad con PySpark DataFrame
├── steam_calidad_datos_local.py    # (3') verificación local sin Spark (pandas)
├── steam_rdd_vs_dataframe.py       # (4) comparación RDD vs DataFrame (Spark)
├── steam_rdd_vs_dataframe_local.py # (4') línea-a-línea vs pandas (sin Spark)
└── output/
    └── games_metadata_anon.csv     # salida del script de anonimización
```

## Verificación del entorno

**Script:** `steam_verificar_entorno.py` — implementa el `00_verificar_entorno.py`
del curso. Comprueba intérprete (Python ≥ 3.10), librerías (pandas obligatoria,
pyspark opcional) y que los CSV de entrada existan con las columnas esperadas.
Devuelve código de salida 0/1, útil como paso previo en cualquier máquina nueva.

```bash
python ingenieria_datos/steam_verificar_entorno.py
```

## Datos de entrada

Los scripts leen los CSV del proyecto (separador `;`):

- `output/games_metadata.csv` — metadatos por juego (`appid`, `name`, `developer`,
  `publisher`, ...). 32.966 registros.
- `output/games_tags.csv` — géneros y categorías en formato one-hot
  (columnas `genre_*`), una fila por `appid`.

## Técnica 1 — Anonimización SHA-256

**Script:** `steam_anonimizar.py`

Reproduce la función `anonimizar()` del ejercicio de aduanas del curso: aplica
`hashlib.sha256` y toma los primeros 16 caracteres hexadecimales; los valores
vacíos, `None` o `'null'` se convierten en cadena vacía. El hash es determinista (un mismo nombre siempre
produce el mismo identificador) e irreversible.

Sobre los datos de Steam anonimiza las columnas `developer` y `publisher`,
generando las columnas `dev_id` y `pub_id`. El resultado se guarda en
`ingenieria_datos/output/games_metadata_anon.csv`.

**Conexión con el proyecto:** el estudio puede analizar cuántos juegos publica
cada estudio/editora, su tasa de éxito, etc., usando `dev_id`/`pub_id` en lugar
de los nombres, lo que protege el dato y mantiene la trazabilidad por agregación.

Ejecución:

```bash
python ingenieria_datos/steam_anonimizar.py
```

Salida real (resumen):

```
Filas procesadas   : 32.966
Developers únicos  : 25.248
Publishers únicos  : 23.420
Ejemplo: developer "Valve" -> 094db509c61fd2da
```

## Técnica 2 — Análisis con Spark RDD

**Script:** `steam_rdd_analisis.py`

Reproduce el patrón del ejercicio de RDDs del curso
(`sc.textFile` → `map(split)` → conteos y `distinct`) usando exclusivamente
operaciones de RDD: `map`, `filter`, `distinct`, `flatMap`, `reduceByKey`,
`count`, `takeOrdered`. Calcula:

- **(a)** total de registros,
- **(b)** developers únicos,
- **(c)** publishers únicos,
- **(d)** top 10 géneros más frecuentes (a partir de las columnas one-hot
  `genre_*` de `games_tags.csv`, ya que `games_metadata.csv` no tiene columna
  de género).

### Cómo ejecutarlo

- **Databricks Free Edition (recomendado):** Spark viene incluido y es gratis.
  Subir los CSV a DBFS/workspace, ajustar `RUTA_METADATA` y `RUTA_TAGS`, y pegar
  el cuerpo de `analizar(sc, ...)` en una celda. El `SparkContext` ya existe
  como `sc`; no hay que crearlo.
- **Local con pyspark:** `pip install pyspark` (no instalado aquí porque es
  pesado) y luego `python ingenieria_datos/steam_rdd_analisis.py`. El script crea
  un `SparkContext` local automáticamente si detecta pyspark.

### Verificación local sin Spark

**Script:** `steam_rdd_analisis_local.py`

Calcula exactamente las mismas métricas con pandas/Python puro, para tener una
referencia reproducible sin instalar Spark (los operadores RDD se mapean a
`nunique`, filtros y suma de columnas one-hot).

```bash
python ingenieria_datos/steam_rdd_analisis_local.py
```

Salida real:

```
(a) Total de registros : 32.966
(b) Developers únicos  : 25.248
(c) Publishers únicos  : 23.419
(d) Top 10 géneros:
     1. indie                  22.593
     2. action                 13.247
     3. casual                 12.929
     4. adventure              12.460
     5. simulation              7.864
     6. strategy                7.518
     7. rpg                     6.743
     8. early_access            2.886
     9. sports                  1.141
    10. massively_multiplayer   1.035
```

> Nota: el conteo de publishers difiere en 1 entre el script de anonimización
> (23.420) y la verificación local (23.419) por un caso límite de espacios en
> blanco en el nombre; ambos métodos normalizan de forma ligeramente distinta.
> Las cifras son consistentes y la diferencia es despreciable.

## Técnica 3 — Calidad de datos (perfilado con PySpark)

**Scripts:** `steam_calidad_datos.py` (Spark) y `steam_calidad_datos_local.py`
(verificación pandas).

Reproduce el notebook de calidad de datos del curso (`03_calidad_datos.ipynb`),
que perfilaba el CSV de aduanas **sin modificar los datos**: normalización de
encabezados, perfilado de nulos (nulos reales con `isNull()` vs strings vacíos
con `== ''`) y perfilado de columnas categóricas (dominio de valores con
`groupBy().count().orderBy()`). Aquí el mismo perfilado se aplica a
`output/games_metadata.csv`.

> Corrección sobre el original: en el notebook del curso el bloque que imprime
> los resultados quedó fuera del bucle `for`, por lo que solo reportaba la
> última columna. En estos scripts el reporte va dentro del bucle.

```bash
python ingenieria_datos/steam_calidad_datos_local.py   # sin Spark
# o en Databricks: pegar el cuerpo de perfilar() de steam_calidad_datos.py
```

Salida real (resumen):

```
Filas: 32,966 | Columnas: 22

=== NORMALIZACION DE ENCABEZADOS ===
  Sin cambios: el ETL del proyecto ya genera encabezados limpios.

=== PERFILADO DE NULOS ===
Columna                    Nulos reales   Strings vacios
name                           1 ( 0.0%)         0 ( 0.0%)
developer                  3,088 ( 9.4%)         1 ( 0.0%)
publisher                  3,158 ( 9.6%)         5 ( 0.0%)
release_date               3,112 ( 9.4%)         0 ( 0.0%)

=== DOMINIO DE COLUMNAS CATEGORICAS ===
controller_support: none 24,180 | full 8,786
is_free           : 0 29,224 | 1 3,742
platform_windows  : 1 29,899 | 0 3,067
```

**Conexión con el proyecto:** el ~9,4 % de nulos en `developer`/`publisher`/
`release_date` corresponde a los juegos del catálogo sin metadata completa —
exactamente el hueco que motivó filtrar el universo de la etapa 1 a 17.565
juegos (ver la sección de sesgo de recolección en la documentación principal).
El dominio de `controller_support` observado es `{none, full}`: el valor
`partial` que admite la Steam Storefront API no aparece en el dataset.

## Técnica 4 — RDD vs DataFrame

**Scripts:** `steam_rdd_vs_dataframe.py` (Spark) y
`steam_rdd_vs_dataframe_local.py` (equivalente local sin Spark).

Reproduce el notebook `02_rdd_vs_dataframe.ipynb` del curso (y cubre el
ejercicio de conteos de `001_ejercicio.ipynb`): las **mismas tres métricas**
calculadas por los dos caminos — total de registros válidos, developers únicos
y registros inválidos descartados.

- **Vía RDD:** `textFile → filter(header) → map(split) → filter → distinct → count`,
  con parseo y limpieza manuales.
- **Vía DataFrame:** `spark.read.csv(header, sep, inferSchema)` y operaciones de
  columna; el parser CSV resuelve esquema y limpieza.

En la versión local el estilo RDD se traduce a Python puro línea por línea y el
estilo DataFrame a pandas vectorizado:

```bash
python ingenieria_datos/steam_rdd_vs_dataframe_local.py
```

Salida real:

```
Metrica                           linea x linea         pandas
--------------------------------------------------------------
(a) Registros validos                    32,944         32,966
(b) Developers unicos                    25,233         25,248
(c) Registros invalidos                      22              0
Tiempo (s)                                 0.04           0.11

Diferencia de 22 registros: filas con el separador ";" embebido en campos
entrecomillados. El split manual (estilo RDD) las rompe; el parser CSV
(pandas / DataFrame) respeta las comillas RFC-4180.
```

**Hallazgo (la lección central del notebook):** 22 juegos tienen `;` dentro de
campos entrecomillados (p. ej. el juego `"ELIZHA;BETH"` o el estudio
`"While !fun continue;"`). El `split` manual del camino RDD rompe esas filas,
igual que en el ejercicio de aduanas las descripciones con saltos de línea
rompían el parseo; un lector CSV real las maneja sin intervención. Es el
argumento práctico a favor de la API DataFrame para datos tabulares.
