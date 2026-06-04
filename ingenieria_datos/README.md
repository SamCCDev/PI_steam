# Apartado de Ingeniería de Datos I

Este apartado integra dos técnicas vistas en la materia **Ingeniería de Datos I**
sobre los datos reales del proyecto (predicción de éxito comercial de videojuegos
en Steam):

1. **Anonimización de datos sensibles con hash SHA-256.**
2. **Análisis con Spark RDD** (conteos y agregaciones).

Los ejercicios originales del curso operaban sobre archivos `xlsx` de
importaciones de aduanas de Bolivia. **Esos archivos no están en este repositorio**
(dependen de `data/*.xlsx` que no se versionan), por lo que aquí se aplican las
**mismas técnicas** a nuestros propios datos de Steam. Así el apartado funciona
como evidencia reproducible para el curso y, de paso, alimenta la documentación
del proyecto principal: la anonimización de `developer`/`publisher` permite
estudiar la distribución de estudios y editoras sin exponer sus nombres.

## Estructura

```
ingenieria_datos/
├── README.md                       # este archivo
├── steam_anonimizar.py             # (1) anonimización SHA-256 sobre Steam
├── steam_rdd_analisis.py           # (2) análisis Spark RDD (Databricks / pyspark)
├── steam_rdd_analisis_local.py     # (2') verificación local sin Spark (pandas)
├── output/
│   └── games_metadata_anon.csv     # salida del script de anonimización
└── referencia/                     # material original del curso (intacto)
    ├── 01_convertir_anonimizar 1.py
    └── 01_ejercicio (1).ipynb
```

La carpeta `referencia/` conserva los dos archivos del docente sin modificar, como
fuente del patrón que se reproduce.

## Datos de entrada

Los scripts leen los CSV del proyecto (separador `;`):

- `output/games_metadata.csv` — metadatos por juego (`appid`, `name`, `developer`,
  `publisher`, ...). 32.966 registros.
- `output/games_tags.csv` — géneros y categorías en formato one-hot
  (columnas `genre_*`), una fila por `appid`.

## Técnica 1 — Anonimización SHA-256

**Script:** `steam_anonimizar.py`

Reproduce la función `anonimizar()` del archivo de referencia
`referencia/01_convertir_anonimizar 1.py`: aplica `hashlib.sha256` y toma los
primeros 16 caracteres hexadecimales; los valores vacíos, `None` o `'null'` se
convierten en cadena vacía. El hash es determinista (un mismo nombre siempre
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

Reproduce el patrón del ejercicio `referencia/01_ejercicio (1).ipynb`
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
