# steam_rdd_analisis.py
# Analisis con Spark RDD aplicado a los datos de Steam del proyecto.
#
# Curso: Ingenieria de Datos I - 2026
# Origen del patron: reproduce el ejercicio de RDDs del curso
#   (sc.textFile -> split por separador -> conteos y distinct), que
#   contaba registros y polizas unicas de las importaciones de Bolivia.
#   Aqui aplicamos las MISMAS operaciones RDD a nuestros datos de Steam.
#
# COMO EJECUTAR:
#   - Databricks Free Edition: Spark viene nativo y gratis. Subir el CSV al
#     workspace/DBFS, ajustar las rutas RUTA_METADATA / RUTA_TAGS y pegar el
#     cuerpo de main() en una celda (o ejecutar este archivo como Job).
#     El SparkContext ya existe como `sc`, no hace falta crearlo.
#   - Local: requiere `pip install pyspark` (NO instalado en este entorno;
#     es pesado). Si esta disponible, `python steam_rdd_analisis.py` crea un
#     SparkContext local automaticamente.
#
# Metricas (operaciones RDD: map/filter/distinct/reduceByKey/countByValue):
#   (a) total de registros
#   (b) developers unicos
#   (c) publishers unicos
#   (d) top 10 generos mas frecuentes
#
# Nota de datos: los generos no estan en games_metadata.csv; viven en
# output/games_tags.csv como columnas one-hot 'genre_*' por appid. Por eso
# el top de generos se calcula sobre ese segundo archivo.

import os

# Rutas relativas a la raiz del repo (este script vive en ingenieria_datos/).
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(BASE_DIR)
RUTA_METADATA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
RUTA_TAGS     = os.path.join(REPO_ROOT, 'output', 'games_tags.csv')
SEPARADOR     = ';'


def analizar(sc, ruta_metadata=RUTA_METADATA, ruta_tags=RUTA_TAGS, sep=SEPARADOR):
    """Ejecuta el analisis RDD sobre los CSV de Steam.

    Args:
        sc: SparkContext ya inicializado (en Databricks es la variable `sc`).
        ruta_metadata: ruta al CSV games_metadata.csv.
        ruta_tags: ruta al CSV games_tags.csv (generos one-hot).
        sep: separador de columnas de los CSV.
    """
    # ---- Lectura y parseo (igual que el ejercicio del curso) -----------
    # sc.textFile -> cada linea es un string; luego split por el separador.
    raw = sc.textFile(ruta_metadata)
    header = raw.first()                 # primera linea = encabezado
    cols = header.split(sep)
    idx_dev = cols.index('developer')
    idx_pub = cols.index('publisher')

    # RDD de filas ya tokenizadas, excluyendo el encabezado.
    rows = (raw
            .filter(lambda line: line != header)
            .map(lambda line: line.split(sep)))
    rows.cache()

    # (a) Total de registros -> count()
    total_registros = rows.count()

    # (b) Developers unicos -> map al campo, filtrar vacios/null, distinct
    developers_unicos = (rows
                         .map(lambda c: c[idx_dev].strip() if len(c) > idx_dev else '')
                         .filter(lambda v: v != '' and v.lower() != 'null')
                         .distinct()
                         .count())

    # (c) Publishers unicos -> mismo patron
    publishers_unicos = (rows
                         .map(lambda c: c[idx_pub].strip() if len(c) > idx_pub else '')
                         .filter(lambda v: v != '' and v.lower() != 'null')
                         .distinct()
                         .count())

    # ---- (d) Top 10 generos (games_tags.csv, columnas genre_* one-hot) --
    raw_tags = sc.textFile(ruta_tags)
    header_tags = raw_tags.first()
    cols_tags = header_tags.split(sep)
    genre_cols = [(i, c) for i, c in enumerate(cols_tags) if c.startswith('genre_')]

    def contar_generos(tokens):
        # Por fila emite (nombre_genero, 1) por cada columna genre_* == '1'.
        out = []
        for i, name in genre_cols:
            if i < len(tokens) and tokens[i].strip() == '1':
                out.append((name.replace('genre_', ''), 1))
        return out

    generos_rdd = (raw_tags
                   .filter(lambda line: line != header_tags)
                   .map(lambda line: line.split(sep))
                   .flatMap(contar_generos)
                   .reduceByKey(lambda a, b: a + b))   # agregacion por clave

    top10_generos = generos_rdd.takeOrdered(10, key=lambda kv: -kv[1])

    # ---- Salida --------------------------------------------------------
    print('=== Analisis RDD de datos de Steam (Spark) ===')
    print(f'(a) Total de registros   : {total_registros:,}')
    print(f'(b) Developers unicos    : {developers_unicos:,}')
    print(f'(c) Publishers unicos    : {publishers_unicos:,}')
    print('(d) Top 10 generos mas frecuentes:')
    for rank, (genero, n) in enumerate(top10_generos, 1):
        print(f'    {rank:2d}. {genero:<22} {n:,}')

    return {
        'total_registros': total_registros,
        'developers_unicos': developers_unicos,
        'publishers_unicos': publishers_unicos,
        'top10_generos': top10_generos,
    }


if __name__ == '__main__':
    # En Databricks `sc` ya existe; localmente intentamos crearlo con pyspark.
    try:
        sc  # type: ignore[name-defined]  # provisto por Databricks
        print('Usando SparkContext existente (Databricks).')
    except NameError:
        try:
            from pyspark import SparkContext
            sc = SparkContext(master='local[*]', appName='steamRDDAnalisis')
            sc.setLogLevel('WARN')
            print('SparkContext local creado con pyspark.')
        except ImportError:
            raise SystemExit(
                'pyspark no esta instalado. Ejecuta este script en Databricks '
                '(Spark nativo) o instala pyspark localmente. Para verificacion '
                'sin Spark usa steam_rdd_analisis_local.py.'
            )

    analizar(sc)
