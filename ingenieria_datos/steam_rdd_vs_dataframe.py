# steam_rdd_vs_dataframe.py
# Comparacion RDD vs DataFrame en Apache Spark sobre los datos de Steam.
#
# Curso: Ingenieria de Datos I - 2026
# Origen del patron: reproduce el notebook RDD vs DataFrame del curso
#   (02_rdd_vs_dataframe.ipynb), que calculaba las mismas metricas sobre el
#   CSV de aduanas con ambas APIs para contrastar ergonomia y resultado.
#   Tambien cubre el ejercicio de conteo de registros y valores unicos
#   (001_ejercicio.ipynb: total + polizas unicas via map/distinct/count).
#
# Las MISMAS tres metricas se calculan dos veces:
#   (a) total de registros validos
#   (b) developers unicos
#   (c) registros invalidos descartados (tokens != n columnas)
#
#   Via RDD       : textFile -> filter(header) -> map(split) -> filter ->
#                   map -> distinct -> count   (parseo y limpieza manuales)
#   Via DataFrame : spark.read.csv(header, sep, inferSchema) -> dropna/
#                   filter -> distinct -> count (el parser resuelve todo)
#
# Ademas de la ergonomia y el tiempo (Catalyst optimiza el plan del
# DataFrame; el RDD ejecuta las lambdas tal cual), la comparacion expone
# la leccion del curso: el split manual rompe las filas con el separador
# embebido en campos entrecomillados; el lector CSV las maneja bien.
#
# COMO EJECUTAR:
#   - Databricks Free Edition: `spark` y `sc` ya existen. Ajustar
#     RUTA_METADATA y pegar el cuerpo de comparar() en una celda.
#   - Local: requiere `pip install pyspark`. Para la verificacion sin Spark
#     usar steam_rdd_vs_dataframe_local.py (Python puro vs pandas).

import os
import time

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(BASE_DIR)
RUTA_METADATA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
SEPARADOR     = ';'


def _es_valido(valor):
    """Filtro comun de ambos caminos: descarta vacios y 'null'."""
    v = valor.strip()
    return v != '' and v.lower() != 'null'


def via_rdd(sc, ruta, sep):
    """Camino 1: API RDD - parseo y limpieza manuales linea por linea."""
    inicio = time.time()

    raw = sc.textFile(ruta)
    header = raw.first()                      # primera linea = encabezado
    cols = header.split(sep)
    n_cols = len(cols)
    idx_dev = cols.index('developer')

    tokens = (raw
              .filter(lambda line: line != header)
              .map(lambda line: line.split(sep)))
    tokens.cache()

    # Limpieza manual: una fila es valida si el split produjo exactamente
    # n_cols tokens (mismo criterio del notebook del curso para detectar
    # filas rotas por saltos de linea o separadores embebidos).
    validos   = tokens.filter(lambda t: len(t) == n_cols)
    invalidos = tokens.count() - validos.count()

    total = validos.count()
    developers = (validos
                  .map(lambda t: t[idx_dev])
                  .filter(_es_valido)
                  .distinct()
                  .count())

    segundos = time.time() - inicio
    return {'total': total, 'developers': developers,
            'invalidos': invalidos, 'segundos': segundos}


def via_dataframe(spark, ruta, sep):
    """Camino 2: API DataFrame - el parser CSV resuelve esquema y limpieza."""
    from pyspark.sql import functions as F

    inicio = time.time()

    df = (spark.read
          .option('header', True)
          .option('sep', sep)
          .option('inferSchema', True)
          .option('mode', 'PERMISSIVE')
          .csv(ruta))

    # El parser ya alineo cada fila al esquema: una fila "rota" queda con
    # la columna clave (appid) en nulo, no hace falta contar tokens a mano.
    total_leidas = df.count()
    df_valido = df.filter(F.col('appid').isNotNull())
    total = df_valido.count()
    invalidos = total_leidas - total

    developers = (df_valido
                  .select(F.trim(F.col('developer')).alias('developer'))
                  .filter((F.col('developer') != '') &
                          (F.lower(F.col('developer')) != 'null'))
                  .distinct()
                  .count())

    segundos = time.time() - inicio
    return {'total': total, 'developers': developers,
            'invalidos': invalidos, 'segundos': segundos}


def comparar(spark, sc, ruta_metadata=RUTA_METADATA, sep=SEPARADOR):
    """Corre ambos caminos y muestra la tabla comparativa."""
    print('=== RDD vs DataFrame - datos de Steam ===\n')

    r = via_rdd(sc, ruta_metadata, sep)
    d = via_dataframe(spark, ruta_metadata, sep)

    print(f'{"Metrica":<32} {"RDD":>14} {"DataFrame":>14}')
    print('-' * 62)
    print(f'{"(a) Registros validos":<32} {r["total"]:>14,} {d["total"]:>14,}')
    print(f'{"(b) Developers unicos":<32} {r["developers"]:>14,} {d["developers"]:>14,}')
    print(f'{"(c) Registros invalidos":<32} {r["invalidos"]:>14,} {d["invalidos"]:>14,}')
    print(f'{"Tiempo (s)":<32} {r["segundos"]:>14.1f} {d["segundos"]:>14.1f}')

    coinciden = (r['total'] == d['total']
                 and r['developers'] == d['developers'])
    print()
    if coinciden:
        print('Resultados identicos por ambos caminos.')
    else:
        # Hallazgo esperado (la leccion central del notebook del curso):
        # textFile + split rompe las filas cuyos campos entrecomillados
        # contienen el separador (p.ej. name = "ELIZHA;BETH"); el lector
        # CSV del DataFrame respeta las comillas y las lee bien. En la
        # verificacion local son 22 filas (32.944 vs 32.966).
        diff = d['total'] - r['total']
        print(f'Diferencia de {diff:,} registros: filas con el separador '
              f'";" embebido en campos entrecomillados.')
        print('El split manual del RDD las rompe; el lector CSV del '
              'DataFrame respeta las comillas RFC-4180.')

    return {'rdd': r, 'dataframe': d, 'coinciden': coinciden}


if __name__ == '__main__':
    # En Databricks `spark` y `sc` ya existen; localmente los creamos.
    try:
        spark, sc  # type: ignore[name-defined]  # provistos por Databricks
        print('Usando SparkSession/SparkContext existentes (Databricks).')
    except NameError:
        try:
            from pyspark.sql import SparkSession
            spark = (SparkSession.builder
                     .appName('steamRDDvsDataFrame')
                     .getOrCreate())
            sc = spark.sparkContext
            sc.setLogLevel('WARN')
            print('SparkSession local creada con pyspark.')
        except ImportError:
            raise SystemExit(
                'pyspark no esta instalado. Ejecuta este script en Databricks '
                '(Spark nativo) o instala pyspark localmente. Para la '
                'verificacion sin Spark usa steam_rdd_vs_dataframe_local.py.'
            )

    comparar(spark, sc)
