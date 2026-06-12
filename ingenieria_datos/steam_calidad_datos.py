# steam_calidad_datos.py
# Perfilado de calidad de datos con PySpark DataFrame sobre los datos de Steam.
#
# Curso: Ingenieria de Datos I - 2026
# Origen del patron: reproduce el notebook de calidad de datos del curso
#   (03_calidad_datos.ipynb), que perfilaba el CSV de importaciones de
#   aduanas: normalizacion de encabezados, perfilado de nulos (nulos reales
#   vs strings vacios) y perfilado de columnas categoricas (dominio de
#   valores via groupBy + count). Aqui aplicamos el MISMO perfilado a
#   output/games_metadata.csv, SIN modificar los datos (solo lectura).
#
# Nota sobre el original: en el notebook del curso el bloque que imprime los
# resultados quedo fuera del bucle `for`, por lo que solo reportaba la ultima
# columna. Aqui el reporte va dentro del bucle y cubre todas las columnas.
#
# COMO EJECUTAR:
#   - Databricks Free Edition: la SparkSession ya existe como `spark`.
#     Subir el CSV, ajustar RUTA_METADATA y pegar el cuerpo de perfilar()
#     en una celda (o ejecutar este archivo como Job).
#   - Local: requiere `pip install pyspark` (NO instalado en este entorno;
#     es pesado). Para la verificacion sin Spark usar
#     steam_calidad_datos_local.py.

import os
import time

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(BASE_DIR)
RUTA_METADATA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
SEPARADOR     = ';'

# Columnas cuyo contenido debe pertenecer a un dominio cerrado de valores.
# controller_support: none/partial/full; is_free y platform_*: 0/1.
COLUMNAS_CATEGORICAS = ['controller_support', 'is_free',
                        'platform_windows', 'platform_mac', 'platform_linux']


def normalizar_encabezados(df):
    """Quita puntos y reemplaza espacios por '_' en los nombres de columna.

    Misma transformacion del ejercicio de aduanas (alli los encabezados
    traian puntos y espacios, p.ej. 'Nro. Registro' -> 'Nro_Registro').
    Devuelve el DataFrame renombrado y la lista de columnas que cambiaron.
    """
    nuevos = [c.replace('.', '').replace(' ', '_') for c in df.columns]
    cambiadas = [(viejo, nuevo) for viejo, nuevo in zip(df.columns, nuevos)
                 if viejo != nuevo]
    return df.toDF(*nuevos), cambiadas


def perfilar(spark, ruta_metadata=RUTA_METADATA, sep=SEPARADOR):
    """Perfila la calidad de games_metadata.csv con la API DataFrame.

    Args:
        spark: SparkSession ya inicializada (en Databricks es `spark`).
        ruta_metadata: ruta al CSV games_metadata.csv.
        sep: separador de columnas del CSV.
    """
    from pyspark.sql import functions as F

    # ---- 1. Carga con esquema inferido (medimos el tiempo de lectura) ---
    inicio = time.time()
    df = (spark.read
          .option('header', True)
          .option('sep', sep)
          .option('inferSchema', True)
          .csv(ruta_metadata))
    total_registros = df.count()
    print(f'CSV leido en {time.time() - inicio:.1f}s')
    print(f'Filas: {total_registros:,} | Columnas: {len(df.columns)}')
    print('\nEsquema inferido:')
    df.printSchema()

    # ---- 2. Normalizacion de encabezados -------------------------------
    df, cambiadas = normalizar_encabezados(df)
    print('=== NORMALIZACION DE ENCABEZADOS ===')
    if cambiadas:
        for viejo, nuevo in cambiadas:
            print(f'  "{viejo}" -> "{nuevo}"')
    else:
        print('  Sin cambios: el ETL del proyecto ya genera encabezados '
              'limpios (snake_case, sin puntos ni espacios).')

    # ---- 3. Perfilado de NULOS ------------------------------------------
    # Tipo 1: nulos reales (None)   -> isNull()
    # Tipo 2: strings vacios ('')   -> == ''  (solo aplica a columnas string)
    print('\n=== PERFILADO DE NULOS ===')
    print(f'{"Columna":<22} {"Nulos reales":>16} {"Strings vacios":>16}')
    print('-' * 56)

    tipos_columnas = dict(df.dtypes)
    columnas_con_problemas = 0
    for columna in df.columns:
        nulos_reales = df.filter(F.col(columna).isNull()).count()

        strings_vacios = 0
        if tipos_columnas[columna] == 'string':
            strings_vacios = df.filter(F.trim(F.col(columna)) == '').count()

        if nulos_reales > 0 or strings_vacios > 0:
            columnas_con_problemas += 1
            pct_nulos  = nulos_reales / total_registros * 100
            pct_vacios = strings_vacios / total_registros * 100
            print(f'{columna:<22} {nulos_reales:>9,} ({pct_nulos:4.1f}%)'
                  f' {strings_vacios:>9,} ({pct_vacios:4.1f}%)')

    if columnas_con_problemas == 0:
        print('Sin nulos ni strings vacios en ninguna columna.')

    # ---- 4. Perfilado de columnas categoricas ---------------------------
    # El dominio observado debe coincidir con el esperado; un valor fuera
    # de dominio (p.ej. un cuarto valor en controller_support) es un
    # problema de calidad.
    print('\n=== DOMINIO DE COLUMNAS CATEGORICAS ===')
    for columna in COLUMNAS_CATEGORICAS:
        if columna not in df.columns:
            print(f'\n-- {columna} -- (no existe en el CSV)')
            continue
        print(f'\n-- {columna} --')
        (df.groupBy(columna)
           .count()
           .orderBy('count', ascending=False)
           .show(10, False))

    return {
        'total_registros': total_registros,
        'columnas': len(df.columns),
        'encabezados_cambiados': cambiadas,
        'columnas_con_problemas': columnas_con_problemas,
    }


if __name__ == '__main__':
    # En Databricks `spark` ya existe; localmente intentamos crearla.
    try:
        spark  # type: ignore[name-defined]  # provista por Databricks
        print('Usando SparkSession existente (Databricks).')
    except NameError:
        try:
            from pyspark.sql import SparkSession
            spark = (SparkSession.builder
                     .appName('steamCalidadDatos')
                     .getOrCreate())
            spark.sparkContext.setLogLevel('WARN')
            print('SparkSession local creada con pyspark.')
        except ImportError:
            raise SystemExit(
                'pyspark no esta instalado. Ejecuta este script en Databricks '
                '(Spark nativo) o instala pyspark localmente. Para la '
                'verificacion sin Spark usa steam_calidad_datos_local.py.'
            )

    perfilar(spark)
