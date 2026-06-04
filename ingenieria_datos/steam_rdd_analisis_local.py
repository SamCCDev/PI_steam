# steam_rdd_analisis_local.py
# Verificacion local (sin Spark) de las metricas calculadas en
# steam_rdd_analisis.py. Usa pandas/Python puro para reproducir los mismos
# numeros de forma deterministica y sin instalar pyspark.
#
# Curso: Ingenieria de Datos I - 2026
# Las operaciones replican el comportamiento de los RDDs:
#   - distinct()        -> set / nunique()
#   - filter()          -> filtrado de vacios y 'null'
#   - reduceByKey()/sum -> suma de columnas one-hot genre_*
#   - takeOrdered()     -> ordenar y tomar top 10
#
# Metricas:
#   (a) total de registros
#   (b) developers unicos
#   (c) publishers unicos
#   (d) top 10 generos mas frecuentes

import os
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(BASE_DIR)
RUTA_METADATA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
RUTA_TAGS     = os.path.join(REPO_ROOT, 'output', 'games_tags.csv')
SEPARADOR     = ';'


def _es_valido(serie):
    """Equivalente al filter() de los RDDs: descarta vacios, NaN y 'null'."""
    s = serie.fillna('').astype(str).str.strip()
    return s[(s != '') & (s.str.lower() != 'null')]


def analizar(ruta_metadata=RUTA_METADATA, ruta_tags=RUTA_TAGS, sep=SEPARADOR):
    meta = pd.read_csv(ruta_metadata, sep=sep, dtype=str)

    # (a) Total de registros
    total_registros = len(meta)

    # (b) Developers unicos (distinct sobre valores validos)
    developers_unicos = _es_valido(meta['developer']).nunique()

    # (c) Publishers unicos
    publishers_unicos = _es_valido(meta['publisher']).nunique()

    # (d) Top 10 generos: suma de columnas one-hot genre_* en games_tags.csv
    tags = pd.read_csv(ruta_tags, sep=sep)
    genre_cols = [c for c in tags.columns if c.startswith('genre_')]
    sumas = (tags[genre_cols]
             .apply(pd.to_numeric, errors='coerce')
             .fillna(0)
             .sum()
             .astype(int))
    sumas.index = [c.replace('genre_', '') for c in sumas.index]
    top10_generos = list(sumas.sort_values(ascending=False).head(10).items())

    print('=== Verificacion local (sin Spark) - datos de Steam ===')
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
    analizar()
