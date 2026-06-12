# steam_rdd_vs_dataframe_local.py
# Verificacion local (sin Spark) de steam_rdd_vs_dataframe.py.
#
# Curso: Ingenieria de Datos I - 2026
# La comparacion RDD vs DataFrame se traduce al mundo local asi:
#   - estilo RDD       -> Python puro linea por linea (open + split + set),
#                         igual que las lambdas de un RDD
#   - estilo DataFrame -> pandas vectorizado (read_csv + nunique),
#                         igual que el plan optimizado de un DataFrame
#
# Las metricas son las mismas del script Spark:
#   (a) total de registros validos
#   (b) developers unicos
#   (c) registros invalidos descartados (tokens != n columnas)

import os
import time
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(BASE_DIR)
RUTA_METADATA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
SEPARADOR     = ';'


def _es_valido(valor):
    """Filtro comun: descarta vacios y 'null' (mismo criterio del script Spark)."""
    v = valor.strip()
    return v != '' and v.lower() != 'null'


def estilo_rdd(ruta, sep):
    """Python puro linea por linea: el equivalente local de la API RDD."""
    inicio = time.time()

    total = 0
    invalidos = 0
    developers = set()

    with open(ruta, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')          # raw.first()
        cols = header.split(sep)
        n_cols = len(cols)
        idx_dev = cols.index('developer')

        for linea in f:                              # map(split) + filter
            tokens = linea.rstrip('\n').split(sep)
            if len(tokens) != n_cols:
                invalidos += 1
                continue
            total += 1
            dev = tokens[idx_dev]
            if _es_valido(dev):
                developers.add(dev.strip())          # distinct()

    segundos = time.time() - inicio
    return {'total': total, 'developers': len(developers),
            'invalidos': invalidos, 'segundos': segundos}


def estilo_dataframe(ruta, sep):
    """pandas vectorizado: el equivalente local de la API DataFrame."""
    inicio = time.time()

    df = pd.read_csv(ruta, sep=sep, dtype=str)

    total_leidas = len(df)
    df_valido = df[df['appid'].notna()]
    total = len(df_valido)
    invalidos = total_leidas - total

    dev = df_valido['developer'].fillna('').str.strip()
    developers = dev[(dev != '') & (dev.str.lower() != 'null')].nunique()

    segundos = time.time() - inicio
    return {'total': total, 'developers': developers,
            'invalidos': invalidos, 'segundos': segundos}


if __name__ == '__main__':
    print('=== RDD vs DataFrame (verificacion local sin Spark) ===\n')

    r = estilo_rdd(RUTA_METADATA, SEPARADOR)
    d = estilo_dataframe(RUTA_METADATA, SEPARADOR)

    print(f'{"Metrica":<32} {"linea x linea":>14} {"pandas":>14}')
    print('-' * 62)
    print(f'{"(a) Registros validos":<32} {r["total"]:>14,} {d["total"]:>14,}')
    print(f'{"(b) Developers unicos":<32} {r["developers"]:>14,} {d["developers"]:>14,}')
    print(f'{"(c) Registros invalidos":<32} {r["invalidos"]:>14,} {d["invalidos"]:>14,}')
    print(f'{"Tiempo (s)":<32} {r["segundos"]:>14.2f} {d["segundos"]:>14.2f}')

    coinciden = (r['total'] == d['total']
                 and r['developers'] == d['developers'])
    print()
    if coinciden:
        print('Resultados identicos por ambos caminos.')
    else:
        # Hallazgo esperado (la leccion central del notebook del curso):
        # el split manual rompe las filas cuyos campos entrecomillados
        # contienen el separador (p.ej. name = "ELIZHA;BETH"); un parser
        # CSV real respeta las comillas y las lee bien.
        diff = d['total'] - r['total']
        print(f'Diferencia de {diff:,} registros: filas con el separador '
              f'";" embebido en campos entrecomillados.')
        print('El split manual (estilo RDD) las rompe; el parser CSV '
              '(pandas / DataFrame) respeta las comillas RFC-4180.')
