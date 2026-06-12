# steam_calidad_datos_local.py
# Verificacion local (sin Spark) del perfilado de calidad de datos que hace
# steam_calidad_datos.py. Usa pandas para reproducir los mismos numeros.
#
# Curso: Ingenieria de Datos I - 2026
# Equivalencias entre la API DataFrame de Spark y pandas:
#   - F.col(c).isNull()          -> df[c].isna()
#   - F.trim(F.col(c)) == ''     -> str.strip() == ''
#   - groupBy(c).count().orderBy -> value_counts() (ya ordena descendente)
#
# Nota: al leer un CSV, tanto Spark como pandas convierten el campo vacio en
# nulo (None/NaN), asi que "nulos reales" aqui equivale a los isNull() de
# Spark. Los "strings vacios" detectables son los que quedan en blanco tras
# strip() (espacios), que el parser no convierte en nulo.
#
# Metricas:
#   (1) normalizacion de encabezados (puntos y espacios)
#   (2) nulos reales y strings vacios por columna
#   (3) dominio de valores de las columnas categoricas

import os
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(BASE_DIR)
RUTA_METADATA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
SEPARADOR     = ';'

COLUMNAS_CATEGORICAS = ['controller_support', 'is_free',
                        'platform_windows', 'platform_mac', 'platform_linux']


def perfilar(ruta_metadata=RUTA_METADATA, sep=SEPARADOR):
    # dtype=str preserva los valores tal cual estan en el CSV (sin inferir
    # tipos), que es lo que necesita un perfilado: ver el dato crudo.
    df = pd.read_csv(ruta_metadata, sep=sep, dtype=str)
    total_registros = len(df)
    print(f'Filas: {total_registros:,} | Columnas: {len(df.columns)}')

    # ---- 1. Normalizacion de encabezados --------------------------------
    nuevos = [c.replace('.', '').replace(' ', '_') for c in df.columns]
    cambiadas = [(v, n) for v, n in zip(df.columns, nuevos) if v != n]
    df.columns = nuevos
    print('\n=== NORMALIZACION DE ENCABEZADOS ===')
    if cambiadas:
        for viejo, nuevo in cambiadas:
            print(f'  "{viejo}" -> "{nuevo}"')
    else:
        print('  Sin cambios: el ETL del proyecto ya genera encabezados '
              'limpios (snake_case, sin puntos ni espacios).')

    # ---- 2. Perfilado de NULOS ------------------------------------------
    print('\n=== PERFILADO DE NULOS ===')
    print(f'{"Columna":<22} {"Nulos reales":>16} {"Strings vacios":>16}')
    print('-' * 56)

    columnas_con_problemas = 0
    for columna in df.columns:
        nulos_reales = int(df[columna].isna().sum())
        sin_nulos = df[columna].dropna()
        strings_vacios = int((sin_nulos.str.strip() == '').sum())

        if nulos_reales > 0 or strings_vacios > 0:
            columnas_con_problemas += 1
            pct_nulos  = nulos_reales / total_registros * 100
            pct_vacios = strings_vacios / total_registros * 100
            print(f'{columna:<22} {nulos_reales:>9,} ({pct_nulos:4.1f}%)'
                  f' {strings_vacios:>9,} ({pct_vacios:4.1f}%)')

    if columnas_con_problemas == 0:
        print('Sin nulos ni strings vacios en ninguna columna.')

    # ---- 3. Dominio de columnas categoricas ------------------------------
    print('\n=== DOMINIO DE COLUMNAS CATEGORICAS ===')
    for columna in COLUMNAS_CATEGORICAS:
        if columna not in df.columns:
            print(f'\n-- {columna} -- (no existe en el CSV)')
            continue
        print(f'\n-- {columna} --')
        conteos = df[columna].value_counts(dropna=False).head(10)
        for valor, n in conteos.items():
            etiqueta = '(nulo)' if pd.isna(valor) else valor
            print(f'  {etiqueta:<12} {n:>8,}')

    return {
        'total_registros': total_registros,
        'columnas': len(df.columns),
        'encabezados_cambiados': cambiadas,
        'columnas_con_problemas': columnas_con_problemas,
    }


if __name__ == '__main__':
    print('=== Perfilado de calidad local (sin Spark) - datos de Steam ===\n')
    perfilar()
