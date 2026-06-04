# steam_anonimizar.py
# Anonimizacion SHA-256 aplicada a los datos de Steam del proyecto.
#
# Curso: Ingenieria de Datos I - 2026
# Origen del patron: reproduce la funcion `anonimizar` y la logica de
#   referencia/01_convertir_anonimizar 1.py (Autor original: William
#   Mauricio Hurtado), que convertia xlsx de importaciones de Bolivia a
#   CSV anonimizado con hash SHA-256.
#
# Como los xlsx de aduanas NO estan en este repo, aplicamos la MISMA
# tecnica a nuestros datos reales: anonimizamos los campos 'developer' y
# 'publisher' de output/games_metadata.csv, generando identificadores
# irreversibles 'dev_id' y 'pub_id'. Esto sirve de evidencia para el curso
# y alimenta la documentacion del proyecto (estudiar la distribucion de
# estudios/editoras sin exponer sus nombres).
#
# Entrada : output/games_metadata.csv          (separador ';')
# Salida  : ingenieria_datos/output/games_metadata_anon.csv (separador ';')

import csv
import hashlib
import os


def anonimizar(valor):
    """Anonimiza un valor sensible usando hash SHA-256.

    Funcion identica a la del archivo de referencia del curso
    (01_convertir_anonimizar 1.py). Convierte el valor a un hash
    irreversible de 16 caracteres. Si el valor es vacio, None o el
    string literal 'null', retorna un string vacio sin aplicar el hash.

    Args:
        valor: El valor a anonimizar (str, int, float o None).

    Returns:
        str: Hash SHA-256 de 16 caracteres si el valor tiene contenido,
             o string vacio '' si el valor es nulo, vacio o 'null'.

    Example:
        >>> anonimizar('Valve')
        '8a2f...'  # 16 hex chars
        >>> anonimizar(None)
        ''
        >>> anonimizar('null')
        ''
    """
    if valor is None or str(valor).strip() == '' or str(valor).lower() == 'null':
        return ''
    return hashlib.sha256(str(valor).encode()).hexdigest()[:16]


# -- Configuracion -----------------------------------------------------
# Rutas relativas a la raiz del repo. El script se ubica en
# ingenieria_datos/, por eso resolvemos contra el directorio padre.
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT       = os.path.dirname(BASE_DIR)
ARCHIVO_ENTRADA = os.path.join(REPO_ROOT, 'output', 'games_metadata.csv')
ARCHIVO_SALIDA  = os.path.join(BASE_DIR, 'output', 'games_metadata_anon.csv')
SEPARADOR       = ';'   # mismo separador de los CSV de Steam

# Columnas con datos a anonimizar -> se agrega una columna *_id con el hash.
# (developer, publisher) son los nombres de estudios/editoras.
COLS_ANONIMIZAR = {
    'developer': 'dev_id',
    'publisher': 'pub_id',
}


def convertir_y_anonimizar(archivo_entrada, archivo_salida, separador, cols_anonimizar):
    """Lee el CSV de Steam y escribe una copia con columnas anonimizadas.

    Para cada columna en `cols_anonimizar` se agrega una nueva columna
    (p.ej. 'dev_id') con el hash SHA-256 del valor original. Las columnas
    originales se conservan para que el material sirva de demostracion del
    mapeo nombre -> hash; en un uso real se descartarian.

    Returns:
        dict: estadisticas del proceso (filas, ejemplos, conjuntos de hash).
    """
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)

    dev_hashes = set()
    pub_hashes = set()
    dev_names  = set()
    pub_names  = set()
    ejemplos   = []
    filas      = 0

    with open(archivo_entrada, 'r', newline='', encoding='utf-8') as fin, \
         open(archivo_salida, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin, delimiter=separador)
        campos_salida = list(reader.fieldnames) + list(cols_anonimizar.values())
        writer = csv.DictWriter(fout, fieldnames=campos_salida, delimiter=separador)
        writer.writeheader()

        for row in reader:
            for col_origen, col_destino in cols_anonimizar.items():
                valor = row.get(col_origen, '')
                row[col_destino] = anonimizar(valor)

            dev = (row.get('developer') or '').strip()
            pub = (row.get('publisher') or '').strip()
            if dev and dev.lower() != 'null':
                dev_names.add(dev)
                dev_hashes.add(row['dev_id'])
            if pub and pub.lower() != 'null':
                pub_names.add(pub)
                pub_hashes.add(row['pub_id'])

            if len(ejemplos) < 5 and dev and pub:
                ejemplos.append((dev, row['dev_id'], pub, row['pub_id']))

            writer.writerow(row)
            filas += 1

    return {
        'filas': filas,
        'developers_unicos': len(dev_names),
        'publishers_unicos': len(pub_names),
        'dev_hashes_unicos': len(dev_hashes),
        'pub_hashes_unicos': len(pub_hashes),
        'ejemplos': ejemplos,
    }


if __name__ == '__main__':
    print('Anonimizacion SHA-256 de datos de Steam')
    print(f'  Entrada : {ARCHIVO_ENTRADA}')
    print(f'  Salida  : {ARCHIVO_SALIDA}')
    print(f'  Columnas anonimizadas: {COLS_ANONIMIZAR}')
    print()

    stats = convertir_y_anonimizar(
        ARCHIVO_ENTRADA,
        ARCHIVO_SALIDA,
        SEPARADOR,
        COLS_ANONIMIZAR,
    )

    print('Estadisticas:')
    print(f'  Filas procesadas        : {stats["filas"]:,}')
    print(f'  Developers unicos        : {stats["developers_unicos"]:,}'
          f'  (hashes distintos: {stats["dev_hashes_unicos"]:,})')
    print(f'  Publishers unicos        : {stats["publishers_unicos"]:,}'
          f'  (hashes distintos: {stats["pub_hashes_unicos"]:,})')
    print()
    print('  Ejemplos nombre -> hash (16 chars):')
    for dev, dev_h, pub, pub_h in stats['ejemplos']:
        print(f'    developer "{dev}" -> {dev_h}')
        print(f'    publisher "{pub}" -> {pub_h}')
    print()
    print(f'OK Archivo anonimizado guardado en: {ARCHIVO_SALIDA}')
