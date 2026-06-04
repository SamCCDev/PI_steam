# 01_convertir_anonimizar.py
# Pipeline de conversión y anonimización: xlsx → CSV
# Ejecutar UNA SOLA VEZ por archivo fuente
#
# Autor: William Mauricio Hurtado
# Curso: Ingeniería de Datos I — 2026
# Descripción: Convierte archivos Excel de importaciones de Bolivia
#              a CSV anonimizado, eliminando o hasheando datos sensibles.

import openpyxl
import csv
import hashlib
import time


def anonimizar(valor):
    """Anonimiza un valor sensible usando hash SHA-256.

    Convierte el valor a un hash irreversible de 16 caracteres.
    Si el valor es vacío, None o el string literal 'null',
    retorna un string vacío sin aplicar el hash.

    Args:
        valor: El valor a anonimizar. Puede ser cualquier tipo
               (str, int, float, None) proveniente de una celda Excel.

    Returns:
        str: Hash SHA-256 de 16 caracteres si el valor tiene contenido,
             o string vacío '' si el valor es nulo, vacío o 'null'.

    Example:
        >>> anonimizar('CHOQUE CRUZ FELIX')
        'ce094d28817d25c5'
        >>> anonimizar(None)
        ''
        >>> anonimizar('null')
        ''
    """
    if valor is None or str(valor).strip() == '' or str(valor).lower() == 'null':
        return ''
    return hashlib.sha256(str(valor).encode()).hexdigest()[:16]


def procesar_fila(row, headers, keep_idx, cols_anonimizar):
    """Procesa una fila del Excel aplicando las reglas de anonimización.

    Para cada columna de la fila decide si anonimizar, normalizar
    a vacío, o copiar el valor tal cual.

    Args:
        row (tuple): Fila del Excel con valores de cada celda.
        headers (list): Lista de nombres de columnas.
        keep_idx (list): Índices de columnas que se conservan en el CSV.
        cols_anonimizar (set): Nombres de columnas que requieren hash.

    Returns:
        list: Lista de strings con los valores procesados de la fila,
              listos para escribir en el CSV.
    """
    fila = []
    for j in keep_idx:
        col = headers[j]
        val = row[j]
        if col in cols_anonimizar:
            fila.append(anonimizar(val))
        elif val is None or str(val).strip().lower() == 'null':
            fila.append('')   # normalizar nulos reales y string 'null'
        else:
            fila.append(str(val))
    return fila


def convertir_y_anonimizar(archivo_entrada, archivo_salida, separador, cols_anonimizar, cols_eliminar):
    """Convierte un archivo Excel a CSV anonimizado.

    Lee el archivo Excel fila por fila, aplica las reglas de
    anonimización y escribe el resultado en un CSV con separador pipe.
    Imprime el progreso cada 50,000 filas.

    Args:
        archivo_entrada (str): Ruta al archivo .xlsx de origen.
        archivo_salida (str): Ruta donde se guardará el CSV resultante.
        separador (str): Carácter separador del CSV. Se recomienda '|'
                         para evitar conflictos con comas en texto libre.
        cols_anonimizar (set): Nombres de columnas a reemplazar con hash.
        cols_eliminar (set): Nombres de columnas a excluir del CSV.

    Returns:
        tuple: (filas_procesadas, tiempo_segundos) con el total de filas
               procesadas y el tiempo de ejecución en segundos.
    """
    start = time.time()
    wb = openpyxl.load_workbook(archivo_entrada, read_only=True)
    ws = wb.active

    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=separador)
        headers = None
        keep_idx = []
        filas_procesadas = 0

        for i, row in enumerate(ws.iter_rows(values_only=True)):

            # Primera fila: procesar encabezados
            if i == 0:
                headers = [str(h) if h is not None else 'SIN_NOMBRE'
                           for h in row]
                keep_idx = [j for j, h in enumerate(headers)
                            if h not in cols_eliminar]
                writer.writerow([headers[j] for j in keep_idx])
                continue

            # Filas de datos: procesar y escribir
            fila = procesar_fila(row, headers, keep_idx, cols_anonimizar)
            writer.writerow(fila)
            filas_procesadas += 1

            # Progreso cada 50,000 filas
            if filas_procesadas % 50000 == 0:
                print(f'  Procesadas: {filas_procesadas:,} filas...')

    wb.close()
    elapsed = time.time() - start
    return filas_procesadas, elapsed


# ── Configuración ─────────────────────────────────────────────────────
# Rutas relativas al proyecto — ejecutar desde la raíz del proyecto
ARCHIVO_ENTRADA = 'data/1.ene_2017.xlsx'
ARCHIVO_SALIDA  = 'data/ene2017_anonimizado.csv'
SEPARADOR       = '|'   # pipe evita conflictos con comas en texto libre

# Columnas que contienen datos personales — se reemplazan con hash SHA-256
COLS_ANONIMIZAR = {
    'Doc. Declarante',   # número de documento del agente aduanero
    'Declarante',        # nombre del agente aduanero
    'Doc. Importador',   # puede ser cédula de identidad
    'Importador'         # nombre del importador
}

# Columnas que se eliminan completamente del CSV
COLS_ELIMINAR = {
    'Descripción Comercial',  # puede revelar estrategia comercial sensible
    'SIN_NOMBRE'              # columna vacía — basura del Excel
}

# ── Ejecución ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Iniciando conversión: {ARCHIVO_ENTRADA}')
    print(f'Columnas a anonimizar: {COLS_ANONIMIZAR}')
    print(f'Columnas a eliminar:   {COLS_ELIMINAR}')
    print()

    filas, elapsed = convertir_y_anonimizar(
        ARCHIVO_ENTRADA,
        ARCHIVO_SALIDA,
        SEPARADOR,
        COLS_ANONIMIZAR,
        COLS_ELIMINAR
    )

    print()
    print(f'✓ Completado: {filas:,} filas en {elapsed:.1f} segundos')
    print(f'✓ Separador usado: pipe |')
    print(f'✓ Archivo guardado: {ARCHIVO_SALIDA}')