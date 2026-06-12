# steam_verificar_entorno.py
# Verificacion del entorno de trabajo para el apartado de ingenieria de datos.
#
# Curso: Ingenieria de Datos I - 2026
# Origen del patron: implementa el script 00_verificar_entorno.py del curso
#   (verificar que el entorno tenga todo lo necesario antes de trabajar).
#   Aqui se verifica el entorno del proyecto Steam: interprete, librerias,
#   disponibilidad opcional de Spark y presencia/esquema de los CSV de datos.
#
# COMO EJECUTAR (desde la raiz del repo o desde ingenieria_datos/):
#   python ingenieria_datos/steam_verificar_entorno.py
#
# Codigo de salida: 0 si todo lo OBLIGATORIO esta presente, 1 si falta algo.
# pyspark y Java son OPCIONALES: solo hacen falta para los scripts *_rdd_* y
# *_calidad_* en su variante Spark; las variantes _local corren sin ellos.

import importlib
import os
import sys

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
SEPARADOR = ';'

# CSV que consumen los scripts del apartado y columnas que deben existir.
ARCHIVOS_REQUERIDOS = {
    os.path.join(REPO_ROOT, 'output', 'games_metadata.csv'):
        ['appid', 'name', 'developer', 'publisher', 'price', 'controller_support'],
    os.path.join(REPO_ROOT, 'output', 'games_tags.csv'):
        ['appid'],          # ademas debe tener columnas one-hot genre_*
}

PYTHON_MINIMO = (3, 10)


def _check(ok, etiqueta, detalle='', marca=None):
    """Imprime una linea de verificacion homogenea y devuelve el booleano."""
    if marca is None:
        marca = 'OK' if ok else 'FALTA'
    print(f'  [{marca:<5}] {etiqueta:<38} {detalle}')
    return ok


def verificar_interprete():
    actual = sys.version_info[:2]
    ok = actual >= PYTHON_MINIMO
    return _check(ok, f'Python >= {PYTHON_MINIMO[0]}.{PYTHON_MINIMO[1]}',
                  f'(version actual: {sys.version.split()[0]})')


def verificar_libreria(nombre, obligatoria=True):
    try:
        mod = importlib.import_module(nombre)
        version = getattr(mod, '__version__', '?')
        return _check(True, nombre, f'(version: {version})')
    except ImportError:
        if obligatoria:
            return _check(False, nombre, 'no instalado')
        _check(True, f'{nombre} (opcional)',
               'no instalado - solo necesario para los scripts Spark',
               marca='AVISO')
        return True


def verificar_datos():
    """Comprueba que los CSV existan y tengan las columnas esperadas."""
    todo_ok = True
    for ruta, columnas_esperadas in ARCHIVOS_REQUERIDOS.items():
        nombre = os.path.relpath(ruta, REPO_ROOT)
        if not os.path.exists(ruta):
            todo_ok = _check(False, nombre, 'archivo no encontrado') and todo_ok
            continue

        with open(ruta, 'r', encoding='utf-8') as f:
            encabezado = f.readline().strip().split(SEPARADOR)

        faltantes = [c for c in columnas_esperadas if c not in encabezado]
        if faltantes:
            todo_ok = _check(False, nombre, f'faltan columnas: {faltantes}') and todo_ok
        else:
            _check(True, nombre, f'({len(encabezado)} columnas)')

        # games_tags.csv ademas debe traer los generos one-hot.
        if 'games_tags' in nombre:
            generos = [c for c in encabezado if c.startswith('genre_')]
            todo_ok = _check(bool(generos), '  columnas one-hot genre_*',
                             f'({len(generos)} generos)') and todo_ok
    return todo_ok


if __name__ == '__main__':
    print('=== Verificacion de entorno - apartado de ingenieria de datos ===\n')

    print('Interprete y librerias:')
    ok = verificar_interprete()
    ok = verificar_libreria('pandas') and ok
    verificar_libreria('pyspark', obligatoria=False)

    print('\nDatos de entrada:')
    ok = verificar_datos() and ok

    print()
    if ok:
        print('Entorno OK: se pueden ejecutar los scripts del apartado.')
        sys.exit(0)
    print('Entorno INCOMPLETO: revisar los items marcados como FALTA.')
    sys.exit(1)
