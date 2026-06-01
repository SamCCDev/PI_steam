"""
==============================================================================
  STEAM ETL PIPELINE — Predictor de Videojuegos (ML Ready)
==============================================================================
  Fuentes: Steam Storefront API + SteamSpy API
  Salida:  games_metadata.csv | games_tags.csv | games_text.csv
  Autor:   Generado desde Documento Técnico v1.0
==============================================================================
"""

import sys
import subprocess

# Instalación automática de dependencias si no existen
try:
    import pandas as pd
    import requests
except ImportError:
    print("[*] Instalando librerías requeridas (pandas, requests)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "requests"])
    import pandas as pd
    import requests

import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

# Directorio donde se guardarán los CSVs finales
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Rate Limiting ──────────────────────────────────────────────────────────
# SteamSpy permite ~1 req/seg sin key. Steam Storefront es más permisiva.
DELAY_BETWEEN_REQUESTS = 1.5   # segundos entre llamadas (ajustar si hay 429s)
MAX_RETRIES            = 3     # intentos máximos ante error 429 / 500
RETRY_BACKOFF          = 5     # segundos extra de espera por cada reintento

# ── Modo de prueba ────────────────────────────────────────────────────────
# Pon SAMPLE_SIZE = None para procesar todos los appids disponibles.
# Por defecto procesará 1000 juegos como solicitaste.
SAMPLE_SIZE = 1000

# ── Timestamp de Inicio para Series de Tiempo (Reseñas) ────────────────────
# 1 de enero de 2024 en formato Unix. (Usado como proxy de jugadores activos)
START_TIMESTAMP = 1704067200

import os
# Cargar variables de entorno desde .env si existe
if Path(".env").exists():
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except Exception as e:
        print(f"[*] No se pudo cargar el archivo .env: {e}")

# ── Steam API Key (gratis en https://steamcommunity.com/dev/apikey) ────────
STEAM_API_KEY = os.environ.get("STEAM_API_KEY", "")
# ── Tags objetivo para One-Hot Encoding ──────────────────────────────────
# Lista de los tags de usuario más relevantes para el modelo.
# Amplía o recorta esta lista según análisis de frecuencia posterior.
TOP_TAGS = [
    "Pixel Art", "First-Person", "Open World", "Crafting", "Permadeath",
    "Story Rich", "Roguelike", "Roguelite", "Survival", "Horror",
    "RPG", "Action RPG", "Turn-Based", "Strategy", "Tower Defense",
    "Platformer", "Metroidvania", "Soulslike", "Co-op", "Multiplayer",
    "Local Co-Op", "PvP", "Battle Royale", "Sandbox", "Simulation",
    "Management", "City Builder", "Sports", "Racing", "Puzzle",
    "Point & Click", "Visual Novel", "Anime", "2D", "3D",
    "Realistic", "Low Poly", "Cartoon", "Dark", "Atmospheric",
    "Relaxing", "Difficult", "Casual", "Indie", "Early Access",
    "Free to Play", "VR", "MOBA", "FPS", "Stealth",
]

# Géneros y categorías estándar de Steam para One-Hot Encoding
STEAM_GENRES = [
    "Action", "Adventure", "Casual", "Free to Play", "Indie",
    "Massively Multiplayer", "RPG", "Racing", "Simulation",
    "Sports", "Strategy", "Violent", "Gore", "Early Access",
]

STEAM_CATEGORIES = [
    "Single-player", "Multi-player", "Co-op", "Online Co-op",
    "Local Co-op", "Full controller support", "Partial Controller Support",
    "Steam Achievements", "Steam Cloud", "Steam Workshop",
    "In-App Purchases", "Steam Trading Cards", "Remote Play Together",
    "Shared/Split Screen Co-op",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "etl.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CAPA DE RED — requests con reintentos y rate-limit
# ─────────────────────────────────────────────────────────────────────────────

import threading
from requests.adapters import HTTPAdapter

class GlobalRateLimiter:
    def __init__(self, requests_per_second=1.0):
        self.lock = threading.Lock()
        self.last_request_time = 0.0
        self.delay = 1.0 / requests_per_second
        self.paused_until = 0.0

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            if now < self.paused_until:
                sleep_time = self.paused_until - now
                time.sleep(sleep_time)
                now = time.time()

            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_request_time = time.time()

    def pause_all(self, duration):
        with self.lock:
            self.paused_until = max(self.paused_until, time.time() + duration)

# Limitadores de tasa cooperativos entre hilos (OPTIMIZADO)
# Storefront: límite real ~200 req/5min = 0.67 req/s. Usamos 0.63 (margen 5%).
# SteamSpy:   límite documentado 1 req/s.
# Reviews:    límite real ~10 req/s. Usamos 5.0 (margen 50%).
limiter_storefront = GlobalRateLimiter(requests_per_second=0.63) # ~1.59s entre llamadas (~190 req/5min, margen 5%)
limiter_steamspy = GlobalRateLimiter(requests_per_second=1.0)    # ~1s entre llamadas
limiter_reviews = GlobalRateLimiter(requests_per_second=5.0)     # ~0.2s entre llamadas (la API tolera ~10 req/s)

# ── Sesiones HTTP persistentes (thread-local) ─────────────────────────────
# requests.Session NO es thread-safe. Usamos threading.local() para crear
# una sesión independiente por hilo, reutilizando conexiones TCP (keep-alive)
# y evitando el overhead de handshake TCP/TLS en cada request (~200ms ahorro).
_thread_local = threading.local()

_SHARED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}

def _get_session() -> requests.Session:
    """Obtiene o crea una sesión HTTP persistente para el hilo actual."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update(_SHARED_HEADERS)
        # Pool de conexiones: hasta 10 conexiones persistentes por host
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session


def safe_get(url: str, params: dict = None, retries: int = MAX_RETRIES) -> dict | None:
    """
    Realiza un GET con reintentos exponenciales ante errores 429/500,
    usando limitadores de tasa globales, sesiones HTTP persistentes
    con connection pooling, y headers de navegador reales.
    """
    if "appdetails" in url:
        limiter = limiter_storefront
    elif "steamspy" in url:
        limiter = limiter_steamspy
    else:
        limiter = limiter_reviews

    is_storefront = "appdetails" in url
    max_attempts = retries + 2 if is_storefront else retries

    session = _get_session()

    for attempt in range(1, max_attempts + 1):
        # Esperar turno global coordinado
        limiter.wait_if_needed()

        try:
            response = session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                # Pausar todas las peticiones globales para evitar ban en cascada
                wait = 60 * attempt if is_storefront else RETRY_BACKOFF * attempt
                log.warning(f"429 Rate Limit en {url}. Pausando peticiones por {wait}s (intento {attempt}/{max_attempts})")
                limiter.pause_all(wait)
                time.sleep(wait)

            elif response.status_code in (500, 502, 503):
                wait = RETRY_BACKOFF * attempt
                log.warning(f"Error {response.status_code} en {url}. Esperando {wait}s (intento {attempt}/{max_attempts})")
                time.sleep(wait)

            else:
                log.error(f"HTTP {response.status_code} inesperado en {url}")
                return None

        except requests.exceptions.Timeout:
            log.warning(f"Timeout en {url} (intento {attempt}/{max_attempts})")
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.exceptions.ConnectionError as e:
            log.warning(f"Error de conexión en {url}: {e} (intento {attempt}/{max_attempts})")
            time.sleep(RETRY_BACKOFF * attempt)
        except (requests.exceptions.JSONDecodeError, ValueError) as e:
            log.warning(f"Error al decodificar JSON en {url}: {e} (intento {attempt}/{max_attempts})")
            time.sleep(RETRY_BACKOFF * attempt)

    log.error(f"Fallaron todos los reintentos para: {url}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CACHÉ BULK DE STEAMSPY (Optimización: elimina llamadas individuales)
# ─────────────────────────────────────────────────────────────────────────────

class SteamSpyBulkCache:
    """
    Pre-carga datos de SteamSpy desde el endpoint paginado `request=all`,
    que retorna owners, ccu, tags, positive, negative, etc. para cada juego.
    Esto elimina la necesidad de hacer llamadas individuales `appdetails` a
    SteamSpy para la mayoría de los juegos.
    """

    def __init__(self, cache_file: Path = OUTPUT_DIR / "steamspy_cache.json"):
        self._cache: dict[int, dict] = {}
        self._loaded = False
        self.cache_file = cache_file

    def try_load_from_disk(self) -> bool:
        import json
        import os
        if self.cache_file.exists():
            mtime = self.cache_file.stat().st_mtime
            # Reutilizar si tiene menos de 24 horas (86400 segundos)
            if time.time() - mtime < 86400:
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._cache = {int(k): v for k, v in data.items()}
                    self._loaded = True
                    log.info(f"[SteamSpy Cache] ✓ Cargado desde disco: {len(self._cache):,} juegos (válido por 24h).")
                    return True
                except Exception as e:
                    log.warning(f"No se pudo cargar el caché de disco: {e}")
            else:
                log.info("[SteamSpy Cache] El caché en disco expiró (>24h). Se descargará nuevamente.")
        return False

    def save_to_disk(self):
        import json
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
            log.info(f"[SteamSpy Cache] ✓ Guardado en disco ({self.cache_file.name}).")
        except Exception as e:
            log.error(f"Error guardando caché en disco: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_from_all_pages(self, target_appids: set[int] = None, max_pages: int = None):
        """
        Descarga el catálogo completo de SteamSpy paginado y lo almacena
        en memoria como caché. Si target_appids se especifica, solo se
        cachean esos appids (para ahorrar memoria).

        Parámetros:
          target_appids: Si se provee, solo almacenar estos appids.
          max_pages: Límite de páginas a descargar (None = todas).
        """
        log.info("[SteamSpy Cache] Descargando catálogo bulk de SteamSpy...")
        if self.try_load_from_disk():
            # Si target_appids se especificó, podríamos filtrar el caché en memoria para limpiar espacio,
            # pero no es estrictamente necesario.
            return

        page = 0
        total_cached = 0

        while True:
            if max_pages is not None and page >= max_pages:
                log.info(f"  Límite de {max_pages} páginas alcanzado.")
                break

            data = safe_get(
                "https://steamspy.com/api.php",
                params={"request": "all", "page": page}
            )

            if not data or len(data) == 0:
                log.info(f"  Página {page} vacía → fin del catálogo.")
                break

            for appid_str, info in data.items():
                if not isinstance(info, dict):
                    continue
                appid_int = int(appid_str)
                if target_appids is None or appid_int in target_appids:
                    self._cache[appid_int] = info
                    total_cached += 1

            log.info(f"  Página {page}: {len(data)} juegos vistos "
                     f"(cacheados: {total_cached:,})")

            page += 1
            # Rate limit duro de SteamSpy: 1 request de tipo "all" por minuto
            log.info(f"  Esperando 62s por rate limit de SteamSpy (pág {page-1}→{page})...")
            time.sleep(62)

        self._loaded = True
        self.save_to_disk()
        log.info(f"[SteamSpy Cache] ✓ Caché en memoria: {len(self._cache):,} juegos.")

    def get(self, appid: int) -> dict | None:
        """Busca un appid en el caché. Retorna None si no está."""
        return self._cache.get(appid)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, appid: int) -> bool:
        return appid in self._cache

# Instancia global del caché (se llena opcionalmente al inicio del pipeline)
spy_bulk_cache = SteamSpyBulkCache()


# ─────────────────────────────────────────────────────────────────────────────
# OBTENCIÓN DE APPIDS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_appids(sample_size: int = None) -> list[int]:
    """
    Obtiene todos los appids probando 2 fuentes:
      1. IStoreService/GetAppList (endpoint oficial vigente, requiere key, paginado)
      2. SteamSpy /all paginado (fallback sin key, 1.000/pág, 1 req/min)
    """

    # ── FUENTE 1: IStoreService (endpoint oficial actual) ─────────────────
    if STEAM_API_KEY and STEAM_API_KEY != "PEGA_TU_KEY_AQUI":
        log.info("Descargando catálogo desde IStoreService/GetAppList (oficial)...")
        appids = []
        last_appid = 0  # cursor de paginación que usa Steam

        while True:
            params = {
                "key":              STEAM_API_KEY,
                "include_games":    1,
                "include_dlc":      0,
                "include_software": 0,
                "include_videos":   0,
                "include_hardware": 0,
                "last_appid":       last_appid,
                "max_results":      50000,  # máximo que permite el endpoint
            }
            data = safe_get(
                "https://api.steampowered.com/IStoreService/GetAppList/v1/",
                params=params
            )

            if not data:
                log.warning("IStoreService no respondió, probando SteamSpy...")
                break

            response   = data.get("response", {})
            apps       = response.get("apps", [])
            have_more  = response.get("have_more_results", False)
            last_appid = response.get("last_appid", 0)

            appids.extend(app["appid"] for app in apps)
            log.info(f"  IStoreService: {len(appids):,} appids acumulados "
                     f"(have_more={have_more})...")

            if not have_more:
                break

            time.sleep(DELAY_BETWEEN_REQUESTS)

        if appids:
            log.info(f"✓ Catálogo completo: {len(appids):,} appids desde IStoreService.")
            if sample_size:
                import random
                random.seed(42)  # Semilla fija para reproducibilidad
                appids = random.sample(appids, min(sample_size, len(appids)))
                log.info(f"  → Modo prueba (muestra aleatoria): {len(appids)} appids.")
            return appids

    # ── FUENTE 2: SteamSpy paginado (fallback sin key) ────────────────────
    log.info("Descargando catálogo desde SteamSpy (paginado, 1.000/pág, 1 req/min)...")
    appids = []
    page   = 0

    # Si sample_size es pequeño, no necesitamos traer las 43 páginas
    max_pages = None
    if sample_size:
        import math
        max_pages = math.ceil(sample_size / 1000) + 1  # solo las páginas necesarias

    while True:
        data = safe_get(
            "https://steamspy.com/api.php",
            params={"request": "all", "page": page}
        )

        if not data or len(data) == 0:
            log.info(f"  SteamSpy: página {page} vacía → fin del catálogo.")
            break

        appids.extend(int(appid) for appid in data.keys())
        log.info(f"  SteamSpy: página {page} → {len(data)} juegos "
                 f"(total: {len(appids):,})")

        page += 1

        if max_pages and page >= max_pages:
            log.info(f"  SteamSpy: límite de {max_pages} páginas alcanzado.")
            break

        # Rate limit duro de SteamSpy: 1 request de tipo "all" por minuto
        log.info(f"  Esperando 62s por rate limit de SteamSpy (pág {page-1}→{page})...")
        time.sleep(62)

    if appids:
        if sample_size:
            appids = appids[:sample_size]
        log.info(f"✓ Catálogo SteamSpy: {len(appids):,} appids obtenidos.")
        return appids

    raise RuntimeError(
        "No se pudo obtener el catálogo. "
        "Verifica tu STEAM_API_KEY o tu conexión a internet."
    )
# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN POR JUEGO
# ─────────────────────────────────────────────────────────────────────────────

def fetch_steam_details(appid: int) -> dict | None:
    """
    Llama a la Steam Storefront API para un appid.
    Retorna el bloque de datos del juego o None si no existe / es DLC.
    """
    url = "https://store.steampowered.com/api/appdetails"
    # NOTA: No usamos el parámetro 'filters' porque el filtro 'basic' de Steam
    # excluye campos críticos: developers, publishers, platforms, release_date.
    # La diferencia de payload (~12KB extra) es despreciable comparada con el
    # cuello de botella real (rate limiter a 1.59s entre requests).
    data = safe_get(url, params={
        "appids": appid, 
        "l": "english"
    })

    if not data:
        return None

    app_data = data.get(str(appid), {})
    if not app_data.get("success"):
        return None

    info = app_data.get("data", {})

    # Filtrar DLCs, demos, etc. Solo queremos juegos base
    if info.get("type") not in ("game",):
        return None

    return info


def fetch_steamspy_details(appid: int, use_cache: bool = True) -> dict | None:
    """
    Llama a la SteamSpy API para obtener ventas estimadas, tags y más.
    Si hay caché bulk cargado, lo usa directamente (0 latencia de red).
    """
    # Primero intentar el caché bulk (si está disponible)
    if use_cache and spy_bulk_cache.is_loaded:
        cached = spy_bulk_cache.get(appid)
        if cached is not None:
            return cached
        # Si no está en caché, hacer la llamada individual como fallback
        log.debug(f"  appid={appid} no encontrado en caché bulk, haciendo llamada individual.")

    url = "https://steamspy.com/api.php"
    data = safe_get(url, params={"request": "appdetails", "appid": appid})
    return data  # SteamSpy siempre retorna algo o None


def fetch_recent_reviews_timeseries(appid: int) -> dict:
    """
    Descarga las reseñas del juego iterando páginas por fecha (las más recientes primero).
    Se detiene al encontrar una reseña anterior a START_TIMESTAMP.
    Retorna un diccionario de métricas mensuales.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    cursor = "*"
    timeseries = {}
    
    pages_fetched = 0
    MAX_PAGES = 10 # Limitar a ~1000 reseñas (máximo) por juego para acelerar significativamente la extracción
    
    while pages_fetched < MAX_PAGES:
        pages_fetched += 1
        params = {
            "json": 1,
            "filter": "recent",
            "language": "all",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": 100,
            "cursor": cursor
        }
        
        data = safe_get(url, params=params)
        if not data or data.get("success") != 1:
            break
            
        reviews = data.get("reviews", [])
        if not reviews:
            break
            
        reached_past = False
        for r in reviews:
            ts = r.get("timestamp_created", 0)
            if ts < START_TIMESTAMP:
                reached_past = True
                break
                
            # Convertir timestamp a YYYY-MM
            month_str = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m')
            
            if month_str not in timeseries:
                timeseries[month_str] = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0,
                    "steam_purchase": 0,
                    "early_access": 0,
                    "sum_playtime_mins": 0
                }
            
            ts_month = timeseries[month_str]
            ts_month["total"] += 1
            
            if r.get("voted_up"):
                ts_month["positive"] += 1
            else:
                ts_month["negative"] += 1
                
            if r.get("steam_purchase"):
                ts_month["steam_purchase"] += 1
                
            if r.get("written_during_early_access"):
                ts_month["early_access"] += 1
                
            author = r.get("author", {})
            ts_month["sum_playtime_mins"] += author.get("playtime_at_review", 0)
            
        if reached_past:
            break
            
        new_cursor = data.get("cursor")
        if not new_cursor or new_cursor == cursor:
            break
            
        cursor = new_cursor
        # No necesitamos sleep adicional: el limiter_reviews global ya controla
        # la tasa a 5 req/s (0.2s entre requests). El sleep anterior de 0.1s era
        # redundante y añadía latencia innecesaria.
        
    # Post-proceso: calcular promedio de horas jugadas
    for month, metrics in timeseries.items():
        if metrics["total"] > 0:
            avg_mins = metrics["sum_playtime_mins"] / metrics["total"]
            metrics["avg_playtime_at_review_hrs"] = round(avg_mins / 60.0, 2)
        else:
            metrics["avg_playtime_at_review_hrs"] = 0.0
            
        del metrics["sum_playtime_mins"]
        
    return timeseries


# ─────────────────────────────────────────────────────────────────────────────
# PARSEO Y LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────

def extract_min_ram(requirements_text: str) -> float:
    """
    Usa regex para extraer el requisito mínimo de RAM en GB desde el
    texto libre de requisitos de Steam (ej: "Memory: 8 GB RAM" → 8.0).
    """
    if not requirements_text:
        return 0.0
    # Busca patrones como "4 GB RAM", "512 MB RAM", "2GB"
    pattern = r"(\d+(?:\.\d+)?)\s*(gb|mb)\s*ram"
    match = re.search(pattern, requirements_text.lower())
    if not match:
        return 0.0
    value, unit = float(match.group(1)), match.group(2)
    return value if unit == "gb" else round(value / 1024, 2)


def parse_supported_languages(lang_string: str) -> int:
    """
    Convierte el string de idiomas de Steam en un conteo entero.
    Ej: "English, Spanish, French<br>Languages with full audio: English" → 3
    """
    if not lang_string:
        return 0
    # Eliminar el bloque de "full audio" y contar comas
    clean = re.sub(r"languages with.*", "", lang_string, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", "", clean)  # quitar HTML tags
    return len([l for l in clean.split(",") if l.strip()])


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def parse_owners_lower_bound(owners_str: str) -> int:
    """
    Parsea el string de owners de SteamSpy (ej: "100,000,000 .. 200,000,000")
    para extraer el límite inferior como entero.
    """
    if not owners_str:
        return 0
    try:
        lower_str = owners_str.split("..")[0].strip()
        clean_str = lower_str.replace(",", "").replace(".", "").replace(" ", "")
        return int(clean_str)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE FILAS POR CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata_row(appid: int, steam: dict, spy: dict) -> dict:
    """
    Construye la fila para games_metadata.csv.
    Combina datos de Steam Storefront y SteamSpy.
    """
    # Precio: Steam lo expresa en centavos bajo price_overview
    price_data = steam.get("price_overview", {})
    price_usd   = safe_float(price_data.get("final", 0)) / 100.0
    is_free     = safe_bool(steam.get("is_free", False))
    if is_free:
        price_usd = 0.0

    # Reseñas (pueden venir de SteamSpy con mayor completitud)
    positive = safe_int(spy.get("positive", 0)) if (spy and isinstance(spy, dict)) else 0
    negative = safe_int(spy.get("negative", 0)) if (spy and isinstance(spy, dict)) else 0
    total_reviews = positive + negative
    rating_pct = round((positive / total_reviews) * 100, 2) if total_reviews > 0 else 0.0

    # Plataformas
    platforms = steam.get("platforms", {})

    # RAM desde texto libre de requisitos mínimos
    req_min_text = steam.get("pc_requirements", {}).get("minimum", "")
    min_ram_gb = extract_min_ram(req_min_text)

    # Fecha de lanzamiento
    release = steam.get("release_date", {})
    release_date = release.get("date", "") if not release.get("coming_soon") else ""

    # Metacritic score (0 si no está disponible)
    metacritic_data = steam.get("metacritic", {})
    metacritic_score = safe_int(metacritic_data.get("score", 0)) if metacritic_data else 0

    return {
        "appid":               appid,
        "name":                str(steam.get("name", "")).strip(),
        "developer":           ", ".join(steam.get("developers", [])),
        "publisher":           ", ".join(steam.get("publishers", [])),
        "release_date":        release_date,
        "price":               price_usd,
        "is_free":             int(is_free),
        # Variables Objetivo
        "owners_lower_bound":  parse_owners_lower_bound(spy.get("owners", "")) if (spy and isinstance(spy, dict)) else 0,
        "ccu":                 safe_int(spy.get("ccu", 0)) if (spy and isinstance(spy, dict)) else 0,
        "rating_porcentaje":   rating_pct,
        "positive":            positive,
        "negative":            negative,
        # Recepción de crítica especializada
        "metacritic_score":    metacritic_score,
        # Proxy de Wishlists (hub followers vía SteamSpy)
        "hub_followers":       safe_int(spy.get("userscore", 0)) if (spy and isinstance(spy, dict)) else 0,
        # Características del producto
        "total_achievements":  safe_int(steam.get("achievements", {}).get("total", 0)),
        "total_dlcs":          len(steam.get("dlc", [])),
        "supported_languages": parse_supported_languages(steam.get("supported_languages", "")),
        "controller_support":  steam.get("controller_support", "none"),
        "platform_windows":    int(safe_bool(platforms.get("windows", False))),
        "platform_mac":        int(safe_bool(platforms.get("mac", False))),
        "platform_linux":      int(safe_bool(platforms.get("linux", False))),
        "min_ram_gb":          min_ram_gb,
    }


def build_tags_row(appid: int, steam: dict, spy: dict) -> dict:
    """
    Construye la fila para games_tags.csv.
    Aplica One-Hot Encoding a géneros, categorías y user tags.
    """
    row = {"appid": appid}

    # ── Géneros (de Steam Storefront) ──────────────────────────────────────
    genre_names = {g.get("description", "") for g in steam.get("genres", [])}
    for genre in STEAM_GENRES:
        col = "genre_" + genre.lower().replace(" ", "_").replace("-", "_")
        row[col] = int(genre in genre_names)

    # ── Categorías (Single-player, Co-op, In-App Purchases, etc.) ─────────
    cat_names = {c.get("description", "") for c in steam.get("categories", [])}
    for cat in STEAM_CATEGORIES:
        col = "cat_" + cat.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        row[col] = int(cat in cat_names)

    # ── User Tags (de SteamSpy, vienen como dict {tag: votos} o list [] si no hay) ────────────
    spy_tags = set()
    if spy and isinstance(spy, dict):
        tags_data = spy.get("tags")
        if isinstance(tags_data, dict):
            spy_tags = set(tags_data.keys())
        elif isinstance(tags_data, list):
            spy_tags = set(tags_data)

    for tag in TOP_TAGS:
        col = "tag_" + tag.lower().replace(" ", "_").replace("-", "_")
        row[col] = int(tag in spy_tags)

    return row


def build_text_row(appid: int, steam: dict) -> dict:
    """
    Construye la fila para games_text.csv.
    Limpia mínimamente el HTML de las descripciones.
    """
    def strip_html(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"<[^>]+>", " ", text).strip()

    return {
        "appid":              appid,
        "short_description":  strip_html(steam.get("short_description", "")),
        "about_the_game":     strip_html(steam.get("detailed_description", "")),
        # reviews_text_sample se populará en un paso separado (Steam Reviews API)
        # Lo dejamos vacío aquí para mantener el schema del CSV
        "reviews_text_sample": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def process_single_game(appid: int) -> dict | None:
    """
    Descarga y parsea la información completa para un solo appid.
    Retorna un diccionario con las filas estructuradas de metadata, tags, text y timeseries.

    Optimización: Pipeline de 2 fases por juego:
      Fase A (secuencial): fetch_steam_details() → si es DLC/demo, saltar (evita
                           desperdiciar llamadas SteamSpy+Reviews en no-juegos).
      Fase B (paralela):   fetch_steamspy_details() y fetch_reviews_timeseries()
                           corren concurrentemente usando un mini ThreadPoolExecutor.
    """
    # ── Fase A: Steam Storefront (secuencial, filtra DLCs) ─────────────
    steam = fetch_steam_details(appid)

    if not steam:
        return None

    # ── Fase B: SteamSpy + Reviews en paralelo ─────────────────────────
    # Estas dos APIs son independientes entre sí y solo necesitan el appid.
    # Lanzarlas concurrentemente ahorra ~1s de latencia por juego.
    with ThreadPoolExecutor(max_workers=2) as mini_pool:
        future_spy = mini_pool.submit(fetch_steamspy_details, appid)
        future_ts  = mini_pool.submit(fetch_recent_reviews_timeseries, appid)

        spy     = future_spy.result()
        ts_data = future_ts.result()

    try:
        row_meta = build_metadata_row(appid, steam, spy)
        row_tags = build_tags_row(appid, steam, spy)
        row_text = build_text_row(appid, steam)
        
        row_ts_list = []
        if ts_data:
            for month, metrics in ts_data.items():
                row_ts_list.append({
                    "appid": appid,
                    "month": month,
                    "review_count": metrics["total"],
                    "positive_count": metrics["positive"],
                    "negative_count": metrics["negative"],
                    "steam_purchase_count": metrics["steam_purchase"],
                    "early_access_count": metrics["early_access"],
                    "avg_playtime_at_review_hrs": metrics["avg_playtime_at_review_hrs"]
                })
        else:
            row_ts_list.append({
                "appid": appid,
                "month": "2024-01",
                "review_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "steam_purchase_count": 0,
                "early_access_count": 0,
                "avg_playtime_at_review_hrs": 0.0
            })
            
        return {
            "meta": row_meta,
            "tags": row_tags,
            "text": row_text,
            "timeseries": row_ts_list
        }
    except Exception as e:
        log.error(f"  → Error parseando appid={appid}: {e}")
        return None


def run_pipeline(sample_size: int = None):
    """
    Orquesta la extracción completa de forma incremental y concurrente:
      1. Lee los CSVs existentes (si existen) para operar de forma incremental.
      2. Obtiene los appids del catálogo de Steam que NO han sido procesados.
      3. Selecciona una muestra aleatoria de los nuevos juegos (si se indica sample_size).
      4. Extrae datos usando multihilo (3 hilos) y guarda progresivamente.
    """
    metadata_path = OUTPUT_DIR / "games_metadata.csv"
    tags_path = OUTPUT_DIR / "games_tags.csv"
    text_path = OUTPUT_DIR / "games_text.csv"
    ts_path = OUTPUT_DIR / "games_timeseries.csv"
    
    existing_appids = set()
    rows_metadata = []
    rows_tags = []
    rows_text = []
    rows_timeseries = []

    if metadata_path.exists():
        try:
            df_exist = pd.read_csv(metadata_path, sep=";")
            existing_appids = set(df_exist["appid"].tolist())
            rows_metadata = df_exist.to_dict("records")
            log.info(f"Se encontraron {len(existing_appids):,} juegos existentes en {metadata_path.name}.")
        except Exception as e:
            log.warning(f"No se pudo leer metadata existente ({e}). Se creará desde cero.")

    if existing_appids:
        if tags_path.exists():
            try:
                rows_tags = pd.read_csv(tags_path, sep=";").to_dict("records")
            except Exception:
                log.warning(f"No se pudo leer {tags_path.name}, se creará desde cero.")
        if text_path.exists():
            try:
                rows_text = pd.read_csv(text_path, sep=";").to_dict("records")
            except Exception:
                log.warning(f"No se pudo leer {text_path.name}, se creará desde cero.")
        if ts_path.exists():
            try:
                rows_timeseries = pd.read_csv(ts_path, sep=";").to_dict("records")
            except Exception:
                log.warning(f"No se pudo leer {ts_path.name}, se creará desde cero.")

    # Obtener catálogo de Steam
    log.info("Obteniendo catálogo completo de Steam...")
    all_appids = get_all_appids(sample_size=None)
    
    # Filtrar procesados
    unprocessed_appids = [aid for aid in all_appids if aid not in existing_appids]
    log.info(f"Juegos sin extraer en el catálogo de Steam: {len(unprocessed_appids):,}")

    # ── Pre-cargar caché bulk de SteamSpy ──────────────────────────────
    # Si hay suficientes juegos para procesar, cargamos el catálogo de
    # SteamSpy en memoria para evitar llamadas individuales.
    if len(unprocessed_appids) >= 100 and not spy_bulk_cache.is_loaded:
        import math
        target_set = set(unprocessed_appids[:sample_size] if sample_size else unprocessed_appids)
        # Solo descargar las páginas necesarias (~1000 juegos/página)
        needed_pages = min(math.ceil(len(target_set) / 500), 100)  # máx 100 páginas
        spy_bulk_cache.load_from_all_pages(target_appids=target_set, max_pages=needed_pages)

    if not unprocessed_appids:
        log.info("✓ Todos los juegos del catálogo ya han sido extraídos. Nada por hacer.")
        return

    # Seleccionar la muestra aleatoria de los no procesados si se especificó sample_size
    if sample_size:
        import random
        random.seed(42)
        appids_to_process = random.sample(unprocessed_appids, min(sample_size, len(unprocessed_appids)))
        log.info(f"Seleccionando muestra aleatoria de {len(appids_to_process):,} juegos nuevos para procesar.")
    else:
        appids_to_process = unprocessed_appids

    total = len(appids_to_process)
    skipped = 0

    log.info(f"Iniciando descarga concurrente con 5 hilos de ejecución...")
    
    import threading
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_game, appid): appid for appid in appids_to_process}
        
        count = 0
        for future in as_completed(futures):
            count += 1
            appid = futures[future]
            res = future.result()
            
            if res is not None:
                with lock:
                    rows_metadata.append(res["meta"])
                    rows_tags.append(res["tags"])
                    rows_text.append(res["text"])
                    rows_timeseries.extend(res["timeseries"])
            else:
                with lock:
                    skipped += 1
            
            if count % 10 == 0 or count == total:
                log.info(f"  → [{count}/{total}] Juegos procesados concurrentemente...")
                
            # Checkpoint parcial cada 50 juegos para no perder progreso
            if count % 50 == 0:
                with lock:
                    meta_cp = list(rows_metadata)
                    tags_cp = list(rows_tags)
                    text_cp = list(rows_text)
                    ts_cp = list(rows_timeseries)
                _export_checkpoint(meta_cp, tags_cp, text_cp, ts_cp)
                log.info(f"  ✓ Checkpoint guardado ({count}/{total} procesados, {skipped} saltados).")

    # ── Exportación final ──────────────────────────────────────────────────
    log.info("Exportando CSVs finales...")
    export_csvs(rows_metadata, rows_tags, rows_text, rows_timeseries)
    log.info(
        f"Pipeline completado: {total - skipped} juegos exportados, "
        f"{skipped} saltados."
    )


def _export_checkpoint(rows_metadata, rows_tags, rows_text, rows_timeseries):
    """Guarda versiones parciales de los CSVs como backup de progreso."""
    export_csvs(rows_metadata, rows_tags, rows_text, rows_timeseries, suffix="_checkpoint")


def process_single_timeseries(appid: int) -> list[dict]:
    """
    Función helper para descargar y procesar las series de tiempo de un appid de forma concurrente.
    """
    try:
        ts_data = fetch_recent_reviews_timeseries(appid)
        rows = []
        if ts_data:
            for month, metrics in ts_data.items():
                rows.append({
                    "appid": appid,
                    "month": month,
                    "review_count": metrics["total"],
                    "positive_count": metrics["positive"],
                    "negative_count": metrics["negative"],
                    "steam_purchase_count": metrics["steam_purchase"],
                    "early_access_count": metrics["early_access"],
                    "avg_playtime_at_review_hrs": metrics["avg_playtime_at_review_hrs"]
                })
        else:
            rows.append({
                "appid": appid,
                "month": "2024-01",
                "review_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "steam_purchase_count": 0,
                "early_access_count": 0,
                "avg_playtime_at_review_hrs": 0.0
            })
        return rows
    except Exception as e:
        log.error(f"  → Error procesando timeseries para appid={appid}: {e}")
        return []


def run_timeseries_only_pipeline():
    """
    Lee los appids desde output/games_metadata.csv,
    compara con los que ya tienen series de tiempo en output/games_timeseries.csv,
    y extrae ÚNICAMENTE la serie de tiempo para los que faltan.
    Usa concurrencia (4 hilos) para acelerar drásticamente la descarga.
    """
    metadata_path = OUTPUT_DIR / "games_metadata.csv"
    ts_path = OUTPUT_DIR / "games_timeseries.csv"
    
    if not metadata_path.exists():
        log.error(f"No se encontró el archivo de metadata en: {metadata_path}")
        log.error("Debes ejecutar el ETL normal primero para generar el archivo de juegos base.")
        return
        
    log.info(f"Cargando appids desde {metadata_path}...")
    try:
        df_meta = pd.read_csv(metadata_path, sep=";")
        all_appids = df_meta["appid"].tolist()
    except Exception as e:
        log.error(f"Error al leer {metadata_path}: {e}")
        return
        
    log.info(f"Se encontraron {len(all_appids):,} juegos en el archivo de metadata.")
    
    existing_appids = set()
    rows_timeseries = []
    
    if ts_path.exists():
        log.info(f"Cargando series de tiempo existentes desde {ts_path}...")
        try:
            df_ts_existing = pd.read_csv(ts_path, sep=";")
            existing_appids = set(df_ts_existing["appid"].unique())
            # Convertir el DataFrame existente a lista de diccionarios para no perderlos
            rows_timeseries = df_ts_existing.to_dict("records")
            log.info(f"Se cargaron {len(existing_appids):,} juegos con series de tiempo existentes.")
        except Exception as e:
            log.warning(f"No se pudo leer {ts_path} correctamente ({e}). Se creará un archivo nuevo.")
            
    # Filtrar los que faltan por procesar
    to_process = [appid for appid in all_appids if appid not in existing_appids]
    log.info(f"Juegos restantes por extraer series de tiempo: {len(to_process):,}")
    
    if not to_process:
        log.info("✓ Todos los juegos ya tienen datos de series de tiempo. Nada que hacer.")
        return
        
    total = len(to_process)
    
    log.info(f"Iniciando descarga concurrente con 4 hilos para {total} juegos...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_timeseries, appid): appid for appid in to_process}
        
        count = 0
        for future in as_completed(futures):
            count += 1
            res = future.result()
            if res:
                rows_timeseries.extend(res)
                
            if count % 20 == 0 or count == total:
                log.info(f"  → [{count}/{total}] Juegos procesados concurrentemente...")
                
            # Guardar checkpoint parcial cada 50 juegos para no perder progreso
            if count % 50 == 0:
                df_ts = pd.DataFrame(rows_timeseries)
                if not df_ts.empty:
                    df_ts["appid"] = df_ts["appid"].astype(int)
                    df_ts["month"] = df_ts["month"].astype(str)
                    df_ts["review_count"] = df_ts["review_count"].fillna(0).astype(int)
                    df_ts["positive_count"] = df_ts["positive_count"].fillna(0).astype(int)
                    df_ts["negative_count"] = df_ts["negative_count"].fillna(0).astype(int)
                    df_ts["steam_purchase_count"] = df_ts["steam_purchase_count"].fillna(0).astype(int)
                    df_ts["early_access_count"] = df_ts["early_access_count"].fillna(0).astype(int)
                    df_ts["avg_playtime_at_review_hrs"] = df_ts["avg_playtime_at_review_hrs"].fillna(0.0).astype(float)
                    df_ts = df_ts.sort_values(by=["appid", "month"])
                    df_ts.to_csv(ts_path, index=False, encoding="utf-8", sep=";")
                    log.info(f"  ✓ Checkpoint guardado: {len(df_ts)} filas escritas.")

    # Guardar archivo final
    df_ts = pd.DataFrame(rows_timeseries)
    if not df_ts.empty:
        df_ts["appid"] = df_ts["appid"].astype(int)
        df_ts["month"] = df_ts["month"].astype(str)
        df_ts["review_count"] = df_ts["review_count"].fillna(0).astype(int)
        df_ts["positive_count"] = df_ts["positive_count"].fillna(0).astype(int)
        df_ts["negative_count"] = df_ts["negative_count"].fillna(0).astype(int)
        df_ts["steam_purchase_count"] = df_ts["steam_purchase_count"].fillna(0).astype(int)
        df_ts["early_access_count"] = df_ts["early_access_count"].fillna(0).astype(int)
        df_ts["avg_playtime_at_review_hrs"] = df_ts["avg_playtime_at_review_hrs"].fillna(0.0).astype(float)
        df_ts = df_ts.sort_values(by=["appid", "month"])
        df_ts.to_csv(ts_path, index=False, encoding="utf-8", sep=";")
        log.info(f"✓ Extracción final de series de tiempo completada. Archivo guardado: {ts_path} ({len(df_ts)} filas).")
    else:
        log.info("No se generó ningún dato de serie de tiempo.")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN A CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_csvs(
    rows_metadata: list[dict],
    rows_tags:     list[dict],
    rows_text:     list[dict],
    rows_timeseries: list[dict] = None,
    suffix:        str = "",
):

    """
    Convierte las listas de filas a DataFrames, aplica tipos de datos
    correctos y exporta a CSV listos para Scikit-Learn / XGBoost.

    Los valores nulos residuales se rellenan con 0 (numérico) o "" (texto)
    para garantizar compatibilidad con los pipelines de ML.
    """
    # ── games_metadata.csv ────────────────────────────────────────────────
    df_meta = pd.DataFrame(rows_metadata)
    if not df_meta.empty:
        # Tipos explícitos para ML
        int_cols   = ["appid", "owners_lower_bound", "ccu", "positive", "negative",
                      "metacritic_score", "hub_followers", "total_achievements", "total_dlcs",
                      "supported_languages", "platform_windows", "platform_mac",
                      "platform_linux", "is_free"]
        float_cols = ["price", "rating_porcentaje", "min_ram_gb"]
        str_cols   = ["name", "developer", "publisher", "release_date",
                      "controller_support"]

        df_meta[int_cols]   = df_meta[int_cols].fillna(0).astype(int)
        df_meta[float_cols] = df_meta[float_cols].fillna(0.0).astype(float)
        df_meta[str_cols]   = df_meta[str_cols].fillna("").astype(str)

        path = OUTPUT_DIR / f"games_metadata{suffix}.csv"
        df_meta.to_csv(path, index=False, encoding="utf-8",sep=";")
        log.info(f"  → {path}  ({len(df_meta)} filas, {len(df_meta.columns)} columnas)")

    # ── games_tags.csv ────────────────────────────────────────────────────
    df_tags = pd.DataFrame(rows_tags)
    if not df_tags.empty:
        # Todas las columnas excepto appid son booleanas (0/1)
        tag_cols = [c for c in df_tags.columns if c != "appid"]
        df_tags[tag_cols] = df_tags[tag_cols].fillna(0).astype(int)
        df_tags["appid"]  = df_tags["appid"].astype(int)

        path = OUTPUT_DIR / f"games_tags{suffix}.csv"
        df_tags.to_csv(path, index=False, encoding="utf-8",sep=";")
        log.info(f"  → {path}  ({len(df_tags)} filas, {len(df_tags.columns)} columnas)")

    # ── games_text.csv ────────────────────────────────────────────────────
    df_text = pd.DataFrame(rows_text)
    if not df_text.empty:
        text_cols = ["short_description", "about_the_game", "reviews_text_sample"]
        df_text[text_cols] = df_text[text_cols].fillna("").astype(str)
        df_text["appid"]   = df_text["appid"].astype(int)

        path = OUTPUT_DIR / f"games_text{suffix}.csv"
        df_text.to_csv(path, index=False, encoding="utf-8",sep=";")
        log.info(f"  → {path}  ({len(df_text)} filas, {len(df_text.columns)} columnas)")

    # ── games_timeseries.csv ──────────────────────────────────────────────
    if rows_timeseries is not None:
        df_ts = pd.DataFrame(rows_timeseries)
        if not df_ts.empty:
            df_ts["appid"] = df_ts["appid"].astype(int)
            df_ts["month"] = df_ts["month"].astype(str)
            df_ts["review_count"] = df_ts["review_count"].fillna(0).astype(int)
            df_ts["positive_count"] = df_ts["positive_count"].fillna(0).astype(int)
            df_ts["negative_count"] = df_ts["negative_count"].fillna(0).astype(int)
            df_ts["steam_purchase_count"] = df_ts["steam_purchase_count"].fillna(0).astype(int)
            df_ts["early_access_count"] = df_ts["early_access_count"].fillna(0).astype(int)
            df_ts["avg_playtime_at_review_hrs"] = df_ts["avg_playtime_at_review_hrs"].fillna(0.0).astype(float)
            
            df_ts = df_ts.sort_values(by=["appid", "month"])

            path = OUTPUT_DIR / f"games_timeseries{suffix}.csv"
            df_ts.to_csv(path, index=False, encoding="utf-8", sep=";")
            log.info(f"  → {path}  ({len(df_ts)} filas, {len(df_ts.columns)} columnas)")


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────────────

def validate_single_appid(appid: int):
    """
    Herramienta de debug: muestra el output crudo de ambas APIs
    para un appid específico sin escribir nada al disco.
    Útil para verificar que el parseo funciona antes de correr el pipeline.

    Uso:
        python steam_etl.py --validate 730    # appid de CS2
    """
    log.info(f"=== Validando appid={appid} ===")

    steam = fetch_steam_details(appid)
    if steam:
        log.info(f"Steam OK → nombre: {steam.get('name')}, tipo: {steam.get('type')}")
    else:
        log.warning("Steam → sin datos o no es juego base.")

    time.sleep(DELAY_BETWEEN_REQUESTS)

    spy = fetch_steamspy_details(appid)
    if spy:
        tags_data = spy.get("tags")
        spy_tags_list = list(tags_data.keys()) if isinstance(tags_data, dict) else list(tags_data) if isinstance(tags_data, list) else []
        log.info(f"SteamSpy OK → owners: {spy.get('owners_lower_bound')}, "
                 f"tags: {spy_tags_list[:5]}...")
    else:
        log.warning("SteamSpy → sin datos.")

    if steam and spy:
        print("\n── Metadata row ──────────────────────────────────────")
        print(pd.Series(build_metadata_row(appid, steam, spy)).to_string())

        print("\n── Tags row (solo columnas con valor=1) ───────────────")
        tags = build_tags_row(appid, steam, spy)
        active_tags = {k: v for k, v in tags.items() if v == 1}
        print(pd.Series(active_tags).to_string())

        print("\n── Text row (primeros 200 chars) ──────────────────────")
        text = build_text_row(appid, steam)
        print(f"short_description: {text['short_description'][:200]}")

        print("\n── Time Series (Reseñas >= Jan 2024) ──────────────────")
        ts = fetch_recent_reviews_timeseries(appid)
        if ts:
            for m, metrics in sorted(ts.items()):
                print(f"{m}: {metrics['total']} reviews "
                      f"(Pos: {metrics['positive']}, Neg: {metrics['negative']}, "
                      f"AvgHrs: {metrics['avg_playtime_at_review_hrs']})")
        else:
            print("Sin datos de series de tiempo para este período.")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CON FILTRO DE CALIDAD (owners garantizados)
# ─────────────────────────────────────────────────────────────────────────────

def get_quality_appids(min_owners: int = 10_000, min_reviews: int = 10, sample_size: int = None) -> list[int]:
    """
    Obtiene una lista de appids garantizando que los juegos tengan datos de
    'owners' significativos en SteamSpy (>= min_owners) y reseñas suficientes.
    """
    log.info(f"[Quality Filter] Descargando catálogo SteamSpy con métricas de owners/reseñas...")
    log.info(f"[Quality Filter] Criterios: owners >= {min_owners:,} | reviews >= {min_reviews}")

    quality_appids = []
    total_seen = 0

    if spy_bulk_cache.try_load_from_disk():
        log.info(f"[Quality Filter] Filtrando a partir del caché físico...")
        for appid_int, info in spy_bulk_cache._cache.items():
            total_seen += 1
            owners_str = info.get("owners", "")
            owners_lb  = parse_owners_lower_bound(owners_str)
            pos        = safe_int(info.get("positive", 0))
            neg        = safe_int(info.get("negative", 0))
            if owners_lb >= min_owners and (pos + neg) >= min_reviews:
                quality_appids.append(appid_int)
        
    else:
        page = 0
        while True:
            data = safe_get(
                "https://steamspy.com/api.php",
                params={"request": "all", "page": page}
            )

            if not data or len(data) == 0:
                log.info(f"  SteamSpy pág {page}: vacía → fin del catálogo.")
                break

            total_seen += len(data)
            passing = 0
            for appid_str, info in data.items():
                if not isinstance(info, dict):
                    continue
                appid_int  = int(appid_str)
                owners_str = info.get("owners", "")
                owners_lb  = parse_owners_lower_bound(owners_str)
                pos        = safe_int(info.get("positive", 0))
                neg        = safe_int(info.get("negative", 0))
                if owners_lb >= min_owners and (pos + neg) >= min_reviews:
                    quality_appids.append(appid_int)
                    spy_bulk_cache._cache[appid_int] = info
                    passing += 1

            log.info(f"  Página {page}: {len(data)} juegos vistos, {passing} pasan el filtro "
                     f"(total filtrados: {len(quality_appids):,})")

            # Optimización: Como SteamSpy devuelve la lista ordenada por 'owners',
            # si una página devuelve 0 juegos válidos, cortamos el loop tempranamente.
            if passing == 0:
                log.info("  0 juegos pasaron el filtro. Cortando escaneo prematuramente para ahorrar tiempo (fin de juegos relevantes).")
                break

            page += 1
            log.info(f"  Esperando 62s por rate limit de SteamSpy (pág {page-1}→{page})...")
            time.sleep(62)

        spy_bulk_cache._loaded = True
        spy_bulk_cache.save_to_disk()

    log.info(f"[Quality Filter] Resultado: {len(quality_appids):,} de {total_seen:,} juegos "
             f"pasan el filtro de calidad.")
    
    if sample_size and len(quality_appids) > sample_size:
        import random
        random.seed(42)
        quality_appids = random.sample(quality_appids, sample_size)
        log.info(f"[Quality Filter] Muestra aleatoria seleccionada: {len(quality_appids)} appids.")

    return quality_appids


def run_quality_pipeline(min_owners: int = 10_000, min_reviews: int = 10, sample_size: int = None):
    """
    Versión del pipeline que pre-filtra el catálogo de Steam usando los datos
    de SteamSpy para garantizar que TODOS los juegos procesados tengan datos
    de 'owners_lower_bound' completos.
    """
    metadata_path = OUTPUT_DIR / "games_metadata.csv"
    tags_path     = OUTPUT_DIR / "games_tags.csv"
    text_path     = OUTPUT_DIR / "games_text.csv"
    ts_path       = OUTPUT_DIR / "games_timeseries.csv"

    existing_appids = set()
    rows_metadata, rows_tags, rows_text, rows_timeseries = [], [], [], []

    if metadata_path.exists():
        try:
            df_exist = pd.read_csv(metadata_path, sep=";")
            existing_appids = set(df_exist["appid"].tolist())
            rows_metadata   = df_exist.to_dict("records")
            log.info(f"Cargados {len(existing_appids):,} juegos existentes desde {metadata_path.name}.")
        except Exception as e:
            log.warning(f"No se pudo leer metadata existente ({e}).")

    if existing_appids:
        for path, lst in [(tags_path, rows_tags), (text_path, rows_text), (ts_path, rows_timeseries)]:
            if path.exists():
                try:
                    lst += pd.read_csv(path, sep=";").to_dict("records")
                except Exception:
                    log.warning(f"No se pudo leer {path.name}.")

    quality_appids = get_quality_appids(
        min_owners=min_owners,
        min_reviews=min_reviews,
        sample_size=sample_size
    )

    to_process = [aid for aid in quality_appids if aid not in existing_appids]
    log.info(f"Juegos de calidad sin procesar: {len(to_process):,}")

    if not to_process:
        log.info("✓ Todos los juegos de calidad ya han sido extraídos.")
        return

    total   = len(to_process)
    skipped = 0

    import threading
    lock = threading.Lock()

    log.info(f"Iniciando descarga concurrente con 5 hilos para {total:,} juegos de calidad...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_game, appid): appid for appid in to_process}

        count = 0
        for future in as_completed(futures):
            count += 1
            appid = futures[future]
            res   = future.result()

            if res is not None:
                with lock:
                    rows_metadata.append(res["meta"])
                    rows_tags.append(res["tags"])
                    rows_text.append(res["text"])
                    rows_timeseries.extend(res["timeseries"])
            else:
                with lock:
                    skipped += 1

            if count % 10 == 0 or count == total:
                log.info(f"  → [{count}/{total}] Juegos procesados...")

            if count % 50 == 0:
                with lock:
                    _export_checkpoint(list(rows_metadata), list(rows_tags),
                                       list(rows_text), list(rows_timeseries))
                log.info(f"  ✓ Checkpoint guardado ({count}/{total}, {skipped} saltados).")

    log.info("Exportando CSVs finales (quality pipeline)...")
    export_csvs(rows_metadata, rows_tags, rows_text, rows_timeseries)
    log.info(f"Quality pipeline completado: {total - skipped} juegos exportados, {skipped} saltados.")


# ─────────────────────────────────────────────────────────────────────────────
# MANTENIMIENTO Y SANITIZACIÓN (Utilidades unificadas)
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_metadata():
    """
    Lee games_metadata.csv y obtiene los datos faltantes (como metacritic_score,
    developers, publishers, etc.) para los juegos antiguos que fueron extraídos 
    con el filtro 'basic' defectuoso.
    """
    metadata_path = OUTPUT_DIR / "games_metadata.csv"
    if not metadata_path.exists():
        log.error(f"No existe {metadata_path}")
        return

    log.info("Cargando dataset para sanitización...")
    df = pd.read_csv(metadata_path, sep=";")
    
    df["developer"] = df["developer"].fillna("")
    if "metacritic_score" not in df.columns:
        df["metacritic_score"] = 0

    # Sanitizamos los que no tienen developer o no tienen metacritic (que probablemente no se consultó)
    # Como metacritic=0 puede ser legítimo, nos basamos principalmente en si es de la tanda antigua.
    # Asumiremos que si se corre este comando, se quieren revisar los que tienen metacritic_score == 0.
    mask = (df["developer"] == "") | (df["metacritic_score"] == 0)
    indices_to_update = df[mask].index.tolist()
    
    total = len(indices_to_update)
    log.info(f"Se encontraron {total:,} juegos que podrían necesitar actualización de metadata.")
    
    if total == 0:
        log.info("Nada que actualizar.")
        return

    count = 0
    modificados = 0
    
    session = _get_session()

    for idx in indices_to_update:
        appid = int(df.at[idx, "appid"])
        count += 1
        
        limiter_storefront.wait_if_needed()
        url = "https://store.steampowered.com/api/appdetails"
        
        for attempt in range(1, 4):
            try:
                res = session.get(url, params={"appids": appid, "l": "english"}, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    if data and str(appid) in data and data[str(appid)].get("success"):
                        steam_data = data[str(appid)]["data"]
                        
                        devs = steam_data.get("developers", [])
                        pubs = steam_data.get("publishers", [])
                        if devs: df.at[idx, "developer"] = devs[0]
                        if pubs: df.at[idx, "publisher"] = pubs[0]
                        
                        rel_date = steam_data.get("release_date", {})
                        df.at[idx, "release_date"] = rel_date.get("date", "")
                        
                        mc = steam_data.get("metacritic", {})
                        if mc: df.at[idx, "metacritic_score"] = mc.get("score", 0)
                        
                        platforms = steam_data.get("platforms", {})
                        df.at[idx, "platform_windows"] = 1 if platforms.get("windows") else 0
                        df.at[idx, "platform_mac"] = 1 if platforms.get("mac") else 0
                        df.at[idx, "platform_linux"] = 1 if platforms.get("linux") else 0
                        
                        modificados += 1
                    break
                elif res.status_code == 429:
                    wait = 60 * attempt
                    limiter_storefront.pause_all(wait)
                    time.sleep(wait)
            except Exception:
                time.sleep(2)
                
        if count % 10 == 0:
            log.info(f"Progreso Sanitización: {count}/{total} (Modificados: {modificados})")
            
        if count % 200 == 0:
            df.to_csv(metadata_path, sep=";", index=False, encoding="utf-8")
            log.info(f"  [✓] Checkpoint de sanitización guardado.")

    df.to_csv(metadata_path, sep=";", index=False, encoding="utf-8")
    log.info(f"¡Sanitización completada! Se actualizaron {modificados} juegos.")


def fill_timeseries():
    """
    Rellena con filas por defecto los appids que existen en metadata pero
    no tienen ninguna fila en timeseries.
    """
    metadata_path = OUTPUT_DIR / "games_metadata.csv"
    ts_path = OUTPUT_DIR / "games_timeseries.csv"

    if not metadata_path.exists() or not ts_path.exists():
        log.error("Faltan archivos base.")
        return

    df_meta = pd.read_csv(metadata_path, sep=";")
    df_ts = pd.read_csv(ts_path, sep=";")

    all_appids = set(df_meta["appid"])
    ts_appids = set(df_ts["appid"])
    missing_appids = all_appids - ts_appids
    
    log.info(f"Juegos en metadata: {len(all_appids):,}")
    log.info(f"Juegos faltantes en timeseries: {len(missing_appids):,}")

    if not missing_appids:
        log.info("Archivo timeseries ya está completo.")
        return

    default_rows = []
    for appid in missing_appids:
        default_rows.append({
            "appid": appid, "month": "2024-01", "review_count": 0,
            "positive_count": 0, "negative_count": 0, "steam_purchase_count": 0,
            "early_access_count": 0, "avg_playtime_at_review_hrs": 0.0
        })

    df_default = pd.DataFrame(default_rows)
    df_ts_complete = pd.concat([df_ts, df_default], ignore_index=True)

    df_ts_complete["appid"] = df_ts_complete["appid"].astype(int)
    df_ts_complete["month"] = df_ts_complete["month"].astype(str)
    df_ts_complete = df_ts_complete.sort_values(by=["appid", "month"])

    df_ts_complete.to_csv(ts_path, index=False, encoding="utf-8", sep=";")
    log.info(f"✓ Timeseries rellenado. Nuevo total de filas: {len(df_ts_complete):,}")


def merge_backups():
    """
    Fusiona el directorio actual 'output' con 'output_copy_mayo8bk'
    y reintenta obtener owners_lower_bound de SteamSpy para juegos que lo tengan en 0.
    """
    output_dir = Path("output")
    backup_dir = Path("output_copy_mayo8bk")
    current_meta_path = output_dir / "games_metadata.csv"
    backup_meta_path = backup_dir / "games_metadata.csv"

    if not current_meta_path.exists() or not backup_meta_path.exists():
        log.error("Archivos de backup o current no existen.")
        return

    log.info("Fusionando backups...")
    df_curr = pd.read_csv(current_meta_path, sep=";")
    df_back = pd.read_csv(backup_meta_path, sep=";")
    
    df_merged = pd.concat([df_curr, df_back]).drop_duplicates(subset=["appid"], keep="first")
    
    condition = (df_merged["owners_lower_bound"] == 0) & (df_merged["positive"] > 0)
    to_update = df_merged[condition].copy()
    log.info(f"Se encontraron {len(to_update)} juegos que necesitan actualizar 'owners'...")

    count, total = 0, len(to_update)
    for idx, row in to_update.iterrows():
        appid = int(row["appid"])
        count += 1
        
        limiter_steamspy.wait_if_needed()
        spy_data = safe_get("https://steamspy.com/api.php", params={"request": "appdetails", "appid": appid})
        
        if spy_data and "owners" in spy_data:
            lower_bound = parse_owners_lower_bound(spy_data["owners"])
            df_merged.loc[df_merged["appid"] == appid, "owners_lower_bound"] = lower_bound
            log.info(f"  [{count}/{total}] {row['name']} actualizado -> {lower_bound:,} dueños.")

    df_merged.to_csv(current_meta_path, index=False, encoding="utf-8", sep=";")
    log.info("✓ Fusión y actualización de owners completada exitosamente.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Steam ETL Pipeline — ML Ready")
    parser.add_argument(
        "--sample",
        type=int,
        default=SAMPLE_SIZE,
        metavar="N",
        help="Procesar solo N appids (prueba). Usa 0 para el catálogo completo.",
    )
    parser.add_argument(
        "--validate",
        type=int,
        default=None,
        metavar="APPID",
        help="Valida el parseo de un appid específico y muestra la salida. No escribe CSVs.",
    )
    parser.add_argument(
        "--timeseries-only",
        action="store_true",
        help="Extrae únicamente las series de tiempo para los appids ya existentes.",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Pre-filtra el catálogo de Steam usando SteamSpy para juegos de alta calidad.",
    )
    parser.add_argument(
        "--min-owners",
        type=int,
        default=10_000,
        metavar="N",
        help="(Solo con --quality-filter) Mínimo de owners. Default: 10,000.",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=10,
        metavar="N",
        help="(Solo con --quality-filter) Mínimo de reseñas. Default: 10.",
    )
    parser.add_argument(
        "--sanitize-metadata",
        action="store_true",
        help="[Mantenimiento] Actualiza juegos antiguos en metadata que no tengan developer o metacritic_score.",
    )
    parser.add_argument(
        "--fill-timeseries",
        action="store_true",
        help="[Mantenimiento] Rellena con 0 los juegos en timeseries que falten de metadata.",
    )
    parser.add_argument(
        "--merge-backups",
        action="store_true",
        help="[Mantenimiento] Fusiona el backup mayo8 y reintenta extraer owners_lower_bound en 0.",
    )
    args = parser.parse_args()

    if args.validate:
        validate_single_appid(args.validate)
    elif args.sanitize_metadata:
        sanitize_metadata()
    elif args.fill_timeseries:
        fill_timeseries()
    elif args.merge_backups:
        merge_backups()
    elif args.timeseries_only:
        run_timeseries_only_pipeline()
    elif args.quality_filter:
        sample = None if args.sample == 0 else args.sample
        run_quality_pipeline(
            min_owners=args.min_owners,
            min_reviews=args.min_reviews,
            sample_size=sample
        )
    else:
        sample = None if args.sample == 0 else args.sample
        run_pipeline(sample_size=sample)
