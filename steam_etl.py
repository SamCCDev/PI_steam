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

# Limitadores de tasa cooperativos entre hilos
limiter_storefront = GlobalRateLimiter(requests_per_second=0.55) # ~1.8s entre llamadas (160 req/5min, extremadamente seguro)
limiter_steamspy = GlobalRateLimiter(requests_per_second=1.0)   # ~1s entre llamadas
limiter_reviews = GlobalRateLimiter(requests_per_second=2.0)    # ~0.5s entre llamadas

def safe_get(url: str, params: dict = None, retries: int = MAX_RETRIES) -> dict | None:
    """
    Realiza un GET con reintentos exponenciales ante errores 429/500,
    usando limitadores de tasa globales y headers de navegador reales.
    """
    if "appdetails" in url:
        limiter = limiter_storefront
    elif "steamspy" in url:
        limiter = limiter_steamspy
    else:
        limiter = limiter_reviews

    is_storefront = "appdetails" in url
    max_attempts = retries + 2 if is_storefront else retries

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }

    for attempt in range(1, max_attempts + 1):
        # Esperar turno global coordinado
        limiter.wait_if_needed()

        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

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
    data = safe_get(url, params={
        "appids": appid, 
        "l": "english",
        "filters": "basic,price_overview,genres,categories,achievements"
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


def fetch_steamspy_details(appid: int) -> dict | None:
    """
    Llama a la SteamSpy API para obtener ventas estimadas, tags y más.
    """
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
        time.sleep(0.1) # La API de reseñas es más rápida, no necesitamos esperar 1.5s
        
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
    """
    # ── Steam Storefront ───────────────────────────────────────────────
    steam = fetch_steam_details(appid)

    if not steam:
        return None

    # ── SteamSpy ───────────────────────────────────────────────────────
    spy = fetch_steamspy_details(appid)

    try:
        ts_data = fetch_recent_reviews_timeseries(appid)
        
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
                      "hub_followers", "total_achievements", "total_dlcs",
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
        help="Extrae únicamente las series de tiempo para los appids ya existentes en games_metadata.csv.",
    )
    args = parser.parse_args()

    if args.validate:
        # Modo diagnóstico: solo inspecciona un juego
        validate_single_appid(args.validate)
    elif args.timeseries_only:
        # Modo extraer solo series de tiempo
        run_timeseries_only_pipeline()
    else:
        # Modo producción / prueba
        sample = None if args.sample == 0 else args.sample
        run_pipeline(sample_size=sample)
