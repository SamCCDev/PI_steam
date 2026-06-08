"""
==============================================================================
  SERVER — Backend del Dashboard con la librería estándar de Python
==============================================================================
  Sin dependencias externas: usa http.server. Sirve la API y el frontend.

      python app/server.py            # arranca en http://127.0.0.1:8000

  Endpoints:
    GET  /                  -> web/index.html
    GET  /api/stats         -> agregados para el panel analítico
    GET  /api/games?q=...   -> búsqueda de juegos reales
    GET  /api/game/<appid>  -> features de un juego real (modo validación)
    POST /api/predict       -> {features, model} -> predicción multi-modelo
    POST /api/recommend     -> {features, K, model} -> mejor paquete de cambios
    POST /api/similar       -> {features, k} -> juegos del mismo camino
==============================================================================
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import inference as inf   # noqa: E402
from app import stats              # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")
VENDOR_DIR = os.path.join(ROOT, "vendor")
# Local: 127.0.0.1:8000. En un host PaaS (Render/Railway) la plataforma inyecta PORT
# y hay que escuchar en 0.0.0.0 para recibir el tráfico del proxy.
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
    ".woff2": "font/woff2", ".map": "application/json",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SteamPredict/2.0"

    # ── utilidades de respuesta ───────────────────────────────────────────
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, abspath):
        if not os.path.isfile(abspath):
            return self._json({"error": "not found", "path": self.path}, 404)
        ext = os.path.splitext(abspath)[1].lower()
        with open(abspath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        # Forzar revalidación: evita que el navegador sirva index.html/JS cacheado tras un
        # redeploy (el http.server por sí solo no manda Cache-Control y Chrome cachea el HTML).
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _safe_join(self, base, rel):
        rel = rel.lstrip("/")
        abspath = os.path.normpath(os.path.join(base, rel))
        if not abspath.startswith(os.path.normpath(base)):
            return None    # bloquea path traversal
        return abspath

    def _body_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, fmt, *args):
        pass   # silencioso

    # ── GET ───────────────────────────────────────────────────────────────
    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        try:
            if path == "/api/stats":
                return self._json(stats.get_stats())
            if path == "/api/games":
                return self._json(inf.search_games(qs.get("q", [""])[0]))
            if path.startswith("/api/game/"):
                return self._json(inf.get_game(int(path.rsplit("/", 1)[-1])))
            if path.startswith("/api/"):
                return self._json({"error": "unknown endpoint"}, 404)

            # estáticos
            if path == "/" or path == "":
                return self._file(os.path.join(WEB_DIR, "index.html"))
            if path.startswith("/vendor/"):
                ap = self._safe_join(VENDOR_DIR, path[len("/vendor/"):])
                return self._file(ap) if ap else self._json({"error": "bad path"}, 400)
            ap = self._safe_join(WEB_DIR, path)
            return self._file(ap) if ap else self._json({"error": "bad path"}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    # ── POST ──────────────────────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path
        data = self._body_json()
        try:
            if path == "/api/predict":
                return self._json(inf.predict(data.get("features", {}),
                                              data.get("model", "todos")))
            if path == "/api/recommend":
                return self._json(inf.recommend(data.get("features", {}),
                                                int(data.get("K", 3)), data.get("model", "mlp")))
            if path == "/api/similar":
                return self._json(inf.similar(data.get("features", {}), int(data.get("k", 8))))
            return self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 60)
    print(f"  SteamPredict  ->  http://{HOST}:{PORT}")
    print(f"  Modelos: {list(inf.MODELS.keys())}  |  juegos: {len(inf.DF):,}")
    print(f"  Features mutables (recomendador): {len(inf.MUTABLE_BOOLS)}")
    print("  Ctrl+C para detener.")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
