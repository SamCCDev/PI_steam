"""
==============================================================================
  EMBED MODEL IN HTML — Inyecta output/model_web.json en el dashboard estático
==============================================================================
  Reemplaza el contenido entre los marcadores /*MODEL_START*/ ... /*MODEL_END*/
  de steampredict_dashboard_comercial.html con el objeto JSON del modelo, para
  que el dashboard funcione abriéndolo directamente (sin servidor ni fetch).

  Ejecutar DESPUÉS de export_model_web.py. Es idempotente (se puede re-correr
  tras regenerar el modelo).
==============================================================================
"""

import json
import re
from pathlib import Path

ROOT = Path(".")
HTML = ROOT / "steampredict_dashboard_comercial.html"
MODEL_JSON = ROOT / "output" / "model_web.json"

model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
# Compacto pero válido como literal JS (JSON es un subconjunto de JS)
model_js = json.dumps(model, ensure_ascii=False, separators=(",", ":"))

html = HTML.read_text(encoding="utf-8")
pattern = re.compile(r"/\*MODEL_START\*/.*?/\*MODEL_END\*/", re.DOTALL)
if not pattern.search(html):
    raise SystemExit("No se encontraron los marcadores /*MODEL_START*/ ... /*MODEL_END*/ en el HTML.")

new_html = pattern.sub(f"/*MODEL_START*/{model_js}/*MODEL_END*/", html)
HTML.write_text(new_html, encoding="utf-8")

kb = len(model_js) / 1024
print(f"Modelo inyectado en {HTML.name} ({kb:.1f} KB de JSON).")
print(f"Clases: {model['classes']}  |  features booleanas: {len(model['boolean'])}")
