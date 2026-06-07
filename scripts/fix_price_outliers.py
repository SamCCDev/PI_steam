"""
==============================================================================
  FIX PRICE OUTLIERS — repara precios inflados del scrape original
==============================================================================
  Problema: ~46 juegos (casi todos AAA: Cyberpunk 2077, ELDEN RING, RDR2...)
  quedaron con precios de 100-200+ USD en games_metadata.csv. El scrape
  original capturó precios de otra región/edición y build_dataset los
  clava en PRICE_CAP_USD=200, lo que distorsiona la feature `price` para
  esos títulos.

  Solución: re-consultar SOLO esos appids contra la Steam Storefront API
  forzando región US (cc=us) y sobrescribir el precio en games_metadata.csv
  (con respaldo .bak). Luego re-correr build_dataset + entrenamientos.

  Uso:
      python scripts/fix_price_outliers.py            # umbral por defecto $70
      python scripts/fix_price_outliers.py --threshold 90
==============================================================================
"""

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "output" / "games_metadata.csv"

API = "https://store.steampowered.com/api/appdetails?appids={appid}&cc=us&l=en&filters=price_overview"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"}
RATE_S = 1.6  # pausa entre llamadas para no gatillar HTTP 429


def fetch_price_usd(appid: int):
    """Precio actual en USD (región US) o None si Steam no lo expone
    (juego delistado / sin price_overview). Para F2P devuelve 0.0."""
    req = urllib.request.Request(API.format(appid=appid), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        payload = json.loads(r.read().decode("utf-8"))
    node = payload.get(str(appid), {})
    if not node.get("success"):
        return None
    data = node.get("data")
    if isinstance(data, list) or not data:          # success pero sin datos -> F2P/delistado raro
        return 0.0
    po = data.get("price_overview")
    if not po:                                       # sin price_overview con success=true => F2P
        return 0.0
    return round(po["final"] / 100.0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=70.0,
                    help="re-consultar juegos con price > umbral (default 70)")
    args = ap.parse_args()

    df = pd.read_csv(META, sep=";", low_memory=False)
    bad = df[df["price"] > args.threshold]
    print(f"{len(bad)} juegos con price > ${args.threshold:g} — re-consultando Steam (cc=us)…")

    backup = META.with_suffix(".csv.bak")
    shutil.copy2(META, backup)
    print(f"respaldo: {backup.name}")

    fixed, skipped = 0, 0
    for _, row in bad.iterrows():
        appid = int(row["appid"])
        try:
            price = fetch_price_usd(appid)
        except Exception as e:
            print(f"  appid {appid}: ERROR {e} — se mantiene {row['price']}")
            skipped += 1
            time.sleep(RATE_S)
            continue
        if price is None:
            print(f"  appid {appid}: sin datos en Steam (delistado) — se mantiene {row['price']}")
            skipped += 1
        else:
            df.loc[df["appid"] == appid, "price"] = price
            name = str(row["name"]).encode("ascii", "replace").decode()
            print(f"  appid {appid}: {name}  ${row['price']:.2f} -> ${price:.2f}")
            fixed += 1
        time.sleep(RATE_S)

    df.to_csv(META, sep=";", index=False)
    print(f"\nlisto: {fixed} corregidos, {skipped} sin cambio. Ahora corre:")
    print("  python scripts/build_dataset.py")
    print("  python scripts/train_models.py")
    print("  python scripts/train_stage1.py")


if __name__ == "__main__":
    main()
