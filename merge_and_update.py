import pandas as pd
from pathlib import Path
import requests
import time
import sys

def parse_owners_lower_bound(owners_str: str) -> int:
    if not owners_str:
        return 0
    try:
        lower_str = owners_str.split("..")[0].strip()
        clean_str = lower_str.replace(",", "").replace(".", "").replace(" ", "")
        return int(clean_str)
    except Exception:
        return 0

def fetch_owners_from_steamspy(appid: int) -> str:
    url = "https://steamspy.com/api.php"
    for attempt in range(3):
        try:
            res = requests.get(url, params={"request": "appdetails", "appid": appid}, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict) and "owners" in data:
                    return data.get("owners", "")
            elif res.status_code == 429:
                print("  → Recibido 429 (Too Many Requests) de SteamSpy, esperando 10s...")
                time.sleep(10)
        except Exception as e:
            print(f"  → Intento {attempt+1} fallido para appid={appid}: {e}")
        time.sleep(2)
    return ""

def main():
    output_dir = Path("output")
    backup_dir = Path("output_copy_mayo8bk")

    # 1. Rutas de archivos
    current_meta_path = output_dir / "games_metadata.csv"
    backup_meta_path = backup_dir / "games_metadata.csv"
    current_tags_path = output_dir / "games_tags.csv"
    backup_tags_path = backup_dir / "games_tags.csv"
    current_text_path = output_dir / "games_text.csv"
    backup_text_path = backup_dir / "games_text.csv"

    # Verificar existencia
    for p in [current_meta_path, backup_meta_path, current_tags_path, backup_tags_path, current_text_path, backup_text_path]:
        if not p.exists():
            print(f"Error: {p} no existe.")
            return

    # 2. Cargar datos
    print("Cargando datasets...")
    df_curr_meta = pd.read_csv(current_meta_path, sep=";")
    df_back_meta = pd.read_csv(backup_meta_path, sep=";")
    df_curr_tags = pd.read_csv(current_tags_path, sep=";")
    df_back_tags = pd.read_csv(backup_tags_path, sep=";")
    df_curr_text = pd.read_csv(current_text_path, sep=";")
    df_back_text = pd.read_csv(backup_text_path, sep=";")

    print(f"Metadata actual: {len(df_curr_meta)} juegos | Metadata copia: {len(df_back_meta)} juegos")

    # 3. Fusionar priorizando el dataset actual para no perder el progreso de dueños ya corregidos
    print("Fusionando dataframes...")
    df_merged_meta = pd.concat([df_curr_meta, df_back_meta]).drop_duplicates(subset=["appid"], keep="first")
    df_merged_tags = pd.concat([df_curr_tags, df_back_tags]).drop_duplicates(subset=["appid"], keep="first")
    df_merged_text = pd.concat([df_curr_text, df_back_text]).drop_duplicates(subset=["appid"], keep="first")

    print(f"Nuevo total de juegos fusionados: {len(df_merged_meta)}")

    # Guardar inmediatamente los tags y textos (ya que no requieren API calls)
    print("Guardando archivos base de tags y texto consolidados...")
    df_merged_tags.to_csv(current_tags_path, index=False, encoding="utf-8", sep=";")
    df_merged_text.to_csv(current_text_path, index=False, encoding="utf-8", sep=";")
    df_merged_meta.to_csv(current_meta_path, index=False, encoding="utf-8", sep=";")
    print("✓ Archivos base guardados exitosamente.")

    # 4. Actualizar owners_lower_bound para los juegos que tienen reseñas pero su owners_lower_bound es 0
    condition = (df_merged_meta["owners_lower_bound"] == 0) & (df_merged_meta["positive"] > 0)
    to_update = df_merged_meta[condition].copy()
    print(f"Se encontraron {len(to_update)} juegos que necesitan actualizar 'owners_lower_bound' desde SteamSpy...")

    count = 0
    total = len(to_update)

    for idx, row in to_update.iterrows():
        appid = int(row["appid"])
        count += 1
        print(f"[{count}/{total}] Obteniendo dueños reales para {row['name']} (appid={appid})...")
        
        owners_str = fetch_owners_from_steamspy(appid)
        if owners_str:
            lower_bound = parse_owners_lower_bound(owners_str)
            df_merged_meta.loc[df_merged_meta["appid"] == appid, "owners_lower_bound"] = lower_bound
            print(f"  → Éxito: {owners_str} -> {lower_bound:,} dueños mínimos.")
        else:
            print("  → No se pudo obtener información de SteamSpy.")
        
        # Guardar checkpoint de metadata cada 25 actualizaciones
        if count % 25 == 0:
            df_merged_meta.to_csv(current_meta_path, index=False, encoding="utf-8", sep=";")
            print("  ✓ Checkpoint de metadata guardado.")
        
        # Sleep educado para evitar rate limit de SteamSpy
        time.sleep(1.5)

    # Guardar estado final de metadata
    df_merged_meta.to_csv(current_meta_path, index=False, encoding="utf-8", sep=";")
    print("✓ Fusión y actualización completadas exitosamente.")

if __name__ == "__main__":
    main()
