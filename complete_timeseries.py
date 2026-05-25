import pandas as pd
from pathlib import Path

def main():
    output_dir = Path("output")
    metadata_path = output_dir / "games_metadata.csv"
    ts_path = output_dir / "games_timeseries.csv"

    if not metadata_path.exists():
        print(f"Error: {metadata_path} no existe.")
        return

    if not ts_path.exists():
        print(f"Error: {ts_path} no existe.")
        return

    print("Cargando archivos...")
    df_meta = pd.read_csv(metadata_path, sep=";")
    df_ts = pd.read_csv(ts_path, sep=";")

    all_appids = set(df_meta["appid"])
    ts_appids = set(df_ts["appid"])

    missing_appids = all_appids - ts_appids
    print(f"Total juegos en metadata: {len(all_appids)}")
    print(f"Juegos ya presentes en series de tiempo: {len(ts_appids)}")
    print(f"Juegos faltantes en series de tiempo: {len(missing_appids)}")

    if len(missing_appids) == 0:
        print("¡El archivo de series de tiempo ya está completo! Todos los juegos tienen al menos un registro.")
        return

    # Crear filas por defecto para los juegos faltantes
    default_rows = []
    for appid in missing_appids:
        default_rows.append({
            "appid": appid,
            "month": "2024-01",  # Usamos enero de 2024 como mes por defecto
            "review_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "steam_purchase_count": 0,
            "early_access_count": 0,
            "avg_playtime_at_review_hrs": 0.0
        })

    df_default = pd.DataFrame(default_rows)
    df_ts_complete = pd.concat([df_ts, df_default], ignore_index=True)

    # Ordenar por appid y mes
    df_ts_complete["appid"] = df_ts_complete["appid"].astype(int)
    df_ts_complete["month"] = df_ts_complete["month"].astype(str)
    df_ts_complete = df_ts_complete.sort_values(by=["appid", "month"])

    # Guardar
    df_ts_complete.to_csv(ts_path, index=False, encoding="utf-8", sep=";")
    print(f"✓ Archivo completado exitosamente: {ts_path}")
    print(f"Nuevo total de filas en series de tiempo: {len(df_ts_complete)}")
    print(f"Juegos únicos ahora representados: {df_ts_complete['appid'].nunique()}")

if __name__ == "__main__":
    main()
