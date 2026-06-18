"""
Adapta el dataset real de Kaggle ("Laptop Price Prediction", Eslam Elsolia) al esquema de columnas que se usa en el resto del proyecto:
brand, laptop_type, cpu, ram_gb, storage_gb, storage_type, gpu, screen_size_in, resolution, touchscreen, weight_kg, os, price_usd.

Uso:
    1. Descarga el dataset de Kaggle y descomprime el zip.
    2. Copia "laptop_data.csv" dentro de data/raw/.
    3. Ejecuta:  python data/prepare_kaggle_dataset.py
    4. Esto sobrescribe data/raw/laptops.csv con los datos reales ya limpios.

Nota: el dataset de Kaggle no incluye "battery_hours" ni "release_year",
por eso esas dos columnas se han quitado de NUMERIC_FEATURES en
src/data_processing.py al pasar a datos reales.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
INPUT_PATH = RAW_DIR / "laptop_data.csv"
OUTPUT_PATH = RAW_DIR / "laptops.csv"

# Tipo de cambio aproximado INR -> USD del periodo del dataset (2018-2019).
# Es una conversión orientativa, no un valor financiero exacto.
INR_TO_USD = 1 / 70.0


def clean_ram(series: pd.Series) -> pd.Series:
    """'8GB' -> 8 (int)"""
    return series.str.replace("GB", "", regex=False).astype(int)


def clean_weight(series: pd.Series) -> pd.Series:
    """'1.37kg' -> 1.37 (float)"""
    return series.str.replace("kg", "", regex=False).astype(float)


def clean_memory(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'128GB SSD + 1TB HDD' -> (128.0, 'SSD')  (nos quedamos con el primer disco)"""
    primary = series.str.split("+").str[0].str.strip()
    extracted = primary.str.extract(r"([\d.]+)(GB|TB)\s*(SSD|HDD|Flash Storage|Hybrid)?")
    capacity = extracted[0].astype(float)
    unit = extracted[1]
    storage_gb = (capacity * 1024).where(unit == "TB", capacity)

    storage_type = extracted[2].map({
        "SSD": "SSD", "HDD": "HDD",
        "Flash Storage": "SSD", "Hybrid": "HDD",
    }).fillna("SSD")
    return storage_gb, storage_type


def clean_cpu(series: pd.Series) -> pd.Series:
    """'Intel Core i5 2.3GHz' -> 'Intel Core i5' (quitamos la velocidad en GHz)"""
    return series.str.replace(r"\s*\d+(\.\d+)?GHz", "", regex=True).str.strip()


def clean_gpu(series: pd.Series) -> pd.Series:
    """Nos quedamos con la marca de la GPU: 'Intel HD Graphics 620' -> 'Intel'"""
    return series.str.split().str[0]


def clean_resolution(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Extrae la resolución numérica y si tiene pantalla táctil."""
    resolution = series.str.extract(r"(\d+x\d+)")[0]
    touchscreen = series.str.contains("Touchscreen", case=False, na=False).astype(int)
    return resolution, touchscreen


def main():
    df = pd.read_csv(INPUT_PATH)

    out = pd.DataFrame()
    out["brand"] = df["Company"]
    out["laptop_type"] = df["TypeName"]
    out["cpu"] = clean_cpu(df["Cpu"])
    out["ram_gb"] = clean_ram(df["Ram"])
    storage_gb, storage_type = clean_memory(df["Memory"])
    out["storage_gb"] = storage_gb
    out["storage_type"] = storage_type
    out["gpu"] = clean_gpu(df["Gpu"])
    out["screen_size_in"] = df["Inches"].astype(float)
    resolution, touchscreen = clean_resolution(df["ScreenResolution"])
    out["resolution"] = resolution
    out["touchscreen"] = touchscreen
    out["weight_kg"] = clean_weight(df["Weight"])
    out["os"] = df["OpSys"]
    out["price_usd"] = (df["Price"] * INR_TO_USD).round(2)

    out = out.dropna().reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset real adaptado y guardado en {OUTPUT_PATH} ({len(out)} filas)")
    print(out.head())


if __name__ == "__main__":
    main()
