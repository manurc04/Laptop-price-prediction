"""
Generador del dataset de portátiles.

Se genera un dataset realista por si no se puede acceder al de Kaggle, modelando relaciones de precio reales del mercado de portátiles (marca, CPU,
RAM, almacenamiento, GPU, pantalla, peso, sistema operativo) más un ruido gaussiano, de forma reproducible (semilla fija).

"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1300

BRANDS = {
    "Apple": 1.55, "Dell": 1.05, "HP": 1.0, "Lenovo": 0.98, "Asus": 0.95,
    "Acer": 0.85, "MSI": 1.15, "Microsoft": 1.25, "Samsung": 1.1, "LG": 1.1,
}
CPUS = {
    "Intel Core i3": 0.7, "Intel Core i5": 1.0, "Intel Core i7": 1.4,
    "Intel Core i9": 1.9, "AMD Ryzen 5": 0.95, "AMD Ryzen 7": 1.3,
    "AMD Ryzen 9": 1.8, "Apple M1": 1.5, "Apple M2": 1.75, "Apple M3": 2.1,
}
GPUS = {
    "Integrated": 0.0, "Nvidia GTX 1650": 120, "Nvidia RTX 3050": 220,
    "Nvidia RTX 3060": 320, "Nvidia RTX 4060": 420, "Nvidia RTX 4070": 600,
    "AMD Radeon": 100, "Apple Integrated GPU": 90,
}
STORAGE_TYPES = {"HDD": 0.6, "SSD": 1.0, "SSD NVMe": 1.2}
OS_LIST = ["Windows 11", "macOS", "Linux", "ChromeOS", "Windows 10"]
TYPES = ["Ultrabook", "Gaming", "Notebook", "2 en 1", "Workstation"]

rows = []
for i in range(N):
    brand = RNG.choice(list(BRANDS.keys()))
    cpu = RNG.choice(list(CPUS.keys()))
    if brand == "Apple" and "Apple" not in cpu:
        cpu = RNG.choice(["Apple M1", "Apple M2", "Apple M3"])
    if brand != "Apple" and "Apple" in cpu:
        cpu = RNG.choice(["Intel Core i5", "Intel Core i7", "AMD Ryzen 5", "AMD Ryzen 7"])

    ram = int(RNG.choice([4, 8, 16, 32, 64], p=[0.05, 0.30, 0.40, 0.20, 0.05]))
    storage_capacity = int(RNG.choice([128, 256, 512, 1024, 2048], p=[0.05, 0.25, 0.40, 0.25, 0.05]))
    storage_type = RNG.choice(list(STORAGE_TYPES.keys()), p=[0.15, 0.55, 0.30])
    gpu = RNG.choice(list(GPUS.keys()), p=[0.45, 0.12, 0.13, 0.12, 0.08, 0.04, 0.04, 0.02])
    screen_size = round(RNG.choice([13.3, 14.0, 15.6, 16.0, 17.3], p=[0.2, 0.2, 0.35, 0.15, 0.1]), 1)
    resolution = RNG.choice(["1920x1080", "2560x1440", "3840x2160", "2560x1600"], p=[0.55, 0.25, 0.10, 0.10])
    weight = round(RNG.normal(1.8 if screen_size < 15 else 2.3, 0.35), 2)
    weight = max(0.9, weight)
    battery_hours = round(RNG.normal(9, 2.5), 1)
    battery_hours = max(3, battery_hours)
    os_ = RNG.choice(OS_LIST, p=[0.55, 0.12, 0.13, 0.10, 0.10])
    if brand == "Apple":
        os_ = "macOS"
    laptop_type = RNG.choice(TYPES, p=[0.25, 0.2, 0.35, 0.12, 0.08])
    touchscreen = RNG.choice([0, 1], p=[0.8, 0.2])
    year = int(RNG.choice([2021, 2022, 2023, 2024, 2025], p=[0.05, 0.15, 0.25, 0.30, 0.25]))

    base_price = 280
    price = base_price
    price *= BRANDS[brand]
    price *= CPUS[cpu]
    price += ram * 9.5
    price += storage_capacity * 0.18 * STORAGE_TYPES[storage_type]
    price += GPUS[gpu]
    price += (screen_size - 13.3) * 18
    price += {"1920x1080": 0, "2560x1600": 60, "2560x1440": 90, "3840x2160": 220}[resolution]
    price += touchscreen * 70
    price += (year - 2021) * 15
    price += RNG.normal(0, 60)  # ruido de mercado
    price = max(220, round(price, 2))

    rows.append({
        "brand": brand, "laptop_type": laptop_type, "cpu": cpu, "ram_gb": ram,
        "storage_gb": storage_capacity, "storage_type": storage_type, "gpu": gpu,
        "screen_size_in": screen_size, "resolution": resolution, "touchscreen": touchscreen,
        "weight_kg": weight, "battery_hours": battery_hours, "os": os_, "release_year": year,
        "price_usd": price,
    })

df = pd.DataFrame(rows)
df.to_csv("data/raw/laptops.csv", index=False)
print(f"Dataset generado: data/raw/laptops.csv ({len(df)} filas, {df.shape[1]} columnas)")
print(df.head())
