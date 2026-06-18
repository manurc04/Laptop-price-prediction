"""Carga el modelo entrenado y predice el precio de un portátil nuevo."""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_price(specs: dict, model=None) -> float:
    """specs debe incluir todas las columnas usadas en el entrenamiento."""
    if model is None:
        model = load_model()
    df = pd.DataFrame([specs])
    return float(model.predict(df)[0])


if __name__ == "__main__":
    example = {
        "ram_gb": 16, "storage_gb": 512, "screen_size_in": 15.6, "weight_kg": 1.9,
        "touchscreen": 0,
        "brand": "Dell", "laptop_type": "Ultrabook", "cpu": "Intel Core i7",
        "storage_type": "SSD", "gpu": "Intel", "resolution": "1920x1080",
        "os": "Windows 10",
    }
    price = predict_price(example)
    print(f"Precio estimado: ${price:,.2f}")
