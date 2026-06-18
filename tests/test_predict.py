from pathlib import Path

import pytest

from src.predict import MODEL_PATH, load_model, predict_price

SAMPLE_SPECS = {
    "ram_gb": 16, "storage_gb": 512, "screen_size_in": 15.6, "weight_kg": 1.9,
    "touchscreen": 0,
    "brand": "Dell", "laptop_type": "Ultrabook", "cpu": "Intel Core i7",
    "storage_type": "SSD", "gpu": "Intel", "resolution": "1920x1080",
    "os": "Windows 10",
}


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Modelo no entrenado todavía")
def test_predict_price_returns_positive_float():
    price = predict_price(SAMPLE_SPECS)
    assert isinstance(price, float)
    assert price > 0
