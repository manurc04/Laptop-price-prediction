"""Funciones de carga y preprocesado del dataset de portátiles."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "laptops.csv"

NUMERIC_FEATURES = [
    "ram_gb", "storage_gb", "screen_size_in", "weight_kg", "touchscreen",
]
CATEGORICAL_FEATURES = [
    "brand", "laptop_type", "cpu", "storage_type", "gpu", "resolution", "os",
]
TARGET = "price_usd"


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga el dataset crudo desde CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: duplicados, nulos y tipos."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Crea el ColumnTransformer (escalado numérico + one-hot categórico)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_features_and_target(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y
