"""Entrena varios modelos de regresión, los compara y guarda el mejor."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import (build_preprocessor, clean_data,
                                   get_features_and_target, load_data)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_candidate_models():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        )
    return models


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def main():
    df = clean_data(load_data())
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    results = {}
    fitted_pipelines = {}

    for name, model in get_candidate_models().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = evaluate(y_test, preds)
        fitted_pipelines[name] = pipe
        print(f"{name:18s} -> {results[name]}")

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nMejor modelo: {best_name} (R2={results[best_name]['R2']})")

    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\nModelo guardado en {MODELS_DIR / 'best_model.joblib'}")
    return results, best_name


if __name__ == "__main__":
    main()
