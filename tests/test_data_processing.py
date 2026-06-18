import pandas as pd

from src.data_processing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                                   build_preprocessor, clean_data,
                                   get_features_and_target, load_data)


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_clean_data_has_no_duplicates():
    df = clean_data(load_data())
    assert df.duplicated().sum() == 0


def test_clean_data_has_no_nulls_in_target():
    df = clean_data(load_data())
    assert df["price_usd"].isna().sum() == 0


def test_get_features_and_target_shapes():
    df = clean_data(load_data())
    X, y = get_features_and_target(df)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(X) == len(y)


def test_build_preprocessor_fits_and_transforms():
    df = clean_data(load_data())
    X, _ = get_features_and_target(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
