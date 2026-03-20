import numpy as np
import pandas as pd


CONTINUOUS_COLUMNS = [
    "aspect",
    "elv",
    "flowAcc",
    "planCurv",
    "profCurv",
    "riverProx",
    "roadProx",
    "slope",
    "SPI",
    "TPI",
    "TRI",
    "TWI",
]
CATEGORICAL_COLUMNS = ["lithology", "soil"]
METADATA_COLUMNS = ["fid", "xcoord", "ycoord"]


def _make_dummy_frame(series, prefix, categories):
    data = {}
    numeric = pd.to_numeric(series, errors="coerce")
    for category in categories:
        column_name = f"{prefix}_{int(category)}"
        data[column_name] = (numeric == category).astype(np.float32)
    return pd.DataFrame(data, index=series.index)


def fit_preprocessor(df):
    continuous = df[CONTINUOUS_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    continuous_min = continuous.min(axis=0).to_numpy(dtype=np.float32)
    continuous_max = continuous.max(axis=0).to_numpy(dtype=np.float32)
    lithology_categories = sorted(
        pd.to_numeric(df["lithology"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    soil_categories = sorted(
        pd.to_numeric(df["soil"], errors="coerce").dropna().astype(int).unique().tolist()
    )

    feature_columns = (
        CONTINUOUS_COLUMNS
        + [f"lithology_{value}" for value in lithology_categories]
        + [f"soil_{value}" for value in soil_categories]
    )

    return {
        "continuous_columns": list(CONTINUOUS_COLUMNS),
        "continuous_min": continuous_min,
        "continuous_max": continuous_max,
        "lithology_categories": lithology_categories,
        "soil_categories": soil_categories,
        "feature_columns": feature_columns,
    }


def transform_with_preprocessor(df, preprocessor):
    continuous = df[preprocessor["continuous_columns"]].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    mins = np.asarray(preprocessor["continuous_min"], dtype=np.float32)
    maxs = np.asarray(preprocessor["continuous_max"], dtype=np.float32)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    continuous_scaled = (continuous.to_numpy(dtype=np.float32) - mins) / denom
    continuous_df = pd.DataFrame(
        continuous_scaled,
        columns=preprocessor["continuous_columns"],
        index=df.index,
    )

    lithology_df = _make_dummy_frame(
        df["lithology"],
        "lithology",
        preprocessor["lithology_categories"],
    )
    soil_df = _make_dummy_frame(
        df["soil"],
        "soil",
        preprocessor["soil_categories"],
    )

    features = pd.concat([continuous_df, lithology_df, soil_df], axis=1)
    return features.reindex(columns=preprocessor["feature_columns"], fill_value=0.0)


def fit_transform_preprocessor(df):
    preprocessor = fit_preprocessor(df)
    features = transform_with_preprocessor(df, preprocessor)
    return features, preprocessor
