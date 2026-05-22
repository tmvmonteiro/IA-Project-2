import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifacts.pkl"
TARGET = "Price"
RANDOM_STATE = 50

NUMERICAL_FEATURES = [
    "TotalArea",
    "LivingArea",
    "ConstructionYear",
    "TotalRooms",
    "NumberOfBedrooms",
    "NumberOfBathrooms",
]

CATEGORICAL_FEATURES = [
    "Municipality",
    "Parish",
    "Type",
    "EnergyCertificate",
    "Parking",
    "Elevator",
    "Floor",
    "ConservationStatus",
]


def clean_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)

    important_columns = [
        "Price",
        "District",
        "City",
        "Town",
        "Type",
        "EnergyCertificate",
        "TotalArea",
        "Parking",
        "Elevator",
        "LivingArea",
        "NumberOfBathrooms",
    ]

    data_clean = data.dropna(subset=important_columns).copy()
    data_clean = data_clean.drop(
        columns=["EnergyEfficiencyLevel", "HasParking"],
        errors="ignore",
    )
    data_clean = data_clean[data_clean["Type"].isin(["House", "Apartment"])].copy()
    data_clean = data_clean[data_clean["District"] == "Porto"].copy()
    data_clean = data_clean.rename(columns={"City": "Municipality", "Town": "Parish"})
    data_clean = data_clean[[TARGET] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

    for col in NUMERICAL_FEATURES:
        data_clean[col] = data_clean[col].fillna(data_clean[col].mean())

    for col in CATEGORICAL_FEATURES:
        data_clean[col] = data_clean[col].where(data_clean[col].notna(), "Unknown").astype(str)

    for col in ["Price", "TotalArea", "LivingArea"]:
        q1 = data_clean[col].quantile(0.25)
        q3 = data_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_clean = data_clean[
            (data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)
        ].copy()

    return data_clean


def make_model():
    stacking_estimators = [
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=30,
                max_features=0.5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=50,
                random_state=50,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "xgboost",
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "lightgbm",
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]

    return StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=Ridge(alpha=10.0),
        cv=5,
        n_jobs=-1,
    )


def make_options(data_clean):
    options = {
        "numerical": {},
        "categorical": {},
        "parishesByMunicipality": {},
    }

    for col in NUMERICAL_FEATURES:
        series = data_clean[col]
        options["numerical"][col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "recommendedMin": float(series.quantile(0.01)),
            "recommendedMax": float(series.quantile(0.99)),
            "median": float(series.median()),
        }

    for col in CATEGORICAL_FEATURES:
        options["categorical"][col] = sorted(data_clean[col].dropna().astype(str).unique().tolist())

    for municipality, group in data_clean.groupby("Municipality"):
        options["parishesByMunicipality"][municipality] = sorted(
            group["Parish"].dropna().astype(str).unique().tolist()
        )

    return options


def main():
    data_clean = clean_data()
    x_raw = data_clean[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data_clean[TARGET].copy()

    x_encoded = pd.get_dummies(x_raw, columns=CATEGORICAL_FEATURES, drop_first=True)
    x_encoded.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", str(column))
        for column in x_encoded.columns
    ]
    feature_columns = list(x_encoded.columns)

    model = make_model()
    model.fit(x_encoded, y)

    artifact = {
        "model": model,
        "featureColumns": feature_columns,
        "numericalFeatures": NUMERICAL_FEATURES,
        "categoricalFeatures": CATEGORICAL_FEATURES,
        "options": make_options(data_clean),
        "trainingRows": int(len(data_clean)),
    }

    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print(f"Training rows: {len(data_clean):,}")
    print(f"Encoded features: {len(feature_columns):,}")


if __name__ == "__main__":
    main()
