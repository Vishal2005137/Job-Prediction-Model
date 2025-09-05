import os
import math
import joblib
import pandas as pd
from typing import Tuple, List, Dict, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CSV_PATH = os.path.join(os.path.dirname(__file__), "rahulSet...csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_store.pkl")


DOMAIN_COLUMNS: List[str] = [
    "AI/ML",
    "CyberSecurity",
    "Cloud Computing",
    "Data Science & Analytics",
    "Software Dev",
    "DevOps & SRE",
    "Blockchain & Web3",
    "IoT",
    "AR/VR & Metaverse",
    "Quantum Computing",
]


def try_import_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
        return XGBRegressor
    except Exception:
        return None


def load_and_clean_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["Year", "Highest Salary (INR LPA)", "Vacancies Offered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in DOMAIN_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Vacancies Offered"]).reset_index(drop=True)
    return df


def derive_primary_domain(df: pd.DataFrame) -> pd.DataFrame:
    present_domains = [c for c in DOMAIN_COLUMNS if c in df.columns]
    if not present_domains:
        df["Domain"] = "Unknown"
        return df
    df["Domain"] = df[present_domains].idxmax(axis=1)
    return df


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ]
    )
    return preprocessor


def get_candidate_models():
    models = [
        ("RandomForest", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("LinearRegression", LinearRegression(n_jobs=None)),
    ]
    XGBRegressor = try_import_xgboost()
    if XGBRegressor is not None:
        models.append(("XGBoost", XGBRegressor(random_state=42, n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)))
    return models


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(((y_true - y_pred) ** 2).mean())
    return {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}


def choose_better_metric(a: Dict[str, float], b: Dict[str, float]) -> bool:
    if a["r2"] != b["r2"]:
        return a["r2"] > b["r2"]
    if a["rmse"] != b["rmse"]:
        return a["rmse"] < b["rmse"]
    return a["mae"] < b["mae"]


def train_for_target(
    df: pd.DataFrame,
    target_key: str,
    target_column: str,
    include_domain_feature: bool,
) -> Dict[str, object]:
    # Feature configuration
    base_features = ["Company Name", "Year", "Work Mode", "Highest Salary (INR LPA)"]
    feature_columns = base_features.copy()
    categorical_cols = ["Company Name", "Work Mode"]
    numeric_cols = ["Year", "Highest Salary (INR LPA)"]

    if include_domain_feature:
        if "Domain" not in df.columns:
            raise ValueError("Domain column missing. Call derive_primary_domain first.")
        feature_columns.append("Domain")
        categorical_cols.append("Domain")

    X = df[feature_columns].copy()
    y = pd.to_numeric(df[target_column], errors="coerce")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    best = {
        "name": None,
        "pipeline": None,
        "metrics": {"r2": -1e9, "mae": 1e9, "rmse": 1e9},
        "feature_columns": feature_columns,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_column": target_column,
    }

    for model_name, model in get_candidate_models():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        if choose_better_metric(metrics, best["metrics"]):
            best.update({
                "name": model_name,
                "pipeline": pipe,
                "metrics": metrics,
            })

    best["target_key"] = target_key
    return best


def train_all(csv_path: str = CSV_PATH, model_path: str = MODEL_PATH) -> Dict[str, object]:
    df = load_and_clean_dataset(csv_path)
    df = derive_primary_domain(df)

    model_store: Dict[str, object] = {"models": {}, "domains": DOMAIN_COLUMNS}

    # Total vacancies target
    total_model = train_for_target(
        df=df,
        target_key="total",
        target_column="Vacancies Offered",
        include_domain_feature=True,
    )
    model_store["models"]["total"] = total_model

    # Domain-specific targets
    for domain_col in [c for c in DOMAIN_COLUMNS if c in df.columns]:
        model_info = train_for_target(
            df=df,
            target_key=domain_col,
            target_column=domain_col,
            include_domain_feature=False,
        )
        model_store["models"][domain_col] = model_info

    joblib.dump(model_store, model_path)
    return model_store


def print_model_store_summary(model_store: Dict[str, object]) -> None:
    print("Saved models:")
    for key, info in model_store["models"].items():
        name = info["name"]
        m = info["metrics"]
        print(f"  - {key}: {name} | R2={m['r2']:.4f} MAE={m['mae']:.2f} RMSE={m['rmse']:.2f}")


if __name__ == "__main__":
    store = train_all()
    joblib.dump(store, MODEL_PATH)
    print(f"Saved model store to: {MODEL_PATH}")
    print_model_store_summary(store)


