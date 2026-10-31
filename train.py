"""Train the Ridge model and save it to model.bin."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "AmesHousing.csv"
MODEL = HERE / "model.bin"
META = HERE / "model_metadata.json"


def load_data():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["SalePrice", "Id"])
    y = np.log1p(df["SalePrice"])
    return X, y


def make_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("preprocessor", make_preprocessor(X)),
        ("model", Ridge()),
    ])
    search = GridSearchCV(
        pipe,
        {"model__alpha": [0.1, 1, 10, 50, 100, 200]},
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    pred = model.predict(X_test)
    log_rmse = mean_squared_error(y_test, pred) ** 0.5
    usd_rmse = mean_squared_error(np.expm1(y_test), np.expm1(pred)) ** 0.5
    r2 = r2_score(y_test, pred)

    with open(MODEL, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model": "Ridge",
        "alpha": search.best_params_["model__alpha"],
        "cv_log_rmse": round(float(-search.best_score_), 4),
        "test_log_rmse": round(float(log_rmse), 4),
        "test_rmse_usd": round(float(usd_rmse), 2),
        "test_r2": round(float(r2), 4),
        "features": X.columns.tolist(),
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps({k: v for k, v in metadata.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    train()
