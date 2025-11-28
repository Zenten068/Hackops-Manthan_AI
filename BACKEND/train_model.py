import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# -------- CONFIG --------
DATA_PATH = Path("ecommerce_fraud_train.csv")  # CSV in same folder
MODEL_PATH = Path("fraud_model.pkl")
TARGET_COL = "is_fraud"  # change if your target column name is different


def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV. Available columns: {list(df.columns)}")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_preprocessor(X: pd.DataFrame):
    """
    Numeric & categorical columns taken from the problem statement.
    We use only those that actually exist in your CSV.
    """
    numeric_cols = [
        "hour_of_day", "day_of_week", "month",
        "time_since_last_tx", "tx_velocity",
        "tx_amount", "account_age_days", "lifetime_spent",
        "num_failed_tx", "credit_score",
        "network_centrality", "shared_devices", "shared_ips",
        "desc_length", "special_chars", "num_urls",
        "sentiment", "isolation_score", "lof_score", "ae_recon_error",
    ]

    categorical_cols = [
        "user_id", "device_type", "browser",
        "ip_country", "vpn_detected",
        "category_l1", "category_l2", "category_l3",
        "language",
    ]

    # Keep only columns that actually exist
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # ----- Simple feature engineering -----

    # Log-transform amount
    if "tx_amount" in X.columns:
        X["tx_amount_log"] = np.log1p(np.clip(X["tx_amount"], a_min=0, a_max=None))
        numeric_cols.append("tx_amount_log")

    # Time flags
    if "hour_of_day" in X.columns:
        X["is_night"] = X["hour_of_day"].between(0, 6).astype(int)
        numeric_cols.append("is_night")

    if "day_of_week" in X.columns:
        X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)
        numeric_cols.append("is_weekend")

    # Missing value flags
    for col in ["credit_score", "account_age_days"]:
        if col in X.columns:
            flag_col = f"{col}_missing"
            X[flag_col] = X[col].isna().astype(int)
            numeric_cols.append(flag_col)

    # ----- Transformers -----
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, X, numeric_cols, categorical_cols


def build_model(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,  # handle imbalance
        random_state=42,
    )


def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Train shape: {X.shape}, fraud rate: {y.mean():.4f}")

    print("Building preprocessor...")
    preprocessor, X_proc, num_cols, cat_cols = build_preprocessor(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class imbalance handling
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight (neg/pos) = {scale_pos_weight:.2f}")

    model = build_model(scale_pos_weight)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_proba = clf.predict_proba(X_val)[:, 1]

    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.8, 40):
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"Best F1 on validation: {best_f1:.4f} at threshold={best_thr:.3f}")

    print("Refitting on full data...")
    clf.fit(X_proc, y)

    artifact = {
        "pipeline": clf,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "best_threshold": float(best_thr),
    }

    print(f"Saving model to {MODEL_PATH} ...")
    joblib.dump(artifact, MODEL_PATH)
    print("Done. fraud_model.pkl created.")


if __name__ == "__main__":
    main()
