# src/run_analysis.py
# Delta Coding Test: Default of Credit Card Clients by Jalaan

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_PATH = os.path.join("data", "raw", "default_of_credit_card_clients.csv")
FIG_DIR = os.path.join("reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"Saved figure: {path}")

def load_data_csv(path: str) -> pd.DataFrame:
    """
    My CSV includes an extra header row (X1, X2...) and the real column names start on row 2.
    Therefore header=1 is correct for THIS file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\nPlace it at: data/raw/default_of_credit_card_clients.csv")

    df = pd.read_csv(path, header=1)  # IMPORTANT: correct for your file structure
    df = df.rename(columns={"default payment next month": "DEFAULT"})
    return df

def basic_qa(df: pd.DataFrame) -> None:
    print("\n=== BASIC QA ===")
    print("Shape:", df.shape)
    print("Missing values (total):", int(df.isna().sum().sum()))
    print("Duplicates:", int(df.duplicated().sum()))
    print("\nTarget distribution (0=No default, 1=Default):")
    print(df["DEFAULT"].value_counts().sort_index())
    print("Default rate:", round(df["DEFAULT"].mean(), 4))

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop ID (identifier)
    - Consolidate undocumented EDUCATION codes (0,5,6 -> 4 "others")
    - Consolidate undocumented MARRIAGE code (0 -> 3 "others")
    """
    print("\n=== CLEANING ===")
    dfc = df.copy()

    if "ID" in dfc.columns:
        dfc = dfc.drop(columns=["ID"])

    if "EDUCATION" in dfc.columns:
        before = sorted(dfc["EDUCATION"].unique())
        dfc["EDUCATION"] = dfc["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        after = sorted(dfc["EDUCATION"].unique())
        print("EDUCATION unique before:", before)
        print("EDUCATION unique after :", after)

    if "MARRIAGE" in dfc.columns:
        before = sorted(dfc["MARRIAGE"].unique())
        dfc["MARRIAGE"] = dfc["MARRIAGE"].replace({0: 3})
        after = sorted(dfc["MARRIAGE"].unique())
        print("MARRIAGE unique before:", before)
        print("MARRIAGE unique after :", after)

    # Validate target
    bad = set(dfc["DEFAULT"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"DEFAULT has unexpected values: {bad}")

    return dfc

def eda(df: pd.DataFrame) -> None:
    """
    EDA outputs saved to reports/figures.
    Keep plots simple and interpretable for a panel.
    """
    print("\n=== EDA ===")

    # 1) Target distribution
    ax = df["DEFAULT"].value_counts().sort_index().plot(kind="bar")
    ax.set_title("Target Distribution (0=No Default, 1=Default)")
    ax.set_xlabel("DEFAULT")
    ax.set_ylabel("Count")
    savefig("01_target_distribution.png")
    plt.close()

    # 2) Default rate by categories
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        if col in df.columns:
            rate = df.groupby(col)["DEFAULT"].mean().sort_index()
            ax = rate.plot(kind="bar")
            ax.set_title(f"Default Rate by {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Default Rate")
            savefig(f"02_default_rate_by_{col.lower()}.png")
            plt.close()

    # 3) Payment status
    if "PAY_0" in df.columns:
        rate = df.groupby("PAY_0")["DEFAULT"].mean().sort_index()
        ax = rate.plot(kind="bar")
        ax.set_title("Default Rate by Most Recent Payment Status (PAY_0)")
        ax.set_xlabel("PAY_0")
        ax.set_ylabel("Default Rate")
        savefig("03_default_rate_by_pay0.png")
        plt.close()

    # 4) Amount distributions (skew check)
    amount_cols = [c for c in df.columns if "AMT" in c or c == "LIMIT_BAL"]
    df[amount_cols].hist(bins=30, figsize=(14, 10))
    savefig("04_amount_distributions.png")
    plt.close()

    # 5) Correlation scan (directional)
    corr = df.corr(numeric_only=True)["DEFAULT"].sort_values(ascending=False)
    print("\nTop correlations with DEFAULT (directional, not causal):")
    print(corr.head(12))

def build_and_eval_models(df: pd.DataFrame) -> None:
    """
    Two-model strategy:
    - Logistic Regression (interpretable baseline; good for panel explanation)
    - Random Forest (nonlinear comparison)
    Metrics:
    - ROC-AUC (ranking quality)
    - PR-AUC (informative under class imbalance)
    """
    print("\n=== MODELING ===")

    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    categorical = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Logistic Regression Pipeline
    preprocess_lr = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop"
    )

    logreg = Pipeline(steps=[
        ("preprocess", preprocess_lr),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])

    logreg.fit(X_train, y_train)
    proba = logreg.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n--- Logistic Regression ---")
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba), 4))
    print(classification_report(y_test, pred))

    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title("LogReg Confusion Matrix (threshold=0.5)")
    savefig("05_logreg_confusion_matrix.png")
    plt.close()

    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("LogReg ROC Curve")
    savefig("06_logreg_roc_curve.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("LogReg Precision-Recall Curve")
    savefig("07_logreg_pr_curve.png")
    plt.close()

    # Random Forest Pipeline
    preprocess_rf = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop"
    )

    rf = Pipeline(steps=[
        ("preprocess", preprocess_rf),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1
        )),
    ])

    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= 0.5).astype(int)

    print("\n--- Random Forest ---")
    print("ROC-AUC:", round(roc_auc_score(y_test, proba_rf), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba_rf), 4))
    print(classification_report(y_test, pred_rf))

    ConfusionMatrixDisplay.from_predictions(y_test, pred_rf)
    plt.title("RF Confusion Matrix (threshold=0.5)")
    savefig("08_rf_confusion_matrix.png")
    plt.close()

def main():
    df = load_data_csv(RAW_PATH)
    basic_qa(df)

    dfc = clean_data(df)
    eda(dfc)
    build_and_eval_models(dfc)

    print("\n Phase 1 complete and data cleaned.")
    print("Charts saved to:", FIG_DIR)

if __name__ == "__main__":
    main()