"""
Shared feature engineering logic for the churn model.

Import this from both the training notebook and the Streamlit app so the
features the model was trained on always match the features it's served with.
"""

import numpy as np
import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CONTRACT_RISK_MAP = {"Month-to-month": 2, "One year": 1, "Two year": 0}

RAW_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used in the training notebook.

    Expects a DataFrame with the raw Telco Customer Churn columns
    (customerID and Churn are not required and are ignored if present).
    """
    df = df.copy()

    # --- Basic cleaning ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Collapse redundant "No internet/phone service" categories down to "No".
    # These values are perfectly correlated with InternetService == "No" / PhoneService == "No",
    # so keeping them separate just duplicates the same signal across six columns without adding
    # information (validated experimentally: ROC-AUC difference was -0.0006, within CV noise of
    # +/-0.011). Collapsed is the default -- must match the notebook's Section 3.1 exactly.
    for col in SERVICE_COLS[1:]:  # skip PhoneService itself, it has no "No X service" variant
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # --- Engineered features (mirrors notebook section 3.2) ---
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"],
    )

    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    df["num_services"] = df[SERVICE_COLS].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    df["long_term_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["has_streaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    df["tenure_charges_interaction"] = df["tenure"] * df["MonthlyCharges"]
    df["charges_per_service"] = df["MonthlyCharges"] / df["num_services"].replace(0, 1)
    df["contract_risk_score"] = df["Contract"].map(CONTRACT_RISK_MAP)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # Drop identifiers/target if present — the model pipeline expects features only
    df = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])

    return df