"""Step 1: data cleaning with pandas.

Problems found in the raw IBM Telco file:
  * TotalCharges is stored as text, and 11 rows hold a blank string " ".
    All 11 are brand-new customers (tenure = 0) who have not been billed yet,
    so the correct value is 0, not "unknown".
  * SeniorCitizen is 0/1 while every other yes/no column is "Yes"/"No".
  * Churn is "Yes"/"No"; models need 1/0.
  * Duplicates: checked on full rows and on customerID.
"""
from __future__ import annotations

import pandas as pd

from . import config

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

CHARGE_BINS = [0, 35, 70, 90, 1000]
CHARGE_LABELS = ["Low (<$35)", "Mid ($35-70)", "High ($70-90)", "Premium ($90+)"]

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> dict:
    """Counts used for the before/after comparison."""
    # Blank strings are "hidden" missing values: pandas does not count them as NaN.
    blanks = int(sum((df[c].astype(str).str.strip() == "").sum()
                     for c in df.select_dtypes(include="object").columns))
    tc = df["TotalCharges"]
    tc_missing = int(tc.isna().sum()) + int((tc.astype(str).str.strip() == "").sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()) + blanks,
        "TotalCharges_missing_or_blank": tc_missing,
        "TotalCharges_dtype": str(tc.dtype),
        "SeniorCitizen_dtype": str(df["SeniorCitizen"].dtype),
        "Churn_dtype": str(df[config.TARGET].dtype),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[config.ID_COL].duplicated().sum()),
    }


def tenure_group(tenure: pd.Series) -> pd.Series:
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)


def charge_bucket(monthly: pd.Series) -> pd.Series:
    return pd.cut(monthly, bins=CHARGE_BINS, labels=CHARGE_LABELS, right=False).astype(str)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Strip stray whitespace in every text column.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Duplicates: drop exact duplicate rows, then keep one row per customer.
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=config.ID_COL, keep="first")

    # 3. Wrong data type: TotalCharges text -> number. Blanks become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 4. Missing values: blank TotalCharges belong to tenure-0 customers who
    #    have not received a bill, so their total is 0.
    new_customer = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customer, "TotalCharges"] = 0.0
    # Any other gap (none in this file) falls back to tenure x monthly charge.
    other_gap = df["TotalCharges"].isna()
    df.loc[other_gap, "TotalCharges"] = df.loc[other_gap, "tenure"] * df.loc[other_gap, "MonthlyCharges"]

    # 5. Consistent encoding.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna(df["SeniorCitizen"])
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)

    # 6. Helper columns for SQL, pivots and the model.
    df["tenure_group"] = tenure_group(df["tenure"])
    df["charge_bucket"] = charge_bucket(df["MonthlyCharges"])
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)

    return df.reset_index(drop=True)


def run(save: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    raw = load_raw()
    before = quality_report(raw)
    cleaned = clean(raw)
    after = quality_report(cleaned)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(config.CLEAN_CSV, index=False)
    return cleaned, before, after
