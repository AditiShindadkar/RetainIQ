"""Central settings: file paths, risk cut-offs and experiment assumptions.

Keeping every "magic number" here means a reviewer can see (and change)
all business assumptions in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "telco_clean.csv"
SCORED_CSV = PROCESSED_DIR / "customers_scored.csv"
DB_PATH = PROCESSED_DIR / "retainiq.db"
SQL_FILE = ROOT / "sql" / "churn_analysis.sql"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXCEL_PATH = REPORTS_DIR / "RetainIQ_MIS.xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Churn"
ID_COL = "customerID"

# Risk groups use the model's churn score.
#   Low    : score < 0.30
#   Medium : 0.30 <= score < 0.70
#   High   : score >= 0.70
RISK_BINS = [0.0, 0.30, 0.70, 1.0001]
RISK_LABELS = ["Low", "Medium", "High"]

# Retention experiment assumptions. These are assumptions, not measured facts.
EXPERIMENT = {
    # Share of would-be churners in the TEST group that the offer saves.
    # 20% relative churn reduction is a conservative, commonly quoted
    # range (10-30%) for targeted retention offers.
    "relative_churn_reduction": 0.20,
    # Cost per treated customer: a $15 bill credit + about $5 of agent time
    # for the reminder call.
    "offer_cost_per_customer": 20.0,
    # How many months of revenue a saved customer is assumed to keep paying.
    "months_retained": 12,
    "alpha": 0.05,
}
