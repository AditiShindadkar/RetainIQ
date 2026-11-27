"""Basic tests.  Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retainiq import cleaning, config, experiment, segmentation, sql_analysis  # noqa: E402


def tiny_raw() -> pd.DataFrame:
    base = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0,
    }
    rows = [
        {**base, "customerID": "A", "tenure": 0, "TotalCharges": " ", "Churn": "No"},
        {**base, "customerID": "B", "tenure": 10, "TotalCharges": "500.5", "Churn": "Yes", "SeniorCitizen": 1},
        {**base, "customerID": "B", "tenure": 10, "TotalCharges": "500.5", "Churn": "Yes", "SeniorCitizen": 1},
        {**base, "customerID": "C", "tenure": 60, "TotalCharges": "3000", "Churn": "No"},
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cleaning
def test_clean_fixes_types_missing_and_duplicates():
    raw = tiny_raw()
    before = cleaning.quality_report(raw)
    assert before["TotalCharges_missing_or_blank"] == 1
    assert before["duplicate_rows"] == 1

    df = cleaning.clean(raw)
    after = cleaning.quality_report(df)
    assert len(df) == 3
    assert after["duplicate_rows"] == 0 and after["duplicate_customer_ids"] == 0
    assert df["TotalCharges"].dtype == float
    assert df["TotalCharges"].isna().sum() == 0
    assert df.loc[df.customerID == "A", "TotalCharges"].item() == 0.0  # tenure-0 -> nothing billed yet
    assert set(df["Churn"]) == {0, 1}
    assert set(df["SeniorCitizen"]) == {"Yes", "No"}


def test_tenure_groups_cover_edges():
    groups = cleaning.tenure_group(pd.Series([0, 12, 13, 24, 25, 48, 49, 72]))
    assert groups.tolist() == ["0-12 months"] * 2 + ["13-24 months"] * 2 + ["25-48 months"] * 2 + ["49-72 months"] * 2


# ------------------------------------------------------------------ SQL
def test_sql_file_has_6_to_8_queries_and_they_run(tmp_path):
    queries = sql_analysis.parse_queries()
    assert 6 <= len(queries) <= 8
    df = cleaning.clean(tiny_raw())
    db = tmp_path / "t.db"
    sql_analysis.load_to_sqlite(df, db)
    out = sql_analysis.run_queries(db)
    assert out["q1_overall_churn"]["customers"][0] == 3
    assert out["q1_overall_churn"]["churn_rate_pct"][0] == pytest.approx(33.33, abs=0.01)


# ------------------------------------------------------------------ segmentation
def test_risk_group_cutoffs():
    groups = segmentation.assign_risk_group([0.0, 0.29, 0.30, 0.69, 0.70, 1.0])
    assert groups.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]


# ------------------------------------------------------------------ experiment
def test_ztest_matches_hand_calculation_and_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest
    z, p = experiment.two_proportion_ztest(60, 100, 45, 100)
    pooled = 105 / 200
    z_hand = (0.60 - 0.45) / np.sqrt(pooled * (1 - pooled) * (2 / 100))
    assert z == pytest.approx(z_hand)
    z_sm, p_sm = proportions_ztest([60, 45], [100, 100])
    assert z == pytest.approx(z_sm) and p == pytest.approx(p_sm)


def test_experiment_split_is_50_50_and_no_effect_means_no_uplift():
    rng = np.random.default_rng(0)
    n = 1000
    scored = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)], "risk_group": "High",
        "Churn": rng.integers(0, 2, n), "churn_score": rng.random(n), "MonthlyCharges": 80.0,
    })
    assigned = experiment.assign_groups(scored)
    assert (assigned["group"] == "Test").sum() == n // 2
    # With 0% assumed effect the test group is never changed, so customers saved
    # can only come from random group differences, and outcomes equal the raw labels.
    out = experiment.simulate_outcomes(assigned, relative_churn_reduction=0.0)
    assert (out["churned_after"] == out["Churn"]).all()
    # With 100% effect, every test customer is retained.
    out_full = experiment.simulate_outcomes(assigned, relative_churn_reduction=1.0)
    assert out_full.loc[out_full.group == "Test", "retained"].all()


def test_cost_per_customer_saved_formula():
    outcomes = pd.DataFrame({
        "group": ["Test"] * 100 + ["Control"] * 100,
        "retained": [1] * 60 + [0] * 40 + [1] * 50 + [0] * 50,
        "churn_score": 0.8, "MonthlyCharges": 100.0,
    })
    cfg = {**config.EXPERIMENT, "offer_cost_per_customer": 20.0}
    r = experiment.analyse(outcomes, cfg)
    assert r.customers_saved == pytest.approx(10)          # (60% - 50%) x 100
    assert r.cost_per_customer_saved == pytest.approx(200)  # $2,000 / 10
