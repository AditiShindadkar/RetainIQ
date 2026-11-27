"""Step 3: group customers into Low / Medium / High risk from the churn score."""
from __future__ import annotations

import pandas as pd

from . import config


def assign_risk_group(scores) -> pd.Series:
    return pd.cut(pd.Series(scores), bins=config.RISK_BINS, labels=config.RISK_LABELS,
                  right=False, include_lowest=True).astype(str)


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Size, share and real churn rate of each risk group."""
    g = df.groupby("risk_group", observed=True)
    out = pd.DataFrame({
        "customers": g.size(),
        "share_of_base_pct": 100 * g.size() / len(df),
        "actual_churn_rate_pct": 100 * g[config.TARGET].mean(),
        "avg_churn_score": g["churn_score"].mean(),
        "avg_monthly_charge": g["MonthlyCharges"].mean(),
        "churners_captured": g[config.TARGET].sum(),
    }).reindex(config.RISK_LABELS)
    out["share_of_all_churners_pct"] = 100 * out["churners_captured"] / df[config.TARGET].sum()
    return out.reset_index().rename(columns={"index": "risk_group"})
