"""Step 5: simulated retention experiment (test vs. control) on High-risk customers.

How the simulation works
------------------------
1. Take every High-risk customer and randomly split them 50/50.
   * Test    -> gets a retention offer + reminder call.
   * Control -> gets nothing (business as usual).
2. Control outcome = what really happened in the data (churned or not).
3. Test outcome   = what really happened, except each customer who churned
   has a `relative_churn_reduction` chance (assumed 20%) of being saved by
   the offer. This is the ASSUMED uplift. It is the only invented part.
4. Compare retention rates with a two-proportion z-test (and a chi-square
   test, which gives the same p-value for a 2x2 table).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from . import config


@dataclass
class ExperimentResult:
    n_test: int
    n_control: int
    retained_test: int
    retained_control: int
    retention_test_pct: float
    retention_control_pct: float
    uplift_pp: float
    ci_low_pp: float
    ci_high_pp: float
    z_stat: float
    p_value_z: float
    chi2_stat: float
    p_value_chi2: float
    significant: bool
    customers_saved: float
    campaign_cost: float
    cost_per_customer_saved: float
    revenue_protected: float
    roi_pct: float
    power_pct: float
    balance_score_test: float
    balance_score_control: float
    balance_charge_test: float
    balance_charge_control: float
    assumed_relative_churn_reduction: float
    offer_cost_per_customer: float
    months_retained: int

    def to_dict(self):
        return asdict(self)


def two_proportion_ztest(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    """Two-sided z-test for p_a vs p_b, using the pooled proportion."""
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se
    return float(z), float(2 * norm.sf(abs(z)))


def assign_groups(customers: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = customers.copy()
    idx = rng.permutation(len(out))
    group = np.where(idx < len(out) // 2, "Test", "Control")
    out["group"] = group
    return out


def simulate_outcomes(assigned: pd.DataFrame, relative_churn_reduction: float,
                      seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    out = assigned.copy()
    would_churn = out[config.TARGET].to_numpy() == 1
    saved_by_offer = (out["group"].to_numpy() == "Test") & would_churn & (rng.random(len(out)) < relative_churn_reduction)
    out["churned_after"] = (would_churn & ~saved_by_offer).astype(int)
    out["retained"] = 1 - out["churned_after"]
    return out


def analyse(outcomes: pd.DataFrame, cfg: dict = config.EXPERIMENT) -> ExperimentResult:
    t = outcomes[outcomes["group"] == "Test"]
    c = outcomes[outcomes["group"] == "Control"]
    n_t, n_c = len(t), len(c)
    r_t, r_c = int(t["retained"].sum()), int(c["retained"].sum())
    p_t, p_c = r_t / n_t, r_c / n_c

    z, p_z = two_proportion_ztest(r_t, n_t, r_c, n_c)
    chi2, p_chi2, _, _ = chi2_contingency([[r_t, n_t - r_t], [r_c, n_c - r_c]], correction=False)

    # 95% confidence interval for the difference (unpooled standard error).
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    zcrit = norm.ppf(1 - cfg["alpha"] / 2)

    # Business maths: how many extra customers did the offer keep?
    saved = diff * n_t
    cost = cfg["offer_cost_per_customer"] * n_t
    cost_per_saved = cost / saved if saved > 0 else float("inf")
    revenue = saved * t["MonthlyCharges"].mean() * cfg["months_retained"]

    # Power: chance this sample size detects the assumed effect at alpha=0.05.
    p_c_true = c["retained"].mean()
    p_t_true = 1 - (1 - p_c_true) * (1 - cfg["relative_churn_reduction"])
    se_power = np.sqrt(p_t_true * (1 - p_t_true) / n_t + p_c_true * (1 - p_c_true) / n_c)
    power = norm.sf(zcrit - (p_t_true - p_c_true) / se_power)

    return ExperimentResult(
        n_test=n_t, n_control=n_c, retained_test=r_t, retained_control=r_c,
        retention_test_pct=100 * p_t, retention_control_pct=100 * p_c,
        uplift_pp=100 * diff, ci_low_pp=100 * (diff - zcrit * se), ci_high_pp=100 * (diff + zcrit * se),
        z_stat=z, p_value_z=p_z, chi2_stat=float(chi2), p_value_chi2=float(p_chi2),
        significant=p_z < cfg["alpha"], customers_saved=float(saved), campaign_cost=float(cost),
        cost_per_customer_saved=float(cost_per_saved), revenue_protected=float(revenue),
        roi_pct=float(100 * (revenue - cost) / cost), power_pct=float(100 * power),
        # Balance check: before treatment, the two groups should look alike.
        balance_score_test=float(t["churn_score"].mean()), balance_score_control=float(c["churn_score"].mean()),
        balance_charge_test=float(t["MonthlyCharges"].mean()), balance_charge_control=float(c["MonthlyCharges"].mean()),
        assumed_relative_churn_reduction=cfg["relative_churn_reduction"],
        offer_cost_per_customer=cfg["offer_cost_per_customer"], months_retained=cfg["months_retained"],
    )


def run_experiment(scored: pd.DataFrame, cfg: dict = config.EXPERIMENT):
    high = scored[scored["risk_group"] == "High"]
    outcomes = simulate_outcomes(assign_groups(high), cfg["relative_churn_reduction"])
    return analyse(outcomes, cfg), outcomes


def sensitivity(scored: pd.DataFrame, reductions=(0.0, 0.05, 0.10, 0.20, 0.30), cfg: dict = config.EXPERIMENT) -> pd.DataFrame:
    """Re-run the experiment for several assumed uplifts: 'what if the offer works less well?'
    The 0% row is an A/A check: any 'uplift' there is pure chance from the random split."""
    rows = []
    for red in reductions:
        r, _ = run_experiment(scored, {**cfg, "relative_churn_reduction": red})
        no_effect = red == 0  # cost per save / ROI mean nothing when the offer does nothing
        rows.append({"assumed_churn_reduction_pct": 100 * red, "uplift_pp": r.uplift_pp,
                     "p_value": r.p_value_z, "significant": r.significant,
                     "cost_per_customer_saved": np.nan if no_effect else r.cost_per_customer_saved,
                     "roi_pct": np.nan if no_effect else r.roi_pct})
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def readout_markdown(r: ExperimentResult, high_risk_count: int, sens: pd.DataFrame | None = None) -> str:
    sens_md = ""
    if sens is not None:
        lines = ["| Assumed churn reduction | Uplift (pp) | p-value | Significant? | Cost per save | ROI |",
                 "|---|---|---|---|---|---|"]
        for s in sens.itertuples(index=False):
            ok = np.isfinite(s.cost_per_customer_saved)
            cps = f"${s.cost_per_customer_saved:,.0f}" if ok else "-"
            roi = f"{s.roi_pct:,.0f}%" if ok else "-"
            lines.append(f"| {s.assumed_churn_reduction_pct:.0f}% | {s.uplift_pp:+.1f} | {fmt_p(s.p_value)} | "
                         f"{'Yes' if s.significant else 'No'} | {cps} | {roi} |")
        noise = sens.loc[sens["assumed_churn_reduction_pct"] == 0, "uplift_pp"]
        noise_txt = (f" Here chance alone produced {noise.iloc[0]:+.1f} pp (not significant), so the measured "
                     "uplift at small assumed effects is partly luck: a real test should be sized for that."
                     if len(noise) else "")
        sens_md = ("\n**Sensitivity: what if the offer works less well than assumed?**\n\n"
                   + "\n".join(lines) + "\n\n*The 0% row is an A/A check: the offer does nothing, so any "
                   "difference is pure chance from the random split." + noise_txt + "*\n")
    decision = ("**Roll out** the offer to all High-risk customers, and keep a 10% holdout "
                "control so the effect is still measured month to month."
                if r.significant and r.roi_pct > 0 else
                "**Do not roll out yet.** The result is not significant or not profitable. "
                "Re-test with a bigger sample or a cheaper offer.")
    return f"""# Experiment Readout: High-Risk Retention Offer

> **Simulated experiment.** Customer outcomes for the control group are real
> (from the dataset). The test group's uplift is an **assumption**
> ({r.assumed_relative_churn_reduction:.0%} of would-be churners saved). This readout shows
> how a real test would be designed, measured and decided.

## 1. Hypothesis
Giving High-risk customers a retention offer (a bill credit plus a reminder call, about
${r.offer_cost_per_customer:.0f} per customer all-in)
will **raise their 1-period retention rate** compared with doing nothing.

* H0 (null): retention(test) = retention(control)
* H1: retention(test) != retention(control) (two-sided, alpha = 0.05)

## 2. Setup
| Item | Value |
|---|---|
| Population | {high_risk_count:,} High-risk customers (churn score >= {config.RISK_BINS[2]:.2f}) |
| Split | Random 50/50 (seed 42) |
| Test group | {r.n_test:,} customers: offer + call |
| Control group | {r.n_control:,} customers: no contact |
| Primary metric | Retention rate (did not churn) |
| Test used | Two-proportion z-test (chi-square check) |
| Power at assumed effect | {min(r.power_pct, 99.9):.1f}% (chance of detecting the assumed effect) |

**Balance check (before treatment).** Randomisation worked if the groups look alike:
| | Test | Control |
|---|---|---|
| Avg churn score | {r.balance_score_test:.3f} | {r.balance_score_control:.3f} |
| Avg monthly charge | ${r.balance_charge_test:.2f} | ${r.balance_charge_control:.2f} |

## 3. Result
| Metric | Test | Control |
|---|---|---|
| Customers | {r.n_test:,} | {r.n_control:,} |
| Retained | {r.retained_test:,} | {r.retained_control:,} |
| Retention rate | **{r.retention_test_pct:.1f}%** | **{r.retention_control_pct:.1f}%** |

* **Uplift:** +{r.uplift_pp:.1f} percentage points (95% CI {r.ci_low_pp:+.1f} to {r.ci_high_pp:+.1f} pp)
* **z = {r.z_stat:.2f}, p-value {fmt_p(r.p_value_z)}** (exact {r.p_value_z:.2e}; chi-square gives the same, chi2 = {r.chi2_stat:.1f}):
  {"statistically significant" if r.significant else "not statistically significant"} at 5%
* **Customers saved:** ~{r.customers_saved:.0f}
* **Campaign cost:** ${r.campaign_cost:,.0f} ({r.n_test:,} x ${r.offer_cost_per_customer:.0f})
* **Cost per customer saved:** **${r.cost_per_customer_saved:,.2f}**
* **Revenue protected ({r.months_retained} months):** ${r.revenue_protected:,.0f}, ROI {r.roi_pct:,.0f}%
{sens_md}
## 4. Decision
{decision}

**Next steps**
1. Run it for real with a holdout and track retention at 30/60/90 days.
2. Test cheaper treatments (SMS, email) against the call to lower cost per save.
3. Try the offer on the Medium-risk group as its own experiment.
"""
