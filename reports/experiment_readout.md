# Experiment Readout: High-Risk Retention Offer

> **Simulated experiment.** Customer outcomes for the control group are real
> (from the dataset). The test group's uplift is an **assumption**
> (20% of would-be churners saved). This readout shows
> how a real test would be designed, measured and decided.

## 1. Hypothesis
Giving High-risk customers a retention offer (a bill credit plus a reminder call, about
$20 per customer all-in)
will **raise their 1-period retention rate** compared with doing nothing.

* H0 (null): retention(test) = retention(control)
* H1: retention(test) != retention(control) (two-sided, alpha = 0.05)

## 2. Setup
| Item | Value |
|---|---|
| Population | 1,746 High-risk customers (churn score >= 0.70) |
| Split | Random 50/50 (seed 42) |
| Test group | 873 customers: offer + call |
| Control group | 873 customers: no contact |
| Primary metric | Retention rate (did not churn) |
| Test used | Two-proportion z-test (chi-square check) |
| Power at assumed effect | 99.9% (chance of detecting the assumed effect) |

**Balance check (before treatment).** Randomisation worked if the groups look alike:
| | Test | Control |
|---|---|---|
| Avg churn score | 0.822 | 0.817 |
| Avg monthly charge | $80.58 | $81.40 |

## 3. Result
| Metric | Test | Control |
|---|---|---|
| Customers | 873 | 873 |
| Retained | 430 | 312 |
| Retention rate | **49.3%** | **35.7%** |

* **Uplift:** +13.5 percentage points (95% CI +8.9 to +18.1 pp)
* **z = 5.71, p-value < 0.0001** (exact 1.11e-08; chi-square gives the same, chi2 = 32.6):
  statistically significant at 5%
* **Customers saved:** ~118
* **Campaign cost:** $17,460 (873 x $20)
* **Cost per customer saved:** **$147.97**
* **Revenue protected (12 months):** $114,096, ROI 553%

**Sensitivity: what if the offer works less well than assumed?**

| Assumed churn reduction | Uplift (pp) | p-value | Significant? | Cost per save | ROI |
|---|---|---|---|---|---|
| 0% | +2.9 | 0.2157 | No | - | - |
| 5% | +5.4 | 0.0208 | Yes | $371 | 160% |
| 10% | +7.9 | 0.0007 | Yes | $253 | 282% |
| 20% | +13.5 | < 0.0001 | Yes | $148 | 553% |
| 30% | +19.2 | < 0.0001 | Yes | $104 | 830% |

*The 0% row is an A/A check: the offer does nothing, so any difference is pure chance from the random split. Here chance alone produced +2.9 pp (not significant), so the measured uplift at small assumed effects is partly luck: a real test should be sized for that.*

## 4. Decision
**Roll out** the offer to all High-risk customers, and keep a 10% holdout control so the effect is still measured month to month.

**Next steps**
1. Run it for real with a holdout and track retention at 30/60/90 days.
2. Test cheaper treatments (SMS, email) against the call to lower cost per save.
3. Try the offer on the Medium-risk group as its own experiment.
