# RetainIQ: Step-by-Step Walkthrough (Interview Prep)

Plain-language explanation of every step, the numbers it produced, and questions you are
likely to be asked. Every number here comes from `python scripts/run_pipeline.py`.

---

## The 30-second pitch

> "I built a churn early-warning system. I cleaned 7,043 customer records, used SQL to find where
> churn concentrates, trained four models and picked logistic regression because it caught 78% of
> churners and is easy to explain. I grouped customers into Low/Medium/High risk: the High group is
> 25% of customers but holds 59% of all churners. Then I designed a test-vs-control experiment for
> a retention offer on that group. Under a 20% assumed effect, retention rose from 35.7% to 49.3%
> (p < 0.0001) at about $148 per customer saved. I shipped it as a Streamlit dashboard and an Excel MIS."

---

## Step 1: Data cleaning

**What I did, simply:** made the data trustworthy before analysing it.

| Problem | Why it matters | Fix |
|---|---|---|
| `TotalCharges` stored as text | You can't add up or average text | Converted to numbers with `pd.to_numeric` |
| 11 blank `TotalCharges` | Hidden missing values (pandas doesn't flag `" "` as missing) | All 11 had `tenure = 0` → new customers never billed → set to **0** |
| `SeniorCitizen` was 0/1, others Yes/No | Inconsistent encoding | Converted to Yes/No |
| `Churn` was Yes/No | Models need numbers | 1 = left, 0 = stayed |
| Duplicates | Would double-count customers | Checked rows and IDs: **0 found** |

**Result:** 7,043 rows × 21 columns in, 7,043 rows × 24 columns out, with 0 missing values. The 3
new columns are tenure group, charge bucket and number of add-ons.

**Interview Q: Why fill with 0 and not the average?**
Because the blanks weren't random. Every one was a brand-new customer with 0 months of tenure. They
truly have been billed $0. Filling with the average (~$2,280) would invent money they never paid.
*Always ask why data is missing before choosing how to fill it.*

---

## Step 2: SQL analysis

**What I did:** loaded the clean table into SQLite and wrote 8 queries (`sql/churn_analysis.sql`).

Key findings:
* Overall churn **26.5%** (1,869 of 7,043)
* Month-to-month **42.7%** vs. two-year **2.8%**: commitment is the biggest lever
* First year **47.4%** vs. 4+ years **9.5%**: churn is an early-life problem
* Electronic check **45.3%** vs. automatic payments **15–17%**
* Month-to-month + fiber + e-check = **60.4%** churn across 1,307 customers
* Month-to-month customers account for **$120.8K of the $139.1K** monthly revenue lost

**Interview Q: Why SQL if you have pandas?**
In a real company the data sits in a database, and analysts answer questions with SQL. SQL also makes
the logic easy for others to review and rerun. SQL concepts used: `GROUP BY`, `CASE WHEN`, `HAVING`
(only segments with 100+ customers, so small groups don't mislead), and ordering.

---

## Step 3: Risk segmentation

**What I did:** gave every customer a churn score (0–1) from the model and bucketed it:
Low < 0.30 ≤ Medium < 0.70 ≤ High.

| Group | Customers | Actual churn | Share of all churners |
|---|---|---|---|
| Low | 2,973 | 4.7% | 7.4% |
| Medium | 2,324 | 27.2% | 33.9% |
| High | 1,746 | 62.8% | 58.7% |

**The business point:** call 1 in 4 customers and you reach ~59% of the people about to leave.

**Interview Q: Isn't scoring customers the model trained on cheating?**
Yes, so I didn't. I used **out-of-fold** scoring: the data is split into 5 parts, and each customer is
scored by a model trained on the *other* 4 parts. Every score is like a score on a new customer.

**Interview Q: How did you pick 0.30 / 0.70?**
They're business-friendly round numbers that give three clearly different groups
(4.7% / 27% / 63% churn). In practice I'd set the High cut-off to match the call team's capacity.

---

## Step 4: Modelling

**What I did:** trained Logistic Regression, Random Forest, XGBoost and LightGBM on 80% of the data
(5,634 customers) and tested on the other 20% (1,409), with the split stratified so both halves have 26.5% churn.

**Metrics in one line each:**
* **Accuracy:** % of all predictions that were right (misleading when only 27% churn)
* **Precision:** of the people we flagged, % who really churned (wasted calls = low precision)
* **Recall:** of the people who really churned, % we caught (missed churners = low recall)
* **F1:** balance of precision and recall
* **ROC-AUC:** how well the model *ranks* churners above stayers (0.5 = coin flip, 1 = perfect)
* **F2:** like F1 but recall counts twice, which is what I used to choose

**Why recall matters more:** missing a churner loses ~12 months of revenue (~$80/month × 12 ≈ $960).
A wasted call costs ~$20. A missed churner is ~50× worse than a false alarm.

**Why Logistic Regression won:**
1. I ranked by **5-fold cross-validated F2 on the training set**, never by the test set. Picking a model
   by its test score is a subtle form of cheating.
2. LR ranked first (0.718). But all four were within one standard deviation (~±0.02–0.03), which is
   effectively a tie.
3. In a tie, take the **simplest, most explainable** model. In fintech and collections, regulators and
   managers need to know *why* a customer was flagged.

Be honest if asked: XGBoost had slightly higher test recall (0.786 vs. 0.778) and LightGBM the
highest ROC-AUC (0.844 vs. 0.839). These gaps are smaller than the noise between CV folds.

**Class imbalance:** only 27% churn, so a lazy model could predict "stays" for everyone and score 73%
accuracy. I used `class_weight="balanced"` (`scale_pos_weight` for XGBoost) to make the models pay
attention to churners.

**A problem I caught and fixed (good interview story):**
My first SHAP output said *"higher monthly bill lowers churn"*, which contradicted my own SQL
(cheap plans churn 10.9%, expensive plans ~33–38%). The cause was **multicollinearity**:
* `TotalCharges` ≈ tenure × monthly bill (correlation 0.83 with tenure)
* `MonthlyCharges` is almost exactly the sum of the services the customer has (R² = 0.999)

When inputs duplicate each other, the model splits their effect in odd ways. I dropped the redundant
columns. CV F2 moved only 0.722 → 0.718 (within noise), and the explanations now match the data.

**Top 10 drivers (SHAP):** SHAP measures how much each factor pushed a customer's score up or down.
Tenure is the strongest (long-time customers stay), then contract type, internet type, paperless
billing, phone lines, dependents, electronic check, and online security.

---

## Step 5: Retention experiment (A/B test)

**Setup:** 1,746 High-risk customers, randomly split 873 test / 873 control.
* Test → retention offer + reminder call (~$20 per customer)
* Control → nothing

**Why a control group?** Some High-risk customers would stay anyway. Without a control you'd take
credit for them. The control tells you what would have happened without the offer.

**How the simulation works (be upfront about this):** control outcomes are the real data. For the
test group I assumed the offer saves 20% of customers who would otherwise churn. That is the only
invented number, and it's in `config.py`.

**Balance check:** avg churn score 0.822 vs. 0.817, avg bill $80.58 vs. $81.40, so randomisation
produced comparable groups.

**Result:**
| | Test | Control |
|---|---|---|
| Retained | 430 / 873 = **49.3%** | 312 / 873 = **35.7%** |

* Uplift **+13.5 percentage points** (95% CI +8.9 to +18.1)
* **Two-proportion z-test: z = 5.71, p < 0.0001** (chi-square gives the same p-value for a 2×2 table)
* **p-value in plain words:** if the offer truly did nothing, a gap this big would almost never happen by chance
* Customers saved ≈ 0.135 × 873 ≈ **118**
* Campaign cost = 873 × $20 = **$17,460**
* **Cost per customer saved = $17,460 / 118 = $148**
* Each saved customer pays ~$81/month, so 12 months ≈ $970. That is far above $148, so the offer pays off.

**Sensitivity (what if I'm wrong about the 20%?):**
| Assumed effect | Uplift | p-value | Cost per save |
|---|---|---|---|
| 0% (A/A) | +2.9 pp | 0.22, not significant | – |
| 5% | +5.4 pp | 0.02 | $371 |
| 10% | +7.9 pp | 0.0007 | $253 |
| 20% | +13.5 pp | < 0.0001 | $148 |
| 30% | +19.2 pp | < 0.0001 | $104 |

The **0% row is an A/A check**: with no real effect, chance alone gave a +2.9 pp gap, and the test
correctly said "not significant". That number is a reminder that small effects need bigger samples.

**Decision:** roll out to High-risk customers, but keep a 10% holdout so the effect keeps being measured.

**Interview Q: Percentage points vs. percent?**
35.7% → 49.3% is +13.5 percentage points, which is a 38% relative increase. Always say which one you mean.

---

## Step 6: Dashboard (Streamlit + Plotly)

Four pages: KPIs → churn by segment → score one customer → experiment results.
Page 3 loads the saved model, takes a customer's details, and returns their probability, risk
group and the recommended action (call / nudge / no action).

## Step 7: Excel MIS

* **Pivot tab:** churn % by Contract × Tenure group, built with `AVERAGEIFS`/`COUNTIFS` so it
  recalculates when data changes (a native PivotTable can be added via Insert → PivotTable).
* **Lookup tab:** type a customer ID → `XLOOKUP` returns risk group, score, contract and tenure.
* I opened the file in Excel and checked every formula result against pandas. They matched.

**Interview Q: Why XLOOKUP over VLOOKUP?** It can look left, it doesn't break when columns are
inserted, it has a built-in "not found" value, and it defaults to exact match.

## Step 8: Engineering

Code split into small modules, one command runs everything, all assumptions live in `config.py`, and
7 tests check cleaning, SQL, risk cut-offs, z-test math (against a hand calculation and statsmodels)
and the cost-per-save formula.

---

## Quick-fire questions

* **Biggest churn driver?** Tenure. New customers on month-to-month contracts are the riskiest.
* **What would you do differently with real data?** Add behaviour over time (late payments, usage
  drops, complaints), run a real pilot, and build an uplift model to find who *responds* to offers,
  not just who is at risk.
* **How does this apply to collections?** Same structure: predict who will miss payment, bucket by
  risk, prioritise agent calls on the high bucket, and A/B test call scripts or payment-plan offers
  with a control group, measuring cost per rupee recovered.
* **Why not deep learning?** 7K rows of tabular data: tree models and LR are the right tools, and
  explainability matters in finance.
