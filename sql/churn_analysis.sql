-- RetainIQ: churn analysis queries (SQLite)
-- Table: customers  (cleaned data, Churn = 1 if the customer left)
-- Each query is separated by a "-- name:" line so Python can run them one by one.

-- name: q1_overall_churn
-- Business question: what share of the customer base did we lose?
SELECT
    COUNT(*)                                   AS customers,
    SUM(Churn)                                 AS churned,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers;

-- name: q2_churn_by_contract
-- Are customers without a long-term commitment more likely to leave?
SELECT
    Contract,
    COUNT(*)                                   AS customers,
    SUM(Churn)                                 AS churned,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- name: q3_churn_by_tenure_group
-- How does churn change as the relationship gets older?
SELECT
    tenure_group,
    COUNT(*)                                   AS customers,
    SUM(Churn)                                 AS churned,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- name: q4_churn_by_payment_method
-- Payment behaviour as a churn signal (similar to collections risk signals).
SELECT
    PaymentMethod,
    COUNT(*)                                   AS customers,
    SUM(Churn)                                 AS churned,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- name: q5_churn_by_charge_bucket
-- Do higher monthly bills push customers away?
SELECT
    charge_bucket,
    COUNT(*)                                   AS customers,
    ROUND(AVG(MonthlyCharges), 2)              AS avg_monthly_charge,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers
GROUP BY charge_bucket
ORDER BY MIN(MonthlyCharges);

-- name: q6_contract_x_tenure
-- Two-way view: which contract + tenure combinations are hotspots?
SELECT
    Contract,
    tenure_group,
    COUNT(*)                                   AS customers,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers
GROUP BY Contract, tenure_group
ORDER BY Contract, MIN(tenure);

-- name: q7_monthly_revenue_at_risk
-- How much monthly revenue walked out the door, by contract type?
SELECT
    Contract,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS lost_monthly_revenue,
    ROUND(SUM(MonthlyCharges), 2)                                      AS total_monthly_revenue,
    ROUND(100.0 * SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END)
          / SUM(MonthlyCharges), 2)                                    AS revenue_churn_pct
FROM customers
GROUP BY Contract
ORDER BY lost_monthly_revenue DESC;

-- name: q8_top_risk_profiles
-- Highest-churn customer profiles with at least 100 customers (enough to trust).
SELECT
    Contract,
    InternetService,
    PaymentMethod,
    COUNT(*)                                   AS customers,
    ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct
FROM customers
GROUP BY Contract, InternetService, PaymentMethod
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC
LIMIT 5;
