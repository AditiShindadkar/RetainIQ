"""RetainIQ dashboard.   Run:  streamlit run app/streamlit_app.py

Pages
  1. Overview        - KPIs: total customers, churn rate, high-risk count
  2. Churn drivers   - churn rate by contract, tenure, payment, charges, risk group
  3. Score a customer- enter details -> churn probability + risk group
  4. Experiment      - test vs. control retention result
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retainiq import config  # noqa: E402
from retainiq.cleaning import TENURE_LABELS, CHARGE_LABELS  # noqa: E402
from retainiq.experiment import fmt_p  # noqa: E402
from retainiq.segmentation import assign_risk_group  # noqa: E402

st.set_page_config(page_title="RetainIQ", page_icon="📉", layout="wide")

BLUE = "#2a78d6"
GRAY = "#8a8984"
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
INK = "#52514e"


@st.cache_data
def load_data():
    if not config.SCORED_CSV.exists():
        return None, None
    scored = pd.read_csv(config.SCORED_CSV)
    results = json.loads((config.REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    return scored, results


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)


def style(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(color=INK), showlegend=False, bargap=0.35)
    fig.update_yaxes(gridcolor="#ececea", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def churn_bar(df, col, order, title):
    g = df.groupby(col)["Churn"].agg(["mean", "size"]).reindex(order).reset_index()
    g["rate"] = 100 * g["mean"]
    fig = px.bar(g, x=col, y="rate", text=g["rate"].map("{:.1f}%".format),
                 custom_data=["size"], title=title, color_discrete_sequence=[BLUE])
    fig.update_traces(textposition="outside", marker_cornerradius=4,
                      hovertemplate="%{x}<br>Churn rate %{y:.1f}%<br>Customers %{customdata[0]:,}<extra></extra>")
    fig.update_yaxes(title="Churn rate (%)", range=[0, g["rate"].max() * 1.25])
    fig.update_xaxes(title=None, tickangle=0,
                     tickvals=g[col], ticktext=[str(v).replace(" (", "<br>(") for v in g[col]])
    return style(fig)


scored, results = load_data()
if scored is None:
    st.error("No outputs found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

st.sidebar.title("RetainIQ")
st.sidebar.caption("Churn risk & retention experiment")
PAGES = ["1. Overview", "2. Churn by segment", "3. Score a customer", "4. Experiment results"]
try:  # ?page=3 opens page 3 directly (handy for links and screenshots)
    start = min(max(int(st.query_params.get("page", 1)) - 1, 0), len(PAGES) - 1)
except ValueError:
    start = 0
page = st.sidebar.radio("Page", PAGES, index=start)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{results['best_model']}**  \nData: IBM Telco, {len(scored):,} customers")

# --------------------------------------------------------------------- page 1
if page.startswith("1"):
    st.title("Customer churn overview")
    high = scored[scored["risk_group"] == "High"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(scored):,}")
    c2.metric("Churn rate", f"{100 * scored['Churn'].mean():.1f}%")
    c3.metric("High-risk customers", f"{len(high):,}")
    c3.caption(f"{100 * len(high) / len(scored):.1f}% of the customer base")
    c4.metric("Monthly revenue lost", f"${scored.loc[scored.Churn == 1, 'MonthlyCharges'].sum():,.0f}")

    st.subheader("Risk groups")
    seg = pd.DataFrame(results["segments"])
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(
            x=seg["risk_group"], y=seg["customers"], marker_color=[RISK_COLORS[g] for g in seg["risk_group"]],
            marker_cornerradius=4, text=[f"{n:,} customers<br>{r:.1f}% churn" for n, r in
                                         zip(seg["customers"], seg["actual_churn_rate_pct"])],
            textposition="outside",
            hovertemplate="%{x} risk<br>%{y:,} customers<extra></extra>"))
        fig.update_layout(title="Customers per risk group (label shows actual churn rate)")
        fig.update_yaxes(title="Customers", range=[0, seg["customers"].max() * 1.3])
        st.plotly_chart(style(fig), width="stretch")
    with right:
        st.markdown("**What each group means**")
        st.markdown(
            "- **High** (score ≥ 0.70): call first, offer retention deal\n"
            "- **Medium** (0.30–0.70): low-cost nudges (email/SMS)\n"
            "- **Low** (< 0.30): no action needed")
        top = seg.set_index("risk_group").loc["High"]
        st.info(f"The High-risk group is **{top.share_of_base_pct:.0f}% of customers** but holds "
                f"**{top.share_of_all_churners_pct:.0f}% of all churners**, so outreach can be focused.")
        st.dataframe(seg[["risk_group", "customers", "actual_churn_rate_pct", "share_of_all_churners_pct"]]
                     .rename(columns={"risk_group": "Risk", "customers": "Customers",
                                      "actual_churn_rate_pct": "Churn %", "share_of_all_churners_pct": "% of churners"})
                     .round(1), hide_index=True, width="stretch")

# --------------------------------------------------------------------- page 2
elif page.startswith("2"):
    st.title("Churn by segment")
    contracts = st.multiselect("Filter contract type", sorted(scored["Contract"].unique()),
                               default=sorted(scored["Contract"].unique()))
    df = scored[scored["Contract"].isin(contracts)] if contracts else scored
    st.caption(f"{len(df):,} customers selected")
    a, b = st.columns(2)
    a.plotly_chart(churn_bar(df, "Contract", ["Month-to-month", "One year", "Two year"], "By contract type"),
                   width="stretch")
    b.plotly_chart(churn_bar(df, "tenure_group", TENURE_LABELS, "By tenure group"), width="stretch")
    a, b = st.columns(2)
    pay_order = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False).index.tolist()
    a.plotly_chart(churn_bar(df, "PaymentMethod", pay_order, "By payment method"), width="stretch")
    b.plotly_chart(churn_bar(df, "charge_bucket", CHARGE_LABELS, "By monthly-charge bucket"), width="stretch")

    st.subheader("Contract × tenure heatmap (churn %)")
    pv = (df.pivot_table(index="Contract", columns="tenure_group", values="Churn", aggfunc="mean")
          .reindex(columns=TENURE_LABELS) * 100)
    fig = px.imshow(pv, text_auto=".1f", color_continuous_scale=["#cde2fb", "#2a78d6", "#0d366b"],
                    aspect="auto", labels=dict(color="Churn %"))
    fig.update_traces(hovertemplate="%{y} · %{x}<br>Churn %{z:.1f}%<extra></extra>")
    st.plotly_chart(style(fig, 300), width="stretch")

    st.subheader("Top 10 churn drivers (SHAP)")
    drivers = pd.read_csv(config.REPORTS_DIR / "top10_churn_drivers.csv")
    st.dataframe(drivers[["plain_english", "mean_abs_shap"]]
                 .rename(columns={"plain_english": "Driver", "mean_abs_shap": "Importance"}),
                 hide_index=True, width="stretch")

# --------------------------------------------------------------------- page 3
elif page.startswith("3"):
    st.title("Score a customer")
    st.caption("Enter a customer's details to get their churn probability and risk group.")
    bundle = load_model()
    pipe, features = bundle["pipeline"], bundle["features"]

    def opts(col):
        return sorted(scored[col].dropna().unique().tolist())

    with st.form("score"):
        c1, c2, c3 = st.columns(3)
        inp = {}
        with c1:
            st.markdown("**Account**")
            inp["tenure"] = st.slider("Tenure (months)", 0, 72, 3)
            inp["Contract"] = st.selectbox("Contract", opts("Contract"), index=opts("Contract").index("Month-to-month"))
            inp["PaymentMethod"] = st.selectbox("Payment method", opts("PaymentMethod"),
                                                index=opts("PaymentMethod").index("Electronic check"))
            inp["PaperlessBilling"] = st.selectbox("Paperless billing", ["Yes", "No"])
        with c2:
            st.markdown("**Customer**")
            inp["gender"] = st.selectbox("Gender", opts("gender"))
            inp["SeniorCitizen"] = st.selectbox("Senior citizen", ["No", "Yes"])
            inp["Partner"] = st.selectbox("Partner", ["No", "Yes"])
            inp["Dependents"] = st.selectbox("Dependents", ["No", "Yes"])
            inp["PhoneService"] = st.selectbox("Phone service", ["Yes", "No"])
            inp["MultipleLines"] = st.selectbox("Multiple lines", opts("MultipleLines"))
        with c3:
            st.markdown("**Internet & add-ons**")
            inp["InternetService"] = st.selectbox("Internet service", opts("InternetService"),
                                                  index=opts("InternetService").index("Fiber optic"))
            for col, label in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                               ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
                               ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]:
                inp[col] = st.selectbox(label, opts(col))
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted or st.query_params.get("demo") == "1":  # ?demo=1 scores the default example
        # Keep inputs consistent: no internet -> add-ons are "No internet service".
        if inp["InternetService"] == "No":
            for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                inp[col] = "No internet service"
        if inp["PhoneService"] == "No":
            inp["MultipleLines"] = "No phone service"
        prob = float(pipe.predict_proba(pd.DataFrame([inp])[features])[0, 1])
        group = assign_risk_group([prob]).iloc[0]
        actions = {"High": "Priority call + retention offer within 48 hours.",
                   "Medium": "Send a low-cost nudge (email/SMS) and offer a contract upgrade.",
                   "Low": "No action. Keep in the normal service journey."}
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[group]
        m1, m2 = st.columns([1, 2])
        m1.metric("Churn probability", f"{prob:.0%}")
        m1.markdown(f"### {icon} {group} risk")
        m2.markdown(f"**Recommended action:** {actions[group]}")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                   "steps": [{"range": [0, 30], "color": "#d6f0d6"}, {"range": [30, 70], "color": "#fdeccc"},
                             {"range": [70, 100], "color": "#f5d0d0"}]}))
        gauge.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=20))
        m2.plotly_chart(gauge, width="stretch")
        if inp["InternetService"] == "No" or inp["PhoneService"] == "No":
            st.caption("Note: add-on / multiple-line answers were set automatically to match the services chosen.")

# --------------------------------------------------------------------- page 4
else:
    st.title("Retention experiment: test vs. control")
    e = results["experiment"]
    st.warning(f"Simulated experiment: control outcomes are real data; the test-group effect assumes the offer "
               f"saves {e['assumed_relative_churn_reduction']:.0%} of would-be churners.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retention: test", f"{e['retention_test_pct']:.1f}%", f"+{e['uplift_pp']:.1f} pp vs control")
    c2.metric("Retention: control", f"{e['retention_control_pct']:.1f}%")
    c3.metric("p-value (z-test)", fmt_p(e["p_value_z"]), "significant" if e["significant"] else "not significant",
              delta_color="normal" if e["significant"] else "inverse")
    c4.metric("Cost per customer saved", f"${e['cost_per_customer_saved']:,.0f}")

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(
            x=["Test (offer + call)", "Control (no contact)"],
            y=[e["retention_test_pct"], e["retention_control_pct"]],
            marker_color=[BLUE, GRAY], marker_cornerradius=4,
            text=[f"{e['retention_test_pct']:.1f}%<br>{e['retained_test']:,} of {e['n_test']:,}",
                  f"{e['retention_control_pct']:.1f}%<br>{e['retained_control']:,} of {e['n_control']:,}"],
            textposition="outside",
            error_y=None, hovertemplate="%{x}<br>Retention %{y:.1f}%<extra></extra>"))
        fig.update_layout(title=f"Retention rate, High-risk customers (95% CI of uplift: "
                                f"{e['ci_low_pp']:+.1f} to {e['ci_high_pp']:+.1f} pp)")
        fig.update_yaxes(title="Retained (%)", range=[0, 70])
        st.plotly_chart(style(fig, 380), width="stretch")
    with right:
        st.markdown("**Business case**")
        # "\$" stops Streamlit reading $...$ as a maths formula.
        st.markdown(
            f"- Customers in test: **{e['n_test']:,}** × \\${e['offer_cost_per_customer']:.0f} = "
            f"**\\${e['campaign_cost']:,.0f}** campaign cost\n"
            f"- Extra customers kept: **~{e['customers_saved']:.0f}**\n"
            f"- Cost per customer saved: **\\${e['cost_per_customer_saved']:,.2f}**\n"
            f"- Revenue protected ({e['months_retained']} months): **\\${e['revenue_protected']:,.0f}**\n"
            f"- ROI: **{e['roi_pct']:,.0f}%**")
        st.markdown("**Balance check** (groups should look alike before treatment)")
        st.dataframe(pd.DataFrame({
            "": ["Avg churn score", "Avg monthly charge"],
            "Test": [f"{e['balance_score_test']:.3f}", f"${e['balance_charge_test']:.2f}"],
            "Control": [f"{e['balance_score_control']:.3f}", f"${e['balance_charge_control']:.2f}"]}),
            hide_index=True, width="stretch")

    st.subheader("What if the offer works less well?")
    sens = pd.DataFrame(results["experiment_sensitivity"])
    sens_view = pd.DataFrame({
        "Assumed churn reduction": sens["assumed_churn_reduction_pct"].map("{:.0f}%".format),
        "Uplift (pp)": sens["uplift_pp"].map("{:+.1f}".format),
        "p-value": sens["p_value"].map(fmt_p),
        "Significant": sens["significant"].map({True: "Yes", False: "No"}),
        "Cost per save": sens["cost_per_customer_saved"].map(lambda v: "—" if pd.isna(v) else f"${v:,.0f}"),
        "ROI": sens["roi_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}%")})
    st.dataframe(sens_view, hide_index=True, width="stretch")
    st.caption("0% row = A/A check: the offer does nothing, so any uplift there is pure chance from the random split.")
    with st.expander("Full experiment readout"):
        st.markdown((config.REPORTS_DIR / "experiment_readout.md").read_text(encoding="utf-8"))
