"""
Customer Churn App — Streamlit

Two clearly separated pages, chosen from the sidebar:
  1. "Explore the Data" — descriptive EDA on historical customers. No model involved.
  2. "Predict Churn Risk" — uses the trained blended model to score customers you provide.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocess import engineer_features, RAW_COLUMNS  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

CHURN_COLORS = {"No": "#2ca02c", "Yes": "#d62728"}

st.set_page_config(page_title="Customer Churn App", page_icon="📉", layout="wide")


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return {
        "xgb": joblib.load(MODEL_DIR / "champion_xgb_pipeline.pkl"),
        "lgbm": joblib.load(MODEL_DIR / "champion_lgbm_pipeline.pkl"),
        "rf": joblib.load(MODEL_DIR / "champion_rf_pipeline.pkl"),
        "logreg": joblib.load(MODEL_DIR / "champion_logreg_pipeline.pkl"),
        "weights": joblib.load(MODEL_DIR / "blend_weights.pkl"),
        "threshold": joblib.load(MODEL_DIR / "decision_threshold.pkl"),
    }


@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_FILE)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn_Numeric"] = df["Churn"].map({"Yes": 1, "No": 0})
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["Addon_Count"] = (df[addons] == "Yes").sum(axis=1)
    return df


def predict_blend(artifacts, X_engineered: pd.DataFrame) -> np.ndarray:
    w = artifacts["weights"]
    return (
        w["xgb"] * artifacts["xgb"].predict_proba(X_engineered)[:, 1]
        + w["lgbm"] * artifacts["lgbm"].predict_proba(X_engineered)[:, 1]
        + w["rf"] * artifacts["rf"].predict_proba(X_engineered)[:, 1]
        + w["logreg"] * artifacts["logreg"].predict_proba(X_engineered)[:, 1]
    )


def risk_tier(p: float) -> str:
    if p < 0.3:
        return "Low"
    elif p < 0.6:
        return "Medium"
    return "High"


def suggested_action(tier: str) -> str:
    return {
        "Low": "No action needed — monitor at next billing cycle.",
        "Medium": "Proactive check-in: highlight underused features or a loyalty perk.",
        "High": "Priority outreach: offer a retention deal (discount, contract upgrade, or bundled service).",
    }[tier]


def top_shap_drivers(artifacts, X_engineered: pd.DataFrame, n=5):
    pipe = artifacts["xgb"]
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X_engineered)
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    feature_names = list(preprocessor.transformers_[0][2]) + list(cat_features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    row = shap_values[0]
    order = np.argsort(np.abs(row))[::-1][:n]
    drivers = pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "impact": [row[i] for i in order],
    })
    drivers["direction"] = np.where(drivers["impact"] > 0, "↑ increases risk", "↓ decreases risk")
    return drivers


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📉 Customer Churn App")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Explore the Data", "🔍 Predict Churn Risk"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown(
    "**📊 Explore the Data**\n"
    "Descriptive patterns in past customers. No model, just history.\n\n"
    "**🔍 Predict Churn Risk**\n"
    "Uses the trained ML model to score a customer you provide."
)


# ===========================================================================
# PAGE 1 — EXPLORE THE DATA (descriptive, no model)
# ===========================================================================
if page == "📊 Explore the Data":
    st.title("📊 Explore the Data")
    st.info(
        "This page shows **descriptive statistics from historical customers** — it explains "
        "*why* certain customers tend to churn. Nothing on this page uses the trained model; "
        "for individual risk scores, switch to **Predict Churn Risk** in the sidebar.",
        icon="ℹ️",
    )

    if not DATA_FILE.exists():
        st.error(
            f"Couldn't find the dataset at `{DATA_FILE}`. Place the Telco CSV in `data/raw/` "
            "to enable this page."
        )
        st.stop()

    raw_df = load_raw_data()

    with st.sidebar.expander("🔧 Filter the data"):
        contract_filter = st.multiselect(
            "Contract type", raw_df["Contract"].unique().tolist(),
            default=raw_df["Contract"].unique().tolist(),
        )
        internet_filter = st.multiselect(
            "Internet service", raw_df["InternetService"].unique().tolist(),
            default=raw_df["InternetService"].unique().tolist(),
        )

    filtered = raw_df[
        raw_df["Contract"].isin(contract_filter) & raw_df["InternetService"].isin(internet_filter)
    ]
    st.caption(f"Showing {len(filtered):,} of {len(raw_df):,} customers based on your filters.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(filtered):,}")
    m2.metric("Churn Rate", f"{filtered['Churn_Numeric'].mean():.1%}")
    m3.metric("Avg. Monthly Charges", f"${filtered['MonthlyCharges'].mean():.2f}")
    m4.metric("Avg. Tenure", f"{filtered['tenure'].mean():.1f} mo")

    st.divider()

    # ROW 1: Target Mix & Lifecycle Patterns
    c1, c2 = st.columns(2)
    with c1:
        churn_counts = filtered["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]

        fig_pie = px.pie(
            churn_counts,
            names="Churn",
            values="Count",
            title="<b>Overall Customer Attrition Mix (Churn Proportion)</b>",
            color="Churn",
            color_discrete_map=CHURN_COLORS,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextorientation="horizontal",
            marker=dict(line=dict(color="#ffffff", width=0.5)),
        )
        fig_pie.update_layout(
            height=480,
            margin=dict(t=50, b=60, l=20, r=20),
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_hist = px.histogram(
            filtered,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=30,
            marginal="box",
            title="<b>Customer Lifespan & Retentiveness (Tenure Distribution)</b>",
            color_discrete_map=CHURN_COLORS,
            opacity=0.65,
            labels={"tenure": "Account Age (Months)", "count": "Customer Volume"},
        )
        fig_hist.update_traces(marker=dict(line=dict(color="#ffffff", width=0.5)))
        fig_hist.update_layout(
            height=520,
            margin=dict(t=50, b=90, l=20, r=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", title=dict(standoff=15)),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ROW 2: Conversion Framework Profiles & Operational Touchpoints
    c3, c4 = st.columns(2)
    with c3:
        # 100% Normalized Percentage Stacked Bars accurately chart operational segment risks
        contract_churn = pd.crosstab(filtered["Contract"], filtered["Churn"], normalize="index").reset_index()
        contract_churn_melt = contract_churn.melt(id_vars="Contract", var_name="Churn", value_name="Proportion")

        fig_contract = px.bar(
            contract_churn_melt,
            x="Contract",
            y="Proportion",
            color="Churn",
            title="<b>Attrition Risk Breakdown by Commitment Strategy (Contract Type)</b>",
            color_discrete_map=CHURN_COLORS,
            labels={"Proportion": "Percentage Composition"},
        )
        fig_contract.update_yaxes(tickformat=".0%")
        fig_contract.update_layout(
            height=480,
            margin=dict(t=50, b=80, l=20, r=20),
            xaxis=dict(title=dict(standoff=15)),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    with c4:
        # Sorted Horizontal Layout ensures longer payment string details never clip on small screens
        pm_churn = pd.crosstab(filtered["PaymentMethod"], filtered["Churn"]).reset_index()
        pm_churn["Total"] = pm_churn["Yes"] + pm_churn["No"]
        pm_churn = pm_churn.sort_values(by="Total", ascending=True)

        fig_payment = px.bar(
            pm_churn,
            y="PaymentMethod",
            x=["No", "Yes"],
            title="<b>Operational Pipeline & Volume Spread by Settlement Option</b>",
            color_discrete_map={"No": CHURN_COLORS["No"], "Yes": CHURN_COLORS["Yes"]},
            labels={"value": "Total Customer Volume", "PaymentMethod": "", "variable": "Churn Status"},
        )
        fig_payment.update_layout(
            height=480,
            margin=dict(t=50, b=80, l=20, r=20),
            barmode="stack",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_payment, use_container_width=True)

    # ROW 3: Cost Threshold Spreads & Feature Bundling Logic
    c5, c6 = st.columns(2)
    with c5:
        fig_box = px.box(
            filtered,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Financial Cost Threshold Variance by Attrition Status",
            color_discrete_map=CHURN_COLORS,
            points="outliers",
            labels={"MonthlyCharges": "Monthly Billing Cost ($)", "Churn": "Did Customer Churn?"},
        )
        fig_box.update_traces(boxmean=True)  # Draw a clean horizontal indicator line mapping means
        fig_box.update_layout(margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with c6:
        addon_churn = filtered.groupby("Addon_Count")["Churn_Numeric"].mean().reset_index()
        # Color scale inversion strategy tracks high risk (red) vs diminishing risk (green)
        fig_addons = px.bar(
            addon_churn,
            x="Addon_Count",
            y="Churn_Numeric",
            title="Calculated Churn Probability Curve via Product Bundling",
            color="Churn_Numeric",
            color_continuous_scale="RdYlGn_r",
            labels={"Addon_Count": "Number of Active Internet Add-on Utilities", "Churn_Numeric": "Churn Probability"},
        )
        fig_addons.update_yaxes(tickformat=".0%")
        fig_addons.update_coloraxes(showscale=False)
        fig_addons.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        fig_addons.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_addons, use_container_width=True)

    # ROW 4: Holistic Co-Linearity Diagnostics
    st.subheader("Correlation Between Key Numeric Signals")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Numeric", "Addon_Count"]
    corr_df = filtered[numeric_cols].copy()
    corr_df["Contract_MonthToMonth"] = (filtered["Contract"] == "Month-to-month").astype(int)
    corr_df["Payment_ElectronicCheck"] = (filtered["PaymentMethod"] == "Electronic check").astype(int)
    corr_df["Internet_FiberOptic"] = (filtered["InternetService"] == "Fiber optic").astype(int)
    matrix = corr_df.corr()
    fig_heatmap = px.imshow(
        matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Holistic Feature Correlation Matrix (Targeting Core Attrition Drivers)",
        labels=dict(color="Correlation Strength"),
    )
    fig_heatmap.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    with st.expander("📌 Key takeaways from this data"):
        st.markdown(
            "- Month-to-month contracts and Electronic check payments both associate strongly "
            "with higher churn.\n"
            "- Churn risk is highest in the first 12 months and drops sharply the longer a "
            "customer stays.\n"
            "- Customers with more bundled add-on services churn less.\n"
            "- Fiber optic internet customers churn more than DSL or no-internet customers."
        )


# ===========================================================================
# PAGE 2 — PREDICT CHURN RISK (model-driven)
# ===========================================================================
else:
    st.title("🔍 Predict Churn Risk")
    st.success(
        "This page runs the **trained model** (a blend of XGBoost, LightGBM, Random Forest, "
        "and Logistic Regression) to score customers — either one at a time or in bulk. "
        "For historical patterns instead of live predictions, switch to **Explore the Data** "
        "in the sidebar.",
        icon="🤖",
    )

    try:
        artifacts = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found in `models/`. Run the training notebook's final "
            "section (Save Model Artifacts) first, so `models/*.pkl` exists next to this app."
        )
        st.stop()

    tab_single, tab_batch = st.tabs(["🔍 Single Customer", "📂 Batch Scoring"])

    with tab_single:
        st.subheader("Enter customer details")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            has_phone = phone_service == "Yes"
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], disabled=not has_phone)
            if not has_phone:
                multiple_lines = "No"

            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            has_internet = internet_service != "No"
            online_security = st.selectbox("Online Security", ["No", "Yes"], disabled=not has_internet)
            online_backup = st.selectbox("Online Backup", ["No", "Yes"], disabled=not has_internet)
            device_protection = st.selectbox("Device Protection", ["No", "Yes"], disabled=not has_internet)
            if not has_internet:
                online_security = "No"
                online_backup = "No"
                device_protection = "No"

        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes"], disabled=not has_internet)
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], disabled=not has_internet)
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], disabled=not has_internet)
            if not has_internet:
                tech_support = "No"
                streaming_tv = "No"
                streaming_movies = "No"
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        col4, col5 = st.columns(2)
        with col4:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        with col5:
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=float(monthly_charges * max(tenure, 1)), step=1.0
            )

        if st.button("Predict Churn Risk", type="primary"):
            raw = pd.DataFrame([{
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner, "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone_service, "MultipleLines": multiple_lines,
                "InternetService": internet_service, "OnlineSecurity": online_security,
                "OnlineBackup": online_backup, "DeviceProtection": device_protection,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies, "Contract": contract,
                "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            }])

            X_eng = engineer_features(raw)
            proba = predict_blend(artifacts, X_eng)[0]
            tier = risk_tier(proba)

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f"{proba:.1%}")
            r2.metric("Risk Tier", tier)
            r3.metric("Decision Threshold", f"{artifacts['threshold']:.2f}")

            color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]
            st.markdown(f"**Suggested action:** :{color}[{suggested_action(tier)}]")

            st.subheader("Top factors driving this prediction")
            drivers = top_shap_drivers(artifacts, X_eng)
            st.dataframe(
                drivers.style.format({"impact": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )

    with tab_batch:
        st.subheader("Score a file of customers")
        st.caption(
            "Upload a CSV with the same columns as the Telco Customer Churn dataset "
            "(customerID and Churn columns are optional and will be ignored)."
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            raw_batch = pd.read_csv(uploaded)
            missing_cols = [c for c in RAW_COLUMNS if c not in raw_batch.columns]

            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {missing_cols}")
            else:
                X_eng_batch = engineer_features(raw_batch)
                proba_batch = predict_blend(artifacts, X_eng_batch)

                results = pd.DataFrame({
                    "customerID": raw_batch["customerID"] if "customerID" in raw_batch.columns
                    else range(len(raw_batch)),
                    "churn_probability": proba_batch,
                    "risk_tier": [risk_tier(p) for p in proba_batch],
                    "predicted_churn": (proba_batch >= artifacts["threshold"]).astype(int),
                }).sort_values("churn_probability", ascending=False).reset_index(drop=True)

                st.success(f"Scored {len(results)} customers.")

                tier_counts = results["risk_tier"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("High Risk", int(tier_counts.get("High", 0)))
                c2.metric("Medium Risk", int(tier_counts.get("Medium", 0)))
                c3.metric("Low Risk", int(tier_counts.get("Low", 0)))

                st.dataframe(results, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download scored results as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="churn_risk_scores.csv",
                    mime="text/csv",
                )