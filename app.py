import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI-IDS Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
        .main { background-color: #f4f6f9; }
        h1 {
            color: #1a1a2e;
            font-size: 28px;
            font-weight: 700;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2, h3 { color: #16213e; font-size: 18px; font-weight: 600; margin-top: 20px; }
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        section[data-testid="stFileUploader"] {
            background-color: #ffffff;
            border: 1px dashed #cccccc;
            border-radius: 8px;
            padding: 10px;
        }
        .stDataFrame { border-radius: 8px; overflow: hidden; }
        .stAlert { border-radius: 8px; }
        .streamlit-expanderHeader { font-size: 14px; color: #444; }
        .footer {
            margin-top: 40px;
            border-top: 1px solid #e0e0e0;
            padding-top: 10px;
            font-size: 12px;
            color: #999;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("rf_ids_model.pkl")
encoder = joblib.load("label_encoder.pkl")

EXPECTED_COLUMNS = list(model.feature_names_in_)

# =========================
# HEADER
# =========================
st.title("AI-Powered Intrusion Detection System")
st.write(
    "Upload a CSV file containing network traffic data. "
    "The system will analyze each flow and classify it as normal or malicious."
)

st.markdown("---")

# =========================
# COLUMN INFO
# =========================
with st.expander("View required column names"):
    st.write(EXPECTED_COLUMNS)

# =========================
# FILE UPLOAD
# =========================
st.subheader("Upload Network Traffic File")
uploaded_file = st.file_uploader(
    "Accepted format: CSV — Maximum size: 200MB",
    type=["csv"],
    label_visibility="visible"
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =========================
    # CLEAN COLUMN NAMES
    # =========================
    original_columns = df.columns.tolist()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_per_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    cleaned_columns = df.columns.tolist()

    changed = []
    for orig, clean in zip(original_columns, cleaned_columns):
        if orig != clean:
            changed.append(f"'{orig}' -> '{clean}'")

    if changed:
        with st.expander("Column names were normalized"):
            for c in changed[:10]:
                st.text(c)
            if len(changed) > 10:
                st.text(f"... and {len(changed) - 10} more")

    # =========================
    # CHECK COLUMNS
    # =========================
    uploaded_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    missing_columns = expected_columns - uploaded_columns
    extra_columns = uploaded_columns - expected_columns

    if missing_columns:
        st.error(f"Missing columns: {sorted(missing_columns)}")
        st.warning("Please fix the CSV file and re-upload.")
        st.stop()

    if extra_columns:
        st.info(f"Extra columns ignored: {sorted(extra_columns)}")

    df = df[EXPECTED_COLUMNS]

    # Fix infinity and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    st.success("All required columns found. Running analysis...")

    st.markdown("---")

    # =========================
    # PREDICT
    # =========================
    predictions = model.predict(df)
    proba = model.predict_proba(df)
    max_proba = np.max(proba, axis=1)

    labels = encoder.inverse_transform(predictions)

    df["Confidence"] = [f"{p*100:.1f}%" for p in max_proba]

    LOW_CONFIDENCE_THRESHOLD = 0.70
    low_confidence_count = (max_proba < LOW_CONFIDENCE_THRESHOLD).sum()

    if low_confidence_count > 0:
        st.warning(
            f"{low_confidence_count} prediction(s) have low confidence (below 70%). "
            "Manual review is recommended for these flows."
        )

    df["Prediction"] = labels

    # =========================
    # METRICS
    # =========================
    st.subheader("Analysis Summary")

    normal_count = (df["Prediction"] == "Normal Traffic").sum()
    attack_count = len(df) - normal_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows Analyzed", f"{len(df):,}")
    col2.metric("Normal Traffic", f"{normal_count:,}")
    col3.metric("Attacks Detected", f"{attack_count:,}")

    st.markdown("---")

    # =========================
    # ATTACK DISTRIBUTION CHART
    # =========================
    st.subheader("Traffic Classification Distribution")

    prediction_counts = df["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3))

    colors = []
    for label in prediction_counts.index:
        if label == "Normal Traffic":
            colors.append("#4CAF50")
        else:
            colors.append("#E53935")

    prediction_counts.plot(
        kind="bar",
        ax=ax,
        color=colors,
        edgecolor="white",
        linewidth=0.5
    )

    ax.set_ylabel("Number of Flows", fontsize=10)
    ax.set_xlabel("")
    ax.set_title("Prediction Distribution", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, fontsize=9)
    plt.tight_layout()

    col_chart, col_space = st.columns([2, 1])
    with col_chart:
        st.pyplot(fig)

    st.markdown("---")

    # =========================
    # SECURITY SUMMARY
    # =========================
    st.subheader("Security Assessment")

    if attack_count == 0:
        st.success(
            "No attacks detected. The analyzed network traffic appears normal."
        )
    else:
        st.error(
            f"Intrusion detected — {attack_count:,} suspicious flow(s) identified. "
            "Immediate investigation is recommended."
        )

        st.subheader("Attack Type Breakdown")

        attack_df = df[df["Prediction"] != "Normal Traffic"]["Prediction"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(5, 3))

        attack_df.plot(
            kind="barh",
            ax=ax2,
            color="#E53935",
            edgecolor="white"
        )

        ax2.set_xlabel("Number of Flows", fontsize=10)
        ax2.set_ylabel("")
        ax2.set_title("Detected Attack Types", fontsize=11)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()

        col_chart2, col_space2 = st.columns([2, 1])
        with col_chart2:
            st.pyplot(fig2)

    st.markdown("---")

    # =========================
    # DETAILED RESULTS TABLE
    # =========================
    st.subheader("Detailed Flow Analysis")
    st.dataframe(df, use_container_width=True)

    # =========================
    # FOOTER
    # =========================
    st.markdown(
        '<div class="footer">AI-IDS — Intrusion Detection System | BAAZA Idriss | 2025/2026</div>',
        unsafe_allow_html=True
    )