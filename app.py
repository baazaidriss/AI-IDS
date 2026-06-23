import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import time
from datetime import datetime

# =========================
# COLOR PALETTE
# =========================
NORMAL_COLOR = "#4a7c59"
ATTACK_COLOR = "#c0392b"
NEUTRAL_COLOR = "#2c3e50"
MEDIUM_COLOR = "#ca6f1e"

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
            color: #2c3e50;
            font-size: 28px;
            font-weight: 700;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2, h3 {
            color: #2c3e50;
            font-size: 18px;
            font-weight: 600;
            margin-top: 20px;
        }
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
        .threat-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .threat-safe {
            background-color: #eaf4ee;
            color: #2d6a4f;
            border: 2px solid #4a7c59;
        }
        .threat-low {
            background-color: #fef9e7;
            color: #7d6608;
            border: 2px solid #d4ac0d;
        }
        .threat-warning {
            background-color: #fdf2e9;
            color: #784212;
            border: 2px solid #ca6f1e;
        }
        .threat-critical {
            background-color: #fdedec;
            color: #922b21;
            border: 2px solid #c0392b;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# CHART SIZE
# =========================
CHART_SIZE = (5, 3)

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
st.markdown("Upload a network traffic file to analyze it for cyber threats using the trained Random Forest model.")
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
    "Accepted formats: CSV, Excel (.xlsx), JSON — Maximum size: 200MB",
    type=["csv", "xlsx", "json"],
    label_visibility="visible"
)

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.info("File format detected: CSV")
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        st.info("File format detected: Excel")
    elif file_name.endswith(".json"):
        df = pd.read_json(uploaded_file)
        st.info("File format detected: JSON")
    else:
        st.error("Unsupported file format. Please upload CSV, Excel or JSON.")
        st.stop()

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
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # =========================
    # PROGRESS BAR
    # =========================
    st.markdown("---")
    st.subheader("Running Analysis...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        (20, "Validating input data..."),
        (40, "Preprocessing features..."),
        (60, "Running Random Forest model..."),
        (80, "Computing confidence scores..."),
        (100, "Analysis complete!")
    ]

    for percent, message in steps:
        time.sleep(0.4)
        progress_bar.progress(percent)
        status_text.text(message)

    status_text.empty()
    progress_bar.empty()

    st.success("All required columns found. Analysis complete!")
    st.markdown("---")

    # =========================
    # PREDICT
    # =========================
    predictions = model.predict(df)
    proba = model.predict_proba(df)
    max_proba = np.max(proba, axis=1)
    labels = encoder.inverse_transform(predictions)

    df["Confidence"] = [f"{p*100:.1f}%" for p in max_proba]

    # =========================
    # RISK SCORE
    # =========================
    def compute_risk_score(prediction, confidence):
        if prediction == "Normal Traffic":
            base = 0
        elif prediction == "Brute Force":
            base = 60
        elif prediction == "DDoS":
            base = 85
        elif prediction == "DoS":
            base = 80
        elif prediction == "Port Scanning":
            base = 50
        else:
            base = 70
        return min(int(base * confidence), 100)

    risk_scores = [
        compute_risk_score(label, conf)
        for label, conf in zip(labels, max_proba)
    ]

    df["Risk Score"] = risk_scores

    def risk_category(score):
        if score <= 30:
            return "Low"
        elif score <= 70:
            return "Medium"
        else:
            return "High"

    df["Risk Level"] = df["Risk Score"].apply(risk_category)

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
    attack_percentage = (attack_count / len(df)) * 100
    high_risk_count = (df["Risk Level"] == "High").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flows Analyzed", f"{len(df):,}")
    col2.metric("Normal Traffic", f"{normal_count:,}")
    col3.metric("Attacks Detected", f"{attack_count:,}")
    col4.metric("High Risk Flows", f"{high_risk_count:,}")

    st.markdown("---")

    # =========================
    # THREAT LEVEL
    # =========================
    st.subheader("Threat Level")

    if attack_percentage == 0:
        threat_level = "SAFE"
        threat_class = "threat-safe"
        threat_msg = "No threats detected. Network traffic appears normal."
    elif attack_percentage < 30:
        threat_level = "LOW"
        threat_class = "threat-low"
        threat_msg = f"{attack_percentage:.1f}% of traffic flagged as suspicious. Monitor closely."
    elif attack_percentage < 60:
        threat_level = "WARNING"
        threat_class = "threat-warning"
        threat_msg = f"{attack_percentage:.1f}% of traffic flagged as attacks. Investigation recommended."
    else:
        threat_level = "CRITICAL"
        threat_class = "threat-critical"
        threat_msg = f"{attack_percentage:.1f}% of traffic identified as attacks. Immediate action required!"

    st.markdown(
        f'<div class="threat-box {threat_class}">'
        f'THREAT LEVEL: {threat_level}<br>'
        f'<span style="font-size:14px; font-weight:normal;">{threat_msg}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================
    # CHARTS — PIE + BAR (same height, centered)
    # =========================
    st.subheader("Traffic Classification Distribution")

    prediction_counts = df["Prediction"].value_counts()

    def get_color(label):
        return NORMAL_COLOR if label == "Normal Traffic" else ATTACK_COLOR

    col_left, col_pie, col_bar, col_right = st.columns([0.5, 2, 2, 0.5])

    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=CHART_SIZE)
        pie_labels = prediction_counts.index.tolist()
        pie_sizes = prediction_counts.values.tolist()
        pie_colors = [get_color(l) for l in pie_labels]

        ax_pie.pie(
            pie_sizes,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9, "color": NEUTRAL_COLOR}
        )
        ax_pie.set_title("Traffic Composition", fontsize=11, color=NEUTRAL_COLOR)
        plt.tight_layout()
        st.pyplot(fig_pie)

    with col_bar:
        fig_bar, ax_bar = plt.subplots(figsize=CHART_SIZE)
        bar_colors = [get_color(l) for l in prediction_counts.index]

        prediction_counts.plot(
            kind="bar",
            ax=ax_bar,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.5
        )

        ax_bar.set_ylabel("Number of Flows", fontsize=10, color=NEUTRAL_COLOR)
        ax_bar.set_xlabel("")
        ax_bar.set_title("Prediction Distribution", fontsize=11, color=NEUTRAL_COLOR)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.tick_params(colors=NEUTRAL_COLOR)
        plt.xticks(rotation=30, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_bar)

    st.markdown("---")

    # =========================
    # SECURITY ASSESSMENT
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

        col_left2, col_mid2, col_right2 = st.columns([1, 2, 1])
        with col_mid2:
            fig2, ax2 = plt.subplots(figsize=CHART_SIZE)
            attack_df.plot(
                kind="barh",
                ax=ax2,
                color=ATTACK_COLOR,
                edgecolor="white"
            )
            ax2.set_xlabel("Number of Flows", fontsize=10, color=NEUTRAL_COLOR)
            ax2.set_ylabel("")
            ax2.set_title("Detected Attack Types", fontsize=11, color=NEUTRAL_COLOR)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.tick_params(colors=NEUTRAL_COLOR)
            plt.tight_layout()
            st.pyplot(fig2)

    st.markdown("---")

    # =========================
    # RISK SCORE DISTRIBUTION
    # =========================
    st.subheader("Risk Score Distribution")

    low_risk = (df["Risk Level"] == "Low").sum()
    med_risk = (df["Risk Level"] == "Medium").sum()
    high_risk = (df["Risk Level"] == "High").sum()

    risk_col1, risk_col2, risk_col3 = st.columns(3)
    risk_col1.metric("Low Risk (0-30)", f"{low_risk:,}")
    risk_col2.metric("Medium Risk (31-70)", f"{med_risk:,}")
    risk_col3.metric("High Risk (71-100)", f"{high_risk:,}")

    col_left3, col_mid3, col_right3 = st.columns([1, 2, 1])
    with col_mid3:
        fig_risk, ax_risk = plt.subplots(figsize=CHART_SIZE)

        risk_labels = ["Low (0-30)", "Medium (31-70)", "High (71-100)"]
        risk_values = [low_risk, med_risk, high_risk]
        risk_colors = [NORMAL_COLOR, MEDIUM_COLOR, ATTACK_COLOR]

        bars = ax_risk.bar(
            risk_labels,
            risk_values,
            color=risk_colors,
            edgecolor="white",
            linewidth=0.5
        )

        for bar, val in zip(bars, risk_values):
            ax_risk.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:,}",
                ha="center",
                fontsize=9,
                color=NEUTRAL_COLOR
            )

        ax_risk.set_ylabel("Number of Flows", fontsize=10, color=NEUTRAL_COLOR)
        ax_risk.set_title("Risk Level Distribution", fontsize=11, color=NEUTRAL_COLOR)
        ax_risk.spines["top"].set_visible(False)
        ax_risk.spines["right"].set_visible(False)
        ax_risk.tick_params(colors=NEUTRAL_COLOR)
        plt.tight_layout()
        st.pyplot(fig_risk)

    st.markdown("---")

    # =========================
    # FILTER + DETAILED TABLE
    # =========================
    st.subheader("Detailed Flow Analysis")

    all_types = ["All"] + sorted(df["Prediction"].unique().tolist())
    selected_type = st.selectbox("Filter by attack type:", all_types)

    all_risk_levels = ["All", "Low", "Medium", "High"]
    selected_risk = st.selectbox("Filter by risk level:", all_risk_levels)

    filtered_df = df.copy()

    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["Prediction"] == selected_type]

    if selected_risk != "All":
        filtered_df = filtered_df[filtered_df["Risk Level"] == selected_risk]

    st.write(f"Showing {len(filtered_df):,} of {len(df):,} connections")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.subheader("Download Incident Report")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_text = f"""
============================================================
       AI-IDS SECURITY INCIDENT REPORT
============================================================
Generated     : {now}
File Analyzed : {uploaded_file.name}
------------------------------------------------------------

EXECUTIVE SUMMARY
-----------------
Total Flows Analyzed : {len(df):,}
Normal Traffic       : {normal_count:,}
Attacks Detected     : {attack_count:,}
Attack Percentage    : {attack_percentage:.1f}%
Threat Level         : {threat_level}
High Risk Flows      : {high_risk_count:,}

RISK BREAKDOWN
--------------
Low Risk  (0-30)    : {low_risk:,} flows
Medium Risk (31-70) : {med_risk:,} flows
High Risk (71-100)  : {high_risk:,} flows

LOW CONFIDENCE FLAGS
--------------------
Predictions below 70% confidence : {low_confidence_count}
Manual review recommended        : {"Yes" if low_confidence_count > 0 else "No"}

ATTACK BREAKDOWN
----------------
"""

    if attack_count > 0:
        attack_counts = df[df["Prediction"] != "Normal Traffic"]["Prediction"].value_counts()
        for attack_type, count in attack_counts.items():
            report_text += f"{attack_type:<25} : {count:,} flows\n"
    else:
        report_text += "No attacks detected.\n"

    report_text += """
------------------------------------------------------------
RECOMMENDATIONS
---------------
"""
    if threat_level == "SAFE":
        report_text += "- Network traffic appears normal. Continue routine monitoring.\n"
    elif threat_level == "LOW":
        report_text += "- Low level of suspicious traffic detected. Monitor closely.\n"
        report_text += "- Review flagged connections in the detailed table.\n"
    elif threat_level == "WARNING":
        report_text += "- Significant attack traffic detected. Investigate immediately.\n"
        report_text += "- Consider blocking suspicious source IPs.\n"
        report_text += "- Review all low-confidence predictions manually.\n"
    else:
        report_text += "- CRITICAL: Majority of traffic is malicious. Immediate action required.\n"
        report_text += "- Isolate affected network segments immediately.\n"
        report_text += "- Contact security team and escalate to incident response.\n"
        report_text += "- Preserve logs for forensic analysis.\n"

    report_text += f"""
------------------------------------------------------------
SYSTEM INFORMATION
------------------
Model          : Random Forest (500 trees)
Training Data  : CICIDS2017 (69,150 samples)
Test Accuracy  : 99.83%
NSL-KDD Valid. : 98.62%
Dashboard      : https://ai-ids-baaza.streamlit.app
GitHub         : https://github.com/baazaidriss/AI-IDS
============================================================
        AI-IDS — BAAZA Idriss — 2025/2026
============================================================
"""

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.download_button(
            label="Download Text Report",
            data=report_text,
            file_name=f"AI-IDS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col_dl2:
        csv_report = df.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_report,
            file_name=f"AI-IDS_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # =========================
    # FOOTER
    # =========================
    st.markdown(
        '<div class="footer">AI-IDS — Intrusion Detection System | BAAZA Idriss | 2025/2026</div>',
        unsafe_allow_html=True
    )