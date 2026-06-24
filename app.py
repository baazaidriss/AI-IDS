import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use("Agg")
import time
from datetime import datetime
import io

# =========================
# COLOR PALETTE
# =========================
NORMAL_COLOR = "#52796f"
ATTACK_COLOR = "#ae2012"
NEUTRAL_COLOR = "#ffffff"
MEDIUM_COLOR = "#ca6f1e"
CHART_BG = "#0e1117"
GRID_COLOR = "#2d2d44"

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
        .main { background-color: #0e1117; }
        h1 {
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            border-bottom: 2px solid #2d2d44;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2, h3 {
            color: #ffffff;
            font-size: 18px;
            font-weight: 600;
            margin-top: 20px;
        }
        p, div, span, label {
            color: #cccccc;
        }
        div[data-testid="metric-container"] {
            background-color: #1a1a2e;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            padding: 15px;
        }
        div[data-testid="metric-container"] label {
            color: #aaaaaa !important;
        }
        div[data-testid="metric-container"] div {
            color: #ffffff !important;
        }
        section[data-testid="stFileUploader"] {
            background-color: #1a1a2e;
            border: 1px dashed #2d2d44;
            border-radius: 8px;
            padding: 10px;
        }
        .stDataFrame { border-radius: 8px; overflow: hidden; }
        .stAlert { border-radius: 8px; }
        .streamlit-expanderHeader { font-size: 14px; color: #cccccc; }
        .footer {
            margin-top: 40px;
            border-top: 1px solid #2d2d44;
            padding-top: 10px;
            font-size: 12px;
            color: #666;
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
            background-color: #1a2e22;
            color: #52796f;
            border: 2px solid #52796f;
        }
        .threat-low {
            background-color: #2e2a0e;
            color: #d4ac0d;
            border: 2px solid #d4ac0d;
        }
        .threat-warning {
            background-color: #2e1e0e;
            color: #ca6f1e;
            border: 2px solid #ca6f1e;
        }
        .threat-critical {
            background-color: #2e0e0e;
            color: #ae2012;
            border: 2px solid #ae2012;
        }
        [data-testid="stMarkdownContainer"] p {
            color: #cccccc;
        }
        .stSelectbox label {
            color: #cccccc !important;
        }
        hr {
            border-color: #2d2d44;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# CHART STYLE FUNCTION
# =========================
def style_ax(ax, title):
    ax.set_facecolor(CHART_BG)
    ax.set_title(title, fontsize=11, color=NEUTRAL_COLOR, pad=8)
    ax.tick_params(colors=NEUTRAL_COLOR, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.yaxis.label.set_color(NEUTRAL_COLOR)
    ax.xaxis.label.set_color(NEUTRAL_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)

def get_color(label):
    return NORMAL_COLOR if label == "Normal Traffic" else ATTACK_COLOR

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
        st.error("Unsupported file format.")
        st.stop()

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

    uploaded_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)
    missing_columns = expected_columns - uploaded_columns
    extra_columns = uploaded_columns - expected_columns

    if missing_columns:
        st.error(f"Missing columns: {sorted(missing_columns)}")
        st.warning("Please fix the file and re-upload.")
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

    risk_scores = [compute_risk_score(l, c) for l, c in zip(labels, max_proba)]
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
    low_risk = (df["Risk Level"] == "Low").sum()
    med_risk = (df["Risk Level"] == "Medium").sum()
    high_risk = (df["Risk Level"] == "High").sum()

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
    # CHARTS — PIE + BAR
    # =========================
    st.subheader("Traffic Classification Distribution")

    prediction_counts = df["Prediction"].value_counts()

    col_left, col_pie, col_bar, col_right = st.columns([0.5, 2, 2, 0.5])

    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
        fig_pie.patch.set_facecolor(CHART_BG)
        ax_pie.set_facecolor(CHART_BG)
        pie_colors = [get_color(l) for l in prediction_counts.index]
        short_labels = [
            "Normal" if l == "Normal Traffic" else l
            for l in prediction_counts.index
        ]

        wedges, texts, autotexts = ax_pie.pie(
            prediction_counts.values.tolist(),
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            labels=short_labels,
            textprops={"fontsize": 8, "color": NEUTRAL_COLOR}
        )
        for at in autotexts:
            at.set_color(NEUTRAL_COLOR)
            at.set_fontsize(8)

        ax_pie.set_title("Traffic Composition", fontsize=11,
                         color=NEUTRAL_COLOR, pad=8)
        plt.tight_layout()
        st.pyplot(fig_pie)

    with col_bar:
        fig_bar, ax_bar = plt.subplots(figsize=(5, 4.2))
        fig_bar.patch.set_facecolor(CHART_BG)
        style_ax(ax_bar, "Prediction Distribution")
        bar_colors = [get_color(l) for l in prediction_counts.index]

        prediction_counts.plot(
            kind="bar",
            ax=ax_bar,
            color=bar_colors,
            edgecolor="none"
        )
        ax_bar.set_ylabel("Number of Flows", fontsize=9, color=NEUTRAL_COLOR)
        ax_bar.set_xlabel("")
        plt.xticks(rotation=30, fontsize=8, color=NEUTRAL_COLOR)
        plt.tight_layout()
        st.pyplot(fig_bar)

    st.markdown("---")

    # =========================
    # SECURITY ASSESSMENT
    # =========================
    st.subheader("Security Assessment")

    if attack_count == 0:
        st.success("No attacks detected. The analyzed network traffic appears normal.")
    else:
        st.error(
            f"Intrusion detected — {attack_count:,} suspicious flow(s) identified. "
            "Immediate investigation is recommended."
        )

        st.subheader("Attack Type Breakdown")
        attack_df = df[df["Prediction"] != "Normal Traffic"]["Prediction"].value_counts()

        col_left2, col_mid2, col_right2 = st.columns([1, 2, 1])
        with col_mid2:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fig2.patch.set_facecolor(CHART_BG)
            style_ax(ax2, "Detected Attack Types")
            attack_df.plot(kind="barh", ax=ax2,
                           color=ATTACK_COLOR, edgecolor="none")
            ax2.set_xlabel("Number of Flows", fontsize=9, color=NEUTRAL_COLOR)
            ax2.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig2)

    st.markdown("---")

    # =========================
    # RISK SCORE DISTRIBUTION
    # =========================
    st.subheader("Risk Score Distribution")

    risk_col1, risk_col2, risk_col3 = st.columns(3)
    risk_col1.metric("Low Risk (0-30)", f"{low_risk:,}")
    risk_col2.metric("Medium Risk (31-70)", f"{med_risk:,}")
    risk_col3.metric("High Risk (71-100)", f"{high_risk:,}")

    col_left3, col_mid3, col_right3 = st.columns([1, 2, 1])
    with col_mid3:
        fig_risk, ax_risk = plt.subplots(figsize=(5, 3.5))
        fig_risk.patch.set_facecolor(CHART_BG)
        style_ax(ax_risk, "Risk Level Distribution")
        risk_labels = ["Low (0-30)", "Medium (31-70)", "High (71-100)"]
        risk_values = [low_risk, med_risk, high_risk]
        risk_colors = [NORMAL_COLOR, MEDIUM_COLOR, ATTACK_COLOR]

        bars = ax_risk.bar(
            risk_labels, risk_values,
            color=risk_colors,
            edgecolor="none"
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

        ax_risk.set_ylabel("Number of Flows", fontsize=9, color=NEUTRAL_COLOR)
        plt.tight_layout()
        st.pyplot(fig_risk)

    st.markdown("---")

    # =========================
    # FILTER + TABLE
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
    # PDF REPORT
    # =========================
    st.subheader("Download Incident Report")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_pdf_report():
        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:

            # ---- PAGE 1: SUMMARY ----
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")

            fig.text(0.5, 0.96, "AI-IDS SECURITY INCIDENT REPORT",
                     ha="center", fontsize=16,
                     fontweight="bold", color="#2c3e50")

            fig.text(0.5, 0.93,
                     f"Generated: {now}   |   File: {uploaded_file.name}",
                     ha="center", fontsize=9, color="gray")

            ax_div = fig.add_axes([0.05, 0.915, 0.9, 0.003])
            ax_div.axhline(0, color=ATTACK_COLOR, linewidth=2)
            ax_div.axis("off")

            threat_color_map = {
                "SAFE": "#52796f",
                "LOW": "#d4ac0d",
                "WARNING": MEDIUM_COLOR,
                "CRITICAL": ATTACK_COLOR
            }
            tcolor = threat_color_map.get(threat_level, "#2c3e50")

            fig.text(0.5, 0.89, f"THREAT LEVEL: {threat_level}",
                     ha="center", fontsize=14,
                     fontweight="bold", color=tcolor)

            fig.text(0.5, 0.865, threat_msg,
                     ha="center", fontsize=9, color="#2c3e50")

            ax_div2 = fig.add_axes([0.05, 0.85, 0.9, 0.003])
            ax_div2.axhline(0, color="#dddddd", linewidth=1)
            ax_div2.axis("off")

            y = 0.82
            fig.text(0.05, y, "EXECUTIVE SUMMARY",
                     fontsize=11, fontweight="bold", color="#2c3e50")
            y -= 0.03

            summary_data = [
                ["Total Flows Analyzed", f"{len(df):,}"],
                ["Normal Traffic", f"{normal_count:,}"],
                ["Attacks Detected", f"{attack_count:,}"],
                ["Attack Percentage", f"{attack_percentage:.1f}%"],
                ["High Risk Flows", f"{high_risk_count:,}"],
                ["Low Confidence Flags", f"{low_confidence_count}"],
            ]
            for label, value in summary_data:
                fig.text(0.08, y, label, fontsize=9, color="#444444")
                fig.text(0.55, y, value, fontsize=9,
                         fontweight="bold", color="#2c3e50")
                y -= 0.03

            y -= 0.01
            ax_div3 = fig.add_axes([0.05, y + 0.015, 0.9, 0.003])
            ax_div3.axhline(0, color="#dddddd", linewidth=1)
            ax_div3.axis("off")
            y -= 0.01

            fig.text(0.05, y, "RISK BREAKDOWN",
                     fontsize=11, fontweight="bold", color="#2c3e50")
            y -= 0.03

            for label, value, color in [
                ["Low Risk (0-30)", f"{low_risk:,} flows", "#52796f"],
                ["Medium Risk (31-70)", f"{med_risk:,} flows", MEDIUM_COLOR],
                ["High Risk (71-100)", f"{high_risk:,} flows", ATTACK_COLOR],
            ]:
                fig.text(0.08, y, label, fontsize=9,
                         color=color, fontweight="bold")
                fig.text(0.55, y, value, fontsize=9, color="#444444")
                y -= 0.03

            y -= 0.01
            ax_div4 = fig.add_axes([0.05, y + 0.015, 0.9, 0.003])
            ax_div4.axhline(0, color="#dddddd", linewidth=1)
            ax_div4.axis("off")
            y -= 0.01

            fig.text(0.05, y, "ATTACK BREAKDOWN",
                     fontsize=11, fontweight="bold", color="#2c3e50")
            y -= 0.03

            if attack_count > 0:
                for attack_type, count in df[
                    df["Prediction"] != "Normal Traffic"
                ]["Prediction"].value_counts().items():
                    fig.text(0.08, y, attack_type,
                             fontsize=9, color=ATTACK_COLOR)
                    fig.text(0.55, y, f"{count:,} flows",
                             fontsize=9, color="#444444")
                    y -= 0.03
            else:
                fig.text(0.08, y, "No attacks detected.",
                         fontsize=9, color="#52796f")
                y -= 0.03

            y -= 0.01
            ax_div5 = fig.add_axes([0.05, y + 0.015, 0.9, 0.003])
            ax_div5.axhline(0, color="#dddddd", linewidth=1)
            ax_div5.axis("off")
            y -= 0.01

            fig.text(0.05, y, "RECOMMENDATIONS",
                     fontsize=11, fontweight="bold", color="#2c3e50")
            y -= 0.03

            if threat_level == "SAFE":
                recs = ["Network traffic appears normal. Continue routine monitoring."]
            elif threat_level == "LOW":
                recs = ["Low level of suspicious traffic detected. Monitor closely.",
                        "Review flagged connections in the detailed table."]
            elif threat_level == "WARNING":
                recs = ["Significant attack traffic detected. Investigate immediately.",
                        "Consider blocking suspicious source IPs.",
                        "Review all low-confidence predictions manually."]
            else:
                recs = ["CRITICAL: Majority of traffic is malicious. Immediate action required.",
                        "Isolate affected network segments immediately.",
                        "Contact security team and escalate to incident response.",
                        "Preserve logs for forensic analysis."]

            for rec in recs:
                fig.text(0.08, y, f"- {rec}",
                         fontsize=9, color="#444444")
                y -= 0.03

            y -= 0.01
            ax_div6 = fig.add_axes([0.05, y + 0.015, 0.9, 0.003])
            ax_div6.axhline(0, color="#dddddd", linewidth=1)
            ax_div6.axis("off")
            y -= 0.01

            fig.text(0.05, y, "SYSTEM INFORMATION",
                     fontsize=11, fontweight="bold", color="#2c3e50")
            y -= 0.03

            sys_info = [
                ["Model", "Random Forest (500 trees)"],
                ["Training Data", "CICIDS2017 (69,150 samples)"],
                ["Test Accuracy", "99.83%"],
                ["NSL-KDD Validation", "98.62%"],
                ["Dashboard", "https://ai-ids-baaza.streamlit.app"],
                ["GitHub", "https://github.com/baazaidriss/AI-IDS"],
            ]
            for label, value in sys_info:
                fig.text(0.08, y, label, fontsize=9, color="#444444")
                fig.text(0.55, y, value, fontsize=9, color="#2c3e50")
                y -= 0.03

            fig.text(0.5, 0.02,
                     "AI-IDS — BAAZA Idriss — 2025/2026",
                     ha="center", fontsize=8, color="gray")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ---- PAGE 2: CHARTS ----
            fig2 = plt.figure(figsize=(8.5, 11))
            fig2.patch.set_facecolor("white")

            fig2.text(0.5, 0.97, "AI-IDS — Visual Analysis",
                      ha="center", fontsize=14,
                      fontweight="bold", color="#2c3e50")

            ax_div_p2 = fig2.add_axes([0.05, 0.955, 0.9, 0.003])
            ax_div_p2.axhline(0, color=ATTACK_COLOR, linewidth=2)
            ax_div_p2.axis("off")

            ax_pie2 = fig2.add_subplot(2, 2, 1)
            ax_pie2.set_facecolor("white")
            pie_colors2 = [get_color(l) for l in prediction_counts.index]
            short_labels2 = [
                "Normal" if l == "Normal Traffic" else l
                for l in prediction_counts.index
            ]
            wedges2, texts2, autotexts2 = ax_pie2.pie(
                prediction_counts.values.tolist(),
                colors=pie_colors2,
                autopct="%1.1f%%",
                startangle=90,
                labels=short_labels2,
                textprops={"fontsize": 7, "color": "#2c3e50"}
            )
            for at in autotexts2:
                at.set_color("#2c3e50")
            ax_pie2.set_title("Traffic Composition",
                              fontsize=10, color="#2c3e50")

            ax_bar2 = fig2.add_subplot(2, 2, 2)
            ax_bar2.set_facecolor("white")
            bar_colors2 = [get_color(l) for l in prediction_counts.index]
            prediction_counts.plot(kind="bar", ax=ax_bar2,
                                   color=bar_colors2, edgecolor="none")
            ax_bar2.set_title("Prediction Distribution",
                              fontsize=10, color="#2c3e50")
            ax_bar2.set_ylabel("Number of Flows",
                               fontsize=8, color="#2c3e50")
            ax_bar2.spines["top"].set_visible(False)
            ax_bar2.spines["right"].set_visible(False)
            ax_bar2.spines["left"].set_color("#dddddd")
            ax_bar2.spines["bottom"].set_color("#dddddd")
            ax_bar2.tick_params(colors="#2c3e50", labelsize=7)
            plt.setp(ax_bar2.xaxis.get_majorticklabels(),
                     rotation=30, fontsize=7)

            if attack_count > 0:
                ax_attack2 = fig2.add_subplot(2, 2, 3)
                ax_attack2.set_facecolor("white")
                attack_df2 = df[
                    df["Prediction"] != "Normal Traffic"
                ]["Prediction"].value_counts()
                attack_df2.plot(kind="barh", ax=ax_attack2,
                                color=ATTACK_COLOR, edgecolor="none")
                ax_attack2.set_title("Attack Type Breakdown",
                                     fontsize=10, color="#2c3e50")
                ax_attack2.set_xlabel("Number of Flows",
                                      fontsize=8, color="#2c3e50")
                ax_attack2.spines["top"].set_visible(False)
                ax_attack2.spines["right"].set_visible(False)
                ax_attack2.spines["left"].set_color("#dddddd")
                ax_attack2.spines["bottom"].set_color("#dddddd")
                ax_attack2.tick_params(colors="#2c3e50", labelsize=7)

            ax_risk2 = fig2.add_subplot(2, 2, 4)
            ax_risk2.set_facecolor("white")
            ax_risk2.bar(["Low", "Medium", "High"],
                         [low_risk, med_risk, high_risk],
                         color=[NORMAL_COLOR, MEDIUM_COLOR, ATTACK_COLOR],
                         edgecolor="none")
            ax_risk2.set_title("Risk Level Distribution",
                               fontsize=10, color="#2c3e50")
            ax_risk2.set_ylabel("Number of Flows",
                                fontsize=8, color="#2c3e50")
            ax_risk2.spines["top"].set_visible(False)
            ax_risk2.spines["right"].set_visible(False)
            ax_risk2.spines["left"].set_color("#dddddd")
            ax_risk2.spines["bottom"].set_color("#dddddd")
            ax_risk2.tick_params(colors="#2c3e50", labelsize=7)

            fig2.text(0.5, 0.02,
                      "AI-IDS — BAAZA Idriss — 2025/2026",
                      ha="center", fontsize=8, color="gray")

            plt.tight_layout(rect=[0, 0.04, 1, 0.94])
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        pdf_buffer.seek(0)
        return pdf_buffer

    pdf_data = generate_pdf_report()

    st.download_button(
        label="Download PDF Report",
        data=pdf_data,
        file_name=f"AI-IDS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

    # =========================
    # FOOTER
    # =========================
    st.markdown(
        '<div class="footer">AI-IDS — Intrusion Detection System | BAAZA Idriss | 2025/2026</div>',
        unsafe_allow_html=True
    )