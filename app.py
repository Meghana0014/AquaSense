"""
AquaSense: Water Purifier Filter Health Monitoring & Predictive Maintenance System
Level 1 Software Demonstrator — Interactive Streamlit Dashboard
"""

from __future__ import annotations
import io
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_generator import SyntheticDataGenerator
from health_engine import (
    AlertRecord,
    BenchmarkResult,
    BenchmarkSuite,
    DataPreprocessor,
    FilterHealthScoringEngine,
    ExplainableAlertEngine,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaSense | Filter Health & Predictive Maintenance",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Modern CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f4c81;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #556987;
        margin-bottom: 1.2rem;
    }
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafd 0%, #eef4fc 100%);
        border: 1px solid #dbe5f1;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(15, 76, 129, 0.04);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    /* Status Badges */
    .status-badge-good {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .status-badge-fair {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        display: inline-block;
        border: 1px solid #fcd34d;
    }
    .status-badge-replace {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    /* Alert Callout */
    .alert-banner-critical {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .alert-banner-warning {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .alert-banner-info {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
    }
    /* Safety Disclaimer */
    .safety-footer {
        background: #f1f5f9;
        border-top: 2px solid #cbd5e1;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 40px;
        font-size: 0.88rem;
        color: #475569;
        text-align: center;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Loader & Cache Helper
# -----------------------------------------------------------------------------
@st.cache_data
def load_standard_scenario(scenario_key: str) -> pd.DataFrame:
    """Loads pre-generated scenario dataset or generates on demand."""
    filename_map = {
        "Normal Operation": "normal_scenario.csv",
        "Deteriorating Filter": "deteriorating_scenario.csv",
        "Abnormal / Anomaly": "abnormal_scenario.csv",
    }
    csv_file = Path("data") / filename_map.get(scenario_key, "normal_scenario.csv")

    if csv_file.exists():
        return pd.read_csv(csv_file)

    # Fallback generation if data dir missing
    generator = SyntheticDataGenerator()
    if scenario_key == "Normal Operation":
        return generator.generate_normal_scenario()
    elif scenario_key == "Deteriorating Filter":
        return generator.generate_deteriorating_scenario()
    else:
        return generator.generate_abnormal_scenario()


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Scenario Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💧 AquaSense Control Center")
    st.caption("Intelligent Filter Health & Telemetry Demonstrator")

    scenario_option = st.selectbox(
        "Select Telemetry Scenario",
        ["Normal Operation", "Deteriorating Filter", "Abnormal / Anomaly", "Upload Custom CSV"],
        index=1,
        help="Switch between simulated reference scenarios or upload telemetry CSV.",
    )

    raw_df: pd.DataFrame
    if scenario_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload Purifier Telemetry CSV",
            type=["csv"],
            help="CSV must contain: timestamp, tds_ppm, daily_usage_liters",
        )
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(raw_df)} records.")
        else:
            st.info("Awaiting CSV upload. Falling back to Deteriorating scenario.")
            raw_df = load_standard_scenario("Deteriorating Filter")
    else:
        raw_df = load_standard_scenario(scenario_option)

    st.markdown("---")
    st.markdown("#### ⚙️ Filter Hardware Specifications")
    with st.expander("Adjust Rated Thresholds", expanded=False):
        rated_volume = st.number_input("Rated Media Capacity (Liters)", min_value=500.0, max_value=10000.0, value=3000.0, step=250.0)
        rated_lifespan = st.number_input("Rated Lifespan (Days)", min_value=30, max_value=730, value=365, step=30)
        tds_warn = st.number_input("TDS Warning Limit (ppm)", min_value=50.0, max_value=300.0, value=150.0, step=10.0)
        tds_crit = st.number_input("TDS Critical Limit (ppm)", min_value=100.0, max_value=500.0, value=200.0, step=10.0)

    st.markdown("---")
    st.markdown("#### ⏱️ Time-Series Replay / Timeline")
    total_days = len(raw_df)
    sim_index = st.slider(
        "Observation Time Window (Days)",
        min_value=1,
        max_value=total_days,
        value=total_days,
        help="Scrub through the timeline to evaluate point-in-time health degradation and alert triggers.",
    )

    st.caption(f"Showing day 1 through day {sim_index} of {total_days}.")

# Slice dataset to simulation window
sim_raw_df = raw_df.iloc[:sim_index].copy()

# -----------------------------------------------------------------------------
# Execute Preprocessing & Health Analytics Pipeline
# -----------------------------------------------------------------------------
preprocessor = DataPreprocessor()
clean_df, quality_report = preprocessor.preprocess(sim_raw_df)

scoring_engine = FilterHealthScoringEngine(
    rated_capacity_liters=rated_volume,
    rated_lifespan_days=int(rated_lifespan),
    tds_warning_ppm=tds_warn,
    tds_critical_ppm=tds_crit,
)
scored_df = scoring_engine.evaluate_dataframe(clean_df)

alert_engine = ExplainableAlertEngine(
    tds_warning_ppm=tds_warn,
    tds_critical_ppm=tds_crit,
    volume_limit_liters=rated_volume,
    age_limit_days=int(rated_lifespan),
)
all_alerts, final_df = alert_engine.evaluate_dataframe(scored_df)

# Most recent telemetry snapshot
latest_row = final_df.iloc[-1]
current_health = latest_row["filter_health_pct"]
current_status = latest_row["operational_status"]
current_tds = latest_row["tds_ppm"]
current_age = int(latest_row["filter_age_days"])
current_volume = latest_row["cumulative_volume_liters"]
latest_alerts = alert_engine.evaluate_record(latest_row)

# -----------------------------------------------------------------------------
# Header Presentation
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>💧 AquaSense Monitoring Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-title'>Intelligent Water Purifier Filter Health & Predictive Maintenance System | Scenario: <b>{scenario_option}</b></div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top KPI Metric Cards Row
# -----------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

# Status Badge Helper
def get_status_html(status: str) -> str:
    if status == "GOOD":
        return "<span class='status-badge-good'>🟢 GOOD</span>"
    elif status == "FAIR":
        return "<span class='status-badge-fair'>🟡 FAIR</span>"
    else:
        return "<span class='status-badge-replace'>🔴 REPLACE</span>"

with kpi_col1:
    health_color = "#16a34a" if current_health >= 70 else ("#d97706" if current_health >= 35 else "#dc2626")
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Estimated Filter Health</div>
            <div class='metric-value' style='color: {health_color};'>{current_health:.1f}%</div>
            <div class='metric-subtext'>Composite Health Index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Operational Status</div>
            <div style='margin-top: 6px; margin-bottom: 6px;'>{get_status_html(current_status)}</div>
            <div class='metric-subtext'>Threshold Recommendation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    tds_color = "#16a34a" if current_tds <= 100 else ("#d97706" if current_tds <= tds_warn else "#dc2626")
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Current TDS Output</div>
            <div class='metric-value' style='color: {tds_color};'>{current_tds:.1f} <span style='font-size: 1rem; color: #64748b;'>ppm</span></div>
            <div class='metric-subtext'>Warn: {tds_warn:.0f} | Crit: {tds_crit:.0f} ppm</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    vol_pct = (current_volume / rated_volume) * 100.0
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Cumulative Volume</div>
            <div class='metric-value'>{current_volume:,.0f} <span style='font-size: 1rem; color: #64748b;'>L</span></div>
            <div class='metric-subtext'>{vol_pct:.1f}% of {rated_volume:,.0f} L limit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col5:
    age_pct = (current_age / rated_lifespan) * 100.0
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Cartridge Age</div>
            <div class='metric-value'>{current_age} <span style='font-size: 1rem; color: #64748b;'>days</span></div>
            <div class='metric-subtext'>{age_pct:.1f}% of {rated_lifespan} days rated</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Active Alert Banner (if any on current simulated observation day)
# -----------------------------------------------------------------------------
if latest_alerts:
    for alert in latest_alerts:
        banner_class = (
            "alert-banner-critical"
            if alert.severity == "CRITICAL"
            else ("alert-banner-warning" if alert.severity == "WARNING" else "alert-banner-info")
        )
        st.markdown(
            f"""
            <div class='{banner_class}'>
                <b>[{alert.severity}] {alert.metric_name}:</b> {alert.explanation}<br/>
                <span style='font-size: 0.88rem; color: #1e293b;'><b>Recommended Maintenance Action:</b> {alert.recommended_action}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_trajectories, tab_alerts, tab_benchmarks, tab_data = st.tabs(
    [
        "📈 Telemetry & Trajectories",
        "🚨 Explainable Alert Log",
        "🧪 Evaluation Benchmark Suite",
        "📋 Data & Preprocessing Explorer",
    ]
)

# =============================================================================
# TAB 1: Live Telemetry & Trajectories
# =============================================================================
with tab_trajectories:
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        # Chart 1: TDS Trajectory with Warning & Critical Lines
        fig_tds = go.Figure()

        # Cleaned TDS Line
        fig_tds.add_trace(
            go.Scatter(
                x=final_df["timestamp"],
                y=final_df["tds_ppm"],
                mode="lines",
                name="TDS Output (ppm)",
                line=dict(color="#0284c7", width=2.5),
            )
        )

        # 7-Day Rolling Trend Line
        fig_tds.add_trace(
            go.Scatter(
                x=final_df["timestamp"],
                y=final_df["tds_rolling_avg"],
                mode="lines",
                name="7-Day Rolling Mean",
                line=dict(color="#6366f1", width=1.8, dash="dot"),
            )
        )

        # Highlight Raw Sensor Glitches/Outliers if present
        if "is_sensor_anomaly" in final_df and final_df["is_sensor_anomaly"].any():
            anomaly_points = final_df[final_df["is_sensor_anomaly"]]
            fig_tds.add_trace(
                go.Scatter(
                    x=anomaly_points["timestamp"],
                    y=anomaly_points["raw_tds_ppm"],
                    mode="markers",
                    name="Raw Sensor Glitch (Mitigated)",
                    marker=dict(color="#ef4444", size=8, symbol="x"),
                )
            )

        # Threshold lines
        fig_tds.add_hline(
            y=tds_warn,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Warning ({tds_warn:.0f} ppm)",
            annotation_position="top right",
        )
        fig_tds.add_hline(
            y=tds_crit,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Critical Spike ({tds_crit:.0f} ppm)",
            annotation_position="top right",
        )

        fig_tds.update_layout(
            title="<b>TDS Water Quality Trajectory (ppm)</b>",
            xaxis_title="Timeline",
            yaxis_title="Total Dissolved Solids (ppm)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        st.plotly_chart(fig_tds, use_container_width=True)

    with col_plot2:
        # Chart 2: Health Score Degradation
        fig_health = go.Figure()

        # Shaded background health bands
        fig_health.add_hrect(y0=70, y1=100, fillcolor="#dcfce7", opacity=0.35, line_width=0, layer="below")
        fig_health.add_hrect(y0=35, y1=70, fillcolor="#fef3c7", opacity=0.35, line_width=0, layer="below")
        fig_health.add_hrect(y0=0, y1=35, fillcolor="#fee2e2", opacity=0.35, line_width=0, layer="below")

        # Health Curve
        fig_health.add_trace(
            go.Scatter(
                x=final_df["timestamp"],
                y=final_df["filter_health_pct"],
                mode="lines",
                name="Filter Health Index (%)",
                line=dict(color="#0f766e", width=3),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.08)",
            )
        )

        # Threshold annotations
        fig_health.add_hline(y=70, line_dash="dot", line_color="#16a34a", annotation_text="Good Threshold (70%)")
        fig_health.add_hline(y=35, line_dash="dot", line_color="#dc2626", annotation_text="Replace Threshold (35%)")

        fig_health.update_layout(
            title="<b>Filter Health Degradation Index (0% - 100%)</b>",
            xaxis_title="Timeline",
            yaxis_title="Estimated Filter Health (%)",
            yaxis_range=[0, 105],
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        st.plotly_chart(fig_health, use_container_width=True)

    col_plot3, col_plot4 = st.columns(2)

    with col_plot3:
        # Chart 3: Cumulative Water Volume vs Capacity Limit
        fig_vol = go.Figure()

        fig_vol.add_trace(
            go.Scatter(
                x=final_df["timestamp"],
                y=final_df["cumulative_volume_liters"],
                mode="lines",
                name="Cumulative Usage (L)",
                line=dict(color="#0369a1", width=2.5),
            )
        )

        fig_vol.add_hline(
            y=rated_volume,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Rated Capacity Limit ({rated_volume:,.0f} L)",
            annotation_position="bottom right",
        )

        fig_vol.update_layout(
            title="<b>Cumulative Volume vs. Rated Media Capacity</b>",
            xaxis_title="Timeline",
            yaxis_title="Cumulative Liters Filtered",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_plot4:
        # Chart 4: Daily Consumption Liters (Detecting burst / leak surges)
        fig_daily = px.bar(
            final_df,
            x="timestamp",
            y="daily_usage_liters",
            title="<b>Daily Water Consumption & Leak Surge Detection</b>",
            labels={"daily_usage_liters": "Daily Usage (Liters)", "timestamp": "Timeline"},
            color="daily_usage_liters",
            color_continuous_scale="Blues",
        )
        fig_daily.add_hline(
            y=50.0,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="High Usage / Leak Surge (50 L/day)",
            annotation_position="top right",
        )
        fig_daily.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
        )
        st.plotly_chart(fig_daily, use_container_width=True)

# =============================================================================
# TAB 2: Explainable Alert Engine & Maintenance Log
# =============================================================================
with tab_alerts:
    st.markdown("### 🚨 Explainable Maintenance Alert Center")
    st.caption("Root-cause diagnostic engine delivering transparent, human-readable explanations and actionable guidance.")

    if not all_alerts:
        st.success("✅ **Clean Operational State:** No maintenance alerts triggered within the selected observation window.")
    else:
        # Summary counts
        crit_count = sum(1 for a in all_alerts if a.severity == "CRITICAL")
        warn_count = sum(1 for a in all_alerts if a.severity == "WARNING")
        info_count = sum(1 for a in all_alerts if a.severity == "INFO")

        alert_kpi1, alert_kpi2, alert_kpi3, alert_kpi4 = st.columns(4)
        with alert_kpi1:
            st.metric("Total Alerts Fired", len(all_alerts))
        with alert_kpi2:
            st.metric("Critical Alerts", crit_count, delta=f"{crit_count} acute risks", delta_color="inverse")
        with alert_kpi3:
            st.metric("Warning Alerts", warn_count, delta=f"{warn_count} approaching limits", delta_color="inverse")
        with alert_kpi4:
            st.metric("Advisory / Info", info_count)

        # Severity filter
        severity_filter = st.radio(
            "Filter Alerts by Severity",
            ["All Alerts", "CRITICAL", "WARNING", "INFO"],
            horizontal=True,
        )

        filtered_alerts = all_alerts
        if severity_filter != "All Alerts":
            filtered_alerts = [a for a in all_alerts if a.severity == severity_filter]

        st.markdown(f"**Showing {len(filtered_alerts)} alert entries:**")

        alert_records_data = []
        for alert in reversed(filtered_alerts):
            alert_records_data.append(
                {
                    "Timestamp": alert.timestamp,
                    "Severity": alert.severity,
                    "Subsystem / Metric": alert.metric_name,
                    "Triggered Value": f"{alert.current_value:.1f}",
                    "Threshold Limit": f"{alert.threshold_value:.1f}",
                    "Root Cause Explanation": alert.explanation,
                    "Actionable Recommended Maintenance": alert.recommended_action,
                }
            )

        alerts_display_df = pd.DataFrame(alert_records_data)
        st.dataframe(
            alerts_display_df,
            width="stretch",
            column_config={
                "Timestamp": st.column_config.TextColumn("Date / Time", width="small"),
                "Severity": st.column_config.TextColumn("Severity", width="small"),
                "Subsystem / Metric": st.column_config.TextColumn("Trigger Metric", width="medium"),
                "Triggered Value": st.column_config.TextColumn("Observed", width="small"),
                "Threshold Limit": st.column_config.TextColumn("Limit", width="small"),
                "Root Cause Explanation": st.column_config.TextColumn("Diagnostic Explanation", width="large"),
                "Actionable Recommended Maintenance": st.column_config.TextColumn("Action Guide", width="large"),
            },
            hide_index=True,
        )

# =============================================================================
# TAB 3: Evaluation Benchmark Suite
# =============================================================================
with tab_benchmarks:
    st.markdown("### 🧪 Level 1 Software Demonstrator Benchmark Evaluation")
    st.caption("Quantitative performance evaluation: detection accuracy, sensitivity, false alert rate (FAR), and latency across all labelled scenarios.")

    run_btn = st.button("🚀 Re-Run Full Benchmark Suite Across All Reference Scenarios", type="primary")

    # Generate benchmark suite
    gen = SyntheticDataGenerator(
        rated_capacity_liters=rated_volume,
        rated_lifespan_days=int(rated_lifespan),
    )
    benchmark_datasets = {
        "Normal": gen.generate_normal_scenario(),
        "Deteriorating": gen.generate_deteriorating_scenario(),
        "Abnormal": gen.generate_abnormal_scenario(),
    }

    suite = BenchmarkSuite(
        preprocessor=preprocessor,
        scoring_engine=scoring_engine,
        alert_engine=alert_engine,
    )

    benchmarks_df = suite.run_all_scenarios(benchmark_datasets)

    # Top benchmark aggregate metrics
    bm_col1, bm_col2, bm_col3, bm_col4 = st.columns(4)
    with bm_col1:
        avg_det_rate = benchmarks_df["detection_rate_pct"].mean()
        st.metric("Detection Rate (Sensitivity)", f"{avg_det_rate:.1f}%", help="Proportion of degraded or anomaly instances correctly flagged.")
    with bm_col2:
        avg_far = benchmarks_df["false_alert_rate_pct"].mean()
        st.metric("False Alert Rate (FAR)", f"{avg_far:.1f}%", help="Proportion of normal operations spuriously flagged with alerts.")
    with bm_col3:
        avg_acc = benchmarks_df["accuracy_pct"].mean()
        st.metric("Mean Accuracy", f"{avg_acc:.1f}%", help="Overall ground truth classification agreement.")
    with bm_col4:
        avg_lat = benchmarks_df["latency_per_record_us"].mean()
        st.metric("Processing Latency", f"{avg_lat:.1f} µs/record", help="Execution latency per time-series sample.")

    st.markdown("#### Scenario Benchmark Breakdown")
    st.dataframe(
        benchmarks_df.rename(
            columns={
                "scenario_name": "Scenario",
                "total_samples": "Samples (Days)",
                "accuracy_pct": "Accuracy (%)",
                "detection_rate_pct": "Detection Rate (%)",
                "false_alert_rate_pct": "False Alert Rate (%)",
                "processing_latency_ms": "Total Ingestion Time (ms)",
                "latency_per_record_us": "Latency / Record (µs)",
                "dashboard_refresh_sim_ms": "Dashboard Refresh (ms)",
                "alerts_triggered_count": "Alerts Triggered",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # Benchmark Visual Comparison Charts
    bm_chart_col1, bm_chart_col2 = st.columns(2)

    with bm_chart_col1:
        fig_bm_perf = go.Figure(
            data=[
                go.Bar(name="Detection Rate (%)", x=benchmarks_df["scenario_name"], y=benchmarks_df["detection_rate_pct"], marker_color="#0284c7"),
                go.Bar(name="False Alert Rate (%)", x=benchmarks_df["scenario_name"], y=benchmarks_df["false_alert_rate_pct"], marker_color="#f59e0b"),
            ]
        )
        fig_bm_perf.update_layout(
            barmode="group",
            title="<b>Detection Rate vs. False Alert Rate by Scenario</b>",
            yaxis_title="Percentage (%)",
            yaxis_range=[0, 115],
            template="plotly_white",
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bm_perf, use_container_width=True)

    with bm_chart_col2:
        fig_bm_lat = go.Figure(
            data=[
                go.Bar(
                    x=benchmarks_df["scenario_name"],
                    y=benchmarks_df["processing_latency_ms"],
                    marker_color="#6366f1",
                    text=benchmarks_df["processing_latency_ms"].apply(lambda v: f"{v:.1f} ms"),
                    textposition="auto",
                )
            ]
        )
        fig_bm_lat.update_layout(
            title="<b>Pipeline Processing Latency (ms)</b>",
            xaxis_title="Scenario",
            yaxis_title="Total Latency (milliseconds)",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bm_lat, use_container_width=True)

# =============================================================================
# TAB 4: Data & Preprocessing Explorer
# =============================================================================
with tab_data:
    st.markdown("### 📋 Telemetry Quality & Preprocessing Explorer")
    st.caption("Inspect pipeline cleaning steps, imputed missing values, spike mitigation, and feature-engineered metrics.")

    dq_col1, dq_col2, dq_col3, dq_col4 = st.columns(4)
    with dq_col1:
        st.metric("Total Records", quality_report["total_records"])
    with dq_col2:
        st.metric("Imputed TDS Nulls", quality_report["missing_tds_imputed"])
    with dq_col3:
        st.metric("Sensor Spikes Mitigated", quality_report["sensor_spikes_mitigated"])
    with dq_col4:
        st.metric("Data Quality Hygiene", f"{quality_report['data_health_score_pct']}%")

    st.markdown("#### Cleaned & Scored Telemetry Table")
    st.dataframe(final_df, width="stretch", hide_index=True)

    # CSV Download Button
    csv_buffer = io.StringIO()
    final_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Cleaned & Scored Telemetry (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"aquasense_processed_{scenario_option.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Required Safety Disclaimer Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class='safety-footer'>
        ⚠️ <b>Safety Disclaimer:</b> AquaSense provides an estimated filter-health status and maintenance guide.
        TDS alone cannot establish whether water is safe to drink or constitute a certified medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)

