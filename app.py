"""
Production Engineering Decision Tool
====================================
A production engineering decision-support tool for horizontal wells
in the Permian Basin (Delaware/Midland sub-basins).

Refactored: modular engine architecture with separated decline, nodal,
economics, and surveillance engines.
"""

from datetime import date
import json
import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Module imports ─────────────────────────────────────────
# Add project root to path so local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decline_engine import (
    evaluate_decline_model,
    fit_decline_curves,
    select_best_model,
    calc_eur,
    generate_monthly_forecast,
    generate_synthetic_production,
)
from nodal_engine import (
    vogel_ipr,
    compute_aof,
    compute_tpr,
    find_operating_point,
    invert_ipr_to_pwf,
    beggs_brill_gradient,
    screen_artificial_lift,
    size_esp,
)
from econ_engine import (
    calculate_economics,
    run_sensitivity_cases,
    tornado_sensitivity,
)
from portfolio_rules import (
    evaluate_well_flags,
    build_portfolio_table,
    portfolio_summary_counts,
    build_forecast_actual_table,
    compute_surveillance_flags,
)
from scenario_store import (
    SCENARIO_PRESETS,
    get_current_inputs,
    load_inputs_to_state,
    scenario_to_json,
    VALID_STATE_KEYS,
)
from report_builder import (
    build_assumptions,
    build_manager_report,
    build_engineer_report,
    build_case_report_html,
)
from unit_conversions import (
    fluid_sg_mixture,
    TUBING_ID_MAP,
    psi_to_ft_head,
)
from validation import (
    validate_reservoir_inputs,
    validate_well_inputs,
    validate_fluid_inputs,
    validate_decline_fit,
)
from survey_engine import (
    calculate_minimum_curvature,
    summarize_survey,
    recommend_lift_setting_windows,
)


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Production Engineering Toolkit — Permian Basin",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --app-bg: #edf3ef;
        --panel-bg: #f9fcfa;
        --panel-soft: #eef4f0;
        --panel-strong: #16322c;
        --panel-dark: #213630;
        --border: #c7d6cf;
        --text: #16211d;
        --muted: #4e635b;
        --tab-bg: #dfe9e2;
        --tab-active-bg: #24453c;
        --tab-active-text: #f7fbf8;
        --accent: #1f6a53;
        --accent-soft: #d6e9df;
        --warn: #9a6700;
        --danger: #b42318;
    }

    html, body, [class*="css"], .stApp, .main {
        font-family: 'Inter', sans-serif;
        color: var(--text) !important;
    }
    code, .stCode { font-family: 'JetBrains Mono', monospace; }
    .main .block-container {
        padding-top: 1.1rem;
        max-width: 1440px;
        color: var(--text) !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--app-bg) !important;
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--panel-bg) !important;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] *,
    [data-testid="stHeader"] *,
    .stMarkdown,
    .stText,
    p, li, label, span, div,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    .stDataFrame, .stTable {
        color: var(--text) !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        color: var(--text) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h2 {
        border-bottom: 2px solid var(--border);
        padding-bottom: 0.2rem;
    }
    h3 { font-weight: 600; }

    .hero-shell {
        background: linear-gradient(135deg, #17342d 0%, #29483f 45%, #3f6157 100%);
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        color: #f5faf7 !important;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 14px 32px rgba(20, 34, 29, 0.16);
    }
    .hero-shell h1, .hero-shell p, .hero-shell div {
        color: #f5faf7 !important;
        border: none !important;
    }

    .highlight-box, .decision-box, .warning-box, .danger-box, .report-box, .assumption-box, .confidence-box, .intake-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        margin: 0.8rem 0;
        border: 1px solid var(--border);
        background: var(--panel-bg);
        color: var(--text) !important;
        box-shadow: 0 10px 24px rgba(19, 35, 29, 0.05);
    }
    .highlight-box { border-left: 4px solid var(--accent); background: linear-gradient(180deg, #fbfdfc 0%, #f2f8f4 100%); }
    .decision-box { border-left: 4px solid #356859; background: #f4faf6; }
    .warning-box { border-left: 4px solid var(--warn); background: #fffaf0; }
    .danger-box { border-left: 4px solid var(--danger); background: #fff5f3; }
    .report-box { border-left: 4px solid var(--panel-dark); }
    .assumption-box { border-left: 4px solid #52796f; background: #f4faf7; }
    .confidence-box { border-left: 4px solid var(--accent); background: #f0f8f4; }
    .intake-box { border-left: 4px solid #335c51; background: #f7fbf8; }

    div[data-testid="stMetric"] {
        background: var(--panel-bg) !important;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.65rem;
        box-shadow: 0 10px 24px rgba(18, 35, 29, 0.06);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.65);
        padding: 0.3rem;
        border-radius: 14px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--tab-bg) !important;
        color: var(--text) !important;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: var(--tab-active-bg) !important;
        color: var(--tab-active-text) !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }

    .stAlert {
        color: var(--text) !important;
        background: var(--panel-bg) !important;
        border: 1px solid var(--border) !important;
    }
    .stAlert p, .stAlert div, .stAlert span {
        color: var(--text) !important;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stSelectbox"],
    [data-testid="stNumberInputContainer"],
    [data-baseweb="select"],
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div {
        background: var(--panel-bg) !important;
        color: var(--text) !important;
    }

    [data-testid="stDataFrame"] div,
    [data-testid="stTable"] div,
    .stDataFrame table,
    .stTable table {
        color: var(--text) !important;
    }

    .small-note {
        color: var(--muted) !important;
        font-size: 0.9rem;
    }
    .section-label {
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def bucket_risk(score):
    if score >= 4.2:
        return "Low"
    if score >= 3.2:
        return "Moderate"
    return "High"


def build_intervention_ranking(rate, depth, wc, gor, deviation, viscosity, pr,
                               gas_lift_available, oil_price, gas_price,
                               discount_rate, decline_model):
    lift_local, _, _ = screen_artificial_lift(rate, depth, gor, wc, viscosity, deviation, pr, gas_lift_available)
    fit_map = dict(zip(lift_local["Method"], lift_local["Weighted Score"]))
    rows = []

    # Do-nothing / natural flow case
    natural_npv = calculate_economics(
        "Natural Flow", max(rate * 0.85, 50), depth,
        oil_price=oil_price, gas_price=gas_price, gor=gor, wc=wc,
        discount_rate=discount_rate, decline_model=decline_model,
    )
    rows.append({
        "Option": "Natural Flow / defer lift",
        "Operational Fit": round(2.5 if rate > 0 else 1.0, 2),
        "Modeled NPV ($)": round(natural_npv["NPV"], 0),
        "CAPEX ($)": 0,
        "Payout (months)": 0,
        "Risk": "High" if rate <= 0 else "Moderate",
    })

    methods = ["ESP", "Rod Pump", "Gas Lift"] if gas_lift_available else ["ESP", "Rod Pump", "Plunger Lift"]
    for method in methods:
        econ = calculate_economics(
            method, rate, depth,
            oil_price=oil_price, gas_price=gas_price, gor=gor, wc=wc,
            discount_rate=discount_rate, decline_model=decline_model,
        )
        score = fit_map.get(method, 0)
        rows.append({
            "Option": method,
            "Operational Fit": round(score, 2),
            "Modeled NPV ($)": round(econ["NPV"], 0),
            "CAPEX ($)": round(econ["CAPEX Total"], 0),
            "Payout (months)": econ["Payout (months)"] if isinstance(econ["Payout (months)"], (int, float)) else 999,
            "Risk": bucket_risk(score),
        })

    df = pd.DataFrame(rows)
    df["Economic Rank"] = df["Modeled NPV ($)"].rank(ascending=False, method="dense")
    df["Fit Rank"] = df["Operational Fit"].rank(ascending=False, method="dense")
    df["Composite Rank"] = (df["Economic Rank"] + df["Fit Rank"]).rank(method="dense")
    return df.sort_values(["Composite Rank", "Modeled NPV ($)"], ascending=[True, False]).reset_index(drop=True)


def init_state_from_preset(preset_name: str):
    preset = SCENARIO_PRESETS[preset_name]
    for key, value in preset.items():
        st.session_state[key] = value


if "preset_initialized" not in st.session_state:
    init_state_from_preset("Base Delaware Well")
    st.session_state["scenario_name"] = "Base Delaware Well"
    st.session_state["preset_initialized"] = True

if "comparison_cases" not in st.session_state:
    st.session_state["comparison_cases"] = []

if "portfolio_manual_rows" not in st.session_state:
    st.session_state["portfolio_manual_rows"] = []


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Case Controls")
st.sidebar.caption("Detailed engineering inputs now live in the `Well Intake` tab.")

scenario_name = st.sidebar.selectbox(
    "Scenario preset",
    list(SCENARIO_PRESETS.keys()),
    index=list(SCENARIO_PRESETS.keys()).index(st.session_state.get("scenario_name", "Base Delaware Well")),
)
if scenario_name != st.session_state.get("scenario_name"):
    init_state_from_preset(scenario_name)
    st.session_state["scenario_name"] = scenario_name

with st.sidebar.expander("Scenario management", expanded=False):
    st.caption("Save the current case or load a prior case package.")
    current_inputs_for_export = get_current_inputs(st.session_state)
    st.download_button(
        "Download current scenario",
        data=scenario_to_json(current_inputs_for_export),
        file_name=f"{st.session_state.get('scenario_name', 'scenario').replace(' ', '_').lower()}_scenario.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded_scenario = st.file_uploader("Load scenario JSON", type=["json"], key="scenario_upload")
    if uploaded_scenario is not None:
        try:
            payload = json.load(uploaded_scenario)
            load_inputs_to_state(payload, st.session_state)
            st.session_state["scenario_name"] = payload.get("scenario_name", st.session_state.get("scenario_name", "Loaded scenario"))
            st.success("Scenario loaded. Refresh widgets by interacting with any input.")
        except Exception as exc:
            st.error(f"Could not load scenario file: {exc}")

st.markdown(
    f"""
<div class="hero-shell">
    <div class="section-label" style="color:#dce9e3 !important;">Production Engineering Workspace</div>
    <h1 style="margin:0 0 0.35rem 0;">Portfolio-first production decision support</h1>
    <p style="margin:0; max-width:980px;">
        Review well priority, capture structured intake data, and move from operating conditions to lift recommendation,
        decline, economics, surveillance, and a manager-ready decision package.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

tab_portfolio, tab_intake, tab_nodal, tab_lift, tab_decline, tab_econ, tab_sens, tab_surv, tab_summary = st.tabs([
    "Portfolio",
    "Well Intake",
    "Nodal + Intervention",
    "Lift Design",
    "Decline Analysis",
    "Economics",
    "Sensitivity / Comparison",
    "Surveillance",
    "Decision Summary",
])

survey_calculated = None
survey_summary = None
survey_error = None
survey_windows = {}

with tab_intake:
    st.header("Well Intake")
    st.markdown(
        """
<div class="intake-box">
<strong>Purpose:</strong> capture the operating data that drives production decisions. Inputs are grouped like an engineering intake sheet,
with unit guidance and constrained fields so the workflow feels more deliberate and less like a generic calculator.
</div>
""",
        unsafe_allow_html=True,
    )

    c_meta1, c_meta2, c_meta3 = st.columns([1.4, 1.2, 1.1])
    with c_meta1:
        st.text_input("Well / case label", key="scenario_name", help="Name used across portfolio tables, exports, and manager summaries.")
    with c_meta2:
        tubing_od = st.selectbox(
            "Tubing OD (in)",
            [2.375, 2.875, 3.5],
            index=[2.375, 2.875, 3.5].index(st.session_state["tubing_od"]),
            key="tubing_od",
            help="Tubing OD is used to infer tubing ID for the simplified tubing-performance calculations.",
        )
    with c_meta3:
        gas_lift_available = st.checkbox(
            "Gas-lift infrastructure available",
            value=st.session_state["gas_lift_available"],
            key="gas_lift_available",
            help="Controls whether gas lift is screened as a viable lift option.",
        )

    ri1, ri2 = st.columns(2)
    with ri1:
        st.markdown("<div class='section-label'>Reservoir And Inflow</div>", unsafe_allow_html=True)
        pr = st.number_input("Reservoir Pressure (psi)", min_value=0, max_value=10000, value=int(st.session_state["pr"]), step=100, key="pr", help="Current reservoir pressure estimate used in IPR and operating-point screening.")
        pb = st.number_input("Bubble Point Pressure (psi)", min_value=0, max_value=10000, value=int(st.session_state["pb"]), step=100, key="pb", help="Bubble point separates undersaturated and saturated inflow behavior.")
        pi = st.number_input("Productivity Index (STB/d/psi)", min_value=0.1, max_value=50.0, value=float(st.session_state["pi"]), step=0.1, key="pi", help="Primary inflow strength indicator for nodal calculations.")
        api_gravity = st.number_input("Oil API Gravity", min_value=10.0, max_value=60.0, value=float(st.session_state["api"]), step=1.0, key="api", help="Used in simplified PVT and flow-property screening.")
    with ri2:
        st.markdown("<div class='section-label'>Well And Operating Conditions</div>", unsafe_allow_html=True)
        depth = st.number_input("True Vertical Depth (ft)", min_value=1000, max_value=20000, value=int(st.session_state["depth"]), step=100, key="depth", help="Vertical depth used in hydrostatic and lift calculations.")
        deviation = st.number_input("Max Deviation (degrees)", min_value=0, max_value=90, value=int(st.session_state["deviation"]), step=1, key="deviation", help="Higher deviation increases rod-lift constraints and affects lift screening.")
        whp = st.number_input("Wellhead Pressure (psi)", min_value=0, max_value=2000, value=int(st.session_state["whp"]), step=25, key="whp", help="Surface backpressure applied in tubing-performance calculations.")
        viscosity = st.number_input("Oil Viscosity (cp)", min_value=0.5, max_value=100.0, value=float(st.session_state["viscosity"]), step=0.5, key="viscosity", help="Oil viscosity estimate used in lift screening and ESP sizing logic.")

    fi1, fi2 = st.columns(2)
    with fi1:
        st.markdown("<div class='section-label'>Fluids</div>", unsafe_allow_html=True)
        wc_pct = st.number_input("Water Cut (%)", min_value=0, max_value=99, value=int(st.session_state["wc_pct"]), step=1, key="wc_pct", help="Current produced water cut used in nodal, lift, surveillance, and economics.")
        gor = st.number_input("GOR (scf/bbl)", min_value=0, max_value=10000, value=int(st.session_state["gor"]), step=50, key="gor", help="Gas-oil ratio used in tubing performance, gas-risk screening, and revenue estimates.")
    with fi2:
        st.markdown("<div class='section-label'>Economics And Constraints</div>", unsafe_allow_html=True)
        oil_price = st.number_input("Oil Price ($/bbl)", min_value=0.0, max_value=200.0, value=float(st.session_state["oil_price"]), step=1.0, key="oil_price", help="Flat realized oil price used in screening economics.")
        gas_price = st.number_input("Gas Price ($/Mscf)", min_value=0.0, max_value=20.0, value=float(st.session_state["gas_price"]), step=0.1, key="gas_price", help="Flat realized gas price used in screening economics.")
        discount_pct = st.number_input("Discount Rate (%)", min_value=0.0, max_value=50.0, value=float(st.session_state["discount_pct"]), step=1.0, key="discount_pct", help="Annual discount rate used for NPV and payout.")
        discount_rate = discount_pct / 100.0

    st.markdown("<div class='section-label'>Trajectory And Survey</div>", unsafe_allow_html=True)
    survey_c1, survey_c2 = st.columns([1.2, 1])
    with survey_c1:
        st.markdown(
            """
<div class="intake-box">
Upload a directional survey when you want the app to compute TVD and dogleg severity from actual stations.
That gives the lift workflow a real placement window instead of relying only on a single max-deviation input.
</div>
""",
            unsafe_allow_html=True,
        )
        survey_upload = st.file_uploader(
            "Upload directional survey (CSV or Excel)",
            type=["csv", "xlsx"],
            key="survey_upload",
            help="Expected columns: MD, Inclination, Azimuth. Common aliases like inc / azi are also accepted.",
        )
    with survey_c2:
        survey_template_df = pd.DataFrame(
            [
                {"MD": 0, "Inclination": 0, "Azimuth": 0},
                {"MD": 1500, "Inclination": 5, "Azimuth": 90},
                {"MD": 4500, "Inclination": 32, "Azimuth": 92},
                {"MD": 9000, "Inclination": 84, "Azimuth": 95},
                {"MD": 12500, "Inclination": 91, "Azimuth": 96},
            ]
        )
        st.download_button(
            "Download survey template (CSV)",
            data=survey_template_df.to_csv(index=False),
            file_name="directional_survey_template.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption("Future connector layer: Enverus or public-well sources can feed the same survey workflow once credentials and source rules are wired in.")

    if survey_upload is not None:
        try:
            if survey_upload.name.lower().endswith(".csv"):
                survey_raw_df = pd.read_csv(survey_upload)
            else:
                survey_raw_df = pd.read_excel(survey_upload)
            survey_calculated = calculate_minimum_curvature(survey_raw_df)
            survey_summary = summarize_survey(survey_calculated)
            survey_windows = {
                lift_name: recommend_lift_setting_windows(survey_calculated, lift_name)
                for lift_name in ["ESP", "Gas Lift", "Rod Pump", "Plunger Lift"]
            }
        except Exception as exc:
            survey_error = str(exc)

    if survey_error:
        st.error(f"Survey could not be processed: {survey_error}")
    elif survey_summary is not None:
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Survey stations", f"{survey_summary['stations']}")
        sm2.metric("Max MD", f"{survey_summary['max_md']:,.0f} ft")
        sm3.metric("Computed TVD", f"{survey_summary['max_tvd']:,.0f} ft")
        sm4.metric("Max inclination", f"{survey_summary['max_inclination']:.1f} deg")
        sm5.metric("Max DLS", f"{survey_summary['max_dls']:.2f} deg/100 ft")

        fig_survey = make_subplots(rows=1, cols=2, subplot_titles=("Trajectory", "Dogleg severity"))
        fig_survey.add_trace(
            go.Scatter(
                x=survey_calculated["MD"],
                y=survey_calculated["TVD"],
                mode="lines+markers",
                name="TVD vs MD",
            ),
            row=1,
            col=1,
        )
        fig_survey.add_trace(
            go.Scatter(
                x=survey_calculated["MD"],
                y=survey_calculated["DLS"],
                mode="lines+markers",
                name="DLS",
            ),
            row=1,
            col=2,
        )
        fig_survey.add_hline(y=3.0, line_dash="dot", line_color="#1f6a53", row=1, col=2)
        fig_survey.add_hline(y=5.0, line_dash="dot", line_color="#9a6700", row=1, col=2)
        fig_survey.update_xaxes(title_text="MD (ft)", row=1, col=1)
        fig_survey.update_xaxes(title_text="MD (ft)", row=1, col=2)
        fig_survey.update_yaxes(title_text="TVD (ft)", autorange="reversed", row=1, col=1)
        fig_survey.update_yaxes(title_text="DLS (deg/100 ft)", row=1, col=2)
        fig_survey.update_layout(template="plotly_white", height=360, showlegend=False)
        st.plotly_chart(fig_survey, width="stretch")

        survey_window_rows = []
        for lift_name, placement in survey_windows.items():
            best_window = placement.get("recommended_window")
            survey_window_rows.append({
                "Lift": lift_name,
                "Preferred DLS limit": f"{placement['limits']['preferred']:.1f}",
                "Recommended MD": "No clear window" if not best_window else f"{best_window['recommended_md']:,.0f} ft",
                "Recommended TVD": "No clear window" if not best_window else f"{best_window['recommended_tvd']:,.0f} ft",
                "Window length": "N/A" if not best_window else f"{best_window['length']:,.0f} ft",
            })
        st.markdown("**Survey-informed placement windows**")
        st.dataframe(pd.DataFrame(survey_window_rows), width="stretch", hide_index=True)

    depth_effective = survey_summary["max_tvd"] if survey_summary is not None else depth
    deviation_effective = survey_summary["max_inclination"] if survey_summary is not None else deviation
    trajectory_source_label = "Directional survey" if survey_summary is not None else "Manual well intake"

    raw_input_summary = pd.DataFrame([
        {"Category": "Reservoir", "Field": "Reservoir Pressure", "Value": pr, "Units": "psi"},
        {"Category": "Reservoir", "Field": "Bubble Point", "Value": pb, "Units": "psi"},
        {"Category": "Reservoir", "Field": "PI", "Value": pi, "Units": "STB/d/psi"},
        {"Category": "Reservoir", "Field": "API Gravity", "Value": api_gravity, "Units": "deg API"},
        {"Category": "Well", "Field": "TVD", "Value": depth, "Units": "ft"},
        {"Category": "Well", "Field": "Deviation", "Value": deviation, "Units": "deg"},
        {"Category": "Well", "Field": "Tubing OD", "Value": tubing_od, "Units": "in"},
        {"Category": "Operations", "Field": "Water Cut", "Value": wc_pct, "Units": "%"},
        {"Category": "Operations", "Field": "GOR", "Value": gor, "Units": "scf/bbl"},
        {"Category": "Operations", "Field": "WHP", "Value": whp, "Units": "psi"},
        {"Category": "Operations", "Field": "Viscosity", "Value": viscosity, "Units": "cp"},
        {"Category": "Economics", "Field": "Oil Price", "Value": oil_price, "Units": "$/bbl"},
        {"Category": "Economics", "Field": "Gas Price", "Value": gas_price, "Units": "$/Mscf"},
        {"Category": "Economics", "Field": "Discount Rate", "Value": discount_pct, "Units": "%"},
        {"Category": "Constraints", "Field": "Gas Lift Available", "Value": "Yes" if gas_lift_available else "No", "Units": ""},
        {"Category": "Trajectory", "Field": "Survey Loaded", "Value": "Yes" if survey_summary is not None else "No", "Units": ""},
        {"Category": "Trajectory", "Field": "Trajectory source", "Value": trajectory_source_label if survey_summary is not None else "Manual well intake", "Units": ""},
        {"Category": "Trajectory", "Field": "Effective TVD for calcs", "Value": round(depth_effective, 1) if survey_summary is not None else depth, "Units": "ft"},
        {"Category": "Trajectory", "Field": "Effective deviation for calcs", "Value": round(deviation_effective, 1) if survey_summary is not None else deviation, "Units": "deg"},
    ])
    raw_input_summary["Value"] = raw_input_summary["Value"].astype(str)
    st.markdown("<div class='section-label'>Raw Input Summary</div>", unsafe_allow_html=True)
    st.dataframe(raw_input_summary, width="stretch", hide_index=True)

wc = wc_pct / 100
d_t = TUBING_ID_MAP.get(tubing_od, 2.441)


# ─────────────────────────────────────────────────────────────
# APP-WIDE CALCULATIONS
# ─────────────────────────────────────────────────────────────

# Input validation
res_val = validate_reservoir_inputs(pr, pb, pi, api_gravity)
well_val = validate_well_inputs(depth_effective, deviation_effective, tubing_od)
fluid_val = validate_fluid_inputs(wc_pct, gor, whp)

# IPR
pwf_range = np.linspace(0, pr * 1.05, 300)
q_ipr = vogel_ipr(pr, pb, pi, pwf_range)
aof = float(np.nanmax(q_ipr))

# TPR
q_tpr_range = np.linspace(10, max(50, aof * 1.2), 150)
bhp_tpr = compute_tpr(q_tpr_range, wc, gor, d_t, whp, depth_effective, api_gravity)

# Operating point (new: returns dict)
op_result = find_operating_point(pr, pb, pi, wc, gor, d_t, whp, depth_effective, api_gravity)
q_op = op_result["q_op"]
pwf_op = op_result["pwf_op"]
rate_for_lift = q_op if q_op is not None else 500

# Fluid SG
fluid_sg = fluid_sg_mixture(wc, api_gravity)

# Decline
prod_df = generate_synthetic_production(months=60)
models, preprocessing_info = fit_decline_curves(prod_df)

# Auto-select best model
best_model_name, model_selection_reason = select_best_model(models)
decline_model = models.get(best_model_name) if best_model_name else models.get("Hyperbolic", models.get("Exponential"))

# Decline confidence
decline_confidence = "high"
if decline_model:
    dcv = validate_decline_fit(decline_model, decline_model.get("rmse"), decline_model.get("r2"))
    decline_confidence = dcv["confidence"]

# Manager action flags
actions = []
if q_op is None:
    actions.append(("critical", "No natural-flow operating point at current conditions. Artificial lift should be treated as immediate work, not future planning."))
elif q_op / max(aof, 1) < 0.35:
    actions.append(("warning", "Natural-flow margin is narrow. A small pressure decline or higher water load can push the well into unstable flow."))
if gor >= 1500:
    actions.append(("warning", "High GOR raises gas-interference risk. Any ESP recommendation needs intake gas handling and tighter surveillance limits."))
if wc >= 0.75:
    actions.append(("warning", "High water cut will drive disposal cost and hydrostatic loading. Late-life lift economics should be checked carefully."))
if deviation_effective >= 70:
    actions.append(("warning", "High deviation penalizes rod-lift reliability and should trigger stronger scrutiny on pump setting depth and trajectory."))
if survey_summary is not None and survey_summary["max_dls"] >= 5:
    actions.append(("warning", "Directional survey shows elevated dogleg severity. Use survey-informed setting windows before locking in lift equipment depth."))
if not gas_lift_available:
    actions.append(("warning", "Gas lift is being penalized because surface gas infrastructure is marked unavailable."))
if decline_confidence == "low":
    actions.append(("warning", "Decline-curve fit confidence is low. Forecasts and EUR should be used with caution."))

# Add nodal warnings to actions
for w in op_result.get("warnings", []):
    # Avoid duplicates
    if not any(w in msg for _, msg in actions):
        actions.append(("warning", w))

# Lift screening (new: returns 3 items)
lift_df, criteria_weights, ranking_notes = screen_artificial_lift(
    rate_for_lift, depth_effective, gor, wc, viscosity, deviation_effective, pr, gas_lift_available
)
best_lift = lift_df.iloc[0]["Method"]
lift_method_options = ["Natural Flow"] + lift_df["Method"].tolist()
methods_to_compare_global = ["ESP", "Rod Pump", "Gas Lift"] if gas_lift_available else ["ESP", "Rod Pump", "Plunger Lift"]
econ_results_global = {
    m: calculate_economics(
        m, rate_for_lift, depth_effective, years=10,
        oil_price=oil_price, gas_price=gas_price, gor=gor, wc=wc,
        discount_rate=discount_rate, decline_model=decline_model,
    )
    for m in methods_to_compare_global
}

current_inputs = {
    "scenario_name": scenario_name,
    "pr": pr, "pb": pb, "pi": pi, "api": api_gravity,
    "depth": depth_effective, "deviation": deviation_effective, "tubing_od": tubing_od,
    "wc_pct": wc_pct, "gor": gor, "whp": whp, "viscosity": viscosity,
    "oil_price": oil_price, "gas_price": gas_price, "discount_pct": discount_rate * 100,
    "gas_lift_available": gas_lift_available,
    "trajectory_source": trajectory_source_label,
}
assumptions = build_assumptions(pr, pb, wc_pct, gor, gas_lift_available)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("Production Engineering Decision Tool")
st.markdown(
    f"""
<div class="highlight-box">
<strong>Scenario:</strong> {scenario_name} |
<strong>Tubing:</strong> {tubing_od:.3f} in |
<strong>Depth:</strong> {depth_effective:,.0f} ft TVD |
<strong>Water cut:</strong> {wc_pct}% |
<strong>GOR:</strong> {gor:,} scf/bbl |
<strong>Trajectory source:</strong> {trajectory_source_label} |
<strong>Status:</strong> {'Natural flow established' if q_op is not None else 'No natural-flow solution'}
</div>
""",
    unsafe_allow_html=True,
)

# Validation warnings
all_val_warnings = res_val["warnings"] + well_val["warnings"] + fluid_val["warnings"]
if all_val_warnings:
    st.markdown(
        "<div class='warning-box'><strong>Input warnings:</strong><br>"
        + "<br>".join([f"&bull; {w}" for w in all_val_warnings])
        + "</div>",
        unsafe_allow_html=True,
    )

if actions:
    for level, text in actions:
        box_class = "danger-box" if level == "critical" else "warning-box"
        st.markdown(f"<div class='{box_class}'><strong>Manager flag:</strong> {text}</div>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("AOF", f"{aof:,.0f} BPD")
m2.metric("Current rate", f"{rate_for_lift:,.0f} BPD")
m3.metric("Flowing BHP", f"{pwf_op:,.0f} psi" if pwf_op is not None else "No solution")
m4.metric("Recommended lift", best_lift)
m5.metric("Pressure regime", "Below Pb" if pr < pb else "Above Pb")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
# TAB 1 — NODAL ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab_nodal:
    st.header("Nodal + Intervention")
    st.markdown(
        """
<div class="highlight-box">
Composite Vogel IPR is paired with a simplified tubing-performance model. Use this page as the operating-mode workspace:
start with natural flow, then test a selected lift strategy and see how much pressure reduction or lift energy is required to reach the target rate.
</div>
""",
        unsafe_allow_html=True,
    )

    default_intervention = best_lift if q_op is None else "Natural Flow"
    if default_intervention not in lift_method_options:
        default_intervention = lift_method_options[0]

    ic1, ic2, ic3 = st.columns([1.2, 1.1, 1.1])
    with ic1:
        selected_intervention = st.selectbox(
            "Operating mode",
            lift_method_options,
            index=lift_method_options.index(default_intervention),
            help="Natural flow uses the current IPR/TPR intersection. Assisted modes let you test a target operating rate and the pressure reduction required from lift.",
            key="selected_intervention_mode",
        )
    with ic2:
        max_target_rate = int(max(300, min(aof * 0.98, 5000)))
        default_target_rate = int(max(rate_for_lift, q_op or rate_for_lift))
        if selected_intervention == "Natural Flow":
            target_rate = q_op if q_op is not None else rate_for_lift
            st.metric("Target operating rate", f"{target_rate:,.0f} BPD" if target_rate else "No natural flow")
        else:
            target_rate = st.slider(
                "Lift target rate (BPD)",
                min_value=100,
                max_value=max_target_rate,
                value=min(default_target_rate, max_target_rate),
                step=25,
                key="selected_intervention_target_rate",
            )
    with ic3:
        if selected_intervention == "Natural Flow":
            st.metric("Mode intent", "As-is well performance")
        else:
            st.metric("Mode intent", f"{selected_intervention} intervention")

    selected_pwf = None
    selected_rate = None
    natural_bhp_at_target = None
    assist_pressure = None
    assist_head_ft = None
    selected_mode_summary = ""

    if selected_intervention == "Natural Flow":
        selected_rate = q_op
        selected_pwf = pwf_op
        selected_mode_summary = (
            f"Natural flow operating point is {q_op:,.0f} BPD."
            if q_op is not None and pwf_op is not None
            else "No natural-flow operating point exists at the current intake conditions."
        )
    else:
        selected_rate = float(target_rate)
        selected_pwf = invert_ipr_to_pwf(pr, pb, pi, selected_rate)
        _, _, natural_bhp_at_target = beggs_brill_gradient(selected_rate, wc, gor, d_t, whp, depth_effective, api_gravity)
        if selected_pwf is not None:
            assist_pressure = max(natural_bhp_at_target - selected_pwf, 0.0)
            assist_head_ft = psi_to_ft_head(assist_pressure, fluid_sg)
            selected_mode_summary = (
                f"{selected_intervention} at {selected_rate:,.0f} BPD would require flowing pressure near {selected_pwf:,.0f} psi, "
                f"or about {assist_pressure:,.0f} psi ({assist_head_ft:,.0f} ft) of lift assistance versus the natural tubing requirement."
            )
        else:
            selected_mode_summary = (
                f"The requested {selected_intervention} target of {selected_rate:,.0f} BPD is above the modeled inflow capacity. "
                "Reduce the target rate or revisit the inflow assumptions."
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_ipr.tolist(), y=pwf_range.tolist(), name="IPR", line=dict(width=3, color="#1f6a53")))
    fig.add_trace(go.Scatter(x=q_tpr_range.tolist(), y=bhp_tpr.tolist(), name=f'Natural TPR ({tubing_od}" tubing)', line=dict(width=3, color="#506b63")))
    if q_op is not None and pwf_op is not None:
        fig.add_trace(
            go.Scatter(
                x=[q_op], y=[pwf_op],
                mode="markers+text",
                text=[f"{q_op:,.0f} BPD"],
                textposition="top center",
                marker=dict(size=14, symbol="diamond", color="#17342d"),
                name="Natural-flow operating point",
            )
        )
    if selected_intervention != "Natural Flow" and selected_pwf is not None and selected_rate is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected_rate], y=[selected_pwf],
                mode="markers+text",
                text=[f"{selected_intervention}: {selected_rate:,.0f} BPD"],
                textposition="bottom center",
                marker=dict(size=16, symbol="circle", color="#b45309"),
                name=f"{selected_intervention} target",
            )
        )
        fig.add_shape(
            type="line",
            x0=selected_rate, y0=selected_pwf, x1=selected_rate, y1=natural_bhp_at_target,
            line=dict(color="#b45309", dash="dot", width=2),
        )
    fig.add_hline(y=pb, line_dash="dash", annotation_text=f"Bubble point = {pb} psi")
    fig.update_layout(
        title="IPR vs Natural TPR With Selected Intervention",
        xaxis_title="Flow Rate (STB/d)",
        yaxis_title="Flowing Bottomhole Pressure (psi)",
        template="plotly_white",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AOF", f"{aof:,.0f} BPD")
    c2.metric("Natural-flow rate", f"{q_op:,.0f} BPD" if q_op is not None else "No intersection")
    c3.metric("Selected mode rate", f"{selected_rate:,.0f} BPD" if selected_rate is not None else "N/A")
    if selected_intervention == "Natural Flow":
        c4.metric("% of AOF", f"{q_op / aof * 100:.0f}%" if q_op is not None and aof > 0 else "N/A")
    else:
        c4.metric("Lift assist needed", f"{assist_pressure:,.0f} psi" if assist_pressure is not None else "Target above inflow")

    if selected_intervention != "Natural Flow":
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Required Pwf", f"{selected_pwf:,.0f} psi" if selected_pwf is not None else "Above inflow")
        mc2.metric("Natural BHP at target", f"{natural_bhp_at_target:,.0f} psi" if natural_bhp_at_target is not None else "N/A")
        mc3.metric("Equivalent head", f"{assist_head_ft:,.0f} ft" if assist_head_ft is not None else "N/A")
        mc4.metric("Chosen lift", selected_intervention)

    st.markdown(
        f"<div class='decision-box'><strong>Intervention readout:</strong> {selected_mode_summary}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Lift screening on the same nodal page", expanded=False):
        st.dataframe(
            lift_df.style.map(
                lambda val: (
                    "background-color: #dff3e4" if isinstance(val, (int, float)) and val >= 4.5
                    else "background-color: #eef7d4" if isinstance(val, (int, float)) and val >= 3.5
                    else "background-color: #fff7cc" if isinstance(val, (int, float)) and val >= 2.5
                    else "background-color: #ffe1bf" if isinstance(val, (int, float)) and val >= 1.5
                    else "background-color: #ffd9d9" if isinstance(val, (int, float))
                    else ""
                ),
                subset=[c for c in lift_df.columns if c != "Method"],
            ),
            width="stretch",
            height=240,
        )
        st.caption("5 = strong fit. 1 = poor fit.")
        st.markdown("**Why the current ranking looks this way**")
        for note in ranking_notes:
            st.markdown(f"- {note}")

    # Stability indicator
    stability = op_result.get("stability", "none")
    stability_colors = {"stable": "#059669", "marginal": "#ca8a04", "none": "#dc2626"}
    st.markdown(
        f'<div class="confidence-box"><strong>Flow stability:</strong> '
        f'<span style="color:{stability_colors.get(stability, "#475569")}; font-weight:700;">{stability.upper()}</span>'
        f' — {"Well has healthy natural-flow margin." if stability == "stable" else "Natural-flow margin is limited; plan for lift." if stability == "marginal" else "No natural flow at current conditions."}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Pressure-decline sensitivity")
    pr_values = np.arange(pr, max(800, pr - 2000), -200)
    if len(pr_values) < 3:
        pr_values = np.arange(pr, 800, -200)
    sens_rows = []
    for p in pr_values:
        op_s = find_operating_point(p, min(pb, p) if p < pb else pb, pi, wc, gor, d_t, whp, depth_effective, api_gravity)
        q_s = op_s["q_op"]
        pwf_s = op_s["pwf_op"]
        if q_s is not None and pwf_s is not None:
            sens_rows.append({"Pr (psi)": p, "Rate (BPD)": q_s, "Pwf (psi)": pwf_s})
    if sens_rows:
        sens_df = pd.DataFrame(sens_rows)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=sens_df["Pr (psi)"].astype(str), y=sens_df["Rate (BPD)"], name="Rate"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=sens_df["Pr (psi)"].astype(str), y=sens_df["Pwf (psi)"], mode="lines+markers", name="Pwf"), secondary_y=True)
        fig2.update_layout(template="plotly_white", height=380)
        fig2.update_yaxes(title_text="Rate (BPD)", secondary_y=False)
        fig2.update_yaxes(title_text="Pwf (psi)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    if selected_intervention == "Natural Flow" and q_op is not None and pwf_op is not None:
        margin_desc = "healthy margin for continued natural flow" if q_op / max(aof, 1) > 0.5 else "limited margin, so lift planning should move from concept to execution planning"
        st.markdown(
            f"""
<div class="decision-box">
<strong>Production-manager readout:</strong> The well is flowing at roughly <strong>{q_op:,.0f} BPD</strong> with <strong>{pr - pwf_op:,.0f} psi</strong> of drawdown. This is {margin_desc}.
</div>
""",
            unsafe_allow_html=True,
        )
    elif selected_intervention == "Natural Flow":
        st.markdown(
            """
<div class="danger-box">
<strong>Production-manager readout:</strong> At these inputs, the well does not show a stable natural-flow intersection. Treat the well as an active lift candidate now.
</div>
""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════
# TAB 2 — LIFT SELECTION
# ═══════════════════════════════════════════════════════════
with tab_lift:
    st.header("Lift Design Support")
    st.markdown(
        f"""
<div class="highlight-box">
Screening uses current well conditions: <strong>{rate_for_lift:,.0f} BPD</strong>, <strong>{depth:,} ft</strong>, <strong>{gor:,} scf/bbl</strong>, <strong>{wc_pct}% water cut</strong>, <strong>{deviation}° deviation</strong>.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="decision-box">
<strong>Workflow intent:</strong> Use this page after the nodal screen to carry the chosen intervention into a lift-specific pre-design review.
The goal is not just to rank lift options, but to decide what has to be checked, sized, or ruled out before the team advances one path.
</div>
""",
        unsafe_allow_html=True,
    )

    def color_score(val):
        if isinstance(val, (int, float)):
            if val >= 4.5:
                return "background-color: #dff3e4"
            if val >= 3.5:
                return "background-color: #eef7d4"
            if val >= 2.5:
                return "background-color: #fff7cc"
            if val >= 1.5:
                return "background-color: #ffe1bf"
                return "background-color: #ffd9d9"
        return ""

    design_lift_default = selected_intervention if selected_intervention != "Natural Flow" else best_lift
    if design_lift_default not in methods_to_compare_global:
        design_lift_default = methods_to_compare_global[0]

    design_c1, design_c2, design_c3 = st.columns([1.2, 1, 1])
    with design_c1:
        design_lift = st.selectbox(
            "Lift detail mode",
            methods_to_compare_global,
            index=methods_to_compare_global.index(design_lift_default),
            help="Select the intervention you want to evaluate in more detail on this page.",
            key="design_lift_mode",
        )
    with design_c2:
        design_econ = econ_results_global.get(design_lift, {})
        st.metric("Screening NPV", f"${design_econ.get('NPV', 0):,.0f}")
    with design_c3:
        design_score = float(lift_df.loc[lift_df["Method"] == design_lift, "Weighted Score"].iloc[0])
        st.metric("Weighted fit score", f"{design_score:.2f}")

    design_basis_items = [
        {"Item": "Operating mode from nodal tab", "Value": selected_intervention},
        {"Item": "Detailed lift path on this tab", "Value": design_lift},
        {"Item": "Current well rate", "Value": f"{rate_for_lift:,.0f} BPD"},
        {"Item": "Top screening recommendation", "Value": best_lift},
        {"Item": "Depth", "Value": f"{depth_effective:,.0f} ft"},
        {"Item": "Water cut", "Value": f"{wc_pct:.0f}%"},
        {"Item": "GOR", "Value": f"{gor:,.0f} scf/bbl"},
        {"Item": "Deviation", "Value": f"{deviation_effective:.0f} deg"},
    ]
    if selected_rate is not None:
        design_basis_items.append({"Item": "Target rate from nodal tab", "Value": f"{selected_rate:,.0f} BPD"})
    if assist_pressure is not None and selected_intervention != "Natural Flow":
        design_basis_items.append({"Item": "Estimated lift assist", "Value": f"{assist_pressure:,.0f} psi"})
    if assist_head_ft is not None and selected_intervention == "ESP":
        design_basis_items.append({"Item": "Equivalent hydraulic head", "Value": f"{assist_head_ft:,.0f} ft"})

    st.markdown("**Design basis carried from the intervention screen**")
    design_basis_df = pd.DataFrame(design_basis_items)
    design_basis_df["Value"] = design_basis_df["Value"].astype(str)
    st.dataframe(design_basis_df, width="stretch", hide_index=True)

    if survey_summary is not None:
        selected_window = survey_windows.get(design_lift, {}).get("recommended_window")
        st.markdown("**Survey-informed placement guidance**")
        if selected_window is None:
            st.markdown(
                f"""
<div class="warning-box">
<strong>Survey readout:</strong> No clear <strong>{design_lift}</strong> placement window met the screening DLS limits from the uploaded survey.
Keep the lift ranking, but treat the setting depth as unresolved until trajectory and completion constraints are reviewed.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            placement_df = pd.DataFrame(
                [
                    {"Item": "Recommended MD", "Value": f"{selected_window['recommended_md']:,.0f} ft"},
                    {"Item": "Recommended TVD", "Value": f"{selected_window['recommended_tvd']:,.0f} ft"},
                    {"Item": "Window MD start", "Value": f"{selected_window['md_start']:,.0f} ft"},
                    {"Item": "Window MD end", "Value": f"{selected_window['md_end']:,.0f} ft"},
                    {"Item": "Window length", "Value": f"{selected_window['length']:,.0f} ft"},
                    {"Item": "Trajectory source", "Value": trajectory_source_label},
                ]
            )
            st.dataframe(placement_df, width="stretch", hide_index=True)

    if selected_intervention != "Natural Flow" and design_lift != selected_intervention:
        st.markdown(
            f"""
<div class="warning-box">
<strong>Workflow note:</strong> The nodal tab is currently testing <strong>{selected_intervention}</strong>, but this tab is open on <strong>{design_lift}</strong>.
That can be useful for comparing alternatives, but engineers should align the two before finalizing a recommendation package.
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("Reference lift screening", expanded=False):
        st.dataframe(
            lift_df.style.map(color_score, subset=[c for c in lift_df.columns if c != "Method"]),
            width="stretch",
            height=240,
        )
        st.caption("5 = strong fit. 1 = poor fit.")

        st.markdown("**Why this lift ranks first**")
        for note in ranking_notes:
            st.markdown(f"- {note}")

        radar_categories = [c for c in lift_df.columns if c not in ["Method", "Weighted Score"]]
        fig_radar = go.Figure()
        for _, row in lift_df.iterrows():
            vals = [row[c] for c in radar_categories]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=radar_categories + [radar_categories[0]],
                    fill="toself",
                    opacity=0.14,
                    name=f"{row['Method']} ({row['Weighted Score']:.2f})",
                )
            )
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5.5])), template="plotly_white", height=500)
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown(
        f"""
<div class="decision-box">
<strong>Current detail focus:</strong> {design_lift}. Top screening recommendation remains <strong>{best_lift}</strong>. {'Gas lift is heavily penalized where gas infrastructure is not available.' if not gas_lift_available else ''}
</div>
""",
        unsafe_allow_html=True,
    )

    # Constraint warnings
    constraint_warnings = []
    if design_lift == "ESP" and gor > 1500:
        constraint_warnings.append("High GOR requires enhanced gas handling (separator + AGH) for any ESP installation.")
    if design_lift == "ESP" and deviation_effective > 70:
        constraint_warnings.append("High deviation narrows ESP setting-depth options. Confirm dogleg severity.")
    if design_lift == "Rod Pump" and deviation_effective > 55:
        constraint_warnings.append("Rod pump at this deviation may have accelerated wear and reduced runlife.")
    if survey_summary is not None and survey_windows.get(design_lift, {}).get("recommended_window") is None:
        constraint_warnings.append(f"Directional survey does not show a clear {design_lift} placement window within the screening DLS limits.")
    if design_lift == "Gas Lift" and not gas_lift_available:
        constraint_warnings.append("Gas lift is selected, but surface gas infrastructure is currently marked unavailable.")
    if design_lift == "Plunger Lift" and wc_pct > 60:
        constraint_warnings.append("High water cut can make plunger-lift cycling less reliable and may reduce liquid-unloading efficiency.")
    if q_op is not None and q_op / max(aof, 1) < 0.35:
        constraint_warnings.append("Narrow natural-flow margin — lift installation timeline should be accelerated.")
    if constraint_warnings:
        st.markdown(
            "<div class='warning-box'><strong>Constraint warnings:</strong><br>"
            + "<br>".join([f"&bull; {w}" for w in constraint_warnings])
            + "</div>",
            unsafe_allow_html=True,
        )

    lift_workflow_titles = {
        "ESP": "ESP pre-design review",
        "Gas Lift": "Gas lift pre-design review",
        "Rod Pump": "Rod-lift pre-design review",
        "Plunger Lift": "Plunger-lift candidacy review",
    }
    st.subheader(lift_workflow_titles.get(design_lift, f"{design_lift} follow-up"))
    if design_lift == "ESP":
        ce1, ce2 = st.columns(2)
        with ce1:
            esp_default_rate = int(max(100, target_rate if selected_intervention == "ESP" and selected_rate is not None else rate_for_lift * 1.1))
            target_esp_rate = st.number_input("Target production rate (BPD)", 100, 5000, esp_default_rate, 50)
        with ce2:
            friction_loss = depth_effective * 0.02
            base_lift_head = assist_head_ft if selected_intervention == "ESP" and assist_head_ft is not None else 0
            tdh_default = int(max(1000, depth_effective + psi_to_ft_head(whp, fluid_sg) + friction_loss + base_lift_head))
            tdh = st.number_input("Total dynamic head (ft)", 1000, 20000, tdh_default, 100)
        vsd_freq = st.slider("VSD operating frequency (Hz)", 40, 70, 60, 1)

        esp_results, pump_info = size_esp(target_esp_rate, tdh, fluid_sg, vsd_freq, gor)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Pump / hydraulic selection**")
            for k in ["Pump Series", "Pump OD (in)", "Operating Range (BPD)", "BEP Rate (BPD)", "Head per Stage (ft)", "Number of Stages", "Total Head (ft)"]:
                st.markdown(f"- **{k}:** {esp_results[k]}")
        with e2:
            st.markdown("**Power / ancillary selection**")
            for k in ["Required BHP", "Motor HP (nameplate)", "Motor Voltage (V)", "Motor Amps", "Cable Size", "Cable Type", "Gas Handling"]:
                st.markdown(f"- **{k}:** {esp_results[k]}")

        q_curve = np.linspace(pump_info["min_q"], pump_info["max_q"], 120)
        bep = pump_info["bep_q"]
        hps = pump_info["head_per_stage"]
        h_curve = hps * esp_results["Number of Stages"] * (1 - 0.5 * ((q_curve - bep) / max(bep, 1)) ** 2)
        h_curve = np.maximum(h_curve, 0)
        fig_esp = go.Figure()
        fig_esp.add_trace(go.Scatter(x=q_curve, y=h_curve, fill="tozeroy", name="Head-capacity curve"))
        fig_esp.add_hline(y=tdh, line_dash="dash", annotation_text=f"TDH = {tdh:,} ft")
        fig_esp.add_vline(x=target_esp_rate, line_dash="dash", annotation_text=f"Target = {target_esp_rate:,} BPD")
        fig_esp.add_vrect(x0=bep * 0.8, x1=bep * 1.2, opacity=0.12, line_width=0, annotation_text="Preferred BEP band")
        fig_esp.update_layout(template="plotly_white", height=420, xaxis_title="Rate (BPD)", yaxis_title="Head (ft)")
        st.plotly_chart(fig_esp, width="stretch")

        st.markdown(
            """
<div class="assumption-box">
<strong>Screening note:</strong> ESP sizing here is screening-grade. It uses a representative pump catalog and simplified affinity-law adjustments. Final design requires vendor coordination, well trajectory review, and detailed thermal/electrical calculations.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Engineering workflow**")
        st.markdown("- Confirm the nodal target rate is the rate you actually want to hold after installation, not just the highest achievable point.")
        st.markdown("- Validate setting depth, free-gas handling, and cable / motor constraints before sending the case to detailed pump design.")
        st.markdown("- Move to vendor-level design only after the hydraulic target, power envelope, and completion geometry all still make sense together.")
    elif design_lift == "Gas Lift":
        gl1, gl2, gl3 = st.columns(3)
        with gl1:
            gas_lift_target = st.number_input(
                "Target production rate (BPD)",
                100, 5000,
                int(max(100, target_rate if selected_intervention == "Gas Lift" and selected_rate is not None else rate_for_lift * 1.1)),
                50,
            )
        with gl2:
            injection_pressure = st.number_input("Available injection pressure (psi)", 200, 4000, int(max(600, whp + 600)), 50)
        with gl3:
            injection_gas = st.number_input("Available injection gas (Mscf/d)", 100, 20000, int(max(500, gor * gas_lift_target / 1500)), 100)

        gl_req_pwf = invert_ipr_to_pwf(pr, pb, pi, gas_lift_target)
        _, _, gl_natural_bhp = beggs_brill_gradient(gas_lift_target, wc, gor, d_t, whp, depth_effective, api_gravity)
        gl_assist = max(gl_natural_bhp - gl_req_pwf, 0.0) if gl_req_pwf is not None else None
        gl_delta_p = max(injection_pressure - whp, 0)

        gm1, gm2, gm3, gm4 = st.columns(4)
        gm1.metric("Required Pwf", f"{gl_req_pwf:,.0f} psi" if gl_req_pwf is not None else "Above inflow")
        gm2.metric("Natural BHP at target", f"{gl_natural_bhp:,.0f} psi")
        gm3.metric("Lift assist estimate", f"{gl_assist:,.0f} psi" if gl_assist is not None else "N/A")
        gm4.metric("Injection delta P", f"{gl_delta_p:,.0f} psi")

        st.markdown(
            f"""
<div class="assumption-box">
<strong>Gas-lift follow-up:</strong> At {gas_lift_target:,.0f} BPD, the well would need roughly
<strong>{gl_assist:,.0f} psi</strong> of effective tubing unloading if the inflow target is achievable.
Use the available injection pressure and gas rate as gating checks before moving to detailed valve-spacing design.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Engineer checklist**")
        st.markdown("- Confirm gas source capacity and compressor margin.")
        st.markdown("- Confirm injection pressure exceeds operating WHP by a workable margin.")
        st.markdown("- Review depth, kick-off, and expected injection point strategy before detailed design.")
        st.markdown("- Treat this as pre-design screening, not valve-spacing or mandrel design.")
        st.markdown("**Engineering workflow**")
        st.markdown("- Use the nodal target as the production objective, then test whether injection pressure and gas availability can realistically support it.")
        st.markdown("- If pressure margin is weak, revisit the target rate before proceeding into valve-depth work.")
        st.markdown("- Advance to detailed gas-lift design only when the surface network and injection margin are both credible.")
    elif design_lift == "Rod Pump":
        rp1, rp2, rp3 = st.columns(3)
        with rp1:
            rod_target = st.number_input(
                "Target production rate (BPD)",
                50, 2500,
                int(max(50, target_rate if selected_intervention == "Rod Pump" and selected_rate is not None else min(rate_for_lift, 1200))),
                25,
            )
        with rp2:
            default_pump_depth = int(depth_effective * 0.85)
            rod_window = survey_windows.get("Rod Pump", {}).get("recommended_window") if survey_summary is not None else None
            if rod_window is not None:
                default_pump_depth = int(rod_window["recommended_md"])
            pump_setting_depth = st.number_input("Pump setting depth (ft)", 1000, int(max(depth_effective, 1000)), int(max(default_pump_depth, 1000)), 100)
        with rp3:
            stroke_length = st.selectbox("Stroke length (in)", [86, 100, 120, 144], index=1)

        suggested_spm = 6 if rod_target < 300 else (8 if rod_target < 700 else 10)
        plunger_diameter = 1.5 if rod_target < 250 else (1.75 if rod_target < 600 else 2.25)

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Target rate", f"{rod_target:,.0f} BPD")
        rm2.metric("Pump depth", f"{pump_setting_depth:,.0f} ft")
        rm3.metric("Suggested SPM", f"{suggested_spm}")
        rm4.metric("Plunger diameter", f"{plunger_diameter:.2f} in")

        st.markdown(
            f"""
<div class="assumption-box">
<strong>Rod-pump follow-up:</strong> This case screens like a rod-lift candidate if mechanical loading and deviation risk remain manageable.
At <strong>{deviation:.0f}°</strong> deviation and <strong>{rod_target:,.0f} BPD</strong> target rate, use this as a pre-design check before rod-string and surface-unit sizing.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Engineer checklist**")
        st.markdown("- Check deviation and dogleg severity before committing to rod lift.")
        st.markdown("- Confirm polished-rod load envelope and gearbox range in detailed design.")
        st.markdown("- Validate pump fillage expectations at current GOR and fluid load.")
        st.markdown("- Move to rod-string design only after the mechanical risk looks acceptable.")
        st.markdown("**Engineering workflow**")
        st.markdown("- Start with the nodal target, then pressure-test whether the well belongs in a rod-lift window from both rate and mechanical-risk perspectives.")
        st.markdown("- If deviation or gas interference is too severe, use this screen to rule rod lift out early rather than forcing a detailed design.")
        st.markdown("- Advance only after the completion geometry, target rate, and surface-unit envelope are aligned.")
    else:
        pl1, pl2, pl3 = st.columns(3)
        with pl1:
            pl_target = st.number_input(
                "Target production rate (BPD)",
                20, 1000,
                int(max(20, target_rate if selected_intervention == "Plunger Lift" and selected_rate is not None else min(rate_for_lift, 400))),
                10,
            )
        with pl2:
            shutin_minutes = st.number_input("Estimated shut-in / build-up time (min)", 10, 720, 90, 10)
        with pl3:
            line_pressure = st.number_input("Sales / flowline pressure (psi)", 50, 1000, int(max(whp, 150)), 25)

        candidate_score = 0
        candidate_score += 1 if gor > 800 else 0
        candidate_score += 1 if wc_pct < 60 else 0
        candidate_score += 1 if pl_target < 400 else 0
        candidate_score += 1 if depth_effective < 12000 else 0

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Target rate", f"{pl_target:,.0f} BPD")
        pm2.metric("Cycle build-up", f"{shutin_minutes} min")
        pm3.metric("Flowline pressure", f"{line_pressure:,.0f} psi")
        pm4.metric("Plunger candidacy", f"{candidate_score}/4")

        st.markdown(
            """
<div class="assumption-box">
<strong>Plunger-lift follow-up:</strong> Use this section to screen whether intermittent lift is still realistic before moving into cycle tuning.
This app is not yet modeling full plunger-cycle optimization, but it can frame whether the well belongs in that conversation.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Engineer checklist**")
        st.markdown("- Confirm gas energy is adequate to unload liquids consistently.")
        st.markdown("- Check whether water cut and line pressure leave enough cyclic margin.")
        st.markdown("- Use this as candidacy screening before any detailed controller tuning.")
        st.markdown("**Engineering workflow**")
        st.markdown("- Use the nodal screen to decide whether intermittent lift is the right conversation at all for this well.")
        st.markdown("- If the target rate depends on continuous drawdown support, plunger lift is likely the wrong path.")
        st.markdown("- Move into cycle optimization only after the well looks like a genuine intermittent-lift candidate.")

# ═══════════════════════════════════════════════════════════
# TAB 3 — DECLINE ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab_decline:
    st.header("Decline Analysis")
    st.markdown(
        """
<div class="highlight-box">
Synthetic data is used here for demonstration. In a field deployment, this tab would accept uploaded production history files. Preprocessing automatically filters shut-ins and outliers before fitting.
</div>
""",
        unsafe_allow_html=True,
    )

    # Preprocessing info
    if preprocessing_info and preprocessing_info.get("flags"):
        st.markdown(
            "<div class='assumption-box'><strong>Data preprocessing:</strong><br>"
            + "<br>".join([f"&bull; {f}" for f in preprocessing_info["flags"]])
            + f"<br>&bull; Data quality score: <strong>{preprocessing_info.get('quality_score', 'N/A')}</strong>"
            + f" ({preprocessing_info.get('n_points_used', '?')}/{preprocessing_info.get('n_points_total', '?')} points used)"
            + "</div>",
            unsafe_allow_html=True,
        )
    if preprocessing_info and preprocessing_info.get("fit_errors"):
        st.warning("Some decline models could not be fit:\n\n- " + "\n- ".join(preprocessing_info["fit_errors"]))

    # Model selection
    model_choice = st.radio(
        "Model selection",
        ["Auto-select best model"] + list(models.keys()),
        horizontal=True,
    )
    if model_choice == "Auto-select best model":
        active_model_name = best_model_name
    else:
        active_model_name = model_choice
    active_model = models.get(active_model_name)

    if active_model_name and model_selection_reason:
        st.markdown(
            f'<div class="confidence-box"><strong>Selected model:</strong> {active_model_name}. '
            f'{model_selection_reason}</div>',
            unsafe_allow_html=True,
        )

    # Plot
    fig_dca = go.Figure()
    fig_dca.add_trace(go.Scatter(x=prod_df["Month"], y=prod_df["Oil Rate (BOPD)"], mode="markers", name="Actual production"))
    t_forecast = np.arange(1, 361)
    eur_results = {}
    for name, model in models.items():
        q_pred = np.maximum(evaluate_decline_model(model, t_forecast), 0)
        eur_val, t_end = calc_eur(model)
        eur_results[name] = eur_val
        r2_str = f" | R²={model.get('r2', 'N/A')}"
        label = f"{name} | EUR {eur_val:,.0f} Mbbl{r2_str}"
        line_style = dict(width=3) if name == active_model_name else dict(width=1, dash="dot")
        fig_dca.add_trace(go.Scatter(x=t_forecast, y=q_pred, name=label, line=line_style))
    fig_dca.add_hline(y=5, line_dash="dot", annotation_text="Economic limit")
    fig_dca.update_layout(template="plotly_white", height=540, yaxis_type="log", xaxis_title="Month", yaxis_title="Oil Rate (BOPD)")
    st.plotly_chart(fig_dca, use_container_width=True)

    # Model comparison table
    st.subheader("Model comparison")
    model_comp_rows = []
    for name, m in models.items():
        model_comp_rows.append({
            "Model": name,
            "qi (BOPD)": f"{m['qi']:.0f}",
            "di (1/mo)": f"{m['di']:.4f}",
            "b": f"{m.get('b', 0):.3f}",
            "R²": f"{m.get('r2', 'N/A')}",
            "RMSE": f"{m.get('rmse', 'N/A')}",
            "MAPE (%)": f"{m.get('mape', 'N/A')}",
            "EUR (Mbbl)": f"{eur_results.get(name, 0):,.0f}",
            "Selected": "✓" if name == active_model_name else "",
        })
    st.dataframe(pd.DataFrame(model_comp_rows), use_container_width=True, hide_index=True)

    # EUR metrics
    d1, d2, d3 = st.columns(3)
    d1.metric("Exponential EUR", f"{eur_results.get('Exponential', 0):,.0f} Mbbl")
    d2.metric("Hyperbolic EUR", f"{eur_results.get('Hyperbolic', 0):,.0f} Mbbl")
    d3.metric("Harmonic EUR", f"{eur_results.get('Harmonic', 0):,.0f} Mbbl")

    # Confidence / reliability note
    if decline_confidence in ("low", "moderate"):
        st.markdown(
            f'<div class="warning-box"><strong>Forecast reliability:</strong> Decline-fit confidence is '
            f'<strong>{decline_confidence}</strong>. EUR and economic forecasts based on this fit should be '
            f'treated as approximate. Consider supplementary data or type-curve benchmarks before committing capital.</div>',
            unsafe_allow_html=True,
        )

    # Fluid trends
    fig_fluid = make_subplots(rows=1, cols=2, subplot_titles=("Water cut trend", "GOR trend"))
    fig_fluid.add_trace(go.Scatter(x=prod_df["Month"], y=prod_df["Water Cut"] * 100, mode="lines+markers"), row=1, col=1)
    fig_fluid.add_trace(go.Scatter(x=prod_df["Month"], y=prod_df["GOR (scf/bbl)"], mode="lines+markers"), row=1, col=2)
    fig_fluid.update_layout(template="plotly_white", height=350, showlegend=False)
    fig_fluid.update_xaxes(title_text="Month", row=1, col=1)
    fig_fluid.update_xaxes(title_text="Month", row=1, col=2)
    fig_fluid.update_yaxes(title_text="Water Cut (%)", row=1, col=1)
    fig_fluid.update_yaxes(title_text="GOR (scf/bbl)", row=1, col=2)
    st.plotly_chart(fig_fluid, use_container_width=True)

    st.markdown(
        """
<div class="decision-box">
<strong>Production-manager readout:</strong> Hyperbolic decline remains the most reasonable base case for this unconventional-well profile. Rising water cut and rising GOR point to tougher late-life lift economics and stronger gas-handling requirements.
</div>
""",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════
# TAB 4 — ECONOMICS
# ═══════════════════════════════════════════════════════════
with tab_econ:
    st.header("Economics")
    st.markdown(
        """
<div class="highlight-box">
Monthly cash-flow model with separated CAPEX, fixed OPEX, variable OPEX, water disposal, and equipment replacement events. All costs are representative screening-grade figures.
</div>
""",
        unsafe_allow_html=True,
    )

    methods_to_compare = ["ESP", "Rod Pump", "Gas Lift"] if gas_lift_available else ["ESP", "Rod Pump", "Plunger Lift"]
    econ_results = {
        m: calculate_economics(
            m, rate_for_lift, depth_effective, years=10,
            oil_price=oil_price, gas_price=gas_price, gor=gor, wc=wc,
            discount_rate=discount_rate, decline_model=decline_model,
        )
        for m in methods_to_compare
    }
    best_econ_method, best_econ_data = max(econ_results.items(), key=lambda x: x[1]["NPV"])

    # Do-nothing comparator
    do_nothing_econ = calculate_economics(
        "Natural Flow", max(rate_for_lift * 0.85, 50), depth_effective, years=10,
        oil_price=oil_price, gas_price=gas_price, gor=gor, wc=wc,
        discount_rate=discount_rate, decline_model=decline_model,
    )

    cols = st.columns(len(methods_to_compare) + 1)
    # Do-nothing column
    with cols[0]:
        st.markdown("### Do Nothing")
        st.metric("CAPEX", "$0")
        st.metric("Annual fixed OPEX", f"${do_nothing_econ['Annual OPEX']:,.0f}")
        st.metric("NPV", f"${do_nothing_econ['NPV']:,.0f}")
        st.metric("Payout", "N/A")
        st.metric("Lifting cost", f"${do_nothing_econ['Lifting Cost ($/BOE)']:.2f}/BOE")

    for i, method in enumerate(methods_to_compare):
        e = econ_results[method]
        with cols[i + 1]:
            st.markdown(f"### {method}")
            st.metric("CAPEX", f"${e['CAPEX Total']:,.0f}")
            st.metric("Annual fixed OPEX", f"${e['Annual OPEX']:,.0f}")
            st.metric("NPV", f"${e['NPV']:,.0f}")
            payout = e['Payout (years)']
            st.metric("Payout", f"{payout} yr" if isinstance(payout, (int, float)) else str(payout))
            st.metric("Lifting cost", f"${e['Lifting Cost ($/BOE)']:.2f}/BOE")
            if e.get("Capital Efficiency (NPV/CAPEX)") != "N/A":
                st.metric("Capital efficiency", f"{e['Capital Efficiency (NPV/CAPEX)']}x")

    fig_cost = make_subplots(rows=1, cols=2, subplot_titles=("CAPEX", "NPV"))
    all_methods = ["Do Nothing"] + methods_to_compare
    all_capex = [0] + [econ_results[m]["CAPEX Total"] for m in methods_to_compare]
    all_npv = [do_nothing_econ["NPV"]] + [econ_results[m]["NPV"] for m in methods_to_compare]
    fig_cost.add_trace(go.Bar(x=all_methods, y=all_capex, name="CAPEX"), row=1, col=1)
    fig_cost.add_trace(go.Bar(x=all_methods, y=all_npv, name="NPV"), row=1, col=2)
    fig_cost.update_layout(template="plotly_white", height=400, showlegend=False)
    st.plotly_chart(fig_cost, use_container_width=True)

    # Cost breakdown table
    st.subheader("Cost breakdown")
    cost_breakdown_rows = []
    for m in methods_to_compare:
        e = econ_results[m]
        cost_breakdown_rows.append({
            "Method": m,
            "Equipment": f"${e.get('CAPEX Equipment', 0):,.0f}",
            "Installation": f"${e.get('CAPEX Installation', 0):,.0f}",
            "Ancillary": f"${e.get('CAPEX Ancillary', 0):,.0f}",
            "Other CAPEX": f"${e.get('CAPEX Other', 0):,.0f}",
            "Total CAPEX": f"${e['CAPEX Total']:,.0f}",
            "Payout (months)": e["Payout (months)"],
            "Total Water Disposal": f"${e['Total Water Disposal']:,.0f}",
            "Replacements": e["Replacement Events in Period"],
        })
    st.dataframe(pd.DataFrame(cost_breakdown_rows), use_container_width=True, hide_index=True)

    st.markdown(
        f"""
<div class="decision-box">
<strong>Economic recommendation:</strong> {best_econ_method} generates the highest NPV in this scenario. Use this ranking only alongside the operating constraints from the lift-screening tab; economics alone should not overrule a bad mechanical fit.
</div>
""",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY / COMPARISON
# ═══════════════════════════════════════════════════════════
with tab_sens:
    st.header("Sensitivity / Comparison")

    sensitivity_df = run_sensitivity_cases(
        current_inputs, decline_model,
        find_op_fn=lambda *a, **kw: find_operating_point(*a, **kw),
        screen_fn=screen_artificial_lift,
    )
    st.markdown(
        "<div class='highlight-box'><strong>Sensitivity view:</strong> This shows where the recommendation is stable and where it changes under a few realistic case shifts.</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Bar(x=sensitivity_df["Scenario"], y=sensitivity_df["Best NPV ($)"], name="Best NPV"))
    fig_sens.update_layout(template="plotly_white", height=380, yaxis_title="Best NPV ($)")
    st.plotly_chart(fig_sens, use_container_width=True)

    # Tornado sensitivity
    st.subheader("Tornado sensitivity (best economics method)")
    if best_econ_method and econ_results.get(best_econ_method):
        tornado_df = tornado_sensitivity(
            econ_results[best_econ_method], best_econ_method, rate_for_lift, depth_effective,
            current_inputs, decline_model,
        )
        st.dataframe(tornado_df, use_container_width=True, hide_index=True)

        # Tornado chart
        fig_tornado = go.Figure()
        base_npv = econ_results[best_econ_method]["NPV"]
        for _, row in tornado_df.iterrows():
            fig_tornado.add_trace(go.Bar(
                y=[row["Parameter"]],
                x=[row["High Case NPV ($)"] - row["Low Case NPV ($)"]],
                base=[row["Low Case NPV ($)"]],
                orientation="h",
                name=row["Parameter"],
                showlegend=False,
            ))
        fig_tornado.add_vline(x=base_npv, line_dash="dash", annotation_text=f"Base NPV ${base_npv:,.0f}")
        fig_tornado.update_layout(template="plotly_white", height=300, xaxis_title="NPV ($)", barmode="overlay")
        st.plotly_chart(fig_tornado, use_container_width=True)

    st.subheader("Scenario comparison set")
    cadd1, cadd2 = st.columns([1, 1])
    with cadd1:
        if st.button("Add current case to comparison set", use_container_width=True):
            case_record = {
                "Scenario": scenario_name,
                "Operating Rate (BPD)": round(rate_for_lift, 0),
                "Recommended Lift": best_lift,
                "Best NPV Method": best_econ_method,
                "Water Cut (%)": wc_pct,
                "GOR (scf/bbl)": gor,
                "Oil Price ($/bbl)": oil_price,
            }
            if case_record not in st.session_state["comparison_cases"]:
                st.session_state["comparison_cases"].append(case_record)
    with cadd2:
        if st.button("Clear comparison set", use_container_width=True):
            st.session_state["comparison_cases"] = []
    if st.session_state["comparison_cases"]:
        st.dataframe(pd.DataFrame(st.session_state["comparison_cases"]), use_container_width=True, hide_index=True)
    else:
        st.info("No saved comparison cases yet.")

    st.subheader("One-well intervention ranking")
    intervention_df = build_intervention_ranking(
        rate_for_lift, depth, wc, gor, deviation, viscosity, pr,
        gas_lift_available, oil_price, gas_price, discount_rate, decline_model,
    )
    st.dataframe(intervention_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# TAB 6 — SURVEILLANCE
# ═══════════════════════════════════════════════════════════
with tab_surv:
    st.header("Surveillance")
    st.markdown(
        "<div class='highlight-box'><strong>Forecast vs actual:</strong> This section is for routine surveillance rather than design. Enter recent actual oil rates to see variance against the screening forecast.</div>",
        unsafe_allow_html=True,
    )

    if "actuals_editor_seed" not in st.session_state or st.button("Reset surveillance table to default"):
        st.session_state["actuals_editor_seed"] = build_forecast_actual_table(prod_df, models)
    actual_editor = st.data_editor(
        st.session_state["actuals_editor_seed"],
        use_container_width=True, num_rows="fixed", hide_index=True, key="actuals_editor",
    )
    actual_editor = pd.DataFrame(actual_editor)
    actual_editor["Variance (%)"] = np.where(
        actual_editor["Forecast Oil Rate (BOPD)"] > 0,
        (actual_editor["Actual Oil Rate (BOPD)"] - actual_editor["Forecast Oil Rate (BOPD)"]) / actual_editor["Forecast Oil Rate (BOPD)"] * 100,
        0,
    )
    st.dataframe(actual_editor, use_container_width=True, hide_index=True)

    avg_var = float(actual_editor["Variance (%)"].mean()) if not actual_editor.empty else 0
    worst_var = float(actual_editor["Variance (%)"].min()) if not actual_editor.empty else 0
    sv1, sv2, sv3 = st.columns(3)
    sv1.metric("Average variance", f"{avg_var:.1f}%")
    sv2.metric("Worst variance", f"{worst_var:.1f}%")
    sv3.metric("Months below -10%", int((actual_editor["Variance (%)"] < -10).sum()))

    fig_sv = go.Figure()
    fig_sv.add_trace(go.Scatter(x=actual_editor["Month"], y=actual_editor["Forecast Oil Rate (BOPD)"], mode="lines+markers", name="Forecast"))
    fig_sv.add_trace(go.Scatter(x=actual_editor["Month"], y=actual_editor["Actual Oil Rate (BOPD)"], mode="lines+markers", name="Actual"))
    fig_sv.update_layout(template="plotly_white", height=380, yaxis_title="Oil Rate (BOPD)")
    st.plotly_chart(fig_sv, use_container_width=True)

    surv_flags = compute_surveillance_flags(actual_editor, gor, wc_pct)
    st.markdown(
        "<div class='report-box'><strong>Surveillance report:</strong><br>"
        + "<br>".join([f"&bull; {item}" for item in surv_flags])
        + "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════
# TAB 7 — PORTFOLIO DASHBOARD (NEW)
# ═══════════════════════════════════════════════════════════
with tab_portfolio:
    st.header("Portfolio Review")
    st.markdown(
        """
<div class="highlight-box">
<strong>Multi-well review view.</strong> Rank wells by urgency, blend current-case data with uploads and manual rows, and drill into the raw operating data behind the flags.
</div>
""",
        unsafe_allow_html=True,
    )

    current_well_row = {
        "well_name": f"Current: {scenario_name}",
        "q_actual": round(rate_for_lift, 1),
        "q_forecast": round(rate_for_lift * 1.05, 1),
        "wc_pct": wc_pct,
        "gor": gor,
        "lifting_cost_boe": econ_results.get(best_econ_method, {}).get("Lifting Cost ($/BOE)", 0) if 'econ_results' in dir() else 10,
        "natural_flow": q_op is not None,
        "margin_pct": op_result.get("margin_pct", 0),
        "decline_confidence": decline_confidence,
        "npv": econ_results.get(best_econ_method, {}).get("NPV", 0) if 'econ_results' in dir() else 0,
        "best_lift": best_lift,
        "deviation": deviation,
    }

    st.subheader("Portfolio Intake")
    st.markdown(
        """
<div class="highlight-box">
Build the review set from three sources: the active well, manual rows entered in-app, and customer portfolio uploads.
Optional built-in sample wells can stay on for demo mode or be turned off for real reviews.
</div>
""",
        unsafe_allow_html=True,
    )

    intake_c1, intake_c2 = st.columns([1.2, 1.4])
    with intake_c1:
        include_current = st.checkbox("Include active well", value=True, help="Adds the current case to the portfolio table.")
        include_samples = st.checkbox("Include built-in sample wells", value=True, help="Useful for demo mode or showing how the portfolio ranking behaves.")
        if st.button("Add active well to manual portfolio set", width="stretch"):
            st.session_state["portfolio_manual_rows"].append(dict(current_well_row))
    with intake_c2:
        portfolio_template_df = pd.DataFrame([current_well_row])
        st.download_button(
            "Download portfolio template (CSV)",
            data=portfolio_template_df.to_csv(index=False),
            file_name="portfolio_template.csv",
            mime="text/csv",
            width="stretch",
        )
        uploaded_portfolio = st.file_uploader("Upload portfolio CSV or Excel", type=["csv", "xlsx"], key="portfolio_upload")

    manual_portfolio_df = pd.DataFrame(
        st.session_state["portfolio_manual_rows"]
        if st.session_state["portfolio_manual_rows"]
        else [dict(current_well_row)]
    )
    manual_portfolio_df = st.data_editor(
        manual_portfolio_df,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="portfolio_manual_editor",
    )
    st.session_state["portfolio_manual_rows"] = manual_portfolio_df.to_dict("records")

    uploaded_portfolio_df = pd.DataFrame()
    if uploaded_portfolio is not None:
        try:
            if uploaded_portfolio.name.lower().endswith(".csv"):
                uploaded_portfolio_df = pd.read_csv(uploaded_portfolio)
            else:
                uploaded_portfolio_df = pd.read_excel(uploaded_portfolio)
        except Exception as exc:
            st.error(f"Could not read uploaded portfolio file: {exc}")

    portfolio_wells = []
    if include_current:
        portfolio_wells.append(current_well_row)

    if include_samples:
        for preset_name, preset in SCENARIO_PRESETS.items():
            if preset_name == scenario_name:
                continue
            preset_wc = preset["wc_pct"] / 100.0
            preset_d_t = TUBING_ID_MAP.get(preset["tubing_od"], 2.441)
            preset_op = find_operating_point(
                preset["pr"], preset["pb"], preset["pi"], preset_wc,
                preset["gor"], preset_d_t, preset["whp"], preset["depth"], preset["api"],
            )
            preset_rate = preset_op["q_op"] if preset_op["q_op"] else 400
            preset_lift, _, _ = screen_artificial_lift(
                preset_rate, preset["depth"], preset["gor"], preset_wc,
                preset["viscosity"], preset["deviation"], preset["pr"], preset["gas_lift_available"],
            )
            preset_econ = calculate_economics(
                preset_lift.iloc[0]["Method"], preset_rate, preset["depth"],
                oil_price=preset["oil_price"], gas_price=preset["gas_price"],
                gor=preset["gor"], wc=preset_wc, discount_rate=preset["discount_pct"] / 100,
                decline_model=decline_model,
            )
            portfolio_wells.append({
                "well_name": preset_name,
                "q_actual": preset_rate * 0.92,
                "q_forecast": preset_rate,
                "wc_pct": preset["wc_pct"],
                "gor": preset["gor"],
                "lifting_cost_boe": preset_econ["Lifting Cost ($/BOE)"],
                "natural_flow": preset_op.get("natural_flow", False),
                "margin_pct": preset_op.get("margin_pct", 0),
                "decline_confidence": "moderate",
                "npv": preset_econ["NPV"],
                "best_lift": preset_lift.iloc[0]["Method"],
                "deviation": preset["deviation"],
            })

    def normalize_portfolio_source(df: pd.DataFrame) -> list[dict]:
        if df.empty:
            return []
        expected_cols = {
            "well_name", "q_actual", "q_forecast", "wc_pct", "gor", "lifting_cost_boe",
            "natural_flow", "margin_pct", "decline_confidence", "npv", "best_lift", "deviation",
        }
        local_df = df.copy()
        for col in expected_cols:
            if col not in local_df.columns:
                local_df[col] = None
        local_df = local_df[list(expected_cols)]
        for col in ["q_actual", "q_forecast", "wc_pct", "gor", "lifting_cost_boe", "margin_pct", "npv", "deviation"]:
            local_df[col] = pd.to_numeric(local_df[col], errors="coerce")
        local_df["natural_flow"] = local_df["natural_flow"].astype("string").str.lower().isin(["true", "1", "yes", "y"])
        local_df["decline_confidence"] = local_df["decline_confidence"].fillna("moderate")
        local_df["best_lift"] = local_df["best_lift"].fillna("Unknown")
        local_df["well_name"] = local_df["well_name"].fillna("Uploaded well")
        return local_df.to_dict("records")

    portfolio_wells.extend(normalize_portfolio_source(manual_portfolio_df))
    portfolio_wells.extend(normalize_portfolio_source(uploaded_portfolio_df))

    deduped_wells = []
    seen_wells = set()
    for well in portfolio_wells:
        well_name = str(well.get("well_name", "Unknown"))
        if well_name in seen_wells:
            continue
        seen_wells.add(well_name)
        deduped_wells.append(well)
    portfolio_wells = deduped_wells

    portfolio_df = build_portfolio_table(portfolio_wells)
    summary = portfolio_summary_counts(portfolio_df)

    # Status counts
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total wells", summary["total_wells"])
    sc2.metric("Critical", summary["critical_wells"])
    sc3.metric("Warning", summary["warning_wells"])
    sc4.metric("Healthy", summary["healthy_wells"])

    # Ranked exception table
    st.subheader("Ranked exception table")
    st.dataframe(portfolio_df, width="stretch", hide_index=True)

    selected_well_name = st.selectbox("Inspect well detail", portfolio_df["Well"].tolist() if not portfolio_df.empty else ["No wells loaded"])
    selected_well = next((w for w in portfolio_wells if w.get("well_name") == selected_well_name), None)
    if selected_well:
        raw_detail_df = pd.DataFrame(
            [{"Field": k, "Value": v} for k, v in selected_well.items()]
        )
        raw_detail_df["Value"] = raw_detail_df["Value"].astype(str)
        st.markdown("<div class='section-label'>Selected Well Raw Data</div>", unsafe_allow_html=True)
        st.dataframe(raw_detail_df, width="stretch", hide_index=True)

    # Detailed flags for each well
    with st.expander("Detailed flags by well"):
        for well in portfolio_wells:
            flags = evaluate_well_flags(well)
            if flags:
                severity_color = {"critical": "#dc2626", "warning": "#ca8a04", "info": "#475569"}
                flag_html = "".join([
                    f'<span style="color:{severity_color.get(f["severity"], "#475569")};">&bull; [{f["severity"].upper()}] {f["flag"]}: {f["detail"]}</span><br>'
                    for f in flags
                ])
                st.markdown(f"**{well['well_name']}**<br>{flag_html}", unsafe_allow_html=True)
            else:
                st.markdown(f"**{well['well_name']}** — No flags.")

    # Download
    st.subheader("Export")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download portfolio table (CSV)",
            data=portfolio_df.to_csv(index=False),
            file_name="portfolio_surveillance.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl2:
        portfolio_json = []
        for well in portfolio_wells:
            flags = evaluate_well_flags(well)
            portfolio_json.append({
                "well_name": well["well_name"],
                "flags": flags,
                "data": {k: v for k, v in well.items() if k != "well_name"},
            })
        st.download_button(
            "Download surveillance data (JSON)",
            data=json.dumps(portfolio_json, indent=2, default=str),
            file_name="portfolio_surveillance.json",
            mime="application/json",
            width="stretch",
        )

    # Portfolio summary report
    st.markdown(
        f"""
<div class="report-box">
<strong>Portfolio summary:</strong> {summary['total_wells']} wells evaluated.
{summary['critical_wells']} with critical flags, {summary['warning_wells']} with warning-only flags,
{summary['healthy_wells']} within normal parameters.
Highest-urgency well: <strong>{portfolio_df.iloc[0]['Well']}</strong> (urgency score {portfolio_df.iloc[0]['Urgency Score']}).
</div>
""",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════
# TAB 8 — DECISION SUMMARY
# ═══════════════════════════════════════════════════════════
with tab_summary:
    st.header("Decision Summary")

    # Recompute econ_results in this scope
    econ_results_summary = {
        m: calculate_economics(
            m, rate_for_lift, depth_effective, years=10,
            oil_price=oil_price, gas_price=gas_price, gor=gor, wc=wc,
            discount_rate=discount_rate, decline_model=decline_model,
        )
        for m in (["ESP", "Rod Pump", "Gas Lift"] if gas_lift_available else ["ESP", "Rod Pump", "Plunger Lift"])
    }
    best_npv_method, best_npv_data = max(econ_results_summary.items(), key=lambda x: x[1]["NPV"])

    eur_hyp = 0
    if "Hyperbolic" in models:
        eur_hyp, _ = calc_eur(models["Hyperbolic"])
    pct_recovered = prod_df["Cum Oil (Mbbls)"].iloc[-1] / max(eur_hyp, 1) * 100 if eur_hyp else np.nan

    st.markdown(
        f"""
<div style="background: linear-gradient(135deg, #17342d 0%, #29483f 100%); color: white; padding: 1.4rem; border-radius: 14px; margin-bottom: 1rem;">
    <h2 style="color: #ffffff; margin-top: 0; border: none;">Well optimization recommendation</h2>
    <div style="color: #d7e7de;">Prepared for production engineering review | Date: {date.today().isoformat()}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Current rate", f"{rate_for_lift:,.0f} BPD")
    s2.metric("Water cut", f"{wc_pct}%")
    s3.metric("GOR", f"{gor:,} scf/bbl")
    s4.metric("Recommended lift", best_lift)
    s5.metric("Best NPV method", best_npv_method)

    summary_table = pd.DataFrame(
        [
            {
                "Method": m,
                "CAPEX": f"${e['CAPEX Total']:,.0f}",
                "Annual fixed OPEX": f"${e['Annual OPEX']:,.0f}",
                "NPV": f"${e['NPV']:,.0f}",
                "Payout": f"{e['Payout (years)']} yr" if isinstance(e["Payout (years)"], (int, float)) else e["Payout (years)"],
                "Lifting Cost": f"${e['Lifting Cost ($/BOE)']:.2f}/BOE",
            }
            for m, e in econ_results_summary.items()
        ]
    )
    st.table(summary_table.set_index("Method"))

    st.markdown(
        build_case_report_html(best_lift, best_npv_method, q_op, aof, wc_pct, gor, eur_hyp, actions, decline_confidence),
        unsafe_allow_html=True,
    )

    ds1, ds2 = st.columns(2)
    with ds1:
        st.subheader("Raw Well Data Snapshot")
        raw_summary_df = pd.DataFrame([
            {"Field": "Case", "Value": scenario_name},
            {"Field": "Reservoir Pressure (psi)", "Value": pr},
            {"Field": "Bubble Point (psi)", "Value": pb},
            {"Field": "PI (STB/d/psi)", "Value": pi},
            {"Field": "Oil API", "Value": api_gravity},
            {"Field": "TVD (ft)", "Value": depth},
            {"Field": "Deviation (deg)", "Value": deviation},
            {"Field": "Trajectory source", "Value": trajectory_source_label},
            {"Field": "Tubing OD (in)", "Value": tubing_od},
            {"Field": "Water Cut (%)", "Value": wc_pct},
            {"Field": "GOR (scf/bbl)", "Value": gor},
            {"Field": "WHP (psi)", "Value": whp},
            {"Field": "Viscosity (cp)", "Value": viscosity},
            {"Field": "Oil Price ($/bbl)", "Value": oil_price},
            {"Field": "Gas Price ($/Mscf)", "Value": gas_price},
            {"Field": "Discount Rate (%)", "Value": discount_pct},
            {"Field": "Gas Lift Available", "Value": "Yes" if gas_lift_available else "No"},
        ])
        raw_summary_df["Value"] = raw_summary_df["Value"].astype(str)
        st.dataframe(raw_summary_df, width="stretch", hide_index=True)
    with ds2:
        st.subheader("Calculated Decision Snapshot")
        calc_summary_df = pd.DataFrame([
            {"Field": "Current Rate (BPD)", "Value": round(rate_for_lift, 1)},
            {"Field": "AOF (BPD)", "Value": round(aof, 1)},
            {"Field": "Flowing BHP (psi)", "Value": None if pwf_op is None else round(pwf_op, 1)},
            {"Field": "Natural Flow", "Value": "Yes" if q_op is not None else "No"},
            {"Field": "Margin to AOF (%)", "Value": op_result.get("margin_pct", 0)},
            {"Field": "Recommended Lift", "Value": best_lift},
            {"Field": "Best NPV Method", "Value": best_npv_method},
            {"Field": "Decline Confidence", "Value": decline_confidence},
            {"Field": "Hyperbolic EUR (Mbbl)", "Value": eur_hyp if eur_hyp else None},
            {"Field": "Recovered / EUR (%)", "Value": round(pct_recovered, 1) if not np.isnan(pct_recovered) else None},
            {"Field": "Survey-based TVD (ft)", "Value": round(depth_effective, 1)},
            {"Field": "Survey-based deviation (deg)", "Value": round(deviation_effective, 1)},
            {"Field": "Max survey DLS (deg/100 ft)", "Value": None if survey_summary is None else round(survey_summary["max_dls"], 2)},
        ])
        calc_summary_df["Value"] = calc_summary_df["Value"].astype(str)
        st.dataframe(calc_summary_df, width="stretch", hide_index=True)

    manager_report = build_manager_report(
        scenario_name, q_op, aof, wc_pct, gor, best_lift, best_npv_method,
        eur_hyp, actions, econ_results_summary, decline_confidence,
    )
    engineer_report = build_engineer_report(
        current_inputs, q_op, pwf_op, aof, best_lift, models,
        eur_results, lift_df, econ_results_summary, preprocessing_info,
    )

    r1, r2 = st.columns(2)
    with r1:
        st.download_button(
            "Download manager report (.txt)",
            data=manager_report,
            file_name="production_engineering_manager_report.txt",
            mime="text/plain",
            width="stretch",
        )
    with r2:
        st.download_button(
            "Download engineer report (.txt)",
            data=engineer_report,
            file_name="production_engineering_engineer_report.txt",
            mime="text/plain",
            width="stretch",
        )

    export_payload = {
        "scenario": scenario_name,
        "inputs": {
            "pr": pr, "pb": pb, "pi": pi, "api": api_gravity,
            "depth": depth, "deviation": deviation, "tubing_od": tubing_od,
            "water_cut_pct": wc_pct, "gor": gor, "whp": whp, "viscosity": viscosity,
            "oil_price": oil_price, "gas_price": gas_price,
            "discount_rate_pct": discount_rate * 100, "gas_lift_available": gas_lift_available,
            "trajectory_source": trajectory_source_label,
        },
        "results": {
            "aof_bpd": round(aof, 1),
            "operating_rate_bpd": round(rate_for_lift, 1),
            "flowing_bhp_psi": None if pwf_op is None else round(pwf_op, 1),
            "recommended_lift": best_lift,
            "best_npv_method": best_npv_method,
            "eur_hyperbolic_mbbl": None if not eur_hyp else round(eur_hyp, 1),
            "decline_confidence": decline_confidence,
            "flow_stability": op_result.get("stability", "unknown"),
            "manager_flags": [msg for _, msg in actions],
            "survey_summary": survey_summary,
        },
    }
    st.download_button(
        "Download scenario summary (JSON)",
        data=json.dumps(export_payload, indent=2),
        file_name="production_engineering_scenario_summary.json",
        mime="application/json",
    )

st.markdown("---")
st.markdown(
    """
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding-bottom:1rem;">
Production Engineering Decision Tool | Streamlit portfolio build<br>
Focus: nodal analysis, lift selection, decline analysis, economics, and portfolio surveillance
</div>
""",
    unsafe_allow_html=True,
)
