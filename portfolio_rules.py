"""
Portfolio Surveillance Rules Engine
====================================
Lightweight, rule-based surveillance layer for production-engineering
manager workflows. Works with manually uploaded or synthetic data.

Patterns inspired by Snowflake Intelligent Production Assistant repo:
- Exception-based alerting
- Manager dashboard logic
- Ranked well-priority tables

No Snowflake infrastructure. Pure Python / Pandas rule evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


# ════════════════════════════════════════════════════════════
# 1. EXCEPTION FLAGS — INDIVIDUAL WELL
# ════════════════════════════════════════════════════════════

def evaluate_well_flags(well: dict) -> List[dict]:
    """
    Evaluate a single well's data against surveillance rules.

    Input well dict should contain keys like:
        q_actual, q_forecast, wc_pct, gor, lifting_cost_boe,
        natural_flow, margin_pct, decline_confidence,
        esp_gas_risk, best_lift, npv, capex, deviation, ...

    Returns list of flag dicts: [{"flag": str, "severity": str, "detail": str}, ...]
    Severity: "critical" | "warning" | "info"
    """
    flags = []

    # ── Below forecast ─────────────────────────────────────
    q_act = well.get("q_actual")
    q_fcast = well.get("q_forecast")
    if q_act is not None and q_fcast is not None and q_fcast > 0:
        variance_pct = (q_act - q_fcast) / q_fcast * 100
        if variance_pct < -20:
            flags.append({
                "flag": "Significantly below forecast",
                "severity": "critical",
                "detail": f"Actual {q_act:.0f} vs forecast {q_fcast:.0f} ({variance_pct:.1f}%)",
            })
        elif variance_pct < -10:
            flags.append({
                "flag": "Below forecast",
                "severity": "warning",
                "detail": f"Actual {q_act:.0f} vs forecast {q_fcast:.0f} ({variance_pct:.1f}%)",
            })

    # ── High lifting cost ──────────────────────────────────
    lc = well.get("lifting_cost_boe")
    if lc is not None and lc > 20:
        flags.append({
            "flag": "High lifting cost",
            "severity": "critical" if lc > 35 else "warning",
            "detail": f"${lc:.2f}/BOE",
        })

    # ── No natural flow ────────────────────────────────────
    if well.get("natural_flow") is False:
        flags.append({
            "flag": "No viable natural flow",
            "severity": "critical",
            "detail": "No IPR/TPR intersection at current conditions.",
        })

    # ── Narrow margin ──────────────────────────────────────
    margin = well.get("margin_pct", 100)
    if margin is not None and 0 < margin < 30:
        flags.append({
            "flag": "Narrow natural-flow margin",
            "severity": "warning",
            "detail": f"Operating at {margin:.1f}% of AOF.",
        })

    # ── High water cut ─────────────────────────────────────
    wc = well.get("wc_pct", 0)
    if wc > 80:
        flags.append({
            "flag": "High water cut",
            "severity": "critical" if wc > 90 else "warning",
            "detail": f"Water cut is {wc}%.",
        })

    # ── High GOR / ESP gas risk ────────────────────────────
    gor = well.get("gor", 0)
    if gor > 2000:
        flags.append({
            "flag": "High GOR — gas risk for ESP",
            "severity": "warning",
            "detail": f"GOR = {gor:,.0f} scf/bbl. ESP requires enhanced gas handling.",
        })
    elif gor > 1500:
        flags.append({
            "flag": "Elevated GOR",
            "severity": "info",
            "detail": f"GOR = {gor:,.0f} scf/bbl. Monitor for gas interference.",
        })

    # ── Low decline confidence ─────────────────────────────
    conf = well.get("decline_confidence", "high")
    if conf == "low":
        flags.append({
            "flag": "Low decline-fit confidence",
            "severity": "warning",
            "detail": "Decline model fit quality is low. Forecast unreliable for investment decisions.",
        })

    # ── Weak economics ─────────────────────────────────────
    npv = well.get("npv")
    if npv is not None and npv < 0:
        flags.append({
            "flag": "Negative NPV",
            "severity": "critical",
            "detail": f"NPV = ${npv:,.0f}. Current lift strategy may be uneconomic.",
        })

    # ── Missing critical data ──────────────────────────────
    required_keys = ["q_actual", "wc_pct", "gor", "lifting_cost_boe"]
    missing = [k for k in required_keys if well.get(k) is None]
    if missing:
        flags.append({
            "flag": "Missing critical input data",
            "severity": "info",
            "detail": f"Missing: {', '.join(missing)}.",
        })

    # ── Lift transition signal ─────────────────────────────
    if well.get("natural_flow") is True and margin is not None and margin < 35:
        if wc > 50 or gor > 1200:
            flags.append({
                "flag": "Likely nearing lift transition",
                "severity": "warning",
                "detail": "Narrow margin + rising fluid challenges suggest lift installation planning.",
            })

    return flags


# ════════════════════════════════════════════════════════════
# 2. PORTFOLIO SCORING
# ════════════════════════════════════════════════════════════

def score_well_urgency(flags: List[dict]) -> float:
    """
    Compute intervention urgency score (0–100) from flags.
    Higher = more urgent.
    """
    score = 0
    for f in flags:
        if f["severity"] == "critical":
            score += 25
        elif f["severity"] == "warning":
            score += 10
        else:
            score += 3
    return min(score, 100)


def build_portfolio_table(wells: List[dict]) -> pd.DataFrame:
    """
    Build a ranked portfolio surveillance table from a list of well dicts.

    Each well dict should have at minimum:
        "well_name", and any keys consumed by evaluate_well_flags.

    Returns DataFrame sorted by urgency score descending.
    """
    rows = []
    for well in wells:
        flags = evaluate_well_flags(well)
        urgency = score_well_urgency(flags)
        n_critical = sum(1 for f in flags if f["severity"] == "critical")
        n_warning = sum(1 for f in flags if f["severity"] == "warning")
        flag_summary = "; ".join([f["flag"] for f in flags[:3]])
        if len(flags) > 3:
            flag_summary += f" (+{len(flags) - 3} more)"

        rows.append({
            "Well": well.get("well_name", "Unknown"),
            "Rate (BOPD)": well.get("q_actual"),
            "Water Cut (%)": well.get("wc_pct"),
            "GOR (scf/bbl)": well.get("gor"),
            "Lifting Cost ($/BOE)": well.get("lifting_cost_boe"),
            "NPV ($)": well.get("npv"),
            "Critical Flags": n_critical,
            "Warning Flags": n_warning,
            "Urgency Score": urgency,
            "Top Flags": flag_summary,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Urgency Score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
    return df


def portfolio_summary_counts(portfolio_df: pd.DataFrame) -> dict:
    """
    Summary statistics for the portfolio table.
    """
    if portfolio_df.empty:
        return {"total_wells": 0, "critical_wells": 0, "warning_wells": 0, "healthy_wells": 0}

    n_total = len(portfolio_df)
    n_critical = int((portfolio_df["Critical Flags"] > 0).sum())
    n_warning_only = int(((portfolio_df["Warning Flags"] > 0) & (portfolio_df["Critical Flags"] == 0)).sum())
    n_healthy = n_total - n_critical - n_warning_only

    return {
        "total_wells": n_total,
        "critical_wells": n_critical,
        "warning_wells": n_warning_only,
        "healthy_wells": max(0, n_healthy),
    }


# ════════════════════════════════════════════════════════════
# 3. SURVEILLANCE FORECAST-VS-ACTUAL
# ════════════════════════════════════════════════════════════

def build_forecast_actual_table(prod_df, models: dict, n_months: int = 6) -> pd.DataFrame:
    """
    Build editable forecast-vs-actual table for surveillance.
    """
    default_model = models.get("Hyperbolic") or models.get("Exponential") or next(iter(models.values()), None)
    from decline_engine import evaluate_decline_model

    forecast_months = prod_df["Month"].tail(n_months).tolist()
    if default_model:
        forecast_values = evaluate_decline_model(default_model, forecast_months)
    else:
        forecast_values = prod_df["Oil Rate (BOPD)"].tail(n_months).to_numpy()

    seed = pd.DataFrame({
        "Month": forecast_months,
        "Forecast Oil Rate (BOPD)": np.round(forecast_values, 0),
        "Actual Oil Rate (BOPD)": prod_df["Oil Rate (BOPD)"].tail(n_months).tolist(),
    })
    return seed


def compute_surveillance_flags(actual_editor: pd.DataFrame, gor: float, wc_pct: float) -> List[str]:
    """
    Compute surveillance text flags from forecast-vs-actual data.
    """
    flags = []
    if actual_editor.empty:
        return ["No surveillance data available."]

    avg_var = float(actual_editor["Variance (%)"].mean())
    if avg_var < -10:
        flags.append("Recent actual rates are running materially below forecast.")
    if (actual_editor["Variance (%)"] < -15).any():
        flags.append("At least one recent month underperformed forecast by more than 15%.")
    if gor > 1500:
        flags.append("Entered GOR remains high enough to justify tighter gas-handling surveillance on any ESP case.")
    if wc_pct > 70:
        flags.append("Water cut is high enough that water-handling cost should be reviewed against late-life lift strategy.")
    if not flags:
        flags.append("No additional surveillance flags beyond the base trigger rules.")
    return flags
