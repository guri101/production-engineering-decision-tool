"""
Economics Engine
================
Monthly cash-flow model with separated CAPEX, fixed OPEX, variable OPEX,
water disposal, and replacement events. Supports scenario sensitivity,
do-nothing comparator, and manager-level summary metrics.

References:
- PetroAlchemy: monthly forecast-to-cashflow structure
- dcapy: decline-driven cash flow validation pattern

All outputs are deterministic and transparent.
"""

import numpy as np
import pandas as pd
from decline_engine import evaluate_decline_model


# ════════════════════════════════════════════════════════════
# 1. COST PARAMETERS
# ════════════════════════════════════════════════════════════

LIFT_COST_PARAMS = {
    "ESP": {
        "capex_equipment": 85000,
        "capex_installation": 180000,
        "capex_ancillary": 55000,       # VSD, transformer, etc.
        "capex_per_ft": 4.5,            # cable + tubing run cost
        "capex_other": 15000,
        "fixed_opex_annual": 12000,     # monitoring, maintenance baseline
        "variable_opex_per_bbl": 0.08,  # power cost per barrel fluid
        "avg_runlife_months": 22,
        "replacement_equipment": 85000,
        "replacement_workover": 120000,
    },
    "Rod Pump": {
        "capex_equipment": 45000,
        "capex_installation": 65000,
        "capex_ancillary": 0,
        "capex_per_ft": 0,
        "capex_other": 95000,           # pumping unit, foundation
        "fixed_opex_annual": 18000,
        "variable_opex_per_bbl": 0.08,
        "avg_runlife_months": 30,
        "replacement_equipment": 45000,
        "replacement_workover": 65000,
    },
    "Gas Lift": {
        "capex_equipment": 35000,
        "capex_installation": 120000,
        "capex_ancillary": 0,
        "capex_per_ft": 0,
        "capex_other": 130000,          # mandrels, compressor share
        "fixed_opex_annual": 8000,
        "variable_opex_per_bbl": 0.06,
        "avg_runlife_months": 48,
        "replacement_equipment": 35000,
        "replacement_workover": 50000,
    },
    "Plunger Lift": {
        "capex_equipment": 8000,
        "capex_installation": 12000,
        "capex_ancillary": 0,
        "capex_per_ft": 0,
        "capex_other": 15000,
        "fixed_opex_annual": 7000,
        "variable_opex_per_bbl": 0.0,
        "avg_runlife_months": 12,
        "replacement_equipment": 8000,
        "replacement_workover": 5000,
    },
    "Natural Flow": {
        "capex_equipment": 0,
        "capex_installation": 0,
        "capex_ancillary": 0,
        "capex_per_ft": 0,
        "capex_other": 0,
        "fixed_opex_annual": 3000,   # minimal surface maintenance
        "variable_opex_per_bbl": 0.02,
        "avg_runlife_months": 999,
        "replacement_equipment": 0,
        "replacement_workover": 0,
    },
}


# ════════════════════════════════════════════════════════════
# 2. MONTHLY CASH-FLOW MODEL
# ════════════════════════════════════════════════════════════

def calculate_economics(method: str, rate: float, depth: float,
                        years: int = 10, oil_price: float = 72.0,
                        gas_price: float = 3.0, gor: float = 1000,
                        wc: float = 0.5, discount_rate: float = 0.10,
                        decline_model: dict = None,
                        water_disposal_per_bbl: float = 0.75) -> dict:
    """
    Monthly cash-flow economics for a given lift method.

    Returns dict with all cost components separated for transparency.
    """
    params = LIFT_COST_PARAMS.get(method, LIFT_COST_PARAMS["Plunger Lift"])

    # ── CAPEX ──────────────────────────────────────────────
    capex = (params["capex_equipment"]
             + params["capex_installation"]
             + params["capex_ancillary"]
             + depth * params["capex_per_ft"]
             + params["capex_other"])

    # ── Production forecast ────────────────────────────────
    t_months = np.arange(1, years * 12 + 1)
    if decline_model is not None:
        q_forecast = evaluate_decline_model(decline_model, t_months)
        first_forecast = float(q_forecast[0]) if len(q_forecast) else 0.0
        if rate is not None and np.isfinite(rate) and rate > 0 and first_forecast > 0:
            # Anchor the forecast to the well's current rate instead of restarting
            # economics near the fitted initial rate.
            q_forecast = q_forecast * (float(rate) / first_forecast)
    else:
        q_forecast = rate * np.exp(-0.05 * (t_months - 1) / 12 * 0.6)
    q_forecast = np.maximum(q_forecast, 5.0)

    # ── Volumes ────────────────────────────────────────────
    monthly_oil_bbl = q_forecast * 30.4
    monthly_gas_mscf = monthly_oil_bbl * gor / 1000.0
    monthly_water_bbl = monthly_oil_bbl * wc / max(1.0 - wc, 0.02)

    # ── Revenue ────────────────────────────────────────────
    monthly_oil_revenue = monthly_oil_bbl * oil_price
    monthly_gas_revenue = monthly_gas_mscf * gas_price
    monthly_revenue = monthly_oil_revenue + monthly_gas_revenue

    # ── Operating costs ────────────────────────────────────
    monthly_water_disposal = monthly_water_bbl * water_disposal_per_bbl
    monthly_fixed_opex = params["fixed_opex_annual"] / 12.0
    monthly_fluid_bbl = monthly_oil_bbl + monthly_water_bbl
    monthly_variable_opex = monthly_fluid_bbl * params["variable_opex_per_bbl"]
    monthly_total_opex = monthly_water_disposal + monthly_fixed_opex + monthly_variable_opex

    # ── Cash flow ──────────────────────────────────────────
    monthly_cashflow = monthly_revenue - monthly_total_opex

    # Replacement events
    runlife = params["avg_runlife_months"]
    replacement_cost = params["replacement_equipment"] + params["replacement_workover"]
    replacement_months = set()
    if runlife < years * 12 and replacement_cost > 0:
        replacement_months = set(range(runlife, years * 12 + 1, runlife))
        for m in replacement_months:
            if m - 1 < len(monthly_cashflow):
                monthly_cashflow[m - 1] -= replacement_cost

    # ── Discounting ────────────────────────────────────────
    monthly_discount = np.array([1.0 / (1.0 + discount_rate / 12) ** m for m in t_months])
    discounted_cashflow = monthly_cashflow * monthly_discount

    # ── Payout ─────────────────────────────────────────────
    cum_cash = -capex
    payout_month = None
    for m, cf in enumerate(monthly_cashflow, start=1):
        cum_cash += cf
        if payout_month is None and cum_cash >= 0:
            payout_month = m

    # ── Aggregate metrics ──────────────────────────────────
    total_opex = float(monthly_total_opex.sum()) + len(replacement_months) * replacement_cost
    total_boe = float(monthly_oil_bbl.sum() + monthly_gas_mscf.sum() / 6.0)
    npv = float(discounted_cashflow.sum() - capex)

    # Capital efficiency = NPV / CAPEX
    cap_efficiency = npv / capex if capex > 0 else np.inf

    return {
        "CAPEX Total": capex,
        "CAPEX Equipment": params["capex_equipment"],
        "CAPEX Installation": params["capex_installation"],
        "CAPEX Ancillary": params["capex_ancillary"],
        "CAPEX Other": params["capex_other"],
        "Annual OPEX": params["fixed_opex_annual"],
        "Variable OPEX Rate ($/bbl fluid)": params["variable_opex_per_bbl"],
        "Avg Runlife (months)": runlife,
        "Replacement Events in Period": len(replacement_months),
        "Replacement Cost Each": replacement_cost,
        "Total Revenue (undiscounted)": float(monthly_revenue.sum()),
        "Total Water Disposal": float(monthly_water_disposal.sum()),
        "Total OPEX (incl replacements)": total_opex,
        "NPV": round(npv, 0),
        "Payout (months)": payout_month if payout_month is not None else ">120",
        "Payout (years)": round(payout_month / 12, 1) if payout_month is not None else ">10",
        "Lifting Cost ($/BOE)": round(total_opex / max(total_boe, 1), 2),
        "Capital Efficiency (NPV/CAPEX)": round(cap_efficiency, 2) if capex > 0 else "N/A",
        "Total BOE": round(total_boe, 0),
        "q_forecast": q_forecast,
        "t_months": t_months,
        "monthly_cashflow": monthly_cashflow,
        "monthly_revenue": monthly_revenue,
        "monthly_opex": monthly_total_opex,
        "discounted_cashflow": discounted_cashflow,
    }


# ════════════════════════════════════════════════════════════
# 3. SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════

def run_sensitivity_cases(base_inputs: dict, decline_model: dict,
                          find_op_fn=None, screen_fn=None) -> pd.DataFrame:
    """
    Run base + 4 sensitivity cases.
    find_op_fn and screen_fn are passed to avoid circular imports.
    """
    cases = [
        ("Base", dict(base_inputs)),
        ("Low oil price", {**base_inputs, "oil_price": max(40, base_inputs["oil_price"] - 15)}),
        ("High water cut", {**base_inputs, "wc_pct": min(95, base_inputs["wc_pct"] + 20)}),
        ("High GOR", {**base_inputs, "gor": min(3000, base_inputs["gor"] + 800)}),
        ("Higher decline", dict(base_inputs)),
    ]

    from unit_conversions import TUBING_ID_MAP

    results = []
    for name, case in cases:
        local_model = decline_model.copy() if decline_model else None
        if name == "Higher decline" and local_model is not None:
            local_model["di"] = float(local_model["di"]) * 1.25

        wc_local = case["wc_pct"] / 100.0
        d_t = TUBING_ID_MAP.get(case["tubing_od"], 2.441)

        if find_op_fn:
            op = find_op_fn(case["pr"], case["pb"], case["pi"], wc_local,
                            case["gor"], d_t, case["whp"], case["depth"], case["api"])
            q_local = op.get("q_op") if isinstance(op, dict) else op[0] if op else None
        else:
            q_local = None
        rate_local = q_local if q_local is not None else 500

        if screen_fn:
            lift_result = screen_fn(rate_local, case["depth"], case["gor"], wc_local,
                                    case["viscosity"], case["deviation"], case["pr"],
                                    case["gas_lift_available"])
            lift_local = lift_result[0] if isinstance(lift_result, tuple) else lift_result
        else:
            lift_local = None

        methods = (["ESP", "Rod Pump", "Gas Lift"] if case["gas_lift_available"]
                   else ["ESP", "Rod Pump", "Plunger Lift"])
        econ_local = {
            m: calculate_economics(
                m, rate_local, case["depth"],
                oil_price=case["oil_price"], gas_price=case["gas_price"],
                gor=case["gor"], wc=wc_local,
                discount_rate=case["discount_pct"] / 100, decline_model=local_model,
            )
            for m in methods
        }
        best_econ = max(econ_local.items(), key=lambda x: x[1]["NPV"])[0]

        results.append({
            "Scenario": name,
            "Operating Rate (BPD)": round(rate_local, 0),
            "Top Lift": lift_local.iloc[0]["Method"] if lift_local is not None else "N/A",
            "Top Lift Score": round(lift_local.iloc[0]["Weighted Score"], 2) if lift_local is not None else 0,
            "Best NPV Method": best_econ,
            "Best NPV ($)": round(econ_local[best_econ]["NPV"], 0),
        })

    return pd.DataFrame(results)


def tornado_sensitivity(base_econ: dict, method: str, rate: float, depth: float,
                        base_inputs: dict, decline_model: dict) -> pd.DataFrame:
    """
    Tornado-style sensitivity: vary one parameter at a time ±20%.
    Returns DataFrame with parameter, low/high NPV delta.
    """
    base_npv = base_econ["NPV"]
    params_to_vary = {
        "Oil Price": ("oil_price", 0.8, 1.2),
        "Water Cut": ("wc_pct", 0.8, 1.2),
        "GOR": ("gor", 0.8, 1.2),
        "Decline Rate": (None, 0.8, 1.2),  # special handling
        "CAPEX": (None, 0.8, 1.2),         # special handling
    }

    rows = []
    for label, (key, lo_mult, hi_mult) in params_to_vary.items():
        npvs = []
        for mult in [lo_mult, hi_mult]:
            inputs = dict(base_inputs)
            local_decline = decline_model.copy() if decline_model else None

            if key is not None:
                inputs[key] = inputs[key] * mult
            elif label == "Decline Rate" and local_decline:
                local_decline["di"] = float(local_decline["di"]) * mult
            elif label == "CAPEX":
                # handled in post
                pass

            wc_local = inputs.get("wc_pct", 50) / 100.0
            econ = calculate_economics(
                method, rate, depth,
                oil_price=inputs.get("oil_price", 72),
                gas_price=inputs.get("gas_price", 3.0),
                gor=inputs.get("gor", 1000),
                wc=wc_local,
                discount_rate=inputs.get("discount_pct", 10) / 100,
                decline_model=local_decline,
            )

            if label == "CAPEX":
                # Approximate: adjust NPV by CAPEX delta
                capex_delta = base_econ["CAPEX Total"] * (mult - 1.0)
                npvs.append(econ["NPV"] - capex_delta)
            else:
                npvs.append(econ["NPV"])

        rows.append({
            "Parameter": label,
            "Low Case NPV ($)": round(npvs[0], 0),
            "High Case NPV ($)": round(npvs[1], 0),
            "NPV Swing ($)": round(abs(npvs[1] - npvs[0]), 0),
        })

    df = pd.DataFrame(rows).sort_values("NPV Swing ($)", ascending=False).reset_index(drop=True)
    return df
