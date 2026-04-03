"""
Nodal Analysis / VLP Engine
============================
Separated IPR and TPR computation with robust operating-point solver,
natural-flow-margin assessment, and validation guards for impossible conditions.

References:
- pyResToolbox: IPR/VLP formula alignment, Vogel composite model structure
- Beggs-Brill simplified: multiphase gradient with liquid holdup
- All calculations are screening-grade, not design-grade

Cache note: all public functions return plain dicts/arrays, no closures.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ════════════════════════════════════════════════════════════
# 1. IPR — INFLOW PERFORMANCE RELATIONSHIP
# ════════════════════════════════════════════════════════════

def vogel_ipr(pr: float, pb: float, pi: float, pwf_range) -> np.ndarray:
    """
    Composite Vogel IPR:
    - Above Pb: Darcy linear (q = PI * (Pr - Pwf))
    - Below Pb: Vogel's empirical curve
    - Transition at Pb is handled to ensure continuity

    Aligned with pyResToolbox composite-IPR approach.
    """
    pwf_range = np.asarray(pwf_range, dtype=float)
    q = np.zeros_like(pwf_range)

    for i, pwf in enumerate(pwf_range):
        pwf = max(float(pwf), 0.0)
        if pr >= pb:
            # Undersaturated reservoir
            if pwf >= pb:
                q[i] = pi * (pr - pwf)
            else:
                q_b = pi * (pr - pb)
                q_max_below = pi * pb / 1.8
                q[i] = q_b + q_max_below * (1.0 - 0.2 * (pwf / pb) - 0.8 * (pwf / pb) ** 2)
        else:
            # Saturated reservoir
            q_max = pi * pr / 1.8
            q[i] = q_max * (1.0 - 0.2 * (pwf / pr) - 0.8 * (pwf / pr) ** 2)
        q[i] = max(q[i], 0.0)

    return q


def compute_aof(pr: float, pb: float, pi: float) -> float:
    """Absolute open flow potential (Pwf = 0)."""
    return float(vogel_ipr(pr, pb, pi, np.array([0.0]))[0])


def invert_ipr_to_pwf(pr: float, pb: float, pi: float, q_target: float) -> float:
    """
    Given a target rate, find the corresponding Pwf on the IPR curve.
    Uses sorted interpolation for robustness.
    """
    pwf_range = np.linspace(0, pr, 800)
    q_ipr = vogel_ipr(pr, pb, pi, pwf_range)
    sort_idx = np.argsort(q_ipr)
    q_sorted = q_ipr[sort_idx]
    pwf_sorted = pwf_range[sort_idx]
    q_target = float(q_target)
    if q_target < q_sorted.min() or q_target > q_sorted.max():
        return None
    f = interp1d(q_sorted, pwf_sorted, kind="linear", fill_value="extrapolate")
    return float(f(q_target))


# ════════════════════════════════════════════════════════════
# 2. TPR — TUBING PERFORMANCE RELATIONSHIP
# ════════════════════════════════════════════════════════════

def beggs_brill_gradient(q_o: float, wc: float, gor: float, d_t: float,
                          whp: float, depth: float, api: float = 38,
                          t_wh: float = 120, t_bh: float = 200):
    """
    Simplified Beggs-Brill multiphase gradient calculation.
    Returns (depths_array, pressures_array, bhp_at_bottom).

    Screening-grade: simplified holdup correlation, no flow-pattern map.
    """
    q_o = max(float(q_o), 1.0)
    wc = float(np.clip(wc, 0, 0.98))
    q_l = q_o / max(1.0 - wc, 0.02)

    gamma_o = 141.5 / (api + 131.5)
    gamma_w = 1.07
    gamma_g = 0.75
    rho_o = gamma_o * 62.4
    rho_w = gamma_w * 62.4
    rho_l = (1.0 - wc) * rho_o + wc * rho_w

    n_seg = 50
    dz = depth / n_seg
    p = whp
    pressures = [p]
    depths_arr = [0]

    for i in range(1, n_seg + 1):
        z = i * dz
        t = t_wh + (t_bh - t_wh) * (z / max(depth, 1))
        t_r = t + 460

        # Solution GOR (Standing)
        x = 0.0125 * api - 0.00091 * t
        rs = gamma_g * (max(p, 14.7) / 18.2 * 10 ** x) ** 1.204 if p > 14.7 else 0
        rs = min(rs, gor)

        # FVF
        bo = 0.972 + 0.000147 * (rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * t) ** 1.175
        bg = 0.0283 * t_r / max(p, 50)

        # Volumes
        free_gas = max(0, (gor - rs) * q_o)
        v_gas = free_gas * bg
        q_w = q_l * wc
        v_oil = q_o * bo
        v_water = q_w
        v_liq = (v_oil + v_water) * 5.615
        v_total = max(v_liq + v_gas, 1e-6)

        a = np.pi / 4 * (d_t / 12) ** 2
        v_sl = (v_liq / 86400) / max(a, 1e-6)
        v_sg = (v_gas / 86400) / max(a, 1e-6)
        v_m = v_sl + v_sg

        lambda_l = v_sl / v_m if v_m > 0 else 1.0
        if v_m > 0:
            n_fr = v_m ** 2 / (32.174 * (d_t / 12))
            hl = min(1.0, lambda_l * (1.0 + 0.3 * (1.0 - lambda_l) * min(n_fr, 5)))
        else:
            hl = 1.0

        rho_g = gamma_g * 0.0764 * p / 14.7 * 520 / t_r
        rho_m = hl * rho_l + (1 - hl) * rho_g
        rho_ns = lambda_l * rho_l + (1 - lambda_l) * max(0.01, rho_g)
        mu_l = 1.5
        re = 1488 * rho_ns * max(v_m, 0.01) * (d_t / 12) / mu_l
        f = 0.0056 + 0.5 * re ** (-0.32)

        dp_grav = rho_m / 144
        dp_fric = f * rho_ns * v_m ** 2 / (2 * 32.174 * (d_t / 12) * 144) if v_m > 0 else 0
        p += (dp_grav + dp_fric) * dz
        pressures.append(p)
        depths_arr.append(z)

    return np.array(depths_arr), np.array(pressures), float(p)


def compute_tpr(q_range, wc: float, gor: float, d_t: float,
                whp: float, depth: float, api: float = 38) -> np.ndarray:
    """Compute TPR (BHP vs rate) for an array of flow rates."""
    q_range = np.asarray(q_range, dtype=float)
    bhp_list = []
    for q in q_range:
        _, _, bhp = beggs_brill_gradient(max(q, 1.0), wc, gor, d_t, whp, depth, api)
        bhp_list.append(bhp)
    return np.array(bhp_list, dtype=float)


# ════════════════════════════════════════════════════════════
# 3. OPERATING POINT SOLVER
# ════════════════════════════════════════════════════════════

def find_operating_point(pr: float, pb: float, pi: float, wc: float,
                         gor: float, d_t: float, whp: float,
                         depth: float, api: float = 38) -> dict:
    """
    Find the natural-flow operating point (IPR–TPR intersection).

    Returns dict:
        - "q_op": float or None
        - "pwf_op": float or None
        - "natural_flow": bool
        - "margin_pct": float (% of AOF)
        - "stability": "stable" | "marginal" | "none"
        - "warnings": list of strings
    """
    warnings = []

    # Validate inputs
    if pr <= 0 or pi <= 0:
        return {"q_op": None, "pwf_op": None, "natural_flow": False,
                "margin_pct": 0, "stability": "none",
                "warnings": ["Invalid reservoir parameters."]}

    aof = compute_aof(pr, pb, pi)
    if aof <= 0:
        return {"q_op": None, "pwf_op": None, "natural_flow": False,
                "margin_pct": 0, "stability": "none",
                "warnings": ["AOF is zero or negative."]}

    def residual(q):
        if q <= 0:
            return 1e6
        pwf_ipr = invert_ipr_to_pwf(pr, pb, pi, q)
        if pwf_ipr is None:
            return -1e6
        _, _, bhp_tpr = beggs_brill_gradient(q, wc, gor, d_t, whp, depth, api)
        return pwf_ipr - bhp_tpr

    q_upper = max(50, aof * 1.15)
    q_test = np.linspace(10, q_upper, 250)
    residuals = np.array([residual(q) for q in q_test])

    q_op, pwf_op = None, None
    for i in range(len(residuals) - 1):
        if np.isnan(residuals[i]) or np.isnan(residuals[i + 1]):
            continue
        if residuals[i] * residuals[i + 1] < 0:
            try:
                q_op = float(brentq(residual, q_test[i], q_test[i + 1], xtol=1.0))
                pwf_op = invert_ipr_to_pwf(pr, pb, pi, q_op)
                break
            except Exception:
                continue

    if q_op is None:
        return {"q_op": None, "pwf_op": None, "natural_flow": False,
                "margin_pct": 0, "stability": "none",
                "warnings": ["No IPR/TPR intersection found at these conditions."]}

    margin_pct = round(q_op / aof * 100, 1)

    # Stability classification
    if margin_pct > 50:
        stability = "stable"
    elif margin_pct > 25:
        stability = "marginal"
        warnings.append("Natural-flow margin is narrow. Small pressure decline could push well to unstable flow.")
    else:
        stability = "marginal"
        warnings.append("Very narrow natural-flow margin. Well is near the edge of self-flowing.")

    # Check gas fraction warning
    if gor > 1500:
        warnings.append("High GOR may cause intermittent flow or heading in this tubing size.")
    if wc > 0.80:
        warnings.append("Very high water cut increases hydrostatic load; natural flow may deteriorate rapidly.")

    return {
        "q_op": round(q_op, 1),
        "pwf_op": round(pwf_op, 1) if pwf_op else None,
        "natural_flow": True,
        "margin_pct": margin_pct,
        "aof": round(aof, 1),
        "stability": stability,
        "warnings": warnings,
    }


# ════════════════════════════════════════════════════════════
# 4. ARTIFICIAL LIFT SCREENING
# ════════════════════════════════════════════════════════════

def screen_artificial_lift(rate: float, depth: float, gor: float, wc: float,
                           viscosity: float, deviation: float, pr: float,
                           gas_lift_available: bool = True) -> tuple:
    """
    Weighted scoring matrix for ESP, Rod Pump, Gas Lift, Plunger Lift.

    Returns (DataFrame, criteria_weights_dict, ranking_notes_list).
    ranking_notes explains why the top method ranks first.
    """
    import pandas as pd

    criteria = {
        "Rate Capability": 0.22,
        "Depth Capability": 0.14,
        "GOR Handling": 0.14,
        "Water Cut Handling": 0.10,
        "Viscosity Handling": 0.08,
        "Deviation Tolerance": 0.10,
        "Reliability / Runlife": 0.10,
        "Surface Infrastructure Fit": 0.07,
        "Operating Flexibility": 0.05,
    }

    scores = {}

    # ── ESP ─────────────────────────────────────────────────
    esp = {
        "Rate Capability": 5 if 200 < rate < 4000 else (4 if rate <= 200 else 3),
        "Depth Capability": 5 if depth < 10000 else (3 if depth < 14000 else 1),
        "GOR Handling": 2 if gor > 2000 else (3 if gor > 1200 else 5),
        "Water Cut Handling": 5,
        "Viscosity Handling": 3 if viscosity < 200 else 1,
        "Deviation Tolerance": 4 if deviation < 65 else (3 if deviation < 80 else 2),
        "Reliability / Runlife": 3,
        "Surface Infrastructure Fit": 4,
        "Operating Flexibility": 4,
    }
    scores["ESP"] = esp

    # ── Rod Pump ───────────────────────────────────────────
    rp = {
        "Rate Capability": 4 if rate < 500 else (3 if rate < 1000 else 1),
        "Depth Capability": 4 if depth < 10000 else (2 if depth < 13000 else 1),
        "GOR Handling": 3 if gor < 1200 else 2,
        "Water Cut Handling": 4,
        "Viscosity Handling": 4 if viscosity < 500 else 2,
        "Deviation Tolerance": 4 if deviation < 40 else (2 if deviation < 60 else 1),
        "Reliability / Runlife": 4,
        "Surface Infrastructure Fit": 4,
        "Operating Flexibility": 3,
    }
    scores["Rod Pump"] = rp

    # ── Gas Lift ───────────────────────────────────────────
    gl = {
        "Rate Capability": 5 if rate > 500 else 3,
        "Depth Capability": 5,
        "GOR Handling": 5,
        "Water Cut Handling": 4,
        "Viscosity Handling": 5,
        "Deviation Tolerance": 5,
        "Reliability / Runlife": 5,
        "Surface Infrastructure Fit": 5 if gas_lift_available else 1,
        "Operating Flexibility": 5,
    }
    scores["Gas Lift"] = gl

    # ── Plunger Lift ───────────────────────────────────────
    pl = {
        "Rate Capability": 2 if rate > 200 else 4,
        "Depth Capability": 3 if depth > 10000 else 4,
        "GOR Handling": 5 if gor > 1000 else 2,
        "Water Cut Handling": 2 if wc > 0.5 else 3,
        "Viscosity Handling": 3,
        "Deviation Tolerance": 3 if deviation < 25 else 1,
        "Reliability / Runlife": 4,
        "Surface Infrastructure Fit": 4,
        "Operating Flexibility": 3,
    }
    scores["Plunger Lift"] = pl

    # ── Build table ────────────────────────────────────────
    rows = []
    for method in ["ESP", "Rod Pump", "Gas Lift", "Plunger Lift"]:
        weighted_total = 0.0
        row = {"Method": method}
        for crit, weight in criteria.items():
            s = scores[method][crit]
            row[crit] = s
            weighted_total += s * weight
        row["Weighted Score"] = round(weighted_total, 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Weighted Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1

    # ── Build "why this ranks first" notes ─────────────────
    top = df.iloc[0]
    top_method = top["Method"]
    ranking_notes = [f"{top_method} ranks first with weighted score {top['Weighted Score']:.2f}."]

    # Identify strongest criteria
    best_criteria = []
    for crit in criteria:
        if top[crit] >= 5:
            best_criteria.append(crit.lower())
    if best_criteria:
        ranking_notes.append(f"Strongest fit areas: {', '.join(best_criteria[:3])}.")

    # Identify weakest criteria
    weak_criteria = []
    for crit in criteria:
        if top[crit] <= 2:
            weak_criteria.append(crit.lower())
    if weak_criteria:
        ranking_notes.append(f"Weakest areas: {', '.join(weak_criteria[:3])}.")

    # Constraint-specific notes
    if top_method == "ESP" and gor > 1500:
        ranking_notes.append("ESP selection at this GOR requires gas handling (separator + AGH).")
    if top_method == "ESP" and deviation > 70:
        ranking_notes.append("High deviation narrows ESP setting-depth options.")
    if top_method == "Gas Lift" and not gas_lift_available:
        ranking_notes.append("Gas lift scored highest on merit but infrastructure is unavailable.")
    if top_method == "Rod Pump" and deviation > 55:
        ranking_notes.append("Rod pump may have reliability issues at this deviation.")

    return df, criteria, ranking_notes


# ════════════════════════════════════════════════════════════
# 5. ESP SIZING
# ════════════════════════════════════════════════════════════

def size_esp(target_rate: float, tdh: float, fluid_sg: float = 1.02,
             frequency: float = 60, gor: float = 1000) -> tuple:
    """
    ESP sizing from representative pump catalog.
    Returns (results_dict, pump_info_dict).
    """
    pump_catalog = [
        {"series": "DN400",  "min_q": 100,  "max_q": 500,  "bep_q": 300,  "head_per_stage": 58, "hp_per_stage": 0.45, "od": 3.38},
        {"series": "DN1100", "min_q": 400,  "max_q": 1400, "bep_q": 900,  "head_per_stage": 52, "hp_per_stage": 0.85, "od": 3.38},
        {"series": "DN1750", "min_q": 800,  "max_q": 2200, "bep_q": 1500, "head_per_stage": 48, "hp_per_stage": 1.25, "od": 4.00},
        {"series": "DN2800", "min_q": 1500, "max_q": 3500, "bep_q": 2500, "head_per_stage": 42, "hp_per_stage": 1.80, "od": 5.13},
        {"series": "DN5000", "min_q": 3000, "max_q": 6000, "bep_q": 4500, "head_per_stage": 38, "hp_per_stage": 2.80, "od": 5.13},
    ]

    selected = next((p for p in pump_catalog if p["min_q"] <= target_rate <= p["max_q"]), None)
    if selected is None:
        selected = pump_catalog[-1] if target_rate > 3000 else pump_catalog[0]

    freq_ratio = frequency / 60
    adj_head = selected["head_per_stage"] * freq_ratio ** 2
    adj_rate = selected["bep_q"] * freq_ratio

    n_stages = int(np.ceil(tdh / max(adj_head, 1)))
    n_stages = max(10, min(n_stages, 500))

    bhp = n_stages * selected["hp_per_stage"] * fluid_sg * freq_ratio ** 3
    bhp *= 1.15  # 15% safety factor

    motor_sizes = [30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750]
    motor_hp = next((m for m in motor_sizes if m >= bhp), motor_sizes[-1])
    motor_voltage = 1000 if motor_hp < 150 else (1500 if motor_hp < 300 else 2400)
    motor_amps = motor_hp * 746 / (motor_voltage * 0.85 * 1.732)

    if motor_amps < 40:
        cable = "#4 AWG"
    elif motor_amps < 60:
        cable = "#2 AWG"
    elif motor_amps < 90:
        cable = "#1 AWG"
    else:
        cable = "#1/0 AWG"

    gas_handling = (
        "Vortex gas separator + AGH (recommended)"
        if gor >= 1000
        else "Standard intake acceptable; separator optional"
    )

    results = {
        "Pump Series": selected["series"],
        "Pump OD (in)": selected["od"],
        "Operating Range (BPD)": f"{selected['min_q']} – {selected['max_q']}",
        "BEP Rate (BPD)": int(adj_rate),
        "Head per Stage (ft)": round(adj_head, 1),
        "Number of Stages": n_stages,
        "Total Head (ft)": round(n_stages * adj_head),
        "Required BHP": round(bhp, 1),
        "Motor HP (nameplate)": motor_hp,
        "Motor Voltage (V)": motor_voltage,
        "Motor Amps": round(motor_amps, 1),
        "Cable Size": cable,
        "Cable Type": "KEOTB armored power cable",
        "Gas Handling": gas_handling,
        "VSD Frequency (Hz)": frequency,
        "Pump Setting Depth (ft)": "Set above KOP; confirm with trajectory and dogleg survey",
    }
    return results, selected
