"""
Decline Curve Analysis Engine
=============================
Deterministic Arps decline models with production-history preprocessing,
fit-quality metrics, model auto-selection, and confidence scoring.

Concepts referenced:
- Python_Automated_DCA: change-point detection / post-change filtering
- dcapy: EUR cross-check from monthly summation vs. closed-form
- pyResToolbox: Arps formula alignment and parameter bounds

All calculations are cache-safe (no lambdas, no closures in returned dicts).
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional


# ════════════════════════════════════════════════════════════
# 1. ARPS DECLINE FUNCTIONS
# ════════════════════════════════════════════════════════════

def arps_exponential(t, qi, di):
    """q(t) = qi * exp(-di * t)"""
    return qi * np.exp(-di * np.asarray(t, dtype=float))


def arps_hyperbolic(t, qi, di, b):
    """q(t) = qi / (1 + b*di*t)^(1/b)"""
    t = np.asarray(t, dtype=float)
    return qi / (1.0 + b * di * t) ** (1.0 / b)


def arps_harmonic(t, qi, di):
    """q(t) = qi / (1 + di*t)  [b = 1 special case]"""
    return qi / (1.0 + di * np.asarray(t, dtype=float))


def evaluate_decline_model(model: dict, t) -> np.ndarray:
    """Evaluate a model dict at times t. Cache-safe: no closures."""
    t = np.asarray(t, dtype=float)
    name = model.get("model", "")
    qi = float(model["qi"])
    di = float(model["di"])
    if name == "Exponential":
        return arps_exponential(t, qi, di)
    elif name == "Hyperbolic":
        return arps_hyperbolic(t, qi, di, float(model.get("b", 1.0)))
    elif name == "Harmonic":
        return arps_harmonic(t, qi, di)
    raise ValueError(f"Unknown decline model: {name}")


# ════════════════════════════════════════════════════════════
# 2. PRODUCTION HISTORY PREPROCESSING
# ════════════════════════════════════════════════════════════
# Pattern adapted from Python_Automated_DCA concepts:
# - detect and flag operational shut-ins (very low rates)
# - detect restart periods after shut-in
# - optionally filter to post-change-point data
# - remove zero-rate tails

def preprocess_production(df, rate_col="Oil Rate (BOPD)", month_col="Month",
                          shutin_threshold_frac=0.15, min_month=4,
                          remove_zero_tail=True):
    """
    Clean production history before fitting.

    Returns:
        dict with keys:
            - "t_fit": array of months to fit
            - "q_fit": array of rates to fit
            - "flags": list of string flags describing preprocessing
            - "mask": boolean mask applied to original df
            - "shutin_months": list of months flagged as shut-in
            - "quality_score": float 0–1 indicating data quality
    """
    df = df.copy()
    flags = []
    n_total = len(df)

    # Basic quality
    if n_total < 6:
        flags.append("Very short production history (<6 months). Fit reliability is low.")

    t = df[month_col].values.astype(float)
    q = df[rate_col].values.astype(float)

    # ── Detect shut-ins ────────────────────────────────────
    median_rate = np.median(q[q > 0]) if np.any(q > 0) else 1.0
    shutin_mask = q < median_rate * shutin_threshold_frac
    shutin_months = t[shutin_mask].tolist()
    if len(shutin_months) > 0:
        flags.append(f"Flagged {len(shutin_months)} shut-in/low-rate months (below {shutin_threshold_frac*100:.0f}% of median).")

    # ── Remove zero-rate tail ──────────────────────────────
    if remove_zero_tail:
        last_nonzero = np.max(np.where(q > 5)[0]) if np.any(q > 5) else len(q) - 1
        if last_nonzero < len(q) - 1:
            tail_len = len(q) - 1 - last_nonzero
            flags.append(f"Trimmed {tail_len} trailing near-zero months.")
            # Only keep up to last_nonzero + 1
            t = t[:last_nonzero + 1]
            q = q[:last_nonzero + 1]
            shutin_mask = shutin_mask[:last_nonzero + 1]

    # ── Apply min_month filter ─────────────────────────────
    month_mask = t >= min_month
    fit_mask = month_mask & ~shutin_mask

    # ── Outlier filter: remove points far from local trend ─
    # Simple: remove points below 40% of median of surviving points
    q_surviving = q[fit_mask]
    if len(q_surviving) > 3:
        local_median = np.median(q_surviving)
        outlier_mask = q < local_median * 0.4
        n_outliers = int(np.sum(fit_mask & outlier_mask))
        if n_outliers > 0:
            flags.append(f"Removed {n_outliers} low outliers (below 40% of median).")
            fit_mask = fit_mask & ~outlier_mask

    t_fit = t[fit_mask]
    q_fit = q[fit_mask]

    # ── Quality score ──────────────────────────────────────
    # Based on: data length, shutin fraction, rate consistency
    quality = 1.0
    if n_total < 12:
        quality -= 0.3
    elif n_total < 24:
        quality -= 0.1
    shutin_frac = len(shutin_months) / max(n_total, 1)
    quality -= min(shutin_frac * 0.5, 0.3)
    if len(q_fit) < 6:
        quality -= 0.3
    quality = max(0.0, min(1.0, quality))

    return {
        "t_fit": t_fit,
        "q_fit": q_fit,
        "flags": flags,
        "shutin_months": shutin_months,
        "quality_score": round(quality, 2),
        "n_points_used": len(t_fit),
        "n_points_total": n_total,
    }


# ════════════════════════════════════════════════════════════
# 3. FIT QUALITY METRICS
# ════════════════════════════════════════════════════════════

def fit_metrics(q_actual: np.ndarray, q_predicted: np.ndarray) -> dict:
    """
    Compute RMSE, MAPE, R² for a decline fit.
    """
    q_a = np.asarray(q_actual, dtype=float)
    q_p = np.asarray(q_predicted, dtype=float)
    n = len(q_a)
    if n == 0:
        return {"rmse": np.nan, "mape": np.nan, "r2": np.nan}

    residuals = q_a - q_p
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((q_a - np.mean(q_a)) ** 2)

    rmse = float(np.sqrt(ss_res / n))

    # MAPE: only where actual > 5 to avoid division blowup
    valid = q_a > 5.0
    if np.sum(valid) > 0:
        mape = float(np.mean(np.abs(residuals[valid] / q_a[valid])) * 100)
    else:
        mape = np.nan

    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"rmse": round(rmse, 2), "mape": round(mape, 2) if not np.isnan(mape) else None, "r2": round(r2, 4)}


# ════════════════════════════════════════════════════════════
# 4. MODEL FITTING
# ════════════════════════════════════════════════════════════

def fit_decline_curves(df, rate_col="Oil Rate (BOPD)", month_col="Month",
                       min_month=4, auto_preprocess=True):
    """
    Fit exponential, hyperbolic, and harmonic decline to production data.

    Returns:
        dict of model_name -> {model dict + metrics + fit metadata}
        preprocessing_info: dict from preprocess_production
    """
    if auto_preprocess:
        pp = preprocess_production(df, rate_col=rate_col, month_col=month_col, min_month=min_month)
        t_fit = pp["t_fit"]
        q_fit = pp["q_fit"]
    else:
        mask = df[month_col] >= min_month
        median_rate = df.loc[mask, rate_col].median()
        mask &= df[rate_col] > median_rate * 0.4
        t_fit = df.loc[mask, month_col].values.astype(float)
        q_fit = df.loc[mask, rate_col].values.astype(float)
        pp = {
            "flags": [],
            "quality_score": None,
            "n_points_used": len(t_fit),
            "n_points_total": len(df),
            "fit_errors": [],
        }

    pp.setdefault("fit_errors", [])

    if len(t_fit) < 3:
        return {}, pp

    results = {}

    def record_fit_error(model_name: str, exc: Exception):
        pp["fit_errors"].append(f"{model_name} fit failed: {type(exc).__name__}: {exc}")

    # ── Exponential ────────────────────────────────────────
    try:
        popt, _ = curve_fit(
            arps_exponential, t_fit, q_fit,
            p0=[max(q_fit), 0.05],
            maxfev=10000,
            bounds=([50, 0.001], [10000, 0.5]),
        )
        model = {"model": "Exponential", "qi": float(popt[0]), "di": float(popt[1]), "b": 0.0}
        q_pred = arps_exponential(t_fit, *popt)
        metrics = fit_metrics(q_fit, q_pred)
        model.update(metrics)
        results["Exponential"] = model
    except Exception as exc:
        record_fit_error("Exponential", exc)

    # ── Hyperbolic ─────────────────────────────────────────
    try:
        popt, _ = curve_fit(
            arps_hyperbolic, t_fit, q_fit,
            p0=[max(q_fit), 0.06, 1.0],
            maxfev=10000,
            bounds=([50, 0.001, 0.01], [10000, 0.5, 2.0]),
        )
        model = {"model": "Hyperbolic", "qi": float(popt[0]), "di": float(popt[1]), "b": float(popt[2])}
        q_pred = arps_hyperbolic(t_fit, *popt)
        metrics = fit_metrics(q_fit, q_pred)
        model.update(metrics)
        results["Hyperbolic"] = model
    except Exception as exc:
        record_fit_error("Hyperbolic", exc)

    # ── Harmonic ───────────────────────────────────────────
    try:
        popt, _ = curve_fit(
            arps_harmonic, t_fit, q_fit,
            p0=[max(q_fit), 0.05],
            maxfev=10000,
            bounds=([50, 0.001], [10000, 0.5]),
        )
        model = {"model": "Harmonic", "qi": float(popt[0]), "di": float(popt[1]), "b": 1.0}
        q_pred = arps_harmonic(t_fit, *popt)
        metrics = fit_metrics(q_fit, q_pred)
        model.update(metrics)
        results["Harmonic"] = model
    except Exception as exc:
        record_fit_error("Harmonic", exc)

    return results, pp


def select_best_model(models: dict) -> tuple:
    """
    Auto-select the best model based on R² and physical reasonableness.

    Returns:
        (model_name, reason_string)

    Selection logic:
    1. Highest R² wins, but hyperbolic with b > 1.5 is penalised
    2. If R² values are within 0.01, prefer hyperbolic (most physically reasonable for unconventionals)
    3. Models with R² < 0.5 are flagged as unreliable
    """
    if not models:
        return None, "No models fitted."

    scored = []
    for name, m in models.items():
        r2 = m.get("r2", 0) or 0
        # Penalize hyperbolic with very high b
        penalty = 0.0
        if name == "Hyperbolic" and m.get("b", 0) > 1.5:
            penalty = 0.05
        adj_r2 = r2 - penalty
        scored.append((name, adj_r2, r2))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_name, best_adj_r2, best_r2 = scored[0]

    # If top two are within 0.01 R², prefer hyperbolic
    if len(scored) > 1:
        second = scored[1]
        if abs(best_adj_r2 - second[1]) < 0.01:
            for s in scored:
                if s[0] == "Hyperbolic":
                    best_name = "Hyperbolic"
                    best_r2 = s[2]
                    break

    # Build reason
    m = models[best_name]
    reason_parts = [f"Highest adjusted R² ({best_r2:.4f})"]
    if best_name == "Hyperbolic":
        reason_parts.append(f"b = {m.get('b', 0):.3f}")
        if m.get("b", 0) > 1.3:
            reason_parts.append("b > 1.3 suggests aggressive late-life optimism — use with caution")
    if best_r2 < 0.7:
        reason_parts.append("Overall fit quality is moderate — forecast uncertainty is elevated")
    if best_r2 < 0.5:
        reason_parts.append("Fit quality is poor — forecast should not be trusted for investment decisions")

    reason = ". ".join(reason_parts) + "."
    return best_name, reason


# ════════════════════════════════════════════════════════════
# 5. EUR CALCULATION
# ════════════════════════════════════════════════════════════

def calc_eur(model: dict, t_start: int = 1, t_end: int = 360,
             econ_limit: float = 5.0) -> tuple:
    """
    EUR from monthly summation of forecast above economic limit.
    Aligned with dcapy pattern of using discrete sum rather than
    closed-form shortcut (more transparent, handles truncation).

    Returns:
        (eur_mbbl, economic_life_months)
    """
    t = np.arange(t_start, t_end + 1)
    q = evaluate_decline_model(model, t)
    above = q >= econ_limit
    if not np.any(above):
        return 0.0, 0
    t_econ = t[above]
    q_econ = q[above]
    eur = float(np.sum(q_econ * 30.4) / 1000.0)
    return round(eur, 1), int(t_econ[-1])


def generate_monthly_forecast(model: dict, months: int = 120,
                              econ_limit: float = 5.0) -> dict:
    """
    Generate a monthly production forecast from a decline model.
    Returns dict with arrays: t_months, q_oil, cum_oil_mbbl, econ_life_month.
    """
    t = np.arange(1, months + 1)
    q = evaluate_decline_model(model, t)
    q = np.maximum(q, 0.0)
    above = q >= econ_limit
    if np.any(above):
        econ_life = int(t[above][-1])
    else:
        econ_life = 0
    cum = np.cumsum(q * 30.4) / 1000.0
    return {
        "t_months": t,
        "q_oil": q,
        "cum_oil_mbbl": cum,
        "econ_life_month": econ_life,
    }


# ════════════════════════════════════════════════════════════
# 6. SYNTHETIC DATA GENERATOR (for demo mode)
# ════════════════════════════════════════════════════════════

def generate_synthetic_production(months: int = 60, seed: int = 42):
    """
    Generate synthetic production data for demonstration.
    Includes noise, downtime events, rising water cut, rising GOR.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    t = np.arange(1, months + 1)
    qi, di, b = 1500, 0.08, 1.2
    q_clean = qi / (1 + b * di * t) ** (1.0 / b)
    noise = rng.normal(0, 0.05, len(t))
    q_noisy = np.maximum(q_clean * (1 + noise), 10)

    for m in [14, 15, 36]:
        if m <= len(q_noisy):
            q_noisy[m - 1] *= 0.3

    wc = np.minimum(0.25 + 0.55 * (1 - np.exp(-0.04 * t)), 0.92)
    gor = 800 + 150 * np.log(1 + t / 6.0)

    df = pd.DataFrame({
        "Month": t,
        "Oil Rate (BOPD)": np.round(q_noisy, 0),
        "Water Cut": np.round(wc, 3),
        "GOR (scf/bbl)": np.round(gor, 0),
        "Clean Rate": np.round(q_clean, 0),
    })
    df["Cum Oil (Mbbls)"] = np.cumsum(df["Oil Rate (BOPD)"] * 30.4) / 1000
    return df
