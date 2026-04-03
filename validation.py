"""
Input validation for production engineering calculations.

Every public function returns a dict: {"valid": bool, "warnings": [...], "errors": [...]}.
Warnings are informational but don't block calculation. Errors should block.
"""

from typing import Any


def validate_reservoir_inputs(pr: float, pb: float, pi: float, api: float) -> dict:
    """Validate reservoir parameters."""
    errors, warnings = [], []

    if pr <= 0:
        errors.append("Reservoir pressure must be positive.")
    if pb <= 0:
        errors.append("Bubble point must be positive.")
    if pi <= 0:
        errors.append("Productivity index must be positive.")
    if not (15 <= api <= 60):
        warnings.append(f"API gravity {api} is outside typical range (15–60).")
    if pb > pr * 1.5:
        warnings.append(f"Bubble point ({pb} psi) is much higher than reservoir pressure ({pr} psi). Check inputs.")
    if pi > 20:
        warnings.append(f"PI = {pi} STB/d/psi is unusually high. Verify units.")
    return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}


def validate_well_inputs(depth: float, deviation: float, tubing_od: float) -> dict:
    errors, warnings = [], []
    if depth <= 0:
        errors.append("TVD must be positive.")
    if depth > 18000:
        warnings.append(f"TVD {depth:,.0f} ft is very deep. Confirm units.")
    if not (0 <= deviation <= 90):
        errors.append("Deviation must be 0–90 degrees.")
    if tubing_od not in (2.375, 2.875, 3.5):
        warnings.append(f"Non-standard tubing OD {tubing_od} in. Using nearest ID estimate.")
    return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}


def validate_fluid_inputs(wc_pct: float, gor: float, whp: float) -> dict:
    errors, warnings = [], []
    if not (0 <= wc_pct <= 100):
        errors.append("Water cut must be 0–100%.")
    if gor < 0:
        errors.append("GOR must be non-negative.")
    if gor > 5000:
        warnings.append(f"GOR {gor:,.0f} scf/bbl is very high. May indicate gas well rather than oil well.")
    if whp < 0:
        errors.append("Wellhead pressure must be non-negative.")
    return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}


def validate_decline_fit(model: dict, rmse: float = None, r2: float = None) -> dict:
    """Assess quality of a decline-curve fit."""
    warnings = []
    qi = model.get("qi", 0)
    di = model.get("di", 0)
    b = model.get("b", 0)

    if qi <= 0:
        warnings.append("qi ≤ 0 is physically impossible.")
    if di <= 0:
        warnings.append("di ≤ 0 means no decline, which is unrealistic.")
    if b < 0:
        warnings.append(f"b = {b:.3f} is negative, which is non-physical for Arps decline.")
    if b > 2.0:
        warnings.append(f"b = {b:.3f} exceeds 2.0. Arps models with b > 2 are mathematically unstable for EUR.")
    if rmse is not None and qi > 0 and rmse / qi > 0.3:
        warnings.append(f"RMSE is {rmse:.0f} vs qi = {qi:.0f}. Fit quality is poor.")
    if r2 is not None and r2 < 0.6:
        warnings.append(f"R² = {r2:.3f}. Fit explains less than 60% of variance.")

    confidence = "high"
    if r2 is not None and r2 < 0.8:
        confidence = "moderate"
    if r2 is not None and r2 < 0.6:
        confidence = "low"
    if b > 1.5:
        confidence = "low"  # b > 1.5 often signals overfitting

    return {"valid": len(warnings) == 0, "warnings": warnings, "confidence": confidence}


def validate_economics_inputs(oil_price: float, gas_price: float, discount_rate: float) -> dict:
    errors, warnings = [], []
    if oil_price <= 0:
        errors.append("Oil price must be positive.")
    if gas_price < 0:
        errors.append("Gas price must be non-negative.")
    if not (0 < discount_rate < 1):
        warnings.append(f"Discount rate {discount_rate:.2%} — confirm this is in decimal form.")
    return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}
