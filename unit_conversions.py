"""
Unit conversion and fluid-property helpers for production engineering.

References:
- Standing correlations for Rs, Bo
- API gravity / specific-gravity conversions
- Standard oilfield unit conversions
"""

import numpy as np


# ── Gravity / density ──────────────────────────────────────
def api_to_sg(api: float) -> float:
    """API gravity → oil specific gravity (water = 1.0)."""
    return 141.5 / (api + 131.5)


def sg_to_api(sg: float) -> float:
    """Oil specific gravity → API gravity."""
    return 141.5 / max(sg, 0.01) - 131.5


def fluid_sg_mixture(wc: float, api: float, gamma_w: float = 1.07) -> float:
    """Composite fluid SG at surface conditions given water cut (fraction) and oil API."""
    gamma_o = api_to_sg(api)
    return (1.0 - wc) * gamma_o + wc * gamma_w


# ── Pressure / head ────────────────────────────────────────
def psi_to_ft_head(psi: float, sg: float) -> float:
    """Convert psi to feet of fluid head."""
    return psi * 2.31 / max(sg, 0.01)


def ft_head_to_psi(ft: float, sg: float) -> float:
    """Convert feet of fluid head to psi."""
    return ft * sg / 2.31


# ── Volume ─────────────────────────────────────────────────
def bbl_to_m3(bbl: float) -> float:
    return bbl * 0.158987


def m3_to_bbl(m3: float) -> float:
    return m3 / 0.158987


def mscf_to_e3m3(mscf: float) -> float:
    return mscf * 0.028317


# ── PVT helpers (Standing-type) ────────────────────────────
def standing_rs(p: float, t: float, api: float, gamma_g: float = 0.75) -> float:
    """
    Standing correlation for solution gas–oil ratio (scf/STB).
    p : pressure (psi), t : temperature (°F)
    """
    if p <= 14.7:
        return 0.0
    x = 0.0125 * api - 0.00091 * t
    rs = gamma_g * (max(p, 14.7) / 18.2 * 10 ** x) ** 1.204
    return max(rs, 0.0)


def standing_bo(rs: float, t: float, gamma_o: float, gamma_g: float = 0.75) -> float:
    """Standing correlation for oil formation volume factor (rb/STB)."""
    bo = 0.972 + 0.000147 * (rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * t) ** 1.175
    return max(bo, 1.0)


def gas_bg(t_rankine: float, p: float) -> float:
    """Ideal-gas formation volume factor (res bbl / scf)."""
    return 0.0283 * t_rankine / max(p, 50.0)


# ── Tubing geometry ────────────────────────────────────────
TUBING_ID_MAP = {
    2.375: 1.995,
    2.875: 2.441,
    3.5: 2.992,
}


def tubing_od_to_id(od: float) -> float:
    """Look up tubing ID (inches) from standard OD."""
    if od in TUBING_ID_MAP:
        return TUBING_ID_MAP[od]
    raise ValueError(f"Unknown tubing OD: {od}. Supported: {list(TUBING_ID_MAP.keys())}")
