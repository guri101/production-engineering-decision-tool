"""
Directional Survey Engine
=========================
Directional-survey normalization, minimum-curvature trajectory calculations,
dogleg-severity screening, and lift-placement window suggestions.

All results are screening-grade and meant to support engineering workflow,
not replace full well-planning or vendor design software.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SURVEY_COLUMN_ALIASES = {
    "md": "MD",
    "measureddepth": "MD",
    "measured_depth": "MD",
    "depthmd": "MD",
    "inclination": "Inclination",
    "inc": "Inclination",
    "angle": "Inclination",
    "incl": "Inclination",
    "azimuth": "Azimuth",
    "azi": "Azimuth",
    "azm": "Azimuth",
}


LIFT_DLS_LIMITS = {
    "ESP": {"preferred": 3.0, "caution": 5.0, "max_inclination": 85.0},
    "Gas Lift": {"preferred": 6.0, "caution": 8.0, "max_inclination": 88.0},
    "Rod Pump": {"preferred": 2.0, "caution": 3.0, "max_inclination": 35.0},
    "Plunger Lift": {"preferred": 8.0, "caution": 10.0, "max_inclination": 90.0},
}


def _normalize_column_name(name: str) -> str:
    key = "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch == "_")
    return SURVEY_COLUMN_ALIASES.get(key, name)


def normalize_survey_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a survey upload into MD / Inclination / Azimuth columns.
    Raises ValueError if required columns are missing or invalid.
    """
    if df is None or df.empty:
        raise ValueError("Survey file is empty.")

    renamed = df.rename(columns={col: _normalize_column_name(col) for col in df.columns}).copy()
    required = ["MD", "Inclination", "Azimuth"]
    missing = [col for col in required if col not in renamed.columns]
    if missing:
        raise ValueError(
            "Survey file must contain MD, Inclination, and Azimuth columns. "
            f"Missing: {', '.join(missing)}"
        )

    cleaned = renamed[required].copy()
    for col in required:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna().sort_values("MD").reset_index(drop=True)

    if len(cleaned) < 2:
        raise ValueError("Survey must contain at least two valid stations.")
    if cleaned["MD"].diff().fillna(1).le(0).any():
        raise ValueError("Survey MD values must increase monotonically.")
    if cleaned["Inclination"].lt(0).any() or cleaned["Inclination"].gt(180).any():
        raise ValueError("Inclination must stay between 0 and 180 degrees.")

    cleaned["Azimuth"] = cleaned["Azimuth"] % 360.0
    return cleaned


def calculate_minimum_curvature(survey_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TVD, northing, easting, course length, and DLS by minimum curvature.
    DLS is returned in deg/100 ft.
    """
    survey = normalize_survey_dataframe(survey_df)
    result = survey.copy()
    result["TVD"] = 0.0
    result["Northing"] = 0.0
    result["Easting"] = 0.0
    result["Course Length"] = result["MD"].diff().fillna(0.0)
    result["Dogleg Angle"] = 0.0
    result["DLS"] = 0.0

    for idx in range(1, len(result)):
        md1, inc1, azi1 = result.loc[idx - 1, ["MD", "Inclination", "Azimuth"]]
        md2, inc2, azi2 = result.loc[idx, ["MD", "Inclination", "Azimuth"]]
        cl = float(md2 - md1)
        inc1_rad, inc2_rad = np.radians([inc1, inc2])
        azi1_rad, azi2_rad = np.radians([azi1, azi2])

        cos_dogleg = (
            np.cos(inc1_rad) * np.cos(inc2_rad)
            + np.sin(inc1_rad) * np.sin(inc2_rad) * np.cos(azi2_rad - azi1_rad)
        )
        cos_dogleg = float(np.clip(cos_dogleg, -1.0, 1.0))
        dogleg = float(np.arccos(cos_dogleg))
        rf = 1.0 if dogleg < 1e-12 else float((2.0 / dogleg) * np.tan(dogleg / 2.0))

        d_north = 0.5 * cl * (
            np.sin(inc1_rad) * np.cos(azi1_rad) + np.sin(inc2_rad) * np.cos(azi2_rad)
        ) * rf
        d_east = 0.5 * cl * (
            np.sin(inc1_rad) * np.sin(azi1_rad) + np.sin(inc2_rad) * np.sin(azi2_rad)
        ) * rf
        d_tvd = 0.5 * cl * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf

        result.loc[idx, "Northing"] = result.loc[idx - 1, "Northing"] + d_north
        result.loc[idx, "Easting"] = result.loc[idx - 1, "Easting"] + d_east
        result.loc[idx, "TVD"] = result.loc[idx - 1, "TVD"] + d_tvd
        result.loc[idx, "Dogleg Angle"] = np.degrees(dogleg)
        result.loc[idx, "DLS"] = 0.0 if cl <= 0 else np.degrees(dogleg) * 100.0 / cl

    return result


def summarize_survey(calculated_survey: pd.DataFrame) -> dict[str, Any]:
    survey = calculated_survey.copy()
    return {
        "stations": int(len(survey)),
        "max_md": float(survey["MD"].max()),
        "max_tvd": float(survey["TVD"].max()),
        "max_inclination": float(survey["Inclination"].max()),
        "max_dls": float(survey["DLS"].max()),
        "avg_dls": float(survey["DLS"].iloc[1:].mean()) if len(survey) > 1 else 0.0,
        "kickoff_md": float(survey.loc[survey["Inclination"] > 3.0, "MD"].min())
        if (survey["Inclination"] > 3.0).any()
        else None,
    }


def _build_windows(mask: pd.Series, survey: pd.DataFrame) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    start_idx = None
    for idx, is_valid in enumerate(mask.tolist()):
        if is_valid and start_idx is None:
            start_idx = idx
        if not is_valid and start_idx is not None:
            end_idx = idx - 1
            windows.append({
                "md_start": float(survey.iloc[start_idx]["MD"]),
                "md_end": float(survey.iloc[end_idx]["MD"]),
                "tvd_start": float(survey.iloc[start_idx]["TVD"]),
                "tvd_end": float(survey.iloc[end_idx]["TVD"]),
                "length": float(survey.iloc[end_idx]["MD"] - survey.iloc[start_idx]["MD"]),
            })
            start_idx = None
    if start_idx is not None:
        end_idx = len(survey) - 1
        windows.append({
            "md_start": float(survey.iloc[start_idx]["MD"]),
            "md_end": float(survey.iloc[end_idx]["MD"]),
            "tvd_start": float(survey.iloc[start_idx]["TVD"]),
            "tvd_end": float(survey.iloc[end_idx]["TVD"]),
            "length": float(survey.iloc[end_idx]["MD"] - survey.iloc[start_idx]["MD"]),
        })
    return [w for w in windows if w["length"] >= 100.0]


def recommend_lift_setting_windows(calculated_survey: pd.DataFrame, lift_type: str) -> dict[str, Any]:
    """
    Return preferred and cautionary placement windows for a given lift type.
    """
    survey = calculated_survey.copy()
    lift_key = lift_type if lift_type in LIFT_DLS_LIMITS else "ESP"
    limits = LIFT_DLS_LIMITS[lift_key]

    preferred_mask = (
        survey["DLS"].le(limits["preferred"]) & survey["Inclination"].le(limits["max_inclination"])
    )
    caution_mask = (
        survey["DLS"].le(limits["caution"]) & survey["Inclination"].le(limits["max_inclination"])
    )

    preferred_windows = _build_windows(preferred_mask, survey)
    caution_windows = _build_windows(caution_mask, survey)

    def pick_window(windows: list[dict[str, float]]) -> dict[str, float] | None:
        if not windows:
            return None
        return sorted(windows, key=lambda item: (item["md_end"], item["length"]), reverse=True)[0]

    recommended = pick_window(preferred_windows) or pick_window(caution_windows)
    if recommended:
        recommended = dict(recommended)
        recommended["recommended_md"] = round((recommended["md_start"] + recommended["md_end"]) / 2.0, 1)
        recommended["recommended_tvd"] = round((recommended["tvd_start"] + recommended["tvd_end"]) / 2.0, 1)

    return {
        "lift_type": lift_key,
        "limits": limits,
        "preferred_windows": preferred_windows,
        "caution_windows": caution_windows,
        "recommended_window": recommended,
    }
