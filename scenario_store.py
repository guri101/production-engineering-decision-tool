"""
Scenario Store
==============
Presets and scenario management for the production engineering tool.
"""

import json
from typing import Dict, Any

SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Base Delaware Well": {
        "pr": 3200, "pb": 2400, "pi": 2.5, "api": 38,
        "depth": 9500, "deviation": 75, "tubing_od": 2.875,
        "wc_pct": 45, "gor": 950, "whp": 150, "viscosity": 3.0,
        "oil_price": 72, "gas_price": 3.0, "discount_pct": 10,
        "gas_lift_available": True,
    },
    "High-GOR Well": {
        "pr": 2600, "pb": 2700, "pi": 1.8, "api": 42,
        "depth": 9200, "deviation": 70, "tubing_od": 2.875,
        "wc_pct": 25, "gor": 2200, "whp": 220, "viscosity": 1.8,
        "oil_price": 72, "gas_price": 3.0, "discount_pct": 10,
        "gas_lift_available": True,
    },
    "Late-Life High-Water-Cut Well": {
        "pr": 1800, "pb": 2400, "pi": 0.9, "api": 34,
        "depth": 9000, "deviation": 60, "tubing_od": 2.375,
        "wc_pct": 82, "gor": 700, "whp": 80, "viscosity": 5.0,
        "oil_price": 68, "gas_price": 3.0, "discount_pct": 12,
        "gas_lift_available": False,
    },
}


VALID_STATE_KEYS = {
    "scenario_name", "pr", "pb", "pi", "api", "depth", "deviation",
    "tubing_od", "wc_pct", "gor", "whp", "viscosity", "oil_price",
    "gas_price", "discount_pct", "gas_lift_available",
}


def get_current_inputs(session_state) -> dict:
    """Extract current inputs from Streamlit session state."""
    return {k: session_state.get(k, SCENARIO_PRESETS["Base Delaware Well"].get(k))
            for k in VALID_STATE_KEYS}


def load_inputs_to_state(payload: dict, session_state):
    """Load a scenario payload into session state."""
    for key, value in payload.items():
        if key in VALID_STATE_KEYS:
            session_state[key] = value


def scenario_to_json(inputs: dict) -> str:
    return json.dumps(inputs, indent=2)


def scenario_from_json(text: str) -> dict:
    return json.loads(text)
