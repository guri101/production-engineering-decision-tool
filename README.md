# Well Production Optimization \& Artificial Lift Selection Tool

## A Production Engineering Portfolio Project — Permian Basin

**Author:** Singh
**Background:** Field Engineer / Application Engineer / Reliability Engineer (ESPs) 

\---

## Project Overview

This is an end-to-end production engineering decision tool that walks through the full lifecycle of a horizontal well in the Permian Basin. It integrates eight core modules into a single decision workflow:

1. **Nodal Analysis** — IPR (Vogel's equation) and TPR (multiphase flow) modeling to find the natural operating point, with flow-stability classification
2. **Artificial Lift Screening** — Weighted scoring matrix with "why this ranks first" explanations and constraint warnings
3. **Decline Curve Analysis** — Arps decline models with automatic preprocessing, fit-quality metrics (R², RMSE, MAPE), model auto-selection, and confidence scoring
4. **Economic Evaluation** — Monthly cash-flow model with separated CAPEX/OPEX components, do-nothing comparator, and capital efficiency metrics
5. **Sensitivity / Comparison** — Scenario sensitivity with tornado analysis and side-by-side case comparison
6. **Surveillance** — Forecast-vs-actual tracking with variance analysis
7. **Portfolio Dashboard** — Multi-well surveillance with exception flags, urgency scoring, and ranked priority table
8. **Decision Summary** — Executive-level recommendation package with downloadable reports

\---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

\---

## Architecture

```
Rev3/
├── app.py
├── decline_engine.py
├── nodal_engine.py
├── econ_engine.py
├── portfolio_rules.py
├── scenario_store.py
├── report_builder.py
├── unit_conversions.py
├── validation.py
├── requirements.txt
└── README.md
```

\---

## What Changed (File-by-File)

### `app.py` (Streamlit entry point)

* **Preserved:** All original UI layout, sidebar, tabs, styling, scenario management
* **Changed:** Imports all calculations from engine modules instead of inline functions
* **Added:** Tab 7 "Portfolio Dashboard" for multi-well surveillance
* **Added:** Input validation warnings displayed at top of page
* **Added:** Flow-stability indicator on nodal tab (STABLE/MARGINAL/NONE)
* **Added:** Model selection radio (auto-select or force model) on decline tab
* **Added:** Model comparison table with R², RMSE, MAPE on decline tab
* **Added:** Decline-fit confidence indicator and low-confidence warning
* **Added:** Do-nothing comparator column in economics tab
* **Added:** Cost breakdown table in economics tab
* **Added:** Tornado sensitivity chart in sensitivity tab
* **Added:** Constraint warning boxes on lift selection tab
* **Improved:** Manager flags now include nodal warnings and decline-confidence warnings
* **Fixed:** Screening note on ESP tab is now accurate (was hardcoded text about a prior change)

### `decline\_engine.py` (Phase A)

* **New module** replacing inline `arps\_\*`, `fit\_decline\_curves`, `calc\_eur`
* **Added:** `preprocess\_production()` — shut-in detection, zero-tail trimming, outlier removal, quality scoring
* **Added:** `fit\_metrics()` — RMSE, MAPE, R² computed per model
* **Added:** `select\_best\_model()` — auto-selection with R²-based ranking, b-value penalty, reason string
* **Added:** `generate\_monthly\_forecast()` — explicit monthly forecast generator
* **Preserved:** All Arps formulas (exponential, hyperbolic, harmonic) unchanged
* **Preserved:** `evaluate\_decline\_model()` — no lambdas, cache-safe

### `nodal\_engine.py` (Phase B)

* **New module** replacing inline `vogel\_ipr`, `beggs\_brill\_gradient`, `find\_operating\_point`, `screen\_artificial\_lift`, `size\_esp`
* **Changed:** `find\_operating\_point()` now returns a dict with stability classification, margin %, AOF, and warnings
* **Added:** `compute\_aof()` convenience function
* **Added:** `ranking\_notes` output from `screen\_artificial\_lift()` — explains why top method ranks first
* **Added:** Constraint-specific warnings in ranking notes (high GOR + ESP, high deviation + rod pump, etc.)
* **Preserved:** All Beggs-Brill gradient math unchanged
* **Preserved:** All lift-screening scores and weights unchanged

### `econ\_engine.py` (Phase C)

* **New module** replacing inline `calculate\_economics`
* **Added:** `LIFT\_COST\_PARAMS` dict — all cost parameters in one place, separated into equipment/installation/ancillary/other
* **Added:** "Natural Flow" as a lift method for do-nothing comparison
* **Added:** Variable OPEX per barrel fluid (separate from fixed OPEX)
* **Added:** Capital efficiency metric (NPV/CAPEX)
* **Added:** Monthly arrays in output (cashflow, revenue, opex, discounted) for potential charting
* **Added:** `tornado\_sensitivity()` — one-at-a-time ±20% sensitivity with NPV swing
* **Preserved:** Monthly discounting, replacement events, payout logic unchanged
* **Preserved:** Water disposal at $0.75/bbl unchanged

### `portfolio\_rules.py` (Phase D)

* **New module** — lightweight surveillance engine
* **Added:** `evaluate\_well\_flags()` — rule-based exception detection (13 flag types)
* **Added:** `score\_well\_urgency()` — 0–100 urgency score from flag severity
* **Added:** `build\_portfolio\_table()` — ranked multi-well table
* **Added:** `portfolio\_summary\_counts()` — status counts for dashboard
* **Moved:** `build\_forecast\_actual\_table()` from app.py
* **Moved:** Surveillance flag logic from app.py to `compute\_surveillance\_flags()`

### `scenario\_store.py`

* **Moved:** `SCENARIO\_PRESETS` from app.py
* **Moved:** `get\_current\_inputs()`, `load\_inputs\_to\_state()` from app.py
* **Added:** `VALID\_STATE\_KEYS` set for input validation

### `report\_builder.py`

* **Moved:** `build\_assumptions()`, `build\_manager\_report()`, `build\_engineer\_report()`, `build\_case\_report\_html()` from app.py
* **Added:** Decline confidence in manager report
* **Added:** Preprocessing info section in engineer report
* **Added:** Limitations section in manager report
* **Improved:** Report formatting with section headers

### `unit\_conversions.py`

* **New module** — fluid property helpers, gravity conversions, tubing lookups
* **Added:** `fluid\_sg\_mixture()` replaces inline calculation
* **Added:** `psi\_to\_ft\_head()`, `standing\_rs()`, `standing\_bo()` for reference
* **Moved:** `TUBING\_ID\_MAP` from app.py

### `validation.py`

* **New module** — input and fit-quality validation
* **Added:** `validate\_reservoir\_inputs()`, `validate\_well\_inputs()`, `validate\_fluid\_inputs()`
* **Added:** `validate\_decline\_fit()` — confidence scoring from model parameters and metrics
* **Added:** `validate\_economics\_inputs()`

\---

## External Repo Concepts: Adopted, Adapted, or Rejected

|Repo|Concept|Status|Notes|
|-|-|-|-|
|**pyResToolbox**|Arps formula definitions|**Validated**|Our formulas match pyResToolbox's Arps implementation. No code imported; used as benchmark.|
|**pyResToolbox**|IPR composite model|**Validated**|Vogel composite (above/below Pb) aligns with pyResToolbox approach.|
|**pyResToolbox**|VLP/TPR integration|**Evaluated, deferred**|Full integration impractical for single-file deployment. Aligned gradient assumptions instead.|
|**Python\_Automated\_DCA**|Change-point detection|**Adapted**|Implemented shut-in detection and post-shutin filtering pattern. Simplified vs. original (no scipy changepoint).|
|**Python\_Automated\_DCA**|Outlier-resistant fitting|**Adapted**|Pre-fit outlier removal (40% of median threshold) and quality scoring.|
|**PetroAlchemy**|Monthly forecast→cashflow structure|**Adopted**|Monthly cash-flow model with separated cost components follows PetroAlchemy pattern.|
|**PetroAlchemy**|Economics reporting|**Adapted**|Cost breakdown table and capital efficiency metric inspired by PetroAlchemy output.|
|**dcapy**|EUR from discrete sum|**Adopted**|EUR calculated from monthly summation (not closed-form shortcut) for transparency.|
|**dcapy**|Probabilistic forecasting|**Rejected**|Too complex for screening tool. Deterministic only.|
|**Intelligent Prod. Assistant**|Exception-based alerting|**Adapted**|13 flag types with severity levels and urgency scoring. Pure Python, no Snowflake.|
|**Intelligent Prod. Assistant**|Manager dashboard layout|**Adapted**|Status counts + ranked exception table + portfolio summary. Pattern borrowed, not code.|
|**Intelligent Prod. Assistant**|Chatbot/AI recommendations|**Rejected**|Explicitly excluded per requirements. All outputs are rule-based.|

\---

## Engineering Validation Notes

1. **Arps formulas** — Exponential, hyperbolic, harmonic formulas are standard textbook implementations. Validated against pyResToolbox output patterns.
2. **b-value constraint** — b is bounded \[0.01, 2.0] during fitting. b > 1.5 triggers a warning and is penalized in auto-selection. b > 2.0 is flagged as non-physical.
3. **Vogel IPR** — Composite model handles above-Pb (linear Darcy) and below-Pb (Vogel empirical) correctly with continuous transition at Pb.
4. **Beggs-Brill gradient** — Simplified holdup correlation without full flow-pattern map. Suitable for screening, not design. This is clearly labeled.
5. **ESP sizing** — Uses a representative catalog, not vendor-specific data. Safety factor (15%) applied to motor sizing. Gas handling recommendation tied to actual GOR.
6. **Economics** — Monthly discounting, not annual approximation. Water disposal is per-barrel-water (not per-barrel-fluid as in some implementations). Replacement events are periodic, not random.
7. **Known simplifications that could lose credibility:**

   * TPR uses a single average liquid viscosity (1.5 cp) rather than pressure-dependent calculation
   * No temperature-dependent PVT corrections in the gradient loop beyond Standing Rs/Bo
   * ESP catalog is illustrative, not a real vendor catalog
   * All cost figures are representative — no field-specific calibration

\---

## UI/Debug Notes

* **Theme:** Forced light theme via CSS variables. Dark-mode conflicts eliminated by explicit color overrides on all major Streamlit components.
* **Stability indicator:** New green/yellow/red flow-stability badge on nodal tab.
* **Confidence indicators:** Decline-fit confidence shown on decline tab and propagated to manager flags and reports.
* **Constraint warnings:** Yellow warning boxes on lift selection tab when engineering constraints affect the recommendation.
* **Input validation:** Warnings shown at top of page when inputs are outside typical ranges.
* **Do-nothing column:** Economics tab now shows a "Do Nothing" comparator alongside lift methods.
* **Tornado chart:** Visual sensitivity ranking on the sensitivity tab.
* **Portfolio tab:** New Tab 7 with multi-well surveillance, urgency scoring, and exportable data.

\---

## Remaining Limitations

1. **Synthetic data only** — No production history upload capability yet. Tab 3 uses synthetic data for demonstration.
2. **Single-well nodal** — No multi-well interference or reservoir simulation coupling.
3. **No real-time data** — Surveillance is manual entry, not connected to SCADA/PI.
4. **Simplified PVT** — Standing correlations only. No lab PVT input.
5. **No gas-lift design** — Gas lift is screened but not designed (no injection-point optimization, valve spacing, or compressor sizing).
6. **No rod-pump design** — Rod pump is screened but not designed (no pump card, dynamometer, or rod-string sizing).
7. **Fixed water disposal cost** — $0.75/bbl is hardcoded. Should be configurable per field.
8. **No tax/royalty** — Economics exclude taxes, royalties, and working interest fractions.
9. **No probabilistic forecasting** — All decline forecasts are deterministic single-curve. P10/P50/P90 not supported.
10. **Portfolio dashboard** — Currently uses preset scenarios as synthetic comparison wells. Real deployment would need database or file upload.

\---

*Built with Python, Streamlit, Plotly, NumPy, SciPy, and Pandas.*
