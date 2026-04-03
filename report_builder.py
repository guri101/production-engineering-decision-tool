"""
Report Builder
==============
Text report generation for manager and engineer audiences.
Reports are grounded in computed outputs only — no speculative advice.
"""

from datetime import date


def build_assumptions(pr, pb, wc_pct, gor, gas_lift_available):
    items = [
        "IPR uses Vogel-style inflow and a simplified multiphase TPR. This is screening-grade, not a replacement for field-validated nodal software.",
        "Decline uses synthetic production history to demonstrate workflow. It is not a history match to a real well file.",
        f"Water cut is modeled at {wc_pct}% and GOR at {gor:,} scf/bbl across the screening case.",
        f"Pressure regime is treated as {'below' if pr < pb else 'above'} bubble point using the entered reservoir pressure and bubble point.",
        f"Gas-lift infrastructure is {'available' if gas_lift_available else 'not available'}, which directly affects method ranking.",
        "Economics are simplified to compare options consistently; they do not include taxes, ownership burdens, downtime logistics, or detailed vendor pricing.",
        "All cost figures are representative and should be replaced with field-specific vendor quotes for investment decisions.",
    ]
    return items


def build_manager_report(scenario_name, q_op, aof, wc_pct, gor, best_lift,
                         best_npv_method, eur_hyp, actions, econ_results,
                         decline_confidence=None):
    """Generate a manager-level report string."""
    natural_flow_text = (
        f"The well is flowing naturally at about {q_op:,.0f} BPD, which is {q_op / max(aof, 1) * 100:.0f}% of modeled AOF."
        if q_op is not None
        else "The well does not show a stable natural-flow operating point at the entered conditions."
    )
    flag_text = "; ".join([msg for _, msg in actions]) if actions else "No immediate trigger flags were created by the current rule set."

    lines = [
        f"Production Engineering Screening Report — {scenario_name}",
        f"Date: {date.today().isoformat()}",
        "",
        "EXECUTIVE SUMMARY",
        f"Engineering-fit ranking places {best_lift} first for the entered well conditions.",
        f"Economic ranking places {best_npv_method} first by modeled NPV.",
        natural_flow_text,
        f"Fluid state: {wc_pct}% water cut, {gor:,} scf/bbl GOR.",
    ]

    if eur_hyp:
        lines.append(f"Hyperbolic EUR screening estimate: {eur_hyp:,.0f} Mbbl.")
    else:
        lines.append("Hyperbolic EUR could not be calculated from the active decline fit.")

    if decline_confidence:
        lines.append(f"Decline-fit confidence: {decline_confidence}.")

    lines += ["", "ECONOMIC SNAPSHOT"]
    for method, econ in econ_results.items():
        payout = econ['Payout (years)']
        payout_str = f"{payout} years" if isinstance(payout, (int, float)) else str(payout)
        lines.append(
            f"  {method}: NPV ${econ['NPV']:,.0f} | CAPEX ${econ['CAPEX Total']:,.0f} | "
            f"Payout {payout_str} | Lifting cost ${econ['Lifting Cost ($/BOE)']:.2f}/BOE"
        )

    lines += ["", "TRIGGER FLAGS", flag_text]
    lines += [
        "",
        "LIMITATIONS",
        "This is a screening-grade analysis using simplified correlations and representative costs.",
        "Results should be validated with detailed nodal software and field-specific cost data before investment.",
    ]
    return "\n".join(lines)


def build_engineer_report(inputs, q_op, pwf_op, aof, best_lift, models,
                          eur_results, lift_df, econ_results, preprocessing_info=None):
    """Generate an engineer-level detail report string."""
    lines = [
        f"Engineering Detail Report — {inputs.get('scenario_name', 'Unknown')}",
        f"Date: {date.today().isoformat()}",
        "",
        "INPUT PARAMETERS",
    ]
    for k, v in inputs.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "NODAL ANALYSIS",
        f"  AOF: {aof:,.0f} BPD",
        f"  Operating rate: {q_op:,.0f} BPD" if q_op is not None else "  Operating rate: no stable natural-flow solution",
        f"  Flowing BHP: {pwf_op:,.0f} psi" if pwf_op is not None else "  Flowing BHP: N/A",
        f"  Recommended lift from screening: {best_lift}",
    ]

    lines += ["", "DECLINE MODELS"]
    for name, model in models.items():
        eur = eur_results.get(name, 0)
        r2 = model.get("r2", "N/A")
        rmse = model.get("rmse", "N/A")
        lines.append(
            f"  {name}: qi={model['qi']:.1f}, di={model['di']:.4f}, "
            f"b={model.get('b', 0):.3f} | R²={r2} | RMSE={rmse} | EUR={eur:,.0f} Mbbl"
        )

    if preprocessing_info:
        lines += ["", "DATA PREPROCESSING"]
        lines.append(f"  Points used: {preprocessing_info.get('n_points_used', 'N/A')} of {preprocessing_info.get('n_points_total', 'N/A')}")
        lines.append(f"  Data quality score: {preprocessing_info.get('quality_score', 'N/A')}")
        for flag in preprocessing_info.get("flags", []):
            lines.append(f"  - {flag}")

    lines += ["", "LIFT SCREENING"]
    for _, row in lift_df.iterrows():
        lines.append(f"  {row['Method']}: weighted score {row['Weighted Score']:.2f}")

    lines += ["", "ECONOMICS"]
    for method, econ in econ_results.items():
        lines.append(
            f"  {method}: NPV ${econ['NPV']:,.0f} | Payout {econ['Payout (months)']} months | "
            f"CAPEX ${econ['CAPEX Total']:,.0f} | Lifting ${econ['Lifting Cost ($/BOE)']:.2f}/BOE"
        )

    return "\n".join(lines)


def build_case_report_html(best_lift, best_npv_method, q_op, aof, wc_pct,
                           gor, eur_hyp, actions, decline_confidence=None):
    """HTML report block for the decision summary tab."""
    flag_html = ("<br>".join([f"&bull; {msg}" for _, msg in actions])
                 if actions
                 else "No immediate trigger flags from the current screening rules.")

    if q_op is not None:
        margin_ratio = q_op / max(aof, 1)
        if margin_ratio < 0.35:
            flow_desc = "flowing naturally with limited margin"
        elif margin_ratio < 0.6:
            flow_desc = "flowing naturally with moderate margin"
        else:
            flow_desc = "flowing naturally with healthy margin"
    else:
        flow_desc = "outside a stable natural-flow operating envelope at the current inputs"

    confidence_html = ""
    if decline_confidence:
        confidence_html = f"<br>Decline-fit confidence: <strong>{decline_confidence}</strong>."

    eur_html = ""
    if eur_hyp:
        eur_html = f" Hyperbolic EUR is approximately <strong>{eur_hyp:,.0f} Mbbl</strong>."

    return f"""
    <div class="report-box">
    <strong>Bottom line:</strong> Engineering-fit ranking places <strong>{best_lift}</strong> first.
    Economic ranking places <strong>{best_npv_method}</strong> first by modeled NPV.
    <br><br>
    <strong>Well condition:</strong> The well is {flow_desc}.
    Water cut is <strong>{wc_pct}%</strong>, GOR is <strong>{gor:,} scf/bbl</strong>,
    and the decline profile indicates increasing lift burden through time.{eur_html}{confidence_html}
    <br><br>
    <strong>Trigger flags:</strong><br>{flag_html}
    </div>
    """
