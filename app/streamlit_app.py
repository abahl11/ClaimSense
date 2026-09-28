"""Streamlit app: score a new workers' compensation claim at reporting time.

Run with ``streamlit run app/streamlit_app.py`` (or ``run.ps1 app`` on Windows) after the
pipeline has trained the models. The app uses only information available when a claim is reported.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # let `streamlit run` import the project package
    sys.path.insert(0, str(ROOT))

from src.config import Config, load_config  # noqa: E402
from src.models.predict import (  # noqa: E402
    Models,
    clean_new_claims,
    explain_claim,
    load_models,
    make_explainer,
    score_claims,
)


@st.cache_resource
def resources() -> tuple[Models, Any]:
    """Load both models and the SHAP explainer once per session."""
    models = load_models(load_config())
    return models, make_explainer(models)


def choice(label: str, options: list[str], default: str) -> str:
    """Select box pre-set to the default option."""
    return st.selectbox(label, options, index=options.index(default))


def claim_form(cfg: Config) -> dict[str, Any] | None:
    """Claim details as entered by the adjuster; ``None`` until the form is submitted."""
    d, limits, c = cfg["app"]["defaults"], cfg["app"]["limits"], cfg["cleaning"]
    codes = cfg["data"]["codes"]
    min_days, max_days = c["days_per_week_range"]
    with st.form("claim"):
        left, right = st.columns(2)
        with left:
            accident_date = st.date_input(
                "Accident date", dt.date.fromisoformat(d["accident_date"])
            )
            accident_time = st.time_input(
                "Accident time", dt.time.fromisoformat(d["accident_time"])
            )
            report_date = st.date_input("Report date", dt.date.fromisoformat(d["report_date"]))
            age = st.number_input("Age", *limits["age"], value=d["age"])
            gender = choice("Gender", c["categories"]["gender"]["valid"], d["gender"])
            marital = choice(
                "Marital status (U = unknown)",
                c["categories"]["marital_status"]["valid"],
                d["marital_status"],
            )
            children = st.number_input(
                "Dependent children", 0, limits["dependents_max"], d["dependent_children"]
            )
            others = st.number_input(
                "Other dependants", 0, limits["dependents_max"], d["dependents_other"]
            )
        with right:
            wages = st.number_input(
                "Weekly wage", min_value=0.0, value=d["weekly_wages"], step=10.0
            )
            part_time = choice(
                "Full-time (F) / part-time (P)",
                [codes["full_time"], codes["part_time"]],
                d["part_time"],
            )
            hours = st.number_input(
                "Hours worked per week", 0.0, float(c["hours_per_week_max"]), d["hours_per_week"]
            )
            days = st.number_input("Days worked per week", min_days, max_days, d["days_per_week"])
            initial = st.number_input(
                "Adjuster's initial estimate", min_value=1, value=d["initial_estimate"]
            )
            description = st.text_area("Claim description", d["description"])
        submitted = st.form_submit_button("Score claim")
    if not submitted:
        return None
    return {
        "accident": dt.datetime.combine(accident_date, accident_time),
        "report": dt.datetime.combine(report_date, dt.time()),
        "age": age,
        "gender": gender,
        "marital_status": marital,
        "dependent_children": children,
        "dependents_other": others,
        "weekly_wages": wages,
        "part_time": part_time,
        "hours_per_week": hours,
        "days_per_week": days,
        "initial_estimate": initial,
        "description": description,
    }


def raw_claim(inputs: dict[str, Any], cfg: Config) -> pd.DataFrame:
    """One claim in the raw ``train.csv`` format, so it goes through the normal cleaning."""
    fmt = cfg["data"]["datetime_format"]
    values = {
        "claim_id": cfg["app"]["claim_id"],
        "accident_date": inputs["accident"].strftime(fmt),
        "report_date": inputs["report"].strftime(fmt),
        **{k: v for k, v in inputs.items() if k not in ("accident", "report")},
    }
    return pd.DataFrame([{cfg.col(name): value for name, value in values.items()}])


def show_scores(scored: pd.Series, models: Models, initial: float) -> None:
    """Cost, interval and high-cost flag."""
    first, second, third = st.columns(3)
    first.metric(
        "Expected ultimate cost",
        f"{scored['expected_cost']:,.0f}",
        help="Mean-type prediction: what to reserve on average for claims like this.",
    )
    second.metric(
        "Typical outcome",
        f"{scored['typical_cost']:,.0f}",
        help="Median-type prediction: half of similar claims end up below this.",
    )
    third.metric("80% range", f"{scored['interval_low']:,.0f} - {scored['interval_high']:,.0f}")
    ratio = scored["expected_cost"] / initial
    st.write(
        f"The model expects **{ratio:.2f}x** the adjuster's initial estimate of {initial:,.0f}."
    )
    probability = scored["high_cost_probability"]
    threshold, cutoff = models.classifier.threshold, models.classifier.cost_threshold
    message = (
        f"{probability:.0%} chance of exceeding {cutoff:,.0f} (flag threshold {threshold:.0%})."
    )
    if scored["flag_for_review"]:
        st.error(f"Flag for early review: {message}")
    else:
        st.success(f"No flag: {message}")


def show_drivers(drivers: pd.DataFrame) -> None:
    """The strongest drivers of this prediction."""
    st.subheader("What drives this estimate")
    st.caption(
        "Multiplicative effect of each input on the predicted cost, holding the others fixed "
        "(SHAP values on the log scale)."
    )
    st.bar_chart(drivers.set_index("driver")["shap"], horizontal=True)
    table = pd.DataFrame(
        {
            "Driver": drivers["driver"],
            "Effect on cost": drivers["factor"].map(lambda f: f"x{f:.2f}"),
        }
    )
    st.table(table)


def show_labels(scored: pd.Series) -> None:
    """What the keyword taxonomy read from the description."""
    parts = [
        f"{field.replace('_', ' ')}: **{str(scored[field]).replace('_', ' ')}**"
        for field in ("body_part", "injury_nature", "injury_cause")
    ]
    st.write("Read from the description: " + ", ".join(parts) + ".")


def main() -> None:
    """Render the app."""
    cfg = load_config()
    st.set_page_config(page_title=cfg["app"]["title"], layout="wide")
    st.title(cfg["app"]["title"])
    st.caption(
        "Decision support for adjusters: predicted ultimate cost, an 80% range and an early-review "
        "flag, from information available at reporting. Trained on synthetic data."
    )
    try:
        models, explainer = resources()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        return
    inputs = claim_form(cfg)
    if inputs is None:
        return
    cleaned = clean_new_claims(raw_claim(inputs, cfg), cfg)
    if cleaned.empty:
        st.warning("The claim could not be scored: check the accident date.")
        return
    scored = score_claims(cleaned, cfg, models).iloc[0]
    show_scores(scored, models, float(inputs["initial_estimate"]))
    show_labels(scored)
    show_drivers(explain_claim(cleaned, cfg, models, explainer, cfg["app"]["top_drivers"]))


main()
