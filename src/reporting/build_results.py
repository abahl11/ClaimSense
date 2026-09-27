"""Builds reports/results.md from the metric files in reports/metrics/.

Nothing in the results report is typed by hand: every number is read from the JSON files
written by the modelling stages, and every sentence that states a finding is templated on
those numbers. Sections whose stage has not run yet say so instead of showing stale values.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from src.config import Config
from src.reporting.outputs import GENERATED_NOTE, markdown_table, write_text

logger = logging.getLogger(__name__)

NOT_RUN = "_Not run yet: run `python -m src.pipeline --stage {stage}`._"
CLASSIFIER_TITLE = "## 8. High-cost claim flag (classification)"


def load_metrics(cfg: Config, key: str) -> dict[str, Any] | None:
    """Parsed metrics file for a ``paths`` key, or ``None`` if the stage has not run."""
    path = cfg.path(key)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def ordinal(q: float) -> str:
    """A quantile as an ordinal percentile, e.g. 0.9 -> '90th'."""
    n = round(q * 100)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def money(value: float) -> str:
    """Currency amount with thousands separators and no decimals."""
    return f"{value:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    """Fraction as a percentage."""
    return f"{value:.{decimals}%}"


def figure(name: str, alt: str) -> str:
    """Markdown image link to a figure in reports/figures/."""
    return f"![{alt}](figures/{name}.png)"


def relative_change(new: float, old: float) -> float:
    """(new - old) / old."""
    return (new - old) / old


def by_name(regression: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Candidate records keyed by name."""
    return {c["name"]: c for c in regression["candidates"]}


def candidate_row(record: dict[str, Any]) -> dict[str, Any]:
    """One row of a model comparison table."""
    cv, test = record["cv"], record["test"]
    return {
        "Model": record["name"],
        "CV RMSLE": f"{cv['rmsle']:.4f} ± {cv['rmsle_std']:.4f}",
        "CV log bias": f"{cv['mean_log_error']:+.3f}",
        "Test RMSLE": f"{test['rmsle']:.4f}",
        "Test R² (log)": f"{test['r2_log']:.3f}",
        "Test MAE": money(test["mae"]),
        "Test MAE (typical)": money(test["mae_typical"]),
        "Test RMSE": money(test["rmse"]),
        "Test total pred / actual": f"{test['bias_ratio']:.3f}",
    }


def comparison_table(records: list[dict[str, Any]]) -> str:
    """Markdown comparison table for a list of candidate records."""
    return markdown_table(pd.DataFrame([candidate_row(r) for r in records]))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def validation_section(v: dict[str, Any]) -> list[str]:
    """Train/test split, time-ordered folds and how different the test period is."""
    folds = pd.DataFrame(
        [
            {
                "Fold": i,
                "Trains on": f"{f['fit']['start']} to {f['fit']['end']}",
                "Training claims": f"{f['fit']['claims']:,}",
                "Validates on": f"{f['eval']['start']} to {f['eval']['end']}",
                "Validation claims": f"{f['eval']['claims']:,}",
                "Validation median cost": money(f["eval"]["median_cost"]),
            }
            for i, f in enumerate(v["folds"], 1)
        ]
    )
    final = v["final"]
    growth = v["test"]["median_cost"] / v["train"]["median_cost"]
    return [
        "## 1. Validation design",
        "",
        f"- **Final test set:** the most recent {v['test']['claims']:,} claims (accidents from "
        f"{final['eval']['start']} to {final['eval']['end']}). It is scored once, after every "
        "choice has been made.",
        f"- **Training period:** {v['train']['claims']:,} claims from {final['fit']['start']} to "
        f"{final['fit']['end']}.",
        f"- **Model selection:** {v['cv_folds']} time-ordered folds inside the training period. "
        "Each fold trains on everything before a block of claims and validates on that block, "
        "so a model is always scored on claims that happened after its training data, as in "
        "deployment. A random K-fold split would mix future claims into training and, with "
        "costs rising this fast, would flatter every model.",
        f"- **Out-of-time calibration:** the high-cost probabilities and the review threshold "
        f"are calibrated on the most recent {pct(v['inner_holdout_fraction'], 0)} of the "
        "training period, with the model fit on the earlier part.",
        "",
        figure("validation_timeline", "Validation timeline"),
        "",
        markdown_table(folds),
        "",
        f"**The test period is much more expensive than the training period.** Its median cost "
        f"is {money(v['test']['median_cost'])} against {money(v['train']['median_cost'])} in "
        f"training ({growth:.1f}x), and {pct(v['test']['share_above_train_high_cost_threshold'])} "
        f"of test claims exceed the training high-cost threshold of "
        f"{money(v['high_cost_threshold'])} ({pct(1 - v['high_cost_quantile'], 0)} by construction "
        "in training). Every test "
        "result below should be read against that shift.",
        "",
    ]


def comparison_section(r: dict[str, Any]) -> list[str]:
    """Stage 1: baselines, model families and losses."""
    records = by_name(r)
    first = [records[n] for n in r["groups"]["stage_one"]]
    best = records[r["choices"]["best_family"]]
    adjuster = records["Baseline B: adjuster's initial estimate"]
    cv_gain = relative_change(best["cv"]["rmsle"], adjuster["cv"]["rmsle"])
    test_gain = relative_change(best["test"]["rmsle"], adjuster["test"]["rmsle"])
    losses = [c for c in first if c["group"] == "loss"]
    best_loss = min(losses, key=lambda c: c["cv"]["rmsle"])
    return [
        "## 2. Model comparison",
        "",
        "All models share one preprocessing pipeline. *CV* columns are the mean (± standard "
        "deviation) over the time-ordered folds and drive every decision; *Test* columns are "
        "the single final evaluation. RMSLE is RMSE on the log scale, so it measures relative "
        "error and treats small and large claims evenly. **RMSE in currency is dominated by the "
        "largest claims**: one claim missed by a million outweighs thousands of small ones, so "
        "it is shown for completeness but not used to choose. Currency metrics use the "
        "expected (mean) prediction; log metrics use the typical (median-type) prediction.",
        "",
        figure("regression_model_comparison", "Model comparison"),
        "",
        comparison_table(first),
        "",
        f"**Takeaway.** {best['name']} is the best model family in cross-validation (RMSLE "
        f"{best['cv']['rmsle']:.4f}), {abs(cv_gain):.0%} "
        f"{'lower' if cv_gain < 0 else 'higher'} than using the adjuster's initial estimate "
        f"({adjuster['cv']['rmsle']:.4f}); on the test set the gap is {abs(test_gain):.0%}. "
        f"The best raw-cost loss is {best_loss['name']} (CV RMSLE "
        f"{best_loss['cv']['rmsle']:.4f}). These losses predict the mean directly, which sits "
        "above the median on the log scale, so RMSLE understates them a little; their test "
        "currency metrics are the fairer comparison.",
        "",
    ]


def drift_context(eda: dict[str, Any] | None) -> str:
    """One sentence on the cost trend found in EDA (if the EDA metrics exist)."""
    if eda is None:
        return "Claim costs rise over the period (see EDA section 5)."
    kind = "a price-level change" if eda["mix_share_of_growth"] < 0.5 else "a change in claim mix"
    return (
        f"EDA showed the median claim cost growing {eda['yearly_median_growth']:.1f}x across the "
        f"period, mainly as {kind}."
    )


def drift_section(r: dict[str, Any], eda: dict[str, Any] | None) -> list[str]:
    """Stage 2: price-drift approaches."""
    records = by_name(r)
    drift = [records[n] for n in r["groups"]["drift"]]
    chosen = records[r["choices"]["best_drift"]]
    by_test = min(drift, key=lambda c: c["test"]["rmsle"])
    agree = by_test["name"] == chosen["name"]
    rows = [
        {
            "Approach": c["name"],
            "CV RMSLE": f"{c['cv']['rmsle']:.4f}",
            "CV log bias": f"{c['cv']['mean_log_error']:+.3f}",
            "Test RMSLE": f"{c['test']['rmsle']:.4f}",
            "Test log bias": f"{c['test']['mean_log_error']:+.3f}",
            "Test total pred / actual": f"{c['test']['bias_ratio']:.3f}",
        }
        for c in drift
    ]
    years = {name: series for name, series in r["drift_bias_by_year"].items()}
    last_year = max(int(y) for y in next(iter(years.values())))
    last = {name: series[str(last_year)] for name, series in years.items()}
    return [
        "## 3. Handling the rise in costs over time",
        "",
        f"{drift_context(eda)} Four ways of handling that were compared with the best model "
        "family: **A** predicts log cost directly; **B** predicts the log ratio of ultimate "
        "cost to the adjuster's initial estimate, which is recorded at current prices; **C** "
        "removes a log-linear cost trend fitted on the last training years and adds it back "
        "when predicting, so costs keep rising after the training period; **D** combines B "
        "and C. Log bias below 0 means the model predicts too low.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        figure("regression_bias_by_year", "Bias by accident year"),
        "",
        f"**Takeaway.** Cross-validation chooses **{chosen['name']}** (CV RMSLE "
        f"{chosen['cv']['rmsle']:.4f}). "
        + (
            "The test set agrees: it is also the best approach there."
            if agree
            else f"On the test set {by_test['name']} scores best ({by_test['test']['rmsle']:.4f}), "
            "a reminder that the future can drift differently from the past; the choice stays "
            "with cross-validation so that the test set is never used to decide."
        )
        + f" In the last test year ({last_year}) total predicted over total actual cost is "
        + ", ".join(f"{name.split(', ', 1)[-1][0]} {value:.2f}" for name, value in last.items())
        + ".",
        "",
    ]


def ablation_section(r: dict[str, Any]) -> list[str]:
    """Stage 3: what the text features add."""
    records = by_name(r)
    ablation = [records[n] for n in r["groups"]["ablation"]]
    none, full = ablation[0], ablation[-1]
    chosen = records[r["choices"]["best_layers"]]
    cv_gain = relative_change(full["cv"]["rmsle"], none["cv"]["rmsle"])
    test_gain = relative_change(full["test"]["rmsle"], none["test"]["rmsle"])
    rows = [
        {
            "Feature set": c["name"],
            "CV RMSLE": f"{c['cv']['rmsle']:.4f}",
            "Test RMSLE": f"{c['test']['rmsle']:.4f}",
            "Test R² (log)": f"{c['test']['r2_log']:.3f}",
        }
        for c in ablation
    ]
    steps = [
        f"adding {later['name'].split(': ')[-1].removeprefix('+ ')}: "
        f"{later['cv']['rmsle'] - earlier['cv']['rmsle']:+.4f}"
        for earlier, later in zip(ablation[:-1], ablation[1:], strict=True)
    ]
    return [
        "## 4. What the text features add (ablation)",
        "",
        "The chosen model and drift approach, refit with each set of description features. "
        "Structured fields (including the initial estimate) are always present, so this "
        "measures what the free-text description adds on top of them.",
        "",
        figure("regression_ablation", "Text-layer ablation"),
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        f"**Takeaway.** Adding all three text layers to the structured fields changes CV RMSLE "
        f"by {cv_gain:+.1%} and test RMSLE by {test_gain:+.1%}. Step by step: "
        + "; ".join(steps)
        + f" (CV RMSLE). The final model uses **{chosen['name']}**: a richer set is only "
        "preferred when "
        "it improves CV RMSLE by more than the configured tolerance.",
        "",
    ]


def tuning_section(best: dict[str, Any] | None) -> list[str]:
    """Stage 4: Optuna tuning."""
    if best is None:
        return ["## 5. Hyperparameter tuning", "", NOT_RUN.format(stage="train_reg"), ""]
    params = pd.DataFrame(
        [
            {"Setting": k, "Value": f"{v:.4g}" if isinstance(v, float) else v}
            for k, v in best["best_params"].items()
        ]
    )
    change = relative_change(best["best_cv_rmsle"], best["untuned_cv_rmsle"])
    return [
        "## 5. Hyperparameter tuning",
        "",
        f"Optuna (seeded TPE sampler, {best['n_trials']} trials) searched the model settings, "
        "scoring each trial by the same time-ordered cross-validation. The test set played no "
        "part. All trials are stored in `reports/metrics/best_params.json`.",
        "",
        markdown_table(params),
        "",
        (
            f"**Takeaway.** Tuning moves CV RMSLE from {best['untuned_cv_rmsle']:.4f} to "
            f"{best['best_cv_rmsle']:.4f} ({change:+.1%}), "
            + (
                "a modest gain: the defaults were already reasonable."
                if change > -0.02
                else "a clear gain."
            )
            if change < 0
            else f"**Takeaway.** Tuning did not beat the defaults (CV RMSLE "
            f"{best['untuned_cv_rmsle']:.4f} untuned vs {best['best_cv_rmsle']:.4f} best trial), "
            "so the untuned settings are kept."
        ),
        "",
    ]


def mae_text(test: dict[str, Any], adjuster: dict[str, Any]) -> str:
    """Explain MAE of the mean prediction vs the typical prediction vs the adjuster."""
    text = (
        f"**On MAE.** MAE is lowest when predicting the median, not the mean. The model's "
        f"typical (median-type) prediction has MAE {money(test['mae_typical'])}; its expected "
        f"(mean) prediction, which is what reserves need, has MAE {money(test['mae'])}; the "
        f"adjusters' estimates have {money(adjuster['mae'])}."
    )
    if adjuster["mae"] < test["mae"] and adjuster["bias_ratio"] < 1:
        text += (
            f" The adjusters' lower MAE comes from estimating low across the board (they book "
            f"{adjuster['bias_ratio']:.2f} of the true total), which suits the many small claims "
            "but leaves the book under-reserved."
        )
    return text


def rise_text(values: pd.Series) -> str:
    """Describe whether a series rises at every step."""
    return "rises with every decile" if values.is_monotonic_increasing else "mostly rises"


def final_section(r: dict[str, Any]) -> list[str]:
    """The final model on the test set: accuracy, calibration and intervals."""
    final = r["final"]
    test = final["test"]
    records = by_name(r)
    adjuster = records["Baseline B: adjuster's initial estimate"]["test"]
    deciles = pd.DataFrame(final["calibration_deciles"])
    deciles_table = deciles.assign(
        mean_predicted=deciles["mean_predicted"].map(money),
        mean_actual=deciles["mean_actual"].map(money),
        ratio=deciles["ratio"].map(lambda x: f"{x:.2f}"),
    ).rename(
        columns={
            "decile": "Predicted-cost decile",
            "claims": "Claims",
            "mean_predicted": "Mean predicted",
            "mean_actual": "Mean actual",
            "ratio": "Predicted / actual",
        }
    )
    interval = final["interval"]
    coverage_years = ", ".join(f"{y}: {pct(c, 0)}" for y, c in interval["coverage_by_year"].items())
    metrics = pd.DataFrame(
        [
            {
                "Metric": "RMSE, log scale (RMSLE)",
                "Final model": f"{test['rmsle']:.4f}",
                "Adjuster estimate": f"{adjuster['rmsle']:.4f}",
            },
            {
                "Metric": "R², log scale",
                "Final model": f"{test['r2_log']:.3f}",
                "Adjuster estimate": f"{adjuster['r2_log']:.3f}",
            },
            {
                "Metric": "MAE, expected (mean) prediction",
                "Final model": money(test["mae"]),
                "Adjuster estimate": money(adjuster["mae"]),
            },
            {
                "Metric": "MAE, typical (median-type) prediction",
                "Final model": money(test["mae_typical"]),
                "Adjuster estimate": money(adjuster["mae_typical"]),
            },
            {
                "Metric": "RMSE (currency)",
                "Final model": money(test["rmse"]),
                "Adjuster estimate": money(adjuster["rmse"]),
            },
            {
                "Metric": "Total predicted / total actual",
                "Final model": f"{test['bias_ratio']:.3f}",
                "Adjuster estimate": f"{adjuster['bias_ratio']:.3f}",
            },
        ]
    )
    trend = final["trend_annual_growth"]
    trend_text = f" The fitted cost trend is {pct(trend)} a year." if trend is not None else ""
    return [
        "## 6. Final model on the test set",
        "",
        f"**{final['name']}** with {final['n_features']} input columns. Expected (mean) costs "
        f"use a smearing factor of {final['smearing']:.3f}, estimated from out-of-fold "
        "residuals within the training period: exponentiating a log-scale prediction gives "
        "roughly the median, which is below the mean for skewed costs, and reserves need the "
        "mean." + trend_text,
        "",
        markdown_table(metrics),
        "",
        figure("regression_calibration_deciles", "Calibration by decile"),
        "",
        markdown_table(deciles_table),
        "",
        f"**Takeaway.** Ranked from cheapest to dearest predicted claims, mean actual cost "
        f"{rise_text(deciles['mean_actual'])}, and the predicted/actual ratio runs from "
        f"{deciles['ratio'].min():.2f} to {deciles['ratio'].max():.2f}. Overall the model books "
        f"{test['bias_ratio']:.2f} of the true total cost, against "
        f"{adjuster['bias_ratio']:.2f} for the adjusters' initial estimates.",
        "",
        mae_text(test, adjuster),
        "",
        "**80% prediction interval** (LightGBM quantile models at 10% and 90% on the same "
        f"target). Test coverage is **{pct(interval['coverage'])}** against the nominal "
        f"{pct(interval['nominal_coverage'], 0)}; {pct(interval['below_interval'])} of claims "
        f"fall below the interval and {pct(interval['above_interval'])} above it. The typical "
        f"interval spans a factor of {interval['median_width_ratio']:.1f} from low to high. "
        f"Coverage by accident year: {coverage_years}.",
        "",
    ]


def error_table(records: list[dict[str, Any]], label: str, key: str) -> str:
    """Markdown table of grouped test errors."""
    frame = pd.DataFrame(records)
    return markdown_table(
        pd.DataFrame(
            {
                label: frame[key].astype(str).str.replace("_", " "),
                "Claims": frame["claims"].map(lambda n: f"{n:,}"),
                "Mean actual": frame["mean_actual"].map(money),
                "Mean predicted": frame["mean_expected"].map(money),
                "Predicted / actual": frame["bias_ratio"].map(lambda x: f"{x:.2f}"),
                "MAE": frame["mae"].map(money),
                "RMSLE": frame["rmsle"].map(lambda x: f"{x:.3f}"),
            }
        )
    )


def error_section(r: dict[str, Any]) -> list[str]:
    """Where the final model is accurate and where it misses."""
    errors = r["final"]["error_analysis"]
    sizes = pd.DataFrame(errors["size_group"])
    top = sizes.iloc[0]
    natures = pd.DataFrame(errors["injury_nature"])
    worst = natures.loc[natures["rmsle"].idxmax()]
    best = natures.loc[natures["rmsle"].idxmin()]
    return [
        "## 7. Error analysis (test set)",
        "",
        "### By size of the actual claim",
        "",
        error_table(errors["size_group"], "Actual-cost band", "size_group"),
        "",
        f"**Takeaway.** For the {top['size_group'].lower()} of claims by actual cost, the model "
        f"predicts on average {money(top['mean_expected'])} against an actual "
        f"{money(top['mean_actual'])} (ratio {top['bias_ratio']:.2f}). "
        + (
            "The very largest claims are under-predicted: much of what makes a claim huge "
            "(complications, litigation, long time off work) happens after it is reported. "
            "This is why the high-cost flag is judged on ranking and review capacity rather "
            "than on exact amounts."
            if top["bias_ratio"] < 1
            else "Even the largest claims are not under-predicted on average."
        ),
        "",
        "### By injury nature",
        "",
        error_table(errors["injury_nature"], "Injury nature", "injury_nature"),
        "",
        f"**Takeaway.** Log-scale error is lowest for {best['injury_nature'].replace('_', ' ')} "
        f"({best['rmsle']:.3f}) and highest for {worst['injury_nature'].replace('_', ' ')} "
        f"({worst['rmsle']:.3f}) among natures with enough claims.",
        "",
        "### By body part",
        "",
        error_table(errors["body_part"], "Body part", "body_part"),
        "",
        "### By weekly wage band",
        "",
        error_table(errors["wage_band"], "Weekly wage band", "wage_band"),
        "",
        "### By accident year",
        "",
        error_table(errors["accident_year"], "Accident year", "accident_year"),
        "",
    ]


def capacity_at(record: dict[str, Any], capacity: float) -> dict[str, Any]:
    """The capacity-table row for one review share."""
    return next(row for row in record["capacity"] if abs(row["capacity"] - capacity) < 1e-9)


def threshold_text(c: dict[str, Any]) -> str:
    """How the capacity-based threshold behaves on the test set."""
    review, at = c["review_capacity"], c["threshold"]["test"]
    text = (
        f"The flag threshold is the calibrated probability at which {pct(review, 0)} of the "
        f"latest training claims would be flagged: **{c['threshold']['probability']:.3f}**. On "
        f"the test set it flags {pct(at['flag_rate'])} of claims (precision "
        f"{pct(at['precision'])}, recall {pct(at['recall'])})."
    )
    if at["flag_rate"] > review:
        text += (
            f" That is more than {pct(review, 0)} because more test claims really are expensive; "
            "a team with fixed capacity would instead review the top-ranked claims each period, "
            "as in the capacity table above."
        )
    return text


def brier_reference(rate: float) -> float:
    """Brier score of always predicting the observed base rate (the no-skill reference)."""
    return rate * (1 - rate)


def calibration_text(c: dict[str, Any]) -> str:
    """Brier scores against the no-skill reference, and the level of the probabilities."""
    cal, rate = c["calibration"], c["base_rate"]["test"]
    text = (
        "**Calibration.** Raw scores are turned into probabilities by a logistic calibration "
        "fit on the latest training claims (out of time). "
    )
    if cal["inner_brier_raw"] is not None:
        text += (
            f"There the Brier score goes from {cal['inner_brier_raw']:.4f} (raw class-weighted "
            f"probabilities) to {cal['inner_brier_calibrated']:.4f}. "
        )
    text += (
        f"On the test set the calibrated Brier score is {cal['test_brier_calibrated']:.4f}, "
        f"against {brier_reference(rate):.4f} for always predicting the test base rate. "
        f"Calibrated probabilities average {pct(cal['test_mean_probability'])} on the test set "
        f"against an observed high-cost rate of {pct(rate)}"
    )
    if cal["test_mean_probability"] < 0.8 * rate:
        text += (
            ": they are calibrated to the training period and the share of expensive claims has "
            "since risen, so in production the calibration would be refreshed as claims close."
        )
    else:
        text += (
            ", so the calibration carried over to the more expensive test period (the scores "
            "rise with the price level through the cost-related inputs)."
        )
    return text


def folds_won(c: dict[str, Any], winner: str, other: str) -> int:
    """Number of CV folds in which ``winner`` has a higher PR-AUC than ``other``."""
    pairs = zip(c["cv"][winner]["folds"], c["cv"][other]["folds"], strict=True)
    return sum(a["pr_auc"] > b["pr_auc"] for a, b in pairs)


def choice_text(c: dict[str, Any]) -> str:
    """Why the flag uses the chosen scorer."""
    chosen = c["chosen"]
    folds = len(c["cv"][chosen]["folds"])
    if c["chosen_kind"] == "cost_rank":
        wins = ", ".join(
            f"{name} in {folds_won(c, chosen, name)} of {folds} folds"
            for name in c["cv"]
            if name not in (chosen, "Baseline: rank by initial estimate")
        )
        return (
            f"**Chosen: {chosen}.** Neither dedicated classifier beat ranking claims by the cost "
            f"model's own predicted cost in cross-validation (it beats {wins}). A classifier "
            "only sees whether a claim crossed the threshold, while the regression learns from "
            "the full cost, so the flag reuses the cost model's prediction and turns it into a "
            "probability with a logistic calibration: one model serves both the reserve and "
            "the flag."
        )
    return f"**Chosen: {chosen}** (highest cross-validated PR-AUC)."


def candidate_rows(results: dict[str, Any], review: float) -> list[dict[str, Any]]:
    """Table rows of PR-AUC, ROC-AUC and capacity recall for every candidate."""
    capacities = [row["capacity"] for row in next(iter(results.values()))["capacity"]]
    rows = []
    for name, r in results.items():
        row = {"Model": name, "PR-AUC": f"{r['pr_auc']:.3f}", "ROC-AUC": f"{r['roc_auc']:.3f}"}
        for capacity in capacities:
            row[f"Recall, top {pct(capacity, 0)}"] = pct(capacity_at(r, capacity)["recall"])
        row[f"Precision, top {pct(review, 0)}"] = pct(capacity_at(r, review)["precision"])
        rows.append(row)
    return rows


def relative_verdict(gap: float) -> str:
    """Honest wording for the recall gap between the flag and the initial-estimate ranking."""
    if gap <= 0:
        return "Here the model does not beat the adjuster's estimate."
    if gap < 0.02:
        return (
            "The gain is marginal: the adjuster's initial estimate already carries most of the "
            "signal for *ranking* claims by severity. The model's value is elsewhere: accurate "
            "cost levels and totals (sections 2 and 6), an uncertainty range, and drivers."
        )
    if gap < 0.05:
        return "A modest gain over the adjuster's own estimate."
    return "A clear gain: the model finds severe claims that the first estimate misses."


def relative_text(c: dict[str, Any]) -> str:
    """Compare the chosen scorer with the initial-estimate baseline on the relative label."""
    review = c["review_capacity"]
    relative = c["relative_label"]
    chosen, baseline = relative[c["chosen"]], relative["Baseline: rank by initial estimate"]
    chosen_at, baseline_at = capacity_at(chosen, review), capacity_at(baseline, review)
    gap = chosen_at["recall"] - baseline_at["recall"]
    return (
        f"**Takeaway.** On the relative label (base rate {pct(chosen['base_rate'])}), reviewing "
        f"the top {pct(review, 0)} of claims catches {pct(chosen_at['recall'])} of the relatively "
        f"severe claims with the chosen flag, against {pct(baseline_at['recall'])} when ranking "
        f"by the adjuster's initial estimate (PR-AUC {chosen['pr_auc']:.3f} vs "
        f"{baseline['pr_auc']:.3f}). " + relative_verdict(gap)
    )


def classifier_section(c: dict[str, Any]) -> list[str]:
    """Phase 6: the high-cost claim flag."""
    review = c["review_capacity"]
    cv_rows = [
        {
            "Model": name,
            "CV PR-AUC": f"{r['pr_auc']:.3f} ± {r['pr_auc_std']:.3f}",
            "CV ROC-AUC": f"{r['roc_auc']:.3f}",
        }
        for name, r in c["cv"].items()
    ]
    chosen = capacity_at(c["test"][c["chosen"]], review)
    baseline = capacity_at(c["test"]["Baseline: rank by initial estimate"], review)
    ceiling = min(1.0, review / c["base_rate"]["test"])
    spread = max(r["pr_auc"] for r in c["test"].values()) - min(
        r["pr_auc"] for r in c["test"].values()
    )
    return [
        CLASSIFIER_TITLE,
        "",
        f"A claim is **high-cost** if its ultimate cost is at least {money(c['cost_threshold'])}, "
        f"the {ordinal(c['label_quantile'])} percentile of training claims. "
        f"{pct(c['base_rate']['train'])} of training claims qualify but "
        f"**{pct(c['base_rate']['test'])} of test claims** do, because costs rose (section 1). "
        "Candidates are compared by cross-validated PR-AUC, which focuses on the rare positive "
        "class; ROC-AUC is shown for reference.",
        "",
        markdown_table(pd.DataFrame(cv_rows)),
        "",
        choice_text(c),
        "",
        "### Test set, fixed threshold: ranking and review capacity",
        "",
        "*Recall, top X%* is the share of all high-cost claims caught when the team reviews "
        "only the X% of claims the model ranks highest.",
        "",
        figure("classifier_pr_curves", "Precision-recall curves"),
        "",
        markdown_table(pd.DataFrame(candidate_rows(c["test"], review))),
        "",
        f"**Takeaway.** Reviewing the top {pct(review, 0)} of claims, the chosen flag catches "
        f"{pct(chosen['recall'])} of all high-cost claims, against {pct(baseline['recall'])} when "
        f"ranking by the adjuster's initial estimate, and {pct(chosen['precision'])} of the "
        f"reviewed claims are genuinely high-cost. With {pct(c['base_rate']['test'])} of test "
        f"claims high-cost, reviewing {pct(review, 0)} can catch at most {pct(ceiling)} of them"
        + (
            f", and every method's PR-AUC lies within {spread:.3f} of the others: after the price "
            "rise so many claims cross the fixed threshold that almost any ranking finds them. "
            "The relative label below is the sharper test."
            if spread < 0.03
            else "."
        ),
        "",
        "### Test set, relative label: top 10% of cost within each accident year",
        "",
        "Every candidate refit (where it learns) on the relative label. This removes the "
        "base-rate shift and asks the real triage question: which claims are unusually severe "
        "for their time?",
        "",
        markdown_table(pd.DataFrame(candidate_rows(c["relative_label"], review))),
        "",
        relative_text(c),
        "",
        "### Threshold and calibration",
        "",
        threshold_text(c),
        "",
        figure("classifier_reliability", "Reliability curves"),
        "",
        calibration_text(c),
        "",
    ]


def all_models_section(r: dict[str, Any]) -> list[str]:
    """Every distinct model evaluated, in one table (duplicates across stages removed)."""
    seen, records = set(), []
    for record in r["candidates"]:
        if record["model_key"] not in seen:
            seen.add(record["model_key"])
            records.append(record)
    return [
        "## Appendix: every model evaluated",
        "",
        "One row per distinct model (a configuration that appears in several stages is listed "
        "once, under its first name).",
        "",
        comparison_table(records),
        "",
    ]


def build_results(cfg: Config) -> None:
    """Write ``reports/results.md`` from whatever metric files exist."""
    validation = load_metrics(cfg, "validation_metrics")
    regression = load_metrics(cfg, "regression_metrics")
    blocks = ["# Results", "", GENERATED_NOTE.format(stage="report"), ""]
    if validation is None or regression is None:
        blocks += [NOT_RUN.format(stage="train_reg"), ""]
    else:
        blocks += validation_section(validation)
        blocks += comparison_section(regression)
        blocks += drift_section(regression, load_metrics(cfg, "eda_metrics"))
        blocks += ablation_section(regression)
        blocks += tuning_section(load_metrics(cfg, "best_params"))
        blocks += final_section(regression)
        blocks += error_section(regression)
    classifier = load_metrics(cfg, "classifier_metrics")
    if classifier is None:
        blocks += [CLASSIFIER_TITLE, "", NOT_RUN.format(stage="train_clf"), ""]
    else:
        blocks += classifier_section(classifier)
    if regression is not None:
        blocks += all_models_section(regression)
    write_text("\n".join(blocks), cfg.path("results_report"))
