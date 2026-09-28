"""Phase 8: model card, resume / interview notes and README results, built from metrics.

Every number in these documents is read from ``reports/metrics/*.json``; nothing is typed by
hand, so re-running the pipeline keeps the documents consistent with the results. The README
keeps its hand-written narrative and only the blocks between ``<!-- BEGIN GENERATED: x -->``
and ``<!-- END GENERATED: x -->`` markers are rewritten.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import Config
from src.reporting.build_results import (
    build_results,
    capacity_at,
    load_metrics,
    money,
    ordinal,
    pct,
    relative_verdict,
)
from src.reporting.outputs import GENERATED_NOTE, markdown_table, write_text

logger = logging.getLogger(__name__)

ADJUSTER = "Baseline B: adjuster's initial estimate"
INITIAL_RANK = "Baseline: rank by initial estimate"


@dataclass
class Facts:
    """The headline numbers every document draws on."""

    quality: dict[str, Any]
    eda: dict[str, Any]
    features: dict[str, Any]
    validation: dict[str, Any]
    regression: dict[str, Any]
    classifier: dict[str, Any]
    explain: dict[str, Any]
    fairness: dict[str, Any]

    @property
    def final(self) -> dict[str, Any]:
        """Final cost model record."""
        return self.regression["final"]

    def candidate(self, name: str) -> dict[str, Any]:
        """A regression candidate record by name."""
        return next(c for c in self.regression["candidates"] if c["name"] == name)

    @property
    def adjuster(self) -> dict[str, Any]:
        """Test metrics of the adjuster's initial estimate."""
        return self.candidate(ADJUSTER)["test"]

    @property
    def rmsle_gain(self) -> float:
        """Relative reduction in test RMSLE versus the adjuster's initial estimate."""
        return 1 - self.final["test"]["rmsle"] / self.adjuster["rmsle"]

    @property
    def text_gain(self) -> float:
        """Relative reduction in CV RMSLE from no text features to the chosen feature set."""
        names = self.regression["groups"]["ablation"]
        none = self.candidate(names[0])["cv"]["rmsle"]
        chosen = self.candidate(self.regression["choices"]["best_layers"])["cv"]["rmsle"]
        return 1 - chosen / none

    def capacity(self, model: str, share: float, label: str = "test") -> dict[str, Any]:
        """Capacity-table row of a classifier candidate on the test set.

        Args:
            model: Candidate name.
            share: Review capacity, e.g. 0.1.
            label: ``test`` (fixed threshold) or ``relative_label`` (within-year top 10%).
        """
        return capacity_at(self.classifier[label][model], share)


def load_facts(cfg: Config) -> Facts | None:
    """All metrics files, or ``None`` if any stage has not been run yet."""
    keys = {
        "quality": "data_quality_metrics",
        "eda": "eda_metrics",
        "features": "features_metrics",
        "validation": "validation_metrics",
        "regression": "regression_metrics",
        "classifier": "classifier_metrics",
        "explain": "explainability_metrics",
        "fairness": "fairness_metrics",
    }
    loaded = {name: load_metrics(cfg, key) for name, key in keys.items()}
    missing = [name for name, value in loaded.items() if value is None]
    if missing:
        logger.warning("Documents not built; missing metrics: %s", missing)
        return None
    return Facts(**loaded)


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------


def results_table(f: Facts) -> str:
    """Headline test-set comparison: final model vs the adjuster and the median baseline."""
    median = f.candidate("Baseline A: training median")["test"]
    rows = []
    for label, test in (
        (f"Final model ({f.final['name']})", f.final["test"]),
        ("Adjuster's initial estimate", f.adjuster),
        ("Training median", median),
    ):
        rows.append(
            {
                "Model (test set, most recent 20% of claims)": label,
                "RMSLE": f"{test['rmsle']:.3f}",
                "R² (log)": f"{test['r2_log']:.3f}",
                "MAE (typical prediction)": money(test["mae_typical"]),
                "Total predicted / actual": f"{test['bias_ratio']:.2f}",
            }
        )
    return markdown_table(pd.DataFrame(rows))


def classifier_table(f: Facts) -> str:
    """Capacity-based recall of the flag vs ranking by the initial estimate, on both labels."""
    c = f.classifier
    shares = [row["capacity"] for row in c["test"][c["chosen"]]["capacity"]]
    labels = (("Fixed threshold", "test"), ("Within-year top 10%", "relative_label"))
    rows = []
    for label_name, label in labels:
        for name, model in (
            ("High-cost flag", c["chosen"]),
            ("Rank by initial estimate", INITIAL_RANK),
        ):
            row = {
                "Test label": label_name,
                "Method": name,
                "PR-AUC": f"{c[label][model]['pr_auc']:.3f}",
            }
            for share in shares:
                row[f"Recall, review top {pct(share, 0)}"] = pct(
                    f.capacity(model, share, label)["recall"]
                )
            rows.append(row)
    return markdown_table(pd.DataFrame(rows))


def trend_insight(eda: dict[str, Any], validation: dict[str, Any]) -> str:
    """The cost-trend insight, worded from the measured mix share."""
    mix = eda["mix_share_of_growth"]
    kind = "a price-level change" if mix < 0.5 else "mostly a change in claim mix"
    effect = "barely changes" if abs(mix) < 0.05 else "only partly explains"
    return (
        f"**Costs rise {eda['yearly_median_growth']:.1f}x over the period, as {kind}.** "
        f"Holding the claim mix fixed {effect} the growth, and "
        f"{pct(validation['test']['share_above_train_high_cost_threshold'], 0)} of the most recent "
        "claims exceed the historical high-cost threshold."
    )


def key_insights(f: Facts, cfg: Config) -> list[str]:
    """Numbered headline insights across the project."""
    eda, v = f.eda, f.validation
    tolerance = cfg["eda"]["adjuster_tolerance"]
    model_line = (
        f"**The model {'beats' if f.rmsle_gain > 0 else 'does not beat'} the adjusters' "
        f"estimate.** Test RMSE on the log scale is {f.final['test']['rmsle']:.3f} against "
        f"{f.adjuster['rmsle']:.3f} ({pct(abs(f.rmsle_gain), 0)} "
        f"{'lower' if f.rmsle_gain > 0 else 'higher'}), and total predicted cost is "
        f"{f.final['test']['bias_ratio']:.2f} of actual against "
        f"{f.adjuster['bias_ratio']:.2f} for the initial estimates."
    )
    text_line = (
        f"**The claim description is worth reading.** Text features cut cross-validated error "
        f"by {pct(f.text_gain, 0)} relative to structured fields alone."
        if f.text_gain > 0
        else "**Text features did not improve cross-validated error.**"
    )
    return [
        f"**Adjusters' first estimates are rough.** The typical initial estimate is off by a "
        f"factor of {eda['adjuster']['typical_factor_off']:.2f} and only "
        f"{pct(eda['adjuster']['share_within_tolerance'], 0)} land within ±{pct(tolerance, 0)} of "
        "the final cost.",
        trend_insight(eda, v),
        model_line,
        text_line,
        *[line for line in f.explain["insights"] if "claim description" not in line][:2],
    ]


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def calibration_limitation(f: Facts) -> str:
    """Whether the flag's probabilities held up on the more expensive test period."""
    cal, rate = f.classifier["calibration"], f.classifier["base_rate"]["test"]
    mean = cal["test_mean_probability"]
    if abs(mean - rate) <= 0.2 * rate:
        return (
            f"The flag's probabilities held up on the test period (mean {pct(mean)} vs an observed "
            f"{pct(rate)}), but calibration must be refreshed as prices move."
        )
    return (
        f"The flag's probabilities are calibrated to the training period (test mean {pct(mean)} "
        f"vs an observed {pct(rate)}) and need refreshing as prices move."
    )


def text_layers_used(f: Facts) -> str:
    """Plain-language list of the text layers in the final model."""
    names = {
        "rules": "keyword taxonomy",
        "tfidf_svd": "TF-IDF",
        "embedding_clusters": "sentence-embedding clusters",
    }
    layers = f.final["description"]["layers"]
    used = [label for key, label in names.items() if layers.get(key)]
    return ", ".join(used) if used else "none"


def dropped_layers(f: Facts) -> list[str]:
    """Text layers evaluated in the ablation but not used by the final model."""
    names = {
        "tfidf_svd": "TF-IDF",
        "embedding_clusters": "sentence-embedding clusters",
        "rules": "keyword taxonomy",
    }
    layers = f.final["description"]["layers"]
    return [label for key, label in names.items() if not layers.get(key, True)]


def model_card(f: Facts, cfg: Config) -> str:
    """Markdown model card."""
    v, c, final = f.validation, f.classifier, f.final
    comparison = f.fairness["gender_comparison"]
    shipped = "excludes" if cfg["fairness"]["exclude_gender"] else "includes"
    genders = pd.DataFrame(f.fairness["gender"])
    interval = final["interval"]
    top = pd.DataFrame(final["error_analysis"]["size_group"]).iloc[0]
    review = c["review_capacity"]
    chosen_at = f.capacity(c["chosen"], review)
    return "\n".join(
        [
            "# Model card: workers' compensation claim cost model and high-cost flag",
            "",
            GENERATED_NOTE.format(stage="report"),
            "",
            "## Model details",
            "",
            f"- **Cost model:** {final['name']}; target approach "
            f"*{final['description']['approach']}*; {final['n_features']} input columns built from "
            "structured claim fields and the free-text description. Predicts the expected "
            "(mean) ultimate cost, a typical (median-type) cost and an 80% interval "
            "(LightGBM quantile models).",
            f"- **High-cost flag:** {c['chosen']}, calibrated with logistic (Platt) scaling; flags "
            f"claims likely to reach {money(c['cost_threshold'])} or more (the "
            f"{ordinal(c['label_quantile'])} percentile of training claims).",
            f"- **Training data:** {v['train']['claims']:,} claims with accidents from "
            f"{v['final']['fit']['start']} to {v['final']['fit']['end']}. Code, configuration and "
            "every result are reproducible with `python -m src.pipeline --stage all` (fixed "
            f"random seed {cfg.random_seed}).",
            "",
            "## Intended use",
            "",
            "- Decision support **at the moment a claim is reported**: a suggested initial "
            "reserve with an uncertainty range, and an early-escalation flag for claims likely "
            "to become expensive, for a claims adjuster to review.",
            "- Portfolio-level checks for reserving (does the total of reserves track expected "
            "cost?) and driver insights for underwriting and risk & safety teams.",
            "",
            "## Out-of-scope uses",
            "",
            "- Automatic denial, reduction or approval of benefits, or any decision about an "
            "individual without human review.",
            "- Pricing an individual worker or employer, or screening job applicants.",
            "- Use on other jurisdictions, lines of business or real claims without revalidation: "
            "the model was trained on synthetic data.",
            "- Causal conclusions: SHAP explanations describe what the model uses, not what causes "
            "costs.",
            "",
            "## Training data",
            "",
            f"Kaggle *Actuarial Loss Estimation* `train.csv`: {f.quality['raw_rows']:,} synthetic "
            f"workers' compensation claims, {f.quality['clean_rows']:,} after cleaning "
            f"({f.quality['raw_rows'] - f.quality['clean_rows']:,} rows dropped; invalid hours, "
            "days and wages set to missing and imputed inside the model pipeline). Known "
            "artifacts: suspected default wage values, accident times of 00:00 meaning "
            "'unknown', and a placeholder initial-estimate value. See "
            "[data_quality_report.md](data_quality_report.md).",
            "",
            "## Evaluation",
            "",
            f"Time-ordered: the most recent {v['test']['claims']:,} claims (from "
            f"{v['final']['eval']['start']}) are a test set used once; every choice was made with "
            f"{v['cv_folds']}-fold time-ordered cross-validation inside the training period.",
            "",
            results_table(f),
            "",
            f"- 80% interval coverage on the test set: {pct(interval['coverage'])}.",
            f"- High-cost flag: reviewing the top {pct(review, 0)} of test claims catches "
            f"{pct(chosen_at['recall'])} of high-cost claims with precision "
            f"{pct(chosen_at['precision'])}; the test base rate is {pct(c['base_rate']['test'])} "
            f"versus {pct(c['base_rate']['train'])} in training, so that is close to the maximum "
            "possible. On the within-year label (top 10% of cost in each accident year) it "
            f"catches {pct(f.capacity(c['chosen'], review, 'relative_label')['recall'])} against "
            f"{pct(f.capacity(INITIAL_RANK, review, 'relative_label')['recall'])} for ranking by "
            "the initial estimate.",
            "",
            "Full results: [results.md](results.md).",
            "",
            "## Fairness",
            "",
            "Test-set error by gender (groups with enough claims):",
            "",
            markdown_table(
                pd.DataFrame(
                    {
                        "Gender": genders["gender"],
                        "Claims": genders["claims"].map(lambda n: f"{n:,}"),
                        "Predicted / actual": genders["bias_ratio"].map(lambda x: f"{x:.2f}"),
                        "RMSLE": genders["rmsle"].map(lambda x: f"{x:.3f}"),
                        "True high-cost rate": genders["high_cost_rate"].map(pct),
                        "Flagged at capacity": genders["capacity_flag_rate"].map(pct),
                    }
                )
            ),
            "",
            f"Removing Gender changes cross-validated RMSLE by "
            f"{comparison['cv_rmsle_change_without_gender']:+.4f}. The shipped model {shipped} "
            "Gender. Details and the age-band analysis: [explainability.md](explainability.md).",
            "",
            "## Limitations",
            "",
            "- **Synthetic data.** Relationships in this data (for example the relative costs "
            "of injury types) may not hold in real claims.",
            f"- **The largest claims are hard to foresee.** For the {top['size_group'].lower()} "
            f"of test claims the model predicts {top['bias_ratio']:.2f} of their actual cost: "
            "most of what makes a claim huge happens after it is reported.",
            f"- **Drift.** Costs rose {f.eda['yearly_median_growth']:.1f}x over the data period; "
            "the model reflects the price level of its training data. " + calibration_limitation(f),
            "- **Text rules are hand-built.** The keyword taxonomy leaves "
            f"{pct(1 - f.features['coverage_injury_cause'], 0)} of descriptions without a cause "
            "label, and misspellings or new jargon will be missed.",
            "",
            "## Ethical considerations",
            "",
            "- Predictions affect injured workers indirectly (through reserves and claim "
            "handling). Keep a human in the loop, and never use the flag to delay or deny care.",
            "- Protected characteristics (gender, age) are monitored group by group; gender is "
            "only used if it adds measurable accuracy, and even then not for individual decisions.",
            "- Each prediction comes with its main drivers, so an adjuster can challenge it.",
            "",
            "## Monitoring and maintenance",
            "",
            "- Track predicted vs actual cost as claims close, by accident quarter, and the share "
            "of claims crossing the high-cost threshold; alert on sustained bias.",
            "- Monitor input drift (wage and initial-estimate distributions, description "
            "vocabulary) and the taxonomy's 'other' rate.",
            "- Retrain at least yearly (and after large price changes), refresh the probability "
            "calibration more often, and re-run the fairness checks at every retrain.",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Resume bullets and interview preparation
# ---------------------------------------------------------------------------


def largest_claims_clause(f: Facts) -> str:
    """How well the largest claims are predicted, as a clause for the limitations answer."""
    top = f.final["error_analysis"]["size_group"][0]
    if top["bias_ratio"] < 1:
        return (
            f"the {top['size_group'].lower()} of claims are under-predicted (the model predicts "
            f"{top['bias_ratio']:.2f} of their actual cost) because what makes them huge happens "
            "after reporting"
        )
    return f"even the {top['size_group'].lower()} of claims are not under-predicted on average"


def flag_bullet(f: Facts, cfg: Config) -> str:
    """Resume bullet on the high-cost flag, SHAP, fairness and the app."""
    c = f.classifier
    comparison = f.fairness["gender_comparison"]
    lead = (
        "- Showed that a dedicated high-cost classifier added no lift over the cost model's own "
        "ranking, so one calibrated model powers both reserving and triage"
        if c["chosen_kind"] == "cost_rank"
        else f"- Built a calibrated high-cost claim flag ({c['chosen']})"
    )
    brier = c["calibration"]["test_brier_calibrated"]
    reference = c["base_rate"]["test"] * (1 - c["base_rate"]["test"])
    gender = (
        "excluded gender after showing it adds no accuracy"
        if comparison["negligible"] and cfg["fairness"]["exclude_gender"]
        else "checked fairness across gender and age bands"
    )
    return (
        f"{lead} (test Brier score {brier:.3f} vs {reference:.3f} for a no-skill forecast); "
        f"added SHAP explanations, fairness checks ({gender}) and a Streamlit triage app."
    )


def interview_prep(f: Facts, cfg: Config) -> str:
    """Resume bullets and likely interview questions, with answers built from the metrics."""
    v, c, final, eda = f.validation, f.classifier, f.final, f.eda
    review = c["review_capacity"]
    rel_chosen = f.capacity(c["chosen"], review, "relative_label")
    rel_initial = f.capacity(INITIAL_RANK, review, "relative_label")
    chosen_at = f.capacity(c["chosen"], review)
    ceiling = min(1.0, review / c["base_rate"]["test"])
    drift = [f.candidate(name) for name in f.regression["groups"]["drift"]]
    by_cv = min(drift, key=lambda d: d["cv"]["rmsle"])
    by_test = min(drift, key=lambda d: d["test"]["rmsle"])
    comparison = f.fairness["gender_comparison"]
    importance = list(f.explain["importance"].items())
    tuning = load_metrics(cfg, "best_params")
    return "\n".join(
        [
            "# Resume and interview preparation",
            "",
            GENERATED_NOTE.format(stage="report"),
            "",
            "Every number below comes from `reports/metrics/`; re-running the pipeline updates it.",
            "",
            "## Resume bullets",
            "",
            f"- Built an end-to-end workers' compensation claim cost model (LightGBM with NLP on "
            f"claim descriptions) on {f.quality['raw_rows']:,} claims; on a time-based holdout it "
            f"cut log-scale error by {pct(f.rmsle_gain, 0)} versus adjusters' initial reserves "
            f"and predicted {final['test']['bias_ratio']:.2f} of total cost versus "
            f"{f.adjuster['bias_ratio']:.2f} for the initial estimates.",
            f"- Designed leakage-free, time-ordered validation that exposed a "
            f"{eda['yearly_median_growth']:.1f}x cost trend, compared four drift-handling "
            f"strategies, and showed through ablation that NLP on the claim description "
            f"({text_layers_used(f)}) reduces error by {pct(f.text_gain, 0)}.",
            flag_bullet(f, cfg),
            "",
            "## Likely interview questions",
            "",
            "**1. Why a time-based split instead of a random one?**",
            "",
            f"Because the model will always predict future claims. A random split would put claims "
            f"from the test period into training. Costs grew {eda['yearly_median_growth']:.1f}x "
            f"across the data, and the test period's median cost is "
            f"{money(v['test']['median_cost'])} against {money(v['train']['median_cost'])} in "
            "training, so a random split would give flattering results that would not hold in "
            f"production. Model selection used {v['cv_folds']} expanding-window folds inside the "
            "training period, and the test set was scored once.",
            "",
            "**2. How did you handle the skewed target?**",
            "",
            "I trained on log cost, which makes errors relative so a few huge claims do not "
            "dominate, and selected models by RMSE on the log scale. Exponentiating a log "
            "prediction gives roughly the median, so for reserves I applied Duan's smearing "
            f"factor ({final['smearing']:.3f}), estimated from out-of-fold residuals so that "
            "boosting's small in-sample residuals do not understate it. I also tried Tweedie "
            "and Gamma losses on the raw cost.",
            "",
            "**3. Costs rise over time. How can a model trained on old claims be trusted?**",
            "",
            f"I compared four approaches with cross-validation: log cost, the ratio to the "
            f"adjuster's estimate (which is recorded at current prices), and trend-adjusted "
            f"versions of both. Cross-validation chose **{by_cv['name']}**"
            + (
                "; the test set agreed."
                if by_cv["name"] == by_test["name"]
                else f"; on the test set {by_test['name']} did slightly better, which shows the "
                "future can drift differently from the past, but I kept the choice made without "
                "looking at the test set."
            )
            + " In production I would monitor predicted vs actual by accident quarter, "
            "recalibrate as claims close and retrain at least yearly.",
            "",
            "**4. What did the NLP add?**",
            "",
            f"In an ablation on the same model, the text features reduced cross-validated RMSLE by "
            f"{pct(f.text_gain, 0)}. The keyword taxonomy gives interpretable labels (body part, "
            f"nature, cause; {pct(f.features['coverage_injury_nature'], 0)} of descriptions get a "
            "nature label); TF-IDF with SVD captures wording beyond the keywords"
            + (
                f". The {' and '.join(dropped_layers(f))} did not improve cross-validation by more "
                "than the pre-set tolerance, so the final model leaves them out: a result, not "
                "a failure."
                if dropped_layers(f)
                else "; the sentence-embedding clusters also earned their place."
            ),
            "",
            "**5. How did you avoid leakage?**",
            "",
            "Every learned step (imputation medians, scaling, one-hot levels, TF-IDF, SVD, KMeans) "
            "is inside one scikit-learn pipeline fit on training rows only, and refit in each "
            "fold. Cleaning rules use fixed thresholds. The high-cost threshold comes from the "
            "training target only. Tests check that the target is never an input and that the "
            "fitted steps only saw training rows.",
            "",
            "**6. How do you choose the classifier threshold?**",
            "",
            f"From review capacity: the probability at which {pct(review, 0)} of the latest "
            f"training claims would be flagged ({c['threshold']['probability']:.3f}). On the test "
            f"set that flags {pct(c['threshold']['test']['flag_rate'])} of claims, because more "
            "claims really are expensive after the price rise; a team with fixed capacity would "
            "simply review the top-ranked claims.",
            "",
            f"**7. Why is recall at {pct(review, 0)} capacity only {pct(chosen_at['recall'])}?**",
            "",
            f"Because {pct(c['base_rate']['test'])} of test claims are high-cost against "
            f"{pct(c['base_rate']['train'])} in training. Reviewing {pct(review, 0)} of claims can "
            f"catch at most {pct(ceiling)} of them. The precision at that capacity is "
            f"{pct(chosen_at['precision'])}, and with a label defined within each accident year "
            f"reviewing the top {pct(review, 0)} catches {pct(rel_chosen['recall'])} of claims "
            f"that are unusually severe for their year (vs {pct(rel_initial['recall'])} ranking by "
            "the initial estimate). "
            + relative_verdict(rel_chosen["recall"] - rel_initial["recall"]).replace("*", ""),
            "",
            "**8. What drives the predictions?**",
            "",
            "By SHAP (one-hot columns summed per concept), the top drivers are "
            + ", ".join(name for name, _ in importance[:4])
            + ". SHAP shows what the model relies on, not causes, and each prediction in the app "
            "comes with its own drivers.",
            "",
            "**9. Did you use gender, and is the model fair?**",
            "",
            f"I compared the model with and without gender: removing it changes cross-validated "
            f"RMSLE by {comparison['cv_rmsle_change_without_gender']:+.4f}. "
            + (
                "That is negligible, so the recommendation is to exclude it."
                if comparison["negligible"]
                else "That is a real difference, so whether to use it is a policy decision."
            )
            + " I also compare error and flag rates across genders and age bands on the test set "
            "and would re-run those checks at every retrain.",
            "",
            "**10. What are the main limitations?**",
            "",
            "The data is synthetic; "
            + largest_claims_clause(f)
            + "; and costs drift, so the model and its calibration need regular refreshing. "
            "Interval coverage on the test set is "
            f"{pct(final['interval']['coverage'])} "
            f"against a nominal {pct(final['interval']['nominal_coverage'], 0)}."
            + (
                f" Tuning with Optuna ({tuning['n_trials']} trials) moved cross-validated RMSLE "
                f"from {tuning['untuned_cv_rmsle']:.4f} to {tuning['best_cv_rmsle']:.4f}."
                if tuning
                else ""
            ),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# README blocks
# ---------------------------------------------------------------------------


def readme_blocks(f: Facts, cfg: Config) -> dict[str, str]:
    """Generated README sections, keyed by marker name."""
    return {
        "results": "\n".join(["", results_table(f), "", classifier_table(f), ""]),
        "insights": "\n".join(
            ["", *[f"{i}. {line}" for i, line in enumerate(key_insights(f, cfg), 1)], ""]
        ),
    }


def replace_block(text: str, name: str, content: str) -> str:
    """Replace the content between a README's BEGIN/END GENERATED markers."""
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED: {name} -->)(.*?)(<!-- END GENERATED: {name} -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"README has no generated block named {name!r}")
    return pattern.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(3), text)


def update_readme(f: Facts, cfg: Config) -> None:
    """Rewrite the generated blocks of README.md in place."""
    path = cfg.path("readme")
    text = path.read_text(encoding="utf-8")
    for name, content in readme_blocks(f, cfg).items():
        text = replace_block(text, name, content)
    write_text(text, path)


def build_documents(cfg: Config) -> None:
    """Stage ``report``: results, model card, interview notes and README results."""
    build_results(cfg)
    facts = load_facts(cfg)
    if facts is None:
        return
    write_text(model_card(facts, cfg), cfg.path("model_card"))
    write_text(interview_prep(facts, cfg), cfg.path("interview_prep"))
    update_readme(facts, cfg)
