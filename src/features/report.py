"""Builds reports/features.md: feature catalogue, taxonomy coverage and the text layers.

Everything is computed on training claims only (accidents before the time-split cutoff):
the choice of k for the embedding clusters, the evidence behind the placeholder-estimate
flag, and every coverage figure. Embeddings are computed for all descriptions (the model is
pretrained and learns nothing from our data), so later stages can reuse the cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.analysis.summaries import CLAIMS, MEDIAN, SHARE, group_summary, readable
from src.config import Config
from src.data.load import load_clean
from src.features.build import (
    build_imputer,
    build_preprocessor,
    feature_names,
    model_input_columns,
)
from src.features.tabular import engineer_features
from src.features.text_embed import embeddings_for, name_clusters, tfidf_svd
from src.features.text_rules import TAXONOMY_FIELDS, coverage
from src.models.split import train_test_split_by_time
from src.reporting.outputs import (
    GENERATED_NOTE,
    figure_markdown,
    markdown_table,
    percent_columns,
    write_json,
    write_text,
)
from src.reporting.plots import hbar, new_figure, save_figure, series_color

logger = logging.getLogger(__name__)

CLUSTER = "embedding_cluster"

# Plain-language definition and rationale for every model input (logical names).
# Values in braces are filled from the config when the report is written.
FEATURE_DOCS: dict[str, tuple[str, str]] = {
    "age": (
        "Claimant age at the accident.",
        "Recovery tends to take longer with age; EDA section 4 shows the pattern here.",
    ),
    "dependent_children": ("Number of dependent children.", "Dependants can raise benefits."),
    "dependents_other": ("Number of other dependants.", "As above; recorded separately."),
    "total_dependents": (
        "Children plus other dependants.",
        "One measure of family obligations; the raw counts are sparse.",
    ),
    "hours_per_week": (
        "Weekly hours (invalid values imputed).",
        "Exposure to risk and the base for wage replacement.",
    ),
    "days_per_week": ("Days worked per week (invalid values imputed).", "Shift pattern."),
    "hours_per_day": ("Hours per week / days per week.", "Long shifts are linked with fatigue."),
    "accident_year": (
        "Calendar year of the accident.",
        "Carries the cost trend. Trees cannot extrapolate beyond the last training year; "
        "Phase 5 tests explicit ways to handle drift.",
    ),
    "accident_hour": (
        "Hour of the accident (0-23).",
        "Time of day relates to shift type and fatigue.",
    ),
    "log_weekly_wages": (
        "log(1 + weekly wage).",
        "Wage replacement scales with pay; the log tames the long right tail.",
    ),
    "log_initial_estimate": (
        "log(1 + adjuster's initial estimate).",
        "The adjuster's view of severity, recorded at current prices (EDA sections 1 and 5).",
    ),
    "report_delay_days": (
        "Days from accident to report (log-transformed in the model).",
        "Reporting delay is a known cost driver; EDA section 3 shows the pattern here.",
    ),
    "hourly_wage": (
        "Weekly wage / hours, hours clipped up to {min_hours:g} (log-transformed).",
        "Pay rate independent of hours; clipping avoids dividing by near-zero hours.",
    ),
    "initial_to_wage_ratio": (
        "Initial estimate / weekly wage (log-transformed).",
        "Roughly how many weeks of pay the adjuster reserved, i.e. expected time off work.",
    ),
    "is_weekend": (
        "Accident on a Saturday or Sunday.",
        "Weekend work often means different staffing and tasks.",
    ),
    "is_night_shift": (
        "Accident after {night_after}:00 or before {night_before:02d}:00, "
        "excluding times of exactly 00:00.",
        "Night work is linked with fatigue. 00:00 is excluded because it mostly means "
        "'time unknown'.",
    ),
    "accident_time_unknown": (
        "Accident time recorded as exactly 00:00.",
        "A recording artifact concentrated in recent years (data quality report).",
    ),
    "is_part_time": (
        "Claimant is part-time.",
        "Part-time workers differ in hours, pay and job type.",
    ),
    "wage_is_default": (
        "Weekly wage equals a suspected system default ({heaped}).",
        "These values track recording eras, not pay (data quality report).",
    ),
    "initial_is_default": (
        "Initial estimate equals a placeholder value ({initial_defaults}).",
        "These estimates are less accurate than others (evidence in section 2).",
    ),
    "initial_is_nominal": (
        "Initial estimate of {nominal} or less.",
        "A nominal placeholder rather than an estimate.",
    ),
    "weekly_wages_was_missing": (
        "Weekly wage was missing or invalid and has been imputed.",
        "Lets the model treat imputed values differently.",
    ),
    "hours_per_week_was_missing": (
        "Hours were missing or invalid and have been imputed.",
        "As above.",
    ),
    "days_per_week_was_missing": ("Days were invalid and have been imputed.", "As above."),
    "gender": (
        "Recorded gender (M, F, U).",
        "Kept for now; Phase 7 compares the model with and without it.",
    ),
    "marital_status": (
        "M, S or U; unknown kept as its own category.",
        "Unknown can itself carry signal.",
    ),
    "age_band": (
        "Age in bands (edges in config).",
        "Lets linear models capture a non-linear age effect.",
    ),
    "accident_month": ("Month of the accident.", "Seasonal patterns, e.g. outdoor work."),
    "accident_weekday": ("Day of the week of the accident.", "Staffing and task mix vary by day."),
    "body_part": (
        "Keyword taxonomy: injured body part, 'multiple' if several areas.",
        "Cost varies widely by body part (EDA section 2).",
    ),
    "injury_nature": (
        "Keyword taxonomy: nature of injury, most severe mention wins.",
        "Cost varies widely by nature of injury (EDA section 2).",
    ),
    "injury_cause": (
        "Keyword taxonomy: how the injury happened.",
        "Mechanism relates to severity (falls, vehicles, lifting).",
    ),
    "num_body_parts_mentioned": (
        "Distinct body parts mentioned (after the finger/hand rule).",
        "More injured areas usually means more treatment.",
    ),
    "description_length": (
        "Characters in the description.",
        "Longer descriptions can signal more complex injuries.",
    ),
}


def small_pct(value: float) -> str:
    """Percentage with one decimal, showing tiny non-zero shares as '<0.1%'."""
    return "<0.1%" if 0 < value < 0.001 else f"{value:.1%}"


def fill_doc(text: str, cfg: Config) -> str:
    """Fill config values into a documentation string."""
    f, flags = cfg["features"], cfg["cleaning"]["flags"]
    return text.format(
        min_hours=f["min_hours_for_rate"],
        night_after=f["night_after_hour"],
        night_before=f["night_before_hour"],
        heaped=", ".join(f"{v:g}" for v in flags["heaped_wage_values"]),
        initial_defaults=", ".join(f"{v:g}" for v in f["initial_default_values"]),
        nominal=flags["nominal_initial_estimate_max"],
    )


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


def catalogue(train_features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per model input: branch, definition, rationale and training statistics."""
    f = cfg["features"]
    branches = {
        "numeric": f["numeric"],
        "log numeric": f["log_numeric"],
        "binary": f["binary"],
        "categorical": f["categorical"],
        "text rules (L1)": f["text_rule_categorical"] + f["text_rule_numeric"],
    }
    imputed = build_imputer(cfg).fit_transform(train_features)  # adds the missing indicators
    rows = []
    for branch, names in branches.items():
        for name in names:
            column = cfg.resolve([name])[0]
            before = train_features[column] if column in train_features else imputed[column]
            definition, rationale = (fill_doc(text, cfg) for text in FEATURE_DOCS[name])
            rows.append(
                {
                    "Feature": name,
                    "Branch": branch,
                    "Definition": definition,
                    "Why": rationale,
                    "Missing before imputation": small_pct(before.isna().mean()),
                    "Share = 1 (flags)": (
                        small_pct(imputed[column].mean()) if branch == "binary" else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def catalogue_section(train_features: pd.DataFrame, cfg: Config) -> list[str]:
    """Markdown for the feature catalogue."""
    e, t = cfg["embeddings"], cfg["tfidf"]
    return [
        "## 1. Feature catalogue",
        "",
        "Every model input, grouped by how the pipeline preprocesses it. Statistics are on "
        "training claims before imputation. The target and anything derived from it are never "
        "inputs.",
        "",
        markdown_table(catalogue(train_features, cfg)),
        "",
        f"Text layers 2 and 3 add **{t['svd_components']} TF-IDF/SVD components** and a "
        f"**one-hot embedding cluster** (k = {e['n_clusters']}); see sections 4 and 5.",
        "",
    ]


# ---------------------------------------------------------------------------
# Placeholder initial estimates
# ---------------------------------------------------------------------------


def estimate_flag_evidence(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Accuracy of initial estimates for candidate 'placeholder' definitions."""
    initial, target = train[cfg.col("initial_estimate")], train[cfg.col("target")]
    log_error = (np.log(target) - np.log(initial)).abs()
    defaults = cfg["features"]["initial_default_values"]
    candidates = {
        "Multiple of 500": initial % 500 == 0,
        "Multiple of 1,000": initial % 1000 == 0,
        f"Equal to {', '.join(f'{v:g}' for v in defaults)} (chosen)": initial.isin(defaults),
    }
    rows = [
        {
            "Definition": name,
            "Share of claims": mask.mean(),
            "Median abs log error, flagged": log_error[mask].median(),
            "Median abs log error, others": log_error[~mask].median(),
        }
        for name, mask in candidates.items()
    ]
    return pd.DataFrame(rows)


def estimate_flag_section(train: pd.DataFrame, cfg: Config) -> list[str]:
    """Why only specific initial-estimate values are flagged as placeholders."""
    table = estimate_flag_evidence(train, cfg)
    return [
        "## 2. Which initial estimates behave like placeholders?",
        "",
        "Adjusters often enter round numbers (EDA section 1), so a natural idea is to flag "
        "round estimates as unreliable. The table tests that on training claims: a flag is only "
        "useful if flagged estimates are further from the final cost (higher error).",
        "",
        markdown_table(percent_columns(table, ["Share of claims"]), floatfmt=".3f"),
        "",
        "Round numbers in general are *not* less accurate, so a 'round number' flag would add "
        "noise. Only the most common single value is clearly worse than the rest, so the "
        "`initial_is_default` flag covers exactly the configured value(s).",
        "",
    ]


# ---------------------------------------------------------------------------
# Layer 1: keyword taxonomy
# ---------------------------------------------------------------------------


def taxonomy_section(
    train_features: pd.DataFrame, cfg: Config
) -> tuple[list[str], dict[str, float]]:
    """Coverage, category counts and unmatched examples for each taxonomy field."""
    rules = cfg["text_rules"]
    other = rules["other_label"]
    covered = coverage(train_features, other)
    coverage_table = pd.DataFrame(
        {"Field": list(covered.index), "Labelled (not 'other')": [f"{v:.1%}" for v in covered]}
    )
    blocks = [
        "## 3. Layer 1: keyword taxonomy",
        "",
        "Rules in `src/features/text_rules.py`; keyword lists in `config/config.yaml`. Keywords "
        "match whole words. For nature and cause the categories are in priority order, so a "
        "description mentioning both a laceration and a fracture is labelled a fracture. A "
        "finger mentioned together with its hand counts as a finger, so *multiple* means "
        "separate areas of the body.",
        "",
        "The dictionaries started from the categories in the project brief and were extended by "
        "reading unmatched descriptions: foreign body, hernia, inflammation, infection, soft "
        "tissue, psychological, bite/sting, dermatitis, amputation and pain were added to "
        "nature; vehicle, assault, bending/twisting, tool use, exposure and animal/insect to "
        "cause; trunk to body part. Descriptions that name an injury but not how it happened "
        "(for example 'STRAIN LOWER BACK') legitimately stay *other* for cause.",
        "",
        "**Coverage on training claims**",
        "",
        markdown_table(coverage_table),
        "",
    ]
    text = train_features[cfg.col("description")]
    target = cfg.col("target")
    for field in TAXONOMY_FIELDS:
        summary = group_summary(train_features, field, target, np.inf)
        summary = summary.sort_values(CLAIMS, ascending=False)[[field, CLAIMS, SHARE, MEDIAN]]
        summary[field] = readable(summary[field])
        unmatched = text[train_features[field] == other]
        examples = unmatched.sample(
            min(rules["unmatched_examples"], len(unmatched)), random_state=cfg.random_seed
        )
        title = readable(pd.Series([field]))[0].capitalize()
        blocks += [
            f"### {title}",
            "",
            markdown_table(percent_columns(summary, [SHARE]), floatfmt=",.0f"),
            "",
            f"Sample of descriptions still labelled *{other}*:",
            "",
            *[f"- `{example}`" for example in examples],
            "",
        ]
    return blocks, {f"coverage_{k}": float(v) for k, v in covered.items()}


# ---------------------------------------------------------------------------
# Layer 2: TF-IDF + SVD
# ---------------------------------------------------------------------------


def tfidf_section(train: pd.DataFrame, cfg: Config) -> tuple[list[str], dict[str, float]]:
    """Vocabulary size, variance explained and the leading terms of the first components."""
    t = cfg["tfidf"]
    pipe = tfidf_svd(cfg).fit(train[cfg.col("description")])
    vocabulary = pipe.named_steps["tfidf"].get_feature_names_out()
    svd = pipe.named_steps["svd"]
    explained = float(svd.explained_variance_ratio_.sum())
    rows = []
    for i, weights in enumerate(svd.components_[: t["components_shown"]]):
        top = np.argsort(np.abs(weights))[::-1][: t["top_terms_shown"]]
        rows.append({"Component": f"svd{i}", "Strongest terms": ", ".join(vocabulary[top])})
    blocks = [
        "## 4. Layer 2: TF-IDF + truncated SVD",
        "",
        f"TF-IDF over words and word pairs (n-grams {tuple(t['ngram_range'])}, terms in at least "
        f"{t['min_df']} training descriptions) gives a vocabulary of "
        f"**{len(vocabulary):,} terms**. "
        f"Truncated SVD compresses it to {t['svd_components']} dense components that together "
        f"keep **{explained:.0%} of the TF-IDF variance**. This captures wording the fixed "
        "keyword lists miss, at a size tree models handle easily. Both steps are fit inside the "
        "pipeline, so the vocabulary never includes words seen only in the test period.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
    ]
    return blocks, {"tfidf_vocabulary": len(vocabulary), "svd_explained_variance": explained}


# ---------------------------------------------------------------------------
# Layer 3: embedding clusters
# ---------------------------------------------------------------------------


def silhouette_table(vectors: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Silhouette score (on a sample) and inertia for each candidate k."""
    e = cfg["embeddings"]
    rows = []
    for k in e["k_candidates"]:
        model = KMeans(n_clusters=k, n_init=e["kmeans_n_init"], random_state=cfg.random_seed)
        labels = model.fit_predict(vectors)
        score = silhouette_score(
            vectors, labels, sample_size=e["silhouette_sample"], random_state=cfg.random_seed
        )
        rows.append({"k": k, "Silhouette": float(score), "Inertia": float(model.inertia_)})
        logger.info("k=%d silhouette=%.4f", k, score)
    return pd.DataFrame(rows)


def plot_silhouette(table: pd.DataFrame, chosen: int | None, cfg: Config) -> Path:
    """Silhouette score against k, with the chosen k marked."""
    p = cfg["plots"]
    fig, ax = new_figure(cfg)
    ax.plot(
        table["k"],
        table["Silhouette"],
        color=series_color(cfg, 0),
        marker="o",
        markersize=5,
        markeredgecolor=p["surface"],
        markeredgewidth=1.5,
    )
    if chosen is not None:
        ax.axvline(chosen, color=p["ink_muted"], linewidth=1)
        ax.text(
            chosen,
            1.0,
            f" chosen k = {chosen}",
            transform=ax.get_xaxis_transform(),
            va="top",
            color=p["ink_secondary"],
        )
    ax.set_xticks(table["k"])
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (higher = better separated)")
    ax.set_title("KMeans on sentence embeddings of training descriptions")
    return save_figure(fig, cfg, "features_silhouette")


def cluster_top_words(texts: pd.Series, labels: np.ndarray, cfg: Config) -> dict[int, str]:
    """Words most over-represented in each cluster relative to all descriptions."""
    e = cfg["embeddings"]
    words = texts.str.split().map(set)
    exploded = pd.DataFrame({"word": words, "cluster": labels}).explode("word").dropna()
    counts = exploded["word"].value_counts()
    common = counts[counts >= e["min_word_count"]].index
    exploded = exploded[exploded["word"].isin(common)]
    overall = counts[common] / len(texts)
    sizes = pd.Series(labels).value_counts()
    top = {}
    for cluster, group in exploded.groupby("cluster"):
        share = group["word"].value_counts() / sizes[cluster]
        lift = (share / overall[share.index])[share >= 0.05]
        top[int(cluster)] = ", ".join(
            lift.sort_values(ascending=False).index[: e["top_words_per_cluster"]]
        )
    return top


def cluster_examples(
    texts: pd.Series, distances: np.ndarray, labels: np.ndarray, n: int
) -> dict[int, list[str]]:
    """The descriptions closest to each cluster centre (distinct texts only)."""
    frame = pd.DataFrame({"text": texts.to_numpy(), "cluster": labels, "distance": distances})
    frame = frame.drop_duplicates("text").sort_values("distance")
    return {int(c): g["text"].head(n).tolist() for c, g in frame.groupby("cluster")}


def plot_clusters(summary: pd.DataFrame, cfg: Config) -> Path:
    """Paired bars: share of training claims and median cost per cluster."""
    fig, (ax_share, ax_cost) = new_figure(cfg, ncols=2, sharey=True)
    labels = summary["Name"]
    hbar(ax_share, summary[SHARE], labels, "{:.1%}", cfg)
    hbar(ax_cost, summary[MEDIAN], labels, "{:,.0f}", cfg)
    ax_share.invert_yaxis()
    ax_share.set_title("Embedding cluster: share of training claims")
    ax_cost.set_title("Embedding cluster: median ultimate cost")
    return save_figure(fig, cfg, "features_embedding_clusters")


def cluster_section(
    train: pd.DataFrame, vectors: np.ndarray, cfg: Config
) -> tuple[list[str], dict[str, Any]]:
    """Choice of k, and a named description of each cluster for the chosen k."""
    e = cfg["embeddings"]
    k = e["n_clusters"]
    report = cfg.path("features_report")
    table = silhouette_table(vectors, cfg)
    blocks = [
        "## 5. Layer 3: sentence-embedding clusters",
        "",
        f"Each description is embedded with the pretrained `{e['model_name']}` model (384 numbers "
        "per description, cached in `data/processed`). The model is never trained on our data, "
        "so embedding cannot leak information. KMeans then groups training descriptions by "
        "meaning, and the cluster id becomes a one-hot feature. Only KMeans learns, and inside "
        "the pipeline it is fit on training rows only.",
        "",
        "**Choosing k.** Silhouette measures how much closer each description is to its own "
        "cluster than to the next one. On short, overlapping texts it is low for every k, so it "
        "is a guide, not a verdict; the final choice also depends on whether each cluster reads "
        "as one recognisable kind of claim.",
        "",
        figure_markdown(
            plot_silhouette(table, k, cfg), report, "Silhouette by k", silhouette_takeaway(table, k)
        ),
        markdown_table(table, floatfmt=("g", ".4f", ",.0f")),
        "",
    ]
    metrics: dict[str, Any] = {
        "silhouette": table.set_index("k")["Silhouette"].to_dict(),
        "n_clusters": k,
    }
    if k is None:
        blocks += ["_k has not been chosen yet (`embeddings.n_clusters` is null)._", ""]
        return blocks, metrics
    model = KMeans(n_clusters=k, n_init=e["kmeans_n_init"], random_state=cfg.random_seed).fit(
        vectors
    )
    labels = model.labels_
    distances = model.transform(vectors)[np.arange(len(labels)), labels]
    texts = train[cfg.col("description")].reset_index(drop=True)
    names = name_clusters(model, cfg)
    frame = train.reset_index(drop=True).assign(**{CLUSTER: labels})
    summary = group_summary(frame, CLUSTER, cfg.col("target"), np.inf)
    summary["Name"] = summary[CLUSTER].map(lambda c: names[int(c)])
    summary = summary.sort_values(CLAIMS, ascending=False)
    words = cluster_top_words(texts, labels, cfg)
    examples = cluster_examples(texts, distances, labels, e["examples_per_cluster"])
    blocks += [
        choice_text(table, k, cfg),
        "",
        figure_markdown(
            plot_clusters(summary, cfg), report, "Embedding clusters", clusters_takeaway(summary)
        ),
    ]
    for row in summary.to_dict("records"):
        cluster = int(row[CLUSTER])
        blocks += [
            f"### Cluster {cluster}: {row['Name']}",
            "",
            f"{row[CLAIMS]:,} training claims ({row[SHARE]:.1%}), median cost "
            f"{row[MEDIAN]:,.0f}. Most over-represented words: {words.get(cluster, '-')}.",
            "",
            *[f"- `{text}`" for text in examples[cluster]],
            "",
        ]
    metrics["clusters"] = {
        int(r[CLUSTER]): {
            "name": r["Name"],
            "claims": int(r[CLAIMS]),
            "median_cost": float(r[MEDIAN]),
        }
        for _, r in summary.iterrows()
    }
    return blocks, metrics


def choice_text(table: pd.DataFrame, chosen: int, cfg: Config) -> str:
    """Explain the choice of k from the silhouette table and the manual read."""
    min_k = cfg["embeddings"]["min_useful_k"]
    useful = table[table["k"] >= min_k].reset_index(drop=True)
    position = useful.index[useful["k"] == chosen][0]
    scores = useful["Silhouette"]
    neighbours = scores.iloc[max(position - 1, 0) : position + 2].drop(position)
    peak = bool((scores.iloc[position] >= neighbours).all())
    peak_text = "a local peak" if peak else "not a local peak"
    return (
        f"**Chosen k = {chosen}.** Reading the clusters showed that k below {min_k} merges kinds "
        "of claim with very different costs (for example knee injuries with ankle sprains), so "
        f"only k >= {min_k} was considered. Across those values silhouette only moves between "
        f"{scores.min():.3f} and {scores.max():.3f}, too little to decide on its own; k = "
        f"{chosen} is {peak_text} in that range and gives clusters that each read as one kind "
        "of claim. Names were written by reading the descriptions closest to each cluster centre "
        "(listed below) and its most over-represented words. Each name is tied in the config to "
        "an anchor description rather than a cluster number, so it follows the meaning."
    )


def silhouette_takeaway(table: pd.DataFrame, chosen: int | None) -> str:
    """Describe the silhouette curve and where the chosen k sits on it."""
    best = table.loc[table["Silhouette"].idxmax()]
    text = (
        f"Silhouette scores range from {table['Silhouette'].min():.3f} to "
        f"{table['Silhouette'].max():.3f}, peaking at k = {int(best['k'])}. Scores this low are "
        "typical for short free-text descriptions, whose topics overlap."
    )
    if chosen is not None:
        score = table.loc[table["k"] == chosen, "Silhouette"]
        if len(score):
            text += f" The chosen k = {chosen} scores {float(score.iloc[0]):.3f}."
    return text


def clusters_takeaway(summary: pd.DataFrame) -> str:
    """Contrast the largest cluster with the dearest and cheapest."""
    largest = summary.iloc[0]
    dearest = summary.loc[summary[MEDIAN].idxmax()]
    cheapest = summary.loc[summary[MEDIAN].idxmin()]
    return (
        f"The largest cluster, **{largest['Name']}**, holds {largest[SHARE]:.0%} of training "
        f"claims. Median cost runs from {cheapest[MEDIAN]:,.0f} (**{cheapest['Name']}**) to "
        f"{dearest[MEDIAN]:,.0f} (**{dearest['Name']}**), a "
        f"{dearest[MEDIAN] / cheapest[MEDIAN]:.1f}x "
        "spread, so the clusters separate cheap and expensive kinds of claim."
    )


# ---------------------------------------------------------------------------
# Output matrix and safeguards
# ---------------------------------------------------------------------------


def matrix_section(train: pd.DataFrame, cfg: Config) -> tuple[list[str], dict[str, Any]]:
    """Fit the full preprocessor on training rows and count output columns per branch."""
    if cfg["embeddings"]["n_clusters"] is None:
        return ["## 6. Model matrix", "", "_Available once k is chosen._", ""], {}
    preprocessor = build_preprocessor(cfg).fit(train)
    names = feature_names(preprocessor)
    counts = pd.Series([n.split("__")[0] for n in names]).value_counts()
    table = counts.rename_axis("Branch").reset_index(name="Columns")
    target = cfg.col("target")
    inputs = model_input_columns(cfg)
    if target in inputs or any(target in name for name in names):
        raise AssertionError("The target leaked into the model inputs")
    return [
        "## 6. Model matrix",
        "",
        f"With every text layer switched on, the fitted pipeline turns each claim into "
        f"**{len(names)} columns**:",
        "",
        markdown_table(table),
        "",
        f"The switches in `text_layers` remove Layer 1, 2 or 3 for the Phase 5 ablation. The "
        f"target (`{target}`) is not among the {len(inputs)} input columns the pipeline reads, "
        "a check that also runs in the test suite.",
        "",
    ], {"n_model_columns": len(names), "columns_per_branch": counts.to_dict()}


def safeguards_blocks() -> list[str]:
    """How this design avoids leakage."""
    return [
        "## 7. Leakage safeguards",
        "",
        "- **Nothing learned outside the pipeline.** Group medians, scalers, one-hot categories, "
        "the TF-IDF vocabulary, SVD and KMeans are all steps of one scikit-learn Pipeline. They "
        "are fit on training rows only, and cross-validation refits them in every fold.",
        "- **Fixed rules only before the split.** Cleaning, feature formulas and the keyword "
        "taxonomy use thresholds fixed in the config and learn nothing from the data.",
        "- **Pretrained embeddings.** The sentence-transformer is used as-is, never trained on "
        "these claims.",
        "- **Design decisions on training claims only.** The placeholder-estimate value, the "
        "coverage iteration and the choice of k all used training claims.",
        "- **No target in the inputs.** The target and anything derived from it (such as the "
        "ultimate/initial ratio) are never model inputs.",
        "",
    ]


def run_features(cfg: Config) -> None:
    """Write ``reports/features.md`` and ``reports/metrics/features.json``."""
    claims = load_clean(cfg)
    embeddings_for(claims[cfg.col("description")], cfg)  # warm the cache for all claims
    train, _, cutoff = train_test_split_by_time(claims, cfg)
    train_features = engineer_features(train, cfg)
    vectors = embeddings_for(train[cfg.col("description")], cfg)
    taxonomy_blocks, taxonomy_metrics = taxonomy_section(train_features, cfg)
    tfidf_blocks, tfidf_metrics = tfidf_section(train, cfg)
    cluster_blocks, cluster_metrics = cluster_section(train, vectors, cfg)
    matrix_blocks, matrix_metrics = matrix_section(train, cfg)
    blocks = [
        "# Features",
        "",
        GENERATED_NOTE.format(stage="features"),
        "",
        f"All statistics use the {len(train):,} training claims (accidents before "
        f"{cutoff:%d %b %Y}).",
        "",
        *catalogue_section(train_features, cfg),
        *estimate_flag_section(train, cfg),
        *taxonomy_blocks,
        *tfidf_blocks,
        *cluster_blocks,
        *matrix_blocks,
        *safeguards_blocks(),
    ]
    write_text("\n".join(blocks), cfg.path("features_report"))
    write_json(
        {**taxonomy_metrics, **tfidf_metrics, **cluster_metrics, **matrix_metrics},
        cfg.path("features_metrics"),
    )
