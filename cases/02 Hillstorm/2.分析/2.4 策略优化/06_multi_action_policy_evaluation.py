from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from hillstrom_common import (  # noqa: E402
    DEFAULT_OUTPUTS,
    GROUP_ORDER,
    format_p,
    prepare_output,
    save_csv,
    stratified_bootstrap_indices,
)


ALPHA = 0.05
ACTION_PROBABILITY = 1 / 3
ACTION_TO_INDEX = {action: index for index, action in enumerate(GROUP_ORDER)}
POLICY_ORDER = ["no_email_all", "mens_all", "womens_all", "personalized_dr"]
REQUIRED_COLUMNS = {
    "row_id",
    "fold",
    "segment",
    "spend",
    "mu_no_email_tlearner",
    "mu_mens_tlearner",
    "mu_womens_tlearner",
    "tau_mens_drlearner",
    "tau_womens_drlearner",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-action email policies.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUTS["uplift"] / "08_oof_predictions.csv.gz",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["policy"])
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"OOF prediction file not found: {path}. Run Stage 05 first."
        )
    predictions = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"OOF prediction file is missing columns: {sorted(missing)}")
    if predictions["row_id"].duplicated().any():
        raise ValueError("row_id must be unique in the OOF prediction file.")
    expected_ids = np.arange(len(predictions))
    if not np.array_equal(predictions["row_id"].to_numpy(), expected_ids):
        raise ValueError("row_id must be consecutive and sorted from zero.")
    observed_groups = set(predictions["segment"].dropna().unique())
    if observed_groups != set(GROUP_ORDER):
        raise ValueError(f"Unexpected experimental groups: {sorted(observed_groups)}")
    if predictions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required OOF fields contain missing values.")
    return predictions


def build_policies(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(predictions)
    candidate_uplift = np.column_stack(
        [
            np.zeros(n),
            predictions["tau_mens_drlearner"].to_numpy(),
            predictions["tau_womens_drlearner"].to_numpy(),
        ]
    )
    personalized = candidate_uplift.argmax(axis=1)
    return {
        "no_email_all": np.full(n, ACTION_TO_INDEX["No E-Mail"], dtype=int),
        "mens_all": np.full(n, ACTION_TO_INDEX["Mens E-Mail"], dtype=int),
        "womens_all": np.full(n, ACTION_TO_INDEX["Womens E-Mail"], dtype=int),
        "personalized_dr": personalized,
    }


def policy_scores(
    outcome: np.ndarray,
    observed_action: np.ndarray,
    nuisance_means: np.ndarray,
    policy_action: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(outcome))
    matches = observed_action == policy_action
    mu_policy = nuisance_means[rows, policy_action]
    mu_observed = nuisance_means[rows, observed_action]
    return {
        "direct_method": mu_policy,
        "ipw": matches * outcome / ACTION_PROBABILITY,
        "aipw": mu_policy
        + matches / ACTION_PROBABILITY * (outcome - mu_observed),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("Bootstrap replications must be positive.")
    output = prepare_output(args.output)
    predictions = load_predictions(args.predictions)

    print("=" * 82)
    print("Hillstorm Multi-Action Policy Evaluation")
    print("=" * 82)
    print(f"Input : {args.predictions}")
    print(f"Output: {output}")

    design = f"""Hillstorm Multi-Action Policy Evaluation

Policies
--------
No E-Mail for all
Mens E-Mail for all
Womens E-Mail for all
Personalized DR policy: argmax of 0, predicted Mens uplift, and predicted
Womens uplift.

Evaluation
----------
Known randomization probability: 1/3 for each action.
Policy values are reported using direct-method, IPW, and AIPW estimators.
The primary estimator is AIPW. The policy and nuisance predictions are fully
out of fold for the evaluated customer. Paired, treatment-stratified fixed-
policy bootstrap intervals quantify evaluation-sample uncertainty.

Bootstrap replications: {args.bootstrap:,}
Random seed: {args.seed}

The bootstrap holds the learned OOF policy fixed and does not include model-
retraining uncertainty.
"""
    (output / "00_policy_design.txt").write_text(design, encoding="utf-8")

    outcome = predictions["spend"].to_numpy(dtype=float)
    observed_action = predictions["segment"].map(ACTION_TO_INDEX).to_numpy(dtype=int)
    nuisance = predictions[
        ["mu_no_email_tlearner", "mu_mens_tlearner", "mu_womens_tlearner"]
    ].to_numpy(dtype=float)
    policies = build_policies(predictions)

    personalized_actions = pd.Series(policies["personalized_dr"]).map(
        {index: action for action, index in ACTION_TO_INDEX.items()}
    )
    allocation = (
        personalized_actions.value_counts()
        .reindex(GROUP_ORDER, fill_value=0)
        .rename("n")
        .reset_index()
    )
    allocation.columns = ["assigned_action", "n"]
    allocation["share"] = allocation["n"] / len(predictions)
    save_csv(allocation, output, "01_personalized_allocation.csv")

    scores = {
        policy_name: policy_scores(outcome, observed_action, nuisance, policy_action)
        for policy_name, policy_action in policies.items()
    }
    rng = np.random.default_rng(args.seed)
    bootstrap_values = {
        policy_name: np.empty(args.bootstrap) for policy_name in POLICY_ORDER
    }
    for replication in range(args.bootstrap):
        indices = stratified_bootstrap_indices(observed_action, rng)
        for policy_name in POLICY_ORDER:
            bootstrap_values[policy_name][replication] = scores[policy_name]["aipw"][indices].mean()

    value_rows = []
    for policy_name in POLICY_ORDER:
        aipw = scores[policy_name]["aipw"]
        low, high = np.quantile(
            bootstrap_values[policy_name], [ALPHA / 2, 1 - ALPHA / 2]
        )
        value_rows.append(
            {
                "policy": policy_name,
                "direct_method_value": scores[policy_name]["direct_method"].mean(),
                "ipw_value": scores[policy_name]["ipw"].mean(),
                "aipw_value": aipw.mean(),
                "aipw_standard_error": aipw.std(ddof=1) / np.sqrt(len(aipw)),
                "aipw_bootstrap_ci_low": low,
                "aipw_bootstrap_ci_high": high,
                "ci_label": "fixed_policy_stratified_bootstrap_95pct",
            }
        )
    policy_values = pd.DataFrame(value_rows)
    static_names = ["no_email_all", "mens_all", "womens_all"]
    best_static = (
        policy_values[policy_values["policy"].isin(static_names)]
        .sort_values("aipw_value", ascending=False)
        .iloc[0]["policy"]
    )
    best_static_value = policy_values.loc[
        policy_values["policy"] == best_static, "aipw_value"
    ].iloc[0]
    policy_values["gain_vs_best_static_point_estimate"] = (
        policy_values["aipw_value"] - best_static_value
    )
    save_csv(policy_values, output, "02_policy_values.csv")

    comparison_rows = []
    personalized_point = scores["personalized_dr"]["aipw"].mean()
    for static_policy in static_names:
        differences = (
            bootstrap_values["personalized_dr"] - bootstrap_values[static_policy]
        )
        low, high = np.quantile(differences, [ALPHA / 2, 1 - ALPHA / 2])
        left = (np.count_nonzero(differences <= 0) + 1) / (args.bootstrap + 1)
        right = (np.count_nonzero(differences >= 0) + 1) / (args.bootstrap + 1)
        comparison_rows.append(
            {
                "policy_a": "personalized_dr",
                "policy_b": static_policy,
                "aipw_value_difference": personalized_point
                - scores[static_policy]["aipw"].mean(),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_p_value_two_sided": min(1.0, 2 * min(left, right)),
                "ci_label": "paired_fixed_policy_stratified_bootstrap_95pct",
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    reject, adjusted, _, _ = multipletests(
        comparisons["bootstrap_p_value_two_sided"], alpha=ALPHA, method="holm"
    )
    comparisons["bootstrap_p_value_holm"] = adjusted
    comparisons["significant_holm"] = reject
    save_csv(comparisons, output, "03_policy_comparisons.csv")

    personalization_supported = bool(
        (
            (comparisons["aipw_value_difference"] > 0)
            & (comparisons["bootstrap_ci_low"] > 0)
            & comparisons["significant_holm"]
        ).all()
    )
    best_comparison = comparisons[comparisons["policy_b"] == best_static].iloc[0]
    decision = f"""Multi-Action Policy Decision

Best static policy by AIPW point estimate: {best_static}
Personalized AIPW gain vs best static: {best_comparison['aipw_value_difference']:.6f}
Paired bootstrap nominal 95% CI: [{best_comparison['bootstrap_ci_low']:.6f}, {best_comparison['bootstrap_ci_high']:.6f}]
Holm-adjusted comparison: {format_p(best_comparison['bootstrap_p_value_holm'])}
Personalization value supported: {personalization_supported}

Decision rule
-------------
The personalized policy must have a positive point estimate, positive paired
bootstrap lower bound, and Holm-adjusted significance against every static
policy. Otherwise the evidence for operational personalization is insufficient.

Policy values measure expected Spend per randomized customer, not profit or ROI.
"""
    (output / "04_policy_decision.txt").write_text(decision, encoding="utf-8")

    print("\nPolicy values")
    print(policy_values.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n" + decision)


if __name__ == "__main__":
    main()
