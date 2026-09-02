from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
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
POLICY_LABELS = {
    "no_email_all": "全量不发送邮件",
    "mens_all": "全量男装邮件",
    "womens_all": "全量女装邮件",
    "personalized_dr": "个性化 DR 策略",
}
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
    parser = argparse.ArgumentParser(description="评估多动作邮件投放策略。")
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
            f"未找到样本外预测文件：{path}。请先运行 Uplift 验证脚本。"
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
    print("Hillstorm 多动作策略评估")
    print("=" * 82)
    print(f"输入：{args.predictions}")
    print(f"输出：{output}")

    design = f"""Hillstorm 多动作策略评估设计

策略
----
所有客户均不发送邮件
所有客户均发送 Mens E-Mail
所有客户均发送 Womens E-Mail
个性化 DR 策略：在 0、预测 Mens Uplift、预测 Womens Uplift 三者中选择最大值对应的动作。

评估
----
每个动作的已知随机分配概率均为 1/3。策略价值同时报告 Direct Method、IPW 和 AIPW 估计，主估计量为 AIPW。被评估客户对应的策略预测与干扰项预测均完全来自样本外。使用按处理组分层、固定策略的配对 Bootstrap 区间量化评估样本的不确定性。

Bootstrap 重复次数：{args.bootstrap:,}
随机种子：{args.seed}

Bootstrap 过程中固定已学习的 OOF 策略，因此不包含模型重新训练带来的不确定性。
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
    best_static_label = POLICY_LABELS[best_static]
    decision = f"""多动作策略决策结论

按 AIPW 点估计得到的最佳静态策略：{best_static_label}
个性化策略相对最佳静态策略的 AIPW 增益：{best_comparison['aipw_value_difference']:.6f}
配对 Bootstrap 95% CI：[{best_comparison['bootstrap_ci_low']:.6f}, {best_comparison['bootstrap_ci_high']:.6f}]
Holm 校正比较：{format_p(best_comparison['bootstrap_p_value_holm'])}
个性化价值获得支持：{'是' if personalization_supported else '否'}

判定规则
--------
个性化策略必须同时满足：点估计为正、配对 Bootstrap 下界为正，并且相对每一个静态策略的比较在 Holm 校正后达到显著。否则，认为当前证据不足以支持在实际运营中采用个性化策略。

策略价值衡量的是每位随机客户的期望 Spend，而不是利润或 ROI。
"""
    (output / "04_policy_decision.txt").write_text(decision, encoding="utf-8")

    print("\n策略价值")
    print(policy_values.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n" + decision)


if __name__ == "__main__":
    main()
