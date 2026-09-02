from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from hillstrom_common import (  # noqa: E402
    DEFAULT_OUTPUTS,
    GROUP_ORDER,
    prepare_output,
    save_csv,
)


ALPHA = 0.05
ACTION_PROBABILITY = 1 / 3
ACTION_TO_INDEX = {action: index for index, action in enumerate(GROUP_ORDER)}
STATIC_POLICIES = {
    "no_email_all": ACTION_TO_INDEX["No E-Mail"],
    "mens_all": ACTION_TO_INDEX["Mens E-Mail"],
    "womens_all": ACTION_TO_INDEX["Womens E-Mail"],
}
REQUIRED_COLUMNS = {
    "row_id",
    "segment",
    "spend",
    "mu_no_email_tlearner",
    "mu_mens_tlearner",
    "mu_womens_tlearner",
    "tau_mens_drlearner",
    "tau_womens_drlearner",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行邮件成本敏感性分析。")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUTS["uplift"] / "08_oof_predictions.csv.gz",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["cost"])
    parser.add_argument(
        "--costs",
        type=float,
        nargs="+",
        default=[0.00, 0.05, 0.10, 0.25, 0.50, 1.00],
    )
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
    if not np.array_equal(predictions["row_id"].to_numpy(), np.arange(len(predictions))):
        raise ValueError("row_id must be consecutive and sorted from zero.")
    observed_groups = set(predictions["segment"].dropna().unique())
    if observed_groups != set(GROUP_ORDER):
        raise ValueError(f"Unexpected experimental groups: {sorted(observed_groups)}")
    if predictions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required OOF fields contain missing values.")
    return predictions


def aipw_scores(
    net_outcome: np.ndarray,
    observed_action: np.ndarray,
    nuisance_net: np.ndarray,
    policy_action: np.ndarray,
) -> np.ndarray:
    rows = np.arange(len(net_outcome))
    matches = observed_action == policy_action
    mu_policy = nuisance_net[rows, policy_action]
    mu_observed = nuisance_net[rows, observed_action]
    return mu_policy + matches / ACTION_PROBABILITY * (net_outcome - mu_observed)


def normal_interval(values: np.ndarray) -> tuple[float, float, float]:
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    critical = stats.norm.ppf(1 - ALPHA / 2)
    mean = values.mean()
    return standard_error, mean - critical * standard_error, mean + critical * standard_error


def main() -> None:
    args = parse_args()
    costs = sorted(set(args.costs))
    if not costs or any(cost < 0 for cost in costs):
        raise ValueError("Costs must be a non-empty list of non-negative values.")

    output = prepare_output(args.output)
    predictions = load_predictions(args.predictions)

    print("=" * 82)
    print("Hillstorm 成本敏感性分析")
    print("=" * 82)
    print(f"输入：{args.predictions}")
    print(f"输出：{output}")

    design = f"""Hillstorm 成本敏感性分析设计

目标
----
评估假设的邮件边际成本变化，如何影响静态策略和个性化策略的价值，以及个性化策略的动作分配。

成本情景
--------
{', '.join(f'{cost:.2f}' for cost in costs)}

净结果
------
发送邮件时，以观察到的 Spend 减去假设的边际邮件成本作为净结果。策略价值采用 AIPW 估计量，随机分配概率已知为 1/3。基于影响函数的正态区间是在已学习的 OOF 策略条件下计算，不包含模型重新训练带来的不确定性。

上述成本仅为情景参数，不代表已确认的真实活动成本、毛利、退订损失或长期客户价值。
"""
    (output / "00_cost_sensitivity_design.txt").write_text(design, encoding="utf-8")

    n = len(predictions)
    outcome = predictions["spend"].to_numpy(dtype=float)
    observed_action = predictions["segment"].map(ACTION_TO_INDEX).to_numpy(dtype=int)
    nuisance = predictions[
        ["mu_no_email_tlearner", "mu_mens_tlearner", "mu_womens_tlearner"]
    ].to_numpy(dtype=float)
    tau_mens = predictions["tau_mens_drlearner"].to_numpy(dtype=float)
    tau_womens = predictions["tau_womens_drlearner"].to_numpy(dtype=float)

    value_rows = []
    allocation_rows = []
    comparison_rows = []
    for cost in costs:
        action_cost = np.array([0.0, cost, cost])
        net_outcome = outcome - action_cost[observed_action]
        nuisance_net = nuisance - action_cost
        personalized = np.column_stack(
            [np.zeros(n), tau_mens - cost, tau_womens - cost]
        ).argmax(axis=1)
        policies = {
            **{
                name: np.full(n, action_index, dtype=int)
                for name, action_index in STATIC_POLICIES.items()
            },
            "personalized_dr": personalized,
        }
        score_by_policy = {}
        for policy_name, policy_action in policies.items():
            scores = aipw_scores(
                net_outcome, observed_action, nuisance_net, policy_action
            )
            score_by_policy[policy_name] = scores
            standard_error, low, high = normal_interval(scores)
            value_rows.append(
                {
                    "email_cost": cost,
                    "policy": policy_name,
                    "aipw_net_value": scores.mean(),
                    "standard_error": standard_error,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_label": "fixed_policy_influence_function_95pct",
                }
            )

        for action_name, action_index in ACTION_TO_INDEX.items():
            allocation_rows.append(
                {
                    "email_cost": cost,
                    "assigned_action": action_name,
                    "n": int((personalized == action_index).sum()),
                    "share": float((personalized == action_index).mean()),
                }
            )

        static_values = {
            name: score_by_policy[name].mean() for name in STATIC_POLICIES
        }
        best_static = max(static_values, key=static_values.get)
        paired = score_by_policy["personalized_dr"] - score_by_policy[best_static]
        standard_error, low, high = normal_interval(paired)
        z_stat = 0.0 if standard_error == 0 else paired.mean() / standard_error
        comparison_rows.append(
            {
                "email_cost": cost,
                "best_static_policy": best_static,
                "personalized_gain_vs_best_static": paired.mean(),
                "standard_error": standard_error,
                "ci_low": low,
                "ci_high": high,
                "z_stat": z_stat,
                "p_value_two_sided": 2 * stats.norm.sf(abs(z_stat)),
                "personalization_supported": paired.mean() > 0 and low > 0,
                "ci_label": "paired_fixed_policy_influence_function_95pct",
            }
        )

    values = pd.DataFrame(value_rows)
    allocation = pd.DataFrame(allocation_rows)
    comparisons = pd.DataFrame(comparison_rows)
    save_csv(values, output, "01_cost_sensitivity_policy_values.csv")
    save_csv(allocation, output, "02_cost_sensitivity_allocation.csv")
    save_csv(comparisons, output, "03_cost_sensitivity_comparisons.csv")

    raw_static_scores = {
        name: aipw_scores(
            outcome,
            observed_action,
            nuisance,
            np.full(n, action_index, dtype=int),
        )
        for name, action_index in STATIC_POLICIES.items()
    }
    no_email_value = raw_static_scores["no_email_all"].mean()
    break_even_rows = []
    for policy_name, label in [("mens_all", "Mens E-Mail"), ("womens_all", "Womens E-Mail")]:
        spend_value = raw_static_scores[policy_name].mean()
        break_even_rows.append(
            {
                "strategy": label,
                "estimated_break_even_email_cost": spend_value - no_email_value,
                "interpretation": "仅表示 Spend 差异，不是利润阈值",
            }
        )
    save_csv(pd.DataFrame(break_even_rows), output, "04_static_break_even_summary.csv")

    supported_costs = comparisons.loc[
        comparisons["personalization_supported"], "email_cost"
    ].tolist()
    conclusion = f"""成本敏感性分析结论

在指定成本网格中，个性化策略相对最佳静态策略获得统计支持的增益：
{', '.join(f'{cost:.2f}' for cost in supported_costs) if supported_costs else '无'}

解释
----
成本网格仅用于情景分析。静态策略的盈亏平衡值是增量 Spend 估计，并非利润阈值。正式投放决策还需要真实投放成本、毛利、退订或疲劳效应，以及 Hillstorm 数据中未包含的实施成本。
"""
    (output / "05_cost_sensitivity_decision.txt").write_text(conclusion, encoding="utf-8")

    print("\n成本敏感性比较")
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n" + conclusion)


if __name__ == "__main__":
    main()
