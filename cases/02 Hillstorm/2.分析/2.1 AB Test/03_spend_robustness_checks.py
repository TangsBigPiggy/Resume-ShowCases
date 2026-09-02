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
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUTS,
    GROUP_ORDER,
    PLANNED_CONTRASTS,
    format_p,
    prepare_output,
    read_hillstrom,
    save_csv,
    validate_groups,
    welch_mean_difference,
)


ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Spend 稳健性检验。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["robustness"])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=64)
    return parser.parse_args()


def bootstrap_group_means(
    values: np.ndarray,
    replications: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    means = np.empty(replications, dtype=float)
    for start in range(0, replications, chunk_size):
        end = min(start + chunk_size, replications)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    return means


def t_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 0.0 if se == 0 else (a.mean() - b.mean()) / se


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        raise ValueError("Bootstrap and permutation counts must be positive.")

    output = prepare_output(args.output)
    df = read_hillstrom(args.data, {"segment", "spend"})
    validate_groups(df)
    if df[["segment", "spend"]].isna().any().any():
        raise ValueError("Segment and Spend must not contain missing values.")

    print("=" * 82)
    print("Hillstorm Spend 稳健性分析")
    print("=" * 82)
    print(f"输入：{args.data}")
    print(f"输出：{output}")

    design = f"""Hillstorm Spend 稳健性分析设计

估计量
------
按随机分配客户计算的人均 Spend（意向治疗，ITT）。

检验程序
--------
1. Welch 均值差异推断。
2. 组内非参数百分位 Bootstrap。
3. 保持三个观察组样本量不变的随机化检验。
4. 对两两比较的随机化检验 p 值进行 Holm 校正。
5. 使用置换 max-|t| 方法控制家族错误率。

Bootstrap 重复次数：{args.bootstrap:,}
置换重复次数：{args.permutations:,}
随机种子：{args.seed}

保留 Spend = 0 的观测；不进行截尾、Winsorization 或任何处理后筛选。置信区间均为 95% 置信区间，不做多重性校正。
"""
    (output / "00_robustness_design.txt").write_text(design, encoding="utf-8")

    values = {
        group: df.loc[df["segment"] == group, "spend"].astype(float).to_numpy()
        for group in GROUP_ORDER
    }
    observed = pd.DataFrame(
        [
            {
                "group_a": group_a,
                "group_b": group_b,
                **welch_mean_difference(values[group_a], values[group_b], alpha=ALPHA),
            }
            for group_a, group_b in PLANNED_CONTRASTS
        ]
    )
    reject, adjusted, _, _ = multipletests(observed["p_value"], alpha=ALPHA, method="holm")
    observed["p_value_holm"] = adjusted
    observed["significant_holm"] = reject
    observed["ci_label"] = "nominal_95pct_unadjusted"
    save_csv(observed, output, "01_observed_welch_results.csv")

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(GROUP_ORDER) + 1)
    boot_means = {
        group: bootstrap_group_means(
            values[group],
            args.bootstrap,
            args.bootstrap_chunk_size,
            np.random.default_rng(child_seeds[index]),
        )
        for index, group in enumerate(GROUP_ORDER)
    }
    bootstrap_rows = []
    for group_a, group_b in PLANNED_CONTRASTS:
        differences = boot_means[group_a] - boot_means[group_b]
        low, high = np.quantile(differences, [ALPHA / 2, 1 - ALPHA / 2])
        bootstrap_rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "observed_difference": values[group_a].mean() - values[group_b].mean(),
                "bootstrap_mean_difference": differences.mean(),
                "bootstrap_standard_error": differences.std(ddof=1),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_ci_method": "percentile",
                "ci_label": "nominal_95pct_unadjusted",
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output, "02_bootstrap_results.csv")

    observed_t = {
        (row.group_a, row.group_b): row.t_stat for row in observed.itertuples(index=False)
    }
    raw_extreme = {contrast: 0 for contrast in PLANNED_CONTRASTS}
    max_extreme = {contrast: 0 for contrast in PLANNED_CONTRASTS}
    all_spend = df["spend"].astype(float).to_numpy()
    sizes = [len(values[group]) for group in GROUP_ORDER]
    cuts = np.cumsum(sizes)[:-1]
    rng = np.random.default_rng(child_seeds[-1])

    for _ in range(args.permutations):
        permuted = rng.permutation(all_spend)
        parts = np.split(permuted, cuts)
        permuted_values = dict(zip(GROUP_ORDER, parts, strict=True))
        statistics = {
            contrast: t_from_arrays(permuted_values[contrast[0]], permuted_values[contrast[1]])
            for contrast in PLANNED_CONTRASTS
        }
        max_abs = max(abs(value) for value in statistics.values())
        for contrast, statistic in statistics.items():
            if abs(statistic) >= abs(observed_t[contrast]):
                raw_extreme[contrast] += 1
            if max_abs >= abs(observed_t[contrast]):
                max_extreme[contrast] += 1

    permutation_rows = []
    for contrast in PLANNED_CONTRASTS:
        permutation_rows.append(
            {
                "group_a": contrast[0],
                "group_b": contrast[1],
                "observed_t": observed_t[contrast],
                "permutation_p_value": (raw_extreme[contrast] + 1) / (args.permutations + 1),
                "permutation_max_abs_t_p_value": (max_extreme[contrast] + 1) / (args.permutations + 1),
                "permutations": args.permutations,
            }
        )
    permutation = pd.DataFrame(permutation_rows)
    reject, adjusted, _, _ = multipletests(
        permutation["permutation_p_value"], alpha=ALPHA, method="holm"
    )
    permutation["permutation_p_value_holm"] = adjusted
    permutation["significant_permutation_holm"] = reject
    permutation["significant_max_abs_t"] = permutation["permutation_max_abs_t_p_value"] < ALPHA
    save_csv(permutation, output, "03_permutation_results.csv")

    comparison = (
        observed.merge(
            bootstrap,
            on=["group_a", "group_b"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            permutation,
            on=["group_a", "group_b"],
            how="inner",
            validate="one_to_one",
        )
    )
    bootstrap_same_direction = (
        ((comparison["difference"] > 0) & (comparison["bootstrap_ci_low"] > 0))
        | ((comparison["difference"] < 0) & (comparison["bootstrap_ci_high"] < 0))
    )
    comparison["robustness_supported"] = (
        comparison["significant_holm"]
        & bootstrap_same_direction
        & comparison["significant_permutation_holm"]
        & comparison["significant_max_abs_t"]
    )
    save_csv(comparison, output, "04_robustness_comparison.csv")

    summary_lines = ["Hillstorm Spend 稳健性分析摘要", ""]
    for row in comparison.itertuples(index=False):
        summary_lines.extend(
            [
                f"{row.group_a} vs {row.group_b}",
                f"观察均值差：{row.difference:.6f}",
                f"Welch 95% CI：[{row.ci_low:.6f}, {row.ci_high:.6f}]",
                f"Bootstrap 95% CI：[{row.bootstrap_ci_low:.6f}, {row.bootstrap_ci_high:.6f}]",
                f"Welch Holm：{format_p(row.p_value_holm)}",
                f"Permutation Holm：{format_p(row.permutation_p_value_holm)}",
                f"Permutation max-|t|：{format_p(row.permutation_max_abs_t_p_value)}",
                f"稳健性支持：{'是' if row.robustness_supported else '否'}",
                "",
            ]
        )
    overall = "结果一致" if comparison["robustness_supported"].all() else "结果混合，需要复核"
    summary_lines.append(f"整体稳健性状态：{overall}")
    summary_text = "\n".join(summary_lines)
    (output / "05_robustness_summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
