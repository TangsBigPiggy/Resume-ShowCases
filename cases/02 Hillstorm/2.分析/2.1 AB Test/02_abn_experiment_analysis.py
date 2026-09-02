from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportions_ztest


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
    validate_binary,
    validate_groups,
    welch_mean_difference,
)


ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Hillstorm A/B/n 实验分析。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["abn"])
    return parser.parse_args()


def apply_holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    reject, adjusted, _, _ = multipletests(
        result["p_value"].to_numpy(), alpha=ALPHA, method="holm"
    )
    result["p_value_holm"] = adjusted
    result["significant_holm"] = reject
    return result


def binary_contrast(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    outcome: str,
) -> dict:
    a = df.loc[df["segment"] == group_a, outcome].astype(int).to_numpy()
    b = df.loc[df["segment"] == group_b, outcome].astype(int).to_numpy()
    events = np.array([a.sum(), b.sum()])
    totals = np.array([len(a), len(b)])
    z_stat, p_value = proportions_ztest(events, totals, alternative="two-sided")
    rate_a, rate_b = a.mean(), b.mean()
    difference = rate_a - rate_b
    unpooled_se = np.sqrt(rate_a * (1 - rate_a) / len(a) + rate_b * (1 - rate_b) / len(b))
    critical = stats.norm.ppf(1 - ALPHA / 2)
    relative_lift = np.nan if rate_b == 0 else difference / rate_b * 100
    return {
        "outcome": outcome,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "events_a": int(a.sum()),
        "events_b": int(b.sum()),
        "rate_a": rate_a,
        "rate_b": rate_b,
        "difference": difference,
        "difference_pp": difference * 100,
        "relative_lift_pct": relative_lift,
        "ci_low": difference - critical * unpooled_se,
        "ci_high": difference + critical * unpooled_se,
        "ci_low_pp": (difference - critical * unpooled_se) * 100,
        "ci_high_pp": (difference + critical * unpooled_se) * 100,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_label": "nominal_95pct_unadjusted",
    }


def main() -> None:
    args = parse_args()
    output = prepare_output(args.output)
    df = read_hillstrom(args.data, {"segment", "visit", "conversion", "spend"})
    validate_groups(df)
    validate_binary(df, ["visit", "conversion"])
    if df[["segment", "visit", "conversion", "spend"]].isna().any().any():
        raise ValueError("Segment and outcome columns must not contain missing values.")

    print("=" * 82)
    print("Hillstorm 随机 A/B/n 实验分析")
    print("=" * 82)
    print(f"输入：{args.data}")
    print(f"输出：{output}")

    design = """Hillstorm 随机 A/B/n 实验分析设计

主要估计量
----------
按随机分配客户计算的人均 Spend（意向治疗，ITT）。

次要结果指标
------------
访问率与转化率。

预设比较
--------
Mens E-Mail vs No E-Mail
Womens E-Mail vs No E-Mail
Mens E-Mail vs Womens E-Mail

统计推断
--------
采用双侧检验，显著性水平 alpha = 0.05。Spend、Visit 和 Conversion 三类指标分别在各自检验族内应用 Holm 多重比较校正。报告的置信区间为 95% 置信区间，不做多重性校正。
"""
    (output / "00_research_design.txt").write_text(design, encoding="utf-8")

    summary_rows = []
    for group in GROUP_ORDER:
        data = df[df["segment"] == group]
        converters = data[data["conversion"] == 1]
        summary_rows.append(
            {
                "segment": group,
                "n": len(data),
                "share": len(data) / len(df),
                "visits": int(data["visit"].sum()),
                "visit_rate": data["visit"].mean(),
                "conversions": int(data["conversion"].sum()),
                "conversion_rate": data["conversion"].mean(),
                "total_spend": data["spend"].sum(),
                "spend_per_customer": data["spend"].mean(),
                "spend_std": data["spend"].std(ddof=1),
                "spend_median": data["spend"].median(),
                "zero_spend_rate": (data["spend"] == 0).mean(),
                "spend_per_converter": converters["spend"].mean() if len(converters) else np.nan,
            }
        )
    group_summary = pd.DataFrame(summary_rows)
    save_csv(group_summary, output, "01_group_summary.csv")

    omnibus_rows = []
    for outcome in ["visit", "conversion"]:
        table = pd.crosstab(df["segment"], df[outcome]).reindex(GROUP_ORDER, fill_value=0)
        chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
        omnibus_rows.append(
            {
                "outcome": outcome,
                "test": "chi_square",
                "statistic": chi2,
                "df_num": dof,
                "df_denom": np.nan,
                "p_value": p_value,
            }
        )

    spend_groups = [
        df.loc[df["segment"] == group, "spend"].astype(float).to_numpy()
        for group in GROUP_ORDER
    ]
    welch_anova = anova_oneway(spend_groups, use_var="unequal", welch_correction=True)
    omnibus_rows.append(
        {
            "outcome": "spend",
            "test": "welch_anova",
            "statistic": welch_anova.statistic,
            "df_num": welch_anova.df_num,
            "df_denom": welch_anova.df_denom,
            "p_value": welch_anova.pvalue,
        }
    )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["significant"] = omnibus["p_value"] < ALPHA
    save_csv(omnibus, output, "02_global_omnibus_tests.csv")

    spend_rows = []
    for group_a, group_b in PLANNED_CONTRASTS:
        a = df.loc[df["segment"] == group_a, "spend"].astype(float).to_numpy()
        b = df.loc[df["segment"] == group_b, "spend"].astype(float).to_numpy()
        row = welch_mean_difference(a, b, alpha=ALPHA)
        row.update(
            {
                "outcome": "spend",
                "group_a": group_a,
                "group_b": group_b,
                "relative_lift_pct": np.nan if row["mean_b"] == 0 else row["difference"] / row["mean_b"] * 100,
                "increment_per_1000": row["difference"] * 1000,
                "effect_unit": "currency_per_customer",
                "ci_label": "nominal_95pct_unadjusted",
            }
        )
        spend_rows.append(row)
    spend_results = apply_holm(pd.DataFrame(spend_rows))
    spend_results["effect_direction"] = np.select(
        [
            spend_results["significant_holm"] & (spend_results["ci_low"] > 0),
            spend_results["significant_holm"] & (spend_results["ci_high"] < 0),
        ],
        ["positive", "negative"],
        default="inconclusive",
    )
    save_csv(spend_results, output, "03_primary_spend_contrasts.csv")

    secondary_tables = []
    for outcome in ["visit", "conversion"]:
        family = pd.DataFrame(
            [binary_contrast(df, group_a, group_b, outcome) for group_a, group_b in PLANNED_CONTRASTS]
        )
        family = apply_holm(family)
        family["effect_direction"] = np.select(
            [
                family["significant_holm"] & (family["ci_low"] > 0),
                family["significant_holm"] & (family["ci_high"] < 0),
            ],
            ["positive", "negative"],
            default="inconclusive",
        )
        secondary_tables.append(family)
    secondary = pd.concat(secondary_tables, ignore_index=True)
    save_csv(secondary, output, "04_secondary_outcome_contrasts.csv")

    decision = spend_results[
        [
            "group_a",
            "group_b",
            "mean_a",
            "mean_b",
            "difference",
            "increment_per_1000",
            "relative_lift_pct",
            "ci_low",
            "ci_high",
            "ci_label",
            "p_value",
            "p_value_holm",
            "significant_holm",
            "effect_direction",
        ]
    ].copy()
    decision.insert(0, "comparison", decision["group_a"] + " vs " + decision["group_b"])
    save_csv(decision, output, "05_primary_decision_table.csv")

    unified_rows = []
    for _, row in spend_results.iterrows():
        unified_rows.append(
            {
                "outcome": "spend",
                "comparison": f"{row['group_a']} vs {row['group_b']}",
                "metric_a": row["mean_a"],
                "metric_b": row["mean_b"],
                "absolute_effect": row["difference"],
                "effect_unit": "currency_per_customer",
                "relative_lift_pct": row["relative_lift_pct"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "ci_label": row["ci_label"],
                "p_value": row["p_value"],
                "p_value_holm": row["p_value_holm"],
                "significant_holm": row["significant_holm"],
            }
        )
    for _, row in secondary.iterrows():
        unified_rows.append(
            {
                "outcome": row["outcome"],
                "comparison": f"{row['group_a']} vs {row['group_b']}",
                "metric_a": row["rate_a"],
                "metric_b": row["rate_b"],
                "absolute_effect": row["difference_pp"],
                "effect_unit": "percentage_point",
                "relative_lift_pct": row["relative_lift_pct"],
                "ci_low": row["ci_low_pp"],
                "ci_high": row["ci_high_pp"],
                "ci_label": row["ci_label"],
                "p_value": row["p_value"],
                "p_value_holm": row["p_value_holm"],
                "significant_holm": row["significant_holm"],
            }
        )
    save_csv(pd.DataFrame(unified_rows), output, "06_all_planned_contrasts.csv")

    print("\n主要结果：每名随机客户的 Spend")
    for _, row in decision.iterrows():
        print(
            f"{row['comparison']}: {row['difference']:.6f} "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}], "
            f"Holm {format_p(row['p_value_holm'])}"
        )
    print("\n生成文件：")
    for path in sorted(output.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
