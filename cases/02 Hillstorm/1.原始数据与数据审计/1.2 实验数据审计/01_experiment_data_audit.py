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
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUTS,
    GROUP_ORDER,
    format_p,
    prepare_output,
    read_hillstrom,
    save_csv,
    validate_binary,
    validate_groups,
)


NUMERIC_COVARIATES = ["recency", "history"]
CATEGORICAL_COVARIATES = [
    "history_segment",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
OUTCOMES = ["visit", "conversion", "spend"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 Hillstorm 实验数据。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["audit"])
    return parser.parse_args()


def pooled_smd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled


def main() -> None:
    args = parse_args()
    output = prepare_output(args.output)
    required = {
        "segment",
        *NUMERIC_COVARIATES,
        *CATEGORICAL_COVARIATES,
        *OUTCOMES,
    }
    df = read_hillstrom(args.data, required)
    validate_groups(df)
    validate_binary(df, ["mens", "womens", "newbie", "visit", "conversion"])

    print("=" * 82)
    print("Hillstorm 实验数据审计")
    print("=" * 82)
    print(f"输入：{args.data}")
    print(f"输出：{output}")

    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[column].dtype) for column in df.columns],
            "non_null": [int(df[column].notna().sum()) for column in df.columns],
            "unique_values": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )
    save_csv(schema, output, "01_schema.csv")

    missingness = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
            "missing_rate": [float(df[column].isna().mean()) for column in df.columns],
        }
    )
    save_csv(missingness, output, "02_missingness.csv")

    counts = df["segment"].value_counts().reindex(GROUP_ORDER)
    segment_counts = counts.rename("n").reset_index()
    segment_counts.columns = ["segment", "n"]
    segment_counts["share"] = segment_counts["n"] / len(df)
    save_csv(segment_counts, output, "03_segment_counts.csv")

    expected = np.repeat(len(df) / len(GROUP_ORDER), len(GROUP_ORDER))
    srm_stat, srm_p = stats.chisquare(counts.to_numpy(), f_exp=expected)
    srm = pd.DataFrame(
        [
            {
                "test": "chi_square_equal_allocation",
                "chi_square": srm_stat,
                "df": len(GROUP_ORDER) - 1,
                "p_value": srm_p,
                "srm_flag_alpha_0_01": srm_p < 0.01,
            }
        ]
    )
    save_csv(srm, output, "04_srm_test.csv")

    numeric_rows = []
    for variable in NUMERIC_COVARIATES:
        arrays = {
            group: df.loc[df["segment"] == group, variable].dropna().astype(float).to_numpy()
            for group in GROUP_ORDER
        }
        anova = stats.f_oneway(*arrays.values())
        smds = [
            abs(pooled_smd(arrays[a], arrays[b]))
            for i, a in enumerate(GROUP_ORDER)
            for b in GROUP_ORDER[i + 1 :]
        ]
        for group in GROUP_ORDER:
            values = arrays[group]
            numeric_rows.append(
                {
                    "variable": variable,
                    "segment": group,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "min": values.min(),
                    "max": values.max(),
                    "anova_p_value": anova.pvalue,
                    "max_absolute_pairwise_smd": max(smds),
                }
            )
    numeric_balance = pd.DataFrame(numeric_rows)
    save_csv(numeric_balance, output, "05_numeric_covariate_balance.csv")

    categorical_rows = []
    distribution_rows = []
    for variable in CATEGORICAL_COVARIATES:
        table = pd.crosstab(df["segment"], df[variable], dropna=False).reindex(GROUP_ORDER)
        chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
        n = table.to_numpy().sum()
        denominator = max(1, min(table.shape) - 1)
        cramer_v = np.sqrt(chi2 / (n * denominator))
        categorical_rows.append(
            {
                "variable": variable,
                "chi_square": chi2,
                "df": dof,
                "p_value": p_value,
                "cramers_v": cramer_v,
            }
        )
        for group in GROUP_ORDER:
            group_total = table.loc[group].sum()
            for level, count in table.loc[group].items():
                distribution_rows.append(
                    {
                        "variable": variable,
                        "level": str(level),
                        "segment": group,
                        "count": int(count),
                        "within_segment_share": count / group_total,
                    }
                )
    categorical_balance = pd.DataFrame(categorical_rows)
    save_csv(categorical_balance, output, "06_categorical_covariate_balance.csv")
    save_csv(
        pd.DataFrame(distribution_rows),
        output,
        "07_categorical_covariate_distribution.csv",
    )

    logic_checks = pd.DataFrame(
        [
            {
                "check": "conversion_without_visit",
                "violation_count": int(((df["conversion"] == 1) & (df["visit"] != 1)).sum()),
            },
            {
                "check": "positive_spend_without_conversion",
                "violation_count": int(((df["spend"] > 0) & (df["conversion"] != 1)).sum()),
            },
            {
                "check": "conversion_without_positive_spend",
                "violation_count": int(((df["conversion"] == 1) & (df["spend"] <= 0)).sum()),
            },
            {
                "check": "negative_spend",
                "violation_count": int((df["spend"] < 0).sum()),
            },
        ]
    )
    save_csv(logic_checks, output, "08_outcome_logic_checks.csv")

    outcome_profile = (
        df.groupby("segment", observed=True)[OUTCOMES]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reindex(GROUP_ORDER)
    )
    outcome_profile.columns = ["_".join(column) for column in outcome_profile.columns]
    save_csv(outcome_profile.reset_index(), output, "09_outcome_profile.csv")

    numeric_summary = df.select_dtypes(include=np.number).describe().T.reset_index(names="column")
    save_csv(numeric_summary, output, "10_numeric_summary.csv")

    duplicate_rows_after_first = int(df.duplicated().sum())
    rows_in_duplicate_groups = int(df.duplicated(keep=False).sum())
    critical_missing = int(missingness.loc[missingness["column"].isin(required), "missing_count"].sum())
    logic_violations = int(logic_checks["violation_count"].sum())
    status = "通过" if srm_p >= 0.01 and critical_missing == 0 and logic_violations == 0 else "需要复核"

    summary = f"""Hillstorm 实验数据审计摘要

状态：{status}
行数：{len(df):,}
列数：{df.shape[1]}
SRM 检验：卡方统计量 = {srm_stat:.6f}，{format_p(srm_p)}
关键字段缺失值：{critical_missing:,}
结果变量逻辑违规：{logic_violations:,}
首次出现之后的完全重复行：{duplicate_rows_after_first:,}
属于完全重复组的行数：{rows_in_duplicate_groups:,}

重复值处理
----------
数据中没有客户唯一标识符，因此无法确认完全相同的记录是否代表重复的实验单位。为避免误删有效观测，所有分析均保留这些记录。
"""
    (output / "00_audit_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("生成文件：")
    for path in sorted(output.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
