from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from hillstrom_common import (  # noqa: E402
    CONTROL,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUTS,
    TREATMENTS,
    prepare_output,
    read_hillstrom,
    save_csv,
    validate_binary,
    validate_groups,
    welch_mean_difference,
)


ALPHA = 0.05
MODERATORS = [
    "recency_band",
    "history_segment",
    "mens",
    "womens",
    "newbie",
    "channel",
    "zip_code",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行预设单变量 HTE 检验。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["hte"])
    return parser.parse_args()


def interaction_indices(
    names: list[str],
    moderator: str,
    treatment: str | None = None,
) -> list[int]:
    indices = []
    for index, name in enumerate(names):
        if ":" not in name or f"C({moderator})" not in name or "C(segment" not in name:
            continue
        if treatment is None or f"[T.{treatment}]" in name:
            indices.append(index)
    return indices


def robust_wald(model, indices: list[int]) -> tuple[float, float, int]:
    if not indices:
        return np.nan, np.nan, 0
    restriction = np.zeros((len(indices), len(model.params)))
    for row, index in enumerate(indices):
        restriction[row, index] = 1.0
    test = model.wald_test(restriction, scalar=True)
    return float(test.statistic), float(test.pvalue), len(indices)


def main() -> None:
    args = parse_args()
    output = prepare_output(args.output)
    required = {
        "segment",
        "spend",
        "recency",
        "history_segment",
        "mens",
        "womens",
        "newbie",
        "channel",
        "zip_code",
    }
    df = read_hillstrom(args.data, required)
    validate_groups(df)
    validate_binary(df, ["mens", "womens", "newbie"])
    if df[list(required)].isna().any().any():
        raise ValueError("Required HTE fields must not contain missing values.")

    df["recency_band"] = pd.cut(
        df["recency"],
        bins=[-np.inf, 3, 6, 9, np.inf],
        labels=["1-3 months", "4-6 months", "7-9 months", "10+ months"],
        ordered=True,
    )
    for column in ["mens", "womens", "newbie"]:
        df[column] = df[column].map({0: "No", 1: "Yes"})

    print("=" * 82)
    print("Hillstorm 预设单变量 HTE 分析")
    print("=" * 82)
    print(f"输入：{args.data}")
    print(f"输出：{output}")

    design = """Hillstorm 预设单变量 HTE 分析设计

结果变量
--------
按随机分配客户计算的人均 Spend。

处理比较
--------
Mens E-Mail vs No E-Mail
Womens E-Mail vs No E-Mail

调节变量
--------
最近购买时间分组、历史消费分组、历史 Mens 类别购买、历史 Womens 类别购买、新客户状态、历史渠道以及 ZIP Code 类型。

统计推断
--------
OLS 交互模型采用 HC3 异方差稳健协方差。七个调节变量的全局交互检验统一进行 Holm 校正；各处理组的特定检验分别进行校正。子组效应报告 95% 置信区间，主要用于描述；是否存在异质性证据以交互项检验结果为准。
"""
    (output / "00_hte_design.txt").write_text(design, encoding="utf-8")

    subgroup_rows = []
    for moderator in MODERATORS:
        for level in df[moderator].dropna().unique():
            subset = df[df[moderator] == level]
            for treatment in TREATMENTS:
                treated = subset.loc[subset["segment"] == treatment, "spend"].astype(float).to_numpy()
                control = subset.loc[subset["segment"] == CONTROL, "spend"].astype(float).to_numpy()
                if len(treated) < 2 or len(control) < 2:
                    continue
                result = welch_mean_difference(treated, control, alpha=ALPHA)
                subgroup_rows.append(
                    {
                        "moderator": moderator,
                        "level": str(level),
                        "treatment": treatment,
                        "control": CONTROL,
                        **result,
                        "estimate_label": "subgroup_effect_estimate",
                        "ci_label": "nominal_95pct_unadjusted",
                    }
                )
    subgroup_effects = pd.DataFrame(subgroup_rows)
    save_csv(subgroup_effects, output, "01_subgroup_effect_estimates.csv")

    global_rows = []
    treatment_rows = []
    for moderator in MODERATORS:
        model_data = df[["spend", "segment", moderator]].dropna().copy()
        formula = (
            'spend ~ C(segment, Treatment(reference="No E-Mail"))'
            f" * C({moderator})"
        )
        model = smf.ols(formula=formula, data=model_data).fit(cov_type="HC3")
        names = list(model.params.index)
        statistic, p_value, terms = robust_wald(
            model, interaction_indices(names, moderator)
        )
        global_rows.append(
            {
                "moderator": moderator,
                "n": len(model_data),
                "interaction_terms": terms,
                "wald_chi_square": statistic,
                "p_value": p_value,
            }
        )
        for treatment in TREATMENTS:
            statistic, p_value, terms = robust_wald(
                model, interaction_indices(names, moderator, treatment)
            )
            treatment_rows.append(
                {
                    "moderator": moderator,
                    "treatment": treatment,
                    "n": len(model_data),
                    "interaction_terms": terms,
                    "wald_chi_square": statistic,
                    "p_value": p_value,
                }
            )

    global_tests = pd.DataFrame(global_rows)
    if global_tests["p_value"].isna().any():
        failed = global_tests.loc[global_tests["p_value"].isna(), "moderator"].tolist()
        raise RuntimeError(f"Interaction test could not be estimated for: {failed}")
    reject, adjusted, _, _ = multipletests(global_tests["p_value"], alpha=ALPHA, method="holm")
    global_tests["p_value_holm"] = adjusted
    global_tests["heterogeneity_supported"] = reject
    save_csv(global_tests, output, "02_global_heterogeneity_tests.csv")

    treatment_tests = pd.DataFrame(treatment_rows)
    treatment_tests["p_value_holm"] = np.nan
    treatment_tests["heterogeneity_supported"] = False
    for treatment in TREATMENTS:
        mask = treatment_tests["treatment"] == treatment
        if treatment_tests.loc[mask, "p_value"].isna().any():
            failed = treatment_tests.loc[mask & treatment_tests["p_value"].isna(), "moderator"].tolist()
            raise RuntimeError(f"Treatment-specific test failed for {treatment}: {failed}")
        reject, adjusted, _, _ = multipletests(
            treatment_tests.loc[mask, "p_value"], alpha=ALPHA, method="holm"
        )
        treatment_tests.loc[mask, "p_value_holm"] = adjusted
        treatment_tests.loc[mask, "heterogeneity_supported"] = reject
    save_csv(treatment_tests, output, "03_treatment_specific_heterogeneity_tests.csv")

    summary = global_tests[
        ["moderator", "p_value", "p_value_holm", "heterogeneity_supported"]
    ].rename(
        columns={
            "p_value": "global_p_value",
            "p_value_holm": "global_p_value_holm",
            "heterogeneity_supported": "global_heterogeneity_supported",
        }
    )
    for treatment, prefix in [("Mens E-Mail", "mens"), ("Womens E-Mail", "womens")]:
        part = treatment_tests[treatment_tests["treatment"] == treatment][
            ["moderator", "p_value", "p_value_holm", "heterogeneity_supported"]
        ].rename(
            columns={
                "p_value": f"{prefix}_p_value",
                "p_value_holm": f"{prefix}_p_value_holm",
                "heterogeneity_supported": f"{prefix}_heterogeneity_supported",
            }
        )
        summary = summary.merge(part, on="moderator", how="left", validate="one_to_one")
    save_csv(summary, output, "04_moderator_summary.csv")

    supported = set(global_tests.loc[global_tests["heterogeneity_supported"], "moderator"])
    supported_subgroups = subgroup_effects[subgroup_effects["moderator"].isin(supported)].copy()
    save_csv(supported_subgroups, output, "05_supported_subgroup_effects.csv")

    print("\n全局异质性检验")
    print(global_tests.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n获得统计支持的调节变量：")
    print("  " + ", ".join(sorted(supported)) if supported else "  Holm 校正后无。")


if __name__ == "__main__":
    main()
