from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.multitest import multipletests


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from hillstrom_common import (  # noqa: E402
    CONTROL,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUTS,
    GROUP_ORDER,
    PRETREATMENT_FEATURES,
    TREATMENTS,
    prepare_output,
    read_hillstrom,
    save_csv,
    validate_binary,
    validate_groups,
    welch_mean_difference,
)


ALPHA = 0.05
ACTION_PROBABILITY = 1 / 3
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["channel", "zip_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行交叉拟合 Uplift 验证。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS["uplift"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--model-iterations", type=int, default=150)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def make_model(args: argparse.Namespace, seed: int) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=args.model_iterations,
        max_leaf_nodes=15,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", regressor)])


def crossfit_predictions(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    x = df[PRETREATMENT_FEATURES]
    action = df["segment"].to_numpy()
    outcome = df["spend"].astype(float).to_numpy()
    action_to_index = {group: index for index, group in enumerate(GROUP_ORDER)}
    treatment_to_index = {group: index for index, group in enumerate(TREATMENTS)}

    mu_tlearner = np.full((n, len(GROUP_ORDER)), np.nan)
    tau_dr = np.full((n, len(TREATMENTS)), np.nan)
    folds = np.full(n, -1, dtype=int)
    fold_rows = []

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    for fold, (train_index, test_index) in enumerate(splitter.split(x, action), start=1):
        folds[test_index] = fold
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        action_train = action[train_index]
        outcome_train = outcome[train_index]

        train_mu = np.empty((len(train_index), len(GROUP_ORDER)))
        for action_name in GROUP_ORDER:
            action_index = action_to_index[action_name]
            action_mask = action_train == action_name
            model = make_model(args, args.seed + fold * 100 + action_index)
            model.fit(x_train.loc[action_mask], outcome_train[action_mask])
            train_mu[:, action_index] = model.predict(x_train)
            mu_tlearner[test_index, action_index] = model.predict(x_test)

        control_index = action_to_index[CONTROL]
        for treatment in TREATMENTS:
            treatment_index = action_to_index[treatment]
            contrast_index = treatment_to_index[treatment]
            pseudo_outcome = (
                train_mu[:, treatment_index]
                - train_mu[:, control_index]
                + (action_train == treatment)
                / ACTION_PROBABILITY
                * (outcome_train - train_mu[:, treatment_index])
                - (action_train == CONTROL)
                / ACTION_PROBABILITY
                * (outcome_train - train_mu[:, control_index])
            )
            cate_model = make_model(args, args.seed + fold * 1000 + contrast_index)
            cate_model.fit(x_train, pseudo_outcome)
            tau_dr[test_index, contrast_index] = cate_model.predict(x_test)

        fold_record = {"fold": fold, "train_n": len(train_index), "test_n": len(test_index)}
        for action_name in GROUP_ORDER:
            fold_record[f"train_{action_name}"] = int((action_train == action_name).sum())
            fold_record[f"test_{action_name}"] = int((action[test_index] == action_name).sum())
        fold_rows.append(fold_record)
        print(f"已完成第 {fold}/{args.folds} 折")

    if np.isnan(mu_tlearner).any() or np.isnan(tau_dr).any() or (folds < 1).any():
        raise RuntimeError("Cross-fitting did not produce a complete OOF prediction set.")

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "fold": folds,
            "segment": action,
            "spend": outcome,
            "mu_no_email_tlearner": mu_tlearner[:, action_to_index[CONTROL]],
            "mu_mens_tlearner": mu_tlearner[:, action_to_index["Mens E-Mail"]],
            "mu_womens_tlearner": mu_tlearner[:, action_to_index["Womens E-Mail"]],
            "tau_mens_tlearner": (
                mu_tlearner[:, action_to_index["Mens E-Mail"]]
                - mu_tlearner[:, action_to_index[CONTROL]]
            ),
            "tau_womens_tlearner": (
                mu_tlearner[:, action_to_index["Womens E-Mail"]]
                - mu_tlearner[:, action_to_index[CONTROL]]
            ),
            "tau_mens_drlearner": tau_dr[:, treatment_to_index["Mens E-Mail"]],
            "tau_womens_drlearner": tau_dr[:, treatment_to_index["Womens E-Mail"]],
        }
    )
    return predictions, pd.DataFrame(fold_rows)


def ranking_curve(
    subset: pd.DataFrame,
    score_column: str,
    treatment: str,
) -> tuple[dict, pd.DataFrame]:
    ranked = subset.sort_values(score_column, ascending=False).reset_index(drop=True)
    treated = (ranked["segment"] == treatment).to_numpy()
    outcome = ranked["spend"].to_numpy()
    conditional_probability = 0.5
    ipw_effect = np.where(
        treated,
        outcome / conditional_probability,
        -outcome / conditional_probability,
    )
    fraction = np.arange(1, len(ranked) + 1) / len(ranked)
    cumulative_gain = np.cumsum(ipw_effect) / len(ranked)
    overall_effect = ipw_effect.mean()
    random_baseline = fraction * overall_effect
    excess = cumulative_gain - random_baseline
    curve_y = np.r_[0.0, excess]
    curve_x = np.r_[0.0, fraction]
    if hasattr(np, "trapezoid"):
        auuc_excess = np.trapezoid(curve_y, x=curve_x)
    else:
        auuc_excess = np.trapz(curve_y, x=curve_x)

    grid = np.linspace(0, 1, 101)
    positions = np.maximum(1, np.ceil(grid[1:] * len(ranked)).astype(int)) - 1
    curve = pd.DataFrame(
        {
            "fraction_targeted": grid,
            "cumulative_incremental_spend": np.r_[0.0, cumulative_gain[positions]],
            "random_baseline": np.r_[0.0, random_baseline[positions]],
            "excess_gain": np.r_[0.0, excess[positions]],
        }
    )
    metrics = {
        "pairwise_sample_n": len(ranked),
        "overall_ipw_effect": overall_effect,
        "auuc_excess_over_random": auuc_excess,
    }
    return metrics, curve


def decile_table(
    subset: pd.DataFrame,
    score_column: str,
    treatment: str,
) -> pd.DataFrame:
    ranked = subset.copy()
    ranked["uplift_decile"] = pd.qcut(
        ranked[score_column].rank(method="first"),
        q=10,
        labels=False,
    ) + 1
    rows = []
    for decile, group in ranked.groupby("uplift_decile", observed=True):
        treated = group.loc[group["segment"] == treatment, "spend"].to_numpy()
        control = group.loc[group["segment"] == CONTROL, "spend"].to_numpy()
        result = welch_mean_difference(treated, control, alpha=ALPHA)
        rows.append(
            {
                "uplift_decile": int(decile),
                "mean_predicted_uplift": group[score_column].mean(),
                **result,
                "ci_label": "nominal_95pct_unadjusted",
            }
        )
    return pd.DataFrame(rows).sort_values("uplift_decile", ascending=False)


def top_bottom_bootstrap(
    subset: pd.DataFrame,
    score_column: str,
    treatment: str,
    replications: int,
    rng: np.random.Generator,
) -> dict:
    ranked = subset.sort_values(score_column, ascending=False).reset_index(drop=True)
    k = int(np.floor(0.30 * len(ranked)))
    if k < 10 or 2 * k >= len(ranked):
        raise ValueError("The pairwise sample is too small for a 30% top-bottom comparison.")
    ranked["rank_group"] = "middle"
    ranked.loc[: k - 1, "rank_group"] = "top_30pct"
    ranked.loc[len(ranked) - k :, "rank_group"] = "bottom_30pct"
    selected = ranked[ranked["rank_group"] != "middle"].copy()

    def effect(frame: pd.DataFrame, rank_group: str) -> float:
        group = frame[frame["rank_group"] == rank_group]
        treated = group.loc[group["segment"] == treatment, "spend"]
        control = group.loc[group["segment"] == CONTROL, "spend"]
        if len(treated) < 2 or len(control) < 2:
            raise ValueError(f"Insufficient treatment coverage in {rank_group}.")
        return treated.mean() - control.mean()

    top_effect = effect(selected, "top_30pct")
    bottom_effect = effect(selected, "bottom_30pct")
    observed_difference = top_effect - bottom_effect
    cells = {
        (rank_group, action): selected.index[
            (selected["rank_group"] == rank_group) & (selected["segment"] == action)
        ].to_numpy()
        for rank_group in ["top_30pct", "bottom_30pct"]
        for action in [CONTROL, treatment]
    }
    if any(len(indices) < 2 for indices in cells.values()):
        raise ValueError("A top-bottom treatment cell contains fewer than two observations.")

    differences = np.empty(replications)
    for replication in range(replications):
        sampled = {
            cell: rng.choice(indices, size=len(indices), replace=True)
            for cell, indices in cells.items()
        }
        top = (
            selected.loc[sampled[("top_30pct", treatment)], "spend"].mean()
            - selected.loc[sampled[("top_30pct", CONTROL)], "spend"].mean()
        )
        bottom = (
            selected.loc[sampled[("bottom_30pct", treatment)], "spend"].mean()
            - selected.loc[sampled[("bottom_30pct", CONTROL)], "spend"].mean()
        )
        differences[replication] = top - bottom

    low, high = np.quantile(differences, [ALPHA / 2, 1 - ALPHA / 2])
    left = (np.count_nonzero(differences <= 0) + 1) / (replications + 1)
    right = (np.count_nonzero(differences >= 0) + 1) / (replications + 1)
    return {
        "top_30pct_n": k,
        "bottom_30pct_n": k,
        "top_30pct_effect": top_effect,
        "bottom_30pct_effect": bottom_effect,
        "top_minus_bottom": observed_difference,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "bootstrap_p_value_two_sided": min(1.0, 2 * min(left, right)),
        "bootstrap_replications": replications,
        "ci_label": "fixed_score_percentile_95pct",
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.bootstrap < 1:
        raise ValueError("At least two folds and one bootstrap replication are required.")

    output = prepare_output(args.output)
    required = {"segment", "spend", *PRETREATMENT_FEATURES}
    df = read_hillstrom(args.data, required).reset_index(drop=True)
    validate_groups(df)
    validate_binary(df, ["mens", "womens", "newbie"])
    if df[["segment", "spend"]].isna().any().any():
        raise ValueError("Segment and Spend must not contain missing values.")
    minimum_group = df["segment"].value_counts().min()
    if minimum_group < args.folds:
        raise ValueError("Each experimental group must contain at least one row per fold.")

    print("=" * 82)
    print("Hillstorm 交叉拟合多变量 Uplift 验证")
    print("=" * 82)
    print(f"输入：{args.data}")
    print(f"输出：{output}")

    design = f"""Hillstorm 交叉拟合多变量 Uplift 验证设计

目标
----
检验实验前客户特征能否在平均随机处理效应之外，形成稳定的样本外 Uplift 排序。

模型
----
T-Learner 基准模型：分别拟合带正则化的梯度提升结果模型。
DR-Learner 主模型：使用随机对照试验已知倾向概率、结果模型干扰项估计，以及带正则化的因果伪结果回归。

验证方式
--------
采用 {args.folds} 折外层交叉拟合。每位客户最终的 CATE 预测均由未在外层训练折中使用该客户的模型生成。评估样本不用于模型选择，也不用于目标覆盖比例选择。

特征
----
{'、'.join(PRETREATMENT_FEATURES)}

证据标准
--------
使用相对随机投放基线的 Excess AUUC、Uplift 十分位结果，以及固定评分下 Top 30% 与 Bottom 30% 的 Bootstrap 差异比较。Top-Bottom 的 p 值在两个模型和两个处理比较之间统一进行 Holm 校正。

Bootstrap 重复次数：{args.bootstrap:,}
随机种子：{args.seed}
"""
    (output / "00_uplift_design.txt").write_text(design, encoding="utf-8")

    predictions, fold_summary = crossfit_predictions(df, args)
    save_csv(fold_summary, output, "01_fold_summary.csv")

    model_specs = [
        ("T-Learner", "Mens E-Mail", "tau_mens_tlearner"),
        ("T-Learner", "Womens E-Mail", "tau_womens_tlearner"),
        ("DR-Learner", "Mens E-Mail", "tau_mens_drlearner"),
        ("DR-Learner", "Womens E-Mail", "tau_womens_drlearner"),
    ]
    model_rows = []
    decile_frames = []
    top_bottom_rows = []
    curve_frames = []
    seed_sequence = np.random.SeedSequence(args.seed + 50_000)
    bootstrap_seeds = seed_sequence.spawn(len(model_specs))

    for index, (model_name, treatment, score_column) in enumerate(model_specs):
        subset = predictions[predictions["segment"].isin([CONTROL, treatment])].copy()
        metrics, curve = ranking_curve(subset, score_column, treatment)
        curve.insert(0, "treatment", treatment)
        curve.insert(0, "model", model_name)
        curve_frames.append(curve)

        deciles = decile_table(subset, score_column, treatment)
        deciles.insert(0, "treatment", treatment)
        deciles.insert(0, "model", model_name)
        decile_frames.append(deciles)

        top_bottom = top_bottom_bootstrap(
            subset,
            score_column,
            treatment,
            args.bootstrap,
            np.random.default_rng(bootstrap_seeds[index]),
        )
        top_bottom_rows.append(
            {"model": model_name, "treatment": treatment, "score_column": score_column, **top_bottom}
        )
        model_rows.append(
            {"model": model_name, "treatment": treatment, "score_column": score_column, **metrics}
        )

    top_bottom_table = pd.DataFrame(top_bottom_rows)
    reject, adjusted, _, _ = multipletests(
        top_bottom_table["bootstrap_p_value_two_sided"],
        alpha=ALPHA,
        method="holm",
    )
    top_bottom_table["bootstrap_p_value_holm"] = adjusted
    top_bottom_table["top_bottom_supported_holm"] = reject
    save_csv(top_bottom_table, output, "04_top_bottom_bootstrap.csv")

    model_summary = pd.DataFrame(model_rows).merge(
        top_bottom_table[
            [
                "model",
                "treatment",
                "top_30pct_effect",
                "bottom_30pct_effect",
                "top_minus_bottom",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_p_value_holm",
                "top_bottom_supported_holm",
            ]
        ],
        on=["model", "treatment"],
        validate="one_to_one",
    )
    model_summary["ranking_evidence_supported"] = (
        (model_summary["auuc_excess_over_random"] > 0)
        & (model_summary["bootstrap_ci_low"] > 0)
        & model_summary["top_bottom_supported_holm"]
    )
    save_csv(model_summary, output, "02_model_summary.csv")
    save_csv(pd.concat(decile_frames, ignore_index=True), output, "03_uplift_deciles.csv")
    save_csv(pd.concat(curve_frames, ignore_index=True), output, "05_uplift_curves.csv")

    decision_rows = []
    for model_name in ["T-Learner", "DR-Learner"]:
        rows = model_summary[model_summary["model"] == model_name]
        supported = rows.loc[rows["ranking_evidence_supported"], "treatment"].tolist()
        decision_rows.append(
            {
                "model": model_name,
                "role": "primary" if model_name == "DR-Learner" else "benchmark",
                "supported_treatment_count": len(supported),
                "supported_treatments": "; ".join(supported),
                "multivariate_ranking_evidence": "supported" if supported else "insufficient",
            }
        )
    decisions = pd.DataFrame(decision_rows)
    save_csv(decisions, output, "06_model_decision.csv")

    primary_supported = decisions.loc[
        decisions["model"] == "DR-Learner", "multivariate_ranking_evidence"
    ].iloc[0]
    primary_supported_label = "充分" if primary_supported == "supported" else "不足"
    decision_text = f"""交叉拟合 Uplift 决策结论

DR-Learner 主模型的排序证据：{primary_supported_label}

判定规则
--------
只有同时满足以下条件，才认为存在可靠的排序证据：Excess AUUC 为正，且 Top 30% 减 Bottom 30% 的 Bootstrap 区间在 Holm 校正检验后仍为正。若未满足该规则，则结论记为“证据不足”，不继续进行额外模型搜索。多动作策略价值由后续策略评估单独检验。
"""
    (output / "07_uplift_decision.txt").write_text(decision_text, encoding="utf-8")

    predictions.to_csv(
        output / "08_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8",
    )
    try:
        data_reference = args.data.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        data_reference = args.data.name
    metadata = {
        "data_path": data_reference,
        "rows": len(df),
        "folds": args.folds,
        "features": PRETREATMENT_FEATURES,
        "action_probability": ACTION_PROBABILITY,
        "bootstrap_replications": args.bootstrap,
        "model_iterations": args.model_iterations,
        "min_samples_leaf": args.min_samples_leaf,
        "random_seed": args.seed,
    }
    (output / "09_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n模型摘要")
    print(model_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n" + decision_text)


if __name__ == "__main__":
    main()
