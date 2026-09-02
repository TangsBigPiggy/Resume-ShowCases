from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(r"E:\DA Cases\Hillstorm")
DEFAULT_DATA_PATH = Path(
    r"E:\DA Cases\Hillstorm\0.原始数据\hillstorm_no_indices.csv"
    r"\hillstorm_no_indices.csv"
)

GROUP_ORDER = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
PLANNED_CONTRASTS = [
    ("Mens E-Mail", "No E-Mail"),
    ("Womens E-Mail", "No E-Mail"),
    ("Mens E-Mail", "Womens E-Mail"),
]

PRETREATMENT_FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "channel",
    "zip_code",
]

DEFAULT_OUTPUTS = {
    "audit": DEFAULT_PROJECT_ROOT / "1.实验数据审计" / "results",
    "abn": DEFAULT_PROJECT_ROOT / "2.AB Test" / "abn_experiment_analysis",
    "robustness": DEFAULT_PROJECT_ROOT / "2.AB Test" / "spend_robustness",
    "hte": DEFAULT_PROJECT_ROOT / "3.异质性效应" / "univariate_hte_analysis",
    "uplift": DEFAULT_PROJECT_ROOT / "4.Uplift" / "crossfitted_uplift_validation",
    "policy": DEFAULT_PROJECT_ROOT / "5.策略优化" / "multi_action_policy_evaluation",
    "cost": DEFAULT_PROJECT_ROOT / "5.策略优化" / "cost_sensitivity_analysis",
}


def prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def read_hillstrom(
    path: str | Path,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def validate_groups(df: pd.DataFrame) -> None:
    observed = set(df["segment"].dropna().unique())
    expected = set(GROUP_ORDER)
    if observed != expected:
        raise ValueError(
            "Experimental groups do not match the expected design. "
            f"Observed: {sorted(observed)}"
        )


def validate_binary(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(
                f"{column} must be binary. Observed values: {sorted(values)}"
            )


def save_csv(df: pd.DataFrame, output: Path, filename: str) -> Path:
    path = output / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "p < 0.001" if value == 0 else f"p = {value:.3e}"
    return f"p = {value:.4f}"


def welch_mean_difference(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    difference = mean_a - mean_b
    se = np.sqrt(var_a / n_a + var_b / n_b)

    if se == 0:
        t_stat = 0.0
        df_welch = np.inf
        p_value = 1.0
        ci_low = ci_high = difference
    else:
        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (
            (var_a / n_a) ** 2 / (n_a - 1)
            + (var_b / n_b) ** 2 / (n_b - 1)
        )
        df_welch = numerator / denominator
        t_stat = difference / se
        p_value = 2 * stats.t.sf(abs(t_stat), df=df_welch)
        critical = stats.t.ppf(1 - alpha / 2, df=df_welch)
        ci_low = difference - critical * se
        ci_high = difference + critical * se

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": difference,
        "standard_error": se,
        "welch_df": df_welch,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def stratified_bootstrap_indices(
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = []
    for value in np.unique(strata):
        positions = np.flatnonzero(strata == value)
        sampled.append(rng.choice(positions, size=len(positions), replace=True))
    return np.concatenate(sampled)
