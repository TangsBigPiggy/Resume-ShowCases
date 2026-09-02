from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hillstrom_common import DEFAULT_DATA_PATH, DEFAULT_PROJECT_ROOT


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete Hillstorm pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--robustness-replications", type=int, default=10_000)
    parser.add_argument("--policy-bootstrap", type=int, default=2_000)
    parser.add_argument("--model-iterations", type=int, default=150)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced settings for a pipeline smoke test only.",
    )
    return parser.parse_args()


def run(script: Path, arguments: list[str]) -> None:
    command = [sys.executable, str(script), *arguments]
    print(f"\nRunning {script.name}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.folds = 3
        args.robustness_replications = 100
        args.policy_bootstrap = 100
        args.model_iterations = 30

    root = args.project_root
    outputs = {
        "audit": root / "1.实验数据审计" / "results",
        "abn": root / "2.AB Test" / "abn_experiment_analysis",
        "robustness": root / "2.AB Test" / "spend_robustness",
        "hte": root / "3.异质性效应" / "univariate_hte_analysis",
        "uplift": root / "4.Uplift" / "crossfitted_uplift_validation",
        "policy": root / "5.策略优化" / "multi_action_policy_evaluation",
        "cost": root / "5.策略优化" / "cost_sensitivity_analysis",
    }
    data_args = ["--data", str(args.data)]
    seed_args = ["--seed", str(args.seed)]

    run(
        PACKAGE_ROOT / "1.实验数据审计" / "01_experiment_data_audit.py",
        [*data_args, "--output", str(outputs["audit"])],
    )
    run(
        PACKAGE_ROOT / "2.AB Test" / "02_abn_experiment_analysis.py",
        [*data_args, "--output", str(outputs["abn"])],
    )
    run(
        PACKAGE_ROOT / "2.AB Test" / "03_spend_robustness_checks.py",
        [
            *data_args,
            "--output",
            str(outputs["robustness"]),
            *seed_args,
            "--bootstrap",
            str(args.robustness_replications),
            "--permutations",
            str(args.robustness_replications),
        ],
    )
    run(
        PACKAGE_ROOT / "3.异质性效应" / "04_univariate_hte_analysis.py",
        [*data_args, "--output", str(outputs["hte"])],
    )
    run(
        PACKAGE_ROOT / "4.Uplift" / "05_crossfitted_uplift_validation.py",
        [
            *data_args,
            "--output",
            str(outputs["uplift"]),
            *seed_args,
            "--folds",
            str(args.folds),
            "--bootstrap",
            str(args.policy_bootstrap),
            "--model-iterations",
            str(args.model_iterations),
        ],
    )

    predictions = outputs["uplift"] / "08_oof_predictions.csv.gz"
    run(
        PACKAGE_ROOT / "5.策略优化" / "06_multi_action_policy_evaluation.py",
        [
            "--predictions",
            str(predictions),
            "--output",
            str(outputs["policy"]),
            *seed_args,
            "--bootstrap",
            str(args.policy_bootstrap),
        ],
    )
    run(
        PACKAGE_ROOT / "5.策略优化" / "07_cost_sensitivity_analysis.py",
        [
            "--predictions",
            str(predictions),
            "--output",
            str(outputs["cost"]),
        ],
    )

    print("\nHillstorm pipeline completed successfully.")


if __name__ == "__main__":
    main()
