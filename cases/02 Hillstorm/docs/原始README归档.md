# Hillstorm 因果分析项目

本代码包实现一条完整且收敛的分析主线：实验审计、A/B/n 主分析、Spend 稳健性检验、预设单变量 HTE、样本外交叉拟合 Uplift、三动作策略评估、成本敏感性分析。

旧的单次 `60/20/20` Uplift 探索和重型嵌套验证未收入本包。

## 默认路径

所有脚本默认读取：

```text
E:\DA Cases\Hillstorm\0.原始数据\hillstorm_no_indices.csv\hillstorm_no_indices.csv
```

运行时可使用 `--data` 和 `--output` 覆盖。`run_all.py` 还支持 `--project-root`，默认值为 `E:\DA Cases\Hillstorm`。

## 环境安装

建议继续使用现有虚拟环境。在 PowerShell 中进入解压目录后执行：

```powershell
& "E:\DA Cases\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## 一次运行全部阶段

```powershell
& "E:\DA Cases\.venv\Scripts\python.exe" .\run_all.py
```

快速衔接测试可使用：

```powershell
& "E:\DA Cases\.venv\Scripts\python.exe" .\run_all.py --quick
```

`--quick` 会减少重采样次数、模型迭代和交叉拟合折数，仅用于检查环境及阶段衔接，不用于正式结论。

自定义项目根目录示例：

```powershell
& "E:\DA Cases\.venv\Scripts\python.exe" .\run_all.py `
  --project-root "E:\DA Cases\Hillstorm"
```

## 目录与输出

```text
Hillstorm_final_code
├── 1.实验数据审计
│   └── 01_experiment_data_audit.py
├── 2.AB Test
│   ├── 02_abn_experiment_analysis.py
│   └── 03_spend_robustness_checks.py
├── 3.异质性效应
│   └── 04_univariate_hte_analysis.py
├── 4.Uplift
│   └── 05_crossfitted_uplift_validation.py
├── 5.策略优化
│   ├── 06_multi_action_policy_evaluation.py
│   └── 07_cost_sensitivity_analysis.py
├── hillstrom_common.py
├── run_all.py
├── requirements.txt
└── README.md
```

默认输出目录：

| 阶段 | 输出目录 |
|---|---|
| 01 | `1.实验数据审计\results` |
| 02 | `2.AB Test\abn_experiment_analysis` |
| 03 | `2.AB Test\spend_robustness` |
| 04 | `3.异质性效应\univariate_hte_analysis` |
| 05 | `4.Uplift\crossfitted_uplift_validation` |
| 06 | `5.策略优化\multi_action_policy_evaluation` |
| 07 | `5.策略优化\cost_sensitivity_analysis` |

## 方法口径

- 主指标始终是所有随机客户的 `Spend per customer`；保留零消费，不做 treatment 后筛选。
- Stage 02 的三个预设 Spend 比较在同一 family 内使用 Holm 校正；区间明确标记为未做多重性调整的 nominal 95% CI。
- Stage 03 同时报告 Welch、组内 Bootstrap、随机化检验与 max-|t| 家族错误率控制。
- Stage 04 只把预设实验前变量作为 moderator；统计结论由交互项检验决定，而不是由子组内显著性决定。
- Stage 05 只保留可解释的 T-Learner 基准和 DR-Learner 主模型。每位客户的最终 CATE 预测均来自未使用该客户结果的外层训练折。
- Stage 06 使用随机实验已知分配概率 `1/3` 做 IPW 与 AIPW 策略价值评估，并比较 No Email、全量 Mens、全量 Womens 和个性化三动作策略。
- Stage 07 的成本是情景参数，不代表真实邮件成本或利润。Spend 不能直接解释为毛利或 ROI。

Stage 05 生成的 `08_oof_predictions.csv.gz` 是 Stage 06 和 Stage 07 的唯一衔接文件。后两个脚本会检查行号、实验组、结果和所需预测列。
