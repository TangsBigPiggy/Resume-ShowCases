# Amazon 耳机市场分析

本项目基于公开的 Amazon Reviews 2023 Electronics 数据，完成耳机类目的数据审计、清洗、商品与评论预处理、市场结构分析、消费者文本分析、商品聚类和行业报告。仓库保留分析代码、人工复核样本、可审计的汇总表、图表与最终报告；原始大数据、行级大表和可由代码重建的中间产物不进入 Git。

当前项目已经完成 Stage 1 与 Stage 2 的主要分析，并形成 `Amazon耳机市场行业分析报告_主报告_v1.0.docx`。当前本地目录未发现 `.pbix` 文件；`3.报告 & 可视化（Power BI）` 现阶段保存的是 Word 报告，Stage 2 的 CSV 表可作为后续 Power BI 数据源。

## 项目结构与阶段

```text
Amazon/
├─ 0.原始数据/
│  └─ 原始数据说明 & 下载链接.txt
├─ 1.EDA & 预处理/
│  ├─ 1.1 EDA/
│  │  ├─ 1_1_eda.py
│  │  └─ stage1_eda/{figures,output}/
│  └─ 1.2 预处理/
│     ├─ preprocessed__pos_ManualReviewed/
│     │  ├─ 1_2_preprocess_v1_baseline.py
│     │  └─ manual_audit_sample.csv
│     └─ preprocessed_pre_ManualReviewed/
│        ├─ 1_2_preprocess_final.py
│        ├─ manual_audit_sample_gold_reviewed.csv
│        └─ stage1_preprocessed/
├─ 2.产品分析 & 相关图表/
│  └─ stage2_product_analysis_v2/
│     ├─ 2_1_product_landscape.py
│     ├─ 2_2_market_text_analysis.py
│     ├─ 2_3_product_clustering.py
│     └─ stage2_outputs/
│        ├─ 2_1_product_landscape/{figures,tables}/
│        ├─ 2_2_market_text/{figures,tables}/
│        └─ 2_3_product_segmentation/{figures,tables}/
└─ 3.报告 & 可视化（Power BI）/
   └─ Amazon耳机市场行业分析报告_主报告_v1.0.docx
```

## 完整分析逻辑

1. **原始数据**：下载 Electronics 商品元数据和评论 JSONL，以 `parent_asin` 连接商品与评论。数据源、字段与下载地址见原始数据说明。
2. **EDA**：流式读取超大 JSONL，检查解析质量、缺失、类目层级、价格、评分、店铺、评论年份、验证购买、星级和文本长度，并输出轻量 CSV 与 PNG。
3. **预处理**：先用规则基线筛选耳机与配件并抽取人工审计样本；人工标注形成 Gold CSV 后，最终脚本训练/校准商品资格与佩戴形态模型，清洗并去重评论，输出商品主表和评论分析表。
4. **产品市场结构**：以 `products_analysis.csv` 计算市场规模、价格带、佩戴形态、品牌/店铺结构、评论集中度和商品定位，生成 2.1 图表与表格。
5. **消费者文本分析**：以商品主表和 `reviews_analysis.parquet` 构造商品级评论语料与词项矩阵，同时保留 Market-weighted 与 Product-equal 两种口径，分析全市场、价格带、评分组、头部/长尾和佩戴形态的关键词差异。
6. **产品聚类**：在文本词项过滤后执行 TF-IDF、TruncatedSVD、UMAP 和 HDBSCAN；KMeans 作为基准，并通过参数搜索和 UMAP 稳健性检查选择最终模型。最终输出商品 Segment、画像、关键词、质量诊断和可视化。
7. **报告与 Power BI**：Word 主报告将市场结构、消费者需求、可靠性风险、核心 Segment、机会判断和方法限制串联起来；Power BI 可直接读取 Stage 2 中保留的 CSV 表建立交互式视图。

当前成果口径约为 47,131 款有效耳机商品、47,119 款可进入完整文本/分群流程的商品和 2,963,761 条非空分析评论。评论量用于消费者反馈与注意力代理，不等同于销量或销售额。

## 环境与复现

建议使用 Python 3.10 或更高版本：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

按 `0.原始数据/原始数据说明 & 下载链接.txt` 下载并解压两个 JSONL 文件，保持与本地相同的嵌套目录，或通过各脚本的命令行参数传入实际路径。推荐按以下顺序运行：

```powershell
python ".\1.EDA & 预处理\1.1 EDA\1_1_eda.py" --help
python ".\1.EDA & 预处理\1.2 预处理\preprocessed__pos_ManualReviewed\1_2_preprocess_v1_baseline.py" --help
python ".\1.EDA & 预处理\1.2 预处理\preprocessed_pre_ManualReviewed\1_2_preprocess_final.py" --help
python ".\2.产品分析 & 相关图表\stage2_product_analysis_v2\2_1_product_landscape.py" --help
python ".\2.产品分析 & 相关图表\stage2_product_analysis_v2\2_2_market_text_analysis.py" --help
python ".\2.产品分析 & 相关图表\stage2_product_analysis_v2\2_3_product_clustering.py" --help
```

先完成基线抽样与人工复核，再运行最终预处理；Stage 2 必须按 2.1 → 2.2 → 2.3 顺序执行。完整文本分析与聚类计算量较大，建议保留充足的磁盘空间和内存。各阶段未上传文件及复现方法见对应目录内的说明。

## 仓库同步边界

- 不上传原始 JSONL、完整评论明细、较大的商品级派生表、Stage 2 cache、模型对象、稀疏矩阵、Parquet/NPZ、重复压缩包和临时文件。
- 代码文件按本地版本原样同步，不为 GitHub 改写路径、参数或分析逻辑。
- `.gitignore` 已覆盖上述文件；日常可在仓库根目录使用 `git add -- "cases/01 Amazon"`，再检查暂存清单后提交。
