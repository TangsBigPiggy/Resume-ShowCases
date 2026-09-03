# 2.1 产品市场结构输出

本目录由 `2_1_product_landscape.py` 生成，输入为 Stage 1 的 `products_analysis.csv`。

- `figures/`：价格、评分、佩戴形态、价格带、品牌/店铺、商品定位和评论集中度图。
- `tables/`：市场概览、佩戴形态、价格带、品牌/店铺、评论集中度曲线与商品定位明细。

为控制仓库体积，未上传 `tables/product_positioning.csv`（约 8.98 MB）和 `tables/review_concentration_curve.csv`（约 2.37 MB）。前者是逐商品定位明细，后者是逐排名点的集中度曲线；两者均由脚本从 `products_analysis.csv` 重新生成。仓库保留对应汇总表和最终图表。

重新运行脚本会覆盖这些派生输出；如需使用其他输入或输出位置，请查看脚本的 `--help`。
