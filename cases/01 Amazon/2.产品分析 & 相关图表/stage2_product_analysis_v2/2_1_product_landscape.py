#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.1 产品市场结构分析
输入：products_analysis.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


DEFAULT_STAGE1_DIR = Path(
    r"E:\DA Cases\Amazon\1.EDA & 预处理\1.2 预处理"
    r"\preprocessed_pre_ManualReviewed\stage1_preprocessed"
)
DEFAULT_PRODUCTS = DEFAULT_STAGE1_DIR / "products_analysis.csv"

PRICE_BINS = [-np.inf, 25, 50, 100, 200, np.inf]
PRICE_LABELS = ["<$25", "$25-49", "$50-99", "$100-199", "$200+"]

# 统一科研配色
PRIMARY = "#4C6FAE"
PINK = "#E07A9D"
YELLOW = "#F1C15A"
GREEN = "#5FA77A"
CYAN = "#67B7C7"
PURPLE = "#C99BC1"
ORANGE = "#E49A45"
MINT = "#8BBF9D"
LIGHT_BLUE = "#A6C4E6"
LIGHT_PINK = "#E9B4C2"
DARK_TEXT = "#26354A"
GRID = "#D9E1EA"

PALETTE = [
    PRIMARY, PINK, YELLOW, GREEN, CYAN,
    PURPLE, ORANGE, MINT, LIGHT_BLUE, LIGHT_PINK,
]

FORM_FACTOR_COLOR = {
    "EARBUD_INEAR": PRIMARY,
    "OVER_EAR": PINK,
    "ON_EAR": CYAN,
    "UNKNOWN": PURPLE,
}

PRICE_COLORS = [PRIMARY, CYAN, YELLOW, ORANGE, PINK]

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "research_density",
    ["#EEF4F8", LIGHT_BLUE, CYAN, PRIMARY],
)


def setup_plot() -> None:
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": DARK_TEXT,
        "axes.labelcolor": DARK_TEXT,
        "axes.edgecolor": "#AEB8C3",
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
    })


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = {
        "product_id", "title", "store", "price", "average_rating",
        "analysis_review_count", "form_factor",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"products_analysis.csv 缺少字段: {missing}")

    numeric_cols = [
        "price", "average_rating", "rating_number",
        "source_review_count", "analysis_review_count",
        "observed_avg_rating", "verified_review_share",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["brand_store"] = (
        df["store"].fillna("").astype(str).str.strip().replace("", "Unknown")
    )
    df["form_factor"] = (
        df["form_factor"].fillna("UNKNOWN").astype(str).str.strip()
    )

    df.loc[df["price"] <= 0, "price"] = np.nan
    df.loc[~df["average_rating"].between(1, 5), "average_rating"] = np.nan
    df["analysis_review_count"] = (
        df["analysis_review_count"].fillna(0).clip(lower=0)
    )

    df["price_band"] = pd.cut(
        df["price"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=False,
    )
    df["log_review_count"] = np.log10(df["analysis_review_count"] + 1)

    return df


def market_overview(df: pd.DataFrame) -> pd.DataFrame:
    price = df["price"].dropna()
    rating = df["average_rating"].dropna()
    reviews = df["analysis_review_count"]

    rows = [
        ("device_products", len(df)),
        (
            "brand_store_count",
            df.loc[df["brand_store"] != "Unknown", "brand_store"].nunique(),
        ),
        ("price_fill_rate", df["price"].notna().mean()),
        ("rating_fill_rate", df["average_rating"].notna().mean()),
        ("price_p25", price.quantile(0.25) if len(price) else np.nan),
        ("price_median", price.median() if len(price) else np.nan),
        ("price_p75", price.quantile(0.75) if len(price) else np.nan),
        ("price_p90", price.quantile(0.90) if len(price) else np.nan),
        ("rating_mean", rating.mean() if len(rating) else np.nan),
        ("rating_median", rating.median() if len(rating) else np.nan),
        ("analysis_review_total", reviews.sum()),
        ("review_count_median", reviews.median()),
        ("review_count_p90", reviews.quantile(0.90)),
        ("products_with_analysis_reviews", int((reviews > 0).sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def form_factor_landscape(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("form_factor", dropna=False)
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
            median_price=("price", "median"),
            median_rating=("average_rating", "median"),
            median_reviews_per_product=("analysis_review_count", "median"),
        )
        .reset_index()
    )
    out["product_share"] = out["product_count"] / out["product_count"].sum()
    total_reviews = out["review_count"].sum()
    out["review_share"] = (
        out["review_count"] / total_reviews if total_reviews else 0
    )
    return out.sort_values("product_count", ascending=False)


def price_band_landscape(df: pd.DataFrame) -> pd.DataFrame:
    x = df[df["price_band"].notna()].copy()
    out = (
        x.groupby("price_band", observed=True)
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
            median_rating=("average_rating", "median"),
            median_reviews_per_product=("analysis_review_count", "median"),
        )
        .reindex(PRICE_LABELS)
        .reset_index()
    )
    out["product_share"] = out["product_count"] / out["product_count"].sum()
    total_reviews = out["review_count"].sum()
    out["review_share"] = (
        out["review_count"] / total_reviews if total_reviews else 0
    )
    return out


def brand_landscape(df: pd.DataFrame, min_products: int) -> pd.DataFrame:
    x = df[df["brand_store"] != "Unknown"].copy()

    out = (
        x.groupby("brand_store")
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
            median_price=("price", "median"),
            median_rating=("average_rating", "median"),
            mean_rating=("average_rating", "mean"),
            median_reviews_per_product=("analysis_review_count", "median"),
        )
        .reset_index()
    )

    out = out[out["product_count"] >= min_products].copy()
    total_reviews = x["analysis_review_count"].sum()
    out["review_share"] = (
        out["review_count"] / total_reviews if total_reviews else 0
    )

    return out.sort_values(
        ["review_count", "product_count"],
        ascending=False,
    )


def review_concentration(df: pd.DataFrame) -> pd.DataFrame:
    x = (
        df[["product_id", "analysis_review_count"]]
        .sort_values("analysis_review_count", ascending=False)
        .reset_index(drop=True)
    )

    total_reviews = x["analysis_review_count"].sum()
    n_products = len(x)

    rows = []
    for share in [0.01, 0.05, 0.10, 0.20]:
        n = max(1, math.ceil(n_products * share))
        review_count = x.head(n)["analysis_review_count"].sum()
        rows.append({
            "top_product_share": share,
            "product_count": n,
            "review_count": review_count,
            "review_share": (
                review_count / total_reviews if total_reviews else 0
            ),
        })

    return pd.DataFrame(rows)


def review_concentration_curve(df: pd.DataFrame) -> pd.DataFrame:
    x = (
        df[["product_id", "analysis_review_count"]]
        .sort_values("analysis_review_count", ascending=False)
        .reset_index(drop=True)
    )

    total_reviews = x["analysis_review_count"].sum()
    n = len(x)

    if n == 0:
        return pd.DataFrame(columns=["top_product_share", "review_share"])

    x["top_product_share"] = (np.arange(n) + 1) / n
    x["review_share"] = (
        x["analysis_review_count"].cumsum() / total_reviews
        if total_reviews else 0
    )

    origin = pd.DataFrame({
        "product_id": [""],
        "analysis_review_count": [0],
        "top_product_share": [0.0],
        "review_share": [0.0],
    })

    return pd.concat([origin, x], ignore_index=True)


def product_positioning(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "product_id", "title", "brand_store", "price", "price_band",
        "average_rating", "observed_avg_rating", "analysis_review_count",
        "log_review_count", "verified_review_share", "form_factor",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def get_form_colors(labels) -> list[str]:
    colors = []
    fallback = iter(PALETTE)
    fallback_map = {}

    for value in labels:
        key = str(value)
        if key in FORM_FACTOR_COLOR:
            colors.append(FORM_FACTOR_COLOR[key])
            continue
        if key not in fallback_map:
            fallback_map[key] = next(fallback, PRIMARY)
        colors.append(fallback_map[key])

    return colors


def pie_plot(
    labels,
    values,
    title: str,
    path: Path,
    value_name: str,
) -> None:
    labels = pd.Series(labels).astype(str).tolist()
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(values) & (values > 0)
    labels = [label for label, keep in zip(labels, mask) if keep]
    values = values[mask]

    if values.sum() <= 0:
        return

    colors = get_form_colors(labels)

    fig, ax = plt.subplots(figsize=(8.4, 6.1))
    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 0.05 else "<0.1%",
        startangle=90,
        counterclock=False,
        pctdistance=0.70,
        wedgeprops={"linewidth": 1.1, "edgecolor": "white"},
        textprops={"fontsize": 9, "weight": "semibold"},
    )

    for txt in autotexts:
        txt.set_color(DARK_TEXT)
        txt.set_bbox({
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 1.2,
        })

    legend_labels = [
        f"{label}  |  {int(value):,}"
        for label, value in zip(labels, values)
    ]

    ax.legend(
        wedges,
        legend_labels,
        title=value_name,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    ax.set_title(title, pad=16)
    ax.axis("equal")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def bar_plot(
    labels,
    values,
    title: str,
    ylabel: str,
    path: Path,
    horizontal: bool = False,
    colors=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    if colors is None:
        colors = PRIMARY

    if horizontal:
        pos = np.arange(len(labels))
        ax.barh(pos, values, color=colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values, color=colors)
        ax.tick_params(axis="x", rotation=35)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def draw_rating_violin(rating: pd.Series, path: Path) -> None:
    values = rating.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return

    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    iqr = q75 - q25
    lower = max(values.min(), q25 - 1.5 * iqr)
    upper = min(values.max(), q75 + 1.5 * iqr)

    fig, ax = plt.subplots(figsize=(5.8, 6.4))

    parts = ax.violinplot(
        [values],
        positions=[1],
        orientation="vertical",
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )

    for body in parts["bodies"]:
        body.set_facecolor(CYAN)
        body.set_edgecolor(PRIMARY)
        body.set_linewidth(1.2)
        body.set_alpha(0.58)

    ax.vlines(1, lower, upper, color=PRIMARY, linewidth=1.4, zorder=4)
    ax.hlines([lower, upper], 0.95, 1.05, color=PRIMARY, linewidth=1.4, zorder=4)

    rect = Rectangle(
        (0.91, q25),
        0.18,
        max(q75 - q25, 1e-6),
        facecolor="white",
        edgecolor=PRIMARY,
        linewidth=1.4,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.hlines(median, 0.91, 1.09, color=PINK, linewidth=2.0, zorder=6)

    ax.scatter([1], [median], s=22, color=PINK, zorder=7)

    ax.annotate(
        f"P75  {q75:.2f}",
        xy=(1.10, q75),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )
    ax.annotate(
        f"Median  {median:.2f}",
        xy=(1.10, median),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        fontsize=9.5,
        weight="semibold",
    )
    ax.annotate(
        f"P25  {q25:.2f}",
        xy=(1.10, q25),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )

    ax.set_title("Average rating distribution")
    ax.set_ylabel("Average rating")
    ax.set_xticks([1])
    ax.set_xticklabels(["All products"])
    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(1, 5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.70)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def shortest_dense_interval(
    values: np.ndarray,
    coverage: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values.sort()

    if len(values) < 2:
        return float(values[0]), float(values[0])

    coverage = float(np.clip(coverage, 0.50, 0.95))
    width_n = max(2, int(math.ceil(len(values) * coverage)))
    starts = values[: len(values) - width_n + 1]
    ends = values[width_n - 1 :]
    widths = ends - starts

    idx = int(np.argmin(widths))
    return float(starts[idx]), float(ends[idx])


def positioning_zoom_bounds(
    df: pd.DataFrame,
    coverage: float = 0.75,
) -> tuple[float, float, float, float, float]:
    x = df[
        df["price"].notna()
        & (df["price"] > 0)
        & df["average_rating"].notna()
        & (df["analysis_review_count"] > 0)
    ].copy()

    log_price = np.log10(x["price"].to_numpy(dtype=float))
    rating = x["average_rating"].to_numpy(dtype=float)

    lp_low, lp_high = shortest_dense_interval(log_price, coverage)
    r_low, r_high = shortest_dense_interval(rating, coverage)

    lp_width = max(lp_high - lp_low, 0.45)
    r_width = max(r_high - r_low, 0.55)

    lp_pad = lp_width * 0.08
    r_pad = r_width * 0.10

    price_low = 10 ** (lp_low - lp_pad)
    price_high = 10 ** (lp_high + lp_pad)
    rating_low = max(1.0, r_low - r_pad)
    rating_high = min(5.0, r_high + r_pad)

    in_zoom = (
        x["price"].between(price_low, price_high)
        & x["average_rating"].between(rating_low, rating_high)
    )
    zoom_share = float(in_zoom.mean())

    return (
        float(price_low),
        float(price_high),
        float(rating_low),
        float(rating_high),
        zoom_share,
    )


def _positioning_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["price"].notna()
        & (df["price"] > 0)
        & df["average_rating"].notna()
        & (df["analysis_review_count"] > 0)
    ].copy()


def draw_positioning_magnifier(
    df: pd.DataFrame,
    overview_path: Path,
    zoom_path: Path,
    top_n: int,
    label_n: int,
    zoom_coverage: float,
) -> None:
    x = _positioning_data(df)
    if x.empty:
        return

    (
        price_low,
        price_high,
        rating_low,
        rating_high,
        zoom_share,
    ) = positioning_zoom_bounds(x, zoom_coverage)

    # Fig09a：全局市场密度，并清晰标示放大区域
    fig, ax = plt.subplots(figsize=(10.4, 6.6))

    hb = ax.hexbin(
        x["price"],
        x["average_rating"],
        gridsize=(52, 30),
        xscale="log",        mincnt=1,
        bins="log",
        cmap=DENSITY_CMAP,
        linewidths=0.18,
        edgecolors="white",
        alpha=0.96,
    )

    colorbar = fig.colorbar(hb, ax=ax, pad=0.015)
    colorbar.set_label("Products per hexbin (log scale)")

    # 放大区域：单层半透明高亮框
    zoom_box = Rectangle(
        (price_low, rating_low),
        price_high - price_low,
        rating_high - rating_low,
        fill=True,
        facecolor=PINK,
        edgecolor=PINK,
        linewidth=2.2,
        linestyle="--",
        alpha=0.24,
        zorder=5,
    )
    ax.add_patch(zoom_box)

    anchor_x = price_high
    anchor_y = rating_low + (rating_high - rating_low) * 0.18
    ax.annotate(
        f"Zoom region\n{zoom_share:.1%} of products",
        xy=(anchor_x, anchor_y),
        xytext=(22, -8),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9.6,
        weight="semibold",
        color=DARK_TEXT,
        arrowprops={
            "arrowstyle": "-|>",
            "color": PINK,
            "linewidth": 1.5,
            "shrinkA": 2,
            "shrinkB": 6,
            "connectionstyle": "arc3,rad=0.08",
        },
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": PINK,
            "linewidth": 1.4,
            "alpha": 0.96,
        },
        zorder=8,
    )

    ax.set_title("Product positioning overview: price × rating density")
    ax.set_xlabel("Price (USD, log scale)")
    ax.set_ylabel("Average rating")
    ax.set_ylim(1, 5)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    overview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(overview_path)
    fig.savefig(overview_path.with_name("fig09_price_rating_positioning.png"))
    plt.close(fig)

    # Fig09b：放大核心密集区，编号高亮头部商品，右侧信息卡独立排版
    zoom = x[
        x["price"].between(price_low, price_high)
        & x["average_rating"].between(rating_low, rating_high)
    ].copy()
    if zoom.empty:
        return

    fig = plt.figure(figsize=(13.2, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 2.2], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis("off")

    use_log_x = (price_high / max(price_low, 1e-9)) > 20

    hb = ax.hexbin(
        zoom["price"],
        zoom["average_rating"],
        gridsize=(58, 34),
        xscale="log" if use_log_x else "linear",
        mincnt=1,
        bins="log",
        cmap=DENSITY_CMAP,
        linewidths=0.16,
        edgecolors="white",
        alpha=0.96,
    )

    colorbar = fig.colorbar(hb, ax=ax, pad=0.02)
    colorbar.set_label("Products per hexbin (log scale)")

    top = zoom.nlargest(max(1, top_n), "analysis_review_count").copy()
    top = top.reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1)
    top["bubble_size"] = 70 + 16 * np.sqrt(top["log_review_count"].clip(lower=0))

    colors = get_form_colors(top["form_factor"].tolist())
    ax.scatter(
        top["price"],
        top["average_rating"],
        s=top["bubble_size"],
        c=colors,
        edgecolors="white",
        linewidths=1.2,
        alpha=0.98,
        zorder=5,
    )

    for _, row in top.iterrows():
        ax.text(
            row["price"],
            row["average_rating"],
            str(int(row["rank"])),
            ha="center",
            va="center",
            fontsize=7.4,
            weight="bold",
            color="white",
            zorder=6,
        )

    ax.set_xlim(price_low, price_high)
    ax.set_ylim(rating_low, rating_high)
    ax.set_title("Zoomed core market: price × rating density", pad=10)
    ax.set_xlabel("Price (USD, log scale)" if use_log_x else "Price (USD)")
    ax.set_ylabel("Average rating")
    ax.grid(color=GRID, linewidth=0.7, alpha=0.42)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 右侧信息卡背景
    card = Rectangle(
        (0.00, 0.00),
        1.00,
        1.00,
        transform=info_ax.transAxes,
        facecolor="#FBFCFE",
        edgecolor="#E2E8F0",
        linewidth=1.0,
        zorder=0,
    )
    info_ax.add_patch(card)

    info_ax.text(
        0.05,
        0.96,
        "Highlighted products",
        transform=info_ax.transAxes,
        fontsize=12.5,
        weight="semibold",
        color=DARK_TEXT,
        va="top",
    )
    info_ax.text(
        0.05,
        0.905,
        (
            f"Zoom: ${price_low:,.0f}–${price_high:,.0f}\n"
            f"Rating: {rating_low:.2f}–{rating_high:.2f}\n"
            f"Products shown: {len(zoom):,} ({zoom_share:.1%})"
        ),
        transform=info_ax.transAxes,
        fontsize=9.1,
        color="#607086",
        va="top",
        linespacing=1.40,
    )

    info_ax.plot(
        [0.05, 0.95],
        [0.765, 0.765],
        transform=info_ax.transAxes,
        color="#E2E8F0",
        linewidth=0.9,
    )

    listed = top.head(max(1, min(label_n, len(top))))
    start_y = 0.72
    usable_bottom = 0.17
    step = min(0.068, max(0.050, (start_y - usable_bottom) / max(len(listed), 1)))
    y = start_y

    for _, row in listed.iterrows():
        rank = int(row["rank"])
        brand = str(row.get("brand_store", "")).strip()
        if not brand or brand == "Unknown":
            brand = str(row.get("title", "Product")).strip()
        brand = brand[:26]

        color = FORM_FACTOR_COLOR.get(str(row.get("form_factor", "")), PRIMARY)

        info_ax.scatter(
            [0.08],
            [y],
            s=94,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            transform=info_ax.transAxes,
            clip_on=False,
            zorder=2,
        )
        info_ax.text(
            0.08,
            y,
            str(rank),
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color="white",
            zorder=3,
        )
        info_ax.text(
            0.14,
            y + 0.012,
            brand,
            transform=info_ax.transAxes,
            fontsize=9.1,
            weight="semibold",
            color=DARK_TEXT,
            va="center",
        )
        info_ax.text(
            0.14,
            y - 0.015,
            (
                f"${row['price']:,.0f}  |  {row['average_rating']:.2f}★"
                f"  |  {int(row['analysis_review_count']):,} reviews"
            ),
            transform=info_ax.transAxes,
            fontsize=7.9,
            color="#6C7888",
            va="center",
        )
        y -= step

    present_forms = list(dict.fromkeys(top["form_factor"].astype(str)))
    legend_top = 0.11
    info_ax.text(
        0.05,
        legend_top + 0.05,
        "Form factor",
        transform=info_ax.transAxes,
        fontsize=9.3,
        weight="semibold",
        color=DARK_TEXT,
    )
    ly = legend_top
    for form in present_forms:
        color = FORM_FACTOR_COLOR.get(form, PRIMARY)
        info_ax.scatter(
            [0.08],
            [ly],
            s=48,
            color=color,
            transform=info_ax.transAxes,
            clip_on=False,
        )
        info_ax.text(
            0.12,
            ly,
            form,
            transform=info_ax.transAxes,
            fontsize=8.0,
            va="center",
            color=DARK_TEXT,
        )
        ly -= 0.035

    fig.savefig(zoom_path)
    plt.close(fig)

def draw_concentration_curve(
    curve: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    if curve.empty:
        return

    x = curve["top_product_share"].to_numpy(dtype=float) * 100
    y = curve["review_share"].to_numpy(dtype=float) * 100

    fig, ax = plt.subplots(figsize=(8.8, 5.7))

    ax.fill_between(x, y, color=LIGHT_BLUE, alpha=0.32)
    ax.plot(x, y, color=PRIMARY, linewidth=2.3, label="Observed concentration")
    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        linewidth=1.3,
        color="#9CA8B5",
        label="Equal-share baseline",
    )

    for _, row in summary.iterrows():
        xi = float(row["top_product_share"]) * 100
        yi = float(row["review_share"]) * 100

        ax.scatter([xi], [yi], s=46, color=PINK, edgecolor="white", linewidth=0.8, zorder=4)
        ax.vlines(xi, 0, yi, color=GRID, linewidth=0.8, linestyle=":")
        ax.annotate(
            f"Top {xi:.0f}% → {yi:.1f}%",
            xy=(xi, yi),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            weight="semibold",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
        )

    ax.set_title("Review concentration across products")
    ax.set_xlabel("Top products ranked by analysis review count (%)")
    ax.set_ylabel("Cumulative share of analysis reviews (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.45)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def create_figures(
    df: pd.DataFrame,
    form: pd.DataFrame,
    price_band: pd.DataFrame,
    brands: pd.DataFrame,
    concentration: pd.DataFrame,
    concentration_curve: pd.DataFrame,
    figdir: Path,
    fig09_top_n: int,
    fig09_label_n: int,
) -> None:
    figdir.mkdir(parents=True, exist_ok=True)

    price = df["price"].dropna()
    if len(price):
        upper = price.quantile(0.99)
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ax.hist(
            price[price <= upper],
            bins=50,
            color=PRIMARY,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        ax.set_title("Product price distribution (≤ P99)")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Products")
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(figdir / "fig01_price_distribution.png")
        plt.close(fig)

    draw_rating_violin(
        df["average_rating"],
        figdir / "fig02_rating_distribution.png",
    )

    pie_plot(
        form["form_factor"],
        form["product_count"],
        "Product share by form factor",
        figdir / "fig03_form_factor_products.png",
        "Products",
    )
    pie_plot(
        form["form_factor"],
        form["review_count"],
        "Analysis review share by form factor",
        figdir / "fig04_form_factor_reviews.png",
        "Analysis reviews",
    )

    bar_plot(
        price_band["price_band"].astype(str),
        price_band["product_count"],
        "Product count by price band",
        "Products",
        figdir / "fig05_price_band_products.png",
        colors=PRICE_COLORS,
    )
    bar_plot(
        price_band["price_band"].astype(str),
        price_band["review_count"],
        "Analysis review scale by price band",
        "Analysis reviews",
        figdir / "fig06_price_band_reviews.png",
        colors=PRICE_COLORS,
    )

    top_products = brands.nlargest(20, "product_count")
    bar_plot(
        top_products["brand_store"],
        top_products["product_count"],
        "Top Brand / Store by product count",
        "Products",
        figdir / "fig07_top_brands_products.png",
        horizontal=True,
        colors=PRIMARY,
    )

    top_reviews = brands.nlargest(20, "review_count")
    bar_plot(
        top_reviews["brand_store"],
        top_reviews["review_count"],
        "Top Brand / Store by analysis review scale",
        "Analysis reviews",
        figdir / "fig08_top_brands_reviews.png",
        horizontal=True,
        colors=PINK,
    )

    draw_positioning_magnifier(
        df,
        figdir / "fig09a_price_rating_positioning_overview.png",
        figdir / "fig09b_price_rating_positioning_zoom.png",
        top_n=fig09_top_n,
        label_n=fig09_label_n,
        zoom_coverage=0.75,
    )

    draw_concentration_curve(
        concentration_curve,
        concentration,
        figdir / "fig10_review_concentration.png",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2.1 Product landscape analysis"
    )
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--min-brand-products", type=int, default=3)
    ap.add_argument("--fig09-top-n", type=int, default=8)
    ap.add_argument("--fig09-label-n", type=int, default=8)
    args = ap.parse_args()

    setup_plot()

    products_path = Path(args.products)
    if not products_path.is_file():
        raise FileNotFoundError(products_path)

    root = (
        Path(args.outdir)
        if args.outdir
        else Path(__file__).resolve().parent
    )
    base = root / "stage2_outputs" / "2_1_product_landscape"
    tables = base / "tables"
    figures = base / "figures"

    print("products:", products_path)
    print("output  :", base)

    df = load_products(products_path)

    overview = market_overview(df)
    form = form_factor_landscape(df)
    price_band = price_band_landscape(df)
    brands = brand_landscape(df, args.min_brand_products)
    concentration = review_concentration(df)
    concentration_curve = review_concentration_curve(df)
    positioning = product_positioning(df)

    save_csv(overview, tables / "market_overview.csv")
    save_csv(form, tables / "form_factor_landscape.csv")
    save_csv(price_band, tables / "price_band_landscape.csv")
    save_csv(brands, tables / "brand_landscape.csv")
    save_csv(concentration, tables / "review_concentration.csv")
    save_csv(
        concentration_curve,
        tables / "review_concentration_curve.csv",
    )
    save_csv(positioning, tables / "product_positioning.csv")

    create_figures(
        df,
        form,
        price_band,
        brands,
        concentration,
        concentration_curve,
        figures,
        fig09_top_n=args.fig09_top_n,
        fig09_label_n=args.fig09_label_n,
    )

    print(f"products analyzed: {len(df):,}")
    print("DONE")


if __name__ == "__main__":
    main()