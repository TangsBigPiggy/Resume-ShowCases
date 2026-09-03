#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.3 产品分群、竞争定位与最终汇总

输入：
- products_analysis.csv
- 2.2 输出的 product_term_counts.npz / product_order.csv / vocabulary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer

try:
    import umap
except ImportError as exc:
    raise SystemExit(
        '缺少 umap-learn，请先运行: pip install umap-learn hdbscan'
    ) from exc

try:
    import hdbscan
except ImportError as exc:
    raise SystemExit(
        '缺少 hdbscan，请先运行: pip install umap-learn hdbscan'
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_STAGE1_DIR = Path(
    r"E:\DA Cases\Amazon\1.EDA & 预处理\1.2 预处理"
    r"\preprocessed_pre_ManualReviewed\stage1_preprocessed"
)
DEFAULT_PRODUCTS = DEFAULT_STAGE1_DIR / "products_analysis.csv"

DEFAULT_STAGE2_DIR = Path(__file__).resolve().parent / "stage2_outputs"
DEFAULT_TEXT_DIR = DEFAULT_STAGE2_DIR / "2_2_market_text"

PRICE_BINS = [-np.inf, 25, 50, 100, 200, np.inf]
PRICE_LABELS = ["<$25", "$25-49", "$50-99", "$100-199", "$200+"]

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
NOISE_COLOR = "#B8BDC5"

SEGMENT_COLORS = [
    PRIMARY, PINK, YELLOW, GREEN, CYAN, PURPLE, ORANGE, MINT,
    LIGHT_BLUE, LIGHT_PINK, "#6F86B6", "#C68CB0",
    "#8FAE7D", "#C6975B", "#80A7B8", "#9A8FC0",
]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "segment_heatmap",
    [PRIMARY, "#F7F9FC", PINK],
)

TERM_TOKEN_RE = re.compile(r"[a-z0-9]+")

GENERIC_CLUSTER_STOPWORDS = {
    "good", "great", "love", "loves", "loved", "like", "likes", "liked",
    "nice", "pretty", "amazing", "awesome", "excellent", "perfect",
    "best", "better", "okay", "ok", "decent", "recommend",
    "recommended", "highly", "really", "just", "very", "much",
    "work", "works", "worked", "working", "use", "used", "using",
    "got", "get", "gets", "getting", "bought", "buy", "purchase",
    "purchased", "one", "two", "three", "day", "days", "month",
    "months", "year", "years", "time", "times", "new",
    "son", "daughter", "husband", "wife", "kid", "kids", "child",
    "children", "gift", "christmas", "money", "product", "amazon",
    "headphone", "headphones", "earphone", "earphones",
    "earbud", "earbuds", "headset", "headsets", "bud", "buds",
    "ear", "ears", "ear buds", "ear bud",
    "ty", "does", "didn", "didnt", "dont", "doesnt",
}

PROTECTED_DOMAIN_TOKENS = {
    "sound", "quality", "audio", "bass", "treble", "volume",
    "battery", "life", "charge", "charging", "case",
    "noise", "cancellation", "canceling", "cancelling", "anc",
    "comfort", "comfortable", "fit", "tips", "tip",
    "bluetooth", "wireless", "connection", "connectivity",
    "disconnect", "latency", "gaming", "game",
    "microphone", "mic", "call", "calls", "voice",
    "cable", "wired", "durability", "durable", "broken",
    "broke", "stopped", "return", "replacement",
    "value", "price", "cheap", "expensive",
    "codec", "ldac", "aac", "spatial", "multipoint",
    "waterproof", "sweat", "sport", "running",
    "headband", "cups", "cup", "padding", "weight",
}

FORM_LABELS = {
    "EARBUD_INEAR": "Earbud / In-ear",
    "OVER_EAR": "Over-ear",
    "ON_EAR": "On-ear",
    "UNKNOWN": "Unknown",
}


def setup_plot() -> None:
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": DARK_TEXT,
        "axes.labelcolor": DARK_TEXT,
        "axes.edgecolor": "#AEB8C3",
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
    })


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="png",
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"图片保存失败: {path}")

    print(
        f"  saved: {path.name} "
        f"({path.stat().st_size / 1024:.1f} KB)"
    )


def load_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = {
        "product_id", "title", "store", "price",
        "average_rating", "analysis_review_count", "form_factor",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"products_analysis.csv 缺少字段: {missing}")

    numeric_cols = [
        "price", "average_rating", "rating_number",
        "analysis_review_count", "verified_review_share",
        "observed_avg_rating",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
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
    df["log_review_count"] = np.log10(df["analysis_review_count"] + 1)

    df["price_band"] = pd.cut(
        df["price"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=False,
    )

    if "verified_review_share" not in df.columns:
        df["verified_review_share"] = np.nan

    return df


def load_text_assets(text_dir: Path):
    cache = text_dir / "cache"

    matrix_path = cache / "product_term_counts.npz"
    order_path = cache / "product_order.csv"
    vocab_path = cache / "vocabulary.json"

    for path in [matrix_path, order_path, vocab_path]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少 2.2 输出文件: {path}")

    counts = sparse.load_npz(matrix_path).tocsr()
    order = pd.read_csv(order_path)
    order["product_id"] = order["product_id"].astype(str).str.strip()

    with vocab_path.open("r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    features = np.empty(len(vocabulary), dtype=object)
    for term, index in vocabulary.items():
        features[int(index)] = term

    if counts.shape[0] != len(order):
        raise ValueError(
            "product_term_counts.npz 与 product_order.csv 行数不一致。"
        )

    return counts, order, features


def prepare_product_matrix(
    products: pd.DataFrame,
    counts,
    order: pd.DataFrame,
):
    merged = (
        order.merge(
            products,
            on="product_id",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    if len(merged) != len(order):
        missing = len(order) - len(merged)
        raise ValueError(
            f"有 {missing:,} 个文本商品无法匹配 products_analysis.csv。"
        )

    return merged, counts


def normalize_term(term: str) -> str:
    return " ".join(TERM_TOKEN_RE.findall(str(term).lower()))


def term_tokens(term: str) -> list[str]:
    return TERM_TOKEN_RE.findall(str(term).lower())


def build_brand_stop_terms(
    products: pd.DataFrame,
    min_products: int,
):
    brand_counts = (
        products.loc[
            products["brand_store"].ne("Unknown"),
            "brand_store",
        ]
        .value_counts()
    )

    brand_phrases = set()
    token_weight = {}

    for brand, count in brand_counts.items():
        if count < min_products:
            continue

        normalized = normalize_term(brand)
        if not normalized:
            continue

        brand_phrases.add(normalized)

        for token in normalized.split():
            if len(token) < 3:
                continue
            if token in PROTECTED_DOMAIN_TOKENS:
                continue
            token_weight[token] = token_weight.get(token, 0) + int(count)

    brand_tokens = {
        token
        for token, count in token_weight.items()
        if count >= min_products
    }

    return brand_phrases, brand_tokens


def filter_cluster_features(
    counts,
    features,
    products: pd.DataFrame,
    min_df: int,
    max_df_ratio: float,
    brand_min_products: int,
):
    n_products = counts.shape[0]

    document_frequency = np.asarray(
        counts.getnnz(axis=0)
    ).ravel().astype(int)

    corpus_count = np.asarray(
        counts.sum(axis=0)
    ).ravel().astype(np.int64)

    brand_phrases, brand_tokens = build_brand_stop_terms(
        products,
        min_products=brand_min_products,
    )

    rows = []
    keep_mask = np.ones(len(features), dtype=bool)

    for i, term in enumerate(features):
        normalized = normalize_term(term)
        tokens = term_tokens(term)
        ngram_n = len(tokens)

        reason = "KEEP"

        if document_frequency[i] < min_df:
            reason = "LOW_DOCUMENT_FREQUENCY"
        elif document_frequency[i] / n_products > max_df_ratio:
            reason = "HIGH_DOCUMENT_FREQUENCY"
        elif not tokens:
            reason = "EMPTY"
        elif normalized in GENERIC_CLUSTER_STOPWORDS:
            reason = "GENERIC_TERM"
        elif all(token in GENERIC_CLUSTER_STOPWORDS for token in tokens):
            reason = "GENERIC_TERM"
        elif normalized in brand_phrases:
            reason = "BRAND_STORE"
        elif any(token in brand_tokens for token in tokens):
            reason = "BRAND_STORE"

        keep = reason == "KEEP"
        keep_mask[i] = keep

        rows.append({
            "term": term,
            "normalized_term": normalized,
            "ngram_n": ngram_n,
            "document_frequency": int(document_frequency[i]),
            "document_share": float(document_frequency[i] / n_products),
            "corpus_count": int(corpus_count[i]),
            "keep_for_clustering": keep,
            "filter_reason": reason,
        })

    audit = pd.DataFrame(rows)

    if keep_mask.sum() < 500:
        raise ValueError(
            f"聚类词汇过滤后仅剩 {keep_mask.sum():,} 个特征。"
        )

    filtered_counts = counts[:, keep_mask].tocsr()
    filtered_features = np.asarray(features)[keep_mask]

    return filtered_counts, filtered_features, audit


def build_text_embedding(
    counts,
    text_components: int,
):
    tfidf_transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf = tfidf_transformer.fit_transform(counts).tocsr()

    n_components = min(
        text_components,
        tfidf.shape[0] - 1,
        tfidf.shape[1] - 1,
    )
    if n_components < 2:
        raise ValueError("文本特征维度不足，无法执行 SVD。")

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
    )
    svd_features = svd.fit_transform(tfidf)

    normalizer = Normalizer(norm="l2")
    embedding = normalizer.fit_transform(svd_features)

    explained_variance = float(
        svd.explained_variance_ratio_.sum()
    )

    return (
        tfidf,
        embedding,
        svd_features,
        explained_variance,
        tfidf_transformer,
        svd,
        normalizer,
    )


def fit_kmeans_candidate(
    x,
    k: int,
    stability_runs: int,
    silhouette_sample: int,
):
    labels_runs = []
    models = []

    for run in range(stability_runs):
        model = KMeans(
            n_clusters=k,
            random_state=42 + run,
            n_init=10,
            max_iter=500,
            algorithm="lloyd",
        )
        labels = model.fit_predict(x)
        models.append(model)
        labels_runs.append(labels)

    labels = labels_runs[0]

    sample_size = min(
        silhouette_sample,
        len(labels),
    )
    silhouette = silhouette_score(
        x,
        labels,
        metric="euclidean",
        sample_size=sample_size,
        random_state=42,
    )

    ari_values = []
    for i in range(len(labels_runs)):
        for j in range(i + 1, len(labels_runs)):
            ari_values.append(
                adjusted_rand_score(
                    labels_runs[i],
                    labels_runs[j],
                )
            )

    stability = (
        float(np.mean(ari_values))
        if ari_values else 1.0
    )

    shares = pd.Series(labels).value_counts(normalize=True)

    row = {
        "model": "KMEANS",
        "k": int(k),
        "silhouette": float(silhouette),
        "stability_ari": stability,
        "inertia": float(models[0].inertia_),
        "largest_cluster_share": float(shares.max()),
    }

    return row, models[0], labels


def evaluate_kmeans(
    x,
    k_min: int,
    k_max: int,
    k_extend_max: int,
    stability_runs: int,
    silhouette_sample: int,
):
    rows = []
    candidates = {}

    def evaluate_range(start, end):
        for k in range(start, end + 1):
            print(f"  KMeans k={k}")
            row, model, labels = fit_kmeans_candidate(
                x,
                k=k,
                stability_runs=stability_runs,
                silhouette_sample=silhouette_sample,
            )
            rows.append(row)
            candidates[k] = (model, labels)

    evaluate_range(k_min, k_max)

    evaluation = pd.DataFrame(rows)
    evaluation["selection_score"] = (
        evaluation["silhouette"]
        * evaluation["stability_ari"].clip(lower=0)
    )

    best_k = int(
        evaluation.sort_values(
            ["selection_score", "silhouette", "stability_ari"],
            ascending=False,
        ).iloc[0]["k"]
    )

    if best_k == k_max and k_extend_max > k_max:
        evaluate_range(k_max + 1, k_extend_max)

        evaluation = pd.DataFrame(rows)
        evaluation["selection_score"] = (
            evaluation["silhouette"]
            * evaluation["stability_ari"].clip(lower=0)
        )
        best_k = int(
            evaluation.sort_values(
                ["selection_score", "silhouette", "stability_ari"],
                ascending=False,
            ).iloc[0]["k"]
        )

    model, labels = candidates[best_k]

    return (
        evaluation.sort_values("k").reset_index(drop=True),
        best_k,
        model,
        labels,
    )


def build_umap_embedding(
    x,
    n_components: int,
    n_neighbors: int,
    random_state: int = 42,
    min_dist: float = 0.0,
):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
        low_memory=True,
        verbose=False,
    )
    embedding = reducer.fit_transform(x)
    return reducer, embedding


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return sorted({
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    })


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return sorted({
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    })


def default_mcs_values(
    n_products: int,
    ratios: list[float],
) -> list[int]:
    return sorted({
        max(50, int(round(n_products * ratio)))
        for ratio in ratios
    })


def fit_hdbscan_candidate(
    embedding,
    min_cluster_size: int,
    min_samples: int,
    silhouette_sample: int,
):
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        gen_min_span_tree=True,
        approx_min_span_tree=True,
        core_dist_n_jobs=-1,
    )
    labels = model.fit_predict(embedding)

    assigned = labels != -1
    unique_clusters = sorted(
        int(label)
        for label in np.unique(labels)
        if label != -1
    )
    cluster_count = len(unique_clusters)
    noise_rate = float(np.mean(labels == -1))

    silhouette = np.nan
    if cluster_count >= 2 and assigned.sum() >= 10:
        sample_size = min(
            silhouette_sample,
            int(assigned.sum()),
        )
        silhouette = float(
            silhouette_score(
                embedding[assigned],
                labels[assigned],
                metric="euclidean",
                sample_size=sample_size,
                random_state=42,
            )
        )

    persistence = getattr(
        model,
        "cluster_persistence_",
        np.array([], dtype=float),
    )

    mean_persistence = (
        float(np.mean(persistence))
        if len(persistence) else np.nan
    )
    min_persistence = (
        float(np.min(persistence))
        if len(persistence) else np.nan
    )

    try:
        relative_validity = float(model.relative_validity_)
    except Exception:
        relative_validity = np.nan

    if cluster_count:
        shares = (
            pd.Series(labels[assigned])
            .value_counts(normalize=True)
        )
        largest_cluster_share = float(shares.max())
    else:
        largest_cluster_share = np.nan

    row = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "cluster_count": int(cluster_count),
        "noise_rate": noise_rate,
        "assigned_share": 1 - noise_rate,
        "silhouette_assigned": silhouette,
        "relative_validity": relative_validity,
        "mean_cluster_persistence": mean_persistence,
        "min_cluster_persistence": min_persistence,
        "largest_cluster_share_assigned": largest_cluster_share,
    }

    return row, model, labels


def apply_hdbscan_quality_gates(
    evaluation: pd.DataFrame,
    max_noise_rate: float,
    min_persistence: float,
    max_clusters: int,
    max_largest_cluster_share: float,
) -> pd.DataFrame:
    x = evaluation.copy()

    x["quality_pass"] = (
        x["cluster_count"].between(3, max_clusters)
        & (x["noise_rate"] <= max_noise_rate)
        & x["mean_cluster_persistence"].ge(min_persistence)
        & x["relative_validity"].gt(0)
        & x["largest_cluster_share_assigned"].le(
            max_largest_cluster_share
        )
    )

    return x


def pareto_frontier_mask(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=bool)

    metrics = np.column_stack([
        df["relative_validity"].fillna(-np.inf).to_numpy(),
        df["mean_cluster_persistence"].fillna(-np.inf).to_numpy(),
        df["silhouette_assigned"].fillna(-np.inf).to_numpy(),
        df["assigned_share"].fillna(-np.inf).to_numpy(),
        -df["largest_cluster_share_assigned"]
        .fillna(np.inf)
        .to_numpy(),
    ])

    frontier = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not frontier[i]:
            continue
        for j in range(len(df)):
            if i == j:
                continue

            no_worse = np.all(metrics[j] >= metrics[i])
            strictly_better = np.any(metrics[j] > metrics[i])

            if no_worse and strictly_better:
                frontier[i] = False
                break

    return frontier


def score_hdbscan_candidates(
    evaluation: pd.DataFrame,
    max_noise_rate: float,
    min_persistence: float,
    max_clusters: int,
    max_largest_cluster_share: float,
) -> pd.DataFrame:
    x = apply_hdbscan_quality_gates(
        evaluation,
        max_noise_rate=max_noise_rate,
        min_persistence=min_persistence,
        max_clusters=max_clusters,
        max_largest_cluster_share=max_largest_cluster_share,
    )

    x["pareto_frontier"] = False
    x["composite_score"] = np.nan

    passing = x[x["quality_pass"]].copy()
    if passing.empty:
        return x

    x.loc[
        passing.index,
        "pareto_frontier",
    ] = pareto_frontier_mask(passing)

    rank_metrics = {
        "relative_validity": 0.30,
        "mean_cluster_persistence": 0.25,
        "silhouette_assigned": 0.20,
        "assigned_share": 0.15,
    }

    score = pd.Series(
        0.0,
        index=passing.index,
        dtype=float,
    )

    for column, weight in rank_metrics.items():
        score += (
            passing[column]
            .rank(pct=True, method="average")
            .fillna(0)
            * weight
        )

    compactness_rank = (
        (-passing["largest_cluster_share_assigned"])
        .rank(pct=True, method="average")
        .fillna(0)
    )
    score += compactness_rank * 0.10

    x.loc[passing.index, "composite_score"] = score
    return x


def evaluate_hdbscan_grid(
    embedding,
    pairs: list[tuple[int, int]],
    stage: str,
    silhouette_sample: int,
    existing: dict,
):
    rows = []

    for min_cluster_size, min_samples in pairs:
        key = (int(min_cluster_size), int(min_samples))
        if key in existing:
            continue

        print(
            f"  HDBSCAN [{stage}] "
            f"min_cluster_size={key[0]}, "
            f"min_samples={key[1]}"
        )

        row, model, labels = fit_hdbscan_candidate(
            embedding,
            min_cluster_size=key[0],
            min_samples=key[1],
            silhouette_sample=silhouette_sample,
        )
        row["search_stage"] = stage

        existing[key] = {
            "model": model,
            "labels": labels,
            "metrics": row,
        }
        rows.append(row)

    return rows


def select_best_hdbscan(
    evaluation: pd.DataFrame,
):
    passing = evaluation[
        evaluation["quality_pass"]
    ].copy()

    if passing.empty:
        return None

    frontier = passing[
        passing["pareto_frontier"]
    ].copy()

    pool = frontier if not frontier.empty else passing

    best = pool.sort_values(
        [
            "composite_score",
            "relative_validity",
            "mean_cluster_persistence",
            "silhouette_assigned",
            "noise_rate",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]

    return (
        int(best["min_cluster_size"]),
        int(best["min_samples"]),
    )


def boundary_extension_pairs(
    evaluation: pd.DataFrame,
    best_key: tuple[int, int],
    n_products: int,
):
    best_mcs, best_ms = best_key

    mcs_values = sorted(
        evaluation["min_cluster_size"].unique().astype(int)
    )
    ms_values = sorted(
        evaluation["min_samples"].unique().astype(int)
    )

    new_mcs = []
    new_ms = []

    if best_mcs == max(mcs_values):
        new_mcs.extend([
            min(int(round(best_mcs * 1.35)), int(n_products * 0.08)),
            min(int(round(best_mcs * 1.70)), int(n_products * 0.08)),
        ])
    elif best_mcs == min(mcs_values):
        new_mcs.extend([
            max(50, int(round(best_mcs * 0.50))),
            max(50, int(round(best_mcs * 0.75))),
        ])

    if best_ms == max(ms_values):
        new_ms.extend([
            max(best_ms + 1, int(round(best_ms * 1.50))),
            max(best_ms + 2, int(round(best_ms * 2.00))),
        ])
    elif best_ms == min(ms_values):
        new_ms.extend([
            max(2, int(round(best_ms * 0.40))),
            max(3, int(round(best_ms * 0.70))),
        ])

    new_mcs = sorted({
        value for value in new_mcs
        if 50 <= value < n_products
    })
    new_ms = sorted({
        value for value in new_ms
        if 2 <= value < max(1000, n_products)
    })

    pairs = []

    if new_mcs:
        neighbor_ms = sorted({
            best_ms,
            max(2, int(round(best_ms * 0.75))),
            max(2, int(round(best_ms * 1.25))),
        })
        pairs.extend(
            (mcs, ms)
            for mcs in new_mcs
            for ms in neighbor_ms
        )

    if new_ms:
        neighbor_mcs = sorted({
            best_mcs,
            max(50, int(round(best_mcs * 0.85))),
            min(
                n_products - 1,
                int(round(best_mcs * 1.15)),
            ),
        })
        pairs.extend(
            (mcs, ms)
            for mcs in neighbor_mcs
            for ms in new_ms
        )

    return sorted(set(pairs))


def local_refinement_pairs(
    best_key: tuple[int, int],
    n_products: int,
):
    best_mcs, best_ms = best_key

    mcs_values = sorted({
        max(50, min(
            n_products - 1,
            int(round(best_mcs * factor)),
        ))
        for factor in [0.72, 0.86, 1.00, 1.14, 1.28]
    })

    ms_values = sorted({
        max(2, int(round(best_ms * factor)))
        for factor in [0.50, 0.75, 1.00, 1.25, 1.60]
    })

    return [
        (mcs, ms)
        for mcs in mcs_values
        for ms in ms_values
    ]


def adaptive_hdbscan_search(
    embedding,
    n_products: int,
    coarse_mcs: list[int],
    coarse_ms: list[int],
    refine_rounds: int,
    silhouette_sample: int,
    max_noise_rate: float,
    min_persistence: float,
    max_clusters: int,
    max_largest_cluster_share: float,
):
    candidates = {}
    history = []

    coarse_pairs = [
        (mcs, ms)
        for mcs in coarse_mcs
        for ms in coarse_ms
    ]

    history.extend(
        evaluate_hdbscan_grid(
            embedding,
            coarse_pairs,
            stage="COARSE",
            silhouette_sample=silhouette_sample,
            existing=candidates,
        )
    )

    evaluation = score_hdbscan_candidates(
        pd.DataFrame(history),
        max_noise_rate=max_noise_rate,
        min_persistence=min_persistence,
        max_clusters=max_clusters,
        max_largest_cluster_share=max_largest_cluster_share,
    )
    best_key = select_best_hdbscan(evaluation)

    if best_key is None:
        return evaluation, None, None, None, {
            "boundary_extended": False,
            "refinement_rounds_completed": 0,
            "final_boundary_warning": None,
        }

    boundary_pairs = boundary_extension_pairs(
        evaluation,
        best_key,
        n_products=n_products,
    )

    boundary_extended = bool(boundary_pairs)

    if boundary_pairs:
        history.extend(
            evaluate_hdbscan_grid(
                embedding,
                boundary_pairs,
                stage="BOUNDARY",
                silhouette_sample=silhouette_sample,
                existing=candidates,
            )
        )

        evaluation = score_hdbscan_candidates(
            pd.DataFrame(history),
            max_noise_rate=max_noise_rate,
            min_persistence=min_persistence,
            max_clusters=max_clusters,
            max_largest_cluster_share=max_largest_cluster_share,
        )
        best_key = select_best_hdbscan(evaluation)

    completed = 0

    for round_id in range(1, refine_rounds + 1):
        if best_key is None:
            break

        pairs = local_refinement_pairs(
            best_key,
            n_products=n_products,
        )

        new_rows = evaluate_hdbscan_grid(
            embedding,
            pairs,
            stage=f"REFINE_{round_id}",
            silhouette_sample=silhouette_sample,
            existing=candidates,
        )

        if not new_rows:
            break

        history.extend(new_rows)

        evaluation = score_hdbscan_candidates(
            pd.DataFrame(history),
            max_noise_rate=max_noise_rate,
            min_persistence=min_persistence,
            max_clusters=max_clusters,
            max_largest_cluster_share=max_largest_cluster_share,
        )

        best_key = select_best_hdbscan(evaluation)
        completed = round_id

    if best_key is None:
        return evaluation, None, None, None, {
            "boundary_extended": boundary_extended,
            "refinement_rounds_completed": completed,
            "final_boundary_warning": None,
        }

    mcs_values = sorted(
        evaluation["min_cluster_size"].unique().astype(int)
    )
    ms_values = sorted(
        evaluation["min_samples"].unique().astype(int)
    )

    warnings = []
    if best_key[0] in {min(mcs_values), max(mcs_values)}:
        warnings.append("min_cluster_size_on_search_boundary")
    if best_key[1] in {min(ms_values), max(ms_values)}:
        warnings.append("min_samples_on_search_boundary")

    selected = candidates[best_key]

    diagnostics = {
        "boundary_extended": boundary_extended,
        "refinement_rounds_completed": completed,
        "final_boundary_warning": warnings or None,
    }

    return (
        evaluation.sort_values(
            [
                "search_stage",
                "min_cluster_size",
                "min_samples",
            ]
        ).reset_index(drop=True),
        best_key,
        selected["model"],
        selected["labels"],
        diagnostics,
    )


def map_labels_to_reference(
    reference_labels,
    alternative_labels,
):
    reference_labels = np.asarray(reference_labels)
    alternative_labels = np.asarray(alternative_labels)

    ref_clusters = sorted(
        int(x)
        for x in np.unique(reference_labels)
        if x != -1
    )
    alt_clusters = sorted(
        int(x)
        for x in np.unique(alternative_labels)
        if x != -1
    )

    mapped = np.full(
        len(alternative_labels),
        -999,
        dtype=int,
    )
    mapped[alternative_labels == -1] = -1

    if not ref_clusters or not alt_clusters:
        return mapped, {}

    ref_index = {
        label: i
        for i, label in enumerate(ref_clusters)
    }
    alt_index = {
        label: i
        for i, label in enumerate(alt_clusters)
    }

    contingency = np.zeros(
        (len(ref_clusters), len(alt_clusters)),
        dtype=np.int64,
    )

    valid = (
        (reference_labels != -1)
        & (alternative_labels != -1)
    )

    for ref_label, alt_label in zip(
        reference_labels[valid],
        alternative_labels[valid],
    ):
        contingency[
            ref_index[int(ref_label)],
            alt_index[int(alt_label)],
        ] += 1

    row_ind, col_ind = linear_sum_assignment(
        -contingency
    )

    mapping = {}

    for row, col in zip(row_ind, col_ind):
        alt_label = alt_clusters[col]
        ref_label = ref_clusters[row]
        mapping[alt_label] = ref_label
        mapped[alternative_labels == alt_label] = ref_label

    return mapped, mapping


def robustness_specs(
    base_neighbors: int,
    base_components: int,
):
    specs = [
        {
            "name": "neighbors_low",
            "n_neighbors": max(10, int(round(base_neighbors * 0.5))),
            "n_components": base_components,
            "random_state": 42,
        },
        {
            "name": "neighbors_high",
            "n_neighbors": min(80, max(
                base_neighbors + 5,
                int(round(base_neighbors * 2.0)),
            )),
            "n_components": base_components,
            "random_state": 42,
        },
        {
            "name": "dimensions_low",
            "n_neighbors": base_neighbors,
            "n_components": max(8, min(
                base_components - 1,
                10,
            )),
            "random_state": 42,
        },
        {
            "name": "dimensions_high",
            "n_neighbors": base_neighbors,
            "n_components": max(
                base_components + 1,
                20,
            ),
            "random_state": 42,
        },
        {
            "name": "seed_alt",
            "n_neighbors": base_neighbors,
            "n_components": base_components,
            "random_state": 52,
        },
    ]

    unique = []
    seen = set()

    for spec in specs:
        key = (
            spec["n_neighbors"],
            spec["n_components"],
            spec["random_state"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)

    return unique


def evaluate_umap_robustness(
    x,
    base_labels,
    selected_hdbscan_key: tuple[int, int],
    base_neighbors: int,
    base_components: int,
    silhouette_sample: int,
):
    rows = []
    mapped_runs = []

    min_cluster_size, min_samples = selected_hdbscan_key

    for spec in robustness_specs(
        base_neighbors,
        base_components,
    ):
        print(
            f"  robustness {spec['name']}: "
            f"neighbors={spec['n_neighbors']}, "
            f"components={spec['n_components']}, "
            f"seed={spec['random_state']}"
        )

        _, embedding = build_umap_embedding(
            x,
            n_components=spec["n_components"],
            n_neighbors=spec["n_neighbors"],
            random_state=spec["random_state"],
            min_dist=0.0,
        )

        metrics, _, labels = fit_hdbscan_candidate(
            embedding,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            silhouette_sample=silhouette_sample,
        )

        mapped, _ = map_labels_to_reference(
            base_labels,
            labels,
        )

        mutually_assigned = (
            (base_labels != -1)
            & (labels != -1)
        )

        if mutually_assigned.sum() >= 2:
            ari = float(
                adjusted_rand_score(
                    base_labels[mutually_assigned],
                    labels[mutually_assigned],
                )
            )
        else:
            ari = np.nan

        base_assigned = base_labels != -1
        comparable = (
            base_assigned
            & (mapped != -999)
        )

        agreement = (
            float(
                np.mean(
                    mapped[comparable]
                    == base_labels[comparable]
                )
            )
            if comparable.any()
            else np.nan
        )

        row = {
            "run": spec["name"],
            "n_neighbors": spec["n_neighbors"],
            "n_components": spec["n_components"],
            "random_state": spec["random_state"],
            "cluster_count": metrics["cluster_count"],
            "noise_rate": metrics["noise_rate"],
            "silhouette_assigned": metrics["silhouette_assigned"],
            "relative_validity": metrics["relative_validity"],
            "mean_cluster_persistence": (
                metrics["mean_cluster_persistence"]
            ),
            "ari_vs_base_assigned": ari,
            "mapped_agreement_base_assigned": agreement,
        }

        rows.append(row)
        mapped_runs.append(mapped)

    return pd.DataFrame(rows), mapped_runs


def summarize_robustness(
    robustness: pd.DataFrame,
    base_cluster_count: int,
    min_median_ari: float,
    min_median_agreement: float,
    max_noise_rate: float,
    max_cluster_count_delta: int,
):
    if robustness.empty:
        return {
            "robustness_pass": False,
            "median_ari": None,
            "median_mapped_agreement": None,
            "max_noise_rate": None,
            "max_cluster_count_delta": None,
        }

    median_ari = float(
        robustness["ari_vs_base_assigned"]
        .dropna()
        .median()
    )
    median_agreement = float(
        robustness["mapped_agreement_base_assigned"]
        .dropna()
        .median()
    )
    observed_max_noise = float(
        robustness["noise_rate"].max()
    )
    observed_cluster_delta = int(
        np.abs(
            robustness["cluster_count"]
            - base_cluster_count
        ).max()
    )

    passed = bool(
        median_ari >= min_median_ari
        and median_agreement >= min_median_agreement
        and observed_max_noise <= max_noise_rate
        and observed_cluster_delta <= max_cluster_count_delta
    )

    return {
        "robustness_pass": passed,
        "median_ari": median_ari,
        "median_mapped_agreement": median_agreement,
        "max_noise_rate": observed_max_noise,
        "max_cluster_count_delta": observed_cluster_delta,
    }


def product_stability_rates(
    base_labels,
    mapped_runs: list[np.ndarray],
):
    if not mapped_runs:
        return np.ones(len(base_labels), dtype=float)

    scores = np.zeros(len(base_labels), dtype=float)
    denominator = np.zeros(len(base_labels), dtype=float)

    for mapped in mapped_runs:
        comparable = mapped != -999
        denominator[comparable] += 1
        scores[comparable] += (
            mapped[comparable]
            == base_labels[comparable]
        )

    stability = np.ones(
        len(base_labels),
        dtype=float,
    )

    valid = denominator > 0
    stability[valid] = (
        scores[valid] / denominator[valid]
    )

    return stability


def choose_final_model(
    kmeans_eval: pd.DataFrame,
    best_k: int,
    hdbscan_eval: pd.DataFrame,
    hdbscan_key,
    robustness_summary: dict | None,
):
    k_row = (
        kmeans_eval[kmeans_eval["k"] == best_k]
        .iloc[0]
    )

    if hdbscan_key is not None:
        h_row = hdbscan_eval[
            (hdbscan_eval["min_cluster_size"] == hdbscan_key[0])
            & (hdbscan_eval["min_samples"] == hdbscan_key[1])
        ].iloc[0]

        robustness_pass = bool(
            robustness_summary
            and robustness_summary["robustness_pass"]
        )

        if bool(h_row["quality_pass"]) and robustness_pass:
            return {
                "selected_model": "HDBSCAN",
                "selection_reason": (
                    "HDBSCAN passed density quality gates and "
                    "UMAP robustness checks; KMeans retained as benchmark."
                ),
                "kmeans_k": int(best_k),
                "kmeans_silhouette": float(k_row["silhouette"]),
                "kmeans_stability_ari": float(k_row["stability_ari"]),
                "hdbscan_min_cluster_size": int(hdbscan_key[0]),
                "hdbscan_min_samples": int(hdbscan_key[1]),
                "hdbscan_cluster_count": int(h_row["cluster_count"]),
                "hdbscan_noise_rate": float(h_row["noise_rate"]),
                "hdbscan_relative_validity": float(
                    h_row["relative_validity"]
                ),
                "hdbscan_mean_persistence": float(
                    h_row["mean_cluster_persistence"]
                ),
                "hdbscan_silhouette_assigned": float(
                    h_row["silhouette_assigned"]
                ),
                "hdbscan_composite_score": float(
                    h_row["composite_score"]
                ),
                "umap_robustness_pass": True,
            }

    return {
        "selected_model": "KMEANS",
        "selection_reason": (
            "No HDBSCAN solution passed both density quality gates "
            "and UMAP robustness checks; KMeans selected as the stable fallback."
        ),
        "kmeans_k": int(best_k),
        "kmeans_silhouette": float(k_row["silhouette"]),
        "kmeans_stability_ari": float(k_row["stability_ari"]),
        "hdbscan_min_cluster_size": (
            int(hdbscan_key[0])
            if hdbscan_key is not None else None
        ),
        "hdbscan_min_samples": (
            int(hdbscan_key[1])
            if hdbscan_key is not None else None
        ),
        "hdbscan_cluster_count": None,
        "hdbscan_noise_rate": None,
        "hdbscan_relative_validity": None,
        "hdbscan_mean_persistence": None,
        "hdbscan_silhouette_assigned": None,
        "hdbscan_composite_score": None,
        "umap_robustness_pass": (
            bool(
                robustness_summary
                and robustness_summary["robustness_pass"]
            )
            if robustness_summary is not None
            else False
        ),
    }


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
):
    mask = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )
    if not mask.any():
        return np.nan

    return float(
        np.average(
            values.loc[mask],
            weights=weights.loc[mask],
        )
    )


def segment_profiles(
    products: pd.DataFrame,
) -> pd.DataFrame:
    total_products = len(products)
    total_reviews = products["analysis_review_count"].sum()

    rows = []

    for segment, group in products.groupby(
        "segment",
        dropna=False,
    ):
        product_count = group["product_id"].nunique()
        review_count = group["analysis_review_count"].sum()

        price = group["price"].dropna()
        rating = group["average_rating"].dropna()

        form_counts = group["form_factor"].value_counts(
            normalize=True
        )
        dominant_form = (
            form_counts.index[0]
            if len(form_counts) else "UNKNOWN"
        )
        dominant_form_share = (
            float(form_counts.iloc[0])
            if len(form_counts) else np.nan
        )

        price_band_counts = (
            group["price_band"]
            .dropna()
            .astype(str)
            .value_counts(normalize=True)
        )
        dominant_price_band = (
            price_band_counts.index[0]
            if len(price_band_counts) else None
        )
        dominant_price_band_share = (
            float(price_band_counts.iloc[0])
            if len(price_band_counts) else np.nan
        )

        rating_weighted = (
            weighted_mean(
                group["average_rating"],
                group["rating_number"],
            )
            if "rating_number" in group.columns
            else np.nan
        )

        row = {
            "segment": int(segment),
            "segment_type": (
                "NOISE"
                if int(segment) == -1
                else "CORE_SEGMENT"
            ),
            "product_count": int(product_count),
            "product_share": (
                product_count / total_products
                if total_products else 0
            ),
            "total_review_count": int(review_count),
            "review_share": (
                review_count / total_reviews
                if total_reviews else 0
            ),
            "median_review_count": float(
                group["analysis_review_count"].median()
            ),
            "p90_review_count": float(
                group["analysis_review_count"].quantile(0.90)
            ),
            "sparse_feedback_rate_le2": float(
                (group["analysis_review_count"] <= 2).mean()
            ),
            "price_observed_count": int(price.notna().sum()),
            "price_coverage_rate": float(
                group["price"].notna().mean()
            ),
            "median_price_observed": (
                float(price.median())
                if len(price) else np.nan
            ),
            "price_p25_observed": (
                float(price.quantile(0.25))
                if len(price) else np.nan
            ),
            "price_p75_observed": (
                float(price.quantile(0.75))
                if len(price) else np.nan
            ),
            "median_rating": (
                float(rating.median())
                if len(rating) else np.nan
            ),
            "mean_rating": (
                float(rating.mean())
                if len(rating) else np.nan
            ),
            "rating_number_weighted_rating": rating_weighted,
            "median_verified_review_share": float(
                group["verified_review_share"].median()
            ),
            "dominant_form_factor": dominant_form,
            "dominant_form_factor_share": dominant_form_share,
            "dominant_price_band": dominant_price_band,
            "dominant_price_band_share": dominant_price_band_share,
            "median_cluster_probability": float(
                group["cluster_probability"].median()
            ),
            "low_confidence_rate_lt_05": float(
                (group["cluster_probability"] < 0.5).mean()
            ),
            "median_stability_rate": float(
                group["segment_stability_rate"].median()
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "segment"
    ).reset_index(drop=True)


def segment_mix(
    products: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    out = (
        products.groupby(
            ["segment", column],
            dropna=False,
        )
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
        )
        .reset_index()
    )

    out["product_share_within_segment"] = (
        out["product_count"]
        / out.groupby("segment")["product_count"]
        .transform("sum")
    )

    out["review_share_within_segment"] = (
        out["review_count"]
        / out.groupby("segment")["review_count"]
        .transform("sum")
        .replace(0, np.nan)
    )

    return out


def top_brands_by_segment(
    products: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    x = products[
        products["brand_store"].ne("Unknown")
    ].copy()

    out = (
        x.groupby(
            ["segment", "brand_store"],
            dropna=False,
        )
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
        )
        .reset_index()
    )

    out["product_share_within_segment"] = (
        out["product_count"]
        / out.groupby("segment")["product_count"]
        .transform("sum")
    )

    out["review_share_within_segment"] = (
        out["review_count"]
        / out.groupby("segment")["review_count"]
        .transform("sum")
        .replace(0, np.nan)
    )

    return (
        out.sort_values(
            [
                "segment",
                "review_count",
                "product_count",
            ],
            ascending=[True, False, False],
        )
        .groupby("segment", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def segment_keywords(
    tfidf,
    features,
    labels,
    top_n: int,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    global_mean = np.asarray(
        tfidf.mean(axis=0)
    ).ravel()

    rows = []

    for segment in sorted(np.unique(labels)):
        if segment == -1:
            continue

        idx = np.where(labels == segment)[0]
        if len(idx) == 0:
            continue

        cluster_mean = np.asarray(
            tfidf[idx].mean(axis=0)
        ).ravel()

        cluster_df = np.asarray(
            tfidf[idx].getnnz(axis=0)
        ).ravel()

        min_cluster_df = max(
            5,
            int(math.ceil(len(idx) * 0.01)),
        )

        eligible = cluster_df >= min_cluster_df
        lift = np.divide(
            cluster_mean + 1e-12,
            global_mean + 1e-12,
        )
        differential = (
            cluster_mean
            * np.log2(np.maximum(lift, 1.0))
        )

        differential[~eligible] = 0.0
        order = np.argsort(differential)[::-1]

        rank = 0

        for j in order:
            if differential[j] <= 0:
                break

            rank += 1

            rows.append({
                "segment": int(segment),
                "rank": rank,
                "term": features[j],
                "segment_mean_tfidf": float(cluster_mean[j]),
                "global_mean_tfidf": float(global_mean[j]),
                "lift_vs_global": float(lift[j]),
                "differential_score": float(differential[j]),
                "segment_document_frequency": int(cluster_df[j]),
                "segment_document_share": float(
                    cluster_df[j] / len(idx)
                ),
            })

            if rank >= top_n:
                break

    return pd.DataFrame(rows)


def segment_quality_summary(
    profiles: pd.DataFrame,
    keywords: pd.DataFrame,
) -> pd.DataFrame:
    out = profiles.copy()

    top_keyword = (
        keywords.sort_values(
            ["segment", "rank"]
        )
        .groupby("segment")
        .first()
        if not keywords.empty
        else pd.DataFrame()
    )

    top_lift = {}
    top_share = {}
    top_term = {}

    if not top_keyword.empty:
        for segment, row in top_keyword.iterrows():
            top_lift[int(segment)] = float(
                row["lift_vs_global"]
            )
            top_share[int(segment)] = float(
                row["segment_document_share"]
            )
            top_term[int(segment)] = str(
                row["term"]
            )

    roles = []
    terms = []
    lifts = []
    shares = []

    for _, row in out.iterrows():
        segment = int(row["segment"])

        if segment == -1:
            role = "FRAGMENTED_NICHE_TAIL"
        elif (
            row["sparse_feedback_rate_le2"] >= 0.50
            and row["review_share"]
            < row["product_share"] * 0.25
        ):
            role = "SPARSE_FEEDBACK_LONG_TAIL"
        else:
            lift = top_lift.get(segment, np.nan)
            share = top_share.get(segment, np.nan)

            if (
                np.isfinite(lift)
                and np.isfinite(share)
                and lift >= 1.5
                and share >= 0.05
            ):
                role = "CORE_INTERPRETABLE_SEGMENT"
            else:
                role = "WEAK_THEME_SEGMENT"

        roles.append(role)
        terms.append(top_term.get(segment))
        lifts.append(top_lift.get(segment, np.nan))
        shares.append(top_share.get(segment, np.nan))

    out["segment_role"] = roles
    out["top_distinctive_term"] = terms
    out["top_term_lift_vs_global"] = lifts
    out["top_term_document_share"] = shares

    return out


def build_segment_descriptors(
    quality: pd.DataFrame,
    keywords: pd.DataFrame,
) -> pd.DataFrame:
    out = quality.copy()

    keyword_map = {}

    if not keywords.empty:
        for segment, group in keywords.groupby("segment"):
            keyword_map[int(segment)] = (
                group.sort_values("rank")
                .head(3)["term"]
                .astype(str)
                .tolist()
            )

    descriptors = []

    for _, row in out.iterrows():
        segment = int(row["segment"])

        if segment == -1:
            descriptors.append(
                "Fragmented / niche tail"
            )
            continue

        if row["segment_role"] == "SPARSE_FEEDBACK_LONG_TAIL":
            descriptors.append(
                "Sparse-feedback commodity long tail"
            )
            continue

        form = FORM_LABELS.get(
            row["dominant_form_factor"],
            str(row["dominant_form_factor"]),
        )
        terms = keyword_map.get(segment, [])
        term_text = (
            " / ".join(terms)
            if terms
            else "No clear keyword signal"
        )

        descriptors.append(
            f"{form} | {term_text}"
        )

    out["segment_descriptor"] = descriptors
    return out


def model_comparison_table(
    selection: dict,
    kmeans_labels,
    hdbscan_labels,
    robustness_summary: dict | None,
) -> pd.DataFrame:
    rows = [{
        "model": "KMEANS",
        "selected": selection["selected_model"] == "KMEANS",
        "cluster_count": int(
            pd.Series(kmeans_labels).nunique()
        ),
        "noise_rate": 0.0,
        "silhouette": selection["kmeans_silhouette"],
        "stability_or_persistence": (
            selection["kmeans_stability_ari"]
        ),
        "density_validity": np.nan,
        "umap_robustness_pass": np.nan,
    }]

    if hdbscan_labels is not None:
        assigned = hdbscan_labels[hdbscan_labels != -1]

        rows.append({
            "model": "HDBSCAN",
            "selected": selection["selected_model"] == "HDBSCAN",
            "cluster_count": int(
                len(np.unique(assigned))
            ),
            "noise_rate": float(
                np.mean(hdbscan_labels == -1)
            ),
            "silhouette": (
                selection.get(
                    "hdbscan_silhouette_assigned"
                )
            ),
            "stability_or_persistence": (
                selection.get(
                    "hdbscan_mean_persistence"
                )
            ),
            "density_validity": (
                selection.get(
                    "hdbscan_relative_validity"
                )
            ),
            "umap_robustness_pass": (
                robustness_summary["robustness_pass"]
                if robustness_summary
                else False
            ),
        })

    return pd.DataFrame(rows)


def fit_umap_2d(
    x,
    n_neighbors: int,
):
    return build_umap_embedding(
        x,
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=42,
        min_dist=0.10,
    )


def segment_color(segment: int) -> str:
    if int(segment) == -1:
        return NOISE_COLOR
    return SEGMENT_COLORS[int(segment) % len(SEGMENT_COLORS)]


def draw_kmeans_evaluation(
    evaluation: pd.DataFrame,
    best_k: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.plot(
        evaluation["k"],
        evaluation["silhouette"],
        marker="o",
        color=PRIMARY,
        label="Silhouette",
    )
    ax.plot(
        evaluation["k"],
        evaluation["stability_ari"],
        marker="o",
        color=PINK,
        label="ARI stability",
    )

    ax.axvline(
        best_k,
        color=YELLOW,
        linestyle="--",
        linewidth=1.5,
        label=f"Selected K={best_k}",
    )

    ax.set_title("KMeans benchmark evaluation")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.grid(color=GRID, linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def draw_hdbscan_search(
    evaluation: pd.DataFrame,
    selected_key,
    path: Path,
) -> None:
    if evaluation.empty:
        return

    x = evaluation["noise_rate"] * 100
    y = evaluation["relative_validity"].fillna(-0.05)
    sizes = (
        60
        + 520
        * evaluation["mean_cluster_persistence"]
        .fillna(0)
        .clip(lower=0)
    )
    color = evaluation["composite_score"].fillna(0)

    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=color,
        cmap="Blues",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )

    passing = evaluation["quality_pass"]
    ax.scatter(
        x[passing],
        y[passing],
        s=sizes[passing] + 20,
        facecolors="none",
        edgecolors=PINK,
        linewidth=1.0,
        label="Quality-pass candidate",
    )

    if selected_key is not None:
        selected = evaluation[
            (evaluation["min_cluster_size"] == selected_key[0])
            & (evaluation["min_samples"] == selected_key[1])
        ].sort_values(
            "composite_score",
            ascending=False,
        ).iloc[0]

        ax.scatter(
            [selected["noise_rate"] * 100],
            [selected["relative_validity"]],
            marker="*",
            s=300,
            color=YELLOW,
            edgecolor=DARK_TEXT,
            linewidth=0.8,
            label="Selected candidate",
            zorder=5,
        )

    ax.set_title("Adaptive HDBSCAN search")
    ax.set_xlabel("Noise rate (%)")
    ax.set_ylabel("Relative validity")
    ax.grid(color=GRID, linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.015,
    )
    cbar.set_label("Composite candidate score")

    save_figure(fig, path)


def draw_robustness(
    robustness: pd.DataFrame,
    path: Path,
) -> None:
    if robustness.empty:
        return

    x = np.arange(len(robustness))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 5.4))

    ax.bar(
        x - width / 2,
        robustness["ari_vs_base_assigned"],
        width=width,
        color=PRIMARY,
        label="ARI vs base",
    )
    ax.bar(
        x + width / 2,
        robustness["mapped_agreement_base_assigned"],
        width=width,
        color=PINK,
        label="Mapped agreement",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        robustness["run"],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Stability score")
    ax.set_title("UMAP / HDBSCAN robustness")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def draw_umap_segments(
    embedding_2d,
    labels,
    path: Path,
) -> None:
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10.0, 7.2))

    for segment in sorted(np.unique(labels)):
        mask = labels == segment
        label = (
            "Noise"
            if segment == -1
            else f"Segment {segment}"
        )

        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=7 if segment != -1 else 4,
            alpha=0.42 if segment != -1 else 0.16,
            c=segment_color(int(segment)),
            label=label,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title("Final product segmentation · UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="best",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def draw_product_review_share(
    profiles: pd.DataFrame,
    path: Path,
) -> None:
    x = profiles.copy()
    x["label"] = x["segment"].map(
        lambda v: "Noise" if v == -1 else f"S{int(v)}"
    )

    pos = np.arange(len(x))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ax.bar(
        pos - width / 2,
        x["product_share"] * 100,
        width=width,
        color=PRIMARY,
        label="Product share",
    )
    ax.bar(
        pos + width / 2,
        x["review_share"] * 100,
        width=width,
        color=PINK,
        label="Review-scale share",
    )

    ax.set_xticks(pos)
    ax.set_xticklabels(x["label"])
    ax.set_ylabel("Share (%)")
    ax.set_title("Segment size vs review-scale share")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def draw_form_factor_mix(
    mix: pd.DataFrame,
    path: Path,
) -> None:
    x = mix.copy()
    x["form_label"] = x["form_factor"].map(
        lambda value: FORM_LABELS.get(
            str(value),
            str(value),
        )
    )

    pivot = x.pivot_table(
        index="segment",
        columns="form_label",
        values="product_share_within_segment",
        aggfunc="sum",
        fill_value=0,
    )

    segments = pivot.index.tolist()
    pos = np.arange(len(segments))
    bottom = np.zeros(len(segments))

    colors = [
        PRIMARY, LIGHT_BLUE, PINK, NOISE_COLOR,
        YELLOW, GREEN,
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    for i, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy() * 100
        ax.bar(
            pos,
            values,
            bottom=bottom,
            color=colors[i % len(colors)],
            label=column,
        )
        bottom += values

    labels = [
        "Noise" if segment == -1 else f"S{int(segment)}"
        for segment in segments
    ]

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Product share within segment (%)")
    ax.set_title("Form-factor mix by segment")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def keyword_heatmap_data(
    keywords: pd.DataFrame,
    max_terms_per_segment: int = 3,
):
    if keywords.empty:
        return None, None

    selected_terms = (
        keywords.sort_values(
            ["segment", "rank"]
        )
        .groupby("segment", group_keys=False)
        .head(max_terms_per_segment)["term"]
        .drop_duplicates()
        .tolist()
    )

    segments = sorted(
        keywords["segment"].unique().tolist()
    )

    table = (
        keywords[keywords["term"].isin(selected_terms)]
        .pivot_table(
            index="term",
            columns="segment",
            values="differential_score",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(columns=segments, fill_value=0)
    )

    return table, segments


def draw_keyword_heatmap(
    keywords: pd.DataFrame,
    path: Path,
) -> None:
    table, segments = keyword_heatmap_data(keywords)
    if table is None or table.empty:
        return

    row_mean = table.mean(axis=1)
    row_std = table.std(axis=1).replace(0, np.nan)
    z = (
        table.sub(row_mean, axis=0)
        .div(row_std, axis=0)
        .fillna(0)
    )

    fig_height = max(6.0, 0.34 * len(z) + 2.0)

    fig, ax = plt.subplots(
        figsize=(
            max(8.0, len(segments) * 1.1 + 4.0),
            fig_height,
        )
    )

    vmax = max(
        1.0,
        float(np.nanmax(np.abs(z.to_numpy()))),
    )

    im = ax.imshow(
        z.to_numpy(),
        aspect="auto",
        cmap=HEATMAP_CMAP,
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(segments)))
    ax.set_xticklabels(
        [f"S{int(segment)}" for segment in segments]
    )
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(z.index)

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(
                j,
                i,
                f"{z.iat[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
            )

    ax.set_title("Distinctive keyword profile by segment")
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.015,
        shrink=0.88,
    )
    cbar.set_label("Relative emphasis (row z-score)")

    save_figure(fig, path)


def draw_price_rating_profile(
    profiles: pd.DataFrame,
    path: Path,
) -> None:
    x = profiles[
        (profiles["segment"] != -1)
        & profiles["median_price_observed"].notna()
        & profiles["median_rating"].notna()
    ].copy()

    if x.empty:
        return

    sizes = (
        150
        + 2200
        * x["review_share"].clip(lower=0)
    )

    colors = [
        segment_color(int(segment))
        for segment in x["segment"]
    ]

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    ax.scatter(
        x["median_price_observed"],
        x["median_rating"],
        s=sizes,
        c=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=1.2,
    )

    for _, row in x.iterrows():
        ax.annotate(
            f"S{int(row['segment'])}\n"
            f"{row['price_coverage_rate'] * 100:.0f}% price cov.",
            xy=(
                row["median_price_observed"],
                row["median_rating"],
            ),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8.3,
        )

    ax.set_xscale("log")
    ax.set_title("Segment market profile: price vs rating")
    ax.set_xlabel("Median observed price (USD, log scale)")
    ax.set_ylabel("Median average rating")
    ax.grid(color=GRID, linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def draw_segment_stability(
    profiles: pd.DataFrame,
    path: Path,
) -> None:
    x = profiles.copy()
    x["label"] = x["segment"].map(
        lambda value: "Noise" if value == -1 else f"S{int(value)}"
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    ax.bar(
        x["label"],
        x["median_stability_rate"] * 100,
        color=[
            segment_color(int(segment))
            for segment in x["segment"]
        ],
    )

    ax.axhline(
        70,
        color=PINK,
        linestyle="--",
        linewidth=1.2,
        label="70% reference",
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Median label agreement across robustness runs (%)")
    ax.set_title("Segment robustness")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.55)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def create_figures(
    kmeans_eval,
    best_k,
    hdbscan_eval,
    hdbscan_key,
    robustness,
    umap_2d,
    final_labels,
    profiles,
    form_mix,
    keywords,
    figures: Path,
):
    figures.mkdir(parents=True, exist_ok=True)

    draw_kmeans_evaluation(
        kmeans_eval,
        best_k,
        figures / "fig01_kmeans_evaluation.png",
    )

    draw_hdbscan_search(
        hdbscan_eval,
        hdbscan_key,
        figures / "fig02_hdbscan_adaptive_search.png",
    )

    if robustness is not None and not robustness.empty:
        draw_robustness(
            robustness,
            figures / "fig03_umap_robustness.png",
        )

    draw_umap_segments(
        umap_2d,
        final_labels,
        figures / "fig04_final_umap_segmentation.png",
    )

    draw_product_review_share(
        profiles,
        figures / "fig05_segment_product_vs_review_share.png",
    )

    draw_form_factor_mix(
        form_mix,
        figures / "fig06_segment_form_factor_mix.png",
    )

    draw_keyword_heatmap(
        keywords,
        figures / "fig07_segment_keyword_heatmap.png",
    )

    draw_price_rating_profile(
        profiles,
        figures / "fig08_segment_price_rating_profile.png",
    )

    draw_segment_stability(
        profiles,
        figures / "fig09_segment_stability.png",
    )


def write_model_manifest(
    models_dir: Path,
    output_path: Path,
) -> None:
    rows = []

    for path in sorted(models_dir.glob("*")):
        if not path.is_file():
            continue

        rows.append({
            "file": path.name,
            "size_mb": path.stat().st_size / (1024 ** 2),
        })

    save_csv(
        pd.DataFrame(rows),
        output_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2.3 Product segmentation and final market profiling"
    )
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--cluster-min-df", type=int, default=25)
    ap.add_argument("--cluster-max-df-ratio", type=float, default=0.50)
    ap.add_argument("--brand-min-products", type=int, default=20)

    ap.add_argument("--text-components", type=int, default=100)

    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--k-extend-max", type=int, default=16)
    ap.add_argument("--kmeans-stability-runs", type=int, default=3)

    ap.add_argument("--umap-cluster-components", type=int, default=15)
    ap.add_argument("--umap-neighbors", type=int, default=30)

    ap.add_argument(
        "--hdbscan-coarse-mcs-ratios",
        default="0.002,0.005,0.01,0.02,0.03,0.04",
    )
    ap.add_argument(
        "--hdbscan-coarse-min-samples",
        default="5,10,20,40,80",
    )
    ap.add_argument(
        "--hdbscan-refine-rounds",
        type=int,
        default=1,
    )

    ap.add_argument("--hdbscan-max-noise-rate", type=float, default=0.35)
    ap.add_argument("--hdbscan-min-persistence", type=float, default=0.25)
    ap.add_argument("--hdbscan-max-clusters", type=int, default=15)
    ap.add_argument(
        "--hdbscan-max-largest-cluster-share",
        type=float,
        default=0.75,
    )

    ap.add_argument(
        "--robustness-min-median-ari",
        type=float,
        default=0.60,
    )
    ap.add_argument(
        "--robustness-min-median-agreement",
        type=float,
        default=0.65,
    )
    ap.add_argument(
        "--robustness-max-noise-rate",
        type=float,
        default=0.45,
    )
    ap.add_argument(
        "--robustness-max-cluster-count-delta",
        type=int,
        default=3,
    )

    ap.add_argument("--silhouette-sample", type=int, default=15_000)
    ap.add_argument("--segment-top-terms", type=int, default=50)
    ap.add_argument("--top-brands-per-segment", type=int, default=20)
    args = ap.parse_args()

    setup_plot()

    products_path = Path(args.products)
    text_dir = Path(args.text_dir)

    if not products_path.is_file():
        raise FileNotFoundError(products_path)
    if not text_dir.is_dir():
        raise FileNotFoundError(text_dir)

    root = (
        Path(args.outdir)
        if args.outdir
        else Path(__file__).resolve().parent
    )

    base = root / "stage2_outputs" / "2_3_product_segmentation"
    tables = base / "tables"
    figures = base / "figures"
    models = base / "models"

    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    print("products:", products_path)
    print("text dir:", text_dir)
    print("output  :", base)

    products = load_products(products_path)
    counts, order, features = load_text_assets(text_dir)
    products_text, counts = prepare_product_matrix(
        products,
        counts,
        order,
    )

    print(
        "products with full-review text features:",
        f"{len(products_text):,}",
    )

    print("\n== 1. clustering-specific vocabulary filter ==")
    (
        cluster_counts,
        cluster_features,
        feature_audit,
    ) = filter_cluster_features(
        counts,
        features,
        products_text,
        min_df=args.cluster_min_df,
        max_df_ratio=args.cluster_max_df_ratio,
        brand_min_products=args.brand_min_products,
    )

    save_csv(
        feature_audit,
        tables / "cluster_feature_filter_audit.csv",
    )
    save_csv(
        feature_audit[
            feature_audit["keep_for_clustering"]
        ].reset_index(drop=True),
        tables / "cluster_vocabulary.csv",
    )

    filter_summary = (
        feature_audit["filter_reason"]
        .value_counts()
        .rename_axis("filter_reason")
        .reset_index(name="term_count")
    )
    save_csv(
        filter_summary,
        tables / "cluster_feature_filter_summary.csv",
    )

    print(
        f"clustering vocabulary: "
        f"{cluster_counts.shape[1]:,} / {counts.shape[1]:,} terms"
    )

    print("\n== 2. TF-IDF + SVD text embedding ==")
    (
        tfidf,
        text_embedding,
        svd_features,
        explained_variance,
        tfidf_transformer,
        svd_model,
        normalizer,
    ) = build_text_embedding(
        cluster_counts,
        text_components=args.text_components,
    )

    print(
        f"text embedding: {text_embedding.shape[0]:,} products x "
        f"{text_embedding.shape[1]:,} dimensions"
    )
    print(
        "SVD explained variance:",
        f"{explained_variance:.4f}",
    )

    joblib.dump(
        tfidf_transformer,
        models / "tfidf_transformer.joblib",
    )
    joblib.dump(
        svd_model,
        models / "svd_model.joblib",
    )
    joblib.dump(
        normalizer,
        models / "svd_normalizer.joblib",
    )

    svd_df = pd.DataFrame(
        svd_features,
        columns=[
            f"text_svd_{i + 1:03d}"
            for i in range(svd_features.shape[1])
        ],
    )
    svd_df.insert(
        0,
        "product_id",
        products_text["product_id"].values,
    )
    svd_df.to_parquet(
        tables / "product_text_svd_features.parquet",
        index=False,
        compression="zstd",
    )

    print("\n== 3. KMeans benchmark ==")
    (
        kmeans_eval,
        best_k,
        kmeans_model,
        kmeans_labels,
    ) = evaluate_kmeans(
        text_embedding,
        k_min=args.k_min,
        k_max=args.k_max,
        k_extend_max=args.k_extend_max,
        stability_runs=args.kmeans_stability_runs,
        silhouette_sample=args.silhouette_sample,
    )

    save_csv(
        kmeans_eval,
        tables / "kmeans_evaluation.csv",
    )
    joblib.dump(
        kmeans_model,
        models / "kmeans_benchmark.joblib",
    )

    print("best KMeans K:", best_k)

    print("\n== 4. base UMAP embedding ==")
    umap_cluster_model, umap_cluster_embedding = (
        build_umap_embedding(
            text_embedding,
            n_components=args.umap_cluster_components,
            n_neighbors=args.umap_neighbors,
            random_state=42,
            min_dist=0.0,
        )
    )

    joblib.dump(
        umap_cluster_model,
        models / "umap_cluster_model.joblib",
    )

    print(
        f"UMAP clustering embedding: "
        f"{umap_cluster_embedding.shape[0]:,} x "
        f"{umap_cluster_embedding.shape[1]:,}"
    )

    print("\n== 5. adaptive HDBSCAN search ==")

    coarse_ratios = parse_float_list(
        args.hdbscan_coarse_mcs_ratios
    )
    if not coarse_ratios:
        raise ValueError(
            "--hdbscan-coarse-mcs-ratios 不能为空。"
        )

    coarse_mcs = default_mcs_values(
        len(products_text),
        coarse_ratios,
    )

    coarse_ms = parse_int_list(
        args.hdbscan_coarse_min_samples
    )
    if not coarse_ms:
        raise ValueError(
            "--hdbscan-coarse-min-samples 不能为空。"
        )

    (
        hdbscan_eval,
        hdbscan_key,
        hdbscan_model,
        hdbscan_labels,
        search_diagnostics,
    ) = adaptive_hdbscan_search(
        umap_cluster_embedding,
        n_products=len(products_text),
        coarse_mcs=coarse_mcs,
        coarse_ms=coarse_ms,
        refine_rounds=args.hdbscan_refine_rounds,
        silhouette_sample=args.silhouette_sample,
        max_noise_rate=args.hdbscan_max_noise_rate,
        min_persistence=args.hdbscan_min_persistence,
        max_clusters=args.hdbscan_max_clusters,
        max_largest_cluster_share=(
            args.hdbscan_max_largest_cluster_share
        ),
    )

    save_csv(
        hdbscan_eval,
        tables / "hdbscan_search_history.csv",
    )
    save_json(
        search_diagnostics,
        tables / "hdbscan_search_diagnostics.json",
    )

    if hdbscan_key is not None:
        joblib.dump(
            hdbscan_model,
            models / "hdbscan_candidate.joblib",
        )

        print(
            "selected HDBSCAN candidate:",
            f"min_cluster_size={hdbscan_key[0]}, "
            f"min_samples={hdbscan_key[1]}",
        )
    else:
        print("no HDBSCAN candidate passed density quality gates")

    robustness = None
    mapped_runs = []
    robustness_summary = None

    if hdbscan_key is not None:
        print("\n== 6. UMAP robustness checks ==")

        robustness, mapped_runs = evaluate_umap_robustness(
            text_embedding,
            base_labels=hdbscan_labels,
            selected_hdbscan_key=hdbscan_key,
            base_neighbors=args.umap_neighbors,
            base_components=args.umap_cluster_components,
            silhouette_sample=args.silhouette_sample,
        )

        base_cluster_count = len(
            np.unique(
                hdbscan_labels[hdbscan_labels != -1]
            )
        )

        robustness_summary = summarize_robustness(
            robustness,
            base_cluster_count=base_cluster_count,
            min_median_ari=args.robustness_min_median_ari,
            min_median_agreement=(
                args.robustness_min_median_agreement
            ),
            max_noise_rate=args.robustness_max_noise_rate,
            max_cluster_count_delta=(
                args.robustness_max_cluster_count_delta
            ),
        )

        save_csv(
            robustness,
            tables / "umap_robustness.csv",
        )
        save_json(
            robustness_summary,
            tables / "umap_robustness_summary.json",
        )

        print(
            "robustness pass:",
            robustness_summary["robustness_pass"],
        )
        print(
            "median ARI:",
            f"{robustness_summary['median_ari']:.4f}",
        )
        print(
            "median mapped agreement:",
            f"{robustness_summary['median_mapped_agreement']:.4f}",
        )

    print("\n== 7. final model selection ==")

    selection = choose_final_model(
        kmeans_eval,
        best_k,
        hdbscan_eval,
        hdbscan_key,
        robustness_summary,
    )

    if selection["selected_model"] == "HDBSCAN":
        final_labels = hdbscan_labels.astype(int)
        probabilities = np.asarray(
            hdbscan_model.probabilities_,
            dtype=float,
        )
        stability_rate = product_stability_rates(
            final_labels,
            mapped_runs,
        )

        joblib.dump(
            hdbscan_model,
            models / "hdbscan_final.joblib",
        )
    else:
        final_labels = kmeans_labels.astype(int)
        probabilities = np.ones(
            len(final_labels),
            dtype=float,
        )
        stability_rate = np.ones(
            len(final_labels),
            dtype=float,
        )

    products_text["segment"] = final_labels
    products_text["cluster_probability"] = probabilities
    products_text["segment_stability_rate"] = stability_rate

    if hdbscan_labels is not None:
        assigned = hdbscan_labels != -1

        if assigned.sum() > 1:
            selection["kmeans_hdbscan_ari_assigned"] = float(
                adjusted_rand_score(
                    kmeans_labels[assigned],
                    hdbscan_labels[assigned],
                )
            )
        else:
            selection["kmeans_hdbscan_ari_assigned"] = None
    else:
        selection["kmeans_hdbscan_ari_assigned"] = None

    selection["hdbscan_search_diagnostics"] = search_diagnostics
    selection["umap_robustness_summary"] = robustness_summary

    print("final model:", selection["selected_model"])
    print("reason     :", selection["selection_reason"])

    save_json(
        selection,
        tables / "model_selection.json",
    )

    comparison = model_comparison_table(
        selection,
        kmeans_labels,
        hdbscan_labels,
        robustness_summary,
    )
    save_csv(
        comparison,
        tables / "model_comparison.csv",
    )

    print("\n== 8. final segment profiling ==")

    profiles = segment_profiles(products_text)

    keywords = segment_keywords(
        tfidf,
        cluster_features,
        final_labels,
        top_n=args.segment_top_terms,
    )

    quality = segment_quality_summary(
        profiles,
        keywords,
    )

    report_summary = build_segment_descriptors(
        quality,
        keywords,
    )

    form_mix = segment_mix(
        products_text,
        "form_factor",
    )
    price_mix = segment_mix(
        products_text,
        "price_band",
    )
    top_brands = top_brands_by_segment(
        products_text,
        top_n=args.top_brands_per_segment,
    )

    save_csv(
        profiles,
        tables / "segment_profiles.csv",
    )
    save_csv(
        quality,
        tables / "segment_quality_summary.csv",
    )
    save_csv(
        report_summary,
        tables / "report_segment_summary.csv",
    )
    save_csv(
        keywords,
        tables / "segment_distinctive_keywords.csv",
    )
    save_csv(
        form_mix,
        tables / "segment_form_factor_mix.csv",
    )
    save_csv(
        price_mix,
        tables / "segment_price_band_mix.csv",
    )
    save_csv(
        top_brands,
        tables / "segment_top_brands.csv",
    )

    output_cols = [
        "product_id", "title", "brand_store",
        "price", "price_band",
        "average_rating", "rating_number",
        "analysis_review_count", "verified_review_share",
        "form_factor", "segment",
        "cluster_probability",
        "segment_stability_rate",
    ]
    output_cols = [
        col
        for col in output_cols
        if col in products_text.columns
    ]

    save_csv(
        products_text[output_cols],
        tables / "product_segments.csv",
    )

    stability_df = pd.DataFrame({
        "product_id": products_text["product_id"].values,
        "segment": final_labels,
        "segment_stability_rate": stability_rate,
    })
    stability_df.to_parquet(
        tables / "product_segment_stability.parquet",
        index=False,
        compression="zstd",
    )

    print("\n== 9. UMAP visualization ==")

    umap_2d_model, umap_2d = fit_umap_2d(
        text_embedding,
        n_neighbors=args.umap_neighbors,
    )

    joblib.dump(
        umap_2d_model,
        models / "umap_2d_model.joblib",
    )

    umap_df = pd.DataFrame({
        "product_id": products_text["product_id"].values,
        "umap_1": umap_2d[:, 0],
        "umap_2": umap_2d[:, 1],
        "segment": final_labels,
        "cluster_probability": probabilities,
        "segment_stability_rate": stability_rate,
    })

    umap_df.to_parquet(
        tables / "product_umap_2d.parquet",
        index=False,
        compression="zstd",
    )

    print("\n== 10. figures ==")

    create_figures(
        kmeans_eval=kmeans_eval,
        best_k=best_k,
        hdbscan_eval=hdbscan_eval,
        hdbscan_key=hdbscan_key,
        robustness=robustness,
        umap_2d=umap_2d,
        final_labels=final_labels,
        profiles=profiles,
        form_mix=form_mix,
        keywords=keywords,
        figures=figures,
    )

    final_config = {
        "product_unit": "one row per product",
        "text_source": "2.2 full-review product term matrix",
        "review_sampling": "none",
        "cluster_definition": (
            "text features only; price, rating, review scale, "
            "verified share, form factor and Brand / Store are profile variables"
        ),
        "cluster_min_df": args.cluster_min_df,
        "cluster_max_df_ratio": args.cluster_max_df_ratio,
        "brand_min_products": args.brand_min_products,
        "original_vocabulary_size": int(counts.shape[1]),
        "clustering_vocabulary_size": int(cluster_counts.shape[1]),
        "svd_components": int(text_embedding.shape[1]),
        "svd_explained_variance": explained_variance,
        "kmeans_search": {
            "k_min": args.k_min,
            "k_max": args.k_max,
            "k_extend_max": args.k_extend_max,
            "selected_k": best_k,
        },
        "umap_base": {
            "n_components": args.umap_cluster_components,
            "n_neighbors": args.umap_neighbors,
            "metric": "cosine",
            "random_state": 42,
        },
        "hdbscan_search": {
            "coarse_mcs_ratios": coarse_ratios,
            "coarse_min_samples": coarse_ms,
            "refine_rounds": args.hdbscan_refine_rounds,
            "quality_gates": {
                "max_noise_rate": args.hdbscan_max_noise_rate,
                "min_mean_cluster_persistence": (
                    args.hdbscan_min_persistence
                ),
                "max_clusters": args.hdbscan_max_clusters,
                "max_largest_cluster_share": (
                    args.hdbscan_max_largest_cluster_share
                ),
                "relative_validity_must_be_positive": True,
            },
            "selected_key": (
                list(hdbscan_key)
                if hdbscan_key is not None
                else None
            ),
            "search_diagnostics": search_diagnostics,
        },
        "robustness_gates": {
            "min_median_ari": args.robustness_min_median_ari,
            "min_median_mapped_agreement": (
                args.robustness_min_median_agreement
            ),
            "max_noise_rate": args.robustness_max_noise_rate,
            "max_cluster_count_delta": (
                args.robustness_max_cluster_count_delta
            ),
        },
        "selected_model": selection["selected_model"],
        "review_scale_note": (
            "analysis_review_count is review scale, not sales"
        ),
        "price_note": (
            "price is excluded from clustering and profiled only "
            "for price-observed products"
        ),
    }

    save_json(
        final_config,
        tables / "segmentation_config.json",
    )

    write_model_manifest(
        models,
        tables / "model_manifest.csv",
    )

    print("\nDONE")
    print("final report table:", tables / "report_segment_summary.csv")
    print("model selection   :", tables / "model_selection.json")
    print("product segments  :", tables / "product_segments.csv")
    print("model manifest    :", tables / "model_manifest.csv")
    print("figures           :", figures)


if __name__ == "__main__":
    main()
