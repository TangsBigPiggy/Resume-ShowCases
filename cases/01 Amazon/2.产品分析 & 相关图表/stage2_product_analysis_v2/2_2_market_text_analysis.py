#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.2 全量市场文本分析

输入：
- products_analysis.csv
- reviews_analysis.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy import sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
    TfidfTransformer,
)


DEFAULT_STAGE1_DIR = Path(
    r"E:\DA Cases\Amazon\1.EDA & 预处理\1.2 预处理"
    r"\preprocessed_pre_ManualReviewed\stage1_preprocessed"
)
DEFAULT_PRODUCTS = DEFAULT_STAGE1_DIR / "products_analysis.csv"
DEFAULT_REVIEWS = DEFAULT_STAGE1_DIR / "reviews_analysis.parquet"

PRICE_BINS = [-np.inf, 25, 50, 100, 200, np.inf]
PRICE_LABELS = ["<$25", "$25-49", "$50-99", "$100-199", "$200+"]

CACHE_VERSION = "2.2.2"
REVIEW_BOUNDARY = "<<<REVIEW_BOUNDARY_2_2_2>>>"

DOMAIN_STOPWORDS = {
    "headphone", "headphones", "earbud", "earbuds",
    "earphone", "earphones", "product", "amazon",
    "bought", "buy", "use", "using", "used",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{1,}")

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

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "research_diverging",
    [PRIMARY, "#F7F9FC", PINK],
)

FORM_ORDER = ["EARBUD_INEAR", "OVER_EAR", "ON_EAR", "UNKNOWN"]
FORM_LABELS = {
    "EARBUD_INEAR": "Earbud / In-ear",
    "OVER_EAR": "Over-ear",
    "ON_EAR": "On-ear",
    "UNKNOWN": "Unknown",
}

RATING_ORDER = ["negative_1_2", "neutral_3", "positive_4_5"]
RATING_LABELS = {
    "negative_1_2": "1–2★",
    "neutral_3": "3★",
    "positive_4_5": "4–5★",
}

TIER_ORDER = [
    "TOP_1_PCT",
    "P1_5_PCT",
    "P5_20_PCT",
    "BOTTOM_80_PCT",
]
TIER_LABELS = {
    "TOP_1_PCT": "Top 1%",
    "P1_5_PCT": "1–5%",
    "P5_20_PCT": "5–20%",
    "BOTTOM_80_PCT": "Bottom 80%",
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
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": DARK_TEXT,
        "axes.labelcolor": DARK_TEXT,
        "axes.edgecolor": "#AEB8C3",
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
    })


def save_figure(fig, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    fig.savefig(
        str(path),
        format="png",
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"图片保存失败：{path}")

    print(
        f"  saved: {path.name} "
        f"({path.stat().st_size / 1024:.1f} KB)"
    )


def verify_figures(figures: Path) -> None:
    expected = [
        "fig01_top_market_bigrams.png",
        "fig02_form_factor_keyword_heatmap.png",
        "fig03_price_band_keyword_heatmap.png",
        "fig04_rating_group_keyword_heatmap.png",
        "fig05_head_vs_longtail_keyword_heatmap.png",
        "fig06_market_vs_product_equal.png",
    ]

    missing = []
    for name in expected:
        path = figures / name
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "以下图片未成功生成："
            + ", ".join(missing)
        )

    print(f"figures written: {len(expected)} -> {figures.resolve()}")


class ParquetTextIterable:
    def __init__(
        self,
        path: Path,
        column: str,
        batch_size: int = 256,
    ):
        self.path = Path(path)
        self.column = column
        self.batch_size = batch_size

    def __iter__(self):
        parquet = pq.ParquetFile(self.path)
        for batch in parquet.iter_batches(
            columns=[self.column],
            batch_size=self.batch_size,
        ):
            for value in batch.column(0).to_pylist():
                yield value or ""


class ReviewBoundaryAnalyzer:
    def __init__(self, ngram_range=(1, 3)):
        self.ngram_range = ngram_range
        self.stopwords = set(ENGLISH_STOP_WORDS)
        self.stopwords.update(DOMAIN_STOPWORDS)

    def __call__(self, document):
        text = document or ""

        for review in text.split(REVIEW_BOUNDARY):
            tokens = [
                token.lower()
                for token in TOKEN_RE.findall(review)
                if token.lower() not in self.stopwords
            ]

            if not tokens:
                continue

            min_n, max_n = self.ngram_range
            for n in range(min_n, max_n + 1):
                if len(tokens) < n:
                    continue
                for i in range(len(tokens) - n + 1):
                    yield " ".join(tokens[i:i + n])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = {
        "product_id", "store", "price",
        "average_rating", "analysis_review_count", "form_factor",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"products_analysis.csv 缺少字段: {missing}")

    for col in [
        "price", "average_rating",
        "analysis_review_count", "verified_review_share",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["brand_store"] = (
        df["store"].fillna("").astype(str).str.strip().replace("", "Unknown")
    )
    df["form_factor"] = (
        df["form_factor"].fillna("UNKNOWN").astype(str).str.strip()
    )
    df["analysis_review_count"] = (
        df["analysis_review_count"].fillna(0).clip(lower=0)
    )

    df.loc[df["price"] <= 0, "price"] = np.nan

    df["price_band"] = pd.cut(
        df["price"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=False,
    )

    return df


def sql_literal(text: str) -> str:
    return text.replace("'", "''")


def build_full_product_corpus(
    reviews_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = sql_literal(str(reviews_path))
    target = sql_literal(str(output_path))
    boundary = sql_literal(REVIEW_BOUNDARY)

    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute(f"""
        COPY (
            SELECT
                cast(product_id as varchar) AS product_id,
                count(*) AS text_review_count,
                sum(length(text_clean)) AS text_char_count,
                string_agg(
                    text_clean,
                    '{boundary}'
                ) AS product_document
            FROM read_parquet('{source}')
            WHERE
                text_clean IS NOT NULL
                AND length(trim(text_clean)) > 0
            GROUP BY product_id
            ORDER BY product_id
        )
        TO '{target}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()


def build_rating_corpus(
    reviews_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = sql_literal(str(reviews_path))
    target = sql_literal(str(output_path))
    boundary = sql_literal(REVIEW_BOUNDARY)

    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute(f"""
        COPY (
            SELECT
                cast(product_id as varchar) AS product_id,
                CASE
                    WHEN rating IN (1, 2) THEN 'negative_1_2'
                    WHEN rating = 3 THEN 'neutral_3'
                    WHEN rating IN (4, 5) THEN 'positive_4_5'
                    ELSE 'missing_or_invalid'
                END AS rating_group,
                count(*) AS review_count,
                string_agg(
                    text_clean,
                    '{boundary}'
                ) AS group_document
            FROM read_parquet('{source}')
            WHERE
                text_clean IS NOT NULL
                AND length(trim(text_clean)) > 0
            GROUP BY product_id, rating_group
            ORDER BY product_id, rating_group
        )
        TO '{target}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()


def cache_state(cache: Path) -> dict:
    return load_json(cache / "cache_state.json", default={}) or {}


def corpus_cache_valid(
    cache: Path,
    product_corpus_path: Path,
    rating_corpus_path: Path | None = None,
) -> bool:
    state = cache_state(cache)
    if state.get("cache_version") != CACHE_VERSION:
        return False
    if not state.get("product_corpus") or not product_corpus_path.is_file():
        return False
    if rating_corpus_path is not None:
        if not state.get("rating_corpus") or not rating_corpus_path.is_file():
            return False
    return True


def update_cache_state(cache: Path, **kwargs) -> None:
    state = cache_state(cache)
    if state.get("cache_version") != CACHE_VERSION:
        state = {}
    state["cache_version"] = CACHE_VERSION
    state.update(kwargs)
    save_json(state, cache / "cache_state.json")


def build_vectorizer(
    max_features: int | None = None,
    min_df: int | float = 1,
    max_df: int | float = 1.0,
    vocabulary: dict | None = None,
) -> CountVectorizer:
    kwargs = {
        "analyzer": ReviewBoundaryAnalyzer((1, 3)),
        "dtype": np.int32,
    }

    if vocabulary is None:
        kwargs.update({
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
        })
    else:
        kwargs["vocabulary"] = vocabulary

    return CountVectorizer(**kwargs)


def matrix_signature(
    max_features: int,
    min_df: int,
    max_df: float,
) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "max_features": int(max_features),
        "min_df": int(min_df),
        "max_df": float(max_df),
        "ngram_range": [1, 3],
        "review_boundary_safe": True,
    }


def load_features(vocabulary: dict) -> np.ndarray:
    features = np.empty(len(vocabulary), dtype=object)
    for term, index in vocabulary.items():
        features[int(index)] = term
    return features


def fit_or_load_count_matrix(
    corpus_path: Path,
    cache: Path,
    max_features: int,
    min_df: int,
    max_df: float,
    rebuild: bool,
):
    matrix_path = cache / "product_term_counts.npz"
    order_path = cache / "product_order.csv"
    vocab_path = cache / "vocabulary.json"
    config_path = cache / "matrix_config.json"

    signature = matrix_signature(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    saved_signature = load_json(config_path, default={}) or {}
    reusable = (
        not rebuild
        and matrix_path.is_file()
        and order_path.is_file()
        and vocab_path.is_file()
        and saved_signature == signature
    )

    if reusable:
        count_matrix = sparse.load_npz(matrix_path).tocsr()
        vocabulary = load_json(vocab_path, default={}) or {}
        features = load_features(vocabulary)
        vectorizer = build_vectorizer(vocabulary=vocabulary)
        print("reuse count matrix:", matrix_path)
        return vectorizer, count_matrix, features

    vectorizer = build_vectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    docs = ParquetTextIterable(
        corpus_path,
        "product_document",
    )
    count_matrix = vectorizer.fit_transform(docs).tocsr()
    features = np.asarray(vectorizer.get_feature_names_out())

    sparse.save_npz(
        matrix_path,
        count_matrix,
        compressed=True,
    )
    save_json(
        {
            term: int(index)
            for term, index in vectorizer.vocabulary_.items()
        },
        vocab_path,
    )
    save_json(signature, config_path)

    return vectorizer, count_matrix, features


def market_term_table(
    count_matrix,
    features,
) -> pd.DataFrame:
    term_count = np.asarray(count_matrix.sum(axis=0)).ravel()
    product_df = np.asarray((count_matrix > 0).sum(axis=0)).ravel()

    total_terms = term_count.sum()
    out = pd.DataFrame({
        "term": features,
        "term_count": term_count,
        "product_document_frequency": product_df,
    })

    out["term_share"] = (
        out["term_count"] / total_terms
        if total_terms else 0
    )
    out["term_per_million"] = out["term_share"] * 1_000_000
    out["ngram_n"] = out["term"].str.count(" ") + 1

    return out.sort_values(
        ["term_count", "product_document_frequency"],
        ascending=False,
    ).reset_index(drop=True)


def product_tfidf_table(
    count_matrix,
    features,
) -> pd.DataFrame:
    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf = transformer.fit_transform(count_matrix)

    out = pd.DataFrame({
        "term": features,
        "tfidf_sum": np.asarray(tfidf.sum(axis=0)).ravel(),
        "tfidf_mean": np.asarray(tfidf.mean(axis=0)).ravel(),
        "idf": transformer.idf_,
    })
    out["ngram_n"] = out["term"].str.count(" ") + 1

    return (
        out.sort_values("tfidf_sum", ascending=False)
        .reset_index(drop=True),
        tfidf,
    )


def l1_normalize_rows(matrix):
    row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    np.divide(
        np.float32(1.0),
        row_sum,
        out=inv,
        where=row_sum > 0,
    )
    return sparse.diags(inv).dot(matrix).tocsr()


def group_incidence(
    groups: pd.Series,
    min_size: int = 1,
    exclude: set[str] | None = None,
):
    series = groups.reset_index(drop=True).astype("string")
    valid = series.notna()

    if exclude:
        valid &= ~series.isin(exclude)

    counts = series[valid].value_counts()
    names = counts[counts >= min_size].index.astype(str).tolist()

    if not names:
        return sparse.csr_matrix((0, len(series))), [], np.array([], dtype=int)

    name_to_code = {name: i for i, name in enumerate(names)}

    product_idx = []
    group_idx = []

    for idx, value in series.items():
        if not valid.iloc[idx]:
            continue
        value = str(value)
        code = name_to_code.get(value)
        if code is None:
            continue
        product_idx.append(idx)
        group_idx.append(code)

    matrix = sparse.csr_matrix(
        (
            np.ones(len(product_idx), dtype=np.float32),
            (group_idx, product_idx),
        ),
        shape=(len(names), len(series)),
    )

    group_sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(int)
    return matrix, names, group_sizes


def market_weighted_group_keywords(
    count_matrix,
    groups: pd.Series,
    features,
    top_n: int,
    min_size: int = 1,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    incidence, names, group_sizes = group_incidence(
        groups,
        min_size=min_size,
        exclude=exclude,
    )
    if not names:
        return pd.DataFrame()

    group_counts = (incidence @ count_matrix).tocsr()
    group_tfidf = TfidfTransformer(
        norm="l2",
        sublinear_tf=True,
    ).fit_transform(group_counts)

    rows = []
    for i, name in enumerate(names):
        counts = group_counts.getrow(i)
        scores = group_tfidf.getrow(i)

        if counts.nnz == 0:
            continue

        score_map = {
            int(index): float(value)
            for index, value in zip(scores.indices, scores.data)
        }
        total = float(counts.sum())

        candidates = []
        for index, count in zip(counts.indices, counts.data):
            score = score_map.get(int(index), 0.0)
            if score > 0:
                candidates.append((int(index), float(count), score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for rank, (index, count, score) in enumerate(
            candidates[:top_n],
            1,
        ):
            rows.append({
                "group": name,
                "rank": rank,
                "term": features[index],
                "term_count": int(count),
                "term_per_million": (
                    count / total * 1_000_000
                    if total else 0
                ),
                "group_tfidf": score,
                "product_count": int(group_sizes[i]),
            })

    return pd.DataFrame(rows)


def product_equal_group_keywords(
    product_share_matrix,
    groups: pd.Series,
    features,
    top_n: int,
    min_size: int = 1,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    incidence, names, group_sizes = group_incidence(
        groups,
        min_size=min_size,
        exclude=exclude,
    )
    if not names:
        return pd.DataFrame()

    group_sum = (incidence @ product_share_matrix).tocsr()

    inv_group_size = np.zeros(len(group_sizes), dtype=np.float64)
    np.divide(
        1.0,
        group_sizes,
        out=inv_group_size,
        where=group_sizes > 0,
    )
    group_mean = (
        sparse.diags(inv_group_size)
        .dot(group_sum)
        .tocsr()
    )

    group_tfidf = TfidfTransformer(
        norm="l2",
        sublinear_tf=False,
    ).fit_transform(group_mean)

    rows = []
    for i, name in enumerate(names):
        mean_row = group_mean.getrow(i)
        score_row = group_tfidf.getrow(i)

        if mean_row.nnz == 0:
            continue

        mean_map = {
            int(index): float(value)
            for index, value in zip(mean_row.indices, mean_row.data)
        }

        candidates = [
            (
                int(index),
                mean_map.get(int(index), 0.0),
                float(score),
            )
            for index, score in zip(score_row.indices, score_row.data)
            if score > 0
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)

        for rank, (index, mean_share, score) in enumerate(
            candidates[:top_n],
            1,
        ):
            rows.append({
                "group": name,
                "rank": rank,
                "term": features[index],
                "mean_product_term_share": mean_share,
                "mean_product_term_per_million": mean_share * 1_000_000,
                "group_tfidf": score,
                "product_count": int(group_sizes[i]),
            })

    return pd.DataFrame(rows)


def save_group_views(
    count_matrix,
    product_share_matrix,
    groups: pd.Series,
    features,
    tables: Path,
    prefix: str,
    top_n: int,
    min_size: int = 1,
    exclude: set[str] | None = None,
):
    market = market_weighted_group_keywords(
        count_matrix,
        groups,
        features,
        top_n=top_n,
        min_size=min_size,
        exclude=exclude,
    )
    equal = product_equal_group_keywords(
        product_share_matrix,
        groups,
        features,
        top_n=top_n,
        min_size=min_size,
        exclude=exclude,
    )

    save_csv(
        market,
        tables / f"{prefix}_market_weighted.csv",
    )
    save_csv(
        equal,
        tables / f"{prefix}_product_equal.csv",
    )

    return market, equal


def price_coverage_table(products_text: pd.DataFrame) -> pd.DataFrame:
    x = products_text.copy()
    x["price_status"] = np.where(
        x["price"].notna(),
        "PRICE_AVAILABLE",
        "PRICE_MISSING",
    )

    out = (
        x.groupby("price_status")
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
        )
        .reset_index()
    )

    out["product_share"] = (
        out["product_count"] / out["product_count"].sum()
    )

    total_reviews = out["review_count"].sum()
    out["review_share"] = (
        out["review_count"] / total_reviews
        if total_reviews else 0
    )

    return out


def review_tiers(products_text: pd.DataFrame) -> pd.Series:
    n = len(products_text)
    order = (
        products_text["analysis_review_count"]
        .sort_values(ascending=False, kind="stable")
        .index
    )

    percentile = pd.Series(
        np.arange(1, n + 1) / n,
        index=order,
        dtype=float,
    ).reindex(products_text.index)

    return pd.cut(
        percentile,
        bins=[0, 0.01, 0.05, 0.20, 1.0],
        labels=[
            "TOP_1_PCT",
            "P1_5_PCT",
            "P5_20_PCT",
            "BOTTOM_80_PCT",
        ],
        include_lowest=True,
    ).astype("string")

def review_tier_summary(
    products_text: pd.DataFrame,
    tiers: pd.Series,
) -> pd.DataFrame:
    x = products_text[
        ["product_id", "analysis_review_count"]
    ].copy()
    x["review_tier"] = tiers

    out = (
        x.groupby("review_tier", dropna=False)
        .agg(
            product_count=("product_id", "nunique"),
            review_count=("analysis_review_count", "sum"),
        )
        .reset_index()
    )

    out["product_share"] = (
        out["product_count"] / out["product_count"].sum()
    )

    total_reviews = out["review_count"].sum()
    out["review_share"] = (
        out["review_count"] / total_reviews
        if total_reviews else 0
    )

    return out


def rating_group_views(
    rating_corpus_path: Path,
    vectorizer: CountVectorizer,
    features,
    top_n: int,
):
    meta = pq.read_table(
        rating_corpus_path,
        columns=["product_id", "rating_group", "review_count"],
    ).to_pandas()

    docs = ParquetTextIterable(
        rating_corpus_path,
        "group_document",
    )
    matrix = vectorizer.transform(docs).tocsr()

    groups = (
        meta["rating_group"]
        .fillna("missing_or_invalid")
        .astype(str)
    )

    valid = groups.isin([
        "negative_1_2",
        "neutral_3",
        "positive_4_5",
    ])

    matrix = matrix[valid.to_numpy()]
    meta = meta.loc[valid].reset_index(drop=True)
    groups = groups.loc[valid].reset_index(drop=True)

    market = market_weighted_group_keywords(
        matrix,
        groups,
        features,
        top_n=top_n,
        min_size=1,
    )
    equal = product_equal_group_keywords(
        l1_normalize_rows(matrix),
        groups,
        features,
        top_n=top_n,
        min_size=1,
    )

    review_counts = (
        meta.groupby("rating_group")["review_count"]
        .sum()
        .rename("review_count")
    )

    if not market.empty:
        market["review_count"] = (
            market["group"].map(review_counts).fillna(0).astype(int)
        )

    if not equal.empty:
        equal["review_count"] = (
            equal["group"].map(review_counts).fillna(0).astype(int)
        )

    return market, equal



def top_market_bigrams(
    global_freq: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    x = global_freq[global_freq["ngram_n"] == 2].copy()
    return (
        x.nlargest(n, "term_count")
        .sort_values("term_count", ascending=True)
        .reset_index(drop=True)
    )


def draw_top_market_bigrams(
    global_freq: pd.DataFrame,
    path: Path,
    n: int = 20,
) -> None:
    x = top_market_bigrams(global_freq, n=n)
    if x.empty:
        return

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    bars = ax.barh(
        x["term"],
        x["term_per_million"],
        color=PRIMARY,
        alpha=0.88,
    )

    for bar, value in zip(bars, x["term_per_million"]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,.1f}",
            va="center",
            ha="left",
            fontsize=8.8,
        )

    ax.set_title("Top market bigrams")
    ax.set_xlabel("Occurrences per million retained terms")
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def keyword_heatmap_matrix(
    df: pd.DataFrame,
    group_order: list[str],
    metric: str = "group_tfidf",
    n_terms: int = 14,
    prefer_phrases: bool = True,
):
    if df is None or df.empty:
        return None

    x = df[df["group"].isin(group_order)].copy()
    if x.empty or metric not in x.columns:
        return None

    x["ngram_n"] = x["term"].str.count(" ") + 1

    if prefer_phrases:
        phrase_x = x[x["ngram_n"] >= 2].copy()
        if phrase_x["term"].nunique() >= n_terms:
            x = phrase_x

    pivot = x.pivot_table(
        index="term",
        columns="group",
        values=metric,
        aggfunc="max",
        fill_value=0.0,
    )

    columns = [g for g in group_order if g in pivot.columns]
    if len(columns) < 2:
        return None

    pivot = pivot.reindex(columns=columns)

    spread = pivot.std(axis=1)
    keep = spread.nlargest(min(n_terms, len(spread))).index
    raw = pivot.loc[keep].copy()

    row_mean = raw.mean(axis=1)
    row_std = raw.std(axis=1).replace(0, np.nan)
    z = raw.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)

    peak_group = raw.to_numpy().argmax(axis=1)
    peak_value = z.max(axis=1).to_numpy()
    order = np.lexsort((-peak_value, peak_group))
    z = z.iloc[order]

    return z


def draw_keyword_heatmap(
    df: pd.DataFrame,
    group_order: list[str],
    label_map: dict[str, str],
    title: str,
    path: Path,
    n_terms: int = 14,
) -> None:
    z = keyword_heatmap_matrix(
        df,
        group_order=group_order,
        metric="group_tfidf",
        n_terms=n_terms,
        prefer_phrases=True,
    )
    if z is None or z.empty:
        return

    fig_width = max(7.4, 1.55 * len(z.columns) + 3.8)
    fig_height = max(5.8, 0.39 * len(z) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))
    im = ax.imshow(
        z.to_numpy(),
        aspect="auto",
        cmap=HEATMAP_CMAP,
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(z.columns)))
    ax.set_xticklabels(
        [label_map.get(c, c) for c in z.columns],
        rotation=0,
    )
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(z.index)

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            value = float(z.iat[i, j])
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=DARK_TEXT,
            )

    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.88)
    cbar.set_label("Relative emphasis (row z-score)")

    save_figure(fig, path)


def global_market_vs_product_equal(
    count_matrix,
    product_share_matrix,
    features,
    n: int = 15,
) -> pd.DataFrame:
    market_count = np.asarray(count_matrix.sum(axis=0)).ravel().astype(float)
    market_total = market_count.sum()
    market_ppm = (
        market_count / market_total * 1_000_000
        if market_total else np.zeros_like(market_count)
    )

    product_equal_share = np.asarray(
        product_share_matrix.mean(axis=0)
    ).ravel()
    product_equal_ppm = product_equal_share * 1_000_000

    ngram_n = np.array(
        [str(term).count(" ") + 1 for term in features],
        dtype=int,
    )
    bigram_idx = np.where(ngram_n == 2)[0]

    if len(bigram_idx) == 0:
        return pd.DataFrame()

    top_idx = bigram_idx[
        np.argsort(market_ppm[bigram_idx])[::-1][:n]
    ]

    out = pd.DataFrame({
        "term": features[top_idx],
        "market_weighted_per_million": market_ppm[top_idx],
        "product_equal_per_million": product_equal_ppm[top_idx],
    })

    return out.sort_values(
        "market_weighted_per_million",
        ascending=True,
    ).reset_index(drop=True)


def draw_market_vs_product_equal(
    count_matrix,
    product_share_matrix,
    features,
    path: Path,
    n: int = 15,
) -> None:
    x = global_market_vs_product_equal(
        count_matrix,
        product_share_matrix,
        features,
        n=n,
    )
    if x.empty:
        return

    y = np.arange(len(x))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    ax.barh(
        y - height / 2,
        x["market_weighted_per_million"],
        height=height,
        color=PRIMARY,
        label="Market-weighted",
    )
    ax.barh(
        y + height / 2,
        x["product_equal_per_million"],
        height=height,
        color=PINK,
        label="Product-equal",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(x["term"])
    ax.set_xlabel("Occurrences per million retained terms")
    ax.set_ylabel("")
    ax.set_title("Market-weighted vs product-equal language")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, path)


def create_figures(
    global_freq: pd.DataFrame,
    form_equal: pd.DataFrame,
    price_equal: pd.DataFrame,
    rating_market: pd.DataFrame,
    tier_equal: pd.DataFrame,
    count_matrix,
    product_share_matrix,
    features,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)

    draw_top_market_bigrams(
        global_freq,
        figures / "fig01_top_market_bigrams.png",
        n=20,
    )

    draw_keyword_heatmap(
        form_equal,
        group_order=FORM_ORDER,
        label_map=FORM_LABELS,
        title="Form-factor keyword differences · Product-equal",
        path=figures / "fig02_form_factor_keyword_heatmap.png",
        n_terms=14,
    )

    draw_keyword_heatmap(
        price_equal,
        group_order=PRICE_LABELS,
        label_map={label: label for label in PRICE_LABELS},
        title="Price-band keyword differences · Product-equal",
        path=figures / "fig03_price_band_keyword_heatmap.png",
        n_terms=15,
    )

    draw_keyword_heatmap(
        rating_market,
        group_order=RATING_ORDER,
        label_map=RATING_LABELS,
        title="Rating-group keyword differences · Market-weighted",
        path=figures / "fig04_rating_group_keyword_heatmap.png",
        n_terms=14,
    )

    draw_keyword_heatmap(
        tier_equal,
        group_order=TIER_ORDER,
        label_map=TIER_LABELS,
        title="Head vs long-tail keyword differences · Product-equal",
        path=figures / "fig05_head_vs_longtail_keyword_heatmap.png",
        n_terms=15,
    )

    draw_market_vs_product_equal(
        count_matrix,
        product_share_matrix,
        features,
        figures / "fig06_market_vs_product_equal.png",
        n=15,
    )

    verify_figures(figures)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2.2 Full-market review text analysis"
    )
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--reviews", default=str(DEFAULT_REVIEWS))
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--max-features", type=int, default=100_000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.95)
    ap.add_argument("--group-top-n", type=int, default=100)
    ap.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="强制重建文本语料与词项矩阵缓存",
    )
    args = ap.parse_args()

    products_path = Path(args.products)
    reviews_path = Path(args.reviews)

    if not products_path.is_file():
        raise FileNotFoundError(products_path)
    if not reviews_path.is_file():
        raise FileNotFoundError(reviews_path)

    root = (
        Path(args.outdir)
        if args.outdir
        else Path(__file__).resolve().parent
    )
    base = root / "stage2_outputs" / "2_2_market_text"
    tables = base / "tables"
    cache = base / "cache"
    figures = base / "figures"

    tables.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    setup_plot()

    product_corpus_path = cache / "product_text_corpus.parquet"
    rating_corpus_path = cache / "rating_product_corpus.parquet"

    print("products:", products_path)
    print("reviews :", reviews_path)
    print("output  :", base)

    products = load_products(products_path)

    print("\n== 1. build product corpus ==")
    product_cache_ok = (
        corpus_cache_valid(
            cache,
            product_corpus_path,
        )
        and not args.rebuild_cache
    )

    if product_cache_ok:
        print("reuse cache:", product_corpus_path)
    else:
        build_full_product_corpus(
            reviews_path,
            product_corpus_path,
        )
        update_cache_state(
            cache,
            product_corpus=True,
        )

    corpus_meta = pq.read_table(
        product_corpus_path,
        columns=[
            "product_id",
            "text_review_count",
            "text_char_count",
        ],
    ).to_pandas()

    corpus_meta["product_id"] = (
        corpus_meta["product_id"].astype(str).str.strip()
    )

    products_text = (
        corpus_meta.merge(
            products,
            on="product_id",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    print(f"product documents: {len(products_text):,}")
    print(
        "text reviews represented:",
        f"{int(products_text['text_review_count'].sum()):,}",
    )

    print("\n== 2. build count matrix ==")
    vectorizer, count_matrix, features = fit_or_load_count_matrix(
        product_corpus_path,
        cache,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        rebuild=args.rebuild_cache,
    )

    save_csv(
        products_text[
            [
                "product_id",
                "text_review_count",
                "text_char_count",
            ]
        ],
        cache / "product_order.csv",
    )

    print(
        f"term matrix: {count_matrix.shape[0]:,} products x "
        f"{count_matrix.shape[1]:,} terms"
    )

    product_share_matrix = l1_normalize_rows(count_matrix)

    print("\n== 3. market-wide frequency ==")
    global_freq = market_term_table(
        count_matrix,
        features,
    )

    save_csv(
        global_freq,
        tables / "global_ngram_frequency.csv",
    )
    save_csv(
        global_freq[global_freq["ngram_n"] == 1].copy(),
        tables / "global_unigram_frequency.csv",
    )
    save_csv(
        global_freq[global_freq["ngram_n"] == 2].copy(),
        tables / "global_bigram_frequency.csv",
    )
    save_csv(
        global_freq[global_freq["ngram_n"] == 3].copy(),
        tables / "global_trigram_frequency.csv",
    )

    print("\n== 4. product-level TF-IDF ==")
    global_tfidf, _ = product_tfidf_table(
        count_matrix,
        features,
    )
    save_csv(
        global_tfidf,
        tables / "global_product_tfidf.csv",
    )

    corpus_summary = pd.DataFrame([{
        "product_documents": len(products_text),
        "text_reviews_represented": int(
            products_text["text_review_count"].sum()
        ),
        "text_characters": int(
            products_text["text_char_count"].sum()
        ),
        "vocabulary_size": len(features),
        "review_sampling": "none",
        "review_boundary_safe_ngram": True,
    }])
    save_csv(
        corpus_summary,
        tables / "text_corpus_summary.csv",
    )

    print("\n== 5. form-factor keywords ==")
    form_market, form_equal = save_group_views(
        count_matrix,
        product_share_matrix,
        products_text["form_factor"],
        features,
        tables,
        prefix="form_factor_keywords",
        top_n=args.group_top_n,
    )

    print("\n== 6. price-band keywords ==")
    price_coverage = price_coverage_table(products_text)
    save_csv(
        price_coverage,
        tables / "price_analysis_coverage.csv",
    )

    price_available = products_text["price"].notna()
    price_groups = products_text["price_band"].astype("string")

    price_market, price_equal = save_group_views(
        count_matrix[price_available.to_numpy()],
        product_share_matrix[price_available.to_numpy()],
        price_groups.loc[price_available].reset_index(drop=True),
        features,
        tables,
        prefix="price_band_keywords",
        top_n=args.group_top_n,
    )

    price_status = pd.Series(
        np.where(
            products_text["price"].notna(),
            "PRICE_AVAILABLE",
            "PRICE_MISSING",
        ),
        index=products_text.index,
        dtype="string",
    )
    save_group_views(
        count_matrix,
        product_share_matrix,
        price_status,
        features,
        tables,
        prefix="price_availability_keywords",
        top_n=args.group_top_n,
    )

    print("\n== 7. Brand / Store keywords ==")
    save_group_views(
        count_matrix,
        product_share_matrix,
        products_text["brand_store"],
        features,
        tables,
        prefix="brand_store_keywords",
        top_n=args.group_top_n,
        exclude={"Unknown"},
    )

    print("\n== 8. head vs long-tail keywords ==")
    tiers = review_tiers(products_text)
    save_csv(
        review_tier_summary(
            products_text,
            tiers,
        ),
        tables / "review_tier_summary.csv",
    )
    tier_market, tier_equal = save_group_views(
        count_matrix,
        product_share_matrix,
        tiers,
        features,
        tables,
        prefix="review_tier_keywords",
        top_n=args.group_top_n,
    )

    print("\n== 9. rating-group keywords ==")
    rating_cache_ok = (
        corpus_cache_valid(
            cache,
            product_corpus_path,
            rating_corpus_path,
        )
        and not args.rebuild_cache
    )

    if rating_cache_ok:
        print("reuse cache:", rating_corpus_path)
    else:
        build_rating_corpus(
            reviews_path,
            rating_corpus_path,
        )
        update_cache_state(
            cache,
            product_corpus=True,
            rating_corpus=True,
        )

    rating_market, rating_equal = rating_group_views(
        rating_corpus_path,
        vectorizer,
        features,
        top_n=args.group_top_n,
    )

    save_csv(
        rating_market,
        tables / "rating_group_keywords_market_weighted.csv",
    )
    save_csv(
        rating_equal,
        tables / "rating_group_keywords_product_equal.csv",
    )

    print("\n== 10. figures ==")
    create_figures(
        global_freq=global_freq,
        form_equal=form_equal,
        price_equal=price_equal,
        rating_market=rating_market,
        tier_equal=tier_equal,
        count_matrix=count_matrix,
        product_share_matrix=product_share_matrix,
        features=features,
        figures=figures,
    )

    config = {
        "review_policy": "all non-empty text_clean reviews",
        "review_sampling": "none",
        "review_boundary_safe_ngram": True,
        "ngram_range": [1, 3],
        "max_features": args.max_features,
        "min_df_product_documents": args.min_df,
        "max_df_product_documents": args.max_df,
        "market_weighted_view": (
            "raw term counts retain observed review-volume contribution"
        ),
        "product_equal_view": (
            "each product is L1-normalized before group aggregation"
        ),
        "tfidf_document_unit": "product",
        "price_band_scope": "price-observed products only",
        "brand_store_scope": "all non-Unknown Brand / Store groups",
        "review_tiers": [
            "TOP_1_PCT",
            "P1_5_PCT",
            "P5_20_PCT",
            "BOTTOM_80_PCT",
        ],
        "review_scale_note": (
            "analysis_review_count is review scale, not sales"
        ),
        "figures": [
            "fig01_top_market_bigrams.png",
            "fig02_form_factor_keyword_heatmap.png",
            "fig03_price_band_keyword_heatmap.png",
            "fig04_rating_group_keyword_heatmap.png",
            "fig05_head_vs_longtail_keyword_heatmap.png",
            "fig06_market_vs_product_equal.png",
        ],
    }

    save_json(
        config,
        tables / "text_analysis_config.json",
    )

    print("DONE")


if __name__ == "__main__":
    main()