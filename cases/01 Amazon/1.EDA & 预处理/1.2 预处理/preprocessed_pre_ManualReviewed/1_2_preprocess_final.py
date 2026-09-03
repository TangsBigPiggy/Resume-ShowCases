#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Reviews 2023 耳机项目预处理。

输入：
- Amazon Electronics metadata
- Amazon Electronics reviews
- 已人工复核的 Gold 样本

输出：
- 商品资格表
- DEVICE 商品分析表
- DEVICE 评论分析表
- 质量检查与模型校准结果
"""
from __future__ import annotations

import argparse
import hashlib
import html
import csv
import gzip
import json
import math
import re
import sqlite3
import unicodedata
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    import joblib
    import numpy as np
    from sklearn.base import clone
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import FeatureUnion, Pipeline
except ImportError as e:
    raise SystemExit(
        "缺少依赖。请先运行：pip install scikit-learn joblib\n"
        f"原始错误：{e}"
    )

TARGET_L2 = "Headphones, Earbuds & Accessories"

DEFAULT_DATA_DIR = Path(r"E:\DA Cases\Amazon\0.原始数据")
DEFAULT_META = DEFAULT_DATA_DIR / "meta_Electronics.jsonl"
DEFAULT_REVIEWS = DEFAULT_DATA_DIR / "Electronics.jsonl"
DEFAULT_GOLD = Path(
    r"E:\DA Cases\Amazon\1.EDA & 预处理\1.2 预处理"
    r"\preprocessed_pre_ManualReviewed\manual_audit_sample_gold_reviewed.csv"
)

LABEL_DEVICE = "DEVICE"
LABEL_ACCESSORY = "ACCESSORY"
LABEL_AMBIGUOUS = "AMBIGUOUS"
VALID_LABELS = {LABEL_DEVICE, LABEL_ACCESSORY, LABEL_AMBIGUOUS}

VALID_FORMS = {"EARBUD_INEAR", "OVER_EAR", "ON_EAR"}
UNKNOWN_FORM = "UNKNOWN"

SEED = 20260901


# 商品类目与标题规则
ACCESSORY_LEAF_EXACT = {
    "cases",
    "earpads",
    "adapters",
    "extension cords",
    "cables",
    "ear tips",
    "replacement parts",
    "stands",
    "chargers",
    "headphone amplifiers",
    "amplifiers",
}

DEVICE_LEAF_EXACT = {
    "earbud headphones",
    "over-ear headphones",
    "on-ear headphones",
    "in-ear headphones",
    "open-ear headphones",
}

GENERIC_DEVICE_LEAF = {
    "headphones & earbuds",
    "headphones, earbuds & accessories",
}


DEVICE_TERM_PATTERNS = [
    r"\bheadphones?\b",
    r"\bearphones?\b",
    r"\bearbuds?\b",
    r"\bear\s+buds?\b",
    r"\bearpods?\b",
    r"\bheadsets?\b",
    r"\bearpieces?\b",
    r"\bin[-\s]?ear\s+monitors?\b",
    r"\bIEMs?\b",
    r"\bAirPods?(?:\s+Pro|\s+Max)?\b",
    r"\bGalaxy\s+Buds?(?:\s+Pro|\s+Live)?\b",
    r"\bPixel\s+Buds?(?:\s+Pro)?\b",
    r"\bBeats\s+(?:Studio\s+)?Buds?\b",
]
DEVICE_TERM_RX = [re.compile(p, re.I) for p in DEVICE_TERM_PATTERNS]

STRONG_ACCESSORY_PATTERNS = [
    r"\b(?:replacement|spare)\s+(?:wireless\s+)?(?:charging\s+)?"
    r"(?:case|ear\s*pads?|earpads?|ear\s*cushions?|ear\s*tips?|eartips?|"
    r"cable|cord|adapter|filter|battery)\b",

    r"\b(?:case|cover|skin|decal|wrap|sleeve|pouch|bag|holder|stand|mount|hanger)"
    r"\s+(?:compatible\s+)?(?:for|with)\b",

    r"\b(?:hard\s+EVA|travel|carrying|storage)\s+(?:case|bag)\s+for\b",

    r"\b(?:ear\s*pads?|earpads?|ear\s*cushions?|ear\s*tips?|eartips?|"
    r"cable|cord|adapter|filter)\s+(?:replacement\s+)?for\b",

    r"\b(?:left|right|single)\s+(?:side\s+)?(?:replacement\s+)?"
    r"(?:earbud|earpiece|earphone)\b",

    r"\bcharging\s+case\s+(?:only|replacement)\b",

    r"\b(?:headphone|earphone|earbud)\s+"
    r"(?:stand|hanger|holder|adapter|cable|cord|case|cover)\b",

    r"\b(?:earbud|earphone|headphone|AirPods?)\s+clean(?:ing|er)\s+(?:kit|pen|tool)\b",
]
STRONG_ACCESSORY_RX = [re.compile(p, re.I) for p in STRONG_ACCESSORY_PATTERNS]

ACCESSORY_CONTEXT_PATTERNS = [
    r"\bear\s*pads?\b",
    r"\bearpads?\b",
    r"\bear\s*cushions?\b",
    r"\bear\s*tips?\b",
    r"\beartips?\b",
    r"\bcharging\s+case\b",
    r"\bprotective\s+case\b",
    r"\bcarrying\s+case\b",
    r"\bstorage\s+case\b",
    r"\badapters?\b",
    r"\bcables?\b",
    r"\bcords?\b",
    r"\bfilters?\b",
    r"\bstands?\b",
    r"\bholders?\b",
    r"\bmounts?\b",
    r"\bskins?\b",
    r"\bdecals?\b",
]
ACCESSORY_CONTEXT_RX = [re.compile(p, re.I) for p in ACCESSORY_CONTEXT_PATTERNS]

DEVICE_BUNDLE_PATTERNS = [
    r"\b(?:headphones?|earphones?|earbuds?|ear\s+buds?|headsets?)\b.*"
    r"\bwith\s+(?:a\s+)?(?:wireless\s+)?charging\s+case\b",

    r"\b(?:headphones?|earphones?|earbuds?|ear\s+buds?|headsets?)\b.*"
    r"\b(?:carrying|storage|protective)\s+case\s+(?:included|included\b|with\b)",

    r"\b(?:headphones?|earphones?|earbuds?|headsets?)\b.*"
    r"(?:\+|plus|bundle(?:d)?\s+with)\s+.*(?:case|cable|adapter)\b",
]
DEVICE_BUNDLE_RX = [re.compile(p, re.I) for p in DEVICE_BUNDLE_PATTERNS]

FORM_PATTERNS = {
    "OVER_EAR": [
        re.compile(r"\bover[-\s]?ear\b", re.I),
        re.compile(r"\baround[-\s]?ear\b", re.I),
        re.compile(r"\bcircumaural\b", re.I),
        re.compile(r"\bfull[-\s]?size\s+headphones?\b", re.I),
    ],
    "ON_EAR": [
        re.compile(r"\bon[-\s]?ear\b", re.I),
        re.compile(r"\bsupra[-\s]?aural\b", re.I),
    ],
    "EARBUD_INEAR": [
        re.compile(r"\bearbuds?\b", re.I),
        re.compile(r"\bear\s+buds?\b", re.I),
        re.compile(r"\bearphones?\b", re.I),
        re.compile(r"\bin[-\s]?ear\b", re.I),
        re.compile(r"\bearpods?\b", re.I),
        re.compile(r"\bIEMs?\b", re.I),
        re.compile(r"\bearpieces?\b", re.I),
    ],
}


# 商品数据结构
@dataclass
class ProductRecord:
    product_id: str
    title: str
    store: str
    leaf_categories: str
    l3_categories: str
    price: Optional[float]
    average_rating: Optional[float]
    rating_number: int
    duplicate_rows: int = 1
    duplicate_title_conflict: int = 0
    duplicate_price_conflict: int = 0
    duplicate_rating_conflict: int = 0


@dataclass
class RuleSignals:
    title_valid: int
    device_term_count: int
    strong_accessory: int
    accessory_precedes_device: int
    accessory_context_count: int
    device_bundle_context: int
    taxonomy_accessory: int
    taxonomy_device: int
    taxonomy_generic: int
    taxonomy_anomaly: int


@dataclass
class Decision:
    product_id: str
    final_label: str
    device_probability: float
    decision_confidence: float
    form_factor: str
    form_probability: float
    reason: str

    title_valid: int
    device_term_count: int
    strong_accessory: int
    accessory_precedes_device: int
    accessory_context_count: int
    device_bundle_context: int
    taxonomy_accessory: int
    taxonomy_device: int
    taxonomy_generic: int
    taxonomy_anomaly: int

    title: str
    store: str
    leaf_categories: str
    l3_categories: str
    price: Optional[float]
    average_rating: Optional[float]
    rating_number: int
    duplicate_rows: int
    duplicate_title_conflict: int
    duplicate_price_conflict: int
    duplicate_rating_conflict: int


def normalize_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def norm_text(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def to_int(v: Any, default: int = 0) -> int:
    try:
        x = float(v)
        if math.isfinite(x):
            return int(x)
    except (TypeError, ValueError):
        pass
    return default


def to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


PRICE_NUM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")


def parse_price(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) and x > 0 else None
    s = str(v).replace(",", "").strip()
    nums = [m.group(1) for m in PRICE_NUM_RE.finditer(s)]
    if len(nums) != 1:
        return None
    try:
        x = float(nums[0])
    except ValueError:
        return None
    return x if math.isfinite(x) and x > 0 else None


def parse_average_rating(v: Any) -> Optional[float]:
    x = to_float(v)
    if x is None or not (1.0 <= x <= 5.0):
        return None
    return x


def _is_jsonl_file(path: Path) -> bool:
    if not path.is_file():
        return False
    n = path.name.lower()
    return n.endswith(".jsonl") or n.endswith(".jsonl.gz") or n.endswith(".json.gz")


def resolve_data_file(path: Path) -> Path:
    path = Path(path)

    variants = [path]
    s = str(path)
    variants.append(Path(s[:-3]) if s.lower().endswith(".gz") else Path(s + ".gz"))

    for p in variants:
        if p.is_file():
            return p

    if path.is_dir():
        hits = [p for p in path.iterdir() if _is_jsonl_file(p)]
        if hits:
            return max(hits, key=lambda x: x.stat().st_size)

    raise FileNotFoundError(f"未找到可读取 JSONL 文件: {path}")


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def flatten_cats(cats: Any) -> list[list[str]]:
    paths: list[list[str]] = []
    if not isinstance(cats, list):
        return paths

    if cats and all(isinstance(x, str) for x in cats):
        p = [norm_text(x) for x in cats if norm_text(x)]
        return [p] if p else []

    for entry in cats:
        if isinstance(entry, list):
            p = [norm_text(x) for x in entry if norm_text(x)]
            if p:
                paths.append(p)
        elif norm_text(entry):
            paths.append([norm_text(entry)])

    return paths


def target_paths(paths: Iterable[list[str]]) -> list[list[str]]:
    return [p for p in paths if TARGET_L2 in p]


def split_pipe_values(s: str) -> set[str]:
    return {x.strip() for x in norm_text(s).split("|") if x.strip()}


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def metadata_row_score(
    title: str,
    store: str,
    paths: list[list[str]],
    rating_number: int,
    price: Optional[float],
    average_rating: Optional[float],
) -> tuple:
    return (
        int(bool(title)),
        len(title),
        max((len(p) for p in paths), default=0),
        int(bool(store)),
        int(price is not None),
        int(average_rating is not None),
        rating_number,
    )


# 读取并聚合目标商品

def load_target_products(meta_path: Path) -> dict[str, ProductRecord]:
    agg: dict[str, dict] = {}
    scanned = 0
    target_rows_n = 0
    target_rows_without_id = 0

    for rec in iter_jsonl(meta_path):
        scanned += 1
        paths = target_paths(flatten_cats(rec.get("categories")))
        if not paths:
            continue

        target_rows_n += 1
        pid = normalize_id(rec.get("parent_asin")) or normalize_id(rec.get("asin"))
        if not pid:
            target_rows_without_id += 1
            continue

        title = norm_text(rec.get("title"))
        store = norm_text(rec.get("store"))
        rating_number = max(to_int(rec.get("rating_number"), 0), 0)
        price = parse_price(rec.get("price"))
        average_rating = parse_average_rating(rec.get("average_rating"))

        leaves = {p[-1] for p in paths if p}
        l3s = {p[2] for p in paths if len(p) >= 3}
        score = metadata_row_score(
            title, store, paths, rating_number, price, average_rating
        )

        if pid not in agg:
            agg[pid] = {
                "best_title": title,
                "best_store": store,
                "best_rating_number": rating_number,
                "best_score": score,
                "best_price": price,
                "best_price_rating_n": rating_number if price is not None else -1,
                "best_average_rating": average_rating,
                "best_avg_rating_n": rating_number if average_rating is not None else -1,
                "leafs": set(leaves),
                "l3s": set(l3s),
                "titles": {title.lower()} if title else set(),
                "prices": {round(price, 2)} if price is not None else set(),
                "avg_ratings": {round(average_rating, 3)} if average_rating is not None else set(),
                "duplicate_rows": 1,
            }
        else:
            a = agg[pid]
            a["duplicate_rows"] += 1
            a["leafs"].update(leaves)
            a["l3s"].update(l3s)
            if title:
                a["titles"].add(title.lower())
            if price is not None:
                a["prices"].add(round(price, 2))
                if rating_number >= a["best_price_rating_n"]:
                    a["best_price"] = price
                    a["best_price_rating_n"] = rating_number
            if average_rating is not None:
                a["avg_ratings"].add(round(average_rating, 3))
                if rating_number >= a["best_avg_rating_n"]:
                    a["best_average_rating"] = average_rating
                    a["best_avg_rating_n"] = rating_number

            if score > a["best_score"]:
                a["best_title"] = title
                a["best_store"] = store
                a["best_score"] = score
            a["best_rating_number"] = max(a["best_rating_number"], rating_number)

        if scanned % 200_000 == 0:
            print(
                f"  meta scanned {scanned:,} | target rows {target_rows_n:,} | "
                f"unique products {len(agg):,}",
                flush=True,
            )

    products: dict[str, ProductRecord] = {}
    for pid, a in agg.items():
        products[pid] = ProductRecord(
            product_id=pid,
            title=a["best_title"],
            store=a["best_store"],
            leaf_categories=" | ".join(sorted(a["leafs"])),
            l3_categories=" | ".join(sorted(a["l3s"])),
            price=a["best_price"],
            average_rating=a["best_average_rating"],
            rating_number=a["best_rating_number"],
            duplicate_rows=a["duplicate_rows"],
            duplicate_title_conflict=int(len(a["titles"]) > 1),
            duplicate_price_conflict=int(len(a["prices"]) > 1),
            duplicate_rating_conflict=int(len(a["avg_ratings"]) > 1),
        )

    print(
        f"metadata done: scanned={scanned:,} target_rows={target_rows_n:,} "
        f"without_id={target_rows_without_id:,} unique_target_products={len(products):,}"
    )
    return products


def title_is_valid(title: str) -> bool:
    s = norm_text(title)
    if not s or s.lower() in {"-", "n/a", "na", "none", "null", "unknown"}:
        return False
    return len(re.findall(r"[A-Za-z]", s)) >= 3


def count_hits(regexes: list[re.Pattern], text: str) -> int:
    return sum(1 for rx in regexes if rx.search(text))


# 生成商品资格规则信号

def analyze_rules(p: ProductRecord) -> RuleSignals:
    title = p.title
    leaves = {x.lower() for x in split_pipe_values(p.leaf_categories)}

    title_valid = int(title_is_valid(title))
    device_matches = [
        m for rx in DEVICE_TERM_RX
        for m in [rx.search(title)]
        if m is not None
    ]
    accessory_matches = [
        m for rx in STRONG_ACCESSORY_RX
        for m in [rx.search(title)]
        if m is not None
    ]

    device_term_count = len(device_matches)
    strong_accessory = int(bool(accessory_matches))

    first_device_pos = min((m.start() for m in device_matches), default=10**9)
    first_accessory_pos = min((m.start() for m in accessory_matches), default=10**9)

    accessory_precedes_device = int(
        bool(accessory_matches)
        and (
            not device_matches
            or first_accessory_pos <= first_device_pos
        )
    )

    accessory_context_count = count_hits(ACCESSORY_CONTEXT_RX, title)
    device_bundle_context = int(any(rx.search(title) for rx in DEVICE_BUNDLE_RX))

    taxonomy_accessory = int(bool(leaves & ACCESSORY_LEAF_EXACT))
    taxonomy_device = int(bool(leaves & DEVICE_LEAF_EXACT))
    taxonomy_generic = int(bool(leaves & GENERIC_DEVICE_LEAF))

    taxonomy_anomaly = int(
        taxonomy_device == 1
        and title_valid == 1
        and device_term_count == 0
    )

    return RuleSignals(
        title_valid=title_valid,
        device_term_count=device_term_count,
        strong_accessory=strong_accessory,
        accessory_precedes_device=accessory_precedes_device,
        accessory_context_count=accessory_context_count,
        device_bundle_context=device_bundle_context,
        taxonomy_accessory=taxonomy_accessory,
        taxonomy_device=taxonomy_device,
        taxonomy_generic=taxonomy_generic,
        taxonomy_anomaly=taxonomy_anomaly,
    )


def build_model_text_from_values(
    title: str,
    leaf_categories: str,
    l3_categories: str,
) -> str:
    title = norm_text(title).lower()
    leaf = norm_text(leaf_categories).lower()
    l3 = norm_text(l3_categories).lower()

    return f"title {title} leaf {leaf} l3 {l3}"


def build_model_text(p: ProductRecord) -> str:
    return build_model_text_from_values(
        p.title,
        p.leaf_categories,
        p.l3_categories,
    )


def resolve_gold_csv(path: Path) -> Path:
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Gold 文件必须是 .csv：{path}\n"
                "如果当前文件是 .xlsx/.xls，请在 Excel/WPS 中“另存为 CSV UTF-8（逗号分隔）”。"
            )
        return path

    if not path.suffix:
        candidate = path.with_suffix(".csv")
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Gold CSV 不存在：{path}")


# 读取人工复核标签

def load_gold_csv(path: Path) -> list[dict]:
    rows: list[dict] = []

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = normalize_id(row.get("product_id"))
            gold = norm_text(row.get("gold_label")).upper()
            form = norm_text(row.get("gold_form_factor")).upper()

            if not pid:
                continue
            if gold not in VALID_LABELS:
                continue

            rows.append({
                "product_id": pid,
                "gold_label": gold,
                "gold_form_factor": form,
                "title": norm_text(row.get("title")),
                "store": norm_text(row.get("store")),
                "leaf_categories": norm_text(row.get("leaf_categories")),
                "l3_categories": norm_text(row.get("l3_categories")),
                "rating_number": max(to_int(row.get("rating_number"), 0), 0),
            })

    if not rows:
        raise ValueError(f"Gold 文件没有有效标注：{path}")

    by_id: dict[str, str] = {}
    for r in rows:
        pid = r["product_id"]
        lab = r["gold_label"]
        if pid in by_id and by_id[pid] != lab:
            raise ValueError(f"Gold 中同一 product_id 出现冲突标签：{pid}")
        by_id[pid] = lab

    unique = {}
    for r in rows:
        unique.setdefault(r["product_id"], r)

    rows = list(unique.values())

    print("gold label distribution:", dict(Counter(r["gold_label"] for r in rows)))

    return rows


def gold_row_to_product(r: dict) -> ProductRecord:
    return ProductRecord(
        product_id=r["product_id"],
        title=r["title"],
        store=r.get("store", ""),
        leaf_categories=r.get("leaf_categories", ""),
        l3_categories=r.get("l3_categories", ""),
        price=None,
        average_rating=None,
        rating_number=max(to_int(r.get("rating_number"), 0), 0),
    )


def make_eligibility_model() -> Pipeline:
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.995,
                max_features=60_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=80_000,
                sublinear_tf=True,
            ),
        ),
    ])

    clf = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=SEED,
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


# 模型校准与交叉验证

def development_probabilities(
    gold_rows: list[dict],
    folds: int,
    seed: int,
) -> list[float]:
    binary_idx = [
        i for i, r in enumerate(gold_rows)
        if r["gold_label"] in {LABEL_DEVICE, LABEL_ACCESSORY}
    ]
    ambiguous_idx = [
        i for i, r in enumerate(gold_rows)
        if r["gold_label"] == LABEL_AMBIGUOUS
    ]

    if len(binary_idx) < folds * 2:
        raise ValueError("Gold 中 DEVICE/ACCESSORY 样本太少，无法做交叉验证。")

    X = [
        build_model_text_from_values(
            gold_rows[i]["title"],
            gold_rows[i]["leaf_categories"],
            gold_rows[i]["l3_categories"],
        )
        for i in binary_idx
    ]
    y = np.array([
        1 if gold_rows[i]["gold_label"] == LABEL_DEVICE else 0
        for i in binary_idx
    ], dtype=int)

    amb_X = [
        build_model_text_from_values(
            gold_rows[i]["title"],
            gold_rows[i]["leaf_categories"],
            gold_rows[i]["l3_categories"],
        )
        for i in ambiguous_idx
    ]

    probs: list[Optional[float]] = [None] * len(gold_rows)
    amb_sum = np.zeros(len(ambiguous_idx), dtype=float)

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    base = make_eligibility_model()

    for fold_no, (train_pos, valid_pos) in enumerate(skf.split(X, y), start=1):
        model = clone(base)

        X_train = [X[i] for i in train_pos]
        y_train = y[train_pos]
        X_valid = [X[i] for i in valid_pos]

        model.fit(X_train, y_train)

        p_valid = model.predict_proba(X_valid)[:, 1]
        for local_pos, p in zip(valid_pos, p_valid):
            original_idx = binary_idx[local_pos]
            probs[original_idx] = float(p)

        if ambiguous_idx:
            amb_sum += model.predict_proba(amb_X)[:, 1]

        print(f"  eligibility CV fold {fold_no}/{folds} done", flush=True)

    if ambiguous_idx:
        amb_mean = amb_sum / folds
        for original_idx, p in zip(ambiguous_idx, amb_mean):
            probs[original_idx] = float(p)

    if any(p is None for p in probs):
        raise RuntimeError("development probabilities 生成不完整。")

    return [float(p) for p in probs]


def decision_from_signals(
    s: RuleSignals,
    prob_device: float,
    device_threshold: float,
    accessory_threshold: float,
) -> tuple[str, float, str]:
    if not s.title_valid:
        return LABEL_AMBIGUOUS, 0.0, "invalid_or_empty_title"

    if (
        s.strong_accessory
        and s.accessory_precedes_device
        and not s.device_bundle_context
    ):
        return (
            LABEL_ACCESSORY,
            max(0.99, 1.0 - prob_device),
            "strong_accessory_relation",
        )

    if s.taxonomy_anomaly:
        if s.taxonomy_accessory and s.accessory_context_count > 0:
            return (
                LABEL_ACCESSORY,
                max(0.95, 1.0 - prob_device),
                "taxonomy_anomaly_with_accessory_context",
            )
        return (
            LABEL_AMBIGUOUS,
            0.0,
            "taxonomy_device_but_title_has_no_device_subject",
        )

    if s.device_term_count == 0:
        if (
            s.taxonomy_accessory
            and s.accessory_context_count > 0
            and prob_device <= accessory_threshold
        ):
            return (
                LABEL_ACCESSORY,
                1.0 - prob_device,
                "accessory_taxonomy_and_context",
            )

        return (
            LABEL_AMBIGUOUS,
            0.0,
            "no_headphone_subject_in_title",
        )

    if prob_device >= device_threshold:
        return (
            LABEL_DEVICE,
            prob_device,
            "model_device_above_threshold_with_device_subject",
        )

    if (
        prob_device <= accessory_threshold
        and (s.taxonomy_accessory or s.accessory_context_count > 0)
    ):
        return (
            LABEL_ACCESSORY,
            1.0 - prob_device,
            "model_accessory_below_threshold_with_accessory_context",
        )

    return (
        LABEL_AMBIGUOUS,
        0.0,
        "probability_or_semantics_not_decisive",
    )


def raw_decision(
    p: ProductRecord,
    prob_device: float,
    device_threshold: float,
    accessory_threshold: float,
) -> tuple[str, float, str, RuleSignals]:
    s = analyze_rules(p)
    lab, conf, reason = decision_from_signals(
        s,
        prob_device,
        device_threshold,
        accessory_threshold,
    )
    return lab, conf, reason, s


def class_precision_recall(
    pred: list[str],
    gold: list[str],
    target: str,
    weights: Optional[list[float]] = None,
) -> dict:
    if weights is None:
        weights = [1.0] * len(pred)

    tp = fp = fn = 0.0

    for p, g, w in zip(pred, gold, weights):
        w = float(max(w, 0.0))

        if p == target and g == target:
            tp += w
        elif p == target and g != target:
            fp += w
        elif p != target and g == target:
            fn += w

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calibrate_device_threshold(
    gold_rows: list[dict],
    probs: list[float],
    target_precision: float,
) -> float:
    gold = [r["gold_label"] for r in gold_rows]
    products = [gold_row_to_product(r) for r in gold_rows]
    signals = [analyze_rules(p) for p in products]

    candidates = []

    for t in np.arange(0.20, 0.996, 0.005):
        pred = [
            decision_from_signals(
                s,
                prob,
                device_threshold=float(t),
                accessory_threshold=0.0,
            )[0]
            for s, prob in zip(signals, probs)
        ]

        m = class_precision_recall(pred, gold, LABEL_DEVICE)

        precision = m["precision"] or 0.0
        recall = m["recall"] or 0.0

        if precision >= target_precision:
            candidates.append((recall, -float(t), precision))

    if not candidates:
        print(
            "WARNING: development set 上无法达到目标 DEVICE precision；"
            "使用保守阈值 0.95。"
        )
        return 0.95

    best = max(candidates)
    threshold = -best[1]

    print(
        f"calibrated DEVICE threshold={threshold:.3f} "
        f"(development precision={best[2]:.4f}, recall={best[0]:.4f})"
    )

    return threshold


def calibrate_accessory_threshold(
    gold_rows: list[dict],
    probs: list[float],
    target_precision: float,
) -> float:
    gold = [r["gold_label"] for r in gold_rows]
    products = [gold_row_to_product(r) for r in gold_rows]
    signals = [analyze_rules(p) for p in products]

    candidates = []

    for t in np.arange(0.005, 0.50, 0.005):
        pred = [
            decision_from_signals(
                s,
                prob,
                device_threshold=1.0,
                accessory_threshold=float(t),
            )[0]
            for s, prob in zip(signals, probs)
        ]

        m = class_precision_recall(pred, gold, LABEL_ACCESSORY)

        precision = m["precision"] or 0.0
        recall = m["recall"] or 0.0

        if precision >= target_precision:
            candidates.append((recall, float(t), precision))

    if not candidates:
        print(
            "WARNING: development set 上无法达到目标 ACCESSORY precision；"
            "使用保守阈值 0.10。"
        )
        return 0.10

    best = max(candidates)
    threshold = best[1]

    print(
        f"calibrated ACCESSORY threshold={threshold:.3f} "
        f"(development precision={best[2]:.4f}, recall={best[0]:.4f})"
    )

    return threshold


def development_metrics_rows(
    gold_rows: list[dict],
    probs: list[float],
    device_threshold: float,
    accessory_threshold: float,
) -> tuple[list[dict], list[dict]]:
    products = [gold_row_to_product(r) for r in gold_rows]
    signals = [analyze_rules(p) for p in products]
    gold = [r["gold_label"] for r in gold_rows]

    pred = []
    detailed = []

    for r, p, sig, prob in zip(gold_rows, products, signals, probs):
        lab, conf, reason = decision_from_signals(
            sig,
            prob,
            device_threshold,
            accessory_threshold,
        )
        pred.append(lab)

        detailed.append({
            "product_id": r["product_id"],
            "gold_label": r["gold_label"],
            "gold_form_factor": r["gold_form_factor"],
            "pred_label_oof": lab,
            "device_probability_oof": round(prob, 6),
            "decision_confidence_oof": round(conf, 6),
            "reason": reason,
            "title": r["title"],
            "leaf_categories": r["leaf_categories"],
            "rating_number": r["rating_number"],
            **asdict(sig),
        })

    rating_weights = [max(r["rating_number"], 1) for r in gold_rows]

    rows = []

    for target in (LABEL_DEVICE, LABEL_ACCESSORY):
        unweighted = class_precision_recall(pred, gold, target)
        weighted = class_precision_recall(pred, gold, target, rating_weights)

        rows.append({
            "metric_scope": "development_oof",
            "label": target,
            "precision": round(unweighted["precision"], 6)
            if unweighted["precision"] is not None else "",
            "recall": round(unweighted["recall"], 6)
            if unweighted["recall"] is not None else "",
            "rating_number_weighted_precision": round(weighted["precision"], 6)
            if weighted["precision"] is not None else "",
            "rating_number_weighted_recall": round(weighted["recall"], 6)
            if weighted["recall"] is not None else "",
            "n_gold": len(gold_rows),
        })

    coverage = sum(x != LABEL_AMBIGUOUS for x in pred) / len(pred)

    rows.append({
        "metric_scope": "development_oof",
        "label": "NON_AMBIGUOUS_COVERAGE",
        "precision": "",
        "recall": round(coverage, 6),
        "rating_number_weighted_precision": "",
        "rating_number_weighted_recall": "",
        "n_gold": len(gold_rows),
    })

    return rows, detailed


def make_form_model() -> Pipeline:
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=70_000,
                sublinear_tf=True,
            ),
        ),
    ])

    clf = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def calibrate_form_threshold(
    gold_rows: list[dict],
    folds: int,
    target_precision: float,
    seed: int,
) -> tuple[Optional[Pipeline], float, list[dict]]:
    rows = [
        r for r in gold_rows
        if r["gold_label"] == LABEL_DEVICE
        and r["gold_form_factor"] in VALID_FORMS
    ]

    if len(rows) < 100:
        print("WARNING: 有效 form gold 太少，不训练 form model。")
        return None, 1.0, []

    X = [
        build_model_text_from_values(
            r["title"],
            r["leaf_categories"],
            r["l3_categories"],
        )
        for r in rows
    ]
    y = np.array([r["gold_form_factor"] for r in rows])

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    base = make_form_model()

    all_classes = sorted(set(y))
    class_to_pos = {c: i for i, c in enumerate(all_classes)}
    oof = np.zeros((len(rows), len(all_classes)), dtype=float)

    for fold_no, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        model = clone(base)
        X_train = [X[i] for i in train_idx]
        y_train = y[train_idx]
        X_valid = [X[i] for i in valid_idx]

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_valid)

        for row_pos, probs in zip(valid_idx, proba):
            for cls, p in zip(model.classes_, probs):
                oof[row_pos, class_to_pos[cls]] = p

        print(f"  form CV fold {fold_no}/{folds} done", flush=True)

    max_prob = oof.max(axis=1)
    pred_class = np.array(all_classes)[oof.argmax(axis=1)]

    candidates = []

    for t in np.arange(0.40, 0.991, 0.01):
        mask = max_prob >= t
        if mask.sum() < 50:
            continue

        precision = float((pred_class[mask] == y[mask]).mean())
        coverage = float(mask.mean())

        if precision >= target_precision:
            candidates.append((coverage, -float(t), precision))

    if candidates:
        best = max(candidates)
        threshold = -best[1]
        achieved = best[2]
        coverage = best[0]
    else:
        threshold = 0.75
        mask = max_prob >= threshold
        achieved = float((pred_class[mask] == y[mask]).mean()) if mask.any() else 0.0
        coverage = float(mask.mean())
        print(
            "WARNING: form model 在 development CV 无法达到目标 precision；"
            "使用保守阈值 0.75。"
        )

    details = []

    for r, pred, mp in zip(rows, pred_class, max_prob):
        final = pred if mp >= threshold else UNKNOWN_FORM
        details.append({
            "product_id": r["product_id"],
            "gold_form_factor": r["gold_form_factor"],
            "pred_form_oof": final,
            "raw_pred_form_oof": pred,
            "max_probability_oof": round(float(mp), 6),
            "title": r["title"],
            "leaf_categories": r["leaf_categories"],
        })

    print(
        f"form threshold={threshold:.3f} "
        f"(development accepted precision={achieved:.4f}, coverage={coverage:.4f})"
    )

    final_model = make_form_model()
    final_model.fit(X, y)

    return final_model, threshold, details


def deterministic_form_hint(p: ProductRecord) -> Optional[str]:
    title = p.title

    hits = Counter()

    for form, regexes in FORM_PATTERNS.items():
        hits[form] = sum(1 for rx in regexes if rx.search(title))

    leaves = {x.lower() for x in split_pipe_values(p.leaf_categories)}

    if "over-ear headphones" in leaves:
        hits["OVER_EAR"] += 3
    if "on-ear headphones" in leaves:
        hits["ON_EAR"] += 3
    if "earbud headphones" in leaves or "in-ear headphones" in leaves:
        hits["EARBUD_INEAR"] += 3

    if not hits:
        return None

    ranked = hits.most_common()

    if not ranked or ranked[0][1] <= 0:
        return None

    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None

    return ranked[0][0]


def train_final_eligibility_model(gold_rows: list[dict]) -> Pipeline:
    rows = [
        r for r in gold_rows
        if r["gold_label"] in {LABEL_DEVICE, LABEL_ACCESSORY}
    ]

    X = [
        build_model_text_from_values(
            r["title"],
            r["leaf_categories"],
            r["l3_categories"],
        )
        for r in rows
    ]
    y = np.array([
        1 if r["gold_label"] == LABEL_DEVICE else 0
        for r in rows
    ], dtype=int)

    model = make_eligibility_model()
    model.fit(X, y)
    return model


def predict_probabilities_batched(
    model: Pipeline,
    products: list[ProductRecord],
    batch_size: int = 5000,
) -> dict[str, float]:
    out: dict[str, float] = {}

    for start in range(0, len(products), batch_size):
        batch = products[start:start + batch_size]
        X = [build_model_text(p) for p in batch]
        probs = model.predict_proba(X)[:, 1]

        for p, prob in zip(batch, probs):
            out[p.product_id] = float(prob)

        print(
            f"  predicted {min(start + batch_size, len(products)):,}/{len(products):,}",
            flush=True,
        )

    return out


def predict_forms_batched(
    model: Optional[Pipeline],
    products: list[ProductRecord],
    threshold: float,
    batch_size: int = 5000,
) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}

    if model is None:
        for p in products:
            hint = deterministic_form_hint(p)
            out[p.product_id] = (hint or UNKNOWN_FORM, 0.0)
        return out

    for start in range(0, len(products), batch_size):
        batch = products[start:start + batch_size]
        X = [build_model_text(p) for p in batch]
        probs = model.predict_proba(X)
        classes = model.classes_

        for p, row_probs in zip(batch, probs):
            pos = int(np.argmax(row_probs))
            raw_form = str(classes[pos])
            max_p = float(row_probs[pos])

            if max_p >= threshold:
                form = raw_form
            else:
                hint = deterministic_form_hint(p)
                form = hint or UNKNOWN_FORM

            out[p.product_id] = (form, max_p)

    return out


def classify_all_products(
    products: dict[str, ProductRecord],
    eligibility_model: Pipeline,
    form_model: Optional[Pipeline],
    device_threshold: float,
    accessory_threshold: float,
    form_threshold: float,
) -> dict[str, Decision]:
    plist = list(products.values())

    print("predicting eligibility probabilities...")
    p_device = predict_probabilities_batched(
        eligibility_model,
        plist,
    )

    preliminary = {}

    device_products = []

    for p in plist:
        lab, conf, reason, sig = raw_decision(
            p,
            p_device[p.product_id],
            device_threshold,
            accessory_threshold,
        )

        preliminary[p.product_id] = (lab, conf, reason, sig)

        if lab == LABEL_DEVICE:
            device_products.append(p)

    print(f"DEVICE candidates before form classification: {len(device_products):,}")

    forms = predict_forms_batched(
        form_model,
        device_products,
        form_threshold,
    )

    decisions: dict[str, Decision] = {}

    for p in plist:
        lab, conf, reason, sig = preliminary[p.product_id]

        if lab == LABEL_DEVICE:
            form, form_prob = forms.get(
                p.product_id,
                (UNKNOWN_FORM, 0.0),
            )
        else:
            form, form_prob = "NOT_APPLICABLE", 0.0

        decisions[p.product_id] = Decision(
            product_id=p.product_id,
            final_label=lab,
            device_probability=round(p_device[p.product_id], 6),
            decision_confidence=round(conf, 6),
            form_factor=form,
            form_probability=round(form_prob, 6),
            reason=reason,

            **asdict(sig),

            title=p.title,
            store=p.store,
            leaf_categories=p.leaf_categories,
            l3_categories=p.l3_categories,
            price=p.price,
            average_rating=p.average_rating,
            rating_number=p.rating_number,
            duplicate_rows=p.duplicate_rows,
            duplicate_title_conflict=p.duplicate_title_conflict,
            duplicate_price_conflict=p.duplicate_price_conflict,
            duplicate_rating_conflict=p.duplicate_rating_conflict,
        )

    return decisions


CONTROL_RX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
HTML_TAG_RX = re.compile(r"<[^>]+>")


# 评论基础清洗与精确去重

def clean_review_text(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    s = html.unescape(v)
    s = unicodedata.normalize("NFKC", s)
    s = HTML_TAG_RX.sub(" ", s)
    s = CONTROL_RX.sub(" ", s)
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_review_rating(v: Any) -> Optional[float]:
    x = to_float(v)
    return x if x is not None and 1.0 <= x <= 5.0 else None


def parse_verified(v: Any) -> Optional[bool]:
    if v is True or v is False:
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return None


def parse_review_datetime(v: Any) -> tuple[Optional[str], Optional[int]]:
    if v is None:
        return None, None
    try:
        if isinstance(v, (int, float)) or (
            isinstance(v, str) and v.strip().isdigit()
        ):
            x = float(v)
            if x > 1e12:
                x /= 1000.0
            dt = datetime.fromtimestamp(x, tz=timezone.utc)
            return dt.date().isoformat(), dt.year
    except (OSError, OverflowError, ValueError):
        return None, None

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None, None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.date().isoformat(), dt.year
        except ValueError:
            return None, None
    return None, None


class DiskDeduper:

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            path.unlink()
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("CREATE TABLE seen (k BLOB PRIMARY KEY) WITHOUT ROWID")
        self.pending = 0

    def is_new(self, parts: Iterable[Any]) -> bool:
        payload = "\x1f".join("" if x is None else str(x) for x in parts)
        key = hashlib.blake2b(payload.encode("utf-8", errors="replace"), digest_size=16).digest()
        cur = self.conn.execute("INSERT OR IGNORE INTO seen(k) VALUES (?)", (key,))
        self.pending += 1
        if self.pending >= 50_000:
            self.conn.commit()
            self.pending = 0
        return cur.rowcount == 1

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


REVIEW_FIELDS = [
    "product_id",
    "asin",
    "rating",
    "review_date",
    "review_year",
    "verified_purchase",
    "helpful_vote",
    "review_title",
    "text_raw",
    "text_clean",
    "text_length_chars",
]


class AnalysisReviewWriter:
    def __init__(self, outdir: Path, table_format: str, batch_size: int = 50_000):
        self.outdir = outdir
        self.batch_size = batch_size
        self.buffer: list[dict] = []
        self.mode = table_format
        self.parquet_writer = None
        self.csv_file = None
        self.csv_writer = None
        self.pa = None
        self.pq = None

        if table_format not in {"auto", "parquet", "csv.gz"}:
            raise ValueError(f"未知 table format: {table_format}")

        use_parquet = table_format in {"auto", "parquet"}
        if use_parquet:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                self.pa = pa
                self.pq = pq
                self.mode = "parquet"
            except ImportError:
                if table_format == "parquet":
                    raise SystemExit(
                        "--table-format parquet 需要 pyarrow。请运行：pip install pyarrow"
                    )
                print("WARNING: pyarrow 未安装，reviews_analysis 自动回退为 csv.gz")
                self.mode = "csv.gz"

        if self.mode == "parquet":
            self.path = outdir / "reviews_analysis.parquet"
            self.schema = self.pa.schema([
                ("product_id", self.pa.string()),
                ("asin", self.pa.string()),
                ("rating", self.pa.float64()),
                ("review_date", self.pa.string()),
                ("review_year", self.pa.int32()),
                ("verified_purchase", self.pa.bool_()),
                ("helpful_vote", self.pa.int64()),
                ("review_title", self.pa.string()),
                ("text_raw", self.pa.string()),
                ("text_clean", self.pa.string()),
                ("text_length_chars", self.pa.int32()),
            ])
        else:
            self.path = outdir / "reviews_analysis.csv.gz"
            self.csv_file = gzip.open(self.path, "wt", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=REVIEW_FIELDS)
            self.csv_writer.writeheader()

    def write(self, row: dict) -> None:
        if self.mode == "parquet":
            self.buffer.append(row)
            if len(self.buffer) >= self.batch_size:
                self.flush()
        else:
            self.csv_writer.writerow(row)

    def flush(self) -> None:
        if self.mode != "parquet" or not self.buffer:
            return
        table = self.pa.Table.from_pylist(self.buffer, schema=self.schema)
        if self.parquet_writer is None:
            self.parquet_writer = self.pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
            )
        self.parquet_writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        if self.mode == "parquet":
            self.flush()
            if self.parquet_writer is not None:
                self.parquet_writer.close()
            elif not self.path.exists():
                table = self.pa.Table.from_pylist([], schema=self.schema)
                self.pq.write_table(table, self.path, compression="zstd")
        elif self.csv_file is not None:
            self.csv_file.close()


def materialize_analysis_reviews(
    reviews_path: Path,
    decisions: dict[str, Decision],
    outdir: Path,
    table_format: str,
) -> tuple[dict[str, dict], list[dict], Path]:
    eligible = {pid for pid, d in decisions.items() if d.final_label == LABEL_DEVICE}
    stats: dict[str, dict] = defaultdict(lambda: {
        "source_review_count": 0,
        "analysis_review_count": 0,
        "valid_rating_review_count": 0,
        "rating_sum": 0.0,
        "verified_true_count": 0,
        "verified_false_count": 0,
        "verified_missing_count": 0,
        "first_review_date": None,
        "last_review_date": None,
    })

    q = Counter()
    product_analysis_counts = Counter()
    writer = AnalysisReviewWriter(outdir, table_format)
    deduper = DiskDeduper(outdir / "_review_dedupe.sqlite")

    try:
        for rec in iter_jsonl(reviews_path):
            q["reviews_scanned"] += 1
            pid = normalize_id(rec.get("parent_asin"))
            if not pid:
                q["missing_parent_asin"] += 1
                continue
            if pid not in eligible:
                q["non_device_reviews"] += 1
                continue

            q["device_source_reviews"] += 1
            s = stats[pid]
            s["source_review_count"] += 1

            rating = parse_review_rating(rec.get("rating"))
            if rating is None:
                q["invalid_or_missing_rating"] += 1
            else:
                s["valid_rating_review_count"] += 1
                s["rating_sum"] += rating

            verified = parse_verified(rec.get("verified_purchase"))
            if verified is True:
                s["verified_true_count"] += 1
                q["verified_true"] += 1
            elif verified is False:
                s["verified_false_count"] += 1
                q["verified_false"] += 1
            else:
                s["verified_missing_count"] += 1
                q["verified_missing"] += 1

            review_date, review_year = parse_review_datetime(rec.get("timestamp"))
            if review_date is None:
                q["invalid_or_missing_timestamp"] += 1
            else:
                if s["first_review_date"] is None or review_date < s["first_review_date"]:
                    s["first_review_date"] = review_date
                if s["last_review_date"] is None or review_date > s["last_review_date"]:
                    s["last_review_date"] = review_date

            raw_text = rec.get("text") if isinstance(rec.get("text"), str) else ""
            text_clean = clean_review_text(raw_text)
            if not text_clean:
                q["empty_text_dropped"] += 1
                continue

            user_id = normalize_id(rec.get("user_id")) or ""
            if not deduper.is_new((pid, user_id, rec.get("timestamp"), rating, text_clean)):
                q["exact_duplicate_dropped"] += 1
                continue

            review_title = clean_review_text(rec.get("title"))
            helpful_vote = max(to_int(rec.get("helpful_vote"), 0), 0)
            asin = normalize_id(rec.get("asin")) or ""

            row = {
                "product_id": pid,
                "asin": asin,
                "rating": rating,
                "review_date": review_date,
                "review_year": review_year,
                "verified_purchase": verified,
                "helpful_vote": helpful_vote,
                "review_title": review_title,
                "text_raw": raw_text,
                "text_clean": text_clean,
                "text_length_chars": len(text_clean),
            }
            writer.write(row)
            q["analysis_reviews_written"] += 1
            s["analysis_review_count"] += 1
            product_analysis_counts[pid] += 1

            if q["reviews_scanned"] % 500_000 == 0:
                print(
                    f"  reviews scanned {q['reviews_scanned']:,} | "
                    f"DEVICE source {q['device_source_reviews']:,} | "
                    f"analysis written {q['analysis_reviews_written']:,}",
                    flush=True,
                )
    finally:
        writer.close()
        deduper.close()

    top100 = []
    for pid, n in product_analysis_counts.most_common(100):
        d = decisions[pid]
        top100.append({
            "product_id": pid,
            "analysis_review_count": n,
            "source_review_count": stats[pid]["source_review_count"],
            "title": d.title,
            "store": d.store,
            "price": d.price,
            "average_rating": d.average_rating,
            "rating_number": d.rating_number,
            "form_factor": d.form_factor,
            "device_probability": d.device_probability,
        })
    write_csv(
        outdir / "review_top100_products.csv",
        top100,
        [
            "product_id", "analysis_review_count", "source_review_count",
            "title", "store", "price", "average_rating", "rating_number",
            "form_factor", "device_probability",
        ],
    )

    top10_n = sum(n for _, n in product_analysis_counts.most_common(10))
    written = q["analysis_reviews_written"]
    qc = [
        {"section": "reviews", "metric": k, "value": v}
        for k, v in sorted(q.items())
    ]
    qc.extend([
        {"section": "reviews", "metric": "eligible_device_products", "value": len(eligible)},
        {"section": "reviews", "metric": "products_with_analysis_reviews", "value": len(product_analysis_counts)},
        {"section": "reviews", "metric": "top10_analysis_review_share", "value": round(top10_n / written, 6) if written else 0},
        {"section": "reviews", "metric": "reviews_analysis_path", "value": str(writer.path)},
        {"section": "reviews", "metric": "reviews_analysis_format", "value": writer.mode},
    ])

    print(f"analysis reviews written: {written:,} -> {writer.path}")
    return stats, qc, writer.path


DECISION_FIELDS = list(Decision.__dataclass_fields__.keys())


def write_small_parquet_if_available(path: Path, rows: list[dict]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return True


# 输出商品与评论分析底表

def export_decisions(
    decisions: dict[str, Decision],
    outdir: Path,
) -> list[dict]:
    rows = [asdict(d) for d in decisions.values()]
    write_csv(outdir / "product_eligibility_final.csv", rows, DECISION_FIELDS)

    counts = Counter(d.final_label for d in decisions.values())
    forms = Counter(
        d.form_factor for d in decisions.values() if d.final_label == LABEL_DEVICE
    )
    reasons = Counter(d.reason for d in decisions.values())
    total = len(decisions)
    summary = []
    for lab in (LABEL_DEVICE, LABEL_ACCESSORY, LABEL_AMBIGUOUS):
        n = counts[lab]
        summary.append({
            "section": "eligibility", "name": lab, "n": n,
            "share": round(n / total, 6) if total else 0,
        })
    device_n = max(counts[LABEL_DEVICE], 1)
    for form, n in forms.most_common():
        summary.append({
            "section": "device_form", "name": form, "n": n,
            "share": round(n / device_n, 6),
        })
    for reason, n in reasons.most_common():
        summary.append({
            "section": "decision_reason", "name": reason, "n": n,
            "share": round(n / total, 6) if total else 0,
        })
    write_csv(
        outdir / "product_eligibility_summary_final.csv",
        summary,
        ["section", "name", "n", "share"],
    )

    suspicious = [
        asdict(d) for d in decisions.values()
        if (
            d.final_label == LABEL_AMBIGUOUS
            or d.taxonomy_anomaly
            or d.duplicate_title_conflict
            or d.duplicate_price_conflict
            or d.duplicate_rating_conflict
        )
    ]
    write_csv(outdir / "product_quality_flags.csv", suspicious, DECISION_FIELDS)
    return summary


def export_products_analysis(
    decisions: dict[str, Decision],
    review_stats: dict[str, dict],
    outdir: Path,
) -> tuple[list[dict], list[dict]]:
    rows = []
    for pid, d in decisions.items():
        if d.final_label != LABEL_DEVICE:
            continue
        s = review_stats.get(pid, {})
        source_n = int(s.get("source_review_count", 0))
        valid_rating_n = int(s.get("valid_rating_review_count", 0))
        rating_sum = float(s.get("rating_sum", 0.0))
        verified_true = int(s.get("verified_true_count", 0))
        analysis_n = int(s.get("analysis_review_count", 0))

        rows.append({
            "product_id": pid,
            "title": d.title,
            "store": d.store,
            "price": d.price,
            "average_rating": d.average_rating,
            "rating_number": d.rating_number,
            "source_review_count": source_n,
            "analysis_review_count": analysis_n,
            "text_review_coverage": round(analysis_n / source_n, 6) if source_n else None,
            "valid_rating_review_count": valid_rating_n,
            "observed_avg_rating": round(rating_sum / valid_rating_n, 4) if valid_rating_n else None,
            "verified_review_count": verified_true,
            "verified_review_share": round(verified_true / source_n, 6) if source_n else None,
            "first_review_date": s.get("first_review_date"),
            "last_review_date": s.get("last_review_date"),
            "form_factor": d.form_factor,
            "leaf_categories": d.leaf_categories,
            "l3_categories": d.l3_categories,
            "device_probability": d.device_probability,
            "decision_confidence": d.decision_confidence,
            "taxonomy_anomaly": d.taxonomy_anomaly,
            "duplicate_rows": d.duplicate_rows,
            "duplicate_title_conflict": d.duplicate_title_conflict,
            "duplicate_price_conflict": d.duplicate_price_conflict,
            "duplicate_rating_conflict": d.duplicate_rating_conflict,
        })

    fields = list(rows[0].keys()) if rows else [
        "product_id", "title", "store", "price", "average_rating", "rating_number",
        "source_review_count", "analysis_review_count", "text_review_coverage",
        "valid_rating_review_count", "observed_avg_rating", "verified_review_count",
        "verified_review_share", "first_review_date", "last_review_date", "form_factor",
        "leaf_categories", "l3_categories", "device_probability", "decision_confidence",
        "taxonomy_anomaly", "duplicate_rows", "duplicate_title_conflict",
        "duplicate_price_conflict", "duplicate_rating_conflict",
    ]
    write_csv(outdir / "products_analysis.csv", rows, fields)
    parquet_written = write_small_parquet_if_available(outdir / "products_analysis.parquet", rows)

    n = len(rows)
    price_fill = sum(r["price"] is not None for r in rows)
    rating_fill = sum(r["average_rating"] is not None for r in rows)
    with_reviews = sum((r["source_review_count"] or 0) > 0 for r in rows)
    with_text = sum((r["analysis_review_count"] or 0) > 0 for r in rows)
    qc = [
        {"section": "products", "metric": "device_products", "value": n},
        {"section": "products", "metric": "price_fill_rate", "value": round(price_fill / n, 6) if n else 0},
        {"section": "products", "metric": "average_rating_fill_rate", "value": round(rating_fill / n, 6) if n else 0},
        {"section": "products", "metric": "products_with_source_reviews", "value": with_reviews},
        {"section": "products", "metric": "products_with_analysis_text", "value": with_text},
        {"section": "products", "metric": "products_analysis_parquet_written", "value": int(parquet_written)},
    ]
    return rows, qc


# 主流程

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Amazon 耳机商品与评论预处理"
    )
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--reviews", default=str(DEFAULT_REVIEWS))
    ap.add_argument(
        "--gold-file",
        default=str(DEFAULT_GOLD),
        help="人工复核 Gold CSV；默认使用项目固定路径",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target-device-precision", type=float, default=0.98)
    ap.add_argument("--target-accessory-precision", type=float, default=0.97)
    ap.add_argument("--target-form-precision", type=float, default=0.95)
    ap.add_argument(
        "--table-format",
        choices=["auto", "parquet", "csv.gz"],
        default="auto",
        help="reviews_analysis 输出格式；auto 优先 parquet，无 pyarrow 时回退 csv.gz",
    )
    ap.add_argument(
        "--skip-reviews",
        action="store_true",
        help="仅调试商品分类时使用；正式 Stage 1 交付不要加这个参数",
    )
    args = ap.parse_args()

    root = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    outdir = root / "stage1_preprocessed"
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path = resolve_data_file(Path(args.meta))
    gold_path = resolve_gold_csv(Path(args.gold_file))

    reviews_path_prepared = None
    if not args.skip_reviews:
        reviews_path_prepared = resolve_data_file(Path(args.reviews))

    print("== Stage 1 paths ==")
    print("meta    :", meta_path)
    print("reviews :", reviews_path_prepared if reviews_path_prepared is not None else "<skipped>")
    print("gold    :", gold_path)
    print("output  :", outdir)

    print("\n== 1. load target metadata ==")
    products = load_target_products(meta_path)

    print("\n== 2. load Gold labels ==")
    gold_rows = load_gold_csv(gold_path)
    if len(gold_rows) < max(100, args.cv_folds * 10):
        raise ValueError(f"有效 Gold 样本过少: {len(gold_rows)}")
    label_counts = Counter(r["gold_label"] for r in gold_rows)

    print("\n== 3. calibrate eligibility model ==")
    dev_probs = development_probabilities(gold_rows, folds=args.cv_folds, seed=args.seed)
    device_threshold = calibrate_device_threshold(
        gold_rows, dev_probs, target_precision=args.target_device_precision
    )
    accessory_threshold = calibrate_accessory_threshold(
        gold_rows, dev_probs, target_precision=args.target_accessory_precision
    )
    if accessory_threshold >= device_threshold:
        accessory_threshold = min(accessory_threshold, device_threshold - 0.10)

    dev_metrics, dev_detail = development_metrics_rows(
        gold_rows, dev_probs, device_threshold, accessory_threshold
    )
    for row in dev_metrics:
        row["metric_scope"] = "internal_oof"
    for row in dev_detail:
        row["validation_scope"] = "internal_oof"
    write_csv(
        outdir / "internal_oof_metrics.csv",
        dev_metrics,
        [
            "metric_scope", "label", "precision", "recall",
            "rating_number_weighted_precision", "rating_number_weighted_recall", "n_gold",
        ],
    )
    if dev_detail:
        write_csv(outdir / "internal_oof_predictions.csv", dev_detail, list(dev_detail[0].keys()))

    print("\n== 4. calibrate form-factor model ==")
    form_model, form_threshold, form_dev_detail = calibrate_form_threshold(
        gold_rows,
        folds=args.cv_folds,
        target_precision=args.target_form_precision,
        seed=args.seed,
    )
    if form_dev_detail:
        for row in form_dev_detail:
            row["validation_scope"] = "internal_oof"
        write_csv(
            outdir / "internal_form_oof_predictions.csv",
            form_dev_detail,
            list(form_dev_detail[0].keys()),
        )

    print("\n== 5. train final models ==")
    eligibility_model = train_final_eligibility_model(gold_rows)
    joblib.dump(eligibility_model, outdir / "eligibility_model.joblib")
    if form_model is not None:
        joblib.dump(form_model, outdir / "form_factor_model.joblib")

    config = {
        "version": "1.0",
        "target_l2": TARGET_L2,
        "reviewed_gold_rows": len(gold_rows),
        "gold_distribution": dict(label_counts),
        "device_threshold": device_threshold,
        "accessory_threshold": accessory_threshold,
        "form_threshold": form_threshold,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "text_cleaning": "NFKC, HTML/control character and whitespace normalization",
    }
    with (outdir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n== 6. classify target products ==")
    decisions = classify_all_products(
        products,
        eligibility_model,
        form_model,
        device_threshold,
        accessory_threshold,
        form_threshold,
    )
    eligibility_summary = export_decisions(decisions, outdir)

    review_stats: dict[str, dict] = {}
    review_qc: list[dict] = []
    review_output_path = None

    if not args.skip_reviews:
        print("\n== 7. build DEVICE review table ==")
        reviews_path = reviews_path_prepared
        assert reviews_path is not None
        review_stats, review_qc, review_output_path = materialize_analysis_reviews(
            reviews_path,
            decisions,
            outdir,
            args.table_format,
        )
    else:
        print("\nWARNING: --skip-reviews enabled; Stage 1 is not fully materialized.")

    print("\n== 8. build product analysis table ==")
    product_rows, product_qc = export_products_analysis(decisions, review_stats, outdir)

    label_counts_final = Counter(d.final_label for d in decisions.values())
    overall_qc = [
        {"section": "eligibility", "metric": "target_unique_products", "value": len(decisions)},
        {"section": "eligibility", "metric": "device_products", "value": label_counts_final[LABEL_DEVICE]},
        {"section": "eligibility", "metric": "accessory_products", "value": label_counts_final[LABEL_ACCESSORY]},
        {"section": "eligibility", "metric": "ambiguous_products", "value": label_counts_final[LABEL_AMBIGUOUS]},
        {"section": "eligibility", "metric": "reviewed_gold_rows", "value": len(gold_rows)},
        {"section": "eligibility", "metric": "device_threshold", "value": device_threshold},
        {"section": "eligibility", "metric": "accessory_threshold", "value": accessory_threshold},
        {"section": "eligibility", "metric": "form_threshold", "value": form_threshold},
    ]
    if review_output_path is not None:
        overall_qc.append({
            "section": "delivery", "metric": "reviews_analysis_file", "value": str(review_output_path)
        })
    overall_qc.extend(product_qc)
    overall_qc.extend(review_qc)
    write_csv(
        outdir / "preprocessing_quality_summary.csv",
        overall_qc,
        ["section", "metric", "value"],
    )

    print("\nDONE")
    print("output:", outdir)
    print("main product table:", outdir / "products_analysis.csv")
    if review_output_path:
        print("main review table:", review_output_path)


if __name__ == "__main__":
    main()
