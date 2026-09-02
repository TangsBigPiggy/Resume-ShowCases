#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Amazon Reviews 2023 Electronics → Headphones, Earbuds & Accessories
Stage 1.1 EDA FINAL

目的
----
在任何自定义 DEVICE/ACCESSORY 或形态筛选之前，只基于 Amazon 平台原始类目树观察：
- 目标类目规模与类目污染
- 商品字段完整性与分布
- 商品 ID / 类目路径 / 重复记录质量
- 评论与商品 join 覆盖
- Verified Purchase、星级、年份、文本长度

本脚本只负责“看清原始目标池”，不做整机筛选，不做 TF-IDF/聚类/NLP。

输入
----
E:\DA Cases\Amazon\0.原始数据\meta_Electronics.jsonl
E:\DA Cases\Amazon\0.原始数据\Electronics.jsonl
支持 .jsonl / .jsonl.gz / .json.gz；若传入同名目录，会自动寻找真实数据文件。

输出
----
stage1_eda/output/*.csv
stage1_eda/figures/*.png
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TARGET_L2 = "Headphones, Earbuds & Accessories"
DEFAULT_DATA_DIR = Path(r"E:\DA Cases\Amazon\0.原始数据")
DEFAULT_META = DEFAULT_DATA_DIR / "meta_Electronics.jsonl"
DEFAULT_REVIEWS = DEFAULT_DATA_DIR / "Electronics.jsonl"
PRICE_NUM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")


@dataclass
class ParseStats:
    lines_seen: int = 0
    blank_lines: int = 0
    bad_json: int = 0
    non_object_json: int = 0
    yielded_objects: int = 0


def setup_font() -> None:
    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            break
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })


def _is_jsonl_file(path: Path) -> bool:
    if not path.is_file():
        return False
    n = path.name.lower()
    return n.endswith(".jsonl") or n.endswith(".jsonl.gz") or n.endswith(".json.gz")


def resolve_data_file(path: Path) -> Path:
    r"""Resolve a JSONL input robustly.

    Supports the user's current Windows layout where the outer folder and the
    actual file have the same name, e.g.::

        E:\DA Cases\Amazon\0.原始数据\Electronics.jsonl\Electronics.jsonl

    It also accepts a direct .jsonl/.gz file. Directory search is deliberately
    bounded to the supplied directory so metadata and reviews cannot be mixed.
    """
    path = Path(path)

    variants = [path]
    s = str(path)
    variants.append(Path(s[:-3]) if s.lower().endswith(".gz") else Path(s + ".gz"))
    for cand in variants:
        if cand.is_file() and _is_jsonl_file(cand):
            return cand

    if path.is_dir():
        # Prefer the exact same-name child used by the current download layout.
        requested = path.name
        exact_names = {requested}
        if requested.lower().endswith(".gz"):
            exact_names.add(requested[:-3])
        else:
            exact_names.add(requested + ".gz")

        exact_hits = [
            child for child in path.iterdir()
            if child.is_file() and _is_jsonl_file(child) and child.name in exact_names
        ]
        if exact_hits:
            return max(exact_hits, key=lambda x: x.stat().st_size)

        # Fallback: any JSONL-like file directly inside this specific folder.
        hits = [child for child in path.iterdir() if _is_jsonl_file(child)]
        if hits:
            return max(hits, key=lambda x: x.stat().st_size)

    raise FileNotFoundError(
        f"未找到可读取 JSONL 文件: {path}\n"
        f"exists={path.exists()} is_dir={path.is_dir()} is_file={path.is_file()}\n"
        "当前默认原始数据目录应为 E:\\DA Cases\\Amazon\\0.原始数据，"
        "且允许 .jsonl 同名文件夹内再放同名 .jsonl 文件。"
    )


def iter_jsonl(path: Path, stats: Optional[ParseStats] = None) -> Iterator[dict]:
    stats = stats if stats is not None else ParseStats()
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stats.lines_seen += 1
            line = line.strip()
            if not line:
                stats.blank_lines += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats.bad_json += 1
                continue
            if not isinstance(obj, dict):
                stats.non_object_json += 1
                continue
            stats.yielded_objects += 1
            yield obj


def norm_text(v: Any) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def normalize_id(v: Any) -> Optional[str]:
    s = norm_text(v)
    return s or None


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


def to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> Optional[int]:
    try:
        x = float(v)
        return int(x) if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def parse_ts(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
            x = float(v)
            if x > 1e12:
                x /= 1000.0
            return datetime.fromtimestamp(x, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def percentile(vals: list[float | int], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def write_csv(path: Path, rows, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def top_rows(counter: Counter, total: int, k: int = 50) -> list[dict]:
    return [
        {"name": name, "n": n, "share": round(n / total, 6) if total else 0}
        for name, n in counter.most_common(k)
    ]


def barh(counter: Counter, title: str, dest: Path, k: int = 15) -> None:
    items = counter.most_common(k)
    if not items:
        return
    labels = [x for x, _ in reversed(items)]
    values = [n for _, n in reversed(items)]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel("count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def hist(vals: list[float], title: str, dest: Path, bins, xlabel: str) -> None:
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.hist(vals, bins=bins, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def eda_meta(meta_path: Path, out: Path, fig: Path) -> set[str]:
    ps = ParseStats()
    n_all = n_target_rows = 0
    ids: set[str] = set()
    id_counts = Counter()
    missing = Counter()
    leaf = Counter()
    l3 = Counter()
    stores = Counter()
    path_depth = Counter()
    path_multiplicity = Counter()
    prices: list[float] = []
    ratings: list[float] = []
    rating_n: list[int] = []
    title_len: list[int] = []

    for rec in iter_jsonl(meta_path, ps):
        n_all += 1
        paths = flatten_cats(rec.get("categories"))
        tpaths = target_paths(paths)
        if not tpaths:
            continue
        n_target_rows += 1
        path_multiplicity[str(len(tpaths))] += 1

        pid = normalize_id(rec.get("parent_asin")) or normalize_id(rec.get("asin"))
        if pid:
            ids.add(pid)
            id_counts[pid] += 1
        else:
            missing["product_id"] += 1

        title = norm_text(rec.get("title"))
        if title:
            title_len.append(len(title))
        else:
            missing["title"] += 1

        store = norm_text(rec.get("store"))
        if store:
            stores[store[:120]] += 1
        else:
            missing["store"] += 1

        for p in tpaths:
            path_depth[str(len(p))] += 1
            leaf[p[-1] if p else "<empty>"] += 1
            l3[p[2] if len(p) >= 3 else "<depth<3>"] += 1

        price = parse_price(rec.get("price"))
        if price is None:
            missing["price"] += 1
        else:
            prices.append(price)

        ar = to_float(rec.get("average_rating"))
        if ar is None or not (1 <= ar <= 5):
            missing["average_rating"] += 1
        else:
            ratings.append(ar)

        rn = to_int(rec.get("rating_number"))
        if rn is None or rn < 0:
            missing["rating_number"] += 1
        else:
            rating_n.append(rn)

        if n_all % 200_000 == 0:
            print(f"  meta scanned {n_all:,} | target rows {n_target_rows:,}", flush=True)

    duplicate_ids = sum(c > 1 for c in id_counts.values())
    duplicate_extra_rows = sum(max(c - 1, 0) for c in id_counts.values())
    write_csv(out / "meta_overview.csv", [
        {"item": "electronics_meta_rows", "value": n_all},
        {"item": "target_meta_rows", "value": n_target_rows},
        {"item": "target_unique_product_ids", "value": len(ids)},
        {"item": "target_share", "value": round(n_target_rows / n_all, 6) if n_all else 0},
        {"item": "duplicate_product_ids", "value": duplicate_ids},
        {"item": "duplicate_rows_beyond_first", "value": duplicate_extra_rows},
        {"item": "price_p50", "value": percentile(prices, .5) if prices else ""},
        {"item": "price_p90", "value": percentile(prices, .9) if prices else ""},
        {"item": "rating_mean", "value": round(sum(ratings) / len(ratings), 4) if ratings else ""},
        {"item": "rating_number_p50", "value": percentile(rating_n, .5) if rating_n else ""},
        {"item": "rating_number_p90", "value": percentile(rating_n, .9) if rating_n else ""},
    ], ["item", "value"])

    fields = ["product_id", "title", "store", "price", "average_rating", "rating_number"]
    write_csv(out / "meta_missingness.csv", [
        {
            "field": f,
            "empty_or_invalid": missing[f],
            "valid": n_target_rows - missing[f],
            "fill_rate": round(1 - missing[f] / n_target_rows, 6) if n_target_rows else 0,
        }
        for f in fields
    ], ["field", "empty_or_invalid", "valid", "fill_rate"])

    write_csv(out / "meta_leaf_category.csv", top_rows(leaf, sum(leaf.values()), 60), ["name", "n", "share"])
    write_csv(out / "meta_l3_category.csv", top_rows(l3, sum(l3.values()), 60), ["name", "n", "share"])
    write_csv(out / "meta_store_top50.csv", top_rows(stores, n_target_rows, 50), ["name", "n", "share"])
    write_csv(out / "meta_path_depth.csv", [
        {"path_depth": k, "n": v} for k, v in sorted(path_depth.items(), key=lambda x: int(x[0]))
    ], ["path_depth", "n"])
    write_csv(out / "meta_path_multiplicity.csv", [
        {"matching_target_paths": k, "n": v} for k, v in sorted(path_multiplicity.items(), key=lambda x: int(x[0]))
    ], ["matching_target_paths", "n"])
    write_csv(out / "parser_quality_meta.csv", [{
        "lines_seen": ps.lines_seen,
        "blank_lines": ps.blank_lines,
        "bad_json": ps.bad_json,
        "non_object_json": ps.non_object_json,
        "yielded_objects": ps.yielded_objects,
    }], ["lines_seen", "blank_lines", "bad_json", "non_object_json", "yielded_objects"])

    rband = Counter()
    for x in rating_n:
        if x == 0: rband["0"] += 1
        elif x < 10: rband["1-9"] += 1
        elif x < 50: rband["10-49"] += 1
        elif x < 200: rband["50-199"] += 1
        elif x < 1000: rband["200-999"] += 1
        else: rband["1000+"] += 1
    write_csv(out / "meta_rating_number_band.csv", [
        {"band": k, "n": rband[k]} for k in ["0", "1-9", "10-49", "50-199", "200-999", "1000+"]
    ], ["band", "n"])

    pband = Counter()
    for x in prices:
        if x < 10: pband["<10"] += 1
        elif x < 25: pband["10-24"] += 1
        elif x < 50: pband["25-49"] += 1
        elif x < 100: pband["50-99"] += 1
        elif x < 200: pband["100-199"] += 1
        else: pband["200+"] += 1
    write_csv(out / "meta_price_band.csv", [
        {"band": k, "n": pband[k]} for k in ["<10", "10-24", "25-49", "50-99", "100-199", "200+"]
    ], ["band", "n"])

    barh(leaf, "Target category — leaf category", fig / "fig01_leaf_category.png")
    barh(l3, "Target category — L3", fig / "fig02_l3_category.png")
    barh(stores, "Store / brand SKU count", fig / "fig03_store.png")
    hist(ratings, "average_rating", fig / "fig04_rating.png", [1,1.5,2,2.5,3,3.5,4,4.5,5.01], "average_rating")
    hist([math.log10(x) for x in rating_n if x > 0], "rating_number log10", fig / "fig05_rating_number_log.png", 20, "log10(rating_number)")
    hist([x for x in prices if x <= 300], "price USD, clipped at 300", fig / "fig06_price.png", 30, "USD")
    hist([x for x in title_len if x <= 250], "title length", fig / "fig07_title_len.png", 30, "chars")

    print(f"meta done: target_rows={n_target_rows:,} unique_target_ids={len(ids):,}")
    return ids


def eda_reviews(reviews_path: Path, target_ids: set[str], out: Path, fig: Path) -> None:
    ps = ParseStats()
    n_all = n_hit = missing_parent = 0
    vp = Counter()
    stars = Counter()
    years = Counter()
    text_len: list[int] = []
    missing = Counter()

    for rec in iter_jsonl(reviews_path, ps):
        n_all += 1
        pid = normalize_id(rec.get("parent_asin"))
        if not pid:
            missing_parent += 1
            continue
        if pid not in target_ids:
            continue
        n_hit += 1

        flag = rec.get("verified_purchase")
        if flag is True: vp["true"] += 1
        elif flag is False: vp["false"] += 1
        else: vp["other_or_missing"] += 1

        r = to_float(rec.get("rating"))
        if r is None or not (1 <= r <= 5):
            missing["rating"] += 1
        else:
            rr = round(r)
            stars[str(int(rr)) if abs(r - rr) < 1e-9 else "non_integer_valid"] += 1

        dt = parse_ts(rec.get("timestamp"))
        if dt is None:
            missing["timestamp"] += 1
        else:
            years[str(dt.year)] += 1

        txt = rec.get("text")
        if not isinstance(txt, str) or not txt.strip():
            missing["text"] += 1
        else:
            text_len.append(len(txt.strip()))

        if n_all % 500_000 == 0:
            print(f"  reviews scanned {n_all:,} | target hits {n_hit:,}", flush=True)

    write_csv(out / "review_overview.csv", [
        {"item": "reviews_scanned", "value": n_all},
        {"item": "reviews_on_target_products", "value": n_hit},
        {"item": "target_review_share", "value": round(n_hit / n_all, 6) if n_all else 0},
        {"item": "reviews_missing_parent_asin", "value": missing_parent},
        {"item": "verified_true", "value": vp["true"]},
        {"item": "verified_false", "value": vp["false"]},
        {"item": "verified_other", "value": vp["other_or_missing"]},
        {"item": "rating_missing_or_invalid", "value": missing["rating"]},
        {"item": "timestamp_missing_or_invalid", "value": missing["timestamp"]},
        {"item": "text_missing", "value": missing["text"]},
        {"item": "text_len_p50", "value": percentile(text_len, .5) if text_len else ""},
        {"item": "text_len_p90", "value": percentile(text_len, .9) if text_len else ""},
    ], ["item", "value"])
    write_csv(out / "review_verified.csv", [
        {"verified_purchase": k, "n": vp[k]} for k in ["true", "false", "other_or_missing"]
    ], ["verified_purchase", "n"])
    write_csv(out / "review_star.csv", [
        {"star": k, "n": stars[k]} for k in ["1", "2", "3", "4", "5", "non_integer_valid"]
    ], ["star", "n"])
    write_csv(out / "review_year.csv", [
        {"year": y, "n": years[y]} for y in sorted(years)
    ], ["year", "n"])
    write_csv(out / "review_join_quality.csv", [
        {"metric": "reviews_scanned", "value": n_all},
        {"metric": "missing_parent_asin", "value": missing_parent},
        {"metric": "target_join_hits", "value": n_hit},
        {"metric": "target_join_hit_rate", "value": round(n_hit / n_all, 6) if n_all else 0},
    ], ["metric", "value"])
    write_csv(out / "parser_quality_reviews.csv", [{
        "lines_seen": ps.lines_seen,
        "blank_lines": ps.blank_lines,
        "bad_json": ps.bad_json,
        "non_object_json": ps.non_object_json,
        "yielded_objects": ps.yielded_objects,
    }], ["lines_seen", "blank_lines", "bad_json", "non_object_json", "yielded_objects"])

    bands = Counter()
    for x in text_len:
        if x < 10: bands["<10"] += 1
        elif x < 25: bands["10-24"] += 1
        elif x < 50: bands["25-49"] += 1
        elif x < 200: bands["50-199"] += 1
        else: bands["200+"] += 1
    write_csv(out / "review_text_len_band.csv", [
        {"band": k, "n": bands[k]} for k in ["<10", "10-24", "25-49", "50-199", "200+"]
    ], ["band", "n"])

    barh(vp, "verified_purchase on target products", fig / "fig08_verified.png", 5)
    barh(Counter({k: stars[k] for k in ["1","2","3","4","5"]}), "star on target products", fig / "fig09_star.png", 5)
    if years:
        ys = sorted(years)
        f, ax = plt.subplots(figsize=(8.2, 3.8))
        ax.bar(ys, [years[y] for y in ys])
        ax.set_title("review year on target products")
        ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        f.savefig(fig / "fig10_year.png")
        plt.close(f)
    hist([x for x in text_len if x <= 500], "review text length, clipped at 500", fig / "fig11_text_len.png", 40, "chars")
    print(f"reviews done: scanned={n_all:,} target_hits={n_hit:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1.1 FINAL EDA for Amazon headphone target category")
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--reviews", default=str(DEFAULT_REVIEWS))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip-reviews", action="store_true")
    args = ap.parse_args()

    setup_font()
    root = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    base = root / "stage1_eda"
    out = base / "output"
    fig = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    meta_path = resolve_data_file(Path(args.meta))
    print("== meta ==", meta_path)
    target_ids = eda_meta(meta_path, out, fig)

    if not args.skip_reviews:
        reviews_path = resolve_data_file(Path(args.reviews))
        print("== reviews ==", reviews_path)
        eda_reviews(reviews_path, target_ids, out, fig)
    else:
        print("skip reviews")

    print("wrote", base)


if __name__ == "__main__":
    main()
