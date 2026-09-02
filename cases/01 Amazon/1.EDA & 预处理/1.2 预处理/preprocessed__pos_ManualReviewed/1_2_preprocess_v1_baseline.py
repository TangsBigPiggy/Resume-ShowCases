#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 1.2 preprocessing V1 BASELINE — historical first-pass screening

This file is intentionally preserved to show the project's one-iteration workflow:
    first rule screening -> manual audit sample -> reviewed Gold -> optimized FINAL.

It is NOT the production preprocessing script. The production script is:
    1_2_preprocess_final.py

Current raw-data layout supported by default:
    E:\DA Cases\Amazon\0.原始数据\meta_Electronics.jsonl\meta_Electronics.jsonl
    E:\DA Cases\Amazon\0.原始数据\Electronics.jsonl\Electronics.jsonl

V1 output:
    v1_baseline_output/product_eligibility_v1.csv
    v1_baseline_output/manual_audit_sample.csv
    v1_baseline_output/baseline_summary.csv
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, random, re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Optional

TARGET_L2 = "Headphones, Earbuds & Accessories"
DEFAULT_DATA_DIR = Path(r"E:\DA Cases\Amazon\0.原始数据")
DEFAULT_META = DEFAULT_DATA_DIR / "meta_Electronics.jsonl"
SEED = 20260831

DEVICE, ACCESSORY, AMBIGUOUS = "DEVICE", "ACCESSORY", "AMBIGUOUS"

ACCESSORY_LEAF = {
    "cases", "earpads", "adapters", "extension cords", "cables", "ear tips",
    "replacement parts", "stands", "chargers", "headphone amplifiers", "amplifiers",
}
DEVICE_LEAF = {"earbud headphones", "over-ear headphones", "on-ear headphones", "in-ear headphones"}

ACC_PATTERNS = [re.compile(x, re.I) for x in [
    r"\bcase\s+for\b", r"\bcover\s+for\b", r"\breplacement\s+(?:ear\s*)?(?:pads?|cushions?|tips?)\b",
    r"\b(?:ear\s*)?(?:pads?|cushions?|tips?)\s+for\b", r"\badapter\s+for\b", r"\bcable\s+for\b",
    r"\bextension\s+(?:cable|cord)\b", r"\bcharging\s+case\b", r"\bheadphone\s+stand\b",
    r"\bstorage\s+(?:case|bag)\b", r"\bcarrying\s+(?:case|bag)\b", r"\bcompatible\s+with\b",
]]
DEV_PATTERNS = [re.compile(x, re.I) for x in [
    r"\bwireless\s+earbuds?\b", r"\bbluetooth\s+(?:headphones?|earbuds?|headset)\b",
    r"\bnoise\s+cancell?ing\s+(?:headphones?|earbuds?|headset)\b", r"\bactive\s+noise\s+cancell?ing\b",
    r"\btrue\s+wireless\b", r"\bover[-\s]?ear\s+headphones?\b", r"\bon[-\s]?ear\s+headphones?\b",
    r"\bin[-\s]?ear\s+(?:headphones?|earbuds?)\b", r"\bgaming\s+headset\b",
]]

@dataclass
class Row:
    product_id: str
    label: str
    confidence: float
    form_factor: str
    taxonomy_device_evidence: int
    taxonomy_accessory_evidence: int
    title_device_evidence: int
    title_accessory_evidence: int
    conflict: int
    reason: str
    title: str
    store: str
    leaf_categories: str
    l3_categories: str
    rating_number: Optional[int]


def _is_jsonl_file(p: Path) -> bool:
    return p.is_file() and p.name.lower().endswith((".jsonl", ".jsonl.gz", ".json.gz"))


def resolve_data_file(path: Path) -> Path:
    path = Path(path)
    variants = [path, Path(str(path)[:-3]) if str(path).lower().endswith('.gz') else Path(str(path)+'.gz')]
    for p in variants:
        if _is_jsonl_file(p): return p
    if path.is_dir():
        exact = [p for p in path.iterdir() if _is_jsonl_file(p) and p.name in {path.name, path.name+'.gz'}]
        if exact: return max(exact, key=lambda x:x.stat().st_size)
        hits = [p for p in path.iterdir() if _is_jsonl_file(p)]
        if hits: return max(hits, key=lambda x:x.stat().st_size)
    raise FileNotFoundError(f"未找到可读取 JSONL 文件: {path}")


def iter_jsonl(path: Path) -> Iterator[dict]:
    op = gzip.open if str(path).lower().endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: x=json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(x, dict): yield x


def norm(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or '')).strip()


def nid(v: Any) -> Optional[str]:
    s=norm(v); return s or None


def to_int(v: Any) -> Optional[int]:
    try:
        x=float(v)
        return int(x) if math.isfinite(x) else None
    except (TypeError,ValueError): return None


def flatten(cats: Any) -> list[list[str]]:
    if not isinstance(cats,list): return []
    if cats and all(isinstance(x,str) for x in cats):
        p=[norm(x) for x in cats if norm(x)]; return [p] if p else []
    out=[]
    for e in cats:
        if isinstance(e,list):
            p=[norm(x) for x in e if norm(x)]
            if p: out.append(p)
    return out


def infer_form(title: str, leaves: set[str]) -> str:
    t=(title+' '+' '.join(leaves)).lower()
    if re.search(r"\b(over[-\s]?ear|circumaural)\b",t): return 'OVER_EAR'
    if re.search(r"\b(on[-\s]?ear|supra[-\s]?aural)\b",t): return 'ON_EAR'
    if re.search(r"\b(earbuds?|in[-\s]?ear|true wireless)\b",t): return 'EARBUD_INEAR'
    if re.search(r"\bheadset\b",t): return 'HEADSET'
    return 'UNKNOWN'


def classify(rec: dict) -> Optional[Row]:
    paths=[p for p in flatten(rec.get('categories')) if TARGET_L2 in p]
    if not paths: return None
    pid=nid(rec.get('parent_asin')) or nid(rec.get('asin'))
    if not pid: return None
    title=norm(rec.get('title')); store=norm(rec.get('store'))
    leaves={p[-1] for p in paths if p}; l3s={p[2] for p in paths if len(p)>=3}
    td=ta=0
    for leaf in leaves:
        low=leaf.lower()
        if low in ACCESSORY_LEAF: ta += 2
        if low in DEVICE_LEAF: td += 2
    xd=sum(bool(r.search(title)) for r in DEV_PATTERNS)
    xa=sum(bool(r.search(title)) for r in ACC_PATTERNS)
    ds=2*td+2*xd; aps=2*ta+3*xa
    conflict=int(ds>0 and aps>0)
    if aps>=4 and aps>=ds+2:
        lab=ACCESSORY; conf=min(.995,.90+.02*(aps-ds))
    elif ds>=4 and aps==0:
        lab=DEVICE; conf=min(.995,.92+.015*ds)
    elif ds>=6 and ds>=aps+4:
        lab=DEVICE; conf=min(.985,.88+.015*(ds-aps))
    elif aps>=6 and aps>=ds+3:
        lab=ACCESSORY; conf=min(.99,.89+.015*(aps-ds))
    else:
        lab=AMBIGUOUS; conf=.50 if ds==aps else .60
    reason=f"taxonomy_device={td};taxonomy_accessory={ta};title_device={xd};title_accessory={xa}" + (';conflict' if conflict else '')
    return Row(pid,lab,round(conf,3),infer_form(title,leaves) if lab==DEVICE else 'NOT_APPLICABLE',td,ta,xd,xa,conflict,reason,title[:500],store[:160],' | '.join(sorted(leaves)),' | '.join(sorted(l3s)),to_int(rec.get('rating_number')))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description='Historical V1 baseline screening')
    ap.add_argument('--meta',default=str(DEFAULT_META)); ap.add_argument('--outdir',default=None)
    ap.add_argument('--audit-n',type=int,default=2500); ap.add_argument('--seed',type=int,default=SEED)
    args=ap.parse_args(); out=(Path(args.outdir) if args.outdir else Path(__file__).resolve().parent/'v1_baseline_output')
    products={}
    for rec in iter_jsonl(resolve_data_file(Path(args.meta))):
        d=classify(rec)
        if d and d.product_id not in products: products[d.product_id]=d
    vals=list(products.values()); fields=list(Row.__dataclass_fields__)
    write_csv(out/'product_eligibility_v1.csv',[asdict(x) for x in vals],fields)
    counts=Counter(x.label for x in vals)
    write_csv(out/'baseline_summary.csv',[{'label':k,'n':counts[k]} for k in (DEVICE,ACCESSORY,AMBIGUOUS)],['label','n'])
    rng=random.Random(args.seed)
    amb=[x for x in vals if x.label==AMBIGUOUS]; con=[x for x in vals if x.conflict]
    top=sorted(vals,key=lambda x:x.rating_number or 0,reverse=True)
    chosen={}
    def add(xs,k):
        xs=[x for x in xs if x.product_id not in chosen]
        if len(xs)>k: xs=rng.sample(xs,k)
        for x in xs: chosen[x.product_id]=x
    add(amb,int(args.audit_n*.35)); add(con,int(args.audit_n*.20)); add(top,int(args.audit_n*.15))
    add([x for x in vals if x.label==DEVICE],int(args.audit_n*.15)); add([x for x in vals if x.label==ACCESSORY],int(args.audit_n*.15))
    add(vals,max(0,args.audit_n-len(chosen)))
    sample=list(chosen.values())[:args.audit_n]; rng.shuffle(sample)
    rows=[]
    for x in sample:
        d=asdict(x); d.update(gold_label='',gold_form_factor='',reviewer_note=''); rows.append(d)
    write_csv(out/'manual_audit_sample.csv',rows,fields+['gold_label','gold_form_factor','reviewer_note'])
    print(f'V1 baseline products={len(vals):,}; audit sample={len(rows):,}; output={out}')

if __name__=='__main__': main()
