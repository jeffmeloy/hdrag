"""
hdrag_test.py - Behavioral analysis suite for HdRAG.

Probes HDC encoding and retrieval properties by indexing synthetic
documents and analyzing scoring behavior.  No inference server needed -
uses HuggingFaceTokenizer directly from GGUF.

Each probe produces a detailed diagnostic report with:
  - Per-document and per-group scoring breakdowns
  - ASCII visualizations (bars, scatter, sparklines, distributions)
  - Computed statistics (correlations, effect sizes, separation ratios)
  - Information-theoretic measures (entropy, mutual information, capacity)

Probes:
  1. ngram         - compositional sensitivity: score ramp across n-gram levels
  2. idf           - frequency discrimination: score gradient across relevance tiers
  3. sparse        - normalization: geometric vs engaged on sparse documents
  4. specificity   - transfer function: lexical overlap → vector correlation
  5. noise         - signal isolation: separation from random content
  6. duplicate     - sensitivity profile: Lipschitz structure under perturbation
  7. topic         - selectivity: cross-topic discrimination matrix
  8. density       - entropy rate: information capacity vs document length
  9. sensitivity   - resolution: query vector stability under incremental changes
 10. retrieval     - end-to-end: ranking quality on mixed corpus

Usage:
  python hdrag_test.py --config hdrag_config.yaml
  python hdrag_test.py --config hdrag_config.yaml --probe ngram
  python hdrag_test.py --config hdrag_config.yaml --probe idf,noise
  python hdrag_test.py --config hdrag_config.yaml --brief
  python hdrag_test.py --config hdrag_config.yaml --json results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
import time
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hdrag_model import Config, HuggingFaceTokenizer, resolve_gguf
from hdrag import HdRAG, score64, strip_structural_keys


# Formatting & visualization helpers

W = 120
BAR = 40

_PC8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
def popcount64(x: np.ndarray) -> np.ndarray:
    return _PC8[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1)


def hbar(val, lo, hi, width=BAR):
    span = hi - lo
    if span <= 0:
        return " " * width
    if lo >= 0:
        fill = int(round((val - lo) / span * width))
        return ("\u2588" * max(fill, 0)).ljust(width)
    zero = int(round(-lo / span * width))
    if val >= 0:
        fill = int(round(val / span * width))
        return (" " * zero + "\u2588" * fill).ljust(width)
    else:
        fill = int(round(-val / span * width))
        start = max(0, zero - fill)
        return (" " * start + "\u2591" * fill).ljust(width)


def hbar_abs(val, hi, width=BAR):
    if hi <= 0:
        return " " * width
    fill = int(round(abs(val) / hi * width))
    ch = "\u2588" if val >= 0 else "\u2591"
    return (ch * max(fill, 0)).ljust(width)


def sparkline(vals, width=20):
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    span = hi - lo if hi > lo else 1.0
    step = max(1, len(vals) // width) if len(vals) > width else 1
    sampled = [vals[i] for i in range(0, len(vals), step)][:width]
    return "".join(blocks[min(8, int((v - lo) / span * 8))] for v in sampled)


def ascii_scatter(xs, ys, labels, width=60, height=16, x_label=""):
    if not xs:
        return ""
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    grid = [[" "] * width for _ in range(height)]
    pm = {}
    for x, y, label in zip(xs, ys, labels):
        col = min(width - 1, int((x - x_min) / x_span * (width - 1)))
        row = min(height - 1, int((y_max - y) / y_span * (height - 1)))
        grid[row][col] = "\u25cf"
        pm[(row, col)] = label
    lines = []
    for r, row in enumerate(grid):
        y_val = y_max - r * y_span / (height - 1) if height > 1 else y_max
        line = f"  {y_val:+8.4f} \u2502" + "".join(row)
        for c in range(width - 1, -1, -1):
            if (r, c) in pm:
                line += f"  \u2190 {pm[(r, c)]}"
                break
        lines.append(line)
    lines.append(f"  {'':>8s} \u2514{'\u2500' * width}")
    x_lo, x_hi = f"{x_min:.2f}", f"{x_max:.2f}"
    lines.append(f"  {'':>8s}  {x_lo}{' ' * (width - len(x_lo) - len(x_hi))}{x_hi}")
    if x_label:
        lines.append(f"  {'':>8s}  {x_label:^{width}s}")
    return "\n".join(lines)


def sep(ch="\u2500", n=W):
    return ch * n


def banner(title):
    return f"\n{'\u2550' * W}\n  {title}\n{'\u2550' * W}\n"


def section(title):
    return f"\n  {'\u2501' * (W - 4)}\n  {title}\n  {'\u2501' * (W - 4)}\n"


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "\u2014"


# Statistical helpers


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        diff = float(a.mean() - b.mean())
        return float("inf") * (1 if diff > 0 else -1) if diff != 0 else 0.0
    pooled = math.sqrt(
        ((na - 1) * float(a.var()) + (nb - 1) * float(b.var())) / (na + nb - 2)
    )
    return float(a.mean() - b.mean()) / pooled if pooled > 0 else float("inf")


def spearman_rho(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    d = rx - ry
    return float(1.0 - 6.0 * (d**2).sum() / (n * (n**2 - 1)))


def pearson_r(xs, ys):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def separation_ratio(hi, lo):
    gap = float(hi.mean() - lo.mean())
    noise = max(float(hi.std()), float(lo.std()), 1e-9)
    return gap / noise


def mad(x):
    """Median absolute deviation (scaled to match σ for normal data)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def robust_d(a, b):
    """Robust effect size: (median_a − median_b) / pooled MAD."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(np.median(a) - np.median(b))
    pooled = max(mad(a), mad(b), 1e-9)
    return diff / pooled


def robust_separation(hi, lo):
    """Robust margin: median_hi − median_lo, normalised by pooled MAD."""
    hi, lo = np.asarray(hi, dtype=float), np.asarray(lo, dtype=float)
    gap = float(np.median(hi) - np.median(lo))
    pooled = max(mad(hi), mad(lo), 1e-9)
    return gap / pooled


def l_moments(x, max_order=4):
    """Probability-weighted L-moments up to order *max_order*.

    Returns dict with keys l1..l4 (raw L-moments) and t3, t4
    (L-skewness, L-kurtosis ratios).  All values are NaN when the
    sample is too small.

    L-moments are linear combinations of order statistics.  They
    characterise location (l1 ≈ mean), scale (l2 ≈ half the Gini
    mean difference), shape (t3 ∈ [−1,+1], t4 ∈ [−¼,+1]) and are
    far more resistant to outliers than classical moments.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    nan = float("nan")
    out = {f"l{k}": nan for k in range(1, max_order + 1)}
    out["t3"] = out["t4"] = nan

    if n < max_order:
        return out

    # Probability-weighted moments  β_r = (1/n) Σ C(i,r)/C(n-1,r) · x_(i+1)
    # where i runs 0..n-1 and x is 0-indexed ascending.
    idx = np.arange(n, dtype=float)
    b = np.zeros(max_order)
    b[0] = x.mean()
    for r in range(1, max_order):
        # C(i,r) / C(n-1,r) computed via running product for stability
        w = np.ones(n)
        for j in range(r):
            w *= (idx - j) / (n - 1 - j)
        b[r] = (w * x).mean()

    out["l1"] = b[0]
    out["l2"] = 2 * b[1] - b[0]
    if max_order >= 3:
        out["l3"] = 6 * b[2] - 6 * b[1] + b[0]
    if max_order >= 4:
        out["l4"] = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]

    if abs(out["l2"]) > 1e-12:
        out["t3"] = out["l3"] / out["l2"]
        out["t4"] = out["l4"] / out["l2"]

    return out


def fmt_lm(lm, compact=False):
    """Format L-moment dict for display."""
    if compact:
        return f"τ₃={lm['t3']:+.3f} τ₄={lm['t4']:+.3f}"
    return (
        f"λ₁={lm['l1']:+.4f}  λ₂={lm['l2']:.4f}  τ₃={lm['t3']:+.4f}  τ₄={lm['t4']:+.4f}"
    )


def effect_label(d):
    ad = abs(d)
    if ad >= 2.0:
        return "huge"
    if ad >= 0.8:
        return "large"
    if ad >= 0.5:
        return "medium"
    if ad >= 0.2:
        return "small"
    return "negligible"


# Infrastructure


@dataclass
class ProbeResult:
    name: str
    ok: bool
    primary_stat: str
    summary: str = ""
    details: str = ""
    metrics: dict = field(default_factory=dict)
    duration: float = 0.0
    warnings: list = field(default_factory=list)


def _json_safe(v):
    """Recursively convert metric values to JSON-safe types."""
    if isinstance(v, (np.floating, np.integer)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(dk): _json_safe(dv) for dk, dv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


class Harness:
    def __init__(self, config, tokenizer, logger):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger

    def index(self, docs):
        tmp = Path(tempfile.mkdtemp(prefix="hdrag_probe_"))
        ds = tmp / "datasets"
        ds.mkdir()
        hd = tmp / "hdrag_data"
        hd.mkdir()
        with open(ds / "test.jsonl", "w") as f:
            for i, d in enumerate(docs):
                f.write(
                    json.dumps(
                        {"instruction": d.get("label", f"doc_{i}"), "output": d["text"]}
                    )
                    + "\n"
                )
        orig_ds, orig_hd = self.config.datasets_dir, self.config.hdrag_dir
        self.config.datasets_dir, self.config.hdrag_dir = str(ds), str(hd)
        engine = HdRAG(
            self.config,
            tokenizer=self.tokenizer,
            logger=self.logger,
        )
        engine.build_index()
        self.config.datasets_dir, self.config.hdrag_dir = orig_ds, orig_hd
        engine._tmp = tmp
        return engine

    def encode(self, engine, query):
        # Same preprocessing as Retriever.search
        query = query.rstrip().rstrip("?!.,;:\u2026\n\r\t").rstrip()
        query = strip_structural_keys(query)
        tids = engine._tokenizer.bulk_tokenize([query])
        qe = engine.hdc.project(token_ids=tids)
        return engine.hdc.encode(unigrams=qe, token_ids=tids)

    def scores(self, engine, q_pos, q_neg, q_lit_uni=None, q_lit_ngram=None):
        corpus = engine.db._open_corpus()
        if not corpus:
            e = np.array([], np.float32)
            return e, e, e, e, e, e
        _, _, pos64, neg64, lit = corpus
        qp = q_pos.ravel().view(np.uint64)[None, :]
        qn = q_neg.ravel().view(np.uint64)[None, :]
        qa = qp | qn
        agree = popcount64((pos64 & qp) | (neg64 & qn)).sum(1).astype(np.float32)
        disagree = popcount64((pos64 & qn) | (neg64 & qp)).sum(1).astype(np.float32)
        raw = agree - disagree
        engaged = popcount64((pos64 | neg64) & qa).sum(1).astype(np.float32).clip(1.0)
        q_lit = float(popcount64(qa).sum())

        # Region-separated scoring when encoder provides partition info
        uni_mask = getattr(engine.hdc, "uni_mask_u64", None)
        ngram_mask = getattr(engine.hdc, "ngram_mask_u64", None)
        has_adaptive = hasattr(engine.hdc, "adaptive_alpha")

        if uni_mask is not None and ngram_mask is not None and has_adaptive:
            # Adaptive alpha: scales n-gram weight by query confidence
            q_du = q_lit_uni if q_lit_uni is not None else q_lit
            q_dn = q_lit_ngram if q_lit_ngram is not None else 0.0
            alpha = engine.hdc.adaptive_alpha(q_du, q_dn)

            # Unigram region
            u_agree = (
                popcount64((pos64 & qp & uni_mask) | (neg64 & qn & uni_mask))
                .sum(1)
                .astype(np.float32)
            )
            u_disagree = (
                popcount64((pos64 & qn & uni_mask) | (neg64 & qp & uni_mask))
                .sum(1)
                .astype(np.float32)
            )
            doc_lit_uni = engine.db._get_lit_uni()
            if len(doc_lit_uni) == 0 or doc_lit_uni.sum() == 0:
                doc_lit_uni = lit
            q_du = q_lit_uni if q_lit_uni is not None else q_lit
            uni_score = (u_agree - u_disagree) / np.sqrt(doc_lit_uni * q_du).clip(1.0)

            # N-gram region
            n_agree = (
                popcount64((pos64 & qp & ngram_mask) | (neg64 & qn & ngram_mask))
                .sum(1)
                .astype(np.float32)
            )
            n_disagree = (
                popcount64((pos64 & qn & ngram_mask) | (neg64 & qp & ngram_mask))
                .sum(1)
                .astype(np.float32)
            )
            doc_lit_ngram = engine.db._get_lit_ngram()
            if len(doc_lit_ngram) == 0 or doc_lit_ngram.sum() == 0:
                # Compute from corpus bitmaps (no sidecar available)
                doc_lit_ngram = (
                    popcount64((pos64 | neg64) & ngram_mask).sum(1).astype(np.float32)
                )
            q_dn = q_lit_ngram if q_lit_ngram is not None else 0.0
            ng_denom = np.sqrt(doc_lit_ngram * q_dn).clip(1.0)
            ngram_score = (n_agree - n_disagree) / ng_denom

            geo = alpha * uni_score + (1.0 - alpha) * ngram_score
        else:
            # Fallback: single-pool normalization
            doc_denom = engine.db._get_lit_uni()
            if len(doc_denom) == 0 or doc_denom.sum() == 0:
                doc_denom = lit
            q_denom = q_lit_uni if q_lit_uni is not None else q_lit
            geo = raw / np.sqrt(doc_denom * q_denom).clip(1.0)

        return (
            raw / engaged,
            geo,
            agree,
            disagree,
            engaged,
            lit,
        )

    def qbits(self, q_pos, q_neg):
        qp = q_pos.ravel().view(np.uint64)
        qn = q_neg.ravel().view(np.uint64)
        qa = qp | qn
        return (
            int(popcount64(qp[None, :]).sum()),
            int(popcount64(qn[None, :]).sum()),
            int(popcount64(qa[None, :]).sum()),
        )

    def doc_bits(self, engine, n):
        corpus = engine.db._open_corpus()
        if not corpus:
            return []
        _, _, p64, n64, lit = corpus
        dims = self.config.hdc_dimensions
        out = []
        for i in range(n):
            p = int(popcount64(p64[i : i + 1]).sum())
            ng = int(popcount64(n64[i : i + 1]).sum())
            a = int(lit[i])
            out.append({"pos": p, "neg": ng, "active": a, "density": a / dims})
        return out

    def ngram_retention(self, a, b, max_n):
        ia = self.tokenizer.bulk_tokenize([a])[0]
        ib = self.tokenizer.bulk_tokenize([b])[0]
        total = shared = 0
        for n in range(1, max_n + 1):
            ga = {tuple(ia[i : i + n]) for i in range(len(ia) - n + 1)}
            gb = {tuple(ib[i : i + n]) for i in range(len(ib) - n + 1)}
            total += len(ga)
            shared += len(ga & gb)
        return shared / total if total else 1.0

    def hamming(self, bm_a, bm_b, dims):
        ap, an = (
            bm_a["pos"].ravel().view(np.uint64),
            bm_a["neg"].ravel().view(np.uint64),
        )
        bp, bn = (
            bm_b["pos"].ravel().view(np.uint64),
            bm_b["neg"].ravel().view(np.uint64),
        )
        dp = int(popcount64((ap ^ bp)[None, :]).sum())
        dn = int(popcount64((an ^ bn)[None, :]).sum())
        return (dp + dn) / (2 * dims)

    def cleanup(self, engine):
        engine.close()
        tmp = getattr(engine, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# Probes


def probe_ngram(h):
    """Score response to increasing n-gram overlap with the query.
    Primary: Spearman rho (token n-gram level -> mean score)

    Documents are constructed from token spans of the prompt — each document
    contains exactly nt consecutive tokens, matching what encode() sees when
    building its sliding-window n-grams.
    """
    t0 = time.time()
    prompt = "What is a geometric shape that uses complex numbers to describe its convex curvatures"
    max_n = h.config.hdc_ngram
    dims = h.config.hdc_dimensions

    # --- tokenize prompt into subword tokens ---
    prompt_tids = h.tokenizer.bulk_tokenize([prompt])[0]
    n_tok = len(prompt_tids)

    # --- build documents from token spans ---
    docs = []
    roundtrip_mismatches = 0
    for nt in range(1, min(max_n + 1, n_tok + 1)):
        for i in range(n_tok - nt + 1):
            span_ids = prompt_tids[i : i + nt]
            text = h.tokenizer.detokenize(list(span_ids))
            # verify round-trip: re-tokenizing should yield same token count
            actual_nt = h.tokenizer.count_tokens(text)
            if actual_nt != nt:
                roundtrip_mismatches += 1
            docs.append(
                {
                    "text": text,
                    "label": f"t{nt}.{i + 1}",
                    "nt": nt,  # intended token span length
                    "actual_nt": actual_nt,  # round-trip token count
                    "roundtrip_ok": actual_nt == nt,
                }
            )

    engine = h.index(docs)
    bm = h.encode(engine, prompt)
    _, s_geo, agree, disagree, engaged, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )
    qp, qn, qa = h.qbits(bm["pos"], bm["neg"])

    lo, hi = float(s_geo.min()), float(s_geo.max())

    # --- group by token span length (= encoder n-gram level) ---
    levels = {}
    for nt in range(1, max_n + 1):
        idx = [i for i, d in enumerate(docs) if d["nt"] == nt]
        if not idx:
            continue
        sg = s_geo[idx]
        lm = l_moments(sg)
        levels[nt] = {
            "mean": float(sg.mean()),
            "std": float(sg.std()),
            "median": float(np.median(sg)),
            "mad": mad(sg),
            "max": float(sg.max()),
            "min": float(sg.min()),
            "n": len(idx),
            "idx": idx,
            "lm": lm,
        }

    ns = sorted(levels.keys())
    means = [levels[n]["mean"] for n in ns]
    medians = [levels[n]["median"] for n in ns]
    rho = spearman_rho(ns, means)
    rho_med = spearman_rho(ns, medians)
    mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    mono_med = all(medians[i] <= medians[i + 1] for i in range(len(medians) - 1))

    # --- per-document correlation ---
    all_nt = [d["nt"] for d in docs]
    all_scores = [float(s_geo[i]) for i in range(len(docs))]
    rho_per_doc = spearman_rho(all_nt, all_scores)

    O = io.StringIO()
    w = O.write
    w(banner("N-GRAM RETRIEVAL ANALYSIS"))
    w(f'\n  PROMPT:      "{prompt}"\n')
    w(f"  Tokens:      {n_tok}\n")
    w(f"  HDC dims:    {dims}\n")
    w(f"  N-gram:      {max_n}\n")
    w(f"  Documents:   {len(docs)}\n")
    if roundtrip_mismatches:
        w(
            f"  ⚠ Round-trip mismatches: {roundtrip_mismatches}/{len(docs)} "
            f"(detokenize→retokenize changed token count)\n"
        )

    w(f"\n  TOKEN MAP\n  {sep(n=60)}\n")
    for i, tid in enumerate(prompt_tids):
        piece = h.tokenizer.detokenize([int(tid)])
        w(f"    [{i:2d}] id={int(tid):6d}  {repr(piece)}\n")

    for n in ns:
        w(f"    {n}-token spans:  {levels[n]['n']}\n")
    w(f"\n  QUERY VECTOR\n  {sep(n=50)}\n")
    w(f"  Active dims:    {qa:5d}  ({pct(qa, dims)})\n")
    w(f"  Positive bits:  {qp:5d}  ({pct(qp, dims)})\n")
    w(f"  Negative bits:  {qn:5d}  ({pct(qn, dims)})\n")
    w(f"  Zero dims:      {dims - qa:5d}  ({pct(dims - qa, dims)})\n")

    for n in ns:
        st = levels[n]
        tag = f"{n}-TOKEN"
        w(
            section(
                f"{tag}    avg={st['mean']:+.4f}  max={st['max']:+.4f}  min={st['min']:+.4f}  std={st['std']:.4f}  n={st['n']}"
            )
        )
        w(
            f"    {'label':>7s}  {'score':>8s}  {'agree':>6s}  {'disag':>6s}  {'engag':>6s}  {'d_lit':>6s}  {'rt':>2s}  bar{' ' * (BAR - 3)}  span\n"
        )
        for i in st["idx"]:
            d = docs[i]
            rt = "ok" if d["roundtrip_ok"] else "!!"
            w(
                f"    {d['label']:>7s}  {s_geo[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {engaged[i]:6.0f}  {lit[i]:6.0f}  {rt:>2s}  {hbar(s_geo[i], lo, hi)}  {repr(d['text'])}\n"
            )

    w(banner("SUMMARY: Mean score by token n-gram level"))
    w(
        f"\n    {'nt':>3s}  {'n_spans':>7s}  {'mean':>9s}  {'median':>9s}  {'std':>9s}  {'mad':>9s}  {'max':>9s}  {'min':>9s}  bar\n"
    )
    w(f"    {sep(n=95)}\n")
    for n in ns:
        st = levels[n]
        w(
            f"    {n:3d}  {st['n']:7d}  {st['mean']:+9.4f}  {st['median']:+9.4f}  {st['std']:9.4f}  {st['mad']:9.4f}  {st['max']:+9.4f}  {st['min']:+9.4f}  {hbar(st['mean'], lo, hi)}\n"
        )

    w(f"\n  L-MOMENT CHARACTERIZATION PER LEVEL\n  {sep(n=80)}\n")
    w(
        f"    {'nt':>3s}  {'λ₁(loc)':>10s}  {'λ₂(scale)':>10s}  {'τ₃(skew)':>10s}  {'τ₄(kurt)':>10s}  shape\n"
    )
    w(f"    {sep(n=70)}\n")
    for n in ns:
        lm = levels[n]["lm"]
        shape = ""
        if not math.isnan(lm["t3"]):
            if lm["t3"] > 0.15:
                shape = "right-skewed"
            elif lm["t3"] < -0.15:
                shape = "left-skewed"
            else:
                shape = "symmetric"
            if not math.isnan(lm["t4"]):
                if lm["t4"] > 0.20:
                    shape += ", heavy-tailed"
                elif lm["t4"] < 0.05:
                    shape += ", light-tailed"
        w(
            f"    {n:3d}  {lm['l1']:+10.4f}  {lm['l2']:10.4f}  {lm['t3']:+10.4f}  {lm['t4']:+10.4f}  {shape}\n"
        )

    w(f"\n  Score increases with token n-gram level: {'YES' if mono else 'NO'}")
    w(f"  (median: {'YES' if mono_med else 'NO'})\n")

    w(f"\n  LEVEL DELTAS\n  {sep(n=50)}\n")
    for i in range(len(means) - 1):
        d = means[i + 1] - means[i]
        rel = 100 * d / abs(means[i]) if abs(means[i]) > 1e-6 else 0
        w(f"  {ns[i]}-token -> {ns[i + 1]}-token:  {d:+.4f}  ({rel:+.1f}%)\n")

    w(f"\n  WITHIN-LEVEL VARIANCE\n  {sep(n=60)}\n")
    for n in ns:
        st = levels[n]
        spread = st["max"] - st["min"]
        cv = st["std"] / abs(st["mean"]) if abs(st["mean"]) > 1e-6 else 0
        cv_mad = st["mad"] / abs(st["median"]) if abs(st["median"]) > 1e-6 else 0
        w(f"  {n}-token:  spread={spread:.4f}  cv={cv:.2f}  cv_mad={cv_mad:.2f}\n")

    w(f"\n  SCORE vs TOKEN N-GRAM LEVEL (all {len(docs)} documents)\n")
    w(
        ascii_scatter(
            xs=all_nt,
            ys=all_scores,
            labels=[d["label"] for d in docs],
            x_label="token n-gram level",
        )
    )
    w(f"\n\n  Spearman rho (level -> mean score):   {rho:+.4f}\n")
    w(f"  Spearman rho (level -> median score): {rho_med:+.4f}\n")
    w(f"  Spearman rho (per-doc):               {rho_per_doc:+.4f}\n")
    w(
        f"  Monotonic (mean): {'yes' if mono else 'no'}  (median): {'yes' if mono_med else 'no'}\n"
    )

    h.cleanup(engine)
    return ProbeResult(
        name="ngram",
        ok=rho > 0.9,
        primary_stat=f"rho={rho:+.4f}",
        summary=f"rho={rho:+.4f}  rho_med={rho_med:+.4f}  rho_doc={rho_per_doc:+.4f}  mono={'yes' if mono else 'no'}  "
        + "  ".join(f"{n}t={levels[n]['mean']:+.4f}" for n in ns),
        details=O.getvalue(),
        metrics={
            "rho": rho,
            "rho_median": rho_med,
            "rho_per_doc": rho_per_doc,
            "monotonic": mono,
            "monotonic_median": mono_med,
            "level_means": {n: levels[n]["mean"] for n in levels},
            "level_medians": {n: levels[n]["median"] for n in levels},
            "level_lmoments": {n: levels[n]["lm"] for n in levels},
            "n_prompt_tokens": n_tok,
            "roundtrip_mismatches": roundtrip_mismatches,
        },
        duration=time.time() - t0,
    )


def probe_idf(h):
    """Score gradient across term-relevance tiers.  Primary: Cohen's d (target vs filler)"""
    t0 = time.time()
    query = "Kubernetes pod autoscaling"
    dims = h.config.hdc_dimensions
    docs = [
        {
            "text": "Kubernetes pod autoscaling allows clusters to dynamically adjust the number of pod replicas based on CPU utilization and custom metrics",
            "label": "exact_match",
            "group": "target",
        },
        {
            "text": "Horizontal pod autoscaler in Kubernetes watches resource metrics and scales deployments automatically",
            "label": "near_match",
            "group": "target",
        },
        {
            "text": "Container orchestration systems manage deployment and scaling of application containers across clusters of machines",
            "label": "related",
            "group": "related",
        },
        {
            "text": "Cloud computing infrastructure provides on-demand access to shared pools of configurable computing resources",
            "label": "cloud",
            "group": "related",
        },
        {
            "text": "Computer systems use various methods to allocate resources efficiently across running processes",
            "label": "vague",
            "group": "filler",
        },
        {
            "text": "The weather today is sunny with a high of 75 degrees and low humidity expected throughout the afternoon",
            "label": "unrelated",
            "group": "filler",
        },
    ]
    engine = h.index(docs)
    bm = h.encode(engine, query)
    s_eng, s_geo, agree, disagree, engaged, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )
    qp, qn, qa = h.qbits(bm["pos"], bm["neg"])
    groups = {}
    for g in ("target", "related", "filler"):
        idx = [i for i, d in enumerate(docs) if d["group"] == g]
        geo = s_geo[idx]
        groups[g] = {"idx": idx, "geo": geo, "lm": l_moments(geo)}
    d_val = cohen_d(groups["target"]["geo"], groups["filler"]["geo"])
    rd_val = robust_d(groups["target"]["geo"], groups["filler"]["geo"])
    lo, hi = float(s_geo.min()), float(s_geo.max())
    O = io.StringIO()
    w = O.write
    w(banner("IDF DISCRIMINATION ANALYSIS"))
    w(f'\n  QUERY: "{query}"     Corpus: {len(docs)} documents\n')
    w(f"  QUERY VECTOR: active={qa} ({pct(qa, dims)})\n")
    w(section("RANKED SCORES"))
    w(
        f"  {'label':>15s}  {'group':>8s}  {'geo':>8s}  {'eng':>8s}  {'agree':>6s}  {'disag':>6s}  {'d_lit':>6s}  bar\n"
    )
    w(f"  {sep(n=110)}\n")
    for i in sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True):
        d = docs[i]
        w(
            f"  {d['label']:>15s}  {d['group']:>8s}  {s_geo[i]:+8.4f}  {s_eng[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {lit[i]:6.0f}  {hbar(s_geo[i], lo, hi)}\n"
        )
    w(section("GROUP DISTRIBUTIONS"))
    w(
        f"  {'group':>10s}  {'n':>3s}  {'mean':>8s}  {'median':>8s}  {'std':>8s}  {'mad':>8s}  {'min':>8s}  {'max':>8s}\n"
    )
    w(f"  {sep(n=80)}\n")
    for g in ("target", "related", "filler"):
        sc = groups[g]["geo"]
        w(
            f"  {g:>10s}  {len(sc):3d}  {float(sc.mean()):+8.4f}  {float(np.median(sc)):+8.4f}  {float(sc.std()):8.4f}  {mad(sc):8.4f}  {float(sc.min()):+8.4f}  {float(sc.max()):+8.4f}\n"
        )
    w(f"\n  L-MOMENTS PER GROUP\n  {sep(n=70)}\n")
    for g in ("target", "related", "filler"):
        lm = groups[g]["lm"]
        w(f"  {g:>10s}:  {fmt_lm(lm)}\n")
    labels = [d["label"] for d in docs]
    ranked = sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True)
    w(f"\n  PAIRWISE GAPS\n  {sep(n=60)}\n")
    for j in range(len(ranked) - 1):
        a, b = ranked[j], ranked[j + 1]
        gap = float(s_geo[a] - s_geo[b])
        w(
            f"  {labels[a]:>15s} -> {labels[b]:<15s}  gap={gap:+.4f}  {hbar_abs(gap, hi - lo, 20)}\n"
        )
    w(f"\n  Cohen's d (target vs filler):  {d_val:+.4f}  ({effect_label(d_val)})\n")
    w(f"  Robust d  (target vs filler):  {rd_val:+.4f}  ({effect_label(rd_val)})\n")
    h.cleanup(engine)
    return ProbeResult(
        name="idf",
        ok=d_val > 0.8,
        primary_stat=f"d={d_val:+.4f} ({effect_label(d_val)})",
        summary=f"d={d_val:+.4f}  rd={rd_val:+.4f}  target_mu={float(groups['target']['geo'].mean()):+.4f}  filler_mu={float(groups['filler']['geo'].mean()):+.4f}",
        details=O.getvalue(),
        metrics={
            "cohen_d": d_val,
            "robust_d": rd_val,
            "group_means": {g: float(groups[g]["geo"].mean()) for g in groups},
            "group_medians": {g: float(np.median(groups[g]["geo"])) for g in groups},
            "group_lmoments": {g: groups[g]["lm"] for g in groups},
        },
        duration=time.time() - t0,
    )


def probe_sparse(h):
    """Normalization effectiveness: length fairness within the supra-ngram regime.

    The scoring denominator sqrt(doc_lit * q_lit) exists to prevent long
    documents from dominating shorter ones purely on bit count.  This probe
    tests that purpose by comparing relevant documents at different lengths
    — all above the n-gram threshold so they have compositional signal —
    against long off-topic filler.

    Three relevant document lengths (all supra-ngram):
      - short:   ~ngram×2 tokens — fewest n-gram windows, the document
                 normalization must protect from being buried
      - medium:  ~ngram×4 tokens — moderate engagement
      - long:    ~ngram×8+ tokens — fully engaged control

    These compete against long off-topic filler of similar length to the
    long relevant doc.  The engaged metric has no length correction, so
    long filler can outscore short relevant docs on sheer bit count.
    Geometric normalization should fix this.

    Health:  geometric normalization ranks the short relevant doc above
             all long filler.
    Primary: rank improvement of short relevant (engaged rank - geo rank).
    """
    t0 = time.time()
    ngram = h.config.hdc_ngram
    query = "photosynthesis in chloroplasts converts light energy into chemical energy"

    # All relevant docs are supra-ngram (above the n-gram threshold) so they
    # have compositional signal.  They differ only in length / number of
    # n-gram windows available.
    docs = [
        # Short supra-ngram: ~ngram×2 tokens, just past the threshold
        {
            "text": (
                "Chloroplast photosynthesis converts light energy into "
                "chemical energy stored in glucose molecules"
            ),
            "label": "short_relevant",
            "group": "relevant",
            "tier": "short",
        },
        # Medium supra-ngram: ~ngram×4 tokens, solid engagement
        {
            "text": (
                "Photosynthesis in chloroplasts converts light energy into "
                "chemical energy. Chlorophyll captures photons in thylakoid "
                "membranes driving the light-dependent reactions that produce "
                "ATP and NADPH for carbon fixation"
            ),
            "label": "medium_relevant",
            "group": "relevant",
            "tier": "medium",
        },
        # Long supra-ngram: ~ngram×8+ tokens, fully engaged control
        {
            "text": (
                "Photosynthesis occurs in chloroplasts where light energy is "
                "captured by chlorophyll and converted into chemical energy "
                "stored in glucose molecules through the Calvin cycle. The "
                "light-dependent reactions in thylakoid membranes produce ATP "
                "and NADPH which drive the Calvin cycle in the stroma to fix "
                "carbon dioxide into organic sugars. This process is the "
                "primary source of atmospheric oxygen and the foundation of "
                "most food chains on Earth"
            ),
            "label": "long_relevant",
            "group": "relevant",
            "tier": "long",
        },
        # Long filler — the threat: high bit count, zero topical relevance.
        {
            "text": (
                "The history of the Roman Empire spans over a thousand years "
                "from the founding of Rome through the fall of Constantinople. "
                "Roman engineering achievements included aqueducts, roads, and "
                "monumental architecture that influenced Western civilization "
                "for centuries. The Roman legal system established principles "
                "of jurisprudence that persist in modern legal frameworks "
                "across Europe and the Americas"
            ),
            "label": "long_filler_1",
            "group": "filler",
            "tier": "long",
        },
        {
            "text": (
                "Medieval European feudal society organized land ownership and "
                "military obligation into a hierarchical system of lords, "
                "vassals, and serfs. Agricultural practices including the "
                "three-field rotation system and the introduction of the heavy "
                "plough transformed productivity. Trade guilds in growing "
                "urban centers regulated craft production and maintained "
                "quality standards for manufactured goods"
            ),
            "label": "long_filler_2",
            "group": "filler",
            "tier": "long",
        },
        {
            "text": (
                "The industrial revolution fundamentally transformed "
                "manufacturing through mechanization powered by steam engines "
                "and later electricity. Factory systems replaced cottage "
                "industries and drove rapid urbanization across Britain and "
                "continental Europe. Innovations in textile production, iron "
                "smelting, and transportation created unprecedented economic "
                "growth and social upheaval throughout the nineteenth century"
            ),
            "label": "long_filler_3",
            "group": "filler",
            "tier": "long",
        },
    ]

    engine = h.index(docs)
    bm = h.encode(engine, query)
    s_eng, s_geo, agree, disagree, engaged, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )

    doc_tokens = [h.tokenizer.count_tokens(d["text"]) for d in docs]

    # --- Index sets by role ---
    rel_i = [i for i, d in enumerate(docs) if d["group"] == "relevant"]
    fill_i = [i for i, d in enumerate(docs) if d["group"] == "filler"]

    geo_order = np.argsort(-s_geo)
    eng_order = np.argsort(-s_eng)

    # --- Per-tier analysis: how does each relevant doc rank vs filler? ---
    tiers = ["short", "medium", "long"]
    tier_results = {}
    for tier in tiers:
        ti = [i for i, d in enumerate(docs) if d["group"] == "relevant" and d["tier"] == tier]
        if not ti:
            continue
        ri = ti[0]
        nw = max(0, doc_tokens[ri] - ngram + 1)
        geo_rank = int(np.where(geo_order == ri)[0][0]) + 1
        eng_rank = int(np.where(eng_order == ri)[0][0]) + 1
        geo_above = all(s_geo[ri] > s_geo[fi] for fi in fill_i)
        eng_above = all(s_eng[ri] > s_eng[fi] for fi in fill_i)
        best_filler_geo = max(s_geo[fi] for fi in fill_i)
        best_filler_eng = max(s_eng[fi] for fi in fill_i)
        geo_gap = float(s_geo[ri] - best_filler_geo)
        eng_gap = float(s_eng[ri] - best_filler_eng)
        tier_results[tier] = {
            "idx": ri,
            "label": docs[ri]["label"],
            "tokens": doc_tokens[ri],
            "ngram_windows": nw,
            "geo_rank": geo_rank,
            "eng_rank": eng_rank,
            "rank_imp": eng_rank - geo_rank,
            "geo_above_filler": geo_above,
            "eng_above_filler": eng_above,
            "geo_gap": geo_gap,
            "eng_gap": eng_gap,
            "gap_imp": geo_gap - eng_gap,
            "geo_score": float(s_geo[ri]),
            "eng_score": float(s_eng[ri]),
        }

    # --- Overall separation ---
    geo_d = cohen_d(s_geo[rel_i], s_geo[fill_i])
    eng_d = cohen_d(s_eng[rel_i], s_eng[fill_i])
    geo_rd = robust_d(s_geo[rel_i], s_geo[fill_i])
    eng_rd = robust_d(s_eng[rel_i], s_eng[fill_i])

    # --- Length bias: correlation between doc_lit and score ---
    doc_lits = np.array([float(lit[i]) for i in range(len(docs))])
    rho_lit_geo = spearman_rho(doc_lits, s_geo)
    rho_lit_eng = spearman_rho(doc_lits, s_eng)
    bias_reduced = abs(rho_lit_geo) < abs(rho_lit_eng)

    # --- Length ratio context ---
    if "short" in tier_results and "long" in tier_results:
        length_ratio = tier_results["long"]["tokens"] / max(tier_results["short"]["tokens"], 1)
        window_ratio = tier_results["long"]["ngram_windows"] / max(tier_results["short"]["ngram_windows"], 1)
    else:
        length_ratio = window_ratio = 0.0

    geo_lo, geo_hi = float(s_geo.min()), float(s_geo.max())
    eng_lo, eng_hi = float(s_eng.min()), float(s_eng.max())

    # --- Output ---
    O = io.StringIO()
    w = O.write
    w(banner("NORMALIZATION EFFECTIVENESS ANALYSIS"))
    w(f'\n  QUERY: "{query}"\n')
    w(f"  N-gram: {ngram}     All relevant docs are supra-ngram (have n-gram signal)\n")
    w(f"  Test: does normalization give shorter relevant docs a fair ranking\n")
    w(f"        against longer off-topic filler?\n\n")
    w(f"  Length ratio (long/short relevant): {length_ratio:.1f}x tokens, {window_ratio:.1f}x n-gram windows\n")

    w(section("SIDE-BY-SIDE: GEOMETRIC vs ENGAGED"))
    w(f"  {'label':>20s}  {'group':>8s}  {'tier':>6s}  {'toks':>5s}  {'ng_win':>6s}  {'geo':>8s}  {'eng':>8s}  {'d_lit':>6s}  {'geo bar':<22s}|{'eng bar':<22s}\n")
    w(f"  {sep(n=W)}\n")
    for i in sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True):
        d = docs[i]
        nw = max(0, doc_tokens[i] - ngram + 1)
        gb = hbar(s_geo[i], geo_lo, geo_hi, 22)
        eb = hbar(s_eng[i], eng_lo, eng_hi, 22)
        w(f"  {d['label']:>20s}  {d['group']:>8s}  {d['tier']:>6s}  {doc_tokens[i]:5d}  {nw:6d}  {s_geo[i]:+8.4f}  {s_eng[i]:+8.4f}  {lit[i]:6.0f}  {gb}|{eb}\n")

    w(section("PER-TIER RANKING vs FILLER"))
    w(f"  Each relevant doc (all supra-ngram) tested against {len(fill_i)} long filler docs.\n")
    w(f"  N-gram windows = max(0, tokens - {ngram} + 1)\n\n")
    w(f"  {'tier':>8s}  {'label':>18s}  {'toks':>5s}  {'ng_win':>6s}  {'geo_rank':>8s}  {'eng_rank':>8s}  {'rank_imp':>8s}  {'geo_gap':>8s}  {'eng_gap':>8s}  {'gap_imp':>8s}  above_filler\n")
    w(f"  {sep(n=W)}\n")
    for tier in tiers:
        if tier not in tier_results:
            continue
        tr = tier_results[tier]
        geo_tag = "yes" if tr["geo_above_filler"] else "NO"
        eng_tag = "yes" if tr["eng_above_filler"] else "NO"
        w(f"  {tier:>8s}  {tr['label']:>18s}  {tr['tokens']:5d}  {tr['ngram_windows']:6d}  "
          f"{tr['geo_rank']:8d}  {tr['eng_rank']:8d}  {tr['rank_imp']:+8d}  "
          f"{tr['geo_gap']:+8.4f}  {tr['eng_gap']:+8.4f}  {tr['gap_imp']:+8.4f}  "
          f"geo={geo_tag} eng={eng_tag}\n")

    w(section("LENGTH BIAS"))
    w(f"  Spearman rho (doc_lit vs score) — lower magnitude = less length-biased:\n\n")
    w(f"  {'metric':>30s}  {'geometric':>10s}  {'engaged':>10s}\n  {sep(n=60)}\n")
    w(f"  {'rho(length, score)':>30s}  {rho_lit_geo:+10.4f}  {rho_lit_eng:+10.4f}\n")
    w(f"  {'length bias reduced?':>30s}  {'yes' if bias_reduced else 'no':>10s}\n")

    w(section("OVERALL SEPARATION (relevant vs filler)"))
    w(f"  {'metric':>30s}  {'geometric':>10s}  {'engaged':>10s}\n  {sep(n=60)}\n")
    w(f"  {'Cohen d':>30s}  {geo_d:+10.4f}  {eng_d:+10.4f}\n")
    w(f"  {'Robust d (med/MAD)':>30s}  {geo_rd:+10.4f}  {eng_rd:+10.4f}\n")

    # --- Structural features ---
    features = []
    for tier in tiers:
        if tier not in tier_results:
            continue
        tr = tier_results[tier]
        if not tr["geo_above_filler"]:
            features.append(
                f"{tier} relevant ({tr['label']}, {tr['tokens']} tokens, "
                f"{tr['ngram_windows']} n-gram windows): geo rank {tr['geo_rank']}, "
                f"gap {tr['geo_gap']:+.4f} vs filler. "
                f"Normalization {'improved' if tr['rank_imp'] > 0 else 'did not improve'} rank "
                f"(eng {tr['eng_rank']} -> geo {tr['geo_rank']})."
            )
    if not bias_reduced:
        features.append(
            f"Length-score correlation not reduced by normalization "
            f"(geo rho={rho_lit_geo:+.4f} vs eng rho={rho_lit_eng:+.4f})."
        )

    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    # --- Health: short relevant doc (the primary case) above all filler ---
    healthy = tier_results.get("short", {}).get("geo_above_filler", False)

    # --- Summary ---
    tier_summary = "  ".join(
        f"{t}:rank_imp={tr['rank_imp']:+d},gap={tr['geo_gap']:+.4f}"
        for t, tr in tier_results.items()
    )

    h.cleanup(engine)
    return ProbeResult(
        name="sparse",
        ok=healthy,
        primary_stat=(
            f"short_above={'yes' if healthy else 'no'}  "
            f"bias_reduced={'yes' if bias_reduced else 'no'}"
        ),
        summary=(
            f"{tier_summary}  bias_reduced={'yes' if bias_reduced else 'no'}  "
            f"rho_geo={rho_lit_geo:+.4f}  rho_eng={rho_lit_eng:+.4f}"
        ),
        details=O.getvalue(),
        metrics={
            "ngram": ngram,
            "length_ratio": length_ratio,
            "window_ratio": window_ratio,
            "tier_results": {
                t: {k: v for k, v in tr.items()}
                for t, tr in tier_results.items()
            },
            "geo_cohen_d": geo_d,
            "eng_cohen_d": eng_d,
            "geo_robust_d": geo_rd,
            "eng_robust_d": eng_rd,
            "rho_lit_geo": rho_lit_geo,
            "rho_lit_eng": rho_lit_eng,
            "bias_reduced": bias_reduced,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_specificity(h):
    """Shared information transfer function: how input-space lexical overlap
    maps to output-space vector correlation.

    Characterizes the functional form of E's response to token overlap:
      - Transfer function shape (threshold, slope, inflection)
      - Mutual information I(shared_count; score)
      - Lexical opacity (how much of the score variance is NOT explained
        by lexical overlap — i.e., the non-lexical information content)
      - Normalization distortion across the overlap spectrum

    Health:  exact match is rank-1 AND margin > 0.01.
    Primary metric: exact-match margin (rank-1 score minus rank-2 score).
    """
    t0 = time.time()
    query = "eigenvalue decomposition of symmetric positive definite matrices"
    dims = h.config.hdc_dimensions

    docs = [
        {
            "text": "The eigenvalue decomposition of a symmetric positive definite matrix A yields A = Q Lambda Q^T where Q is orthogonal and Lambda is diagonal with positive entries",
            "label": "exact",
        },
        {
            "text": "Eigenvalues and eigenvectors are fundamental concepts in linear algebra used to analyze linear transformations and matrix properties",
            "label": "broad_eigen",
        },
        {
            "text": "Matrix decomposition methods include LU, QR, SVD, and eigendecomposition, each suited to different computational tasks",
            "label": "broad_decomp",
        },
        {
            "text": "Linear algebra is a branch of mathematics concerning linear equations, linear maps, and their representations in vector spaces and through matrices",
            "label": "field_level",
        },
        {
            "text": "Mathematics encompasses many fields including algebra, geometry, analysis, and number theory",
            "label": "super_generic",
        },
        {
            "text": "The Krebs cycle is a series of chemical reactions used by aerobic organisms to release stored energy",
            "label": "unrelated",
        },
    ]
    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, agree, disagree, _, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )
    qp, qn, qa = h.qbits(bm["pos"], bm["neg"])
    q_lit_uni_val = float(bm["lit_uni"][0])

    # --- measure actual token overlap per document -------------------------
    doc_lit_uni = engine.db._get_lit_uni()
    q_tids = set(h.tokenizer.bulk_tokenize([query])[0])
    uni_lits = []
    for i in range(len(docs)):
        ul = (
            float(doc_lit_uni[i])
            if len(doc_lit_uni) > i and doc_lit_uni[i] > 0
            else float(lit[i])
        )
        uni_lits.append(ul)
    for i, d in enumerate(docs):
        d_tids = set(h.tokenizer.bulk_tokenize([d["text"]])[0])
        shared = q_tids & d_tids
        d["shared_tokens"] = len(shared)
        d["shared_frac"] = len(shared) / len(q_tids) if q_tids else 0.0
        d["geo"] = float(s_geo[i])
        d["lit"] = uni_lits[i]
        d["norm_factor"] = float(np.sqrt(uni_lits[i] * q_lit_uni_val))

    # --- group by measured token overlap -----------------------------------
    overlap_i = [i for i, d in enumerate(docs) if d["shared_tokens"] > 0]
    disjoint_i = [i for i, d in enumerate(docs) if d["shared_tokens"] == 0]

    ov_scores = s_geo[overlap_i] if overlap_i else np.array([0.0])
    dj_scores = s_geo[disjoint_i] if disjoint_i else np.array([0.0])

    exact_i = 0
    exact_is_top = int(np.argmax(s_geo)) == exact_i
    ov_mean = float(ov_scores.mean())
    dj_mean = float(dj_scores.mean())
    group_gap = ov_mean - dj_mean

    # --- exact-match margin: rank-1 score minus rank-2 score ----------------
    sorted_scores = np.sort(s_geo)[::-1]
    exact_margin = (
        float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    )

    # --- normalization distortion: how much does the score denominator vary? -
    lit_vals = np.array([d["lit"] for d in docs])
    norm_vals = np.array([d["norm_factor"] for d in docs])
    norm_ratio = (
        float(norm_vals.max() / norm_vals.min())
        if norm_vals.min() > 0
        else float("inf")
    )
    raw_ratio = (
        float(np.sqrt(lit_vals.max()) / np.sqrt(lit_vals.min()))
        if lit_vals.min() > 0
        else float("inf")
    )

    # --- Spearman: shared_token_count vs score ---
    shared_counts = [d["shared_tokens"] for d in docs]
    score_vals = [d["geo"] for d in docs]
    rho_tokens = spearman_rho(shared_counts, score_vals)

    # --- Mutual information I(shared_count; score) via binned estimator ---
    # Bin scores into quantiles to estimate discrete MI
    sc_arr = np.array(score_vals)
    sh_arr = np.array(shared_counts)
    n_docs = len(docs)

    # Group by shared_count (natural bins)
    unique_k = sorted(set(shared_counts))
    # H(score) - marginal entropy of scores (binned into unique-k-count groups)
    # Assign each score to its rank bin for MI estimation
    rank_bins = np.argsort(np.argsort(sc_arr))  # rank of each score

    # H(shared_count) — entropy of the overlap distribution
    k_counts = np.array([shared_counts.count(k) for k in unique_k])
    p_k = k_counts / n_docs
    H_shared = float(-np.sum(p_k * np.log2(p_k + 1e-12)))

    # H(score | shared_count = k) — conditional entropy per overlap level
    H_cond_per_k = {}
    for k in unique_k:
        k_scores = [s for s, c in zip(score_vals, shared_counts) if c == k]
        if len(k_scores) <= 1:
            H_cond_per_k[k] = 0.0
        else:
            # Variance-based entropy estimate for continuous variable (differential entropy)
            var_k = float(np.var(k_scores))
            H_cond_per_k[k] = 0.5 * math.log2(2 * math.pi * math.e * max(var_k, 1e-12))

    # H(score) — marginal differential entropy
    var_total = float(np.var(score_vals))
    H_score = 0.5 * math.log2(2 * math.pi * math.e * max(var_total, 1e-12))

    # Conditional entropy H(score | shared_count) = sum p(k) * H(score|k)
    H_score_given_k = sum(
        (shared_counts.count(k) / n_docs) * H_cond_per_k[k] for k in unique_k
    )
    # MI = H(score) - H(score | shared_count)
    mutual_info = H_score - H_score_given_k

    # Lexical opacity = 1 - R^2 (fraction of score variance NOT explained by overlap)
    r_sq = rho_tokens**2
    lexical_opacity = 1.0 - r_sq

    lo, hi = float(s_geo.min()), float(s_geo.max())

    # --- Transfer function: score as f(shared_count) ---
    # Compute per-k mean score for transfer function characterization
    k_means = {}
    for k in unique_k:
        k_scores = [s for s, c in zip(score_vals, shared_counts) if c == k]
        k_means[k] = float(np.mean(k_scores))

    # Find inflection region: largest jump between consecutive k values
    sorted_k = sorted(k_means.keys())
    max_jump = 0.0
    inflection_k = 0
    for i in range(len(sorted_k) - 1):
        jump = k_means[sorted_k[i + 1]] - k_means[sorted_k[i]]
        if jump > max_jump:
            max_jump = jump
            inflection_k = sorted_k[i]

    # --- Structural features ---
    features = []
    if norm_ratio > 1.5:
        features.append(
            f"Normalization distortion: {norm_ratio:.2f}x range. "
            f"The scoring denominator varies by {norm_ratio:.2f}x across the overlap spectrum."
        )
    if not exact_is_top:
        features.append("Exact match is not rank-1.")
    if group_gap <= 0:
        features.append(
            f"Lexical opacity in partial-overlap regime: overlap mean ({ov_mean:+.4f}) "
            f"does not exceed disjoint mean ({dj_mean:+.4f}). The encoding is "
            f"opaque to partial lexical overlap (opacity={lexical_opacity:.2f})."
        )

    O = io.StringIO()
    w = O.write
    w(banner("SHARED INFORMATION TRANSFER FUNCTION"))
    w(f'\n  QUERY: "{query}"\n  QUERY VECTOR: active={qa} ({pct(qa, dims)})\n')
    w(f"  Query tokens: {len(q_tids)}\n")

    w(section("PER-DOCUMENT ANALYSIS (sorted by score)"))
    w(
        f"  {'label':>15s}  {'shared':>6s}  {'geo':>8s}  {'agree':>6s}  {'disag':>6s}  {'lit':>6s}  {'norm':>8s}  bar\n  {sep(n=110)}\n"
    )
    for i in sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True):
        d = docs[i]
        tag = (
            f"  [{d['shared_tokens']} tokens]"
            if d["shared_tokens"] > 0
            else "  [BLIND]"
        )
        w(
            f"  {d['label']:>15s}  {d['shared_tokens']:6d}  {s_geo[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {lit[i]:6.0f}  {d['norm_factor']:8.1f}  {hbar(s_geo[i], lo, hi)}{tag}\n"
        )

    w(section("TRANSFER FUNCTION"))
    w("  Score as a function of shared token count.\n\n")
    w(
        f"  {'k (shared)':>12s}  {'n_docs':>6s}  {'mean_score':>10s}  bar\n  {sep(n=50)}\n"
    )
    for k in sorted_k:
        n_k = shared_counts.count(k)
        w(f"  {k:12d}  {n_k:6d}  {k_means[k]:+10.4f}  {hbar(k_means[k], lo, hi, 20)}\n")
    w(
        f"\n  Inflection region: k={inflection_k} -> k={inflection_k + 1 if inflection_k + 1 in k_means else '?'}  (jump={max_jump:+.4f})\n"
    )
    if len(sorted_k) >= 3:
        w(f"  Below k={inflection_k + 1}: mean score is near or below noise floor.\n")
        w(
            f"  Above k={inflection_k}: score rises sharply with additional shared tokens.\n"
        )

    w(section("INFORMATION-THEORETIC MEASURES"))
    w(f"  H(score):                    {H_score:8.4f} bits  (marginal score entropy)\n")
    w(
        f"  H(shared_count):             {H_shared:8.4f} bits  (overlap distribution entropy)\n"
    )
    w(
        f"  H(score | shared_count):     {H_score_given_k:8.4f} bits  (residual uncertainty)\n"
    )
    w(
        f"  I(shared_count; score):      {mutual_info:8.4f} bits  (information overlap provides about score)\n"
    )
    w(f"\n  Spearman rho:                {rho_tokens:+.4f}\n")
    w(f"  Lexical opacity (1 - rho^2): {lexical_opacity:.4f}\n")
    w(
        f"  Interpretation: {lexical_opacity * 100:.0f}% of score variance is driven by\n"
    )
    w("  non-lexical features (n-gram position, IDF weighting, composition).\n")

    w(f"\n  CONDITIONAL ENTROPY by overlap level\n  {sep(n=50)}\n")
    for k in sorted_k:
        n_k = shared_counts.count(k)
        h_str = f"{H_cond_per_k[k]:8.4f}" if n_k > 1 else "    ---"
        w(f"  k={k:2d}:  H(score|k)={h_str}  n={n_k}\n")

    w(section("SEPARATION METRICS"))
    w(f"  Exact match is rank-1: {'yes' if exact_is_top else 'NO'}\n")
    w(f"  Exact-match margin (rank-1 - rank-2): {exact_margin:+.4f}\n")
    w(f"  Overlap docs (shared > 0):   n={len(overlap_i)}  mean={ov_mean:+.4f}\n")
    w(f"  Disjoint docs (shared = 0):  n={len(disjoint_i)}  mean={dj_mean:+.4f}\n")
    w(f"  Group gap (overlap - disjoint):       {group_gap:+.4f}\n")

    w(section("NORMALIZATION DISTORTION"))
    w(f"  score denom = sqrt(lit_uni * q_lit_uni)\n  {sep(n=60)}\n")
    for d in sorted(docs, key=lambda x: x["norm_factor"]):
        w(
            f"    {d['label']:>15s}  lit={d['lit']:6.0f}  norm_denom={d['norm_factor']:8.1f}\n"
        )
    w(f"\n  Raw sqrt(lit) ratio:  {raw_ratio:.2f}x\n")
    w(f"  Norm denom ratio:     {norm_ratio:.2f}x\n")

    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    w("\n  SCORE vs SHARED TOKEN COUNT\n")
    w(
        ascii_scatter(
            xs=shared_counts,
            ys=score_vals,
            labels=[d["label"] for d in docs],
            x_label="shared query tokens",
        )
    )

    healthy = exact_is_top and exact_margin > 0.01
    h.cleanup(engine)
    return ProbeResult(
        name="specificity",
        ok=healthy,
        primary_stat=f"margin={exact_margin:+.4f}  opacity={lexical_opacity:.2f}",
        summary=f"margin={exact_margin:+.4f}  MI={mutual_info:.3f}  opacity={lexical_opacity:.2f}  norm={norm_ratio:.2f}x",
        details=O.getvalue(),
        metrics={
            "exact_margin": exact_margin,
            "group_gap": group_gap,
            "exact_is_top": exact_is_top,
            "norm_ratio": norm_ratio,
            "raw_ratio": raw_ratio,
            "rho_token_count": rho_tokens,
            "lexical_opacity": lexical_opacity,
            "mutual_information": mutual_info,
            "H_score": H_score,
            "H_shared_count": H_shared,
            "H_score_given_k": H_score_given_k,
            "inflection_k": inflection_k,
            "disjoint_count": len(disjoint_i),
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_noise(h):
    """Signal-to-noise separation.  Primary: Cohen's d (signal vs noise)"""
    t0 = time.time()
    query = "quantum entanglement between photon pairs"
    rng = np.random.default_rng(42)
    noise_docs = [
        {
            "text": "".join(
                rng.choice(
                    list("abcdefghijklmnopqrstuvwxyz "), size=rng.integers(50, 200)
                )
            ),
            "label": f"noise_{i}",
            "group": "noise",
        }
        for i in range(8)
    ]
    signal_docs = [
        {
            "text": "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly affects the other regardless of distance",
            "label": "relevant",
            "group": "signal",
        },
        {
            "text": "Photon pair generation through spontaneous parametric down-conversion is a primary method for creating entangled photon states in quantum optics experiments",
            "label": "highly_relevant",
            "group": "signal",
        },
        {
            "text": "Classical physics describes the motion of macroscopic objects under forces like gravity and electromagnetism",
            "label": "same_field",
            "group": "signal",
        },
    ]
    docs = signal_docs + noise_docs
    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, agree, disagree, _, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )
    sig_i = [i for i, d in enumerate(docs) if d["group"] == "signal"]
    noi_i = [i for i, d in enumerate(docs) if d["group"] == "noise"]
    d_val = cohen_d(s_geo[sig_i], s_geo[noi_i])
    rd_val = robust_d(s_geo[sig_i], s_geo[noi_i])
    margin = float(s_geo[sig_i].min() - s_geo[noi_i].max())
    lm_sig = l_moments(s_geo[sig_i])
    lm_noi = l_moments(s_geo[noi_i])
    lo, hi = float(s_geo.min()), float(s_geo.max())
    O = io.StringIO()
    w = O.write
    w(banner("NOISE FLOOR ANALYSIS"))
    w(
        f'\n  QUERY: "{query}"\n  Corpus: {len(signal_docs)} signal + {len(noise_docs)} noise\n'
    )
    w(section("RANKED RESULTS"))
    w(
        f"  {'label':>20s}  {'group':>8s}  {'geo':>8s}  {'agree':>6s}  {'disag':>6s}  {'d_lit':>6s}  bar\n  {sep(n=100)}\n"
    )
    for i in sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True):
        d = docs[i]
        tag = " <-" if d["group"] == "noise" else ""
        w(
            f"  {d['label']:>20s}  {d['group']:>8s}  {s_geo[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {lit[i]:6.0f}  {hbar(s_geo[i], lo, hi)}{tag}\n"
        )
    w(f"\n  GROUP STATISTICS\n  {sep(n=70)}\n")
    for grp, idx, lm in [("signal", sig_i, lm_sig), ("noise", noi_i, lm_noi)]:
        v = s_geo[idx]
        w(
            f"  {grp:>8s}: mu={float(v.mean()):+.4f}  med={float(np.median(v)):+.4f}  sigma={float(v.std()):.4f}  mad={mad(v):.4f}  [{float(v.min()):+.4f}, {float(v.max()):+.4f}]  {sparkline(v.tolist(), 15)}\n"
        )
        w(f"  {'':>8s}  L-mom: {fmt_lm(lm)}\n")
    w(f"\n  Separation margin (signal_min - noise_max): {margin:+.4f}\n")
    w(
        f"  Cohen's d (signal vs noise):                {d_val:+.4f}  ({effect_label(d_val)})\n"
    )
    w(
        f"  Robust d  (signal vs noise):                {rd_val:+.4f}  ({effect_label(rd_val)})\n"
    )
    h.cleanup(engine)
    return ProbeResult(
        name="noise",
        ok=d_val > 0.8,
        primary_stat=f"d={d_val:+.4f} ({effect_label(d_val)})",
        summary=f"d={d_val:+.4f}  rd={rd_val:+.4f}  margin={margin:+.4f}",
        details=O.getvalue(),
        metrics={
            "cohen_d": d_val,
            "robust_d": rd_val,
            "margin": margin,
            "lm_signal": lm_sig,
            "lm_noise": lm_noi,
        },
        duration=time.time() - t0,
    )


def probe_duplicate(h):
    """Encoding sensitivity profile: Lipschitz structure of the encoding map.

    Measures the function  f: retention → score  by applying perturbations of
    known type and magnitude, then characterizing the map's local sensitivity,
    continuity class, and directional response.

    Perturbation classes:
      tokenization  — whitespace/punctuation changes (zero n-gram disruption)
      lexical       — synonym substitution (moderate n-gram disruption)
      structural    — phrase collapse, reordering (high n-gram disruption)

    Primary: Spearman rho (retention -> score)
    """
    t0 = time.time()
    query = "gradient descent optimization for neural networks"
    max_n = h.config.hdc_ngram
    base = "Gradient descent is an iterative optimization algorithm used to minimize a loss function by updating parameters in the direction of steepest descent"
    variants = [
        {"text": base, "label": "original", "pclass": "identity"},
        {"text": base + ".", "label": "period_added", "pclass": "tokenization"},
        {"text": "  " + base + "  ", "label": "ws_padded", "pclass": "tokenization"},
        {
            "text": base.replace("iterative", "repeated"),
            "label": "synonym_swap",
            "pclass": "lexical",
        },
        {
            "text": base.replace("optimization algorithm", "method"),
            "label": "phrase_collapse",
            "pclass": "structural",
        },
        {
            "text": " ".join(base.split()[::-1]),
            "label": "reversed",
            "pclass": "structural",
        },
    ]
    control = {
        "text": "The recipe for chocolate cake requires flour, sugar, cocoa powder, eggs, and butter mixed together and baked at 350 degrees",
        "label": "unrelated",
    }
    for v in variants:
        v["ret"] = h.ngram_retention(base, v["text"], max_n)
    docs = [
        {"text": v["text"], "label": v["label"], "group": "variant"} for v in variants
    ]
    docs.append(
        {"text": control["text"], "label": control["label"], "group": "control"}
    )
    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, agree, disagree, _, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )
    for i, v in enumerate(variants):
        v["geo"] = float(s_geo[i])
    rets = [v["ret"] for v in variants]
    geos = [v["geo"] for v in variants]
    rho = spearman_rho(rets, geos)
    r = pearson_r(rets, geos)
    lo = min(float(s_geo.min()), float(s_geo[len(variants)]))
    hi = max(float(s_geo.max()), float(s_geo[len(variants)]))
    ranked = sorted(variants, key=lambda v: v["ret"], reverse=True)

    # --- Local sensitivity per perturbation ---
    orig_geo = variants[0]["geo"]
    sensitivity = []
    for v in variants[1:]:
        disrupted = 1.0 - v["ret"]
        delta_score = abs(orig_geo - v["geo"])
        local_L = delta_score / disrupted if disrupted > 0.001 else float("inf")
        sensitivity.append(
            {
                "label": v["label"],
                "pclass": v["pclass"],
                "disrupted": disrupted,
                "delta_score": delta_score,
                "lipschitz_local": local_L,
            }
        )

    # --- Per-class aggregate sensitivity ---
    pclasses = {}
    for s in sensitivity:
        pc = s["pclass"]
        pclasses.setdefault(pc, []).append(s)
    class_stats = {}
    for pc, items in pclasses.items():
        Ls = [
            s["lipschitz_local"] for s in items if s["lipschitz_local"] != float("inf")
        ]
        deltas = [s["delta_score"] for s in items]
        class_stats[pc] = {
            "mean_L": float(np.mean(Ls)) if Ls else float("inf"),
            "max_L": float(np.max(Ls)) if Ls else float("inf"),
            "mean_delta": float(np.mean(deltas)),
            "n": len(items),
        }

    # --- Continuity analysis: identity-boundary discontinuity ---
    # Perturbations with ret=1.0 but score ≠ original reveal a cusp
    identity_variants = [v for v in variants[1:] if v["ret"] >= 0.999]
    jump_size = (
        max(abs(v["geo"] - orig_geo) for v in identity_variants)
        if identity_variants
        else 0.0
    )

    # --- Isometry constant: baseline correlation of independent draws ---
    control_geo = float(s_geo[len(variants)])
    # Count how many variants fall below the isometry constant
    below_floor = sum(1 for v in variants if v["geo"] < control_geo)

    # --- Curvature: are perturbed variants anti-correlated (below noise floor)? ---
    anti_correlated = [v for v in variants[1:] if v["geo"] < 0]

    O = io.StringIO()
    w = O.write
    w(banner("ENCODING SENSITIVITY PROFILE"))
    w(f'\n  QUERY: "{query}"\n  N-gram: {max_n}     Base: "{base[:70]}..."\n')

    w(section("VARIANT SCORES"))
    w(
        f"  {'label':>20s}  {'retention':>9s}  {'geo':>8s}  {'agree':>6s}  {'disag':>6s}  {'d_lit':>6s}  bar\n  {sep(n=110)}\n"
    )
    for i, d in enumerate(docs):
        if d["group"] == "variant":
            v = variants[i]
            w(
                f"  {v['label']:>20s}  {v['ret']:9.3f}  {s_geo[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {lit[i]:6.0f}  {hbar(s_geo[i], lo, hi)}\n"
            )
        else:
            w(
                f"  {d['label']:>20s}  {'---':>9s}  {s_geo[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {lit[i]:6.0f}  {hbar(s_geo[i], lo, hi)}  <- control\n"
            )

    w(f"\n  RETENTION-ORDERED RANKING\n  {sep(n=70)}\n")
    for i, v in enumerate(ranked):
        arrow = ""
        if i < len(ranked) - 1:
            nxt = ranked[i + 1]
            if v["ret"] > nxt["ret"]:
                ok = v["geo"] >= nxt["geo"]
                arrow = f"  {'v' if ok else '^!'} {nxt['label']}"
        w(
            f"  {i + 1}. {v['label']:>20s}  ret={v['ret']:.3f}  geo={v['geo']:+.4f}{arrow}\n"
        )

    w("\n  RETENTION vs SCORE\n")
    w(
        ascii_scatter(
            xs=rets,
            ys=geos,
            labels=[v["label"] for v in variants],
            x_label="n-gram retention",
        )
    )

    w(section("LOCAL SENSITIVITY (Lipschitz profile)"))
    w("  Sensitivity = |delta_score| / (1 - retention)  at each perturbation point.\n")
    w("  Higher values indicate the encoding amplifies that perturbation class.\n\n")
    w(
        f"  {'label':>20s}  {'class':>14s}  {'disrupted':>9s}  {'|delta|':>8s}  {'L_local':>8s}  bar\n  {sep(n=80)}\n"
    )
    max_L_finite = max(
        (
            s["lipschitz_local"]
            for s in sensitivity
            if s["lipschitz_local"] != float("inf")
        ),
        default=1.0,
    )
    for s in sorted(sensitivity, key=lambda x: x["lipschitz_local"], reverse=True):
        L_str = (
            f"{s['lipschitz_local']:8.4f}"
            if s["lipschitz_local"] != float("inf")
            else "     inf"
        )
        L_bar = (
            s["lipschitz_local"]
            if s["lipschitz_local"] != float("inf")
            else max_L_finite
        )
        w(
            f"  {s['label']:>20s}  {s['pclass']:>14s}  {100 * s['disrupted']:8.1f}%  {s['delta_score']:8.4f}  {L_str}  {hbar_abs(L_bar, max_L_finite, 20)}\n"
        )

    w(section("PERTURBATION CLASS SENSITIVITY"))
    w("  Aggregate sensitivity by perturbation type.\n\n")
    w(
        f"  {'class':>14s}  {'n':>3s}  {'mean_L':>8s}  {'max_L':>8s}  {'mean_|d|':>8s}  bar\n  {sep(n=60)}\n"
    )
    for pc in ["tokenization", "lexical", "structural"]:
        if pc in class_stats:
            cs = class_stats[pc]
            mL = f"{cs['mean_L']:8.4f}" if cs["mean_L"] != float("inf") else "     inf"
            xL = f"{cs['max_L']:8.4f}" if cs["max_L"] != float("inf") else "     inf"
            bar_v = cs["mean_L"] if cs["mean_L"] != float("inf") else max_L_finite
            w(
                f"  {pc:>14s}  {cs['n']:3d}  {mL}  {xL}  {cs['mean_delta']:8.4f}  {hbar_abs(bar_v, max_L_finite, 20)}\n"
            )

    w(section("CONTINUITY ANALYSIS"))
    w(f"  Identity-boundary jump (ret=1.0, score≠original): {jump_size:.4f}\n")
    if jump_size > 0.01:
        w("  The encoding map has a discontinuity at retention=1.0.\n")
        w("  Tokenization-level perturbations that preserve all n-gram content\n")
        w("  still shift n-gram window alignment, producing orthogonal bindings.\n")
    else:
        w("  The encoding map is approximately continuous at the identity.\n")

    w(f"\n  Isometry constant (control score):  {control_geo:+.4f}\n")
    w("  This is the baseline correlation between independent draws from\n")
    w("  the encoding's output distribution — the map's noise floor.\n")
    if below_floor > 0:
        w(f"\n  Variants below isometry constant:   {below_floor}/{len(variants)}\n")
        w("  These perturbations produce vectors that are anti-correlated\n")
        w("  with the query, indicating negative curvature in those\n")
        w("  perturbation directions.\n")

    if anti_correlated:
        w(f"\n  Anti-correlated variants (score < 0): {len(anti_correlated)}\n")
        w("  The n-gram promotion layer actively disagrees with the query\n")
        w("  when window alignment is disrupted — the encoding's output\n")
        w("  space has regions of negative inner product under perturbation.\n")

    w(section("CORRELATION STRUCTURE"))
    w(f"  Spearman rho (retention -> score): {rho:+.4f}\n")
    w(f"  Pearson r:                         {r:+.4f}\n\n")
    w("  The encoding map f: retention -> score has a monotonic tendency\n")
    w(
        f"  (rho={rho:+.4f}) with {'strong' if abs(rho) > 0.8 else 'moderate' if abs(rho) > 0.5 else 'weak'} rank preservation.\n"
    )

    # --- Structural features (replaces "weaknesses") ---
    features = []
    if jump_size > 0.01:
        features.append(
            f"Discontinuity at identity: jump={jump_size:.4f}. The encoding amplifies "
            f"tokenization-boundary perturbations ~{max(s['lipschitz_local'] for s in sensitivity if s['pclass'] == 'tokenization' and s['lipschitz_local'] != float('inf')):.1f}x "
            f"relative to structural perturbations."
        )
    if below_floor > 0:
        features.append(
            f"Negative curvature: {below_floor}/{len(variants)} perturbations produce "
            f"vectors below the isometry constant ({control_geo:+.4f}). The n-gram "
            f"promotion layer generates anti-correlated signal under window misalignment."
        )
    # Sensitivity ratio across classes
    tok_L = class_stats.get("tokenization", {}).get("mean_L", 0)
    struct_L = class_stats.get("structural", {}).get("mean_L", 0)
    if tok_L != float("inf") and struct_L > 0:
        ratio = tok_L / struct_L
        if ratio > 2.0:
            features.append(
                f"Anisotropic sensitivity: tokenization class {ratio:.1f}x more "
                f"sensitive than structural class per unit disruption."
            )

    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="duplicate",
        ok=rho > 0.7,
        primary_stat=f"rho={rho:+.4f}",
        summary=f"rho={rho:+.4f}  r={r:+.4f}  jump={jump_size:.4f}  iso={control_geo:+.4f}",
        details=O.getvalue(),
        metrics={
            "rho": rho,
            "pearson_r": r,
            "identity_jump": jump_size,
            "isometry_constant": control_geo,
            "below_isometry": below_floor,
            "anti_correlated_count": len(anti_correlated),
            "class_sensitivity": {pc: cs["mean_L"] for pc, cs in class_stats.items()},
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_topic(h):
    """Cross-topic selectivity matrix.  Primary: minimum diagonal dominance margin"""
    t0 = time.time()
    dims = h.config.hdc_dimensions
    topics = {
        "marine": [
            "Coral reef ecosystems support an enormous diversity of marine life including fish, invertebrates, and algae in warm shallow tropical waters",
            "Deep sea hydrothermal vents host extremophile organisms that derive energy from chemosynthesis rather than photosynthesis",
            "Marine mammal communication involves complex vocalizations used for echolocation, mating calls, and social coordination among pods",
        ],
        "compiler": [
            "Lexical analysis tokenizes source code into a stream of tokens by matching patterns defined by regular expressions in the language specification",
            "Abstract syntax trees represent the hierarchical structure of source code after parsing, enabling semantic analysis and code generation phases",
            "Register allocation assigns program variables to CPU registers using graph coloring algorithms to minimize memory spills during code generation",
        ],
        "medieval": [
            "The feudal system organized medieval European society into a hierarchy of lords, vassals, and serfs bound by obligations of military service and agricultural labor",
            "Gothic cathedral construction employed flying buttresses and ribbed vaults to achieve unprecedented height and allow large stained glass windows",
            "The Black Death of 1347 killed an estimated one-third of Europe's population and fundamentally transformed economic and social structures",
        ],
    }
    queries = {
        "marine": "coral reef fish diversity ocean ecosystem",
        "compiler": "lexer parser abstract syntax tree code generation",
        "medieval": "feudal lords vassals medieval European society",
    }
    docs = []
    ranges = {}
    idx = 0
    for topic, texts in topics.items():
        start = idx
        for i, text in enumerate(texts):
            docs.append({"text": text, "label": f"{topic}_{i}", "topic": topic})
            idx += 1
        ranges[topic] = (start, idx)
    engine = h.index(docs)
    cross = {}
    cross_med = {}
    per_q = {}
    for qt, qtext in queries.items():
        bm = h.encode(engine, qtext)
        _, sg, ag, dg, eg, lt = h.scores(
            engine,
            bm["pos"],
            bm["neg"],
            q_lit_uni=float(bm["lit_uni"][0]),
            q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
        )
        _, _, qa = h.qbits(bm["pos"], bm["neg"])
        per_q[qt] = {"geo": sg, "agree": ag, "disagree": dg, "lit": lt, "qa": qa}
        for dt, (s, e) in ranges.items():
            cross[(qt, dt)] = float(sg[s:e].mean())
            cross_med[(qt, dt)] = float(np.median(sg[s:e]))
    tn = list(topics.keys())
    margins = []
    margins_med = []
    for qt in tn:
        own = cross[(qt, qt)]
        best_other = max(cross[(qt, dt)] for dt in tn if dt != qt)
        margins.append(own - best_other)
        own_med = cross_med[(qt, qt)]
        best_other_med = max(cross_med[(qt, dt)] for dt in tn if dt != qt)
        margins_med.append(own_med - best_other_med)
    min_margin = min(margins)
    min_margin_med = min(margins_med)
    O = io.StringIO()
    w = O.write
    w(banner("TOPIC SEPARATION ANALYSIS"))
    w(f"\n  Topics: {', '.join(tn)}     Docs/topic: {len(list(topics.values())[0])}\n")
    for qt, qtext in queries.items():
        d = per_q[qt]
        sg = d["geo"]
        lo_q, hi_q = float(sg.min()), float(sg.max())
        w(section(f"QUERY: {qt}"))
        w(f'  "{qtext}"\n  Query active dims: {d["qa"]} ({pct(d["qa"], dims)})\n\n')
        w(
            f"  {'label':>15s}  {'topic':>10s}  {'geo':>8s}  {'agree':>6s}  {'disag':>6s}  {'d_lit':>6s}  bar\n  {sep(n=100)}\n"
        )
        for i, doc in enumerate(docs):
            tag = " <-" if doc["topic"] == qt else ""
            w(
                f"  {doc['label']:>15s}  {doc['topic']:>10s}  {sg[i]:+8.4f}  {d['agree'][i]:6.0f}  {d['disagree'][i]:6.0f}  {d['lit'][i]:6.0f}  {hbar(sg[i], lo_q, hi_q)}{tag}\n"
            )
    w(banner("CROSS-TOPIC SCORE MATRIX"))
    hdr = f"  {'query / doc':>15s}"
    for dt in tn:
        hdr += f"  {dt:>12s}"
    hdr += f"  {'margin':>10s}"
    w(f"\n{hdr}\n  {sep(n=65)}\n")
    for qi, qt in enumerate(tn):
        row = f"  {qt:>15s}"
        for dt in tn:
            s = cross[(qt, dt)]
            star = " *" if qt == dt else "  "
            row += f"  {s:+10.4f}{star}"
        row += f"  {margins[qi]:+10.4f}"
        w(row + "\n")
    w(f"\n  DIAGONAL DOMINANCE\n  {sep(n=50)}\n")
    for qt in tn:
        own = cross[(qt, qt)]
        for dt in tn:
            if dt != qt:
                m = own - cross[(qt, dt)]
                w(f"  {qt} vs {dt}: {m:+.4f}  {hbar_abs(m, max(margins), 20)}\n")
    w(f"\n  Minimum dominance margin (mean):   {min_margin:+.4f}\n")
    w(f"  Minimum dominance margin (median): {min_margin_med:+.4f}\n")

    # L-moments: per-topic score distributions under own query
    w(f"\n  L-MOMENTS (own-topic scores under own query)\n  {sep(n=70)}\n")
    for qt in tn:
        s, e = ranges[qt]
        own_scores = per_q[qt]["geo"][s:e]
        lm = l_moments(own_scores)
        w(f"  {qt:>10s}:  {fmt_lm(lm)}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="topic",
        ok=min_margin > 0,
        primary_stat=f"min_margin={min_margin:+.4f}",
        summary=f"min_margin={min_margin:+.4f}  min_margin_med={min_margin_med:+.4f}",
        details=O.getvalue(),
        metrics={
            "min_margin": min_margin,
            "min_margin_median": min_margin_med,
            "margins": margins,
            "margins_median": margins_med,
        },
        duration=time.time() - t0,
    )


def probe_density(h):
    """Encoding entropy rate: information capacity of the representation
    as a function of input length.

    Characterizes the encoding channel E: text -> {-1, 0, +1}^d by measuring:
      1. Per-document entropy (bits of information in the ternary vector)
      2. Entropy rate (marginal entropy per additional input token)
      3. Phase transition between sub-ngram and supra-ngram regimes
      4. Normalization iso-entropy (whether scoring operates on
         comparable-entropy representations)

    Structural invariants: non-zero, no saturation, polarity balance.
    Health:  all invariants pass.
    Primary: entropy rate characterization and normalization distortion.
    """
    t0 = time.time()
    dims = h.config.hdc_dimensions
    ngram = h.config.hdc_ngram
    docs = [
        {"text": "cat", "label": "1_word"},
        {"text": "black cat", "label": "2_words"},
        {"text": "the quick brown fox", "label": "4_words"},
        {
            "text": "the quick brown fox jumps over the lazy dog near the river",
            "label": "12_words",
        },
        {
            "text": "Machine learning algorithms learn patterns from training data to make predictions on unseen examples using statistical methods and optimization techniques",
            "label": "sentence",
        },
        {
            "text": "Neural networks are computational models inspired by biological neural systems. They consist of layers of interconnected nodes that process information through weighted connections. Deep learning extends this by using many hidden layers to learn hierarchical representations of data. Training involves backpropagation of gradients through the network to minimize a loss function.",
            "label": "paragraph",
        },
        {
            "text": "The history of artificial intelligence stretches back to ancient myths about artificial beings. Modern AI began in the 1950s when researchers started exploring whether machines could think. Early approaches focused on symbolic reasoning and logic-based systems. The field experienced cycles of optimism and disappointment known as AI winters. The resurgence of neural networks in the 2010s, powered by large datasets and GPU computing, led to breakthrough results in image recognition, natural language processing, and game playing. Today, large language models represent the frontier of AI capability, demonstrating emergent abilities that were not explicitly programmed.",
            "label": "multi_paragraph",
        },
    ]
    engine = h.index(docs)
    bits = h.doc_bits(engine, len(docs))
    for i, b in enumerate(bits):
        ntok = h.tokenizer.count_tokens(docs[i]["text"])
        b["label"] = docs[i]["label"]
        b["tokens"] = ntok
        b["polarity"] = b["pos"] / b["active"] if b["active"] else 0.5
        b["regime"] = "sub-ngram" if b["tokens"] < ngram else "supra-ngram"

    # --- invariant checks --------------------------------------------------
    all_nonzero = all(b["active"] > 0 for b in bits)
    no_saturation = all(b["density"] < 1.0 for b in bits)
    max_polarity_dev = max(abs(b["polarity"] - 0.5) for b in bits)
    polarity_ok = max_polarity_dev < 0.10

    # --- density volatility ------------------------------------------------
    all_dens = np.array([b["density"] for b in bits])
    dens_cv = float(all_dens.std() / all_dens.mean()) if all_dens.mean() > 0 else 0.0
    dens_range = float(all_dens.max() - all_dens.min())
    dens_lm = l_moments(all_dens)
    dens_cv_mad = (
        mad(all_dens) / abs(float(np.median(all_dens)))
        if np.median(all_dens) > 0
        else 0.0
    )

    # --- normalization distortion ------------------------------------------
    lit_uni = engine.db._get_lit_uni()
    if len(lit_uni) > 0 and lit_uni.sum() > 0:
        raw_lits = lit_uni[: len(bits)]
    else:
        raw_lits = np.array([float(b["active"]) for b in bits])
    norm_ratio = (
        float(np.sqrt(raw_lits.max()) / np.sqrt(raw_lits.min()))
        if raw_lits.min() > 0
        else float("inf")
    )

    # --- Per-document entropy (ternary: d+ dims positive, d- dims negative, d0 zero) ---
    for b in bits:
        d = b["density"]  # fraction of non-zero dims
        pol = b["polarity"]  # fraction of active dims that are positive
        p_pos = d * pol  # P(dim = +1)
        p_neg = d * (1 - pol)  # P(dim = -1)
        p_zero = 1 - d  # P(dim = 0)
        # Per-dimension entropy
        terms = []
        for p in [p_pos, p_neg, p_zero]:
            if p > 1e-12:
                terms.append(-p * math.log2(p))
        H_dim = sum(terms)
        b["H_dim"] = H_dim
        b["H_total"] = H_dim * dims
        b["bits_per_token"] = b["H_total"] / max(b["tokens"], 1)

    # --- Entropy rate: marginal entropy per additional token ---
    sorted_bits = sorted(bits, key=lambda b: b["tokens"])
    entropy_rates = []
    for i in range(1, len(sorted_bits)):
        prev, curr = sorted_bits[i - 1], sorted_bits[i]
        dt = curr["tokens"] - prev["tokens"]
        dH = curr["H_total"] - prev["H_total"]
        if dt > 0:
            entropy_rates.append(
                {
                    "from": prev["label"],
                    "to": curr["label"],
                    "dt": dt,
                    "dH": dH,
                    "rate": dH / dt,
                }
            )

    # --- Asymptotic entropy rate (stable regime: longest docs) ---
    supra_bits = [b for b in sorted_bits if b["regime"] == "supra-ngram"]
    asymptotic_rate = 0.0
    if len(supra_bits) >= 2:
        # Rate between two longest supra-ngram docs
        last_two = supra_bits[-2:]
        dt = last_two[1]["tokens"] - last_two[0]["tokens"]
        dH = last_two[1]["H_total"] - last_two[0]["H_total"]
        asymptotic_rate = dH / dt if dt > 0 else 0.0

    # --- Phase transition characterization ---
    sub = [b for b in bits if b["regime"] == "sub-ngram"]
    supra = [b for b in bits if b["regime"] == "supra-ngram"]
    sub_dens = np.array([b["density"] for b in sub]) if sub else np.array([])
    supra_dens = np.array([b["density"] for b in supra]) if supra else np.array([])
    sub_H = np.array([b["H_total"] for b in sub]) if sub else np.array([])
    supra_H = np.array([b["H_total"] for b in supra]) if supra else np.array([])

    regime_gap = 0.0
    if len(sub_dens) and len(supra_dens):
        regime_gap = abs(float(sub_dens.mean()) - float(supra_dens.mean()))

    supra_cv = 0.0
    if len(supra_dens) >= 2 and supra_dens.mean() > 0:
        supra_cv = float(supra_dens.std() / supra_dens.mean())

    # Entropy CV in supra regime
    supra_H_cv = 0.0
    if len(supra_H) >= 2 and supra_H.mean() > 0:
        supra_H_cv = float(supra_H.std() / supra_H.mean())

    max_d = max(b["density"] for b in bits)
    tokens = np.array([b["tokens"] for b in bits], dtype=float)
    r_val = pearson_r(tokens, all_dens)

    # Maximum entropy distortion between any two docs
    H_totals = np.array([b["H_total"] for b in bits])
    H_mean = float(H_totals.mean())
    max_H_distortion = (
        float((H_totals.max() - H_totals.min()) / H_mean) if H_mean > 0 else 0.0
    )

    # Transition zone boundaries
    transition_lo = (
        min(b["tokens"] for b in bits if b["regime"] == "supra-ngram")
        if supra
        else ngram
    )
    transition_hi = transition_lo
    # Find where density stabilizes (first supra doc within 10% of last supra doc's density)
    if len(supra) >= 2:
        stable_dens = supra[-1]["density"]
        for b in supra:
            if abs(b["density"] - stable_dens) / max(stable_dens, 1e-6) < 0.10:
                transition_hi = b["tokens"]
                break
        else:
            transition_hi = supra[-1]["tokens"]

    # --- Structural features ---
    features = []
    if max_H_distortion > 0.20:
        features.append(
            f"Entropy distortion: {max_H_distortion:.2f} (max|H_i - H_j|/H_mean). "
            f"The scoring function operates on representations with unequal information content."
        )
    if supra_cv > 0.3:
        features.append(
            f"Supra-ngram density variance: CV={supra_cv:.2f}. "
            f"The m_cap mechanism produces density CV={supra_cv:.2f} in the post-transition regime "
            f"(entropy CV={supra_H_cv:.2f})."
        )
    # Phase transition exists if density spikes then drops
    if len(supra) >= 2:
        peak_dens = max(b["density"] for b in supra)
        stable_dens = supra[-1]["density"]
        if peak_dens > stable_dens * 1.5:
            peak_label = max(supra, key=lambda b: b["density"])["label"]
            features.append(
                f"Phase transition at ~{transition_lo} tokens: density peaks at "
                f"{peak_dens:.3f} ({peak_label}) then converges to ~{stable_dens:.3f}. "
                f"Documents near the n-gram boundary exhaust the m_cap budget."
            )

    O = io.StringIO()
    w = O.write
    w(banner("ENCODING ENTROPY RATE ANALYSIS"))
    w(f"\n  HDC dims: {dims}     N-gram: {ngram}     Documents: {len(docs)}\n")

    w(section("PER-DOCUMENT ENCODING"))
    w(
        f"  {'label':>24s}  {'tokens':>6s}  {'regime':>12s}  {'active':>6s}  {'density':>8s}  {'H_total':>8s}  {'bits/tok':>8s}  {'polarity':>9s}  bar\n  {sep(n=120)}\n"
    )
    for i, b in enumerate(bits):
        w(
            f"  {b['label']:>24s}  {b['tokens']:6d}  {b['regime']:>12s}  {b['active']:6d}  {b['density']:8.4f}  {b['H_total']:8.0f}  {b['bits_per_token']:8.1f}  {b['polarity']:9.4f}  {hbar_abs(b['density'], max_d, 20)}\n"
        )

    w(section("ENTROPY PROFILE"))
    w(
        "  Per-dimension entropy H(d) for ternary vector with density d and balanced polarity:\n"
    )
    w("  H_dim = -p+ log2(p+) - p- log2(p-) - p0 log2(p0)\n\n")
    w(
        f"  {'label':>24s}  {'tokens':>6s}  {'density':>8s}  {'H_dim':>8s}  {'H_total':>8s}  bar\n  {sep(n=80)}\n"
    )
    max_H = max(b["H_total"] for b in bits)
    for b in sorted_bits:
        w(
            f"  {b['label']:>24s}  {b['tokens']:6d}  {b['density']:8.4f}  {b['H_dim']:8.4f}  {b['H_total']:8.0f}  {hbar_abs(b['H_total'], max_H, 20)}\n"
        )

    w(
        f"\n  MARGINAL ENTROPY RATE (dH/d_tokens between consecutive documents)\n  {sep(n=70)}\n"
    )
    w(
        f"  {'from':>24s}  {'to':>24s}  {'dt':>4s}  {'dH':>8s}  {'rate':>10s}\n  {sep(n=80)}\n"
    )
    for er in entropy_rates:
        w(
            f"  {er['from']:>24s}  {er['to']:>24s}  {er['dt']:4d}  {er['dH']:+8.0f}  {er['rate']:+10.2f} bits/tok\n"
        )
    w(
        f"\n  Asymptotic entropy rate (stable regime): {asymptotic_rate:+.2f} bits/token\n"
    )
    if abs(asymptotic_rate) < 1.0:
        w("  The encoding is near capacity — additional input tokens produce\n")
        w("  diminishing information gains in the representation.\n")

    w(section("INVARIANT CHECKS"))
    w(f"  {'check':>30s}  {'result':>10s}  detail\n  {sep(n=80)}\n")
    w(
        f"  {'All docs non-zero':>30s}  {'PASS' if all_nonzero else 'FAIL':>10s}  min_active={min(b['active'] for b in bits)}\n"
    )
    w(
        f"  {'No saturation (< 1.0)':>30s}  {'PASS' if no_saturation else 'FAIL':>10s}  max_density={max_d:.4f}\n"
    )
    w(
        f"  {'Polarity balanced (±10%)':>30s}  {'PASS' if polarity_ok else 'FAIL':>10s}  max_dev={max_polarity_dev:.4f}\n"
    )

    w(section("NORMALIZATION ISO-ENTROPY"))
    w("  The scoring function assumes vectors are comparable.\n")
    w(
        f"  Maximum entropy distortion: {max_H_distortion:.4f}  ({max_H_distortion * 100:.1f}% of H_mean)\n\n"
    )
    w(f"  Corpus median lit:   {float(np.median(raw_lits)):.0f}\n")
    w(f"  Norm denom ratio:    {norm_ratio:.2f}x\n")
    w(
        f"  lit_uni range:       [{float(raw_lits.min()):.1f}, {float(raw_lits.max()):.1f}]\n"
    )
    w(f"  Density CV:          {dens_cv:.3f}\n")
    w(f"  Density CV (MAD):    {dens_cv_mad:.3f}\n")
    w(
        f"  Entropy CV:          {float(H_totals.std() / H_totals.mean()) if H_totals.mean() > 0 else 0:.3f}\n"
    )
    w(
        f"  Density range:       [{float(all_dens.min()):.4f}, {float(all_dens.max()):.4f}]\n"
    )
    w(f"\n  Density L-moments:   {fmt_lm(dens_lm)}\n")
    H_lm = l_moments(H_totals)
    w(f"  Entropy L-moments:   {fmt_lm(H_lm)}\n")

    w(section("REGIME TRANSITION"))
    w(
        f"  Sub-ngram (tokens < {ngram}):   pure unigram encoding, no n-gram contribution\n"
    )
    w(
        f"  Supra-ngram (tokens >= {ngram}): unigram + m_cap-limited n-grams + promotion\n\n"
    )
    if len(sub_dens):
        w(
            f"  Sub-ngram:    n={len(sub)}  density mean={float(sub_dens.mean()):.4f}  entropy mean={float(sub_H.mean()):.0f}\n"
        )
    if len(supra_dens):
        w(
            f"  Supra-ngram:  n={len(supra)}  density mean={float(supra_dens.mean()):.4f}  entropy mean={float(supra_H.mean()):.0f}  density CV={supra_cv:.3f}\n"
        )
    if regime_gap > 0:
        w(f"  Density gap:  {regime_gap:.4f}\n")
    w(f"  Transition zone: ~{transition_lo}-{transition_hi} tokens\n")

    w("\n  TOKEN COUNT vs DENSITY\n")
    w(
        ascii_scatter(
            xs=[b["tokens"] for b in bits],
            ys=[b["density"] for b in bits],
            labels=[b["label"] for b in bits],
            x_label="token count",
        )
    )

    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    w(
        f"\n  Pearson r (tokens -> density): {r_val:+.4f}  (non-monotonic expected due to phase transition)\n"
    )

    healthy = all_nonzero and no_saturation and polarity_ok
    h.cleanup(engine)
    return ProbeResult(
        name="density",
        ok=healthy,
        primary_stat=f"H_rate={asymptotic_rate:+.1f}bits/tok  CV={dens_cv:.3f}",
        summary=f"norm={norm_ratio:.2f}x  CV={dens_cv:.3f}  H_distort={max_H_distortion:.3f}  supra_cv={supra_cv:.3f}",
        details=O.getvalue(),
        metrics={
            "all_nonzero": all_nonzero,
            "no_saturation": no_saturation,
            "max_density": max_d,
            "max_polarity_deviation": max_polarity_dev,
            "norm_distortion_ratio": norm_ratio,
            "density_cv": dens_cv,
            "density_cv_mad": dens_cv_mad,
            "density_lmoments": dens_lm,
            "supra_ngram_cv": supra_cv,
            "regime_gap": regime_gap,
            "asymptotic_entropy_rate": asymptotic_rate,
            "max_entropy_distortion": max_H_distortion,
            "supra_entropy_cv": supra_H_cv,
            "transition_tokens": transition_lo,
            "entropies": {b["label"]: b["H_total"] for b in bits},
            "densities": {b["label"]: b["density"] for b in bits},
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_sensitivity(h):
    """Query vector stability under incremental word additions.  Primary: mean Hamming delta

    Each step adds one word, but reports both word count and actual token count
    so the encoder's true granularity is visible.
    """
    t0 = time.time()
    dims = h.config.hdc_dimensions
    words = [
        "quantum",
        "field",
        "theory",
        "predicts",
        "particle",
        "interactions",
        "using",
        "gauge",
        "symmetry",
        "principles",
    ]

    # --- token counts per word and cumulative ---
    word_ntok = []
    for wd in words:
        word_ntok.append(len(h.tokenizer.bulk_tokenize([wd])[0]))
    cum_ntok = []
    for i in range(1, len(words) + 1):
        cum_ntok.append(h.tokenizer.count_tokens(" ".join(words[:i])))

    base_docs = [
        {"text": " ".join(words), "label": "full"},
        {
            "text": "Classical mechanics describes motion of macroscopic objects",
            "label": "filler_1",
        },
        {
            "text": "Thermodynamics studies energy transfer between systems",
            "label": "filler_2",
        },
    ]
    engine = h.index(base_docs)
    bitmaps = [h.encode(engine, " ".join(words[:i])) for i in range(1, len(words) + 1)]
    deltas = [
        h.hamming(bitmaps[i - 1], bitmaps[i], dims) for i in range(1, len(bitmaps))
    ]
    steps = []
    for i, bm in enumerate(bitmaps):
        p, n, a = h.qbits(bm["pos"], bm["neg"])
        steps.append(
            {
                "w": i + 1,
                "ntok": cum_ntok[i],
                "word_ntok": word_ntok[i],
                "pos": p,
                "neg": n,
                "active": a,
                "density": a / dims,
                "delta": deltas[i - 1] if i > 0 else 0.0,
            }
        )
    da = np.array(deltas)
    mean_d = float(da.mean())
    cv = float(da.std() / da.mean()) if da.mean() > 0 else float("inf")
    cv_mad = mad(da) / abs(float(np.median(da))) if np.median(da) > 0 else float("inf")
    da_lm = l_moments(da)
    all_changed = all(d > 0 for d in deltas)
    max_dens = max(s["density"] for s in steps)
    max_delt = max(deltas) if deltas else 1.0
    O = io.StringIO()
    w = O.write
    w(banner("QUERY SENSITIVITY ANALYSIS"))
    w(f"\n  HDC dims: {dims}\n  Words: {' -> '.join(words)}\n")
    w(section("INCREMENTAL ENCODING"))
    w(
        f"  {'#':>3s}  {'active':>6s}  {'density':>8s}  {'ntok':>5s}  {'w_tok':>5s}  {'delta_H':>10s}  {'dens bar':<20s}  {'delta bar':<20s}  word added\n  {sep(n=120)}\n"
    )
    for s in steps:
        dbar = hbar_abs(s["delta"], max_delt, 20) if s["w"] > 1 else " " * 20
        delta_str = f"{s['delta']:10.6f}" if s["w"] > 1 else f"{'---':>10s}"
        w(
            f"  {s['w']:3d}  {s['active']:6d}  {s['density']:8.4f}  {s['ntok']:5d}  {s['word_ntok']:5d}  {delta_str}  {hbar_abs(s['density'], max_dens, 20)}  {dbar}  +{words[s['w'] - 1]}\n"
        )
    w(f"\n  HAMMING DELTA STATISTICS\n  {sep(n=50)}\n")
    w(f"  Mean:    {mean_d:.6f}\n  Median:  {float(np.median(da)):.6f}\n")
    w(f"  Std:     {float(da.std()):.6f}\n  MAD:     {mad(da):.6f}\n")
    w(f"  CV:      {cv:.4f}\n  CV(MAD): {cv_mad:.4f}\n")
    w(f"  Range:   [{float(da.min()):.6f}, {float(da.max()):.6f}]\n")
    w(f"  L-mom:   {fmt_lm(da_lm)}\n")
    w(f"  Spark:   {sparkline(deltas, 30)}\n")
    w(f"\n  All additions produced change: {'yes' if all_changed else 'no'}\n")
    w(f"  Mean Hamming delta per word:   {mean_d:.6f}\n")
    w(
        f"  CV: {cv:.4f}  ({'stable' if cv < 0.5 else 'moderate' if cv < 1.0 else 'high variance'})\n"
    )
    w("\n  NOTE: ntok = cumulative token count, w_tok = tokens added by this word.\n")
    w("  The encoder operates on tokens, not words. Multi-subword words\n")
    w("  contribute more tokens per step than single-token words.\n")
    h.cleanup(engine)
    return ProbeResult(
        name="sensitivity",
        ok=all_changed and cv < 2.0,
        primary_stat=f"mu_delta={mean_d:.6f}  CV={cv:.4f}",
        summary=f"mu_delta={mean_d:.6f}  CV={cv:.4f}  all_changed={'yes' if all_changed else 'no'}",
        details=O.getvalue(),
        metrics={
            "mean_delta": mean_d,
            "median_delta": float(np.median(da)),
            "cv": cv,
            "cv_mad": cv_mad,
            "delta_lmoments": da_lm,
            "all_changed": all_changed,
            "cum_tokens": cum_ntok,
        },
        duration=time.time() - t0,
    )


def probe_retrieval(h):
    """End-to-end ranking quality.  Primary: Cohen's d (on-topic vs off-topic)"""
    t0 = time.time()
    dims = h.config.hdc_dimensions
    query = "how do transformers use self-attention mechanisms for sequence modeling"
    on = [
        {
            "text": "The transformer architecture uses multi-head self-attention to compute pairwise relationships between all positions in an input sequence, enabling parallel processing of sequential data",
            "label": "transformers_exact",
            "topic": "on",
        },
        {
            "text": "Self-attention mechanisms allow each token in a sequence to attend to every other token, computing weighted sums based on learned query, key, and value projections",
            "label": "self_attention",
            "topic": "on",
        },
        {
            "text": "Positional encoding in transformers provides sequence order information since self-attention is inherently permutation-invariant",
            "label": "positional",
            "topic": "on",
        },
        {
            "text": "BERT and GPT represent two paradigms of transformer pretraining: masked language modeling for bidirectional understanding and autoregressive generation",
            "label": "bert_gpt",
            "topic": "on",
        },
    ]
    off = [
        {
            "text": "Recurrent neural networks process sequences through hidden state propagation but suffer from vanishing gradients on long sequences",
            "label": "rnns",
            "topic": "related_ml",
        },
        {
            "text": "Convolutional neural networks apply learned filters across spatial dimensions to detect features in images",
            "label": "cnns",
            "topic": "related_ml",
        },
        {
            "text": "Gradient boosting combines many weak decision tree learners into a strong ensemble",
            "label": "gbm",
            "topic": "other_ml",
        },
        {
            "text": "The French Revolution of 1789 transformed European politics by overthrowing the monarchy",
            "label": "french_rev",
            "topic": "history",
        },
        {
            "text": "Photosynthesis in C4 plants concentrates carbon dioxide in bundle sheath cells",
            "label": "c4_plants",
            "topic": "biology",
        },
        {
            "text": "Volcanic eruptions release magma, gases, and tephra through fissures in the Earth's crust",
            "label": "volcanoes",
            "topic": "geology",
        },
    ]
    docs = on + off
    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, agree, disagree, _, lit = h.scores(
        engine,
        bm["pos"],
        bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )
    qp, qn, qa = h.qbits(bm["pos"], bm["neg"])
    on_i = list(range(len(on)))
    off_i = list(range(len(on), len(docs)))
    d_val = cohen_d(s_geo[on_i], s_geo[off_i])
    rd_val = robust_d(s_geo[on_i], s_geo[off_i])
    lm_on = l_moments(s_geo[on_i])
    lm_off = l_moments(s_geo[off_i])
    ranked = np.argsort(-s_geo)
    prec = {}
    for k in [1, 2, 4, len(on)]:
        top_k = ranked[:k]
        correct = sum(1 for i in top_k if docs[i]["topic"] == "on")
        prec[k] = correct / k
    run = 0
    for idx in ranked:
        if docs[idx]["topic"] != "on":
            break
        run += 1
    lo, hi = float(s_geo.min()), float(s_geo.max())
    O = io.StringIO()
    w = O.write
    w(banner("RETRIEVAL PIPELINE ANALYSIS"))
    w(
        f'\n  QUERY: "{query}"\n  Corpus: {len(on)} on-topic + {len(off)} off-topic\n  QUERY VECTOR: active={qa} ({pct(qa, dims)})\n'
    )
    w(section("RANKED RESULTS"))
    w(
        f"  {'rank':>4s}  {'label':>20s}  {'topic':>14s}  {'geo':>8s}  {'agree':>6s}  {'disag':>6s}  {'d_lit':>6s}  bar\n  {sep(n=110)}\n"
    )
    for rank, idx in enumerate(ranked):
        d = docs[idx]
        tag = " <-" if d["topic"] == "on" else ""
        w(
            f"  {rank + 1:4d}  {d['label']:>20s}  {d['topic']:>14s}  {s_geo[idx]:+8.4f}  {agree[idx]:6.0f}  {disagree[idx]:6.0f}  {lit[idx]:6.0f}  {hbar(s_geo[idx], lo, hi)}{tag}\n"
        )
    w(f"\n  SCORE BY TOPIC GROUP\n  {sep(n=70)}\n")
    tg = {}
    for i, d in enumerate(docs):
        tg.setdefault(d["topic"], []).append(i)
    for topic in sorted(tg, key=lambda t: -float(s_geo[tg[t]].mean())):
        idx = tg[topic]
        v = s_geo[idx]
        w(
            f"  {topic:>14s}: mu={float(v.mean()):+.4f}  med={float(np.median(v)):+.4f}  [{float(v.min()):+.4f}, {float(v.max()):+.4f}]  n={len(idx)}  {sparkline(v.tolist(), 12)}\n"
        )
    w(f"\n  PRECISION\n  {sep(n=40)}\n")
    for k, p in sorted(prec.items()):
        w(f"  P@{k}: {p:.2f}  {hbar_abs(p, 1.0, 20)}\n")
    w(f"\n  Correct run before first miss: {run}/{len(on)}\n")
    w(f"  Cohen's d (on vs off):         {d_val:+.4f}  ({effect_label(d_val)})\n")
    w(f"  Robust d  (on vs off):         {rd_val:+.4f}  ({effect_label(rd_val)})\n")
    w(f"\n  L-MOMENTS\n  {sep(n=60)}\n")
    w(f"  {'on-topic':>14s}:  {fmt_lm(lm_on)}\n")
    w(f"  {'off-topic':>14s}:  {fmt_lm(lm_off)}\n")
    top = int(ranked[0])
    aratio = (
        float(agree[top] / (agree[top] + disagree[top]))
        if (agree[top] + disagree[top]) > 0
        else 0
    )
    w(
        f"\n  TOP RESULT\n  {sep(n=50)}\n  Document:    {docs[top]['label']}\n  Score:       {s_geo[top]:+.4f}\n  Agree ratio: {aratio:.4f}\n"
    )

    h.cleanup(engine)
    return ProbeResult(
        name="retrieval",
        ok=d_val > 0.8,
        primary_stat=f"d={d_val:+.4f} ({effect_label(d_val)})",
        summary=f"d={d_val:+.4f}  rd={rd_val:+.4f}  P@1={prec[1]:.2f}  P@4={prec[4]:.2f}  run={run}/{len(on)}",
        details=O.getvalue(),
        metrics={
            "cohen_d": d_val,
            "robust_d": rd_val,
            "precision": prec,
            "correct_run": run,
            "lm_on": lm_on,
            "lm_off": lm_off,
        },
        duration=time.time() - t0,
    )


# Registry & Runner

def probe_paraphrase(h):
    """Paraphrase robustness: does the encoding recognise meaning-preserving reformulations?

    Builds pairs of documents that express the same idea with different wording.
    Measures how much score survives paraphrase vs the exact-match baseline.

    Primary: median paraphrase retention (paraphrase_score / exact_score).
    Health: median retention > 0.3 (paraphrases score at least 30% of exact).
    """
    t0 = time.time()
    query = "machine learning algorithms for image classification"

    pairs = [
        {
            "exact": "Machine learning algorithms for image classification use convolutional neural networks to extract features from pixel data and assign category labels",
            "para": "Categorizing pictures with artificial intelligence relies on layered neural architectures that detect visual patterns and map them to predefined classes",
            "label": "cnn_classification",
        },
        {
            "exact": "Gradient descent optimization minimizes the loss function by iteratively adjusting model parameters in the direction of steepest decrease",
            "para": "The training procedure repeatedly tweaks network weights to reduce prediction errors by following the slope of the error surface downhill",
            "label": "gradient_descent",
        },
        {
            "exact": "Recurrent neural networks process sequential data by maintaining hidden state vectors that capture temporal dependencies across time steps",
            "para": "Networks designed for ordered data keep an internal memory that tracks patterns evolving over successive elements in the sequence",
            "label": "rnn_sequential",
        },
        {
            "exact": "Transfer learning adapts a pretrained model to a new task by fine-tuning the final layers while keeping earlier feature extractors frozen",
            "para": "Reusing a previously trained network for a different problem involves retraining only the output layers and preserving the learned low-level representations",
            "label": "transfer_learning",
        },
        {
            "exact": "Data augmentation increases training set diversity by applying random transformations like rotation flipping and color jittering to existing images",
            "para": "Expanding the variety of training examples through geometric and photometric perturbations of the original samples helps prevent overfitting",
            "label": "augmentation",
        },
    ]

    filler_docs = [
        {"text": "The Roman Empire expanded through military conquest and administrative organization across the Mediterranean basin", "label": "filler_1"},
        {"text": "Photosynthesis converts carbon dioxide and water into glucose using light energy captured by chlorophyll molecules", "label": "filler_2"},
        {"text": "Medieval castle architecture employed thick stone walls moats and drawbridges for defensive purposes", "label": "filler_3"},
    ]

    # Build corpus: exact + paraphrase + filler for each pair
    docs = []
    for p in pairs:
        docs.append({"text": p["exact"], "label": f"{p['label']}_exact"})
        docs.append({"text": p["para"], "label": f"{p['label']}_para"})
    docs.extend(filler_docs)

    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, agree, disagree, _, lit = h.scores(
        engine, bm["pos"], bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )

    # Collect per-pair metrics
    pair_metrics = []
    for pi, p in enumerate(pairs):
        ei = pi * 2       # exact doc index
        pi_idx = pi * 2 + 1  # para doc index
        exact_s = float(s_geo[ei])
        para_s = float(s_geo[pi_idx])
        retention = para_s / exact_s if abs(exact_s) > 1e-6 else 0.0
        pair_metrics.append({
            "label": p["label"],
            "exact_score": exact_s,
            "para_score": para_s,
            "retention": retention,
            "gap": exact_s - para_s,
        })

    filler_start = len(pairs) * 2
    filler_scores = s_geo[filler_start:]

    retentions = np.array([pm["retention"] for pm in pair_metrics])
    exact_scores = np.array([pm["exact_score"] for pm in pair_metrics])
    para_scores = np.array([pm["para_score"] for pm in pair_metrics])
    gaps = np.array([pm["gap"] for pm in pair_metrics])

    med_ret = float(np.median(retentions))
    med_gap = float(np.median(gaps))
    lm_ret = l_moments(retentions)
    lm_exact = l_moments(exact_scores)
    lm_para = l_moments(para_scores)

    # How many paraphrases outscore best filler?
    best_filler = float(filler_scores.max()) if len(filler_scores) > 0 else 0.0
    para_above_filler = int((para_scores > best_filler).sum())

    lo = float(s_geo.min())
    hi = float(s_geo.max())

    O = io.StringIO()
    w = O.write
    w(banner("PARAPHRASE ROBUSTNESS ANALYSIS"))
    w(f'\n  QUERY: "{query}"\n')
    w(f"  Pairs: {len(pairs)}     Filler: {len(filler_docs)}\n")

    w(section("PER-PAIR SCORES"))
    w(f"  {'label':>20s}  {'exact':>8s}  {'para':>8s}  {'retain':>7s}  {'gap':>8s}  exact bar         |para bar\n")
    w(f"  {sep(n=100)}\n")
    for pm in pair_metrics:
        eb = hbar(pm["exact_score"], lo, hi, 18)
        pb = hbar(pm["para_score"], lo, hi, 18)
        w(f"  {pm['label']:>20s}  {pm['exact_score']:+8.4f}  {pm['para_score']:+8.4f}  {pm['retention']:7.3f}  {pm['gap']:+8.4f}  {eb}|{pb}\n")

    w(section("FILLER BASELINE"))
    for i, fd in enumerate(filler_docs):
        fi = filler_start + i
        w(f"  {fd['label']:>20s}  {s_geo[fi]:+8.4f}  {hbar(s_geo[fi], lo, hi, 18)}\n")
    w(f"\n  Best filler: {best_filler:+.4f}\n")
    w(f"  Paraphrases above filler: {para_above_filler}/{len(pairs)}\n")

    w(section("RETENTION STATISTICS"))
    w(f"  Mean retention:    {float(retentions.mean()):.4f}\n")
    w(f"  Median retention:  {med_ret:.4f}\n")
    w(f"  MAD:               {mad(retentions):.4f}\n")
    w(f"  Range:             [{float(retentions.min()):.4f}, {float(retentions.max()):.4f}]\n")
    w(f"  L-moments:         {fmt_lm(lm_ret)}\n")
    w(f"\n  Exact scores:      med={float(np.median(exact_scores)):+.4f}  L-mom: {fmt_lm(lm_exact)}\n")
    w(f"  Para scores:       med={float(np.median(para_scores)):+.4f}  L-mom: {fmt_lm(lm_para)}\n")
    w(f"  Median gap:        {med_gap:+.4f}\n")

    features = []
    if med_ret < 0.3:
        features.append(f"Paraphrase retention below 30% (median={med_ret:.3f}). The encoding is highly sensitive to surface form.")
    if para_above_filler < len(pairs):
        features.append(f"Only {para_above_filler}/{len(pairs)} paraphrases outscore the best filler ({best_filler:+.4f}).")
    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="paraphrase",
        ok=med_ret > 0.3,
        primary_stat=f"med_ret={med_ret:.3f}  above_filler={para_above_filler}/{len(pairs)}",
        summary=f"med_ret={med_ret:.3f}  med_gap={med_gap:+.4f}  above_filler={para_above_filler}/{len(pairs)}",
        details=O.getvalue(),
        metrics={
            "median_retention": med_ret,
            "mean_retention": float(retentions.mean()),
            "median_gap": med_gap,
            "para_above_filler": para_above_filler,
            "best_filler": best_filler,
            "lm_retention": lm_ret,
            "pair_metrics": pair_metrics,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_query_length(h):
    """Retrieval quality as a function of query length.

    Measures P@1, target-filler gap, and score at each query length from
    2 tokens up to 16 tokens.  Maps the practical minimum query length.

    Primary: minimum token count where P@1 = 1.0.
    Health: P@1 = 1.0 achieved by 8 tokens.
    """
    t0 = time.time()

    # Target document covers the full topic
    target = "Transformer neural networks use multi-head self-attention mechanisms to model long-range dependencies in sequential data for natural language processing tasks"
    filler_docs = [
        {"text": "The migration patterns of Arctic terns span from pole to pole covering over forty thousand kilometers annually", "label": "filler_1"},
        {"text": "Baroque music composers like Bach and Vivaldi developed complex polyphonic structures using counterpoint techniques", "label": "filler_2"},
        {"text": "Volcanic eruptions release magma ash and gases from deep within the Earth through fissures in the tectonic plates", "label": "filler_3"},
        {"text": "Ancient Egyptian hieroglyphics were deciphered using the Rosetta Stone which contained the same text in three scripts", "label": "filler_4"},
        {"text": "The process of fermentation converts sugars into alcohol and carbon dioxide through the action of yeast enzymes", "label": "filler_5"},
    ]
    docs = [{"text": target, "label": "target"}] + filler_docs

    engine = h.index(docs)

    # Build incrementally longer queries from related words
    words = ["transformer", "attention", "mechanism", "sequence", "modeling",
             "neural", "network", "language", "processing", "multi-head",
             "self-attention", "dependencies", "long-range", "data", "natural", "tasks"]

    results = []
    for nw in range(1, len(words) + 1):
        q = " ".join(words[:nw])
        tids = h.tokenizer.bulk_tokenize([q])[0]
        ntoks = len(tids)
        bm = h.encode(engine, q)
        _, sg, _, _, _, _ = h.scores(
            engine, bm["pos"], bm["neg"],
            q_lit_uni=float(bm["lit_uni"][0]),
            q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
        )
        target_score = float(sg[0])
        filler_max = float(sg[1:].max())
        gap = target_score - filler_max
        p_at_1 = 1.0 if target_score > filler_max else 0.0
        results.append({
            "n_words": nw,
            "n_tokens": ntoks,
            "query": q,
            "target_score": target_score,
            "filler_max": filler_max,
            "gap": gap,
            "p_at_1": p_at_1,
        })

    # Find first P@1 = 1.0
    first_hit = len(words) + 1
    for r in results:
        if r["p_at_1"] == 1.0:
            first_hit = r["n_tokens"]
            break

    gaps = np.array([r["gap"] for r in results])
    target_scores = np.array([r["target_score"] for r in results])
    lm_gaps = l_moments(gaps)

    O = io.StringIO()
    w = O.write
    w(banner("QUERY LENGTH RETRIEVAL CURVE"))
    w(f"\n  Target: \"{target[:70]}...\"\n")
    w(f"  Filler: {len(filler_docs)} unrelated docs\n")
    w(f"  Query words: {' → '.join(words[:8])}...\n")

    w(section("RETRIEVAL BY QUERY LENGTH"))
    w(f"  {'words':>5s}  {'tokens':>6s}  {'P@1':>4s}  {'target':>8s}  {'filler':>8s}  {'gap':>8s}  gap bar\n")
    w(f"  {sep(n=80)}\n")
    max_gap = max(abs(r["gap"]) for r in results)
    for r in results:
        p1 = "✓" if r["p_at_1"] == 1.0 else "✗"
        w(f"  {r['n_words']:5d}  {r['n_tokens']:6d}  {p1:>4s}  {r['target_score']:+8.4f}  {r['filler_max']:+8.4f}  {r['gap']:+8.4f}  {hbar_abs(r['gap'], max(max_gap, 0.01), 20)}\n")

    w(f"\n  First P@1=1.0 at {first_hit} tokens\n")
    w(f"  Gap L-moments: {fmt_lm(lm_gaps)}\n")

    # Scatter: tokens → gap
    w("\n  QUERY LENGTH vs TARGET-FILLER GAP\n")
    w(ascii_scatter(
        xs=[r["n_tokens"] for r in results],
        ys=[r["gap"] for r in results],
        labels=[f"{r['n_words']}w" for r in results],
        x_label="query tokens",
    ))

    features = []
    if first_hit > 8:
        features.append(f"P@1 not achieved until {first_hit} tokens. Short queries may fail.")
    negative_gaps = [r for r in results if r["gap"] < 0]
    if negative_gaps:
        features.append(f"{len(negative_gaps)} query lengths have negative gap (target outscored by filler).")
    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="query_length",
        ok=first_hit <= 8,
        primary_stat=f"P@1_at={first_hit}tok  med_gap={float(np.median(gaps)):+.4f}",
        summary=f"P@1_at={first_hit}tok  gaps=[{float(gaps.min()):+.3f},{float(gaps.max()):+.3f}]",
        details=O.getvalue(),
        metrics={
            "first_p1_tokens": first_hit,
            "median_gap": float(np.median(gaps)),
            "lm_gaps": lm_gaps,
            "per_length": results,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_corpus_scale(h):
    """Noise separation as corpus size grows.

    Builds corpora of increasing size (same signal, more noise) and measures
    Cohen's d and robust d at each scale.  Tests whether discrimination
    degrades sub-linearly, linearly, or super-linearly with corpus size.

    Primary: d at largest scale.
    Health: d > 1.5 at 500 documents.
    """
    t0 = time.time()
    query = "quantum entanglement between photon pairs"
    rng = np.random.default_rng(77)

    signal_docs = [
        {"text": "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly affects the other regardless of distance", "label": "signal_0", "group": "signal"},
        {"text": "Photon pair generation through spontaneous parametric down-conversion creates entangled states for quantum optics experiments", "label": "signal_1", "group": "signal"},
        {"text": "Bell inequality violations demonstrate that entangled particles exhibit correlations beyond classical physics", "label": "signal_2", "group": "signal"},
    ]

    # Pre-generate a large pool of noise docs
    max_noise = 500
    noise_pool = [
        {
            "text": "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz     "), size=rng.integers(40, 180))),
            "label": f"noise_{i}",
            "group": "noise",
        }
        for i in range(max_noise)
    ]

    scales = [5, 10, 25, 50, 100, 250, 500]
    scale_results = []

    for n_noise in scales:
        docs = signal_docs + noise_pool[:n_noise]
        engine = h.index(docs)
        bm = h.encode(engine, query)
        _, sg, _, _, _, _ = h.scores(
            engine, bm["pos"], bm["neg"],
            q_lit_uni=float(bm["lit_uni"][0]),
            q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
        )
        sig_i = list(range(len(signal_docs)))
        noi_i = list(range(len(signal_docs), len(docs)))
        d_val = cohen_d(sg[sig_i], sg[noi_i])
        rd_val = robust_d(sg[sig_i], sg[noi_i])
        margin = float(sg[sig_i].min() - sg[noi_i].max())
        noise_p99 = float(np.percentile(sg[noi_i], 99))
        lm_noi = l_moments(sg[noi_i])
        scale_results.append({
            "n_noise": n_noise,
            "n_total": len(docs),
            "cohen_d": d_val,
            "robust_d": rd_val,
            "margin": margin,
            "signal_med": float(np.median(sg[sig_i])),
            "noise_med": float(np.median(sg[noi_i])),
            "noise_p99": noise_p99,
            "lm_noise": lm_noi,
        })
        h.cleanup(engine)

    final_d = scale_results[-1]["cohen_d"]
    final_rd = scale_results[-1]["robust_d"]

    # Compute degradation rate via Spearman
    log_scales = [math.log(r["n_noise"]) for r in scale_results]
    d_values = [r["cohen_d"] for r in scale_results]
    rho_scale = spearman_rho(log_scales, d_values)

    O = io.StringIO()
    w = O.write
    w(banner("CORPUS SCALE ANALYSIS"))
    w(f'\n  QUERY: "{query}"\n')
    w(f"  Signal: {len(signal_docs)} docs (fixed)     Noise: 5..{max_noise} (scaled)\n")

    w(section("DISCRIMINATION BY CORPUS SIZE"))
    w(f"  {'noise':>6s}  {'total':>6s}  {'Cohen d':>9s}  {'robust d':>9s}  {'margin':>8s}  {'noi_med':>8s}  {'noi_p99':>8s}  d bar\n")
    w(f"  {sep(n=90)}\n")
    max_d = max(abs(r["cohen_d"]) for r in scale_results)
    for r in scale_results:
        w(f"  {r['n_noise']:6d}  {r['n_total']:6d}  {r['cohen_d']:+9.4f}  {r['robust_d']:+9.4f}  {r['margin']:+8.4f}  {r['noise_med']:+8.4f}  {r['noise_p99']:+8.4f}  {hbar_abs(r['cohen_d'], max_d, 20)}\n")

    w(f"\n  Scale-d correlation (Spearman rho): {rho_scale:+.4f}\n")
    if rho_scale < -0.5:
        w("  Interpretation: d degrades significantly with corpus size.\n")
    else:
        w("  Interpretation: d is stable across corpus scale.\n")

    w("\n  NOISE DISTRIBUTION L-MOMENTS AT LARGEST SCALE\n")
    w(f"  {fmt_lm(scale_results[-1]['lm_noise'])}\n")

    # Scatter
    w("\n  CORPUS SIZE vs COHEN'S d\n")
    w(ascii_scatter(
        xs=[r["n_noise"] for r in scale_results],
        ys=[r["cohen_d"] for r in scale_results],
        labels=[f"n={r['n_noise']}" for r in scale_results],
        x_label="noise documents",
    ))

    features = []
    if final_d < 1.5:
        features.append(f"d={final_d:+.4f} at n={max_noise} — discrimination weak at scale.")
    if rho_scale < -0.7:
        features.append(f"Strong negative scale correlation (rho={rho_scale:+.4f}). Discrimination degrades rapidly with corpus growth.")
    p99_close = [r for r in scale_results if r["signal_med"] - r["noise_p99"] < 0.05]
    if p99_close:
        features.append(f"99th percentile noise approaches signal at {len(p99_close)} scale(s).")
    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="corpus_scale",
        ok=final_d > 1.5,
        primary_stat=f"d@{max_noise}={final_d:+.4f}  rho={rho_scale:+.4f}",
        summary=f"d@{max_noise}={final_d:+.4f}  rd={final_rd:+.4f}  rho_scale={rho_scale:+.4f}",
        details=O.getvalue(),
        metrics={
            "final_d": final_d,
            "final_rd": final_rd,
            "rho_scale": rho_scale,
            "per_scale": scale_results,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_adversarial(h):
    """Adversarial vocabulary: documents sharing query tokens in wrong contexts.

    Tests whether the n-gram layer protects against vocabulary-level false
    positives.  Decoy documents contain most query tokens but in semantically
    unrelated contexts.

    Primary: true target outscores all decoys.
    Health: target-decoy margin > 0.
    """
    t0 = time.time()
    query = "neural network training with backpropagation gradient updates"

    docs = [
        {
            "text": "Neural network training with backpropagation computes gradient updates by applying the chain rule to propagate error signals from output to input layers",
            "label": "true_match",
            "group": "target",
        },
        {
            "text": "The neural pathways in the brain form a biological network whose training through repetition causes gradient changes in synaptic strength through updates to neurotransmitter levels via backpropagation of chemical signals",
            "label": "neuro_decoy",
            "group": "decoy",
        },
        {
            "text": "A network of hiking trails with gradient elevation updates was built for training marathon runners using neural feedback from heart rate monitors with backpropagation of fitness data",
            "label": "hiking_decoy",
            "group": "decoy",
        },
        {
            "text": "Software updates to the corporate network training program included gradient improvements to the neural interface and backpropagation of employee feedback through management layers",
            "label": "corporate_decoy",
            "group": "decoy",
        },
        {
            "text": "The fishing net work involves training apprentices in gradient water depths using neural instincts with updates on tide backpropagation patterns",
            "label": "fishing_decoy",
            "group": "decoy",
        },
        {
            "text": "Medieval castle architecture employed thick stone walls and defensive moats to protect against invading armies",
            "label": "unrelated",
            "group": "filler",
        },
        {
            "text": "Photosynthesis converts sunlight into chemical energy within chloroplast organelles of green plants",
            "label": "unrelated_2",
            "group": "filler",
        },
    ]

    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, agree, disagree, _, lit = h.scores(
        engine, bm["pos"], bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )

    target_i = [i for i, d in enumerate(docs) if d["group"] == "target"]
    decoy_i = [i for i, d in enumerate(docs) if d["group"] == "decoy"]
    filler_i = [i for i, d in enumerate(docs) if d["group"] == "filler"]

    target_score = float(s_geo[target_i[0]])
    decoy_scores = s_geo[decoy_i]
    filler_scores = s_geo[filler_i]
    best_decoy = float(decoy_scores.max())
    target_decoy_margin = target_score - best_decoy
    target_rank = 1 + int((s_geo > target_score).sum())

    # Cohen's d: target+filler vs decoy (are decoys closer to filler or target?)
    d_decoy_filler = cohen_d(decoy_scores, filler_scores)
    rd_decoy_filler = robust_d(decoy_scores, filler_scores)
    lm_decoy = l_moments(decoy_scores)

    lo, hi = float(s_geo.min()), float(s_geo.max())

    O = io.StringIO()
    w = O.write
    w(banner("ADVERSARIAL VOCABULARY ANALYSIS"))
    w(f'\n  QUERY: "{query}"\n')
    w(f"  Target: 1     Decoys: {len(decoy_i)} (shared vocabulary, wrong context)\n")
    w(f"  Filler: {len(filler_i)} (no shared vocabulary)\n")

    w(section("RANKED SCORES"))
    w(f"  {'label':>20s}  {'group':>8s}  {'geo':>8s}  {'agree':>6s}  {'disag':>6s}  bar\n")
    w(f"  {sep(n=80)}\n")
    for i in sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True):
        d = docs[i]
        w(f"  {d['label']:>20s}  {d['group']:>8s}  {s_geo[i]:+8.4f}  {agree[i]:6.0f}  {disagree[i]:6.0f}  {hbar(s_geo[i], lo, hi)}\n")

    w(section("ADVERSARIAL RESISTANCE"))
    w(f"  Target score:          {target_score:+.4f}\n")
    w(f"  Best decoy score:      {best_decoy:+.4f}\n")
    w(f"  Target-decoy margin:   {target_decoy_margin:+.4f}\n")
    w(f"  Target rank:           {target_rank}\n")
    w(f"\n  Decoy scores:  med={float(np.median(decoy_scores)):+.4f}  MAD={mad(decoy_scores):.4f}\n")
    w(f"  Filler scores: med={float(np.median(filler_scores)):+.4f}  MAD={mad(filler_scores):.4f}\n")
    w(f"  Decoy L-moments: {fmt_lm(lm_decoy)}\n")
    w(f"\n  Decoy vs filler d:     {d_decoy_filler:+.4f} ({effect_label(d_decoy_filler)})\n")
    w(f"  Decoy vs filler rd:    {rd_decoy_filler:+.4f}\n")

    features = []
    if target_decoy_margin <= 0:
        features.append(f"Target outscored by decoy (margin={target_decoy_margin:+.4f}). N-gram layer failed to protect.")
    if d_decoy_filler > 1.0:
        features.append(f"Decoys score significantly above filler (d={d_decoy_filler:+.4f}). Shared vocabulary inflates scores.")
    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="adversarial",
        ok=target_decoy_margin > 0,
        primary_stat=f"margin={target_decoy_margin:+.4f}  rank={target_rank}",
        summary=f"margin={target_decoy_margin:+.4f}  rank={target_rank}  decoy_d={d_decoy_filler:+.4f}",
        details=O.getvalue(),
        metrics={
            "target_score": target_score,
            "best_decoy": best_decoy,
            "margin": target_decoy_margin,
            "target_rank": target_rank,
            "decoy_filler_d": d_decoy_filler,
            "decoy_filler_rd": rd_decoy_filler,
            "lm_decoy": lm_decoy,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_mmr(h):
    """MMR diversity: does MMR deduplication surface diverse relevant results?

    Builds a corpus with near-duplicate clusters and distinct relevant docs.
    Compares top-K with and without MMR to measure diversity improvement.

    Primary: diversity gain (unique clusters in MMR top-K vs raw top-K).
    Health: MMR surfaces at least 2 distinct clusters in top-4.
    """
    t0 = time.time()
    query = "how does photosynthesis convert sunlight into chemical energy"

    # Cluster A: near-duplicates about photosynthesis mechanism
    cluster_a = [
        {"text": "Photosynthesis converts sunlight into chemical energy through light-dependent reactions in the thylakoid membranes of chloroplasts", "label": "photo_a1", "cluster": "A"},
        {"text": "Photosynthesis converts sunlight to chemical energy via light-dependent reactions within the thylakoid membranes of chloroplast organelles", "label": "photo_a2", "cluster": "A"},
        {"text": "The process of photosynthesis converts sunlight into chemical energy through light dependent reactions in thylakoid membranes inside chloroplasts", "label": "photo_a3", "cluster": "A"},
    ]
    # Cluster B: about the Calvin cycle (different aspect, still relevant)
    cluster_b = [
        {"text": "The Calvin cycle fixes carbon dioxide into glucose molecules using ATP and NADPH produced during the light reactions of photosynthesis", "label": "calvin_b1", "cluster": "B"},
        {"text": "Carbon fixation through the Calvin cycle uses energy carriers from light reactions to synthesize glucose from atmospheric CO2 in photosynthetic organisms", "label": "calvin_b2", "cluster": "B"},
    ]
    # Cluster C: about chlorophyll (different aspect, still relevant)
    cluster_c = [
        {"text": "Chlorophyll molecules absorb red and blue wavelengths of light while reflecting green and transfer excited electrons to the photosynthetic reaction center", "label": "chloro_c1", "cluster": "C"},
    ]
    # Filler
    filler = [
        {"text": "The stock market experienced significant volatility during the fourth quarter due to interest rate uncertainty", "label": "filler_1", "cluster": "X"},
        {"text": "Medieval blacksmiths forged weapons and tools using bellows-heated charcoal fires and iron ore", "label": "filler_2", "cluster": "X"},
        {"text": "The migratory patterns of monarch butterflies span thousands of kilometers across North America", "label": "filler_3", "cluster": "X"},
    ]

    docs = cluster_a + cluster_b + cluster_c + filler
    engine = h.index(docs)
    bm = h.encode(engine, query)
    _, s_geo, _, _, _, _ = h.scores(
        engine, bm["pos"], bm["neg"],
        q_lit_uni=float(bm["lit_uni"][0]),
        q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
    )

    # Raw top-K ranking
    K = 4
    raw_order = np.argsort(-s_geo)[:K]
    raw_clusters = [docs[i]["cluster"] for i in raw_order]
    raw_unique = len(set(raw_clusters) - {"X"})

    # Compute pairwise similarity for MMR
    corpus = engine.db._open_corpus()
    _, _, pos64, neg64, lit_arr = corpus
    qp = bm["pos"].ravel().view(np.uint64)[None, :]
    qn = bm["neg"].ravel().view(np.uint64)[None, :]

    # Simple MMR: greedily select to maximize relevance - lambda*max_sim_to_selected
    mmr_lambda = 0.7
    candidates = list(range(len(docs)))
    selected = []
    for _ in range(K):
        best_i = None
        best_score = -float("inf")
        for ci in candidates:
            relevance = float(s_geo[ci])
            max_sim = 0.0
            for si in selected:
                sim = float(score64(pos64[ci:ci+1], neg64[ci:ci+1],
                                   pos64[si].view(np.uint64), neg64[si].view(np.uint64))[0])
                sim /= max(float(lit_arr[ci]), 1.0)
                max_sim = max(max_sim, sim)
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = ci
        if best_i is not None:
            selected.append(best_i)
            candidates.remove(best_i)

    mmr_clusters = [docs[i]["cluster"] for i in selected]
    mmr_unique = len(set(mmr_clusters) - {"X"})

    diversity_gain = mmr_unique - raw_unique

    O = io.StringIO()
    w = O.write
    w(banner("MMR DIVERSITY ANALYSIS"))
    w(f'\n  QUERY: "{query}"\n')
    w(f"  Clusters: A (3 near-dupes), B (2 Calvin cycle), C (1 chlorophyll)\n")
    w(f"  Filler: {len(filler)}     K={K}     MMR λ={mmr_lambda}\n")

    w(section("RAW RANKING (top-{})".format(K)))
    for rank, i in enumerate(raw_order, 1):
        w(f"  {rank}. {docs[i]['label']:>12s}  cluster={docs[i]['cluster']}  score={s_geo[i]:+.4f}\n")
    w(f"  Unique relevant clusters: {raw_unique}\n")

    w(section("MMR RANKING (top-{})".format(K)))
    for rank, i in enumerate(selected, 1):
        w(f"  {rank}. {docs[i]['label']:>12s}  cluster={docs[i]['cluster']}  score={s_geo[i]:+.4f}\n")
    w(f"  Unique relevant clusters: {mmr_unique}\n")

    w(section("DIVERSITY COMPARISON"))
    w(f"  Raw unique clusters:   {raw_unique}\n")
    w(f"  MMR unique clusters:   {mmr_unique}\n")
    w(f"  Diversity gain:        {diversity_gain:+d}\n")

    # Full score table
    w(section("ALL SCORES"))
    lo, hi = float(s_geo.min()), float(s_geo.max())
    w(f"  {'label':>12s}  {'cluster':>7s}  {'geo':>8s}  bar\n")
    w(f"  {sep(n=60)}\n")
    for i in sorted(range(len(docs)), key=lambda i: s_geo[i], reverse=True):
        w(f"  {docs[i]['label']:>12s}  {docs[i]['cluster']:>7s}  {s_geo[i]:+8.4f}  {hbar(s_geo[i], lo, hi)}\n")

    features = []
    if mmr_unique < 2:
        features.append(f"MMR failed to surface multiple clusters (got {mmr_unique}).")
    if diversity_gain <= 0:
        features.append("MMR did not improve diversity over raw ranking.")
    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="mmr",
        ok=mmr_unique >= 2,
        primary_stat=f"raw={raw_unique}  mmr={mmr_unique}  gain={diversity_gain:+d}",
        summary=f"raw_clusters={raw_unique}  mmr_clusters={mmr_unique}  gain={diversity_gain:+d}",
        details=O.getvalue(),
        metrics={
            "raw_unique_clusters": raw_unique,
            "mmr_unique_clusters": mmr_unique,
            "diversity_gain": diversity_gain,
            "raw_order": [int(i) for i in raw_order],
            "mmr_order": selected,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_code_prose(h):
    """Cross-domain retrieval: code vs prose about the same concept.

    Tests whether a prose query can retrieve relevant code and vice versa.
    The encoding must bridge the tokenization gap between natural language
    and programming constructs.

    Primary: cross-domain P@1 (prose query finds code, code query finds prose).
    Health: at least one cross-domain P@1 = 1.0.
    """
    t0 = time.time()

    docs = [
        {
            "text": "Binary search finds a target value in a sorted array by repeatedly dividing the search interval in half comparing the middle element to the target",
            "label": "bsearch_prose",
            "domain": "prose",
            "topic": "bsearch",
        },
        {
            "text": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
            "label": "bsearch_code",
            "domain": "code",
            "topic": "bsearch",
        },
        {
            "text": "A linked list is a data structure where each node contains a value and a pointer to the next node allowing dynamic memory allocation and efficient insertion",
            "label": "linkedlist_prose",
            "domain": "prose",
            "topic": "linkedlist",
        },
        {
            "text": "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    def insert(self, val):\n        node = Node(val)\n        node.next = self.head\n        self.head = node",
            "label": "linkedlist_code",
            "domain": "code",
            "topic": "linkedlist",
        },
        {
            "text": "Quicksort partitions an array around a pivot element placing smaller elements before it and larger elements after then recursively sorts the partitions",
            "label": "quicksort_prose",
            "domain": "prose",
            "topic": "quicksort",
        },
        {
            "text": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
            "label": "quicksort_code",
            "domain": "code",
            "topic": "quicksort",
        },
        {
            "text": "The French Revolution of 1789 overthrew the monarchy and established republican government in France",
            "label": "filler_1",
            "domain": "filler",
            "topic": "none",
        },
        {
            "text": "Coral reef ecosystems support enormous biodiversity in warm tropical ocean waters",
            "label": "filler_2",
            "domain": "filler",
            "topic": "none",
        },
    ]

    engine = h.index(docs)

    # Prose query → should find both prose and code about that topic
    prose_queries = {
        "bsearch": "binary search algorithm sorted array divide and conquer",
        "linkedlist": "linked list data structure nodes and pointers",
        "quicksort": "quicksort partition pivot recursive sorting",
    }
    # Code-style query → should find both
    code_queries = {
        "bsearch": "binary_search arr target lo hi mid",
        "linkedlist": "class Node LinkedList insert head next",
        "quicksort": "def quicksort arr pivot left right",
    }

    cross_results = []
    for topic in ["bsearch", "linkedlist", "quicksort"]:
        prose_idx = next(i for i, d in enumerate(docs) if d["topic"] == topic and d["domain"] == "prose")
        code_idx = next(i for i, d in enumerate(docs) if d["topic"] == topic and d["domain"] == "code")

        # Prose query → code doc?
        bm = h.encode(engine, prose_queries[topic])
        _, sg, _, _, _, _ = h.scores(
            engine, bm["pos"], bm["neg"],
            q_lit_uni=float(bm["lit_uni"][0]),
            q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
        )
        prose_finds_prose = float(sg[prose_idx])
        prose_finds_code = float(sg[code_idx])
        prose_best_other = float(max(sg[i] for i in range(len(docs)) if docs[i]["topic"] != topic))

        # Code query → prose doc?
        bm = h.encode(engine, code_queries[topic])
        _, sg2, _, _, _, _ = h.scores(
            engine, bm["pos"], bm["neg"],
            q_lit_uni=float(bm["lit_uni"][0]),
            q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
        )
        code_finds_code = float(sg2[code_idx])
        code_finds_prose = float(sg2[prose_idx])
        code_best_other = float(max(sg2[i] for i in range(len(docs)) if docs[i]["topic"] != topic))

        cross_results.append({
            "topic": topic,
            "prose_q_prose_d": prose_finds_prose,
            "prose_q_code_d": prose_finds_code,
            "prose_cross_gap": prose_finds_code - prose_best_other,
            "code_q_code_d": code_finds_code,
            "code_q_prose_d": code_finds_prose,
            "code_cross_gap": code_finds_prose - code_best_other,
        })

    # Aggregate
    prose_to_code_gaps = [r["prose_cross_gap"] for r in cross_results]
    code_to_prose_gaps = [r["code_cross_gap"] for r in cross_results]
    prose_cross_p1 = sum(1 for g in prose_to_code_gaps if g > 0)
    code_cross_p1 = sum(1 for g in code_to_prose_gaps if g > 0)
    med_p2c = float(np.median(prose_to_code_gaps))
    med_c2p = float(np.median(code_to_prose_gaps))
    lm_p2c = l_moments(np.array(prose_to_code_gaps))
    lm_c2p = l_moments(np.array(code_to_prose_gaps))

    O = io.StringIO()
    w = O.write
    w(banner("CODE vs PROSE CROSS-DOMAIN ANALYSIS"))
    w(f"\n  Topics: {', '.join(t for t in ['bsearch', 'linkedlist', 'quicksort'])}\n")
    w(f"  Each topic has a prose doc and a code doc.\n")

    w(section("PROSE QUERY → CODE DOCUMENT"))
    w(f"  {'topic':>12s}  {'same_prose':>10s}  {'cross_code':>10s}  {'best_other':>10s}  {'cross_gap':>10s}  found?\n")
    w(f"  {sep(n=80)}\n")
    for r in cross_results:
        found = "✓" if r["prose_cross_gap"] > 0 else "✗"
        w(f"  {r['topic']:>12s}  {r['prose_q_prose_d']:+10.4f}  {r['prose_q_code_d']:+10.4f}  {r['prose_cross_gap'] + r['prose_q_code_d'] - r['prose_cross_gap']:+10.4f}  {r['prose_cross_gap']:+10.4f}  {found}\n")
    w(f"\n  Prose→Code P@1: {prose_cross_p1}/{len(cross_results)}\n")
    w(f"  Median cross gap: {med_p2c:+.4f}\n")

    w(section("CODE QUERY → PROSE DOCUMENT"))
    w(f"  {'topic':>12s}  {'same_code':>10s}  {'cross_prose':>10s}  {'best_other':>10s}  {'cross_gap':>10s}  found?\n")
    w(f"  {sep(n=80)}\n")
    for r in cross_results:
        found = "✓" if r["code_cross_gap"] > 0 else "✗"
        w(f"  {r['topic']:>12s}  {r['code_q_code_d']:+10.4f}  {r['code_q_prose_d']:+10.4f}  {r['code_cross_gap'] + r['code_q_prose_d'] - r['code_cross_gap']:+10.4f}  {r['code_cross_gap']:+10.4f}  {found}\n")
    w(f"\n  Code→Prose P@1: {code_cross_p1}/{len(cross_results)}\n")
    w(f"  Median cross gap: {med_c2p:+.4f}\n")

    w(section("CROSS-DOMAIN SUMMARY"))
    total_cross = prose_cross_p1 + code_cross_p1
    w(f"  Total cross-domain hits: {total_cross}/{2 * len(cross_results)}\n")
    w(f"  Prose→Code L-moments: {fmt_lm(lm_p2c)}\n")
    w(f"  Code→Prose L-moments: {fmt_lm(lm_c2p)}\n")

    features = []
    if prose_cross_p1 == 0:
        features.append("Prose queries cannot find code documents. Token universes are disjoint.")
    if code_cross_p1 == 0:
        features.append("Code queries cannot find prose documents. Token universes are disjoint.")
    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    h.cleanup(engine)
    return ProbeResult(
        name="code_prose",
        ok=total_cross >= 1,
        primary_stat=f"cross={total_cross}/{2*len(cross_results)}  p2c={prose_cross_p1}  c2p={code_cross_p1}",
        summary=f"cross={total_cross}/{2*len(cross_results)}  p2c_gap={med_p2c:+.4f}  c2p_gap={med_c2p:+.4f}",
        details=O.getvalue(),
        metrics={
            "total_cross_hits": total_cross,
            "prose_to_code_p1": prose_cross_p1,
            "code_to_prose_p1": code_cross_p1,
            "median_p2c_gap": med_p2c,
            "median_c2p_gap": med_c2p,
            "lm_p2c": lm_p2c,
            "lm_c2p": lm_c2p,
            "per_topic": cross_results,
        },
        duration=time.time() - t0,
        warnings=features,
    )


def probe_ngram_sweep(h):
    """N-gram order sweep: how encoding behavior changes from ngram=2 to max.

    Builds a separate index at each n-gram order and runs a compact battery:
      - Noise separation (Cohen's d): can the encoding distinguish signal from random?
      - Topic selectivity (min margin): does each topic rank itself highest?
      - Short query positive: minimum token count to produce a positive score
      - Scoring weight: alpha (unigram) and 1-alpha (n-gram) at each order

    Primary: noise Cohen's d across the sweep.
    Health: all orders achieve d > 0.8.
    """
    t0 = time.time()
    max_n = h.config.hdc_ngram
    dims = h.config.hdc_dimensions

    if max_n < 3:
        return ProbeResult(
            name="ngram_sweep",
            ok=True,
            primary_stat="skipped (max_ngram < 3)",
            summary="skipped",
            details="  N-gram sweep requires hdc_ngram >= 3 to have multiple values to compare.\n",
            duration=time.time() - t0,
        )

    # ── Shared corpus for noise + topic + short query tests ──────────

    rng = np.random.default_rng(99)

    noise_corpus = [
        {
            "text": "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly affects the other regardless of distance",
            "label": "signal_1",
            "group": "signal",
        },
        {
            "text": "Photon pair generation through spontaneous parametric down-conversion creates entangled states for quantum optics experiments",
            "label": "signal_2",
            "group": "signal",
        },
        {
            "text": "Bell inequality violations demonstrate that entangled particles exhibit correlations that cannot be explained by local hidden variable theories",
            "label": "signal_3",
            "group": "signal",
        },
    ] + [
        {
            "text": "".join(
                rng.choice(list("abcdefghijklmnopqrstuvwxyz "), size=rng.integers(50, 200))
            ),
            "label": f"noise_{i}",
            "group": "noise",
        }
        for i in range(6)
    ]
    noise_query = "quantum entanglement between photon pairs"

    topic_corpus = [
        {"text": "Coral reef ecosystems support enormous diversity of marine life including fish and invertebrates in warm tropical waters", "label": "marine_0", "topic": "marine"},
        {"text": "Deep sea hydrothermal vents host extremophile organisms that derive energy from chemosynthesis", "label": "marine_1", "topic": "marine"},
        {"text": "Lexical analysis tokenizes source code into tokens by matching patterns defined by regular expressions", "label": "compiler_0", "topic": "compiler"},
        {"text": "Abstract syntax trees represent hierarchical structure of source code enabling semantic analysis and code generation", "label": "compiler_1", "topic": "compiler"},
        {"text": "The feudal system organized medieval European society into a hierarchy of lords vassals and serfs", "label": "medieval_0", "topic": "medieval"},
        {"text": "Gothic cathedral construction employed flying buttresses and ribbed vaults to achieve unprecedented height", "label": "medieval_1", "topic": "medieval"},
    ]
    topic_queries = {
        "marine": "coral reef fish diversity ocean ecosystem",
        "compiler": "lexer parser abstract syntax tree code generation",
        "medieval": "feudal lords vassals medieval European society",
    }

    # Short queries: 2 tokens through 8 tokens
    short_base_words = ["quantum", "field", "theory", "predicts", "particle", "interactions", "using", "gauge"]
    short_corpus = [
        {"text": "Quantum field theory predicts particle interactions using gauge symmetry principles and renormalization techniques in high energy physics", "label": "target"},
        {"text": "The weather today is sunny with clear skies and mild temperatures throughout the afternoon", "label": "filler_1"},
        {"text": "Ancient Roman architecture featured arches columns and domes that influenced building design for centuries", "label": "filler_2"},
    ]

    # ── Sweep ────────────────────────────────────────────────────────

    orig_ngram = h.config.hdc_ngram
    results = {}

    for n in range(2, max_n + 1):
        h.config.hdc_ngram = n
        r = {"ngram": n}

        # Alpha weights
        r["alpha_uni"] = (n - 1) / n
        r["alpha_ng"] = 1.0 / n

        # ── Noise separation ──
        engine = h.index(noise_corpus)
        bm = h.encode(engine, noise_query)
        _, s_geo, _, _, _, _ = h.scores(
            engine, bm["pos"], bm["neg"],
            q_lit_uni=float(bm["lit_uni"][0]),
            q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
        )
        sig_i = [i for i, d in enumerate(noise_corpus) if d["group"] == "signal"]
        noi_i = [i for i, d in enumerate(noise_corpus) if d["group"] == "noise"]
        r["noise_d"] = cohen_d(s_geo[sig_i], s_geo[noi_i])
        r["noise_margin"] = float(s_geo[sig_i].min() - s_geo[noi_i].max())
        r["signal_mean"] = float(s_geo[sig_i].mean())
        r["noise_mean"] = float(s_geo[noi_i].mean())
        h.cleanup(engine)

        # ── Topic selectivity ──
        engine = h.index(topic_corpus)
        topics = list(topic_queries.keys())
        topic_ranges = {}
        for topic in topics:
            topic_ranges[topic] = [i for i, d in enumerate(topic_corpus) if d["topic"] == topic]

        margins = []
        for qt in topics:
            bm = h.encode(engine, topic_queries[qt])
            _, sg, _, _, _, _ = h.scores(
                engine, bm["pos"], bm["neg"],
                q_lit_uni=float(bm["lit_uni"][0]),
                q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
            )
            own = float(sg[topic_ranges[qt]].mean())
            best_other = max(
                float(sg[topic_ranges[dt]].mean())
                for dt in topics if dt != qt
            )
            margins.append(own - best_other)
        r["topic_min_margin"] = min(margins)
        r["topic_margins"] = margins
        h.cleanup(engine)

        # ── Short query threshold ──
        engine = h.index(short_corpus)
        r["short_scores"] = {}
        r["positive_at"] = 0
        for ntok in range(2, len(short_base_words) + 1):
            q = " ".join(short_base_words[:ntok])
            bm = h.encode(engine, q)
            _, sg, _, _, _, _ = h.scores(
                engine, bm["pos"], bm["neg"],
                q_lit_uni=float(bm["lit_uni"][0]),
                q_lit_ngram=float(bm.get("lit_ngram", np.zeros(1))[0]),
            )
            target_score = float(sg[0])  # target doc
            best_filler = float(max(sg[1], sg[2]))
            r["short_scores"][ntok] = {
                "target": target_score,
                "filler": best_filler,
                "gap": target_score - best_filler,
            }
            if target_score > best_filler and r["positive_at"] == 0:
                r["positive_at"] = ntok
        if r["positive_at"] == 0:
            r["positive_at"] = len(short_base_words) + 1  # never
        h.cleanup(engine)

        results[n] = r

    h.config.hdc_ngram = orig_ngram

    # ── Output ───────────────────────────────────────────────────────

    O = io.StringIO()
    w = O.write
    w(banner("N-GRAM ORDER SWEEP"))
    w(f"\n  HDC dims: {dims}     Sweep: ngram=2..{max_n}\n")
    w(f"  Testing noise separation, topic selectivity, and short query threshold\n")
    w(f"  at each n-gram order to find the behavioral sweet spot.\n")

    w(section("SCORING WEIGHTS"))
    w(f"  {'ngram':>5s}  {'α_uni':>7s}  {'α_ng':>7s}  {'uni bar':<20s}  {'ng bar':<20s}\n")
    w(f"  {sep(n=70)}\n")
    for n in sorted(results):
        r = results[n]
        ub = hbar_abs(r["alpha_uni"], 1.0, 20)
        nb = hbar_abs(r["alpha_ng"], 1.0, 20)
        w(f"  {n:5d}  {r['alpha_uni']:7.3f}  {r['alpha_ng']:7.3f}  {ub}  {nb}\n")

    w(section("NOISE SEPARATION BY N-GRAM ORDER"))
    w(f"  Query: \"{noise_query}\"\n")
    w(f"  Signal: {len(sig_i)} docs     Noise: {len(noi_i)} random docs\n\n")
    w(f"  {'ngram':>5s}  {'Cohen d':>9s}  {'margin':>8s}  {'sig_mu':>8s}  {'noi_mu':>8s}  {'label':>10s}  bar\n")
    w(f"  {sep(n=80)}\n")
    max_d = max(abs(r["noise_d"]) for r in results.values())
    for n in sorted(results):
        r = results[n]
        label = effect_label(r["noise_d"])
        w(f"  {n:5d}  {r['noise_d']:+9.4f}  {r['noise_margin']:+8.4f}  {r['signal_mean']:+8.4f}  {r['noise_mean']:+8.4f}  {label:>10s}  {hbar_abs(r['noise_d'], max_d, 20)}\n")

    w(section("TOPIC SELECTIVITY BY N-GRAM ORDER"))
    w(f"  Topics: {', '.join(topics)}     Docs/topic: 2\n\n")
    w(f"  {'ngram':>5s}  {'min_margin':>10s}  {'separated':>9s}  {'margins':>30s}  bar\n")
    w(f"  {sep(n=80)}\n")
    max_m = max(abs(r["topic_min_margin"]) for r in results.values())
    for n in sorted(results):
        r = results[n]
        ok = "yes" if r["topic_min_margin"] > 0 else "NO"
        mar_str = "  ".join(f"{m:+.4f}" for m in r["topic_margins"])
        w(f"  {n:5d}  {r['topic_min_margin']:+10.4f}  {ok:>9s}  {mar_str:>30s}  {hbar_abs(r['topic_min_margin'], max(max_m, 0.01), 20)}\n")

    w(section("SHORT QUERY THRESHOLD BY N-GRAM ORDER"))
    w(f"  Words: {' → '.join(short_base_words)}\n")
    w(f"  'positive_at' = minimum word count where target outscores all filler.\n\n")
    w(f"  {'ngram':>5s}  {'pos_at':>6s}  ")
    for ntok in range(2, len(short_base_words) + 1):
        w(f"  {ntok}w_gap")
    w(f"\n  {sep(n=80)}\n")
    for n in sorted(results):
        r = results[n]
        pa = r["positive_at"]
        pa_str = f"{pa}" if pa <= len(short_base_words) else "never"
        w(f"  {n:5d}  {pa_str:>6s}  ")
        for ntok in range(2, len(short_base_words) + 1):
            if ntok in r["short_scores"]:
                gap = r["short_scores"][ntok]["gap"]
                w(f"  {gap:+6.3f}")
            else:
                w(f"  {'---':>6s}")
        w("\n")

    w(section("SWEEP SUMMARY"))
    w(f"\n  {'ngram':>5s}  {'noise_d':>9s}  {'topic_margin':>12s}  {'short_pos':>9s}  {'α_uni':>7s}  recommendation\n")
    w(f"  {sep(n=70)}\n")
    for n in sorted(results):
        r = results[n]
        pa = r["positive_at"]
        pa_str = f"{pa} words" if pa <= len(short_base_words) else "never"
        # Simple heuristic recommendation
        rec = ""
        if r["noise_d"] < 0.8:
            rec = "weak noise separation"
        elif r["topic_min_margin"] <= 0:
            rec = "topics not separated"
        elif pa > 5:
            rec = "needs long queries"
        else:
            rec = "✓"
        w(f"  {n:5d}  {r['noise_d']:+9.4f}  {r['topic_min_margin']:+12.4f}  {pa_str:>9s}  {r['alpha_uni']:7.3f}  {rec}\n")

    # ── Scatter: ngram order vs noise d ──
    w("\n  NOISE SEPARATION vs N-GRAM ORDER\n")
    w(
        ascii_scatter(
            xs=[n for n in sorted(results)],
            ys=[results[n]["noise_d"] for n in sorted(results)],
            labels=[f"n={n}" for n in sorted(results)],
            x_label="n-gram order",
        )
    )

    # ── Health: all orders achieve d > 0.8 ──
    all_strong = all(r["noise_d"] > 0.8 for r in results.values())
    best_n = max(results, key=lambda n: results[n]["noise_d"])
    worst_n = min(results, key=lambda n: results[n]["noise_d"])

    features = []
    for n in sorted(results):
        r = results[n]
        if r["noise_d"] < 0.8:
            features.append(
                f"ngram={n}: noise separation below threshold (d={r['noise_d']:+.4f})"
            )
        if r["topic_min_margin"] <= 0:
            features.append(
                f"ngram={n}: topic selectivity fails (min_margin={r['topic_min_margin']:+.4f})"
            )

    if features:
        w(section("STRUCTURAL FEATURES"))
        for i, feat in enumerate(features):
            w(f"  {i + 1}. {feat}\n")

    return ProbeResult(
        name="ngram_sweep",
        ok=all_strong,
        primary_stat=f"best=n{best_n}(d={results[best_n]['noise_d']:+.2f})  worst=n{worst_n}(d={results[worst_n]['noise_d']:+.2f})",
        summary=(
            f"sweep=2..{max_n}  "
            + "  ".join(f"n{n}:d={results[n]['noise_d']:+.2f},m={results[n]['topic_min_margin']:+.3f}" for n in sorted(results))
        ),
        details=O.getvalue(),
        metrics={
            "sweep_range": [2, max_n],
            "best_ngram": best_n,
            "worst_ngram": worst_n,
            "per_ngram": {
                n: {
                    "noise_d": r["noise_d"],
                    "noise_margin": r["noise_margin"],
                    "topic_min_margin": r["topic_min_margin"],
                    "positive_at": r["positive_at"],
                    "alpha_uni": r["alpha_uni"],
                    "alpha_ng": r["alpha_ng"],
                }
                for n, r in results.items()
            },
        },
        duration=time.time() - t0,
        warnings=features,
    )


ALL_PROBES = {
    "ngram": probe_ngram,
    "idf": probe_idf,
    "sparse": probe_sparse,
    "specificity": probe_specificity,
    "noise": probe_noise,
    "duplicate": probe_duplicate,
    "topic": probe_topic,
    "density": probe_density,
    "sensitivity": probe_sensitivity,
    "retrieval": probe_retrieval,
    "paraphrase": probe_paraphrase,
    "query_length": probe_query_length,
    "corpus_scale": probe_corpus_scale,
    "adversarial": probe_adversarial,
    "mmr": probe_mmr,
    "code_prose": probe_code_prose,
    "ngram_sweep": probe_ngram_sweep,
}


class _Tee:
    """Write to both a file and the original stream."""

    def __init__(self, stream, path):
        self._stream = stream
        self._file = open(path, "w", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def main():
    ap = argparse.ArgumentParser(description="HdRAG behavioral analysis suite")
    ap.add_argument("--config", default="hdrag_config.yaml")
    ap.add_argument(
        "--probe",
        default=None,
        help=f"Comma-separated probe names. Available: {','.join(ALL_PROBES)}",
    )
    ap.add_argument("--brief", action="store_true", help="Summary table only")
    ap.add_argument("--json", default=None, metavar="PATH")
    ap.add_argument(
        "--results",
        default="hdrag_results.txt",
        metavar="PATH",
        help="Save text output to file (default: hdrag_results.txt)",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tee = _Tee(sys.stdout, args.results)
    sys.stdout = tee

    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    for nm in ("httpx", "httpcore"):
        logging.getLogger(nm).setLevel(logging.WARNING)
    logger = logging.getLogger("hdrag_probe")
    if not args.debug:
        logger.setLevel(logging.WARNING)

    config = Config.load(args.config)

    # Tokenizer from inference GGUF (shared vocabulary)
    gguf_path = resolve_gguf(config.gguf_model, config.model_dir)
    if not gguf_path:
        print(f"ERROR: GGUF not found: {config.gguf_model}")
        sys.exit(1)

    print(banner("HdRAG Behavioral Analysis"))
    print(f"  GGUF:     {Path(gguf_path).stem}")
    print(f"  HDC dims: {config.hdc_dimensions}")
    print(f"  N-gram:   {config.hdc_ngram}")
    print("  Loading tokenizer...", end=" ", flush=True)
    tokenizer = HuggingFaceTokenizer.from_gguf(gguf_path, logger=logger)
    print(f"vocab={tokenizer.vocab_size:,}")

    harness = Harness(config, tokenizer, logger)

    if args.probe:
        names = [t.strip() for t in args.probe.split(",")]
        for nm in names:
            if nm not in ALL_PROBES:
                print(f"ERROR: Unknown probe '{nm}'. Available: {','.join(ALL_PROBES)}")
                sys.exit(1)
        probes = {nm: ALL_PROBES[nm] for nm in names}
    else:
        probes = ALL_PROBES

    results = []
    t_total = time.time()
    for name, fn in probes.items():
        print(f"\n{sep(n=W)}")
        print(f"  Running: {name}", end="", flush=True)
        try:
            r = fn(harness)
            results.append(r)
            tag = "ok" if r.ok else "??"
            print(f"  [{tag}]  ({r.duration:.1f}s)  {r.primary_stat}  {r.summary}")
            if not args.brief:
                print(r.details)
        except Exception as e:
            results.append(
                ProbeResult(
                    name=name, ok=False, primary_stat="ERROR", details=f"  ERROR: {e}"
                )
            )
            print(f"  ERROR: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

    total = time.time() - t_total
    ok_count = sum(1 for r in results if r.ok)

    print(banner("SUMMARY"))
    print(f"  {'probe':>15s}  {'primary statistic':>40s}  {'time':>6s}  health")
    print(f"  {sep(n=75)}")
    for r in results:
        tag = "ok" if r.ok else "??"
        if r.ok and r.warnings:
            tag = "ok*"
        print(f"  {r.name:>15s}  {r.primary_stat:>40s}  {r.duration:5.1f}s  {tag}")
    print(f"\n  {ok_count}/{len(results)} healthy  ({total:.1f}s total)")

    # --- Encoding Map Characterization ---
    all_features = [(r.name, w) for r in results for w in r.warnings]
    if all_features:
        print(banner("ENCODING MAP CHARACTERIZATION"))
        # Group features by category
        sensitivity_feats = [(n, f) for n, f in all_features if n == "duplicate"]
        transfer_feats = [(n, f) for n, f in all_features if n == "specificity"]
        capacity_feats = [(n, f) for n, f in all_features if n == "density"]
        other_feats = [
            (n, f)
            for n, f in all_features
            if n not in ("duplicate", "specificity", "density")
        ]

        if sensitivity_feats:
            print()
            print(f"  {sep(n=70)}")
            for name, feat in sensitivity_feats:
                print(f"  {feat}")

        if transfer_feats:
            print("\n  TRANSFER FUNCTION (overlap → correlation)")
            print(f"  {sep(n=70)}")
            for name, feat in transfer_feats:
                print(f"  {feat}")

        if capacity_feats:
            print("\n  CAPACITY (entropy rate)")
            print(f"  {sep(n=70)}")
            for name, feat in capacity_feats:
                print(f"  {feat}")

        if other_feats:
            print("\n  OTHER")
            print(f"  {sep(n=70)}")
            for name, feat in other_feats:
                print(f"  [{name}] {feat}")

    # --- Encoding Signature (compact numerical fingerprint) ---
    print(banner("ENCODING SIGNATURE"))
    sig = {}
    for r in results:
        sig.update(r.metrics)
    sig_lines = []
    # Pick the most informative metrics from each probe
    for r in results:
        m = r.metrics
        if r.name == "ngram":
            sig_lines.append(
                f"  Compositional ramp:    rho={m.get('rho', '?'):+.4f}  rho_med={m.get('rho_median', '?'):+.4f}"
            )
        elif r.name == "idf":
            sig_lines.append(
                f"  IDF discrimination:    d={m.get('cohen_d', '?'):+.4f}  rd={m.get('robust_d', '?'):+.4f}"
            )
        elif r.name == "noise":
            sig_lines.append(
                f"  Signal isolation:      d={m.get('cohen_d', '?'):+.4f}  rd={m.get('robust_d', '?'):+.4f}  margin={m.get('margin', '?'):+.4f}"
            )
        elif r.name == "topic":
            sig_lines.append(
                f"  Topic selectivity:     min_margin={m.get('min_margin', '?'):+.4f}  min_margin_med={m.get('min_margin_median', '?'):+.4f}"
            )
        elif r.name == "sparse":
            tr = m.get("tier_results", {})
            parts = []
            for t in ["short", "medium", "long"]:
                if t in tr:
                    parts.append(f"{t}:{'above' if tr[t].get('geo_above_filler') else 'below'}")
            sig_lines.append(
                f"  Normalization:         {' '.join(parts)}  bias_reduced={'yes' if m.get('bias_reduced') else 'no'}"
            )
        elif r.name == "duplicate":
            sig_lines.append(
                f"  Sensitivity:           rho={m.get('rho', '?'):+.4f}  jump={m.get('identity_jump', '?'):.4f}  iso={m.get('isometry_constant', '?'):+.4f}"
            )
        elif r.name == "specificity":
            sig_lines.append(
                f"  Transfer function:     MI={m.get('mutual_information', '?'):.3f}bits  opacity={m.get('lexical_opacity', '?'):.2f}"
            )
        elif r.name == "density":
            sig_lines.append(
                f"  Entropy rate:          asymptotic={m.get('asymptotic_entropy_rate', '?'):+.1f}bits/tok  H_distort={m.get('max_entropy_distortion', '?'):.3f}  CV_mad={m.get('density_cv_mad', '?'):.3f}"
            )
        elif r.name == "sensitivity":
            med_d = m.get("median_delta", "?")
            cv_m = m.get("cv_mad", "?")
            sig_lines.append(
                f"  Query resolution:      mu_delta={m.get('mean_delta', '?'):.4f}  med_delta={med_d:.4f}  CV={m.get('cv', '?'):.4f}  CV_mad={cv_m:.4f}"
            )
        elif r.name == "retrieval":
            sig_lines.append(
                f"  End-to-end retrieval:  d={m.get('cohen_d', '?'):+.4f}  rd={m.get('robust_d', '?'):+.4f}  P@1={m.get('precision', {}).get(1, '?'):.2f}"
            )
        elif r.name == "paraphrase":
            sig_lines.append(
                f"  Paraphrase:            med_ret={m.get('median_retention', '?'):.3f}  above_filler={m.get('para_above_filler', '?')}/{len(m.get('pair_metrics', []))}"
            )
        elif r.name == "query_length":
            sig_lines.append(
                f"  Query length:          P@1_at={m.get('first_p1_tokens', '?')}tok  med_gap={m.get('median_gap', '?'):+.4f}"
            )
        elif r.name == "corpus_scale":
            sig_lines.append(
                f"  Corpus scale:          d@500={m.get('final_d', '?'):+.4f}  rho={m.get('rho_scale', '?'):+.4f}"
            )
        elif r.name == "adversarial":
            sig_lines.append(
                f"  Adversarial:           margin={m.get('margin', '?'):+.4f}  rank={m.get('target_rank', '?')}  decoy_d={m.get('decoy_filler_d', '?'):+.4f}"
            )
        elif r.name == "mmr":
            sig_lines.append(
                f"  MMR diversity:         raw={m.get('raw_unique_clusters', '?')}  mmr={m.get('mmr_unique_clusters', '?')}  gain={m.get('diversity_gain', '?'):+d}"
            )
        elif r.name == "code_prose":
            sig_lines.append(
                f"  Code/prose:            cross={m.get('total_cross_hits', '?')}/6  p2c={m.get('prose_to_code_p1', '?')}  c2p={m.get('code_to_prose_p1', '?')}"
            )
        elif r.name == "ngram_sweep":
            best = m.get("best_ngram", "?")
            worst = m.get("worst_ngram", "?")
            pn = m.get("per_ngram", {})
            if pn:
                parts = [f"n{n}:d={v.get('noise_d', 0):+.2f}" for n, v in sorted(pn.items())]
                sig_lines.append(f"  N-gram sweep:          best=n{best}  worst=n{worst}  {' '.join(parts)}")
            else:
                sig_lines.append(f"  N-gram sweep:          best=n{best}  worst=n{worst}")

    for line in sig_lines:
        print(line)
    print(f"{'=' * W}\n")

    if args.json:
        jdata = {
            "gguf": Path(gguf_path).stem,
            "hdc_dims": config.hdc_dimensions,
            "hdc_ngram": config.hdc_ngram,
            "total_time": total,
            "probes": [],
        }
        for r in results:
            jr = {
                "name": r.name,
                "ok": r.ok,
                "primary_stat": r.primary_stat,
                "summary": r.summary,
                "duration": r.duration,
                "warnings": r.warnings,
                "metrics": {},
            }
            for k, v in r.metrics.items():
                jr["metrics"][k] = _json_safe(v)
            jdata["probes"].append(jr)
        with open(args.json, "w") as f:
            json.dump(jdata, f, indent=2)
        print(f"  Results written to {args.json}")

    print(f"\n  Results saved to {args.results}")
    sys.stdout = tee._stream
    tee.close()

    sys.exit(0 if ok_count == len(results) else 1)


if __name__ == "__main__":
    main()