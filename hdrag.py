"""
hdrag - Hyperdimensional Retrieval-Augmented Generation (Memory Engine)

Indexing:  text → tokenize → embed → project → ternary quantize → pos/neg bitmaps
Retrieval: query → encode → prune → threshold → rerank → context

This module has zero dependency on the Gradio UI (hdrag_gradio.py)
or the inference pipeline (generation lives in hdrag_model.InferenceEngine).
It depends on hdrag_model for Config, tokenizer, and GGUF helpers.
"""

from __future__ import annotations

import gc, hashlib, json, logging, math, os, sqlite3
import statistics, time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gguf import GGUFReader

from hdrag_model import (
    Config,
    Tokenizer,
    trim_working_set,
    gguf_bytes,
    gguf_int,
    gguf_field,
)


# ── Tensor Registry ────────────────────────────────────────────────

_T: dict[str, torch.Tensor] = {}


def tset(k: str, v: torch.Tensor) -> torch.Tensor:
    _T[k] = v.cpu() if v.device.type != "cpu" else v
    return _T[k]


def tget(k: str) -> Optional[torch.Tensor]:
    return _T.get(k)


def tdel(k: str):
    _T.pop(k, None)


# ── Constants & Core Math ──────────────────────────────────────────

_M1, _M2, _M4, _H01 = (
    np.uint64(x)
    for x in (
        0x5555555555555555,
        0x3333333333333333,
        0x0F0F0F0F0F0F0F0F,
        0x0101010101010101,
    )
)


def bstride(d_hdc: int) -> int:
    d = (d_hdc + 7) // 8
    return d + (8 - d % 8) % 8


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> 1) & _M1)
    x = (x & _M2) + ((x >> 2) & _M2)
    x = (x + (x >> 4)) & _M4
    return ((x * _H01) >> 56).astype(np.int32)


def score64(dp, dn, qp, qn) -> np.ndarray:
    if qp.ndim == 1:
        qp, qn = qp[None, :], qn[None, :]
    agree = (dp & qp) | (dn & qn)
    disagree = (dp & qn) | (dn & qp)
    return (popcount64(agree).sum(1) - popcount64(disagree).sum(1)).astype(np.float32)


def bind_ternary_bits(ap: np.ndarray, an: np.ndarray, bp: np.ndarray, bn: np.ndarray):
    """Ternary multiply in packed bitspace: (+)(+)→+, (+)(-)→-, (0)(x)→0."""
    return (ap & bp) | (an & bn), (ap & bn) | (an & bp)


def bind_ternary_bits_into(
    ap: np.ndarray,
    an: np.ndarray,
    bp: np.ndarray,
    bn: np.ndarray,
    outp: np.ndarray,
    outn: np.ndarray,
    tmp: np.ndarray,
) -> None:
    """In-place ternary multiply in packed bitspace to avoid allocation."""
    np.bitwise_and(ap, bp, out=outp)
    np.bitwise_and(an, bn, out=tmp)
    np.bitwise_or(outp, tmp, out=outp)

    np.bitwise_and(ap, bn, out=outn)
    np.bitwise_and(an, bp, out=tmp)
    np.bitwise_or(outn, tmp, out=outn)


def align8(n: int) -> int:
    return (n + 7) & ~7


def mask64_for_dims(dims: int, stride_bytes: int) -> np.ndarray:
    n_words = stride_bytes // 8
    m = np.full((n_words,), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    pad_bits = n_words * 64 - dims
    if pad_bits > 0:
        keep = 64 - pad_bits
        m[-1] = (
            np.uint64((1 << keep) - 1) if keep < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
        )
    return m[None, :]


def adaptive_threshold(vals: torch.Tensor) -> float:
    n = vals.numel()
    if n < 4:
        return vals.max().item() if n > 0 else 0.0
    x = vals.sort()[0]
    m = n // 2
    median = (x[m - 1] + x[m]) / 2
    idx = torch.arange(m, device=x.device, dtype=x.dtype)
    L2 = 2 * (x[:m] * idx).sum() / (m * (m - 1)) - x[:m].mean()
    return (median + L2 * math.log(n)).item() if L2 > 1e-9 else median.item()


def compress_context(text: str) -> str:
    if not text:
        return text
    words = text.split()
    if not words:
        return text
    freq = Counter(words)
    imp = {w: 1 / n for w, n in freq.items()}
    thr = statistics.median(imp.values())
    return " ".join(w for w in words if imp[w] >= thr)


# ── Data Utilities ─────────────────────────────────────────────────


def chunks(xs: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def extract_text(item: dict) -> str:
    def _join(msgs, role_key="from", text_key=None):
        out = []
        for m in msgs:
            rk = m.get(role_key, m.get("role", ""))
            if rk == "system":
                continue
            out.append(m.get(text_key or "value", m.get("content", "")))
        return "\n\n".join(x for x in out if x)

    if (
        isinstance(item, list)
        and item
        and isinstance(item[0], dict)
        and "from" in item[0]
    ):
        return _join(item)
    for key in ("conversations", "conversation"):
        if key in item:
            return _join(item[key])
    if "messages" in item:
        return _join(item["messages"], role_key="role", text_key="content")

    human = [
        "human",
        "user",
        "prompt",
        "problem",
        "question",
        "instruction",
        "message_1",
    ]
    assist = [
        "gpt",
        "assistant",
        "response",
        "answer",
        "output",
        "message_2",
    ]
    parts = []
    for k in human:
        if item.get(k):
            parts.append(item[k])
            break
    for k in assist:
        if item.get(k):
            parts.append(item[k])
            break
    if parts:
        if item.get("input"):
            parts.insert(1, item["input"])
        return "\n\n".join(parts)
    return item.get("text", item.get("content", ""))


def discover_datasets(directory: Path) -> list:
    exts = {".json", ".jsonl", ".parquet", ".txt", ".md", ".html", ".xml"}
    if not directory.exists():
        return []
    if not directory.is_dir():
        return (
            [{"name": directory.stem, "path": str(directory)}]
            if directory.suffix in exts
            else []
        )
    return [
        {"name": fp.stem, "path": str(fp)}
        for fp in sorted(directory.glob("**/*"))
        if fp.suffix in exts
    ]


def iter_dataset(
    path: Path,
    tokenizer: Tokenizer = None,
    chunk_size: int = 1024,
) -> Generator[dict, None, None]:
    if path.suffix in (".txt", ".md", ".html", ".xml"):
        text = path.read_text(encoding="utf-8")
        if tokenizer and chunk_size:
            tokens = tokenizer.tokenize(text)
            overlap = max(64, chunk_size // 8)
            step = chunk_size - overlap
            total = (len(tokens) + step - 1) // step
            for i, start in enumerate(range(0, len(tokens), step)):
                chunk = tokens[start : start + chunk_size]
                if len(chunk) < overlap and i > 0:
                    break
                yield {
                    "text": tokenizer.detokenize(chunk),
                    "chunk_idx": i,
                    "total_chunks": total,
                }
        else:
            yield {"text": text}
    elif path.suffix == ".parquet":
        for rec in pd.read_parquet(path).to_dict(orient="records"):
            yield {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in rec.items()
            }
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        yield from (data if isinstance(data, list) else [data])
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


# ── Database ───────────────────────────────────────────────────────


class Database:
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS memories (
            hdv_idx INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL, metadata JSON, token_count INTEGER);
        CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS idf (
            token_id INTEGER PRIMARY KEY, doc_freq INTEGER NOT NULL,
            weight REAL NOT NULL) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_memory_id ON memories(id);
    """

    def __init__(self, db_path: Path, config: Config, logger=None):
        self.db_path = db_path
        self.config = config
        self.corpus_file = db_path.parent / "corpus_hdc.idx"
        self.vocab_file = db_path.parent / "vocab_hdc.idx"
        self.logger = logger
        self._stride = bstride(config.hdc_dimensions)
        self._init_db()

    def _init_db(self):
        with self._conn() as c:
            c.executescript(self.SCHEMA)
            row = c.execute(
                "SELECT value FROM config WHERE key='hdc_dimensions'"
            ).fetchone()
            if row:
                if json.loads(row[0]) != self.config.hdc_dimensions:
                    raise ValueError("HDC dimension mismatch with existing index")
            else:
                c.execute(
                    "INSERT INTO config VALUES (?,?)",
                    ("hdc_dimensions", json.dumps(self.config.hdc_dimensions)),
                )

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA cache_size=-{self.config.sqlite_cache_kb}",
            "PRAGMA temp_store=MEMORY",
        ):
            c.execute(pragma)
        try:
            yield c
            c.commit()
        except:
            c.rollback()
            raise
        finally:
            c.close()

    def _query(self, sql, params=()):
        with self._conn() as c:
            return c.execute(sql, params).fetchall()

    def _chunked(self, sql, ids, extract):
        result = {}
        for chunk in chunks(ids, self.config.sqlite_max_vars):
            ph = ",".join("?" * len(chunk))
            for row in self._query(sql.format(ph), tuple(chunk)):
                k, v = extract(row)
                result[k] = v
        return result

    def _open_corpus(self):
        if not self.corpus_file.exists():
            return None
        n = self.count()
        if n == 0:
            return None
        data = np.memmap(self.corpus_file, dtype=np.uint8, mode="r")
        layout = self.get_config("corpus_layout") or "interleaved"
        if layout == "blocked":
            half = n * self._stride
            return (
                data,
                data[:half].reshape(n, self._stride),
                data[half:].reshape(n, self._stride),
            )
        else:
            d = self._stride
            view = data.reshape(n, d * 2)
            return data, view[:, :d], view[:, d:]

    def count(self) -> int:
        return self._query("SELECT COUNT(*) FROM memories")[0][0]

    def exists(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        return set(
            self._chunked(
                "SELECT id FROM memories WHERE id IN ({})",
                ids,
                lambda r: (r["id"], True),
            )
        )

    def get_memories(self, indices: list[int]) -> dict[int, dict]:
        if not indices:
            return {}
        return self._chunked(
            "SELECT hdv_idx, id, text, metadata, token_count "
            "FROM memories WHERE hdv_idx IN ({})",
            indices,
            lambda r: (
                r["hdv_idx"],
                {
                    "id": r["id"],
                    "text": r["text"],
                    "hdv_idx": r["hdv_idx"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "token_count": r["token_count"],
                },
            ),
        )

    def get_token_counts(self) -> np.ndarray:
        n = self.count()
        counts = np.ones(n, dtype=np.int32)
        for r in self._query("SELECT hdv_idx, token_count FROM memories"):
            counts[r["hdv_idx"]] = r["token_count"] or 1
        return counts

    def get_bitmaps(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        corpus = self._open_corpus()
        if not corpus or not indices:
            d = self._stride
            return np.empty((0, d), np.uint8), np.empty((0, d), np.uint8)
        _, pos, neg = corpus
        idx = np.array(indices)
        result = pos[idx].copy(), neg[idx].copy()
        del corpus
        return result

    def save_idf(self, df_counts: dict, n_docs: int):
        rows = [
            (tid, df, math.log((n_docs + 1) / (df + 1)))
            for tid, df in df_counts.items()
        ]
        with self._conn() as c:
            c.execute("DELETE FROM idf")
            c.executemany("INSERT INTO idf VALUES (?,?,?)", rows)

    def load_idf(self) -> dict[int, float]:
        return {
            r["token_id"]: r["weight"]
            for r in self._query("SELECT token_id, weight FROM idf")
        }

    def get_config(self, key: str):
        rows = self._query("SELECT value FROM config WHERE key=?", (key,))
        if not rows:
            return None
        try:
            return json.loads(rows[0]["value"])
        except (json.JSONDecodeError, ValueError):
            return rows[0]["value"]

    def set_config(self, key: str, value: Any):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO config VALUES (?,?)",
                (key, json.dumps(value)),
            )

    def insert(self, items: list[dict], bitmaps: dict[str, np.ndarray]):
        pos, neg = bitmaps["pos"], bitmaps["neg"]
        with self._conn() as c:
            start = c.execute(
                "SELECT COALESCE(MAX(hdv_idx),-1)+1 FROM memories"
            ).fetchone()[0]
            c.executemany(
                "INSERT INTO memories "
                "(hdv_idx,id,text,metadata,token_count) VALUES (?,?,?,?,?)",
                [
                    (
                        start + i,
                        it["id"],
                        it["text"],
                        json.dumps(it["metadata"]),
                        it["token_count"],
                    )
                    for i, it in enumerate(items)
                ],
            )
        with open(self.corpus_file, "ab") as f:
            for i in range(len(pos)):
                f.write(pos[i].tobytes())
                f.write(neg[i].tobytes())
        self.set_config("corpus_layout", "interleaved")

    def finalize_index(self):
        n = self.count()
        if n == 0:
            return
        layout = self.get_config("corpus_layout")
        if layout == "blocked":
            return
        d = self._stride
        if (
            not self.corpus_file.exists()
            or self.corpus_file.stat().st_size != n * d * 2
        ):
            return
        if self.logger:
            self.logger.info(f"Reorganizing {n:,} HDVs to blocked layout")
        tmp = self.corpus_file.with_suffix(".tmp")
        with open(self.corpus_file, "rb") as src, open(tmp, "wb") as dst:
            for i in range(n):
                src.seek(i * d * 2)
                dst.write(src.read(d))
            for i in range(n):
                src.seek(i * d * 2 + d)
                dst.write(src.read(d))
        tmp.replace(self.corpus_file)
        self.set_config("corpus_layout", "blocked")

    def search(self, q_pos, q_neg, target, candidates=None, logger=None):
        corpus = self._open_corpus()
        if not corpus:
            return np.array([], np.int32), np.array([], np.float32)
        data, pos_map, neg_map = corpus
        n = pos_map.shape[0]

        corpus_idx = candidates if candidates is not None else np.arange(n)
        nc = len(corpus_idx)
        if nc == 0:
            del data
            return np.array([], np.int32), np.array([], np.float32)

        pos_64, neg_64 = pos_map.view(np.uint64), neg_map.view(np.uint64)
        qp64 = q_pos.ravel().view(np.uint64)
        qn64 = q_neg.ravel().view(np.uint64)
        n_cols = pos_64.shape[1]

        max_passes = max(2, int(np.ceil(np.log2(nc / target)))) if nc > target else 1
        chunk_sz = max(1, n_cols // max_passes)
        cumul = np.zeros(nc, np.float32)
        surv = np.arange(nc)
        prev_top = None
        buf_p = buf_n = None

        for i in range(max_passes):
            s = i * chunk_sz
            e = n_cols if i == max_passes - 1 else (i + 1) * chunk_sz
            cs = corpus_idx[surv]

            if i == 1:
                buf_p = np.ascontiguousarray(pos_64[cs])
                buf_n = np.ascontiguousarray(neg_64[cs])

            if i >= 1:
                cumul[surv] += score64(
                    buf_p[:, s:e], buf_n[:, s:e], qp64[s:e], qn64[s:e]
                )
            else:
                cumul[surv] += score64(
                    pos_64[cs, s:e], neg_64[cs, s:e], qp64[s:e], qn64[s:e]
                )

            k = min(target, len(surv))
            top_idx = np.argpartition(cumul[surv], -k)[-k:]
            curr_top = set(surv[top_idx])

            if prev_top is not None and curr_top == prev_top:
                if logger:
                    logger.info(
                        f"[Prune] {nc:,}\u2192{len(surv):,} stable@{i + 1}/{max_passes}"
                    )
                break
            prev_top = curr_top

            if len(surv) > target:
                keep = max(target, len(surv) // 2)
                idx = np.argpartition(cumul[surv], -keep)[-keep:]
                surv = surv[idx]
                if i >= 1:
                    buf_p, buf_n = buf_p[idx], buf_n[idx]
        else:
            if logger:
                logger.info(f"[Prune] {nc:,}\u2192{len(surv):,} floor@{max_passes}")

        scores = cumul[surv]
        order = np.argsort(-scores)
        del data
        return corpus_idx[surv[order]], scores[order]

    def clear(self):
        with self._conn() as c:
            c.execute("DELETE FROM memories")
            c.execute("DELETE FROM idf")
            c.execute("DELETE FROM config WHERE key != 'hdc_dimensions'")
        if self.corpus_file.exists():
            self.corpus_file.unlink()

    def stats(self) -> dict:
        return {
            "db_mb": self.db_path.stat().st_size / 1e6 if self.db_path.exists() else 0,
            "corpus_hdv_mb": self.corpus_file.stat().st_size / 1e6
            if self.corpus_file.exists()
            else 0,
        }

    def source_counts(self) -> dict:
        return {
            r["src"]: r["n"]
            for r in self._query(
                "SELECT json_extract(metadata,'$.source') as src, "
                "COUNT(*) as n FROM memories GROUP BY src ORDER BY n DESC"
            )
        }

    def compute_sparsity(self) -> dict:
        corpus = self._open_corpus()
        if not corpus:
            return {"positive": 0, "negative": 0, "zero": 1}
        data, pos, neg = corpus
        dims = self.config.hdc_dimensions
        pu = np.unpackbits(pos, axis=1)[:, :dims]
        nu = np.unpackbits(neg, axis=1)[:, :dims]
        total = pu.size
        result = {
            "positive": float(pu.sum() / total),
            "negative": float(nu.sum() / total),
            "zero": float((total - (pu | nu).sum()) / total),
        }
        del data
        return result

    def sample_similarities(self) -> list[float]:
        corpus = self._open_corpus()
        if not corpus:
            return []
        data, pos, neg = corpus
        n = pos.shape[0]
        if n < 2:
            del data
            return []
        max_pairs = n * (n - 1) // 2
        ns = min(max_pairs, int(max_pairs**0.5))
        p64, n64 = pos.view(np.uint64), neg.view(np.uint64)
        ia = np.random.randint(0, n, ns * 2)
        ib = np.random.randint(0, n, ns * 2)
        mask = ia != ib
        ia, ib = ia[mask][:ns], ib[mask][:ns]
        result = score64(p64[ia], n64[ia], p64[ib], n64[ib]).tolist()
        del data
        return result

    def dimension_activation(self) -> tuple[np.ndarray, np.ndarray]:
        corpus = self._open_corpus()
        if not corpus:
            return np.array([]), np.array([])
        data, pos, neg = corpus
        dims = self.config.hdc_dimensions
        pf = np.unpackbits(pos, axis=1)[:, :dims].mean(axis=0)
        nf = np.unpackbits(neg, axis=1)[:, :dims].mean(axis=0)
        del data
        return pf, nf

    def close(self):
        pass


# ── HDC Encoder ────────────────────────────────────────────────────


class HDCEncoder:
    def __init__(self, config: Config, hdrag_dir: Path, db: Database):
        self.config = config
        self.hdrag_dir = hdrag_dir
        self.db = db
        self.proj_path = hdrag_dir / "projection.pt"
        self.idf: dict[int, float] = db.load_idf()
        self.median_doc_length = db.get_config("median_doc_length") or 0.0
        self._emb_dim = db.get_config("emb_dim") or 0
        self._stride = bstride(config.hdc_dimensions)
        self._corpus_vocab: list[int] = []
        self._remap: dict[int, int] = {}
        self._bind_ws: Optional[dict[str, np.ndarray]] = None
        self._bind_ws_shape = (0, 0)
        if self.proj_path.exists():
            tset(
                "projection",
                torch.load(self.proj_path, weights_only=True, map_location="cpu"),
            )
        self._build_remap()

    def _build_remap(self):
        if self.idf:
            self._corpus_vocab = sorted(self.idf.keys())
            self._remap = {tok: i for i, tok in enumerate(self._corpus_vocab)}
            mx = max(self._corpus_vocab) + 1
            self._remap_arr = np.full(mx, -1, dtype=np.int32)
            for tid, idx in self._remap.items():
                self._remap_arr[tid] = idx
        else:
            self._corpus_vocab, self._remap = [], {}
            self._remap_arr = np.zeros(1, dtype=np.int32)
        for a in ("_vocab_f16", "_vocab_ng_u8", "_vocab_ng64", "_vocab_bag"):
            if hasattr(self, a):
                delattr(self, a)

    def _init_projection(self, emb_dim: int):
        proj = tget("projection")
        if proj is not None:
            if proj.shape[0] != emb_dim:
                raise ValueError(
                    f"Projection dim mismatch: {proj.shape[0]} != {emb_dim}"
                )
            return proj
        g = torch.Generator(device="cpu").manual_seed(self.config.hdc_seed)
        proj = torch.randn(
            emb_dim,
            self.config.hdc_dimensions,
            generator=g,
            dtype=torch.float16,
        )
        proj /= proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
        torch.save(proj, self.proj_path)
        return tset("projection", proj)

    def build_vocab_index(self):
        if self.db.vocab_file.exists():
            return
        self._build_remap()
        if not self._corpus_vocab:
            return

        emb = tget("embedding_table")
        if emb is None:
            return
        self._emb_dim = emb.shape[1]
        self.db.set_config("emb_dim", self._emb_dim)
        proj = self._init_projection(emb.shape[1])
        n, dim = self.config.hdc_ngram, self.config.hdc_dimensions
        stride = self._stride
        n_words = stride // 8
        mask64 = mask64_for_dims(dim, stride)

        base = (
            emb[self._corpus_vocab].to(device="cuda", dtype=torch.float16) @ proj.cuda()
        )
        vocab_f16 = base.cpu().numpy()

        tau = emb.shape[1] ** -0.5
        pos = (base > tau).to(torch.uint8).cpu().numpy()
        neg = (base < -tau).to(torch.uint8).cpu().numpy()
        pos_b = np.packbits(pos, axis=1)
        neg_b = np.packbits(neg, axis=1)
        pad = stride - pos_b.shape[1]
        if pad:
            pos_b = np.pad(pos_b, ((0, 0), (0, pad)))
            neg_b = np.pad(neg_b, ((0, 0), (0, pad)))

        pos64 = pos_b.view(np.uint64) & mask64
        neg64 = neg_b.view(np.uint64) & mask64

        g = torch.Generator(device="cpu").manual_seed(self.config.hdc_seed)
        rho_w = torch.randperm(n_words, generator=g).numpy()
        powers_w = [np.arange(n_words, dtype=np.int64)]
        cur = rho_w.copy()
        for _ in range(n - 1):
            powers_w.append(cur.copy())
            cur = cur[rho_w]

        nv = vocab_f16.shape[0]
        uni_bytes = nv * dim * 2
        bits_offset = align8(uni_bytes)
        pad_bytes = bits_offset - uni_bytes

        with open(self.db.vocab_file, "wb") as f:
            f.write(vocab_f16.tobytes())
            if pad_bytes:
                f.write(b"\x00" * pad_bytes)
            for k in range(n):
                p = pos64[:, powers_w[k]].copy()
                q = neg64[:, powers_w[k]].copy()
                f.write(p.view(np.uint8).tobytes())
                f.write(q.view(np.uint8).tobytes())

        del base
        torch.cuda.empty_cache()

    def _ensure_vocab(self):
        if hasattr(self, "_vocab_f16"):
            return
        if self._emb_dim == 0:
            raise ValueError("emb_dim unknown \u2014 run build_vocab_index first")
        if not self.db.vocab_file.exists():
            self.build_vocab_index()
        n, dim = self.config.hdc_ngram, self.config.hdc_dimensions
        stride = self._stride
        nv = len(self._corpus_vocab)
        uni_bytes = nv * dim * 2
        bits_offset = align8(uni_bytes)
        self._vocab_f16 = np.memmap(
            self.db.vocab_file,
            dtype=np.float16,
            mode="r",
            offset=0,
            shape=(nv, dim),
        )
        self._vocab_bag = torch.from_numpy(self._vocab_f16.astype(np.float32))
        bits = np.memmap(
            self.db.vocab_file,
            dtype=np.uint8,
            mode="r",
            offset=bits_offset,
            shape=(n * 2, nv, stride),
        )
        self._vocab_ng_u8 = bits.reshape(n, 2, nv, stride)
        self._vocab_ng64 = self._vocab_ng_u8.view(np.uint64)

    def project(self, token_ids=None, flat_ids=None, offsets=None):
        """Token IDs → continuous HDC unigram vectors (batch, dim) float16."""
        self._ensure_vocab()
        dim = self.config.hdc_dimensions
        if flat_ids is not None:
            batch = len(offsets) - 1
            flat_np = flat_ids.astype(np.int64, copy=False)
            offs = offsets
        elif token_ids:
            batch = len(token_ids)
            arrs = [np.asarray(ids, dtype=np.int64) for ids in token_ids]
            flat_np = np.concatenate(arrs) if len(arrs) > 1 else arrs[0].copy()
            offs = np.zeros(batch + 1, dtype=np.int64)
            np.cumsum([len(a) for a in arrs], out=offs[1:])
        else:
            raise ValueError("provide token_ids or flat_ids+offsets")

        seg_np = np.repeat(np.arange(batch, dtype=np.int64), np.diff(offs))
        clipped = np.clip(flat_np, 0, len(self._remap_arr) - 1)
        remapped = self._remap_arr[clipped]
        mapped = remapped >= 0
        safe = np.clip(remapped, 0, None)

        idf_w = tget("idf_weights")
        safe_t = torch.from_numpy(safe.astype(np.int64))
        mapped_f = torch.from_numpy(mapped).float()
        w = (
            idf_w[torch.from_numpy(flat_np)].float() * mapped_f
            if idf_w is not None
            else mapped_f
        )
        offsets_t = torch.from_numpy(offs.astype(np.int64))
        unigrams = F.embedding_bag(
            safe_t,
            self._vocab_bag,
            offsets_t,
            per_sample_weights=w,
            mode="sum",
            include_last_offset=True,
        )
        w_sums = torch.zeros(batch, dtype=torch.float32)
        w_sums.scatter_add_(0, torch.from_numpy(seg_np), w)
        unigrams /= w_sums.unsqueeze(-1).clamp(1e-6)
        unigrams = F.normalize(unigrams, p=2, dim=1)
        unigrams *= math.sqrt(dim / self._emb_dim)
        return unigrams.half()

    def _ensure_bind_workspace(self, rows: int, n_words: int) -> dict[str, np.ndarray]:
        """Allocate (or grow) reusable uint64 buffers for n-gram bind chain."""
        r0, c0 = self._bind_ws_shape
        if (self._bind_ws is None) or (rows > r0) or (n_words != c0):
            alloc_rows = max(rows, r0)
            shape = (alloc_rows, n_words)
            self._bind_ws = {
                "bp": np.empty(shape, dtype=np.uint64),
                "bn": np.empty(shape, dtype=np.uint64),
                "bufp": np.empty(shape, dtype=np.uint64),
                "bufn": np.empty(shape, dtype=np.uint64),
                "tmp": np.empty(shape, dtype=np.uint64),
                "kp_buf": np.empty(shape, dtype=np.uint64),
                "kn_buf": np.empty(shape, dtype=np.uint64),
            }
            self._bind_ws_shape = (alloc_rows, n_words)
        return {k: v[:rows, :n_words] for k, v in self._bind_ws.items()}

    def release_workspace(self):
        """Free the pre-allocated binding workspace buffers."""
        if self._bind_ws is not None:
            self._bind_ws = None
            self._bind_ws_shape = (0, 0)
            gc.collect()

    def encode(self, unigrams=None, token_ids=None, flat_ids=None, offsets=None):
        if self._emb_dim == 0:
            raise ValueError("emb_dim unknown \u2014 run build_vocab_index first")
        self._ensure_vocab()
        n, dim = self.config.hdc_ngram, self.config.hdc_dimensions
        stride = self._stride
        n_words = stride // 8
        mask64 = mask64_for_dims(dim, stride)
        tau = self._emb_dim**-0.5

        flat_np = seg_np = mapped = safe = offs = None
        if flat_ids is not None:
            flat_np = flat_ids.astype(np.int64, copy=False)
            offs = offsets
        elif token_ids:
            arrs = [np.asarray(ids, dtype=np.int64) for ids in token_ids]
            flat_np = np.concatenate(arrs) if len(arrs) > 1 else arrs[0].copy()
            offs = np.zeros(len(token_ids) + 1, dtype=np.int64)
            np.cumsum([len(a) for a in arrs], out=offs[1:])

        if flat_np is not None:
            seg_np = np.repeat(np.arange(len(offs) - 1, dtype=np.int64), np.diff(offs))
            clipped = np.clip(flat_np, 0, len(self._remap_arr) - 1)
            remapped = self._remap_arr[clipped]
            mapped = remapped >= 0
            safe = np.clip(remapped, 0, None)

        if unigrams is None:
            unigrams = self.project(
                token_ids=token_ids, flat_ids=flat_ids, offsets=offsets
            )
        batch = unigrams.shape[0]

        uni_np = unigrams.float().cpu().numpy()
        u_pos = np.packbits(uni_np > tau, axis=1)
        u_neg = np.packbits(uni_np < -tau, axis=1)
        pad = stride - u_pos.shape[1]
        if pad:
            u_pos = np.pad(u_pos, ((0, 0), (0, pad)))
            u_neg = np.pad(u_neg, ((0, 0), (0, pad)))
        u_p64 = u_pos.view(np.uint64) & mask64
        u_n64 = u_neg.view(np.uint64) & mask64

        out_p64, out_n64 = u_p64, u_n64

        if flat_np is not None and n > 1:
            nw = len(flat_np) - n + 1
            if nw > 0:
                valid = seg_np[:nw] == seg_np[n - 1 :]
                for k in range(n):
                    valid &= mapped[k : k + nw]
                inv = ~valid

                ws = self._ensure_bind_workspace(rows=nw, n_words=n_words)
                bp, bn = ws["bp"], ws["bn"]
                bufp, bufn = ws["bufp"], ws["bufn"]
                tmp, kp_buf, kn_buf = ws["tmp"], ws["kp_buf"], ws["kn_buf"]

                idx0 = safe[:nw]
                np.take(self._vocab_ng64[0, 0], idx0, axis=0, out=bp)
                np.take(self._vocab_ng64[0, 1], idx0, axis=0, out=bn)

                for k in range(1, n):
                    idxk = safe[k : k + nw]
                    np.take(self._vocab_ng64[k, 0], idxk, axis=0, out=kp_buf)
                    np.take(self._vocab_ng64[k, 1], idxk, axis=0, out=kn_buf)
                    bind_ternary_bits_into(bp, bn, kp_buf, kn_buf, bufp, bufn, tmp)
                    bp, bufp = bufp, bp
                    bn, bufn = bufn, bn

                if inv.any():
                    bp[inv] = 0
                    bn[inv] = 0
                bp &= mask64
                bn &= mask64

                coh = np.sqrt(popcount64(bp | bn).sum(axis=1).astype(np.float32) + 1e-6)

                wseg = seg_np[:nw]
                uni_density = (
                    popcount64(u_p64 | u_n64).sum(axis=1).astype(np.float32) / dim
                )
                agg_p = np.zeros((batch, n_words), dtype=np.uint64)
                agg_n = np.zeros((batch, n_words), dtype=np.uint64)

                for bi in range(batch):
                    rows = np.where(wseg == bi)[0]
                    if rows.size == 0:
                        continue
                    doc_coh = coh[rows]
                    thr = adaptive_threshold(torch.from_numpy(doc_coh))
                    sel = rows[doc_coh >= thr] if thr > 0 else rows
                    if sel.size == 0:
                        continue

                    mean_bits = (coh[sel] ** 2).mean()
                    p = mean_bits / dim
                    target = max(uni_density[bi], 0.01)
                    if 0 < p < 1:
                        m_cap = max(
                            1,
                            int(np.ceil(np.log1p(-target) / np.log1p(-p))),
                        )
                        if sel.size > m_cap:
                            top_idx = np.argpartition(coh[sel], -m_cap)[-m_cap:]
                            sel = sel[top_idx]

                    ap = np.bitwise_or.reduce(bp[sel], axis=0)
                    an = np.bitwise_or.reduce(bn[sel], axis=0)
                    coll = ap & an
                    ap &= ~coll
                    an &= ~coll
                    agg_p[bi] = ap
                    agg_n[bi] = an

                agg_p &= mask64
                agg_n &= mask64

                ng_pos = np.unpackbits(
                    agg_p.view(np.uint8).reshape(batch, stride), axis=1
                )[:, :dim].astype(np.float32)
                ng_neg = np.unpackbits(
                    agg_n.view(np.uint8).reshape(batch, stride), axis=1
                )[:, :dim].astype(np.float32)
                agree = (ng_pos * (uni_np > 0)) + (ng_neg * (uni_np < 0))
                boosted = uni_np * (1.0 + agree)

                out_pos = np.packbits(boosted > tau, axis=1)
                out_neg = np.packbits(boosted < -tau, axis=1)
                pad = stride - out_pos.shape[1]
                if pad:
                    out_pos = np.pad(out_pos, ((0, 0), (0, pad)))
                    out_neg = np.pad(out_neg, ((0, 0), (0, pad)))
                out_p64 = out_pos.view(np.uint64) & mask64
                out_n64 = out_neg.view(np.uint64) & mask64

        pos_p = out_p64.view(np.uint8).reshape(batch, stride).copy()
        neg_p = out_n64.view(np.uint8).reshape(batch, stride).copy()
        return {"pos": pos_p, "neg": neg_p}

    def clear(self):
        tdel("projection")
        for a in ("_vocab_f16", "_vocab_ng_u8", "_vocab_ng64", "_vocab_bag"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        for f in (self.db.corpus_file, self.db.vocab_file, self.proj_path):
            if f.exists():
                f.unlink()
        self._corpus_vocab, self._remap = [], {}
        self._remap_arr = np.zeros(1, dtype=np.int32)
        self._bind_ws = None
        self._bind_ws_shape = (0, 0)


# ── Embedding Extractor ───────────────────────────────────────────


class EmbeddingExtractor:
    """Extracts and caches the token embedding table from a GGUF model.

    Only needed during index building. After the vocab index is built,
    the embedding table can be released — search/retrieval never touches it.
    """

    def __init__(self, gguf_path: str, cache_dir: Path, logger: logging.Logger):
        self._gguf_path = gguf_path
        self._emb_path = cache_dir / "embeddings.pt"
        self.logger = logger
        self._loaded = False

    def ensure(self):
        """Load cached embeddings or extract from GGUF."""
        if self._loaded:
            return
        if self._emb_path.exists():
            emb = torch.load(self._emb_path, map_location="cpu", weights_only=True)
            tset("embedding_table", emb)
            self._loaded = True
            self.logger.info(f"Loaded cached embeddings: {emb.shape}")
            return
        self._extract()

    def release(self):
        """Free the embedding table from memory."""
        tdel("embedding_table")
        torch.cuda.empty_cache()
        gc.collect()
        self._loaded = False

    @staticmethod
    def set_idf_weights(idf: dict[int, float], vocab_size: int, special_ids: set[int]):
        """Build IDF weight tensor and store in tensor registry."""
        w = torch.ones(vocab_size, dtype=torch.float32)
        for tid, weight in idf.items():
            if 0 <= tid < vocab_size:
                w[tid] = weight
        for tid in special_ids:
            if 0 <= tid < vocab_size:
                w[tid] = 0.0
        tset("idf_weights", w)

    def _extract(self):
        self.logger.info(f"Extracting embeddings from {self._gguf_path}")
        reader = GGUFReader(str(self._gguf_path))
        for t in reader.tensors:
            if t.name != "token_embd.weight":
                continue
            shape = tuple(int(x) for x in reversed(t.shape))
            raw = np.array(t.data)
            qn = t.tensor_type.name

            if qn == "F32":
                data = raw.view(np.float32).copy().reshape(shape)
            elif qn == "F16":
                data = raw.view(np.float16).copy().reshape(shape).astype(np.float32)
            elif qn == "BF16":
                u16 = raw.view(np.uint16).reshape(shape)
                data = np.zeros(shape, dtype=np.float32)
                data.view(np.uint32)[:] = u16.astype(np.uint32) << 16
            elif qn == "Q8_0":
                nb = int(np.prod(shape)) // 32
                blk = raw[: nb * 34].reshape(nb, 34)
                d = np.frombuffer(blk[:, :2].copy().tobytes(), dtype=np.float16).astype(
                    np.float32
                )
                qs = blk[:, 2:].copy().view(np.int8).astype(np.float32)
                data = (qs * d[:, None]).reshape(shape)
            elif qn == "Q4_0":
                nb = int(np.prod(shape)) // 32
                blk = raw[: nb * 18].reshape(nb, 18)
                d = (
                    np.frombuffer(blk[:, :2].copy().tobytes(), dtype=np.float16)
                    .astype(np.float32)
                    .reshape(nb, 1)
                )
                qs = blk[:, 2:]
                lo = (qs & 0xF).astype(np.float32) - 8.0
                hi = (qs >> 4).astype(np.float32) - 8.0
                vals = np.empty((nb, 32), dtype=np.float32)
                vals[:, :16] = d * lo
                vals[:, 16:] = d * hi
                data = vals.reshape(shape)
            elif qn == "Q6_K":
                nb = int(np.prod(shape)) // 256
                blk = raw[: nb * 210].reshape(nb, 210)
                ql = blk[:, :128]
                qh = blk[:, 128:192]
                sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32)
                d = (
                    np.frombuffer(blk[:, 208:210].copy().tobytes(), dtype=np.float16)
                    .astype(np.float32)
                    .reshape(nb, 1)
                )
                data = np.zeros((nb, 256), dtype=np.float32)
                for half in range(2):
                    ql_h = ql[:, half * 64 : (half + 1) * 64].astype(np.int32)
                    qh_h = qh[:, half * 32 : (half + 1) * 32].astype(np.int32)
                    s = half * 8
                    q1 = ((ql_h[:, :32] & 0xF) | (((qh_h >> 0) & 3) << 4)).astype(
                        np.float32
                    ) - 32
                    q2 = ((ql_h[:, 32:] & 0xF) | (((qh_h >> 2) & 3) << 4)).astype(
                        np.float32
                    ) - 32
                    q3 = ((ql_h[:, :32] >> 4) | (((qh_h >> 4) & 3) << 4)).astype(
                        np.float32
                    ) - 32
                    q4 = ((ql_h[:, 32:] >> 4) | (((qh_h >> 6) & 3) << 4)).astype(
                        np.float32
                    ) - 32
                    b = half * 128
                    data[:, b : b + 32] = d * sc[:, s : s + 1] * q1
                    data[:, b + 32 : b + 64] = d * sc[:, s + 2 : s + 3] * q2
                    data[:, b + 64 : b + 96] = d * sc[:, s + 4 : s + 5] * q3
                    data[:, b + 96 : b + 128] = d * sc[:, s + 6 : s + 7] * q4
                data = data.reshape(shape)
            else:
                try:
                    from gguf.quants import dequantize

                    data = dequantize(raw, qn).reshape(shape)
                    self.logger.info(f"Dequantized {qn} via gguf.quants")
                except (ImportError, Exception) as e:
                    raise ValueError(
                        f"Unsupported quantization: {qn} "
                        f"(gguf.quants fallback failed: {e})"
                    )

            emb = torch.from_numpy(data.copy()).float()
            torch.save(emb, self._emb_path)
            tset("embedding_table", emb)
            self._loaded = True
            self.logger.info(f"Extracted embeddings: {emb.shape} ({qn}\u2192fp32)")
            return
        raise KeyError("token_embd.weight not found in GGUF")


# ── Retriever ──────────────────────────────────────────────────────


class Retriever:
    def __init__(
        self,
        db: Database,
        hdc: HDCEncoder,
        tokenizer: Tokenizer,
        config: Config,
        logger: logging.Logger,
    ):
        self.db, self.hdc = db, hdc
        self._tokenizer = tokenizer
        self.config, self.logger = config, logger
        self._turns: list[tuple[np.ndarray, np.ndarray]] = []
        self._token_counts = db.get_token_counts()

    def add_turn(self, text: str):
        """Encode a model response and store its ternary bitvector.

        Only model responses are tracked — they represent where the
        conversation *actually went*, not where the user pointed it.
        The current query is already the search target; blending
        previous queries would feed the retrieval system's own
        input back to itself (echo), not the model's expansion
        of the topic (resonance).
        """
        tids = self._tokenizer.bulk_tokenize([text])
        bm = self.hdc.encode(token_ids=tids)
        self._turns.append((bm["pos"][0].copy(), bm["neg"][0].copy()))

    def clear_turns(self):
        self._turns.clear()

    def _ternary_blend(self, query_bv: tuple, content_count: int) -> tuple:
        """Blend current query with previous model responses.

        Operates entirely in bitvector space — no float→ternary
        bottleneck. The query's zero (abstain) dimensions let response
        history signal propagate unopposed; the query's lit dimensions
        carry its specific intent. Adaptive weighting: short referential
        queries lean on response history, substantive queries stand alone.

        Response-only tracking means decay=0.5 loses one weight level
        per exchange instead of two, doubling effective memory horizon.
        """
        if not self._turns:
            return query_bv

        dim = self.hdc.config.hdc_dimensions
        ngram = self.hdc.config.hdc_ngram
        query_weight = min(0.5, content_count / (ngram * 2))

        decay = 0.5
        n = len(self._turns)
        hist_weights = [decay ** (n - i) for i in range(n)]
        hist_total = sum(hist_weights)

        # Normalize: response history gets (1 - query_weight), query gets query_weight
        if hist_total > 1e-6:
            hist_scale = (1.0 - query_weight) / hist_total
            hist_weights = [w * hist_scale for w in hist_weights]
        weights = hist_weights + [query_weight]

        self.logger.info(
            f"[Blend] content={content_count} ngram={ngram} "
            f"qw={query_weight:.2f} responses={n}"
        )

        # Accumulate weighted votes: +1, 0, -1 per dimension
        accum = np.zeros(dim, dtype=np.float32)
        all_bv = list(self._turns) + [query_bv]
        for (pos_u8, neg_u8), w in zip(all_bv, weights):
            p = np.unpackbits(pos_u8)[:dim].astype(np.float32)
            n_bits = np.unpackbits(neg_u8)[:dim].astype(np.float32)
            accum += w * (p - n_bits)

        # Threshold back to ternary
        stride = self.hdc._stride
        out_pos = np.packbits((accum > 0).astype(np.uint8))
        out_neg = np.packbits((accum < 0).astype(np.uint8))
        # Pad to stride
        if len(out_pos) < stride:
            out_pos = np.pad(out_pos, (0, stride - len(out_pos)))
            out_neg = np.pad(out_neg, (0, stride - len(out_neg)))
        # Mask padding bits
        mask64 = mask64_for_dims(dim, stride)
        out_p = (out_pos[:stride].view(np.uint64) & mask64).view(np.uint8)
        out_n = (out_neg[:stride].view(np.uint64) & mask64).view(np.uint8)
        return (out_p, out_n)

    def _dedup(self, results: list[dict]) -> list[dict]:
        """Fast textual subset dedup — catches exact containment."""
        n = len(results)
        if n <= 1:
            return results
        sets = [set(r["memory"]["text"].lower().split()) for r in results]
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                if sets[i] <= sets[j]:
                    keep[i] = False
                    break
                elif sets[j] <= sets[i]:
                    keep[j] = False
        out = [r for i, r in enumerate(results) if keep[i]]
        if len(out) < n:
            self.logger.info(f"[Dedup] subset: {n}\u2192{len(out)}")
        return out

    def _hdc_mmr(self, results: list[dict], q_pos, q_neg, budget: int) -> list[dict]:
        """Coverage-aware greedy selection in packed ternary HDC space.

        Marginal score = retrieval_score × (novelty / null_novelty)

        null_novelty = 1 - d_c is the expected novelty of a vector
        statistically independent of the current coverage accumulator.
        The ratio acts as a likelihood test: >1 means the candidate
        is more novel than chance → score preserved or boosted,
        <1 means more redundant than chance → score suppressed.

        No λ parameter — coverage density IS the adaptive tradeoff.
        Early selections (sparse coverage) → near-pure relevance.
        Late selections (dense coverage) → diversity dominates.
        Collision zeroing on the accumulator creates natural back-
        pressure: ambiguous dimensions don't count as covered, so
        the system stays in relevance mode longer when the selected
        set contains contradictory signals.
        """
        n = len(results)
        if n <= 1:
            return results

        dim = self.config.hdc_dimensions
        indices = [r["memory"]["hdv_idx"] for r in results]
        tcounts = [r["memory"].get("token_count", 0) for r in results]
        scores = np.array([r["hdc_score"] for r in results], dtype=np.float32)

        dp, dn = self.db.get_bitmaps(indices)
        dp64, dn64 = dp.view(np.uint64), dn.view(np.uint64)
        mask = mask64_for_dims(dim, self.hdc._stride).ravel()

        # Precompute per-candidate lit dimensions
        cand_lit = np.array(
            [popcount64((dp64[i] & mask) | (dn64[i] & mask)).sum() for i in range(n)],
            dtype=np.float32,
        )

        cov_p = np.zeros_like(dp64[0])
        cov_n = np.zeros_like(dn64[0])

        alive = np.ones(n, dtype=bool)
        accepted, total = [], 0

        while alive.any():
            cov_lit = popcount64(cov_p | cov_n).sum()
            d_c = cov_lit / dim
            # Null model: expected novelty of an independent vector
            # Floor at 1/dim avoids division by zero at full saturation
            null_novelty = max(1.0 - d_c, 1.0 / dim)

            # Score all alive candidates against current coverage
            marginal = np.full(n, -np.inf, dtype=np.float32)
            for i in np.where(alive)[0]:
                if total + tcounts[i] > budget:
                    alive[i] = False
                    continue
                if cand_lit[i] == 0:
                    alive[i] = False
                    continue
                cp = dp64[i] & mask
                cn = dn64[i] & mask
                novel = popcount64((cp | cn) & ~(cov_p | cov_n)).sum()
                novelty = novel / cand_lit[i]
                marginal[i] = scores[i] * (novelty / null_novelty)

            best = np.argmax(marginal)
            if marginal[best] <= 0:
                break

            accepted.append(results[best])
            total += tcounts[best]
            alive[best] = False

            # Update coverage — collision zeroing preserves ternary invariant
            cov_p |= dp64[best] & mask
            cov_n |= dn64[best] & mask
            coll = cov_p & cov_n
            cov_p &= ~coll
            cov_n &= ~coll

        final_cov = popcount64(cov_p | cov_n).sum() / dim
        self.logger.info(
            f"[HDC-MMR] {n}\u2192{len(accepted)} "
            f"tokens={total:,} cov={final_cov:.1%}"
        )
        return accepted

    def search(
        self,
        query: str,
        token_budget: int,
        track: bool = True,
        enabled_sources: set = None,
    ) -> list[dict]:
        if self.db.count() == 0:
            return []
        self.logger.info(f"[Search] Query: {len(query)} chars")

        tids = self._tokenizer.bulk_tokenize([query])
        qe = self.hdc.project(token_ids=tids)
        bm = self.hdc.encode(unigrams=qe, token_ids=tids)

        if track and self._turns:
            # Count content tokens for adaptive weight
            flat = np.asarray(tids[0], dtype=np.int64)
            remap = self.hdc._remap_arr
            clipped = np.clip(flat, 0, len(remap) - 1)
            content_count = int((remap[clipped] >= 0).sum())

            query_bv = (bm["pos"][0].copy(), bm["neg"][0].copy())
            blended = self._ternary_blend(query_bv, content_count)

            # Query is NOT stored — only model responses go into _turns
            # via add_turn() called from the UI after generation.

            # Replace bm with blended for search
            bm["pos"][0] = blended[0]
            bm["neg"][0] = blended[1]

        # First turn with no response history: search unblended.
        # Response gets added to _turns after generation via add_turn().

        max_doc = token_budget // self.config.min_context
        eligible = np.where(self._token_counts <= max_doc)[0]
        mdl = max(1, int(np.median(self._token_counts[eligible])))

        # Widen pruning target: hdc_ngram multiplier gives _hdc_mmr a
        # deeper candidate pool to reorder. The pruning stage is cheap
        # (packed bitwise scan), the expensive part is bitmap fetch +
        # coverage loop in _hdc_mmr which is O(target) not O(corpus).
        target = (
            max(1, int((token_budget * self.config.min_context) // mdl))
            * self.config.hdc_ngram
        )

        indices, scores = self.db.search(
            bm["pos"], bm["neg"], target, eligible, self.logger
        )
        thr = adaptive_threshold(torch.from_numpy(scores))
        pre = len(indices)

        if thr and thr != 0.0:
            m = scores >= thr
            indices, scores = indices[m], scores[m]

        self.logger.info(
            f"[Search] {len(eligible):,}\u2192{pre}\u2192{len(indices)} "
            f"(prune\u2192threshold)"
            + (f" thr={thr:.1f}" if thr and thr != 0.0 else "")
        )

        if len(indices) == 0:
            return []

        sel = indices.tolist()
        smap = dict(zip(sel, scores.tolist()))
        mems = self.db.get_memories(sel)

        if len(mems) < len(sel):
            self.logger.info(f"[Search] get_memories: {len(sel)}\u2192{len(mems)}")

        if enabled_sources:
            pre_src = len(mems)
            mems = {
                i: m
                for i, m in mems.items()
                if m.get("metadata", {}).get("source") in enabled_sources
            }
            if len(mems) < pre_src:
                self.logger.info(f"[Search] source filter: {pre_src}\u2192{len(mems)}")

        results = [
            {"memory": mems[i], "hdc_score": smap.get(i, 0.0)} for i in sel if i in mems
        ]
        results = self._dedup(results)
        if len(results) > 1:
            results = self._hdc_mmr(results, bm["pos"], bm["neg"], token_budget)
        return results


# ── HdRAG Orchestrator ────────────────────────────────────────────


class HdRAG:
    """Hyperdimensional RAG memory engine.

    Accepts any object satisfying the Tokenizer protocol (typically
    an InferenceEngine instance). Has zero knowledge of the inference
    pipeline, Gradio UI, or conversation logging.
    """

    def __init__(
        self,
        config: Config,
        tokenizer: Tokenizer,
        gguf_path: str = "",
        logger: logging.Logger = None,
    ):
        self.config = config
        self._tokenizer = tokenizer
        self._gguf_path = gguf_path
        self.logger = logger or logging.getLogger(__name__)

        self._hdrag_dir = Path(config.hdrag_dir)
        self._hdrag_dir.mkdir(parents=True, exist_ok=True)

        self.db = Database(self._hdrag_dir / "index.db", config, self.logger)
        self.hdc = HDCEncoder(config, self._hdrag_dir, self.db)
        self.retriever = Retriever(self.db, self.hdc, tokenizer, config, self.logger)
        self.enabled_sources: set[str] = set(self.db.source_counts())

        # Load IDF weights into tensor registry if index exists
        if self.hdc.idf:
            EmbeddingExtractor.set_idf_weights(
                self.hdc.idf,
                tokenizer.vocab_size,
                tokenizer.special_ids,
            )

        self.logger.info(f"HdRAG initialized: {self.db.count():,} memories")

    # ── Retrieval ──

    def search(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> list[dict]:
        return self.retriever.search(
            query,
            token_budget or self.config.max_context_tokens,
            track,
            self.enabled_sources,
        )

    def get_context(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> str:
        return "\n\n---\n\n".join(
            r["memory"]["text"] for r in self.search(query, token_budget, track)
        )

    # ── Conversation state ──

    def add_turn(self, text: str):
        self.retriever.add_turn(text)

    def clear_turns(self):
        self.retriever.clear_turns()

    # ── Source filtering ──

    def source_counts(self) -> dict[str, int]:
        return self.db.source_counts()

    # ── Introspection ──

    @property
    def count(self) -> int:
        return self.db.count()

    def stats(self) -> dict:
        return {
            "memories": self.db.count(),
            "vocab_size": len(self.hdc.idf),
            "median_tokens": self.hdc.median_doc_length,
            "hdc_dims": self.config.hdc_dimensions,
            "hdc_ngram": self.config.hdc_ngram,
            "model": Path(self.config.gguf_model).stem,
            "sources": self.db.source_counts(),
            "tensors": list(_T.keys()),
            "db": self.db.stats(),
        }

    # ── Resource management ──

    def trim_memory(self, *child_procs):
        gc.collect()
        trim_working_set(*child_procs)

    # ── Indexing ──

    def clear_index(self, *child_procs):
        gc.collect()
        trim_working_set(*child_procs)
        self.db.clear()
        self.hdc.clear()
        self.logger.info("Index cleared")

    def build_index(self, progress_cb: Callable = None) -> int:
        """Full rebuild: clear existing index, scan datasets, encode, store.

        Returns number of documents indexed.
        """
        if not self._gguf_path:
            raise FileNotFoundError(
                "GGUF path required for embedding extraction during indexing"
            )

        emb = EmbeddingExtractor(self._gguf_path, self._hdrag_dir, self.logger)
        emb.ensure()

        files = discover_datasets(Path(self.config.datasets_dir))
        if not files:
            emb.release()
            return 0

        self.logger.info("Clearing existing index...")
        for a in ("_vocab_f16", "_vocab_ng_u8", "_vocab_ng64", "_vocab_bag"):
            if hasattr(self.hdc, a):
                delattr(self.hdc, a)
        gc.collect()
        for f in (self.db.corpus_file, self.db.vocab_file):
            if f.exists():
                f.unlink()
        self.db.clear()

        self.logger.info("Pass 1: Building vocabulary...")

        # Start server in CPU-only mode for tokenization (no GPU memory)
        if hasattr(self._tokenizer, "start_for_indexing"):
            self._tokenizer.start_for_indexing()

        special = self._tokenizer.special_ids
        vocab_df: Counter = Counter()
        docs: list[dict] = []

        for fi, f in enumerate(files):
            path, name = Path(f["path"]), f["name"]
            count, pending = 0, []
            for item in iter_dataset(
                path,
                self._tokenizer,
                self.config.max_context_tokens,
            ):
                if "chunk_idx" in item:
                    text = item["text"].strip()
                    meta = {
                        "chunk_idx": item["chunk_idx"],
                        "total_chunks": item.get("total_chunks"),
                    }
                else:
                    text, meta = extract_text(item).strip(), None
                if text:
                    pending.append((text, meta))
                if len(pending) >= self.config.batch_size * 4:
                    count += self._ingest(pending, name, special, vocab_df, docs)
                    pending = []
            if pending:
                count += self._ingest(pending, name, special, vocab_df, docs)
            self.logger.info(f"  [{name}] {count:,} records")
            if progress_cb:
                progress_cb((fi + 1) / len(files) * 0.3, f"Pass 1: {name}")

        seen, unique = set(), []
        for d in docs:
            if d["id"] not in seen:
                seen.add(d["id"])
                unique.append(d)
        docs = unique
        if not docs:
            self.logger.info("No new documents")
            emb.release()
            return 0

        self.logger.info(f"Full rebuild: {len(docs):,} documents")

        # Free GPU memory — tokenization is done, encoding needs the GPU.
        # Server restarts automatically on next tokenize() or generate().
        if hasattr(self._tokenizer, "stop_server"):
            self._tokenizer.stop_server()

        self.db.set_config("hdc_seed", self.config.hdc_seed)
        self.db.set_config("hdc_ngram", self.config.hdc_ngram)

        n_docs = len(docs)
        idf = {tid: math.log((n_docs + 1) / (df + 1)) for tid, df in vocab_df.items()}
        self.db.save_idf(dict(vocab_df), n_docs)
        self.db.set_config(
            "median_doc_length",
            statistics.median(d["token_count"] for d in docs),
        )

        all_tids = np.concatenate([d["token_ids"] for d in docs])
        tid_offsets = np.zeros(len(docs) + 1, dtype=np.int64)
        np.cumsum([len(d["token_ids"]) for d in docs], out=tid_offsets[1:])
        for d in docs:
            d.pop("token_ids", None)
        self.logger.info(
            f"Token IDs: {len(all_tids):,} tokens, {all_tids.nbytes / 1e6:.0f}MB"
        )

        self.hdc.idf = idf
        self.hdc.median_doc_length = self.db.get_config("median_doc_length")
        EmbeddingExtractor.set_idf_weights(
            idf, self._tokenizer.vocab_size, self._tokenizer.special_ids
        )
        self.hdc._build_remap()
        self.logger.info(
            f"Corpus vocab: {len(self.hdc._corpus_vocab):,}"
            f"/{self._tokenizer.vocab_size:,}"
        )
        self.hdc.build_vocab_index()

        emb.release()
        self.logger.info("Freed embedding table")

        self.logger.info(f"Pass 2: Encoding (ngram={self.config.hdc_ngram})...")
        bs = self.config.batch_size
        nb = (len(docs) + bs - 1) // bs

        for i, start in enumerate(range(0, len(docs), bs)):
            end = min(start + bs, len(docs))
            batch_flat = all_tids[tid_offsets[start] : tid_offsets[end]]
            batch_offs = tid_offsets[start : end + 1] - tid_offsets[start]
            bitmaps = self.hdc.encode(flat_ids=batch_flat, offsets=batch_offs)
            self.db.insert(
                [
                    {k: d[k] for k in ("id", "text", "metadata", "token_count")}
                    for d in docs[start:end]
                ],
                bitmaps,
            )
            if (i + 1) % self.config.batch_log_interval == 0 or i + 1 == nb:
                self.logger.info(
                    f"  Batch {i + 1:,}/{nb:,} ({100 * (i + 1) / nb:.0f}%)"
                )
            if (i + 1) % self.config.batch_log_interval == 0:
                gc.collect()
            if progress_cb:
                progress_cb(0.3 + (i + 1) / nb * 0.7, f"Pass 2: batch {i + 1}")

        del all_tids, tid_offsets
        self.db.finalize_index()
        self.retriever._token_counts = self.db.get_token_counts()
        self.hdc.release_workspace()
        self.enabled_sources = set(self.db.source_counts())
        gc.collect()
        self.logger.info(f"Index complete: {self.db.count():,} memories")
        return len(docs)

    def _ingest(self, items, source, special, vocab_df, docs) -> int:
        texts, metas = zip(*items)
        ids = [
            hashlib.blake2b(
                f"{source}:{'chunk' + str(m['chunk_idx']) + ':' if m and 'chunk_idx' in m else ''}{t}".encode(),
                digest_size=8,
            ).hexdigest()
            for t, m in zip(texts, metas)
        ]

        existing = self.db.exists(ids)
        new = [
            (mid, txt, meta)
            for mid, txt, meta in zip(ids, texts, metas)
            if mid not in existing
        ]
        if not new:
            return 0

        new_texts = [txt for _, txt, _ in new]
        all_toks = self._tokenizer.bulk_tokenize(new_texts)

        for (mid, text, meta), toks in zip(new, all_toks):
            if not toks:
                continue  # tokenization failed (500) — skip
            vocab_df.update(set(toks) - special)
            doc_meta = {"source": source}
            if meta:
                doc_meta.update(meta)
            docs.append(
                {
                    "id": mid,
                    "text": text,
                    "metadata": doc_meta,
                    "token_count": len(toks),
                    "token_ids": np.array(toks, dtype=np.int32),
                }
            )
        return len(new)

    def close(self):
        self.db.close()

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()
