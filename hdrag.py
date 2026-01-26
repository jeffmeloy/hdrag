"""
hdrag - Hyperdimensional Retrieval-Augmented Generation
Usage: python hdrag.py [--config hdrag_config.yaml] [--debug]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sqlite3
import statistics
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Generator, Optional

import pandas as pd
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


@dataclass
class Config:
    chat_history_dir: str
    hdrag_dir: str
    datasets_dir: str
    model_dir: str
    model_name: str
    gguf_name: str
    llama_server_url: str
    temperature: float
    top_p: float
    max_new_tokens: int
    max_length_tokens: int
    hdc_dimensions: int
    hdc_seed: int
    batch_size: int
    vocab_chunk_multiplier: int
    hash_digest_size: int
    export_log_interval: int
    batch_log_interval: int
    text_chunk_size: int
    text_chunk_overlap: int
    max_context_tokens: int
    min_context: int
    sqlite_max_vars: int
    sqlite_cache_kb: int
    gradio_port: int
    system_prompt: str

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        if not Path(path).exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        missing = valid_fields - set(filtered.keys())
        if missing:
            raise ValueError(f"Missing config fields: {missing}")
        return cls(**filtered)


def chunks(xs: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def otsu_threshold(vals: torch.Tensor) -> float:
    if vals.numel() < 2:
        return None
    x, _ = torch.sort(vals)
    n = x.numel()
    sum_total = torch.sum(x)
    sum_left = torch.cumsum(x, dim=0)
    w0 = torch.arange(1, n, device=x.device) / n
    w1 = 1.0 - w0
    m0 = sum_left[:-1] / torch.arange(1, n, device=x.device)
    m1 = (sum_total - sum_left[:-1]) / torch.arange(n - 1, 0, -1, device=x.device)
    scores = w0 * w1 * (m0 - m1).pow(2)
    if scores.max() == 0:
        return 0.0
    max_idx = torch.argmax(scores)
    return ((x[max_idx] + x[max_idx + 1]) / 2.0).item()


def compress_context(text: str) -> str:
    if not text:
        return text
    words = text.split()
    if not words:
        return text
    word_freq = Counter(words)
    word_importance = {w: 1 / freq for w, freq in word_freq.items()}
    threshold = statistics.median(word_importance.values())
    compressed = [w for w in words if word_importance[w] >= threshold]
    return " ".join(compressed)


def discover_datasets(directory: Path) -> list:
    datasets = []
    if not directory.exists():
        return datasets
    if not directory.is_dir():
        if directory.suffix in [
            ".json",
            ".jsonl",
            ".parquet",
            ".txt",
            ".md",
            ".html",
            ".xml",
        ]:
            return [{"name": directory.stem, "path": str(directory)}]
        return datasets
    for fp in sorted(directory.glob("**/*")):
        if fp.suffix in [".json", ".jsonl", ".parquet", ".txt", ".md", ".html", ".xml"]:
            datasets.append({"name": fp.stem, "path": str(fp)})
    return datasets


def iter_dataset(
    path: Path, tokenizer=None, chunk_size: int = 1024, chunk_overlap: int = 128
) -> Generator[dict, None, None]:
    if path.suffix in [".txt", ".md", ".html", ".xml"]:
        text = path.read_text(encoding="utf-8")
        if tokenizer and chunk_size:
            tokens = tokenizer.encode(text)
            step = chunk_size - chunk_overlap
            for i, start in enumerate(range(0, len(tokens), step)):
                chunk_tokens = tokens[start : start + chunk_size]
                if len(chunk_tokens) < chunk_overlap and i > 0:
                    break
                yield {
                    "text": tokenizer.decode(chunk_tokens),
                    "chunk_idx": i,
                    "total_chunks": (len(tokens) + step - 1) // step,
                }
        else:
            yield {"text": text}
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        for record in df.to_dict(orient="records"):
            yield {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in record.items()
            }
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:
            yield data
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def extract_text(item: dict) -> str:
    if (
        isinstance(item, list)
        and item
        and isinstance(item[0], dict)
        and "from" in item[0]
    ):
        msgs = [
            m.get("value", m.get("content", ""))
            for m in item
            if m.get("from", m.get("role", "")) != "system"
        ]
        return "\n\n".join(m for m in msgs if m)

    if "conversations" in item or "conversation" in item:
        conv = item.get("conversations", item.get("conversation", []))
        msgs = [
            m.get("value", m.get("content", ""))
            for m in conv
            if m.get("from", m.get("role", "")) != "system"
        ]
        return "\n\n".join(m for m in msgs if m)

    if "messages" in item:
        msgs = [
            m.get("content", "") for m in item["messages"] if m.get("role") != "system"
        ]
        return "\n\n".join(m for m in msgs if m)

    human_keys = [
        "human",
        "user",
        "prompt",
        "problem",
        "question",
        "instruction",
        "message_1",
    ]
    assistant_keys = ["gpt", "assistant", "response", "answer", "output", "message_2"]
    parts = []
    for k in human_keys:
        if item.get(k):
            parts.append(item[k])
            break
    for k in assistant_keys:
        if item.get(k):
            parts.append(item[k])
            break
    if parts:
        if item.get("input"):
            parts.insert(1, item["input"])
        return "\n\n".join(parts)

    return item.get("text", item.get("content", ""))


class CUDAManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"
        self._resident: dict[str, torch.Tensor] = {}

    def to_device(
        self, tensor: torch.Tensor, non_blocking: bool = True
    ) -> torch.Tensor:
        if tensor.device == self.device:
            return tensor
        return tensor.to(self.device, non_blocking=non_blocking)

    def register(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        gpu_tensor = self.to_device(tensor)
        self._resident[name] = gpu_tensor
        return gpu_tensor

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._resident.get(name)

    def unregister(self, name: str) -> None:
        if name in self._resident:
            del self._resident[name]

    def clear(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()

    @property
    def memory_stats(self) -> dict:
        if not self.is_cuda:
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "resident_tensors": list(self._resident.keys()),
        }


CUDA = CUDAManager()


class Database:
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS memories (
            hdv_idx INTEGER PRIMARY KEY,
            id TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL,
            metadata JSON,
            token_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS idf (
            token_id INTEGER PRIMARY KEY,
            doc_freq INTEGER NOT NULL,
            weight REAL NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_memory_id ON memories(id);
    """

    def __init__(
        self, db_path: Path, config: Config, logger: Optional[logging.Logger] = None
    ):
        self.db_path = db_path
        self.config = config
        self.corpus_hdv_file = db_path.parent / "corpus_hdc.idx"
        self.logger = logger

        # 8-byte aligned stride for uint64 view compatibility
        d = (config.hdc_dimensions + 7) // 8
        self._bitmap_stride = d + (8 - d % 8) % 8

        self._init_db()

        self._m1 = np.uint64(0x5555555555555555)
        self._m2 = np.uint64(0x3333333333333333)
        self._m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
        self._h01 = np.uint64(0x0101010101010101)

    def _bytes_per_bitmap(self) -> int:
        d = (self.config.hdc_dimensions + 7) // 8
        return d + (8 - d % 8) % 8

    def _init_db(self) -> None:
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
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute(f"PRAGMA cache_size=-{self.config.sqlite_cache_kb}")
        c.execute("PRAGMA temp_store=MEMORY")
        try:
            yield c
            c.commit()
        except:
            c.rollback()
            raise
        finally:
            c.close()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(sql, params).fetchall()

    def _chunked_query(self, sql_template: str, ids: list, extractor: Callable) -> dict:
        result = {}
        for chunk in chunks(ids, self.config.sqlite_max_vars):
            ph = ",".join("?" * len(chunk))
            for row in self._query(sql_template.format(ph), tuple(chunk)):
                k, v = extractor(row)
                result[k] = v
        return result

    def count(self) -> int:
        return self._query("SELECT COUNT(*) FROM memories")[0][0]

    def exists(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        found = self._chunked_query(
            "SELECT id FROM memories WHERE id IN ({})", ids, lambda r: (r["id"], True)
        )
        return set(found.keys())

    def get_memories(self, indices: list[int]) -> dict[int, dict]:
        if not indices:
            return {}
        return self._chunked_query(
            "SELECT hdv_idx, id, text, metadata, token_count FROM memories WHERE hdv_idx IN ({})",
            indices,
            lambda r: (
                r["hdv_idx"],
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    "token_count": r["token_count"],
                    "hdv_idx": r["hdv_idx"],
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
        if not indices or not self.corpus_hdv_file.exists():
            d = self._bytes_per_bitmap()
            return np.empty((0, d), dtype=np.uint8), np.empty((0, d), dtype=np.uint8)
        n = self.count()
        d = self._bytes_per_bitmap()
        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")
        half = n * d
        idx = np.array(indices)
        pos_bits = hdv_data[:half].reshape(n, d)
        neg_bits = hdv_data[half:].reshape(n, d)
        result_pos = pos_bits[idx].copy()
        result_neg = neg_bits[idx].copy()
        del hdv_data
        return result_pos, result_neg

    def save_idf(self, df_counts: dict[int, int], n_docs: int) -> None:
        idf_values = [
            (tid, df, math.log((n_docs + 1) / (df + 1)))
            for tid, df in df_counts.items()
        ]
        with self._conn() as c:
            c.execute("DELETE FROM idf")
            c.executemany("INSERT INTO idf VALUES (?,?,?)", idf_values)

    def load_idf(self) -> dict[int, float]:
        return {
            r["token_id"]: r["weight"]
            for r in self._query("SELECT token_id, weight FROM idf")
        }

    def get_config(self, key: str) -> Optional[Any]:
        rows = self._query("SELECT value FROM config WHERE key=?", (key,))
        if not rows:
            return None
        try:
            return json.loads(rows[0]["value"])
        except (json.JSONDecodeError, ValueError):
            return rows[0]["value"]

    def set_config(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO config VALUES (?,?)", (key, json.dumps(value))
            )

    def insert(self, items: list[dict], bitmaps: dict[str, np.ndarray]) -> None:
        pos, neg = bitmaps["pos"], bitmaps["neg"]
        with self._conn() as c:
            start = c.execute(
                "SELECT COALESCE(MAX(hdv_idx),-1)+1 FROM memories"
            ).fetchone()[0]
            c.executemany(
                "INSERT INTO memories (hdv_idx, id, text, metadata, token_count) VALUES (?,?,?,?,?)",
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
        with open(self.corpus_hdv_file, "ab") as f:
            for i in range(len(pos)):
                f.write(pos[i].tobytes())
                f.write(neg[i].tobytes())

    def finalize_index(self) -> None:
        n = self.count()
        if n == 0:
            return
        d = self._bytes_per_bitmap()
        expected = n * d * 2
        if (
            not self.corpus_hdv_file.exists()
            or self.corpus_hdv_file.stat().st_size != expected
        ):
            return
        if self.logger:
            self.logger.info(f"Reorganizing {n:,} HDVs to blocked layout")

        temp_file = self.corpus_hdv_file.with_suffix(".tmp")
        with open(self.corpus_hdv_file, "rb") as src, open(temp_file, "wb") as dst:
            for i in range(n):
                src.seek(i * d * 2)
                dst.write(src.read(d))
            for i in range(n):
                src.seek(i * d * 2 + d)
                dst.write(src.read(d))
        temp_file.replace(self.corpus_hdv_file)

    def search(
        self,
        q_pos: np.ndarray,
        q_neg: np.ndarray,
        target: int,
        candidates: np.ndarray = None,
        logger=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.corpus_hdv_file.exists():
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        n = self.count()
        if n == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        corpus_idx = candidates if candidates is not None else np.arange(n)
        n_candidates = len(corpus_idx)

        if n_candidates == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        n_passes = (
            max(2, int(np.ceil(np.log2(n_candidates / target))))
            if n_candidates > target
            else 1
        )

        if logger:
            logger.info(f"[Prune] n={n_candidates:,} target={target} passes={n_passes}")

        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")
        half = n * self._bitmap_stride
        pos_bits = hdv_data[:half].reshape(n, self._bitmap_stride)
        neg_bits = hdv_data[half:].reshape(n, self._bitmap_stride)
        pos_64 = pos_bits.view(np.uint64)
        neg_64 = neg_bits.view(np.uint64)
        q_pos_64 = q_pos.ravel().view(np.uint64)
        q_neg_64 = q_neg.ravel().view(np.uint64)
        n_cols = pos_64.shape[1]
        chunk_size = max(1, n_cols // n_passes)
        cumulative = np.zeros(n_candidates, dtype=np.float32)
        survivors = np.arange(n_candidates)

        for i in range(n_passes):
            start = i * chunk_size
            end = n_cols if i == n_passes - 1 else start + chunk_size
            corpus_survivors = corpus_idx[survivors]
            cumulative[survivors] += self._score_bitmap_64(
                pos_64[corpus_survivors, start:end],
                neg_64[corpus_survivors, start:end],
                q_pos_64[start:end],
                q_neg_64[start:end],
            )

            if i < n_passes - 1 and len(survivors) > 1:
                keep = max(1, len(survivors) // 2)
                idx = np.argpartition(cumulative[survivors], -keep)[-keep:]
                survivors = survivors[idx]

        if logger:
            logger.info(f"[Prune] {n_candidates:,}→{len(survivors):,} survivors")

        scores = cumulative[survivors]
        order = np.argsort(-scores)
        del hdv_data
        return corpus_idx[survivors[order]], scores[order]

    def _score_bitmap_64(
        self,
        d_pos_64: np.ndarray,
        d_neg_64: np.ndarray,
        q_pos_64: np.ndarray,
        q_neg_64: np.ndarray,
    ) -> np.ndarray:
        if q_pos_64.ndim == 1:
            q_pos_64 = q_pos_64[None, :]
            q_neg_64 = q_neg_64[None, :]
        agree = (d_pos_64 & q_pos_64) | (d_neg_64 & q_neg_64)
        disagree = (d_pos_64 & q_neg_64) | (d_neg_64 & q_pos_64)
        return (
            self._popcount64(agree).sum(axis=1) - self._popcount64(disagree).sum(axis=1)
        ).astype(np.float32)

    def _popcount64(self, x: np.ndarray) -> np.ndarray:
        x = x - ((x >> 1) & self._m1)
        x = (x & self._m2) + ((x >> 2) & self._m2)
        x = (x + (x >> 4)) & self._m4
        return ((x * self._h01) >> 56).astype(np.int32)

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM memories")
            c.execute("DELETE FROM idf")
            c.execute("DELETE FROM config WHERE key != 'hdc_dimensions'")
        if self.corpus_hdv_file.exists():
            self.corpus_hdv_file.unlink()

    def stats(self) -> dict:
        return {
            "db_mb": self.db_path.stat().st_size / 1e6 if self.db_path.exists() else 0,
            "corpus_hdv_mb": self.corpus_hdv_file.stat().st_size / 1e6
            if self.corpus_hdv_file.exists()
            else 0,
        }

    def source_counts(self) -> dict:
        rows = self._query(
            "SELECT json_extract(metadata, '$.source') as src, COUNT(*) as n FROM memories GROUP BY src ORDER BY n DESC"
        )
        return {r["src"]: r["n"] for r in rows}

    def compute_sparsity(self) -> dict:
        if not self.corpus_hdv_file.exists():
            return {"positive": 0, "negative": 0, "zero": 1}
        n = self.count()
        if n == 0:
            return {"positive": 0, "negative": 0, "zero": 1}

        d = self._bytes_per_bitmap()
        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")
        half = n * d
        pos_bits = hdv_data[:half].reshape(n, d)
        neg_bits = hdv_data[half:].reshape(n, d)

        dims = self.config.hdc_dimensions
        pos_unpacked = np.unpackbits(pos_bits, axis=1)[:, :dims]
        neg_unpacked = np.unpackbits(neg_bits, axis=1)[:, :dims]
        total = pos_unpacked.size
        pos_count = pos_unpacked.sum()
        neg_count = neg_unpacked.sum()
        zero_count = total - (pos_unpacked | neg_unpacked).sum()

        del hdv_data
        return {
            "positive": float(pos_count / total),
            "negative": float(neg_count / total),
            "zero": float(zero_count / total),
        }

    def sample_similarities(self, n_samples: int = 5000) -> list[float]:
        n = self.count()
        if not self.corpus_hdv_file.exists() or n < 2:
            return []

        d = self._bytes_per_bitmap()
        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")
        half = n * d
        pos_bits = hdv_data[:half].reshape(n, d)
        neg_bits = hdv_data[half:].reshape(n, d)
        d_pos_64 = pos_bits.view(np.uint64)
        d_neg_64 = neg_bits.view(np.uint64)

        n_samples = min(n_samples, n * (n - 1) // 2)
        idx_a = np.random.randint(0, n, n_samples * 2)
        idx_b = np.random.randint(0, n, n_samples * 2)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask][:n_samples], idx_b[mask][:n_samples]

        scores = self._score_bitmap_64(
            d_pos_64[idx_a], d_neg_64[idx_a], d_pos_64[idx_b], d_neg_64[idx_b]
        )

        del hdv_data
        return scores.tolist()

    def dimension_activation(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.corpus_hdv_file.exists():
            return np.array([]), np.array([])
        n = self.count()
        if n == 0:
            return np.array([]), np.array([])

        d = self._bytes_per_bitmap()
        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")
        half = n * d
        pos_bits = hdv_data[:half].reshape(n, d)
        neg_bits = hdv_data[half:].reshape(n, d)

        dims = self.config.hdc_dimensions
        pos_unpacked = np.unpackbits(pos_bits, axis=1)[:, :dims]
        neg_unpacked = np.unpackbits(neg_bits, axis=1)[:, :dims]
        pos_freq = pos_unpacked.mean(axis=0)
        neg_freq = neg_unpacked.mean(axis=0)

        del hdv_data
        return pos_freq, neg_freq

    def close(self) -> None:
        pass


class HDCEncoder:
    def __init__(self, config: Config, hdrag_dir: Path, db: Database):
        self.config = config
        self.hdrag_dir = hdrag_dir
        self.db = db
        self.proj_path = hdrag_dir / "projection.pt"
        self._load_projection()
        self.idf: dict[int, float] = db.load_idf()
        self.median_doc_length = db.get_config("median_doc_length") or 0.0

    def _bytes_per_bitmap(self) -> int:
        d = (self.config.hdc_dimensions + 7) // 8
        return d + (8 - d % 8) % 8

    def _load_projection(self) -> None:
        if self.proj_path.exists():
            proj = torch.load(self.proj_path, weights_only=True, map_location="cpu")
            CUDA.register("projection", proj)

    def _init_projection(self, emb_dim: int) -> None:
        proj = CUDA.get("projection")
        if proj is not None:
            if proj.shape[0] != emb_dim:
                raise ValueError(
                    f"Projection dim mismatch: {proj.shape[0]} != {emb_dim}"
                )
            return

        g = torch.Generator(device="cpu").manual_seed(self.config.hdc_seed)
        proj = torch.randn(
            emb_dim, self.config.hdc_dimensions, generator=g, dtype=torch.float16
        )
        proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
        torch.save(proj, self.proj_path)
        CUDA.register("projection", proj)

    def encode(self, embeddings: torch.Tensor) -> dict[str, np.ndarray]:
        if CUDA.get("projection") is None:
            self._init_projection(embeddings.shape[1])
        proj = CUDA.get("projection")

        emb = CUDA.to_device(embeddings).to(torch.float16)
        projected = emb @ proj

        threshold = embeddings.shape[1] ** -0.5
        pos = (projected > threshold).cpu().numpy()
        neg = (projected < -threshold).cpu().numpy()

        pos_packed = np.packbits(pos, axis=1)
        neg_packed = np.packbits(neg, axis=1)

        target = self._bytes_per_bitmap()
        pad = target - pos_packed.shape[1]
        if pad:
            pos_packed = np.pad(pos_packed, ((0, 0), (0, pad)))
            neg_packed = np.pad(neg_packed, ((0, 0), (0, pad)))

        return {"pos": pos_packed, "neg": neg_packed}

    def clear(self) -> None:
        CUDA.unregister("projection")
        if self.proj_path.exists():
            self.proj_path.unlink()


class Deduplicator:
    def __init__(self, config: Config):
        self.config = config

    def dedup(
        self,
        results: list[dict],
        bitmaps: Optional[tuple[np.ndarray, np.ndarray]] = None,
        scores: Optional[np.ndarray] = None,
        logger=None,
    ) -> tuple[list[dict], Optional[tuple[np.ndarray, np.ndarray]]]:
        n = len(results)
        if n <= 1:
            return results, bitmaps

        keep = np.ones(n, dtype=bool)
        token_sets = [set(r["memory"]["text"].lower().split()) for r in results]

        pre = keep.sum()
        keep &= self._token_subset_dedup(token_sets, keep)
        if logger and keep.sum() < pre:
            logger.info(f"[Dedup] subset: {pre}→{keep.sum()}")
        if keep.sum() <= 1:
            return self._apply_mask(results, bitmaps, keep)

        pre = keep.sum()
        keep &= self._near_dedup(token_sets, keep)
        if logger and keep.sum() < pre:
            logger.info(f"[Dedup] near: {pre}→{keep.sum()}")

        return self._apply_mask(results, bitmaps, keep)

    def _token_subset_dedup(
        self, token_sets: list[set[str]], keep: np.ndarray
    ) -> np.ndarray:
        indices = np.where(keep)[0].tolist()
        mask = np.ones(len(token_sets), dtype=bool)
        for i in range(len(indices)):
            if not mask[indices[i]]:
                continue
            for j in range(i + 1, len(indices)):
                if not mask[indices[j]]:
                    continue
                set_i, set_j = token_sets[indices[i]], token_sets[indices[j]]
                if set_i <= set_j:
                    mask[indices[i]] = False
                    break
                elif set_j <= set_i:
                    mask[indices[j]] = False
        return mask

    def _near_dedup(self, token_sets: list[set[str]], keep: np.ndarray) -> np.ndarray:
        indices = np.where(keep)[0].tolist()
        n = len(indices)
        if n <= 1:
            return keep

        pair_jaccards = []
        for i in range(n):
            for j in range(i + 1, n):
                set_i, set_j = token_sets[indices[i]], token_sets[indices[j]]
                union = len(set_i | set_j)
                if union > 0:
                    pair_jaccards.append(len(set_i & set_j) / union)

        if not pair_jaccards:
            return keep

        threshold = otsu_threshold(torch.tensor(pair_jaccards, dtype=torch.float32))
        if threshold == 0.0:
            return keep

        mask = np.ones(len(token_sets), dtype=bool)
        for i in range(n):
            if not mask[indices[i]]:
                continue
            for j in range(i + 1, n):
                if not mask[indices[j]]:
                    continue
                set_i, set_j = token_sets[indices[i]], token_sets[indices[j]]
                union = len(set_i | set_j)
                if union > 0 and len(set_i & set_j) / union > threshold:
                    mask[indices[j]] = False
        return mask

    def _apply_mask(
        self,
        results: list[dict],
        bitmaps: Optional[tuple[np.ndarray, np.ndarray]],
        mask: np.ndarray,
    ) -> tuple[list[dict], Optional[tuple[np.ndarray, np.ndarray]]]:
        filtered = [r for i, r in enumerate(results) if mask[i]]
        if bitmaps is not None:
            pos, neg = bitmaps
            return filtered, (pos[mask], neg[mask])
        return filtered, None


class Retriever:
    """
    Progressive pruning HDC retriever.
    Pipeline: HDC encode → score all (chunked + prune) → Otsu → dedup → rerank → budget fill
    """

    def __init__(
        self,
        db: Database,
        hdc: HDCEncoder,
        dedup: Deduplicator,
        model,
        config: Config,
        logger: logging.Logger,
    ):
        self.db = db
        self.hdc = hdc
        self.dedup = dedup
        self.model = model
        self.config = config
        self.logger = logger
        self._turns: list[torch.Tensor] = []
        self._token_counts = self.db.get_token_counts()

    def add_turn(self, text: str) -> None:
        emb = self.model.extract_embeddings([text]).squeeze(0).cpu()
        self._turns.append(emb)

    def clear_turns(self) -> None:
        self._turns.clear()

    def _blend(self, query_emb: torch.Tensor) -> torch.Tensor:
        q = query_emb.squeeze(0) if query_emb.dim() == 2 else query_emb
        all_embs = self._turns + [q]
        if len(all_embs) == 1:
            return q.unsqueeze(0)
        weights = torch.tensor(
            [1.0 / (len(all_embs) - i) for i in range(len(all_embs))]
        )
        blended = (torch.stack(all_embs) * weights.unsqueeze(-1)).sum(0)
        return F.normalize(blended.unsqueeze(0), p=2, dim=1)

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

        query_emb = self.model.extract_embeddings([query])
        if track:
            search_emb = self._blend(query_emb)
            self._turns.append(query_emb.squeeze(0).cpu())
        else:
            search_emb = query_emb

        bitmaps = self.hdc.encode(search_emb)

        # Derive target from retrieval semantics: enough candidates to fill budget min_context times
        max_doc_size = token_budget // self.config.min_context
        median_tokens = max(1, self.hdc.median_doc_length)
        target = token_budget // median_tokens * self.config.min_context
        eligible = np.where(self._token_counts <= max_doc_size)[0]

        indices, scores = self.db.search(
            bitmaps["pos"],
            bitmaps["neg"],
            target,
            candidates=eligible,
            logger=self.logger,
        )

        score_threshold = otsu_threshold(torch.from_numpy(scores))
        pre_otsu = len(indices)

        if score_threshold != 0.0:
            mask = scores >= score_threshold
            indices, scores = indices[mask], scores[mask]

        self.logger.info(
            f"[Search] {len(eligible):,}→{pre_otsu}→{len(indices)} (prune→Otsu)"
            + (f" threshold={score_threshold:.1f}" if score_threshold != 0.0 else "")
        )

        if len(indices) == 0:
            return []

        sel_idx = indices.tolist()
        score_map = {i: s for i, s in zip(sel_idx, scores.tolist())}
        memories = self.db.get_memories(sel_idx)

        if enabled_sources:
            memories = {
                i: m
                for i, m in memories.items()
                if m.get("metadata", {}).get("source") in enabled_sources
            }
            sel_idx = [i for i in sel_idx if i in memories]

        if len(sel_idx) > 1:
            temp = [
                {"memory": memories[i], "hdv_idx": i} for i in sel_idx if i in memories
            ]
            temp_scores = np.array(
                [score_map[i] for i in sel_idx if i in memories], dtype=np.float32
            )
            bitmaps = self.db.get_bitmaps([t["hdv_idx"] for t in temp])
            deduped, _ = self.dedup.dedup(temp, bitmaps, temp_scores, self.logger)
        else:
            deduped = [
                {"memory": memories[i], "hdv_idx": i} for i in sel_idx if i in memories
            ]

        deduped_tokens = sum(r["memory"].get("token_count", 0) for r in deduped)
        if len(deduped) > 1 and deduped_tokens > token_budget:
            texts = [r["memory"]["text"] for r in deduped]
            doc_embs = self.model.extract_embeddings(texts, use_idf=False)
            q_emb = self.model.extract_embeddings([query], use_idf=False)
            sims = (q_emb @ doc_embs.T).squeeze(0)
            order = torch.argsort(sims, descending=True).tolist()
            deduped = [deduped[i] for i in order]

        results, final_tokens = [], 0
        for r in deduped:
            tc = r["memory"].get("token_count", 0)
            if final_tokens + tc <= token_budget:
                results.append(
                    {
                        "memory": r["memory"],
                        "hdc_score": score_map.get(r["hdv_idx"], 0.0),
                    }
                )
                final_tokens += tc

        self.logger.info(
            f"[Search] {len(sel_idx)}→{len(deduped)}→{len(results)} ({final_tokens:,} tokens)"
        )
        return results


class ModelManager:
    EMBED_KEY_PATTERNS = ["embed_tokens", "wte", "word_embeddings"]

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.idf_weights: Optional[torch.Tensor] = None

    def _download_model(self, name: str) -> str:
        if os.path.exists(name):
            return name
        local_path = os.path.join(self.config.model_dir, name.replace("/", "_"))
        if not os.path.exists(local_path):
            self.logger.info(f"Downloading {name}")
            snapshot_download(
                repo_id=name, local_dir=local_path, local_dir_use_symlinks=False
            )
        return local_path

    def load_embedding_table(self) -> None:
        model_path = Path(self._download_model(self.config.model_name))
        with open(model_path / "config.json", encoding="utf-8") as f:
            model_config = json.load(f)
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(model_config.get("torch_dtype", "float32"), torch.float32)

        shard_path = model_path / "model.safetensors"
        if not shard_path.exists():
            index_path = model_path / "model.safetensors.index.json"
            if index_path.exists():
                data = json.loads(index_path.read_text())
                weight_map = data.get("weight_map", {})
                for k, v in weight_map.items():
                    if any(p in k for p in self.EMBED_KEY_PATTERNS):
                        shard_path = model_path / v
                        break

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(p in key for p in self.EMBED_KEY_PATTERNS) and "weight" in key:
                    embedding_table = f.get_tensor(key).to(dtype=dtype)
                    CUDA.register("embedding_table", embedding_table)
                    break

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def load_full_model(self) -> None:
        model_path = self._download_model(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.config.gguf_name:
            self.logger.info(
                f"GGUF mode: using llama.cpp server at {self.config.llama_server_url}"
            )
            if CUDA.get("embedding_table") is None:
                self.load_embedding_table()
            self.model = None
            return

        self.logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        embedding_table = self.model.get_input_embeddings().weight.data.clone()
        CUDA.register("embedding_table", embedding_table)

    def set_idf_weights(self, idf: dict[int, float]) -> None:
        vocab_size = len(self.tokenizer)
        idf_weights = torch.ones(vocab_size, dtype=torch.float32)
        for tid, weight in idf.items():
            if 0 <= tid < vocab_size:
                idf_weights[tid] = weight
        for tid in self.tokenizer.all_special_ids:
            idf_weights[tid] = 0.0
        self.idf_weights = CUDA.register("idf_weights", idf_weights)

    def extract_embeddings(
        self, texts: list[str], use_idf: bool = True
    ) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length_tokens,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        attn = inputs.attention_mask

        embedding_table = CUDA.get("embedding_table")
        idf_weights = CUDA.get("idf_weights")

        input_ids_gpu = CUDA.to_device(input_ids)
        attn_gpu = CUDA.to_device(attn)

        embeddings = embedding_table[input_ids_gpu]
        mask = attn_gpu.unsqueeze(-1).float()

        if use_idf and idf_weights is not None:
            weights = idf_weights[input_ids_gpu]
            w_expanded = weights.unsqueeze(-1) * mask
            pooled = (embeddings * w_expanded).sum(1) / w_expanded.sum(1).clamp(1e-9)
        else:
            pooled = (embeddings * mask).sum(1) / mask.sum(1).clamp(1e-9)

        return F.normalize(pooled, p=2, dim=1).cpu()

    def _generate_via_server(self, prompt: str):
        import requests

        payload = {
            "prompt": prompt,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "n_predict": self.config.max_new_tokens,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.config.llama_server_url}/completion",
                json=payload,
                stream=True,
                timeout=(5, None),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if content := chunk.get("content", ""):
                            yield content
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield f"\n\n[Error: Cannot connect to llama.cpp server at {self.config.llama_server_url}]"
        except requests.exceptions.Timeout:
            yield "\n\n[Error: Connection timeout]"
        except requests.exceptions.RequestException as e:
            yield f"\n\n[Error: {e}]"

    def generate_stream(self, prompt: str):
        if self.model is None:
            yield from self._generate_via_server(prompt)
            return

        device = next(self.model.parameters()).device
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text

    def generate(self, prompt: str) -> str:
        return "".join(self.generate_stream(prompt))


class ConversationLogger:
    def __init__(self, history_dir: str):
        self.dir = Path(history_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file: Optional[Path] = None

    def log(self, query: str, response: str) -> None:
        if self.file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file = self.dir / f"conversation_{ts}.jsonl"
        with open(self.file, "a") as f:
            f.write(json.dumps({"from": "human", "value": query}) + "\n")
            f.write(json.dumps({"from": "gpt", "value": response}) + "\n")

    def new_conversation(self) -> None:
        self.file = None


class HdRAG:
    def __init__(self, config_path: str = "hdrag_config.yaml", debug: bool = False):
        self.config_path = config_path
        self.config = Config.load(config_path)
        self.logger = self._setup_logging(debug)

        self.hdrag_dir = Path(self.config.hdrag_dir)
        self.hdrag_dir.mkdir(parents=True, exist_ok=True)

        self.db = Database(self.hdrag_dir / "index.db", self.config, self.logger)
        self.hdc = HDCEncoder(self.config, self.hdrag_dir, self.db)
        self.model = ModelManager(self.config, self.logger)
        self.dedup = Deduplicator(self.config)
        self.retriever = Retriever(
            self.db, self.hdc, self.dedup, self.model, self.config, self.logger
        )
        self.chat_log = ConversationLogger(self.config.chat_history_dir)

        self.enabled_sources: set = set(self.db.source_counts().keys())
        self.logger.info(f"HdRAG initialized with {self.db.count():,} memories")

    def _setup_logging(self, debug: bool) -> logging.Logger:
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
            force=True,
        )
        return logging.getLogger(__name__)

    def load_model(self) -> None:
        self.model.load_full_model()
        if self.hdc.idf:
            self.model.set_idf_weights(self.hdc.idf)

    def new_conversation(self) -> None:
        self.chat_log.new_conversation()
        self.retriever.clear_turns()

    def search(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> list[dict]:
        return self.retriever.search(
            query,
            token_budget or self.config.max_context_tokens,
            track,
            enabled_sources=self.enabled_sources,
        )

    def get_context(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> str:
        results = self.search(query, token_budget, track)
        return "\n\n---\n\n".join(r["memory"]["text"] for r in results)

    def add_turn(self, text: str) -> None:
        self.retriever.add_turn(text)

    def clear_index(self) -> None:
        import gc

        gc.collect()
        self.db.clear()
        self.hdc.clear()
        self.logger.info("Index cleared")

    def stats(self) -> dict:
        return {
            "memories": self.db.count(),
            "vocab_size": len(self.hdc.idf),
            "median_tokens": self.hdc.median_doc_length,
            "hdc_dims": self.config.hdc_dimensions,
            "model": self.config.model_name,
            "sources": self.db.source_counts(),
            "cuda": CUDA.memory_stats,
            "db": self.db.stats(),
        }

    def extend_index(self, progress_cb: Callable[[float, str], None] = None) -> int:
        if CUDA.get("embedding_table") is None:
            self.model.load_embedding_table()

        files = discover_datasets(Path(self.config.datasets_dir))
        if not files:
            return 0

        self.logger.info("Clearing existing index...")
        import gc

        gc.collect()
        if self.db.corpus_hdv_file.exists():
            self.db.corpus_hdv_file.unlink()
        self.db.clear()

        self.logger.info("Pass 1: Building vocabulary...")
        special = set(self.model.tokenizer.all_special_ids)
        chunk_size = self.config.batch_size * self.config.vocab_chunk_multiplier
        vocab_df: Counter[int] = Counter()
        docs = []

        for i, f in enumerate(files):
            path, name = Path(f["path"]), f["name"]
            count, pending = 0, []

            for item in iter_dataset(
                path,
                tokenizer=self.model.tokenizer,
                chunk_size=self.config.text_chunk_size,
                chunk_overlap=self.config.text_chunk_overlap,
            ):
                if "chunk_idx" in item:
                    text = item["text"].strip()
                    chunk_meta = {
                        "chunk_idx": item["chunk_idx"],
                        "total_chunks": item.get("total_chunks"),
                    }
                else:
                    text = extract_text(item).strip()
                    chunk_meta = None

                if text:
                    pending.append((text, chunk_meta))

                if len(pending) >= chunk_size:
                    count += self._process_chunk(pending, name, special, vocab_df, docs)
                    pending = []

            if pending:
                count += self._process_chunk(pending, name, special, vocab_df, docs)

            self.logger.info(f"  [{name}] {count:,} records")
            if progress_cb:
                progress_cb((i + 1) / len(files) * 0.3, f"Pass 1: {name}")

        seen, unique = set(), []
        for d in docs:
            if d["id"] not in seen:
                seen.add(d["id"])
                unique.append(d)
        docs = unique

        if not docs:
            self.logger.info("No new documents")
            return 0

        self.logger.info(f"Full rebuild: {len(docs):,} documents")
        self.db.set_config("hdc_seed", self.config.hdc_seed)

        self.logger.info("Computing IDF...")
        n_docs = len(docs)
        idf = {tid: math.log((n_docs + 1) / (df + 1)) for tid, df in vocab_df.items()}
        self.db.save_idf(dict(vocab_df), n_docs)

        self.db.set_config(
            "median_doc_length", statistics.median(d["token_count"] for d in docs)
        )
        self.hdc.idf = idf
        self.hdc.median_doc_length = self.db.get_config("median_doc_length")
        self.model.set_idf_weights(idf)

        self.logger.info("Pass 2: Encoding...")
        bs = self.config.batch_size
        n_batches = (len(docs) + bs - 1) // bs

        for i, start in enumerate(range(0, len(docs), bs)):
            batch = docs[start : start + bs]
            embs = self.model.extract_embeddings([d["text"] for d in batch])
            bitmaps = self.hdc.encode(embs)

            self.db.insert(
                [
                    {
                        "id": d["id"],
                        "text": d["text"],
                        "metadata": d["metadata"],
                        "token_count": d["token_count"],
                    }
                    for d in batch
                ],
                bitmaps,
            )

            if (i + 1) % self.config.batch_log_interval == 0 or i + 1 == n_batches:
                self.logger.info(
                    f"  Batch {i + 1:,}/{n_batches:,} ({100 * (i + 1) / n_batches:.0f}%)"
                )
            if progress_cb:
                progress_cb(0.3 + (i + 1) / n_batches * 0.7, f"Pass 2: batch {i + 1}")

        CUDA.clear()
        self.logger.info(f"Index complete: {self.db.count():,} memories")

        self.db.finalize_index()
        self.retriever._token_counts = self.db.get_token_counts()

        return len(docs)

    def _process_chunk(
        self,
        items: list[tuple[str, Optional[dict]]],
        source: str,
        special: set[int],
        vocab_df: Counter[int],
        docs: list[dict],
    ) -> int:
        texts = [t for t, _ in items]
        chunk_metas = [m for _, m in items]

        ids = []
        for text, meta in items:
            if meta and "chunk_idx" in meta:
                id_str = f"{source}:chunk{meta['chunk_idx']}:{text}"
            else:
                id_str = f"{source}:{text}"
            ids.append(
                hashlib.blake2b(
                    id_str.encode(), digest_size=self.config.hash_digest_size
                ).hexdigest()
            )

        existing = self.db.exists(ids)
        new_items = [
            (mid, txt, meta)
            for mid, txt, meta in zip(ids, texts, chunk_metas)
            if mid not in existing
        ]

        if not new_items:
            return 0

        new_ids = [x[0] for x in new_items]
        new_texts = [x[1] for x in new_items]
        new_metas = [x[2] for x in new_items]

        encodings = self.model.tokenizer(
            new_texts,
            padding=False,
            truncation=True,
            max_length=self.config.max_length_tokens,
        )

        for mid, text, meta, toks in zip(
            new_ids, new_texts, new_metas, encodings.input_ids
        ):
            doc_tokens = set(toks) - special
            vocab_df.update(doc_tokens)

            doc_meta = {"source": source}
            if meta:
                doc_meta.update(meta)

            docs.append(
                {
                    "id": mid,
                    "text": text,
                    "metadata": doc_meta,
                    "token_count": len(toks),
                }
            )

        return len(new_items)

    def save_config(self) -> None:
        self.config.save(self.config_path)

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()


def create_gradio_app(hdrag: HdRAG):
    model_max = 8192
    if hdrag.model.tokenizer:
        raw = getattr(hdrag.model.tokenizer, "model_max_length", None)
        if raw and raw < 1_000_000:
            model_max = raw

    slider_max = max(min(model_max // 2, 131072), hdrag.config.max_context_tokens)

    with gr.Blocks(title="HdRAG") as app:
        gr.Markdown(
            f"# 🧠 HdRAG\n**Model:** {hdrag.config.model_name} | **Memories:** {hdrag.db.count():,}"
        )

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=450, label="Conversation")

            with gr.Row():
                msg = gr.Textbox(
                    label="Message", placeholder="Ask something...", scale=4, lines=2
                )
                budget_slider = gr.Slider(
                    500,
                    slider_max,
                    value=hdrag.config.max_context_tokens,
                    step=100,
                    label="Token Budget",
                    scale=1,
                )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("New Conversation")
                use_memory = gr.Checkbox(value=True, label="Use Memory")
                compress_ctx = gr.Checkbox(value=False, label="Compress Context")

            with gr.Accordion("🔍 Debug: Full Prompt Viewer", open=False):
                debug_viewer = gr.Code(
                    label="Full Prompt",
                    language="markdown",
                    lines=20,
                    value="Send a message to see the full prompt...",
                )

            def respond(message, history, budget, use_mem, compress):
                if not message.strip():
                    yield "", history, ""
                    return

                def extract_text(content):
                    if isinstance(content, list):
                        return " ".join(
                            b.get("text", "")
                            for b in content
                            if b.get("type") == "text"
                        )
                    return str(content)

                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": ""})
                yield "", history, f"🔍 Query: {message}"

                memory = ""
                if use_mem:
                    memory = hdrag.get_context(message, budget, track=True)
                    if compress and memory:  # apply compression
                        memory = compress_context(memory)

                sys_prompt = (
                    f"{hdrag.config.system_prompt}\n\n<working_memory>\n{memory}\n</working_memory>"
                    if memory
                    else hdrag.config.system_prompt
                )

                msgs = [{"role": "system", "content": sys_prompt}]
                msgs += [
                    {"role": t["role"], "content": extract_text(t["content"])}
                    for t in history[:-2]
                ]
                msgs.append({"role": "user", "content": message})

                prompt = hdrag.model.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )

                response = ""
                for chunk in hdrag.model.generate_stream(prompt):
                    response += chunk
                    history[-1] = {"role": "assistant", "content": response}
                    yield (
                        "",
                        history,
                        f"🔍 Query: {message}\n\n🤖 Full Prompt:\n{prompt}",
                    )

                hdrag.add_turn(response)
                hdrag.chat_log.log(message, response)

            def clear_conversation():
                hdrag.new_conversation()
                return [], "Send a message to see the full prompt..."

            msg.submit(
                respond,
                [msg, chatbot, budget_slider, use_memory, compress_ctx],
                [msg, chatbot, debug_viewer],
            )
            send_btn.click(
                respond,
                [msg, chatbot, budget_slider, use_memory, compress_ctx],
                [msg, chatbot, debug_viewer],
            )
            clear_btn.click(clear_conversation, outputs=[chatbot, debug_viewer])

        with gr.Tab("🔍 Search"):
            search_input = gr.Textbox(label="Query", placeholder="Search memories...")
            search_budget = gr.Slider(
                500,
                slider_max,
                value=hdrag.config.max_context_tokens,
                step=100,
                label="Token Budget",
            )
            search_btn = gr.Button("Search", variant="primary")
            search_output = gr.JSON(label="Results")

            def do_search(query, budget):
                if not query.strip():
                    return []
                return [
                    {
                        "score": r["hdc_score"],
                        "source": r["memory"]["metadata"].get("source", "?"),
                        "tokens": r["memory"].get("token_count", 0),
                        "text": r["memory"]["text"][:500]
                        + ("..." if len(r["memory"]["text"]) > 500 else ""),
                    }
                    for r in hdrag.search(query, token_budget=budget, track=False)
                ]

            search_btn.click(do_search, [search_input, search_budget], search_output)

        with gr.Tab("⚙️ Config"):
            gr.Markdown("## System Configuration")

            with gr.Accordion("HDC Settings", open=False):
                hdc_dims = gr.Slider(
                    1000,
                    50000,
                    value=hdrag.config.hdc_dimensions,
                    step=1000,
                    label="Dimensions",
                )
                hdc_seed = gr.Number(
                    value=hdrag.config.hdc_seed, label="Seed", precision=0
                )

            with gr.Accordion("Retrieval Settings", open=False):
                max_tokens = gr.Slider(
                    500,
                    slider_max,
                    value=hdrag.config.max_context_tokens,
                    step=100,
                    label="Max Context Tokens",
                )

            with gr.Accordion("Model Settings", open=False):
                gr.Markdown(f"**Model:** `{hdrag.config.model_name}`")
                model_temp = gr.Slider(
                    0.0,
                    2.0,
                    value=hdrag.config.temperature,
                    step=0.05,
                    label="Temperature",
                )
                model_top_p = gr.Slider(
                    0.0, 1.0, value=hdrag.config.top_p, step=0.05, label="Top P"
                )
                model_max_new = gr.Slider(
                    128,
                    slider_max,
                    value=hdrag.config.max_new_tokens,
                    step=128,
                    label="Max Output Tokens",
                )

            with gr.Accordion("Datasets", open=False):
                gr.Markdown(f"**Directory:** `{hdrag.config.datasets_dir}`")
                source_counts = hdrag.db.source_counts()
                if source_counts:
                    gr.Markdown("**Select datasets to include in search:**")
                    dataset_checkboxes = gr.CheckboxGroup(
                        choices=[
                            f"{src} ({count:,})" for src, count in source_counts.items()
                        ],
                        value=[
                            f"{src} ({count:,})" for src, count in source_counts.items()
                        ],
                        label="Enabled Datasets",
                    )

                    def update_enabled_sources(selected):
                        hdrag.enabled_sources = {s.rsplit(" (", 1)[0] for s in selected}
                        return f"✓ {len(hdrag.enabled_sources)} datasets enabled"

                    dataset_status = gr.Textbox(
                        label="Dataset Status",
                        interactive=False,
                        value=f"{len(source_counts)} datasets enabled",
                    )
                    dataset_checkboxes.change(
                        update_enabled_sources, [dataset_checkboxes], [dataset_status]
                    )
                else:
                    gr.Markdown("*No indexed datasets found. Click 'Index' to build.*")

            with gr.Row():
                save_btn = gr.Button("💾 Save Config", variant="secondary")
                reindex_btn = gr.Button("🔄 Index", variant="primary")

            status = gr.Textbox(label="Status", interactive=False)

            def update_config(dims, seed, tokens, temp, top_p, max_new):
                hdrag.config.hdc_dimensions = int(dims)
                hdrag.config.hdc_seed = int(seed)
                hdrag.config.max_context_tokens = int(tokens)
                hdrag.config.temperature = float(temp)
                hdrag.config.top_p = float(top_p)
                hdrag.config.max_new_tokens = int(max_new)
                hdrag.save_config()
                return "✓ Saved"

            inputs = [
                hdc_dims,
                hdc_seed,
                max_tokens,
                model_temp,
                model_top_p,
                model_max_new,
            ]
            save_btn.click(update_config, inputs, status)
            reindex_btn.click(
                lambda *_: f"✓ Indexed {hdrag.extend_index()} memories", inputs, status
            )

        with gr.Tab("📊 Stats"):
            stats_output = gr.JSON(label="System Statistics")
            with gr.Row():
                sparsity_plot = gr.Plot(label="HDV Sparsity")
                similarity_plot = gr.Plot(label="Corpus Similarity Distribution")

            dimension_plot = gr.Plot(label="Dimension Activation")
            refresh_btn = gr.Button("Refresh Stats", variant="primary")

            def compute_stats():
                stats = hdrag.stats()

                sparsity = hdrag.db.compute_sparsity()
                sparsity_fig = go.Figure(
                    go.Bar(
                        x=["Positive (+1)", "Zero (0)", "Negative (-1)"],
                        y=[
                            sparsity["positive"],
                            sparsity["zero"],
                            sparsity["negative"],
                        ],
                        marker_color=["#4CAF50", "#9E9E9E", "#f44336"],
                        text=[
                            f"{v:.1%}"
                            for v in [
                                sparsity["positive"],
                                sparsity["zero"],
                                sparsity["negative"],
                            ]
                        ],
                        textposition="auto",
                    )
                )
                sparsity_fig.update_layout(
                    title="Ternary Distribution Across All HDVs",
                    yaxis_title="Fraction",
                    yaxis_tickformat=".0%",
                    height=300,
                    margin=dict(t=40, b=40),
                )

                sims = hdrag.db.sample_similarities(5000)
                if sims:
                    sim_fig = go.Figure(
                        go.Histogram(
                            x=sims, nbinsx=50, marker_color="#2196F3", opacity=0.75
                        )
                    )
                    sim_fig.add_vline(
                        x=np.median(sims),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"median: {np.median(sims):.0f}",
                    )
                    sim_fig.update_layout(
                        title="Pairwise Document Similarity (5k samples)",
                        xaxis_title="HDC Score",
                        yaxis_title="Count",
                        height=300,
                        margin=dict(t=40, b=40),
                    )
                else:
                    sim_fig = go.Figure()
                    sim_fig.add_annotation(text="Not enough documents", showarrow=False)

                pos_freq, neg_freq = hdrag.db.dimension_activation()
                if len(pos_freq) > 0:
                    step = max(1, len(pos_freq) // 500)
                    x = np.arange(0, len(pos_freq), step)

                    dim_fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Positive Activation", "Negative Activation"),
                        vertical_spacing=0.1,
                    )
                    dim_fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pos_freq[::step],
                            mode="lines",
                            line=dict(color="#4CAF50", width=1),
                            name="+1",
                        ),
                        row=1,
                        col=1,
                    )
                    dim_fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=neg_freq[::step],
                            mode="lines",
                            line=dict(color="#f44336", width=1),
                            name="-1",
                        ),
                        row=2,
                        col=1,
                    )
                    dim_fig.update_layout(
                        height=350, margin=dict(t=40, b=40), showlegend=False
                    )
                    dim_fig.update_xaxes(title_text="Dimension", row=2, col=1)
                    dim_fig.update_yaxes(title_text="Freq", tickformat=".0%")
                else:
                    dim_fig = go.Figure()
                    dim_fig.add_annotation(text="No index loaded", showarrow=False)

                return stats, sparsity_fig, sim_fig, dim_fig

            refresh_btn.click(
                compute_stats,
                outputs=[stats_output, sparsity_plot, similarity_plot, dimension_plot],
            )

    return app


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="HdRAG")
    parser.add_argument("--config", default="hdrag_config.yaml", help="Config path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    hdrag = HdRAG(args.config, debug=args.debug)
    hdrag.load_model()
    app = create_gradio_app(hdrag)
    app.launch(server_port=hdrag.config.gradio_port)


if __name__ == "__main__":
    main()
