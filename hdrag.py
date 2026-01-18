"""
hdrag - Hyperdimensional Retrieval-Augmented Generation
Usage: >python hdrag.py --config hdrag_config.yaml [--debug]
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

# Module level - precompute once
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


@dataclass
class Config:
    # Directories
    chat_history_dir: str
    hdrag_dir: str
    datasets_dir: str
    model_dir: str
    # Model
    model_name: str
    temperature: float
    top_p: float
    max_new_tokens: int
    max_length_tokens: int
    # HDC Encoding
    hdc_dimensions: int
    hdc_seed: int
    # Indexing
    batch_size: int
    vocab_chunk_multiplier: int
    hash_digest_size: int
    export_log_interval: int
    batch_log_interval: int
    # .txt file chunking
    text_chunk_size: int
    text_chunk_overlap: int
    # Retrieval
    max_context_tokens: int
    min_context: int
    hdc_search_mb: int
    # Database
    sqlite_max_vars: int
    sqlite_cache_kb: int
    sqlite_mmap_bytes: int
    # UI
    gradio_port: int
    # Prompt
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
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        missing = valid_fields - set(filtered_data.keys())
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        return cls(**filtered_data)


def chunks(xs: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def otsu_threshold(vals: torch.Tensor) -> Optional[float]:
    if vals.numel() < 2:
        return None
    x, _ = torch.sort(vals)
    median_val = torch.median(x)
    upper = x[x >= median_val]
    if upper.numel() < 2:
        return median_val.item()
    n = upper.numel()
    sum_total = torch.sum(upper)
    sum_left = torch.cumsum(upper, dim=0)
    w0 = torch.arange(1, n, device=upper.device) / n
    w1 = 1.0 - w0
    m0 = sum_left[:-1] / torch.arange(1, n, device=upper.device)
    m1 = (sum_total - sum_left[:-1]) / torch.arange(n - 1, 0, -1, device=upper.device)
    scores = w0 * w1 * (m0 - m1).pow(2)
    if scores.max() == 0:
        return median_val.item()
    max_idx = torch.argmax(scores)
    return ((upper[max_idx] + upper[max_idx + 1]) / 2.0).item()


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
    for file_path in sorted(directory.glob("**/*")):
        if file_path.suffix in [
            ".json",
            ".jsonl",
            ".parquet",
            ".txt",
            ".md",
            ".html",
            ".xml",
        ]:
            datasets.append({"name": file_path.stem, "path": str(file_path)})
    return datasets


def iter_dataset(
    path: Path, tokenizer=None, chunk_size: int = 1024, chunk_overlap: int = 128
) -> Generator[dict, None, None]:
    """Iterate over dataset, yielding documents. Handles .txt chunking."""

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
    # Direct conversation array (no wrapper key)
    if (
        isinstance(item, list)
        and item
        and isinstance(item[0], dict)
        and "from" in item[0]
    ):
        messages = [
            msg.get("value", msg.get("content", ""))
            for msg in item
            if msg.get("from", msg.get("role", "")) not in ["system"]
        ]
        return "\n\n".join(m for m in messages if m)

    # Conversation arrays
    if "conversations" in item or "conversation" in item:
        conv_key = "conversations" if "conversations" in item else "conversation"
        conv = item.get(conv_key, [])
        messages = [
            msg.get("value", msg.get("content", ""))
            for msg in conv
            if msg.get("from", msg.get("role", "")) not in ["system"]
        ]
        return "\n\n".join(m for m in messages if m)

    # Messages array (OpenAI format)
    if "messages" in item:
        messages = [
            m.get("content", "")
            for m in item["messages"]
            if m.get("role") not in ["system"]
        ]
        return "\n\n".join(m for m in messages if m)

    # Top-level human/assistant pairs
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
        # Also grab optional input field
        if item.get("input"):
            parts.insert(1, item["input"])
        return "\n\n".join(parts)

    # Plain text fields
    if "text" in item:
        return item["text"]

    if "content" in item:
        return item["content"]

    return ""


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
        self.token_hdv_file = db_path.parent / "token_hdc.idx"
        self.vocab_index_file = db_path.parent / "vocab.idx"
        self.logger = logger
        self._init_db()

        # SWAR constants
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
        c.execute(f"PRAGMA mmap_size={self.config.sqlite_mmap_bytes}")
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
        counts = np.ones(n, dtype=np.int32)  # default 1 for missing
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

    # -------------------------------------------------------------------------
    # IDF
    # -------------------------------------------------------------------------

    def save_idf(self, df_counts: dict[int, int], n_docs: int) -> None:
        idf_values = [
            (tid, df, math.log((n_docs + 1) / (df + 1)))
            for tid, df in df_counts.items()
        ]
        with self._conn() as c:
            c.execute("DELETE FROM idf")
            c.executemany("INSERT INTO idf VALUES (?,?,?)", idf_values)

        if idf_values:
            self.set_config("median_idf", statistics.median(v[2] for v in idf_values))

    def load_idf(self) -> dict[int, float]:
        return {
            r["token_id"]: r["weight"]
            for r in self._query("SELECT token_id, weight FROM idf")
        }

    def load_df(self) -> dict[int, int]:
        return {
            r["token_id"]: r["doc_freq"]
            for r in self._query("SELECT token_id, doc_freq FROM idf")
        }

    # -------------------------------------------------------------------------
    # Vocab Index (inverted index: token_id → doc_ids)
    # -------------------------------------------------------------------------

    def save_vocab_index(
        self, vocab_index: dict[int, list[int]], vocab_size: int
    ) -> None:
        """Save inverted index in CSR binary format."""
        row_ptrs = np.zeros(vocab_size + 1, dtype=np.uint64)
        all_postings = []

        for tid in range(vocab_size):
            row_ptrs[tid + 1] = row_ptrs[tid]
            if tid in vocab_index:
                all_postings.extend(vocab_index[tid])
                row_ptrs[tid + 1] += len(vocab_index[tid])

        postings = (
            np.array(all_postings, dtype=np.uint32)
            if all_postings
            else np.array([], dtype=np.uint32)
        )

        with open(self.vocab_index_file, "wb") as f:
            f.write(b"VIDX")
            f.write(np.array([1, vocab_size, len(postings)], dtype=np.uint32).tobytes())
            f.write(row_ptrs.tobytes())
            f.write(postings.tobytes())

        if self.logger:
            self.logger.info(
                f"Vocab index: {len(vocab_index):,} terms, {len(postings):,} postings, "
                f"{self.vocab_index_file.stat().st_size / 1e6:.1f}MB"
            )

    def load_vocab_index(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load inverted index as mmap'd CSR arrays (row_ptrs, postings)."""
        if not self.vocab_index_file.exists():
            return None, None

        with open(self.vocab_index_file, "rb") as f:
            if f.read(4) != b"VIDX":
                return None, None
            _, vocab_size, n_postings = np.frombuffer(f.read(12), dtype=np.uint32)

        header_bytes = 16
        row_ptrs = np.memmap(
            self.vocab_index_file,
            dtype=np.uint64,
            mode="r",
            offset=header_bytes,
            shape=(vocab_size + 1,),
        )
        postings = np.memmap(
            self.vocab_index_file,
            dtype=np.uint32,
            mode="r",
            offset=header_bytes + (vocab_size + 1) * 8,
            shape=(n_postings,),
        )
        return row_ptrs, postings

    # -------------------------------------------------------------------------
    # Token HDC Index
    # -------------------------------------------------------------------------

    def save_token_hdc(self, pos: np.ndarray, neg: np.ndarray) -> None:
        """Save token HDC index in blocked layout."""
        n_tokens = len(pos)
        with open(self.token_hdv_file, "wb") as f:
            for i in range(n_tokens):
                f.write(pos[i].tobytes())
            for i in range(n_tokens):
                f.write(neg[i].tobytes())
        if self.logger:
            self.logger.info(
                f"Token HDC index: {n_tokens:,} tokens, "
                f"{self.token_hdv_file.stat().st_size / 1e6:.1f}MB"
            )

    def load_token_hdc(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load token HDC index."""
        if not self.token_hdv_file.exists():
            return None, None

        d = self._bytes_per_bitmap()
        data = np.memmap(self.token_hdv_file, dtype=np.uint8, mode="r")
        n_tokens = len(data) // (d * 2)
        half = n_tokens * d

        pos = data[:half].reshape(n_tokens, d)
        neg = data[half:].reshape(n_tokens, d)
        return pos, neg

    def expand_tokens(
        self,
        query_token_ids: list[int],
        token_pos: np.ndarray,
        token_neg: np.ndarray,
        k: int = 5,
    ) -> list[int]:
        """Expand query tokens using HDC similarity over vocab."""
        expanded = set(query_token_ids)

        t_pos_64 = token_pos.view(np.uint64)
        t_neg_64 = token_neg.view(np.uint64)
        n_tokens = len(t_pos_64)

        for tid in query_token_ids:
            if tid >= n_tokens:
                continue

            scores = self._score_bitmap_64(
                t_pos_64, t_neg_64, t_pos_64[tid], t_neg_64[tid]
            )
            scores[tid] = -np.inf  # exclude self

            topk = np.argpartition(scores, -k)[-k:]
            expanded.update(topk.tolist())

        return list(expanded)

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Insert / Finalize
    # -------------------------------------------------------------------------

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
        """Reorganize corpus HDC to blocked layout (all pos, then all neg)."""
        n = self.count()
        if n == 0:
            return
        d = self._bytes_per_bitmap()
        expected = n * d * 2
        if not self.corpus_hdv_file.exists():
            return
        if self.corpus_hdv_file.stat().st_size != expected:
            if self.logger:
                self.logger.warning(
                    f"Index file size mismatch: {self.corpus_hdv_file.stat().st_size} vs {expected}"
                )
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

        if self.logger:
            self.logger.info(
                f"Index complete: {self.corpus_hdv_file.stat().st_size / 1e9:.2f} GB"
            )

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        q_pos: np.ndarray,
        q_neg: np.ndarray,
        candidate_indices: list[int],
        k: int,
    ) -> tuple[list[int], list[float]]:
        """Score candidate documents against query (no coarse pass)."""
        if not self.corpus_hdv_file.exists() or not candidate_indices:
            return [], []

        n = self.count()
        if n == 0:
            return [], []

        d = self._bytes_per_bitmap()
        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")

        half = n * d
        pos_bits = hdv_data[:half].reshape(n, d)
        neg_bits = hdv_data[half:].reshape(n, d)

        q_pos_64 = q_pos.ravel().view(np.uint64)
        q_neg_64 = q_neg.ravel().view(np.uint64)

        # Score only candidates (already cache-sized by caller)
        cand_idx = np.array(candidate_indices)
        d_pos_64 = pos_bits[cand_idx].view(np.uint64)
        d_neg_64 = neg_bits[cand_idx].view(np.uint64)

        scores = self._score_bitmap_64(d_pos_64, d_neg_64, q_pos_64, q_neg_64)

        # Top k
        top_k = min(k, len(scores))
        if top_k == 0:
            del hdv_data
            return [], []

        top_pos = np.argpartition(scores, -top_k)[-top_k:]
        top_pos = top_pos[np.argsort(scores[top_pos])[::-1]]

        result_indices = cand_idx[top_pos].tolist()
        result_scores = scores[top_pos].tolist()

        del hdv_data
        return result_indices, result_scores

    def search_full(
        self, q_pos: np.ndarray, q_neg: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        """Fallback: score all documents (for queries with no vocab matches)."""
        if not self.corpus_hdv_file.exists():
            return [], []

        n = self.count()
        if n == 0:
            return [], []

        d = self._bytes_per_bitmap()
        hdv_data = np.memmap(self.corpus_hdv_file, dtype=np.uint8, mode="r")

        half = n * d
        pos_bits = hdv_data[:half].reshape(n, d)
        neg_bits = hdv_data[half:].reshape(n, d)

        q_pos_64 = q_pos.ravel().view(np.uint64)
        q_neg_64 = q_neg.ravel().view(np.uint64)
        d_pos_64 = pos_bits.view(np.uint64)
        d_neg_64 = neg_bits.view(np.uint64)

        scores = self._score_bitmap_64(d_pos_64, d_neg_64, q_pos_64, q_neg_64)

        top_k = min(k, len(scores))
        top_pos = np.argpartition(scores, -top_k)[-top_k:]
        top_pos = top_pos[np.argsort(scores[top_pos])[::-1]]

        result_indices = top_pos.tolist()
        result_scores = scores[top_pos].tolist()

        del hdv_data
        return result_indices, result_scores

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Clear / Stats
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM memories")
            c.execute("DELETE FROM idf")
            c.execute("DELETE FROM config WHERE key != 'hdc_dimensions'")
        if self.corpus_hdv_file.exists():
            self.corpus_hdv_file.unlink()
        if self.token_hdv_file.exists():
            self.token_hdv_file.unlink()
        if self.vocab_index_file.exists():
            self.vocab_index_file.unlink()

    def stats(self) -> dict:
        return {
            "db_mb": self.db_path.stat().st_size / 1e6 if self.db_path.exists() else 0,
            "corpus_hdv_mb": self.corpus_hdv_file.stat().st_size / 1e6
            if self.corpus_hdv_file.exists()
            else 0,
            "token_hdv_mb": self.token_hdv_file.stat().st_size / 1e6
            if self.token_hdv_file.exists()
            else 0,
            "vocab_index_mb": self.vocab_index_file.stat().st_size / 1e6
            if self.vocab_index_file.exists()
            else 0,
        }

    def source_counts(self) -> dict:
        rows = self._query("""
            SELECT json_extract(metadata, '$.source') as src, COUNT(*) as n
            FROM memories GROUP BY src ORDER BY n DESC
        """)
        return {r["src"]: r["n"] for r in rows}

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def compute_sparsity(self) -> dict:
        """Compute ternary distribution across all corpus HDVs."""
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

        result = {
            "positive": float(pos_count / total),
            "negative": float(neg_count / total),
            "zero": float(zero_count / total),
        }

        del hdv_data
        return result

    def sample_similarities(self, n_samples: int = 5000) -> list[float]:
        """Sample pairwise similarities for corpus diversity analysis."""
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

        scores = []
        for a, b in zip(idx_a, idx_b):
            score = self._score_bitmap_64(
                d_pos_64[a : a + 1],
                d_neg_64[a : a + 1],
                d_pos_64[b],
                d_neg_64[b],
            )[0]
            scores.append(float(score))

        del hdv_data
        return scores

    def dimension_activation(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute dimension activation frequencies."""
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

        corr = np.corrcoef(pos_freq, neg_freq)[0, 1]
        if self.logger:
            self.logger.debug(f"Pos/Neg frequency correlation: {corr:.3f}")

        del hdv_data
        return pos_freq, neg_freq

    def close(self) -> None:
        """Close database connections."""
        pass  # Nothing to close with ephemeral mmaps


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

        # Pad to uint64 alignment
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
        self._popcount_table = np.array(
            [bin(i).count("1") for i in range(256)], dtype=np.int32
        )

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def dedup(
        self,
        results: list[dict],
        bitmaps: Optional[tuple[np.ndarray, np.ndarray]] = None,
        scores: Optional[np.ndarray] = None,
    ) -> tuple[list[dict], Optional[tuple[np.ndarray, np.ndarray]]]:
        n = len(results)
        if n <= 1:
            return results, bitmaps

        keep = np.ones(n, dtype=bool)

        # Stage 0: Otsu filter on retrieval scores
        if scores is not None and len(scores) > 1:
            threshold = otsu_threshold(torch.from_numpy(scores))
            if threshold is not None:
                keep &= scores >= threshold
                if keep.sum() <= 1:
                    return self._apply_mask(results, bitmaps, keep)

        texts = [r["memory"]["text"] for r in results]
        normalized = [self._normalize(t) for t in texts]
        token_sets = [set(t.split()) for t in normalized]

        keep &= self._exact_dedup(normalized)
        if keep.sum() <= 1:
            return self._apply_mask(results, bitmaps, keep)

        if keep.sum() > 1:
            keep &= self._token_subset_dedup(token_sets, keep)
            if keep.sum() <= 1:
                return self._apply_mask(results, bitmaps, keep)

        if bitmaps is not None and keep.sum() > 2:
            keep &= self._near_dedup(bitmaps, keep)

        return self._apply_mask(results, bitmaps, keep)

    def _exact_dedup(self, normalized: list[str]) -> np.ndarray:
        seen = set()
        keep = np.zeros(len(normalized), dtype=bool)
        for i, text in enumerate(normalized):
            if text not in seen:
                seen.add(text)
                keep[i] = True
        return keep

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

    def _near_dedup(
        self,
        bitmaps: tuple[np.ndarray, np.ndarray],
        keep: np.ndarray,
        threshold: float = 0.9,
    ) -> np.ndarray:
        """Near-duplicate detection. Vectorized upper-triangle with early exit."""
        indices = np.where(keep)[0]
        n = len(indices)
        if n <= 1:
            return keep

        pos, neg = bitmaps
        pos_active = pos[indices]
        neg_active = neg[indices]

        # Self-similarity for normalization
        self_sims = (
            self._popcount_table[pos_active].sum(axis=1)
            + self._popcount_table[neg_active].sum(axis=1)
        ).astype(np.float32)
        self_sims[self_sims == 0] = 1

        alive = np.ones(n, dtype=bool)

        for i in range(n - 1):
            if not alive[i]:
                continue

            # Candidates: j > i and still alive
            candidates = np.where(alive[i + 1 :])[0] + (i + 1)
            if len(candidates) == 0:
                continue

            # Vectorized similarity: doc i vs all candidates
            pi, ni = pos_active[i], neg_active[i]
            pj, nj = pos_active[candidates], neg_active[candidates]

            agree = self._popcount_table[pj & pi].sum(axis=1) + self._popcount_table[
                nj & ni
            ].sum(axis=1)
            disagree = self._popcount_table[pj & ni].sum(axis=1) + self._popcount_table[
                nj & pi
            ].sum(axis=1)

            sims = (agree - disagree) / np.sqrt(self_sims[i] * self_sims[candidates])

            # Mark duplicates
            alive[candidates[sims > threshold]] = False

        # Map back to original space
        mask = np.ones(len(pos), dtype=bool)
        mask[indices[~alive]] = False
        return mask

    def _apply_mask(
        self,
        results: list[dict],
        bitmaps: Optional[tuple[np.ndarray, np.ndarray]],
        mask: np.ndarray,
    ) -> tuple[list[dict], Optional[tuple[np.ndarray, np.ndarray]]]:
        filtered_results = [r for i, r in enumerate(results) if mask[i]]
        if bitmaps is not None:
            pos, neg = bitmaps
            filtered_bitmaps = (pos[mask], neg[mask])
        else:
            filtered_bitmaps = None
        return filtered_results, filtered_bitmaps


class Retriever:
    def __init__(
        self,
        db: Database,
        hdc: HDCEncoder,
        dedup: Deduplicator,
        model: ModelManager,
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
        self._token_pos, self._token_neg = db.load_token_hdc()
        self._vocab_row_ptrs, self._vocab_postings = db.load_vocab_index()
        self._idf = self.db.load_idf()
        self._median_idf = self.db.get_config("median_idf") or 0
        self._token_counts = self.db.get_token_counts()

    def _max_candidates(self) -> int:
        """Max docs to score - sized to cache budget."""
        bytes_per_doc = 2 * self.db._bytes_per_bitmap()
        budget_bytes = self.config.hdc_search_mb * 1024 * 1024
        return int(budget_bytes * 0.9 / bytes_per_doc)

    def _get_vocab_candidates(
        self, query_tokens: list[int], idf: dict[int, float]
    ) -> list[int]:
        """Get candidate doc indices from vocab index, weighted by IDF."""
        if self._vocab_row_ptrs is None:
            return []

        row_ptrs, postings = self._vocab_row_ptrs, self._vocab_postings
        vocab_size = len(row_ptrs) - 1

        chunks, weights = [], []
        for tid in query_tokens:
            if tid >= vocab_size:
                continue
            start, end = row_ptrs[tid], row_ptrs[tid + 1]
            if start < end:
                chunks.append(postings[start:end])
                weights.append(
                    np.full(end - start, idf.get(tid, 1.0), dtype=np.float32)
                )

        if not chunks:
            return []

        hdv_ids = np.concatenate(chunks)
        scores = np.bincount(
            hdv_ids, weights=np.concatenate(weights), minlength=self.db.count()
        )

        max_cand = self._max_candidates()
        nz = np.count_nonzero(scores)
        if nz == 0:
            return []

        k = min(max_cand, nz)
        top = np.argpartition(scores, -k)[-k:]
        return top[np.argsort(scores[top])[::-1]].tolist()

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

    def release_indices(self) -> None:
        import gc

        # Explicitly delete memmaps to release file handles (required on Windows)
        if self._token_pos is not None:
            del self._token_pos
        if self._token_neg is not None:
            del self._token_neg
        if self._vocab_row_ptrs is not None:
            del self._vocab_row_ptrs
        if self._vocab_postings is not None:
            del self._vocab_postings
        self._token_pos = None
        self._token_neg = None
        self._vocab_row_ptrs = None
        self._vocab_postings = None
        gc.collect()

    def search(
        self,
        query: str,
        token_budget: int,
        track: bool = True,
        enabled_sources: set = None,
    ) -> list[dict]:
        if self.db.count() == 0:
            return []

        self.logger.info(f"[Search] {len(query)}: Length query")

        query_tokens = self.model.tokenizer.encode(query, add_special_tokens=False)
        candidates = self._get_vocab_candidates(query_tokens, self.hdc.idf)

        max_doc_size = token_budget / self.config.min_context
        candidates = [c for c in candidates if self._token_counts[c] <= max_doc_size]

        self.logger.info(f"[Search] {len(candidates):,} candidates")

        if not candidates:
            return []

        query_emb = self.model.extract_embeddings([query])
        if track:
            search_emb = self._blend(query_emb)
            self._turns.append(query_emb.squeeze(0).cpu())
        else:
            search_emb = query_emb

        bitmaps = self.hdc.encode(search_emb)
        indices, scores = self.db.search(
            bitmaps["pos"], bitmaps["neg"], candidates, len(candidates)
        )

        if not indices:
            return []

        ranked = sorted(
            [(i, s, self._token_counts[i]) for i, s in zip(indices, scores)],
            key=lambda x: x[1],
            reverse=True,
        )

        # Token-budget-aware cap: keep enough to survive dedup
        dedup_budget = token_budget * 10
        cumulative, cutoff = 0, len(ranked)
        for idx, (_, _, tc) in enumerate(ranked):
            cumulative += tc
            if cumulative >= dedup_budget:
                cutoff = idx + 1
                break
        ranked = ranked[:cutoff]

        sel_idx = [r[0] for r in ranked]
        score_map = {r[0]: r[1] for r in ranked}
        memories = self.db.get_memories(sel_idx)

        # Filter by enabled sources
        if enabled_sources:
            memories = {
                idx: mem
                for idx, mem in memories.items()
                if mem.get("metadata", {}).get("source") in enabled_sources
            }
            sel_idx = [i for i in sel_idx if i in memories]

        if len(sel_idx) > 1:
            temp, temp_scores = [], []
            for i in sel_idx:
                if i in memories:
                    temp.append({"memory": memories[i], "hdv_idx": i})
                    temp_scores.append(score_map[i])

            bitmaps = self.db.get_bitmaps(sel_idx)
            scores_arr = np.array(temp_scores, dtype=np.float32)
            deduped, _ = self.dedup.dedup(temp, bitmaps, scores_arr)
        else:
            deduped = [
                {"memory": memories[i], "hdv_idx": i} for i in sel_idx if i in memories
            ]

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
            f"[Search] {len(ranked)}→{len(deduped)}→{len(results)} ({final_tokens:,} tokens)"
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
        self.logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    def extract_embeddings(self, texts: list[str]) -> torch.Tensor:
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

        if idf_weights is not None:
            weights = idf_weights[input_ids_gpu]
        else:
            weights = torch.ones_like(attn_gpu, dtype=torch.float32)

        w_expanded = weights.unsqueeze(-1) * mask
        pooled = (embeddings * w_expanded).sum(1) / w_expanded.sum(1).clamp(1e-9)
        return F.normalize(pooled, p=2, dim=1).cpu()

    def generate_stream(self, prompt: str):
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

        # Dataset filtering - all sources enabled by default
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

        # Load token HDC for query expansion
        self._token_pos, self._token_neg = self.db.load_token_hdc()

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

    def compress_expand_context(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> str:
        """
        Get context with compression:
        1. Retrieve context normally
        2. Extract unique tokens preserving order
        3. Remove tokens with IDF < median(IDF of context tokens)
        4. Return unique high-IDF tokens in original relative order
        """
        import time

        # Get normal context first
        context = self.get_context(query, token_budget, track)
        if not context:
            return ""

        # Tokenize
        tokens = self.model.tokenizer.encode(context, add_special_tokens=False)
        if not tokens:
            return ""

        # Get unique tokens preserving order
        seen = set()
        unique_tokens = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        # Get IDF data
        idf = self.hdc.idf
        if not idf:
            return context

        # Calculate LOCAL median IDF from the retrieved context tokens
        context_idfs = [idf.get(t, 0) for t in unique_tokens if idf.get(t, 0) > 0]
        if not context_idfs:
            return context
        local_median_idf = float(np.median(context_idfs))
        if local_median_idf <= 0:
            return context

        self.logger.info(
            f"[Compress] Raw context: {len(context)} chars, "
            f"{len(tokens)} tokens, {len(unique_tokens)} unique, median IDF: {local_median_idf:.3f}"
        )

        # Filter to high-IDF tokens (IDF >= local median)
        high_idf_tokens = [
            t for t in unique_tokens if idf.get(t, 0) >= local_median_idf
        ]
        if not high_idf_tokens:
            return context

        self.logger.info(
            f"[Compress] Filtered to {len(high_idf_tokens)} high-IDF tokens"
        )

        # Decode back to text
        compressed = self.model.tokenizer.decode(
            high_idf_tokens, skip_special_tokens=True
        )
        return compressed

    def add_turn(self, text: str) -> None:
        self.retriever.add_turn(text)

    def clear_index(self) -> None:
        self.retriever.release_indices()
        # Also release HdRAG's own memmap references
        if hasattr(self, "_token_pos") and self._token_pos is not None:
            del self._token_pos
            self._token_pos = None
        if hasattr(self, "_token_neg") and self._token_neg is not None:
            del self._token_neg
            self._token_neg = None
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

        # Release mmap handles before clearing (required on Windows)
        self.logger.info("Clearing existing index...")
        self.retriever.release_indices()
        # Also release HdRAG's own memmap references
        if hasattr(self, "_token_pos") and self._token_pos is not None:
            del self._token_pos
            self._token_pos = None
        if hasattr(self, "_token_neg") and self._token_neg is not None:
            del self._token_neg
            self._token_neg = None
        import gc

        gc.collect()
        if self.db.corpus_hdv_file.exists():
            self.db.corpus_hdv_file.unlink()
        self.db.clear()

        # Pass 1: Build vocabulary
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

        # Dedupe
        seen, unique = set(), []
        for d in docs:
            if d["id"] not in seen:
                seen.add(d["id"])
                unique.append(d)

        if len(unique) < len(docs):
            self.logger.info(f"Removed {len(docs) - len(unique):,} duplicates")
        docs = unique

        if not docs:
            self.logger.info("No new documents")
            return 0

        self.logger.info(f"Full rebuild: {len(docs):,} documents")
        self.db.set_config("hdc_seed", self.config.hdc_seed)

        # Compute IDF
        self.logger.info("Computing IDF...")
        n_docs = len(docs)
        idf = {tid: math.log((n_docs + 1) / (df + 1)) for tid, df in vocab_df.items()}
        self.db.save_idf(dict(vocab_df), n_docs)

        self.logger.info(f"Vocabulary: {len(idf):,} terms")
        self.db.set_config(
            "median_doc_length", statistics.median(d["token_count"] for d in docs)
        )
        self.hdc.idf = idf
        self.hdc.median_doc_length = self.db.get_config("median_doc_length")
        self.model.set_idf_weights(idf)

        # Build token HDC index
        self.logger.info("Building token HDC index...")
        emb_table = CUDA.get("embedding_table")
        token_bitmaps = self.hdc.encode(emb_table)
        self.db.save_token_hdc(token_bitmaps["pos"], token_bitmaps["neg"])

        # Pass 2: Encode
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

        # Build vocab index (token_id → list of hdv_idx)
        self.logger.info("Building vocab index...")
        vocab_index: dict[int, list[int]] = {}
        for hdv_idx, doc in enumerate(docs):
            for token_id in doc.get("tokens", set()):
                if token_id not in vocab_index:
                    vocab_index[token_id] = []
                vocab_index[token_id].append(hdv_idx)

        self.db.save_vocab_index(vocab_index, len(self.model.tokenizer))

        CUDA.clear()
        self.logger.info(f"Index complete: {self.db.count():,} memories")

        # Reorganize to blocked layout
        self.db.finalize_index()

        # Reload indices for retriever
        self.retriever._token_pos, self.retriever._token_neg = self.db.load_token_hdc()
        self.retriever._vocab_row_ptrs, self.retriever._vocab_postings = (
            self.db.load_vocab_index()
        )
        self.retriever._token_counts = self.db.get_token_counts()
        self.retriever._idf = self.db.load_idf()

        return len(docs)

    def _process_chunk(
        self,
        items: list[tuple[str, Optional[dict]]],
        source: str,
        special: set[int],
        vocab_df: Counter[int],
        docs: list[dict],
    ) -> int:
        """Process a batch of texts, updating vocab and docs."""

        texts = [t for t, _ in items]
        chunk_metas = [m for _, m in items]

        # Generate IDs
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

        # Filter to new items
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
            # Track tokens for vocab index
            doc_tokens = set(toks) - special
            vocab_df.update(doc_tokens)

            # Build metadata
            doc_meta = {"source": source}
            if meta:
                doc_meta.update(meta)

            docs.append(
                {
                    "id": mid,
                    "text": text,
                    "metadata": doc_meta,
                    "token_count": len(toks),
                    "tokens": doc_tokens,  # Store for vocab index
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
                compress_expand = gr.Checkbox(value=False, label="Compress Context")

            with gr.Accordion("🔍 Debug: Full Prompt Viewer", open=False):
                debug_viewer = gr.Code(
                    label="Full Prompt",
                    language="markdown",
                    lines=20,
                    value="Send a message to see the full prompt...",
                )

            def respond(message, history, budget, use_mem, use_compress_expand):
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

                if use_mem:
                    if use_compress_expand:
                        memory = hdrag.compress_expand_context(
                            message, budget, track=True
                        )
                    else:
                        memory = hdrag.get_context(message, budget, track=True)
                else:
                    memory = ""
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
                [msg, chatbot, budget_slider, use_memory, compress_expand],
                [msg, chatbot, debug_viewer],
            )
            send_btn.click(
                respond,
                [msg, chatbot, budget_slider, use_memory, compress_expand],
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

                # Get available sources and their counts
                source_counts = hdrag.db.source_counts()
                if source_counts:
                    gr.Markdown("**Select datasets to include in search:**")
                    dataset_checkboxes = gr.CheckboxGroup(
                        choices=[
                            f"{src} ({count:,})" for src, count in source_counts.items()
                        ],
                        value=[
                            f"{src} ({count:,})" for src, count in source_counts.items()
                        ],  # all selected
                        label="Enabled Datasets",
                    )

                    def update_enabled_sources(selected):
                        # Extract source name from "source (count)" format
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

                # Sparsity gauge
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

                # Similarity histogram
                sims = hdrag.db.sample_similarities(5000)
                if sims:
                    sim_fig = go.Figure(
                        go.Histogram(
                            x=sims,
                            nbinsx=50,
                            marker_color="#2196F3",
                            opacity=0.75,
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

                # Dimension activation
                pos_freq, neg_freq = hdrag.db.dimension_activation()
                if len(pos_freq) > 0:
                    # Downsample for display if too many dims
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
                        height=350,
                        margin=dict(t=40, b=40),
                        showlegend=False,
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
