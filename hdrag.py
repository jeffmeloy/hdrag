"""
hdrag - Hyperdimensional Retrieval-Augmented Generation (Memory Engine)

Indexing:  text → strip keys → tokenize → project → IDF weight → ternary quantize → pos/neg bitmaps
Retrieval: query → strip keys → encode → prune → threshold → MMR → context

This module has zero dependency on the Gradio UI (hdrag_gradio.py)
or the inference pipeline (generation lives in hdrag_model.InferenceEngine).
It depends on hdrag_model for Config and tokenizer only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import re
import os
import sqlite3
import zlib
import statistics
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hdrag_model import Config, Tokenizer


class ArrayStore:
    """Scoped lifecycle manager for shared numpy arrays."""

    __slots__ = ("_store",)

    def __init__(self):
        self._store: dict[str, np.ndarray] = {}

    def set(self, k: str, v: np.ndarray) -> np.ndarray:
        self._store[k] = v
        return v

    def get(self, k: str) -> Optional[np.ndarray]:
        return self._store.get(k)

    def pop(self, k: str) -> Optional[np.ndarray]:
        return self._store.pop(k, None)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def clear(self):
        self._store.clear()

    def __contains__(self, k: str) -> bool:
        return k in self._store

    def nbytes(self) -> int:
        return sum(v.nbytes for v in self._store.values() if hasattr(v, "nbytes"))


_arrays = ArrayStore()


# Structural key stripping for better retrieval focus (e.g. "question:", "answer:", etc.)
_STRUCTURAL_PREFIX_RE = re.compile(r"^\w+\s*:\s*", re.MULTILINE)


# Flat key pairs
_HUMAN_KEYS = (
    "human",
    "user",
    "prompt",
    "problem",
    "question",
    "instruction",
    "message_1",
)

_ASSIST_KEYS = ("gpt", "assistant", "response", "answer", "output", "message_2")


def bstride(d_hdc: int) -> int:
    d = (d_hdc + 7) // 8
    return d + (8 - d % 8) % 8


def score64(dp, dn, qp, qn) -> np.ndarray:
    if qp.ndim == 1:
        qp, qn = qp[None, :], qn[None, :]
    agree = np.bitwise_count((dp & qp) | (dn & qn)).sum(1, dtype=np.int32)
    disag = np.bitwise_count((dp & qn) | (dn & qp)).sum(1, dtype=np.int32)
    return (agree - disag).astype(np.float32)


def score64_partitioned(
    dp,
    dn,
    qp,
    qn,
    uni_mask,
    ngram_mask,
    doc_lit_uni,
    q_lit_uni,
    doc_lit_ngram,
    q_lit_ngram,
    alpha,
) -> np.ndarray:
    """Region-separated scoring with independent normalization."""
    if qp.ndim == 1:
        qp, qn = qp[None, :], qn[None, :]
    agree = (dp & qp) | (dn & qn)
    disag = (dp & qn) | (dn & qp)

    ua = np.bitwise_count(agree & uni_mask).sum(1, dtype=np.int32).astype(np.float32)
    ud = np.bitwise_count(disag & uni_mask).sum(1, dtype=np.int32).astype(np.float32)
    uni = (ua - ud) / np.sqrt(doc_lit_uni * q_lit_uni).clip(1.0)

    na = np.bitwise_count(agree & ngram_mask).sum(1, dtype=np.int32).astype(np.float32)
    nd = np.bitwise_count(disag & ngram_mask).sum(1, dtype=np.int32).astype(np.float32)
    ngram = (na - nd) / np.sqrt(doc_lit_ngram * q_lit_ngram).clip(1.0)

    return alpha * uni + (1.0 - alpha) * ngram


def mask64_for_dims(dims: int, stride_bytes: int) -> np.ndarray:
    n_words = stride_bytes // 8
    m = np.full(n_words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    pad_bits = n_words * 64 - dims
    if pad_bits > 0:
        keep = 64 - pad_bits
        m[-1] = (
            np.uint64((1 << keep) - 1) if keep < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
        )
    return m[None, :]


def _region_mask_u64(start: int, end: int, total: int, stride: int) -> np.ndarray:
    """Packed uint64 bit mask with bits [start, end) set."""
    bits = np.zeros(total, dtype=np.uint8)
    bits[start:end] = 1
    packed = np.packbits(bits)
    pad = stride - len(packed)
    if pad > 0:
        packed = np.pad(packed, (0, pad))
    return packed.view(np.uint64)[None, :]


def adaptive_threshold(vals: np.ndarray) -> float:
    n = len(vals)
    if n < 4:
        return float(vals.max()) if n > 0 else 0.0
    x = np.sort(vals)
    m = n // 2
    median = (x[m - 1] + x[m]) * 0.5
    lower = x[:m]
    L2 = 2.0 * np.dot(lower, np.arange(m)) / (m * (m - 1)) - lower.mean()
    return (median + L2 * math.log(n)) if L2 > 1e-9 else median


def strip_structural_keys(text: str) -> str:
    return _STRUCTURAL_PREFIX_RE.sub("", text)


def _chunks(xs: list, n: int) -> Generator[list, None, None]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def extract_text(item: dict) -> str:
    def _join(msgs, role_key="from", text_key=None):
        out = []
        for m in msgs:
            rk = m.get(role_key, m.get("role", ""))
            if rk == "system":
                continue
            text = m.get(text_key or "value", m.get("content", ""))
            if text:
                out.append(f"{rk}: {text}" if rk else text)
        return "\n\n".join(out)

    # Direct conversation list
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

    parts = []
    for k in _HUMAN_KEYS:
        if item.get(k):
            parts.append(f"{k}: {item[k]}")
            break
    for k in _ASSIST_KEYS:
        if item.get(k):
            parts.append(f"{k}: {item[k]}")
            break
    if parts:
        if item.get("input"):
            parts.insert(1, f"input: {item['input']}")
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


def _chunk_text(text: str, chunk_size: int, tokenizer) -> list[str]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [text] if text.strip() else []

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def _flush():
        nonlocal buf, buf_tokens
        if buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = [], 0

    def _split_long(block: str):
        nonlocal buf, buf_tokens
        words, wbuf, wcount = block.split(), [], 0
        for word in words:
            wn = tokenizer.count_tokens(word)
            if wcount + wn <= chunk_size:
                wbuf.append(word)
                wcount += wn
            else:
                if wbuf:
                    chunks.append(" ".join(wbuf))
                wbuf, wcount = [word], wn
        if wbuf:
            chunks.append(" ".join(wbuf))
        buf, buf_tokens = [], 0

    for para in paras:
        n = tokenizer.count_tokens(para)
        if buf_tokens + n <= chunk_size:
            buf.append(para)
            buf_tokens += n
            continue
        _flush()
        if n <= chunk_size:
            buf, buf_tokens = [para], n
        else:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                sn = tokenizer.count_tokens(sent)
                if buf_tokens + sn <= chunk_size:
                    buf.append(sent)
                    buf_tokens += sn
                elif sn <= chunk_size:
                    if buf:
                        chunks.append(" ".join(buf))
                        buf, buf_tokens = [], 0
                    buf, buf_tokens = [sent], sn
                else:
                    if buf:
                        chunks.append(" ".join(buf))
                        buf, buf_tokens = [], 0
                    _split_long(sent)
    _flush()
    return chunks


def iter_dataset(
    path: Path,
    tokenizer: Tokenizer = None,
    chunk_size: int = 1024,
) -> Generator[dict, None, None]:
    if path.suffix in (".txt", ".md", ".html", ".xml"):
        text = path.read_text(encoding="utf-8")
        if tokenizer and chunk_size:
            parts = _chunk_text(text, chunk_size, tokenizer)
            for i, part in enumerate(parts):
                yield {"text": part, "chunk_idx": i, "total_chunks": len(parts)}
        else:
            yield {"text": text}
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        try:
            pf.read_row_group(0).to_pylist()
            use_row_groups = True
        except Exception:
            use_row_groups = False

        if use_row_groups:
            for i in range(pf.metadata.num_row_groups):
                yield from pf.read_row_group(i).to_pylist()
        else:
            schema = pf.schema_arrow
            flat_cols = [
                f.name
                for f in schema
                if str(f.type).startswith(
                    ("string", "utf8", "large_string", "int", "float", "double", "bool")
                )
            ] or None
            yield from pq.read_table(path, columns=flat_cols).to_pylist()
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


_trim_log = logging.getLogger(__name__ + ".trim")


def _linux_malloc_trim():
    """Release free heap pages back to the OS on Linux."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).malloc_trim(0)
    except Exception:
        pass


class _WinMemHelper:
    """Lazy-initialized Windows memory management via ctypes."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        # kernel32 prototypes
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.SetProcessWorkingSetSizeEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        k32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL

        # psapi
        psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        psapi.EmptyWorkingSet.restype = wintypes.BOOL

        self.k32 = k32
        self.psapi = psapi

    @classmethod
    def get(cls) -> Optional["_WinMemHelper"]:
        if os.name != "nt":
            return None
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = cls()
                    except Exception:
                        return None
        return cls._instance

    def trim_working_set(self, *handles):
        for h in handles:
            self.psapi.EmptyWorkingSet(h)

    def current_process(self):
        return self.k32.GetCurrentProcess()

    def open_child(self, pid: int):
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_QUERY_INFORMATION = 0x0400
        return self.k32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid
        )

    def purge_standby_list(self):
        """Purge the system standby page list (requires SeProfileSingleProcessPrivilege)."""
        ctypes = self._ctypes
        wintypes = self._wintypes

        SE_PRIVILEGE_ENABLED = 0x00000002
        TOKEN_ADJUST_PRIVILEGES = 0x0020
        TOKEN_QUERY = 0x0008

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class LUID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

        class TOKEN_PRIVILEGES(ctypes.Structure):
            _fields_ = [
                ("PrivilegeCount", wintypes.DWORD),
                ("Privileges", LUID_AND_ATTRIBUTES * 1),
            ]

        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.LookupPrivilegeValueW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            ctypes.POINTER(LUID),
        ]
        advapi32.LookupPrivilegeValueW.restype = wintypes.BOOL
        advapi32.AdjustTokenPrivileges.argtypes = [
            wintypes.HANDLE,
            wintypes.BOOL,
            ctypes.POINTER(TOKEN_PRIVILEGES),
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        advapi32.AdjustTokenPrivileges.restype = wintypes.BOOL
        ntdll.NtSetSystemInformation.argtypes = [
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        ntdll.NtSetSystemInformation.restype = ctypes.c_long

        hToken = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            self.current_process(),
            TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
            ctypes.byref(hToken),
        ):
            return

        try:
            luid = LUID()
            if not advapi32.LookupPrivilegeValueW(
                None, "SeProfileSingleProcessPrivilege", ctypes.byref(luid)
            ):
                return
            tp = TOKEN_PRIVILEGES()
            tp.PrivilegeCount = 1
            tp.Privileges[0].Luid = luid
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
            advapi32.AdjustTokenPrivileges(
                hToken,
                False,
                ctypes.byref(tp),
                ctypes.sizeof(tp),
                None,
                None,
            )
            if ctypes.get_last_error() == 1300:
                return
        finally:
            self.k32.CloseHandle(hToken)


def soft_trim():
    """Move faulted mmap pages from working set to standby."""
    gc.collect()
    if os.name != "nt":
        _linux_malloc_trim()
        return
    try:
        wm = _WinMemHelper.get()
        if wm:
            wm.trim_working_set(wm.current_process())
    except Exception as exc:
        _trim_log.info("soft_trim error: %s", exc)


def trim_memory(*child_procs):
    """Release unused memory (VRAM, Python GC, OS pages)."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass

    gc.collect()

    if os.name != "nt":
        _linux_malloc_trim()
        return

    try:
        wm = _WinMemHelper.get()
        if not wm:
            return
        handles_to_close = []
        handles = [wm.current_process()]

        for proc in child_procs:
            if proc is None or proc.poll() is not None:
                continue
            try:
                h = wm.open_child(proc.pid)
                if h:
                    handles.append(h)
                    handles_to_close.append(h)
            except Exception:
                pass

        wm.trim_working_set(*handles)
        for h in handles_to_close:
            wm.k32.CloseHandle(h)
        wm.purge_standby_list()

    except Exception as exc:
        _trim_log.info("trim_memory Windows error: %s", exc)


class Database:
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS memories (
            hdv_idx INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
            text BLOB NOT NULL, metadata JSON, token_count INTEGER,
            uni_lit INTEGER);
        CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS idf (
            token_id INTEGER PRIMARY KEY, doc_freq INTEGER NOT NULL,
            weight REAL NOT NULL) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_memory_id ON memories(id);
    """

    def __init__(self, db_path: Path, config: Config, logger=None):
        self.db_path = db_path
        self.config = config
        self.logger = logger
        self._stride = bstride(config.hdc_dimensions)
        self._lock = threading.Lock()

        # Sidecar file paths
        parent = db_path.parent
        self.corpus_file = parent / "corpus_hdc.idx"
        self.vocab_file = parent / "vocab_hdc.idx"
        self.lit_file = parent / "lit.idx"
        self.uni_lit_file = parent / "uni_lit.idx"
        self.ngram_lit_file = parent / "ngram_lit.idx"

        # Mmap state
        self._corpus_cache = None
        self._mmap_raw = None
        self._mmap_lit = None
        self._mmap_uni_lit = None
        self._mmap_ngram_lit = None
        self._count: int | None = None

        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA cache_size=-{self.config.sqlite_cache_kb}",
            "PRAGMA temp_store=MEMORY",
        ):
            self._db.execute(pragma)
        self._init_db()

    def _init_db(self):
        with self._conn() as c:
            c.executescript(self.SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(memories)")}
            if "uni_lit" not in cols:
                c.execute("ALTER TABLE memories ADD COLUMN uni_lit INTEGER")
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
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except:
                self._db.rollback()
                raise

    def _query(self, sql, params=()):
        with self._conn() as c:
            return c.execute(sql, params).fetchall()

    def _chunked(self, sql, ids, extract):
        result = {}
        with self._conn() as c:
            for chunk in _chunks(ids, self.config.sqlite_max_vars):
                ph = ",".join("?" * len(chunk))
                for row in c.execute(sql.format(ph), tuple(chunk)).fetchall():
                    k, v = extract(row)
                    result[k] = v
        return result

    def _load_sidecar(
        self, path: Path, n: int, dtype=np.float32
    ) -> Optional[np.ndarray]:
        """Load a sidecar file via mmap if it exists and has the right size."""
        expected = n * np.dtype(dtype).itemsize
        if path.exists() and path.stat().st_size == expected:
            return np.memmap(path, dtype=dtype, mode="r", shape=(n,))
        return None

    def _ensure_sidecar(
        self,
        path: Path,
        cache_attr: str,
        n: int,
        generate_fn: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Generic sidecar accessor: return cached mmap, or load from disk, or generate."""
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached

        if n == 0:
            return np.zeros(0, dtype=np.float32)

        # Try loading existing file
        loaded = self._load_sidecar(path, n)
        if loaded is not None:
            setattr(self, cache_attr, loaded)
            return loaded

        # Generate, save, and mmap
        arr = generate_fn()
        if arr is not None:
            arr.tofile(path)
            del arr
            loaded = self._load_sidecar(path, n)
            if loaded is not None:
                setattr(self, cache_attr, loaded)
                return loaded

        return np.zeros(n, dtype=np.float32)

    def _convert_interleaved_to_blocked(self, n: int):
        """Reorganize interleaved [pos|neg, pos|neg, ...] → blocked [pos..., neg...]."""
        d = self._stride
        if (
            self.get_config("corpus_layout") == "blocked"
            or not self.corpus_file.exists()
            or self.corpus_file.stat().st_size != n * d * 2
        ):
            return
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

    def _open_corpus(self):
        if self._corpus_cache is not None:
            return self._corpus_cache
        if not self.corpus_file.exists():
            return None
        n = self.count()
        if n == 0:
            return None

        d = self._stride
        self._convert_interleaved_to_blocked(n)

        raw = np.memmap(self.corpus_file, dtype=np.uint8, mode="r")
        self._mmap_raw = raw
        half = n * d
        pos, neg = raw[:half].reshape(n, d), raw[half : 2 * half].reshape(n, d)
        pos64, neg64 = pos.view(np.uint64), neg.view(np.uint64)

        # Ensure lit sidecar exists, then mmap it
        expected = n * np.dtype(np.float32).itemsize
        if not self.lit_file.exists() or self.lit_file.stat().st_size != expected:
            tmp_lit = np.bitwise_count(pos64 | neg64).sum(axis=1, dtype=np.float32)
            tmp_lit.tofile(self.lit_file)
            del tmp_lit
        self._mmap_lit = self._load_sidecar(self.lit_file, n)
        lit = self._mmap_lit

        # Open sidecar mmaps for uni/ngram lit if they exist
        self._mmap_uni_lit = self._load_sidecar(self.uni_lit_file, n)
        self._mmap_ngram_lit = self._load_sidecar(self.ngram_lit_file, n)

        self._corpus_cache = (pos, neg, pos64, neg64, lit)
        if self.logger:
            mb = self.corpus_file.stat().st_size / 1e6
            self.logger.info(f"Corpus mmap: {mb:.0f}MB, {n:,} vectors")
        return self._corpus_cache

    def invalidate_corpus(self):
        self._corpus_cache = None
        self._mmap_raw = self._mmap_lit = self._mmap_uni_lit = self._mmap_ngram_lit = (
            None
        )

    def release_mmap_pages(self):
        """Close all mmap file handles so the OS can reclaim physical pages."""
        self.invalidate_corpus()
        gc.collect()

    def _get_lit_uni(self) -> np.ndarray:
        """Unigram-only lit counts via mmap sidecar."""
        n = self.count()

        def _generate():
            tmp = np.zeros(n, dtype=np.float32)
            populated = False
            try:
                for r in self._query(
                    "SELECT hdv_idx, uni_lit FROM memories WHERE uni_lit IS NOT NULL"
                ):
                    tmp[r["hdv_idx"]] = float(r["uni_lit"])
                    populated = True
            except Exception:
                pass
            if not populated:
                corpus = self._open_corpus()
                if corpus:
                    np.copyto(tmp, corpus[4][:n])
            return tmp

        return self._ensure_sidecar(self.uni_lit_file, "_mmap_uni_lit", n, _generate)

    def _get_lit_ngram(self, ngram_mask=None) -> np.ndarray:
        """N-gram region lit counts via mmap sidecar."""
        n = self.count()

        def _generate():
            if ngram_mask is None:
                return None
            corpus = self._open_corpus()
            if not corpus:
                return None
            _, _, pos64, neg64, _ = corpus
            return np.bitwise_count((pos64 | neg64) & ngram_mask).sum(
                axis=1, dtype=np.float32
            )

        return self._ensure_sidecar(
            self.ngram_lit_file, "_mmap_ngram_lit", n, _generate
        )

    def count(self) -> int:
        if self._count is None:
            self._count = self._query("SELECT COUNT(*) FROM memories")[0][0]
        return self._count

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
            "SELECT hdv_idx, id, text, metadata, token_count FROM memories WHERE hdv_idx IN ({})",
            indices,
            lambda r: (
                r["hdv_idx"],
                {
                    "id": r["id"],
                    "text": zlib.decompress(r["text"]).decode(),
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

    def get_source_map(self) -> dict[int, str]:
        return {
            r[0]: r[1] or ""
            for r in self._query(
                "SELECT hdv_idx, json_extract(metadata,'$.source') FROM memories"
            )
        }

    def get_bitmaps(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        corpus = self._open_corpus()
        if not corpus or not indices:
            d = self._stride
            return np.empty((0, d), np.uint8), np.empty((0, d), np.uint8)
        pos, neg, _, _, _ = corpus
        idx = np.array(indices)
        return np.ascontiguousarray(pos[idx]), np.ascontiguousarray(neg[idx])

    def get_search_arrays(
        self, indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        corpus = self._open_corpus()
        if not corpus or not indices:
            cols = self._stride // 8
            e = np.empty((0, cols), np.uint64)
            return e, e, np.empty(0, np.float32)
        _, _, pos64, neg64, lit = corpus
        idx = np.array(indices)
        return (
            np.ascontiguousarray(pos64[idx]),
            np.ascontiguousarray(neg64[idx]),
            lit[idx].copy(),
        )

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

    def load_df(self) -> dict[int, int]:
        return {
            r["token_id"]: r["doc_freq"]
            for r in self._query("SELECT token_id, doc_freq FROM idf")
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
                "INSERT OR REPLACE INTO config VALUES (?,?)", (key, json.dumps(value))
            )

    def _insert_memory_rows(self, c, items, hdv_start, lit_uni):
        """Shared row insertion used by both insert() and insert_rows()."""
        c.executemany(
            "INSERT INTO memories (hdv_idx,id,text,metadata,token_count,uni_lit) VALUES (?,?,?,?,?,?)",
            [
                (
                    hdv_start + i,
                    it["id"],
                    zlib.compress(it["text"].encode(), 1),
                    json.dumps(it["metadata"]),
                    it["token_count"],
                    int(lit_uni[i]) if lit_uni is not None else None,
                )
                for i, it in enumerate(items)
            ],
        )

    def insert(self, items: list[dict], bitmaps: dict[str, np.ndarray]):
        pos, neg = bitmaps["pos"], bitmaps["neg"]
        lit_uni = bitmaps.get("lit_uni")
        with self._conn() as c:
            start = c.execute(
                "SELECT COALESCE(MAX(hdv_idx),-1)+1 FROM memories"
            ).fetchone()[0]
            self._insert_memory_rows(c, items, start, lit_uni)
        with open(self.corpus_file, "ab") as f:
            for i in range(len(pos)):
                f.write(pos[i].tobytes())
                f.write(neg[i].tobytes())
        self.set_config("corpus_layout", "interleaved")
        self.invalidate_corpus()
        self._count = None

    def insert_rows(self, items: list[dict], bitmaps: dict, hdv_start: int):
        """Insert metadata rows into SQLite (no corpus file I/O)."""
        with self._conn() as c:
            self._insert_memory_rows(c, items, hdv_start, bitmaps.get("lit_uni"))
        self._count = None

    def finalize_index(self):
        n = self.count()
        if n == 0:
            return

        self._convert_interleaved_to_blocked(n)
        self.invalidate_corpus()

        # Generate missing sidecar files
        if self.corpus_file.exists() and not self.lit_file.exists():
            corpus = self._open_corpus()
            if corpus:
                arr = np.bitwise_count(corpus[2] | corpus[3]).sum(
                    axis=1, dtype=np.float32
                )
                arr.tofile(self.lit_file)
                del arr
                self.invalidate_corpus()

        if self.corpus_file.exists() and not self.uni_lit_file.exists():
            uni_lits = np.zeros(n, dtype=np.float32)
            populated = False
            try:
                for r in self._query(
                    "SELECT hdv_idx, uni_lit FROM memories WHERE uni_lit IS NOT NULL"
                ):
                    uni_lits[r["hdv_idx"]] = float(r["uni_lit"])
                    populated = True
            except Exception:
                pass
            if populated:
                uni_lits.tofile(self.uni_lit_file)

        with self._conn() as c:
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            c.execute("PRAGMA journal_mode=DELETE")

        # Force WAL/SHM removal
        self._db.close()
        for f in (self.db_path.with_suffix(".db-wal"), self.db_path.with_suffix(".db-shm")):
            if f.exists():
                f.unlink()
        trim_memory()
    
        # Reopen in WAL for query-time reads
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA cache_size=-{self.config.sqlite_cache_kb}",
            "PRAGMA temp_store=MEMORY",
        ):
            self._db.execute(pragma)
        self._count = None

    def search(
        self,
        q_pos,
        q_neg,
        target,
        candidates=None,
        logger=None,
        q_lit_uni: float = None,
        q_lit_ngram: float = None,
        uni_mask=None,
        ngram_mask=None,
        alpha: float = None,
    ):
        """Progressive pruning in geometric-mean-normalized HDC space."""
        corpus = self._open_corpus()
        if not corpus:
            return np.array([], np.int32), np.array([], np.float32)
        _, _, pos64, neg64, lit = corpus
        n = pos64.shape[0]

        corpus_idx = candidates if candidates is not None else np.arange(n)
        nc = len(corpus_idx)
        if nc == 0:
            return np.array([], np.int32), np.array([], np.float32)

        qp64 = q_pos.ravel().view(np.uint64)
        qn64 = q_neg.ravel().view(np.uint64)
        q_lit = float(np.bitwise_count(qp64 | qn64).sum(dtype=np.int32))
        n_cols = pos64.shape[1]

        max_passes = max(2, int(np.ceil(np.log2(nc / target)))) if nc > target else 1

        # Geometric column schedule: early passes (large candidate set) use
        # fewer columns for cheap coarse ranking; later passes (small set)
        # use more columns for finer discrimination.  Total columns = n_cols.
        # With ratio 2, widths double each pass → work per pass ≈ constant.
        if max_passes <= 1:
            col_slices = [(0, n_cols)]
        else:
            base = n_cols / (2.0**max_passes - 1)
            widths = [max(1, int(round(base * 2.0**i))) for i in range(max_passes)]
            widths[-1] = n_cols - sum(widths[:-1])
            col_slices = []
            pos = 0
            for w in widths:
                col_slices.append((pos, min(pos + w, n_cols)))
                pos += w

        cumul = np.zeros(nc, np.float32)
        surv = np.arange(nc)

        for s, e in col_slices:
            cs = corpus_idx[surv]
            cumul[surv] += score64(pos64[cs, s:e], neg64[cs, s:e], qp64[s:e], qn64[s:e])

            if len(surv) > target:
                keep = max(target, len(surv) // 2)
                surv = surv[np.argpartition(cumul[surv], -keep)[-keep:]]

        if logger:
            logger.info(f"[Prune] {nc:,}→{len(surv):,} passes: {max_passes}")

        # Final scoring on survivors only
        cs_final = corpus_idx[surv]

        use_partitioned = (
            uni_mask is not None
            and ngram_mask is not None
            and alpha is not None
            and q_lit_ngram is not None
        )

        if use_partitioned:
            scores = score64_partitioned(
                pos64[cs_final],
                neg64[cs_final],
                qp64,
                qn64,
                uni_mask,
                ngram_mask,
                self._get_lit_uni()[cs_final],
                q_lit_uni,
                self._get_lit_ngram(ngram_mask=ngram_mask)[cs_final],
                q_lit_ngram,
                alpha,
            )
        else:
            lit_uni = self._get_lit_uni()
            doc_denom = lit_uni[cs_final] if len(lit_uni) > 0 else lit[cs_final]
            q_denom = q_lit_uni if q_lit_uni is not None else q_lit
            scores = cumul[surv] / np.sqrt(doc_denom * q_denom).clip(1.0)

        order = np.argsort(-scores)
        return corpus_idx[surv[order]], scores[order]

    def clear(self):
        with self._conn() as c:
            c.execute("DELETE FROM memories")
            c.execute("DELETE FROM idf")
            c.execute("DELETE FROM config WHERE key != 'hdc_dimensions'")
        self.invalidate_corpus()
        trim_memory()
        for f in (
            self.corpus_file,
            self.lit_file,
            self.uni_lit_file,
            self.ngram_lit_file,
        ):
            if f.exists():
                f.unlink()
        self._count = None
        trim_memory()

    def close(self):
        self.invalidate_corpus()
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def stats(self) -> dict:
        return {
            "db_mb": self.db_path.stat().st_size / 1e6 if self.db_path.exists() else 0,
            "corpus_hdv_mb": self.corpus_file.stat().st_size / 1e6
            if self.corpus_file.exists()
            else 0,
            "corpus_resident": self._corpus_cache is not None,
            "corpus_mmap": self._mmap_raw is not None,
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
        _, _, pos64, neg64, _ = corpus
        dims, n = self.config.hdc_dimensions, pos64.shape[0]
        total = n * dims
        pos_bits = np.bitwise_count(pos64).sum(dtype=np.int64)
        neg_bits = np.bitwise_count(neg64).sum(dtype=np.int64)
        active = float(pos_bits + neg_bits)
        return {
            "positive": float(pos_bits / total),
            "negative": float(neg_bits / total),
            "zero": float((total - active) / total),
        }

    def sample_similarities(self) -> list[float]:
        corpus = self._open_corpus()
        if not corpus:
            return []
        _, _, p64, n64, _ = corpus
        n = p64.shape[0]
        if n < 2:
            return []
        max_pairs = n * (n - 1) // 2
        ns = min(max_pairs, int(max_pairs**0.5))
        ia = np.random.randint(0, n, ns * 2)
        ib = np.random.randint(0, n, ns * 2)
        mask = ia != ib
        ia, ib = ia[mask][:ns], ib[mask][:ns]
        return score64(p64[ia], n64[ia], p64[ib], n64[ib]).tolist()

    def dimension_activation(self) -> tuple[np.ndarray, np.ndarray]:
        corpus = self._open_corpus()
        if not corpus:
            return np.array([]), np.array([])
        pos, neg, _, _, _ = corpus
        dims = self.config.hdc_dimensions
        n = pos.shape[0]
        pf = np.zeros(dims, dtype=np.float64)
        nf = np.zeros(dims, dtype=np.float64)
        chunk = max(1, min(10000, 500_000_000 // dims))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            pf += np.unpackbits(np.ascontiguousarray(pos[s:e]), axis=1)[:, :dims].sum(
                axis=0
            )
            nf += np.unpackbits(np.ascontiguousarray(neg[s:e]), axis=1)[:, :dims].sum(
                axis=0
            )
        pf /= n
        nf /= n
        return pf.astype(np.float32), nf.astype(np.float32)


class HDCEncoder:
    """Dimension-partitioned skip-gram HDC encoder."""

    def __init__(self, config: Config, hdrag_dir: Path, db):
        self.config = config
        self.hdrag_dir = hdrag_dir
        self.db = db
        self.idf: dict[int, float] = db.load_idf()
        self.median_doc_length = db.get_config("median_doc_length") or 0.0
        self._stride = bstride(config.hdc_dimensions)
        self._corpus_vocab: list[int] = []
        self._remap: dict[int, int] = {}
        self._vocab_t: Optional[torch.Tensor] = None

        self._build_remap()
        self._init_hash_constants()
        self._init_partition()

    def _init_partition(self):
        """Compute dimension partition, codeword width, and region masks."""
        D = self.config.hdc_dimensions
        N = self.config.hdc_ngram
        stride = self._stride
        D_ngram = D // 2
        D_uni = D - D_ngram
        P_onset = max(1, N * (N - 1) // 2)
        C_local = math.ceil(math.log(max(D_ngram, 2)) / math.log(3))

        self._base_activate = max(
            C_local, round(math.sqrt(D_ngram * C_local / P_onset))
        )
        self._C_local = C_local
        self._D_uni = D_uni
        self._D_ngram = D_ngram
        self._uni_mask = _region_mask_u64(0, D_uni, D, stride)
        self._ngram_mask = _region_mask_u64(D_uni, D, D, stride)
        self._full_mask = mask64_for_dims(D, stride)

    def _init_hash_constants(self):
        rng = np.random.default_rng(self.config.hdc_seed ^ 0x4E475241)
        self._token_mix = np.uint64(int(rng.integers(1, 2**63, dtype=np.uint64)) | 1)
        self._hash_a = np.uint64(int(rng.integers(1, 2**63, dtype=np.uint64)) | 1)
        self._hash_b = np.uint64(int(rng.integers(1, 2**63, dtype=np.uint64)) | 1)

    def _build_remap(self):
        if self.idf:
            self._corpus_vocab = sorted(self.idf.keys())
            self._remap = {tok: i for i, tok in enumerate(self._corpus_vocab)}
            vocab_arr = np.array(self._corpus_vocab, dtype=np.int32)
            if len(vocab_arr) > 0:
                mx = int(vocab_arr.max()) + 1
                self._remap_arr = np.full(mx, -1, dtype=np.int32)
                self._remap_arr[vocab_arr] = np.arange(len(vocab_arr), dtype=np.int32)
            else:
                self._remap_arr = np.zeros(1, dtype=np.int32)
        else:
            self._corpus_vocab, self._remap = [], {}
            self._remap_arr = np.zeros(1, dtype=np.int32)
        if hasattr(self, "_vocab_f16"):
            delattr(self, "_vocab_f16")
        self._vocab_t = None

    def build_vocab_index(self):
        """Generate random HDC vectors for each token in the corpus vocabulary."""
        if self.db.vocab_file.exists():
            return
        self._build_remap()
        if not self._corpus_vocab:
            return

        dim = self.config.hdc_dimensions
        nv = len(self._corpus_vocab)
        seed = np.uint64(self.config.hdc_seed)
        base = np.empty((nv, dim), dtype=np.float32)
        dim_idx = np.arange(dim, dtype=np.uint64)

        # Stafford variant 13 mix constants
        MIX_A, MIX_B = np.uint64(0x9E3779B97F4A7C15), np.uint64(0x6C62272E07BB0142)
        MIX_C = np.uint64(0xBF58476D1CE4E5B9)
        MIX_D, MIX_E = np.uint64(0x517CC1B727220A95), np.uint64(0x94D049BB133111EB)
        S32, S31, S11 = np.uint64(32), np.uint64(31), np.uint64(11)
        SCALE = 1.0 / (1 << 53)

        CHUNK = 512
        for start in range(0, nv, CHUNK):
            end = min(start + CHUNK, nv)
            tids = np.array(self._corpus_vocab[start:end], dtype=np.uint64)[:, None]

            h1 = (tids ^ seed) * MIX_A + dim_idx * MIX_B
            h1 ^= h1 >> S32
            h1 *= MIX_C
            h1 ^= h1 >> S31

            h2 = (tids ^ seed) * MIX_D + dim_idx * MIX_E
            h2 ^= h2 >> S32
            h2 *= MIX_E
            h2 ^= h2 >> S31

            u1 = (h1 >> S11).astype(np.float64) * SCALE + 1e-15
            u2 = (h2 >> S11).astype(np.float64) * SCALE

            base[start:end] = (
                np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            ).astype(np.float32)

        # L2-normalize
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        base /= np.maximum(norms, 1e-8)

        # IDF-weighted mean subtraction (decorrelate high-frequency tokens)
        raw_df = self.db.load_df()
        cw = np.array(
            [self.idf.get(t, 0.0) * raw_df.get(t, 0) for t in self._corpus_vocab],
            np.float32,
        )
        cw_sum = cw.sum()
        if cw_sum > 0:
            cw /= cw_sum
            base -= (cw[:, None] * base).sum(axis=0, keepdims=True)

        with open(self.db.vocab_file, "wb") as f:
            f.write(base.astype(np.float16).tobytes())

    def compute_token_weights(self, vocab_size: int, special_ids: set[int]):
        w = np.zeros(vocab_size, dtype=np.float32)
        vocab_arr = np.array(self._corpus_vocab, dtype=np.int64)
        valid = vocab_arr < vocab_size
        idf_vals = np.array(
            [self.idf.get(t, 0.0) for t in self._corpus_vocab], np.float32
        )
        w[vocab_arr[valid]] = idf_vals[valid]
        special_arr = np.array(
            [t for t in special_ids if 0 <= t < vocab_size], dtype=np.int64
        )
        if len(special_arr):
            w[special_arr] = 0.0
        _arrays.set("idf_weights", w)

    def _open_vocab_mmap(self):
        if hasattr(self, "_vocab_f16") and self._vocab_f16 is not None:
            return
        if not self.db.vocab_file.exists():
            self.build_vocab_index()
        self._vocab_f16 = np.memmap(
            self.db.vocab_file,
            dtype=np.float16,
            mode="r",
            offset=0,
            shape=(len(self._corpus_vocab), self.config.hdc_dimensions),
        )

    def _ensure_vocab(self):
        """Load vocab tensor on demand.  compact() drops it after each query."""
        self._open_vocab_mmap()
        if self._vocab_t is None:
            self._vocab_t = torch.from_numpy(
                np.array(self._vocab_f16, dtype=np.float32)
            )

    def load_vocab_full(self):
        self._ensure_vocab()

    def release_vocab_full(self):
        self._vocab_t = None

    def compact(self):
        """Release the float32 vocab tensor between queries."""
        # self._vocab_t = None
        pass

    def deep_compact(self):
        """Full release of both the float32 tensor and the float16 mmap."""
        self._vocab_t = None
        if hasattr(self, "_vocab_f16") and self._vocab_f16 is not None:
            del self._vocab_f16
            self._vocab_f16 = None
        gc.collect()

    def _flat_offsets(self, token_ids=None, flat_ids=None, offsets=None):
        if flat_ids is not None:
            return flat_ids.astype(np.int64, copy=False), offsets
        if token_ids:
            arrs = [np.asarray(ids, dtype=np.int64) for ids in token_ids]
            flat = np.concatenate(arrs) if len(arrs) > 1 else arrs[0].copy()
            offs = np.zeros(len(token_ids) + 1, dtype=np.int64)
            np.cumsum([len(a) for a in arrs], out=offs[1:])
            return flat, offs
        raise ValueError("provide token_ids or flat_ids+offsets")

    def _remap_flat(self, flat_np):
        clipped = np.clip(flat_np, 0, len(self._remap_arr) - 1)
        remapped = self._remap_arr[clipped]
        return remapped >= 0, np.clip(remapped, 0, None)

    def project_query(self, token_ids):
        self._open_vocab_mmap()  # float16 mmap, no torch tensor
        flat = np.array(token_ids[0], dtype=np.int64)
        valid, safe = self._remap_flat(flat)
        rows = self._vocab_f16[safe[valid]].astype(np.float32)
        idf_w = _arrays.get("idf_weights")
        w = idf_w[flat[valid]] if idf_w is not None else np.ones(valid.sum())
        vec = (rows * w[:, None]).sum(axis=0, keepdims=True)
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
        return vec.astype(np.float16)

    def project(self, token_ids=None, flat_ids=None, offsets=None):
        self._ensure_vocab()
        flat_np, offs = self._flat_offsets(token_ids, flat_ids, offsets)
        mapped, safe = self._remap_flat(flat_np)

        idf_w = _arrays.get("idf_weights")
        safe_t = torch.from_numpy(safe.astype(np.int64))
        mapped_f = torch.from_numpy(mapped).float()
        w = (
            torch.from_numpy(idf_w[flat_np]) * mapped_f
            if idf_w is not None
            else mapped_f
        )

        unigrams = F.embedding_bag(
            safe_t,
            self._vocab_t,
            torch.from_numpy(offs.astype(np.int64)),
            per_sample_weights=w,
            mode="sum",
            include_last_offset=True,
        )
        unigrams = F.normalize(unigrams, p=2, dim=1)
        return unigrams.half().numpy()

    def _content_ngrams(
        self,
        flat_np: np.ndarray,
        seg_np: np.ndarray,
        mapped: np.ndarray,
        batch: int,
        dims: int,
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Skip-gram pairs hashed into the n-gram dimension region [D_uni, D)."""
        n = self.config.hdc_ngram
        T = len(flat_np)
        C = self._base_activate
        D_uni, D_ngram = self._D_uni, self._D_ngram

        token_hashes = flat_np.astype(np.uint64) * self._token_mix
        token_hashes ^= token_hashes >> np.uint64(32)

        idf_w = _arrays.get("idf_weights")
        token_idf = idf_w[flat_np] if idf_w is not None else np.ones(T, np.float32)

        positive_idf = token_idf[token_idf > 0]
        idf_scale = n / max(
            float(np.median(positive_idf)) if len(positive_idf) else 1.0, 1e-9
        )

        # Narrow accumulator: only D_ngram columns instead of full dims.
        accum = np.zeros((batch, D_ngram), dtype=np.int32)
        pair_counts = np.zeros(batch, dtype=np.int32)
        hash_b_slots = np.arange(C, dtype=np.uint64) * self._hash_b
        flat_size = batch * D_ngram
        D_ngram_u64 = np.uint64(D_ngram)

        # Use bitwise AND when D_ngram is a power of 2 (avoids uint64 division).
        dim_pow2 = (D_ngram & (D_ngram - 1)) == 0
        dim_mask = np.uint64(D_ngram - 1) if dim_pow2 else None

        for g in range(1, n):
            nw = T - g
            if nw <= 0:
                continue

            valid = (
                mapped[:nw] & mapped[g : nw + g] & (seg_np[:nw] == seg_np[g : nw + g])
            )
            v_idx = np.where(valid)[0]
            if len(v_idx) == 0:
                continue

            v_addrs = token_hashes[v_idx] + token_hashes[v_idx + g]
            v_seg = seg_np[v_idx].astype(np.intp)
            v_weights = np.clip(
                np.round(
                    np.minimum(token_idf[v_idx], token_idf[v_idx + g]) * idf_scale
                ).astype(np.int16),
                1,
                n,
            )

            pair_counts += np.bincount(v_seg, minlength=batch)

            h = v_addrs[:, None] * self._hash_a + hash_b_slots
            h ^= h >> np.uint64(32)

            # Fused linear index: compute seg*D_ngram + dim directly in uint64,
            # avoiding a separate (V,C)-shaped dim_idx intermediate.
            seg_offset = v_seg.astype(np.uint64)[:, None] * D_ngram_u64
            if dim_pow2:
                linear = (seg_offset + (h & dim_mask)).ravel().astype(np.intp)
            else:
                linear = (seg_offset + (h % D_ngram_u64)).ravel().astype(np.intp)

            # Sign + weight: use np.where to skip the intermediate `signs` array.
            w_col = v_weights[:, None].astype(np.int32)
            sign_bits = (h >> np.uint64(33)) & np.uint64(1)
            weighted = np.where(sign_bits, w_col, np.negative(w_col)).ravel()

            accum.ravel()[:] += np.bincount(
                linear, weights=weighted, minlength=flat_size
            ).astype(np.int32)

        # Adaptive threshold
        vpd = pair_counts.astype(np.float32) * C / D_ngram
        thresholds = np.maximum(2, np.ceil(np.sqrt(vpd))).astype(np.int32)[:, None]

        # Expand narrow accum back to full width for packbits alignment.
        # Columns [0, D_uni) are zero → produce zero bits, masked out anyway.
        full_accum = np.zeros((batch, dims), dtype=np.int32)
        full_accum[:, D_uni : D_uni + D_ngram] = accum

        ngram_p = np.packbits((full_accum >= thresholds).astype(np.uint8), axis=1)
        ngram_n = np.packbits((full_accum <= -thresholds).astype(np.uint8), axis=1)

        pad = stride - ngram_p.shape[1]
        if pad > 0:
            ngram_p = np.pad(ngram_p, ((0, 0), (0, pad)))
            ngram_n = np.pad(ngram_n, ((0, 0), (0, pad)))

        return (
            ngram_p.view(np.uint64) & self._ngram_mask,
            ngram_n.view(np.uint64) & self._ngram_mask,
        )

    @property
    def score_alpha(self) -> float:
        N = self.config.hdc_ngram
        return (N - 1) / N

    def adaptive_alpha(self, q_lit_uni: float, q_lit_ngram: float) -> float:
        """Per-query unigram weight based on n-gram confidence."""
        if q_lit_ngram <= 0 or self._D_ngram <= 0:
            return 1.0
        expected_ratio = self._D_ngram / max(self._D_uni, 1)
        actual_ratio = q_lit_ngram / max(q_lit_uni, 1.0)
        ng_confidence = min(1.0, actual_ratio / expected_ratio)
        max_ng_weight = 1.0 / self.config.hdc_ngram
        return 1.0 - max_ng_weight * ng_confidence

    @property
    def uni_mask_u64(self) -> np.ndarray:
        return self._uni_mask

    @property
    def ngram_mask_u64(self) -> np.ndarray:
        return self._ngram_mask

    @property
    def D_uni(self) -> int:
        return self._D_uni

    @property
    def D_ngram(self) -> int:
        return self._D_ngram

    def encode(self, unigrams=None, token_ids=None, flat_ids=None, offsets=None):
        """Encode documents into ternary HDC bitmaps."""
        self._ensure_vocab()
        n, dim = self.config.hdc_ngram, self.config.hdc_dimensions
        stride = self._stride
        tau = dim**-0.5

        # Prepare token arrays
        flat_np = seg_np = mapped = None
        if flat_ids is not None or token_ids:
            flat_np, offs = self._flat_offsets(token_ids, flat_ids, offsets)
            mapped, _ = self._remap_flat(flat_np)
            seg_np = np.repeat(np.arange(len(offs) - 1, dtype=np.int64), np.diff(offs))

        # Unigram projection + ternary quantization
        if unigrams is None:
            unigrams = self.project(
                token_ids=token_ids, flat_ids=flat_ids, offsets=offsets
            )
        batch = unigrams.shape[0]

        uni_np = unigrams.astype(np.float32)
        u_pos = np.packbits(uni_np > tau, axis=1)
        u_neg = np.packbits(uni_np < -tau, axis=1)
        pad = stride - u_pos.shape[1]
        if pad:
            u_pos = np.pad(u_pos, ((0, 0), (0, pad)))
            u_neg = np.pad(u_neg, ((0, 0), (0, pad)))

        u_p64 = u_pos.view(np.uint64) & self._uni_mask
        u_n64 = u_neg.view(np.uint64) & self._uni_mask

        lit_uni = np.bitwise_count(u_p64 | u_n64).sum(axis=1, dtype=np.int32)

        # N-gram layer
        if flat_np is not None and n > 1 and len(flat_np) > 1:
            ngram_p, ngram_n = self._content_ngrams(
                flat_np, seg_np, mapped, batch, dim, stride
            )
            lit_ngram = np.bitwise_count(ngram_p | ngram_n).sum(axis=1, dtype=np.int32)
            out_p64 = u_p64 | ngram_p
            out_n64 = u_n64 | ngram_n
        else:
            lit_ngram = np.zeros(batch, dtype=np.int32)
            out_p64 = u_p64
            out_n64 = u_n64

        return {
            "pos": out_p64.view(np.uint8).reshape(batch, stride).copy(),
            "neg": out_n64.view(np.uint8).reshape(batch, stride).copy(),
            "lit_uni": lit_uni,
            "lit_ngram": lit_ngram,
        }

    def release_workspace(self):
        pass

    def clear(self):
        if hasattr(self, "_vocab_f16"):
            delattr(self, "_vocab_f16")
        self._vocab_t = None
        self.db.invalidate_corpus()
        trim_memory()
        for f in (
            self.db.corpus_file,
            self.db.vocab_file,
            self.db.lit_file,
            self.db.uni_lit_file,
        ):
            if f.exists():
                f.unlink()
        self._corpus_vocab, self._remap = [], {}
        self._remap_arr = np.zeros(1, dtype=np.int32)


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
        self._token_counts = db.get_token_counts()
        self._source_map = db.get_source_map()
        self._eligible_cache: tuple[int, np.ndarray, int] | None = None

    def _get_eligible(self, max_doc: int) -> tuple[np.ndarray, int]:
        if self._eligible_cache is not None and self._eligible_cache[0] == max_doc:
            return self._eligible_cache[1], self._eligible_cache[2]
        eligible = np.where(self._token_counts <= max_doc)[0]
        mdl = (
            max(1, int(np.median(self._token_counts[eligible]))) if len(eligible) else 1
        )
        self._eligible_cache = (max_doc, eligible, mdl)
        return eligible, mdl

    def _mmr(
        self,
        candidates: list[dict],
        q_pos: np.ndarray,
        q_neg: np.ndarray,
        dp64: np.ndarray,
        dn64: np.ndarray,
        lit: np.ndarray,
        budget: int,
    ) -> list[dict]:
        """Incremental multiplicative MMR in ternary bitspace. O(n·k) memory."""
        n = len(candidates)
        if n <= 1:
            return candidates

        qp64 = q_pos.ravel().view(np.uint64)[None, :]
        qn64 = q_neg.ravel().view(np.uint64)[None, :]
        q_sims = score64(dp64, dn64, qp64, qn64)
        q_range = q_sims.max() - q_sims.min()
        q_norm = (q_sims - q_sims.min()) / (q_range + 1e-6)
        tcounts = np.array([c["token_count"] for c in candidates], dtype=np.int32)

        selected, total = [], 0
        mask = np.ones(n, dtype=bool)
        max_sim = np.zeros(n, dtype=np.float32)

        while mask.any():
            scores = q_norm * (1.0 - np.clip(max_sim, 0.0, None))
            scores[~mask] = -np.inf
            best = int(np.argmax(scores))
            if total + tcounts[best] > budget:
                mask[best] = False
                continue
            selected.append(best)
            total += int(tcounts[best])
            mask[best] = False
            if mask.any():
                raw_b = score64(dp64, dn64, dp64[best], dn64[best])
                np.maximum(
                    max_sim, raw_b / np.sqrt(lit * lit[best]).clip(1.0), out=max_sim
                )

        self.logger.info(
            f"[MMR] n={n} rel=[{q_sims.min():.3f},{q_sims.max():.3f}] "
            f"selected={len(selected)} tokens={total:,}"
        )
        out = [candidates[i] for i in selected]
        out.sort(key=lambda r: r["hdc_score"], reverse=True)
        return out

    def search(
        self,
        query: str,
        token_budget: int,
        track: bool = True,
        enabled_sources: set = None,
    ) -> list[dict]:
        """Retrieve memories relevant to *query* within *token_budget*."""
        if self.db.count() == 0:
            return []

        query = query.rstrip().rstrip("?!.,;:…\n\r\t").rstrip()
        query = strip_structural_keys(query)
        if not query:
            return []

        self.logger.info(f"[Search] Query: {len(query)} chars")

        tids = self._tokenizer.bulk_tokenize([query])
        #qe = self.hdc.project(token_ids=tids)
        qe = self.hdc.project_query(token_ids=tids)
        bm = self.hdc.encode(unigrams=qe, token_ids=tids)
        q_lit_uni = float(bm["lit_uni"][0])
        q_lit_ngram = float(bm.get("lit_ngram", np.zeros(1))[0])

        uni_mask = getattr(self.hdc, "uni_mask_u64", None)
        ngram_mask = getattr(self.hdc, "ngram_mask_u64", None)
        alpha = (
            self.hdc.adaptive_alpha(q_lit_uni, q_lit_ngram)
            if hasattr(self.hdc, "adaptive_alpha")
            else None
        )

        # 1. HDC progressive pruning
        max_doc = token_budget // self.config.min_context
        eligible, mdl = self._get_eligible(max_doc)
        target = max(1, int((token_budget * self.config.min_context) // mdl))
        indices, scores = self.db.search(
            bm["pos"],
            bm["neg"],
            target,
            eligible,
            self.logger,
            q_lit_uni=q_lit_uni,
            q_lit_ngram=q_lit_ngram,
            uni_mask=uni_mask,
            ngram_mask=ngram_mask,
            alpha=alpha,
        )
        if len(indices) == 0:
            return []

        # Filter positives
        pos_mask = scores > 0
        indices, scores = indices[pos_mask], scores[pos_mask]
        if len(indices) == 0:
            self.logger.info(f"[Search] {len(eligible):,}→0 (all ≤ 0)")
            return []

        # Build lightweight candidates
        candidates = [
            {
                "hdv_idx": idx,
                "hdc_score": sc,
                "token_count": int(self._token_counts[idx]),
                "source": self._source_map.get(idx, ""),
            }
            for idx, sc in zip(indices.tolist(), scores.tolist())
            if not enabled_sources or self._source_map.get(idx, "") in enabled_sources
        ]
        if not candidates:
            return []

        if len(candidates) > 1:
            # 2. Adaptive threshold
            net_scores = np.array([c["hdc_score"] for c in candidates], np.float32)
            thr = adaptive_threshold(net_scores)
            if thr and thr != 0.0:
                keep = net_scores >= thr
                if 0 < keep.sum() < len(candidates):
                    self.logger.info(
                        f"[Adaptive Threshold] {len(candidates)}→{int(keep.sum())} "
                        f"(thr={thr:.3f}, range=[{net_scores.min():.3f},{net_scores.max():.3f}])"
                    )
                    candidates = [c for c, k in zip(candidates, keep) if k]

            # 3. MMR diversity selection
            if len(candidates) > 1:
                bm_indices = [c["hdv_idx"] for c in candidates]
                dp64, dn64, lit = self.db.get_search_arrays(bm_indices)
                candidates = self._mmr(
                    candidates, bm["pos"], bm["neg"], dp64, dn64, lit, token_budget
                )

        # 4. Deferred text fetch
        final_idxs = [c["hdv_idx"] for c in candidates]
        mems = self.db.get_memories(final_idxs)

        return [
            {"memory": mems[c["hdv_idx"]], "hdc_score": c["hdc_score"]}
            for c in candidates
            if c["hdv_idx"] in mems
        ]


class HdRAG:
    """Hyperdimensional RAG memory engine."""

    def __init__(
        self,
        config: Config,
        tokenizer: Tokenizer,
        logger: logging.Logger = None,
    ):
        self.config = config
        self._tokenizer = tokenizer
        self.logger = logger or logging.getLogger(__name__)
        self._hdrag_dir = Path(config.hdrag_dir)
        self._hdrag_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self._hdrag_dir / "index.db", config, self.logger)
        self.hdc = HDCEncoder(config, self._hdrag_dir, self.db)
        self.retriever = Retriever(self.db, self.hdc, tokenizer, config, self.logger)
        self.enabled_sources: set[str] = set(self.db.source_counts())
        self._init_idf_weights(tokenizer)
        self.logger.info(f"HdRAG initialized: {self.db.count():,} memories")

    def _init_idf_weights(self, tokenizer: Tokenizer):
        if not self.hdc.idf:
            return
        if self.db.vocab_file.exists():
            self.hdc.compute_token_weights(tokenizer.vocab_size, tokenizer.special_ids)
        else:
            w = np.ones(tokenizer.vocab_size, dtype=np.float32)
            vocab_arr = np.array(
                [t for t in self.hdc.idf if 0 <= t < tokenizer.vocab_size], np.int64
            )
            w[vocab_arr] = np.array([self.hdc.idf[t] for t in vocab_arr], np.float32)
            special_arr = np.array(
                [t for t in tokenizer.special_ids if 0 <= t < tokenizer.vocab_size],
                np.int64,
            )
            if len(special_arr):
                w[special_arr] = 0.0
            _arrays.set("idf_weights", w)

    def search(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> list[dict]:
        results = self.retriever.search(
            query,
            token_budget or self.config.max_context_tokens,
            track,
            self.enabled_sources,
        )
        self.compact()
        return results

    def get_context(
        self, query: str, token_budget: int = None, track: bool = True
    ) -> str:
        return "\n\n---\n\n".join(
            r["memory"]["text"] for r in self.search(query, token_budget, track)
        )

    def source_counts(self) -> dict[str, int]:
        return self.db.source_counts()

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
            "arrays": _arrays.keys(),
            "array_mb": _arrays.nbytes() / 1e6,
            "db": self.db.stats(),
        }

    def compact(self, *child_procs):
        """Release encoding memory, trim working set."""
        self.hdc.compact()
        self.retriever._eligible_cache = None
        soft_trim()

    def clear_index(self, *child_procs):
        self.db.clear()
        self.hdc.clear()

    def build_index(self, progress_cb: Callable = None) -> int:
        files = discover_datasets(Path(self.config.datasets_dir))
        if not files:
            return 0
        docs = self._pass1_vocabulary(files, progress_cb)
        if not docs:
            return 0
        self._pass2_encode(docs, progress_cb)
        return len(docs)

    def _clear_for_rebuild(self):
        """Release all caches and delete index files before a full rebuild."""
        if hasattr(self.hdc, "_vocab_f16"):
            delattr(self.hdc, "_vocab_f16")
        self.hdc._vocab_t = None
        self.db.invalidate_corpus()
        trim_memory()
        for f in (
            self.db.corpus_file,
            self.db.vocab_file,
            self.db.lit_file,
            self.db.uni_lit_file,
            self.db.ngram_lit_file,
        ):
            if f.exists():
                f.unlink()
        self.db.clear()

    def _pass1_vocabulary(self, files, progress_cb) -> list[dict]:
        """Pass 1: tokenize all documents and build document-frequency table."""
        self.logger.info("Clearing existing index...")
        self._clear_for_rebuild()
        self.logger.info("Pass 1: Building vocabulary...")

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
                self.config.max_context_tokens // self.config.min_context,
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
            trim_memory()

        # Deduplicate
        seen = set()
        docs = [d for d in docs if d["id"] not in seen and not seen.add(d["id"])]
        if not docs:
            self.logger.info("No new documents")
            return []

        self.logger.info(f"Full rebuild: {len(docs):,} documents")
        if hasattr(self._tokenizer, "stop_server"):
            self._tokenizer.stop_server()

        # Persist IDF
        n_docs = len(docs)
        idf = {tid: math.log((n_docs + 1) / (df + 1)) for tid, df in vocab_df.items()}
        self.db.save_idf(dict(vocab_df), n_docs)
        self.db.set_config("hdc_seed", self.config.hdc_seed)
        self.db.set_config("hdc_ngram", self.config.hdc_ngram)
        self.db.set_config(
            "median_doc_length", statistics.median(d["token_count"] for d in docs)
        )

        # Prepare encoder
        self.hdc.idf = idf
        self.hdc.median_doc_length = self.db.get_config("median_doc_length")
        self.hdc._build_remap()
        self.logger.info(
            f"Corpus vocab: {len(self.hdc._corpus_vocab):,}/{self._tokenizer.vocab_size:,}"
        )
        self.hdc.build_vocab_index()
        self.hdc.compute_token_weights(
            self._tokenizer.vocab_size, self._tokenizer.special_ids
        )

        return docs

    def _pass2_encode(self, docs, progress_cb):
        """Pass 2: encode all documents into ternary bitmaps and write corpus."""
        all_tids = np.concatenate([d["token_ids"] for d in docs])
        tid_offsets = np.zeros(len(docs) + 1, dtype=np.int64)
        np.cumsum([len(d["token_ids"]) for d in docs], out=tid_offsets[1:])
        for d in docs:
            d.pop("token_ids", None)
        self.logger.info(
            f"Token IDs: {len(all_tids):,} tokens, {all_tids.nbytes / 1e6:.0f}MB"
        )

        trim_memory()
        self.hdc.load_vocab_full()
        self.logger.info(f"Pass 2: Encoding (ngram={self.config.hdc_ngram})...")

        bs = self.config.batch_size
        n_total = len(docs)
        nb = (n_total + bs - 1) // bs
        stride = self.db._stride

        tmp_pos = self.db.corpus_file.with_suffix(".pos.tmp")
        tmp_neg = self.db.corpus_file.with_suffix(".neg.tmp")
        self.db.set_config("corpus_layout", "blocked")

        all_lit = np.empty(n_total, dtype=np.float32)
        all_uni_lit = np.empty(n_total, dtype=np.float32)
        all_ngram_lit = np.empty(n_total, dtype=np.float32)

        with open(tmp_pos, "wb") as fp, open(tmp_neg, "wb") as fn:
            for i, start in enumerate(range(0, n_total, bs)):
                end = min(start + bs, n_total)
                batch_flat = all_tids[tid_offsets[start] : tid_offsets[end]]
                batch_offs = tid_offsets[start : end + 1] - tid_offsets[start]
                bitmaps = self.hdc.encode(flat_ids=batch_flat, offsets=batch_offs)
                self.db.insert_rows(
                    [
                        {k: d[k] for k in ("id", "text", "metadata", "token_count")}
                        for d in docs[start:end]
                    ],
                    bitmaps,
                    start,
                )
                fp.write(bitmaps["pos"].tobytes())
                fn.write(bitmaps["neg"].tobytes())

                bsz = end - start
                p64 = bitmaps["pos"].reshape(bsz, stride).view(np.uint64)
                n64 = bitmaps["neg"].reshape(bsz, stride).view(np.uint64)
                all_lit[start:end] = np.bitwise_count(p64 | n64).sum(
                    axis=1, dtype=np.float32
                )
                all_uni_lit[start:end] = bitmaps["lit_uni"].astype(np.float32)
                all_ngram_lit[start:end] = bitmaps.get(
                    "lit_ngram", np.zeros(bsz, np.int32)
                ).astype(np.float32)

                if (i + 1) % self.config.batch_log_interval == 0 or i + 1 == nb:
                    self.logger.info(
                        f"  Batch {i + 1:,}/{nb:,} ({100 * (i + 1) / nb:.0f}%)"
                    )
                    trim_memory()
                if progress_cb:
                    progress_cb(0.3 + (i + 1) / nb * 0.7, f"Pass 2: batch {i + 1}")

        # Concatenate pos + neg into final blocked corpus file
        with open(self.db.corpus_file, "wb") as out:
            for tmp in (tmp_pos, tmp_neg):
                with open(tmp, "rb") as src:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                tmp.unlink()

        all_lit.tofile(self.db.lit_file)
        all_uni_lit.tofile(self.db.uni_lit_file)
        all_ngram_lit.tofile(self.db.ngram_lit_file)
        del all_lit, all_uni_lit, all_ngram_lit, all_tids, tid_offsets

        self.db.finalize_index()
        self.retriever._token_counts = self.db.get_token_counts()
        self.retriever._source_map = self.db.get_source_map()
        self.retriever._eligible_cache = None
        self.hdc.release_vocab_full()
        self.hdc.release_workspace()
        self.enabled_sources = set(self.db.source_counts())
        self.db.release_mmap_pages()
        self.hdc.deep_compact()
        trim_memory()
        self.logger.info(f"Index complete: {self.db.count():,} memories")

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

        stripped = [strip_structural_keys(txt) for _, txt, _ in new]
        all_toks = self._tokenizer.bulk_tokenize(stripped)

        for (mid, text, meta), toks in zip(new, all_toks):
            if not toks:
                continue
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
