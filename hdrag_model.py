"""
hdrag_model - Shared foundation for HdRAG.

Config, GGUF utilities, llama.cpp server management,
inference engine (tokenizer + generation via server HTTP API),
and conversation logging.

Tokenization is delegated entirely to llama-server's /tokenize endpoint.
The GGUF file is the single source of truth for vocabulary, merge rules,
and pre-tokenizer patterns. No Python reimplementation of BPE.

Server has two mutually exclusive modes:
  tokenize  — CPU only (-ngl 0, -c 8), boots in ~1s, for indexing
  inference — full GPU (-ngl N, -c K), boots in ~6s, for chat

They never coexist. Transitions are explicit: indexing kills inference
server, starts tokenize server, tokenizes, kills it. Next chat message
starts inference server.

This module has zero dependency on the HDC memory engine (hdrag.py)
or the Gradio UI (hdrag_gradio.py).
"""

from __future__ import annotations

import json, logging, os, re, signal, subprocess, time, threading
from dataclasses import dataclass, asdict, fields
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Protocol, runtime_checkable

import numpy as np
import yaml
import requests
from gguf import GGUFReader


# ── Tokenizer Protocol ────────────────────────────────────────────


@runtime_checkable
class Tokenizer(Protocol):
    """Structural interface consumed by the HDC memory engine."""

    def tokenize(self, text: str) -> list[int]: ...
    def bulk_tokenize(self, texts: list[str]) -> list[list[int]]: ...
    def count_tokens(self, text: str) -> int: ...
    def detokenize(self, tokens: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def special_ids(self) -> set[int]: ...


# ── OS Utilities ──────────────────────────────────────────────────


def trim_working_set(*child_procs: subprocess.Popen):
    """Release unused physical pages back to the OS (Windows only)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetProcessWorkingSetSizeEx(
            k32.GetCurrentProcess(),
            ctypes.c_size_t(-1),
            ctypes.c_size_t(-1),
            0,
        )
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_QUERY_INFORMATION = 0x0400
        for proc in child_procs:
            if proc is None or proc.poll() is not None:
                continue
            h = k32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION,
                False,
                proc.pid,
            )
            if h:
                k32.SetProcessWorkingSetSizeEx(
                    h,
                    ctypes.c_size_t(-1),
                    ctypes.c_size_t(-1),
                    0,
                )
                k32.CloseHandle(h)
    except Exception:
        pass


# ── GGUF Helpers ──────────────────────────────────────────────────


def gguf_bytes(field) -> Generator[bytes, None, None]:
    for idx in field.data:
        part = field.parts[idx]
        if part.dtype == np.uint8 and part.size > 0:
            yield bytes(part)


def gguf_int(reader, key: str, default=None):
    field = reader.fields.get(key)
    if not field or not field.data:
        return default
    part = field.parts[field.data[0]]
    if part.size > 0 and part.dtype.kind in ("i", "u"):
        return int(part[0])
    return default


def gguf_field(reader, key: str, default=None):
    field = reader.fields.get(key)
    if not field or not field.data:
        return default
    part = field.parts[field.data[0]]
    if part.size == 0:
        return default
    if part.dtype.kind == "b":
        return bool(part[0])
    if part.dtype == np.uint8:
        return bytes(part).decode("utf-8", errors="replace")
    if part.dtype.kind in ("i", "u"):
        return int(part[0])
    if part.dtype.kind == "f":
        return float(part[0])
    return default


def resolve_gguf(name: str, model_dir: str) -> str:
    if not name:
        return ""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p)
    direct = Path(model_dir) / name
    if direct.exists():
        return str(direct)
    hits = list(Path(model_dir).rglob(name))
    if len(hits) == 1:
        return str(hits[0])
    return ""


# ── Config ─────────────────────────────────────────────────────────


@dataclass
class Config:
    chat_history_dir: str
    hdrag_dir: str
    datasets_dir: str
    model_dir: str
    gguf_model: str
    llama_cpp_dir: str
    llama_server_url: str
    llama_gpu_layers: int
    temperature: float
    top_p: float
    max_new_tokens: int
    hdc_dimensions: int
    hdc_seed: int
    hdc_ngram: int
    batch_size: int
    export_log_interval: int
    batch_log_interval: int
    max_context_tokens: int
    min_context: int
    sqlite_max_vars: int
    sqlite_cache_kb: int
    gradio_port: int
    system_prompt: str
    llama_context_size: int = 8192
    llama_batch_size: int = 2048
    llama_ubatch_size: int = 512
    llama_cache_type_k: str = "q8_0"
    llama_cache_type_v: str = "q8_0"
    llama_parallel: int = 1
    llama_cache_reuse: int = 8192  # deprecated, ignored
    stop_sequences: list = None

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> Config:
        if not Path(path).exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        required = {
            f.name
            for f in fields(cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        missing = required - set(filtered)
        if missing:
            raise ValueError(f"Missing config fields: {missing}")
        return cls(**filtered)


# ── LLama Server ───────────────────────────────────────────────────

# Server modes
TOKENIZE = "tokenize"  # CPU only, vocab lookup, -ngl 0, minimal context
INFERENCE = "inference"  # Full GPU, generation, full context + KV cache


class LlamaServer:
    """Manages a single llama-server process in one of two modes.

    tokenize  — CPU only (-ngl 0, -c 8). Loads vocab + merge table only.
                No GPU memory, no KV cache. Boots in ~1 second.
                Used during indexing for /tokenize endpoint.

    inference — Full GPU (-ngl N, -c K, KV cache, --jinja).
                Full model on VRAM. Boots in ~6 seconds.
                Used during chat for /v1/chat/completions.

    Modes are mutually exclusive. Switching modes stops the current
    process and starts a new one. The /tokenize endpoint works in
    both modes, so chat-time token counting doesn't trigger a switch.
    """

    def __init__(
        self, config: Config, logger: logging.Logger, model_pathname: str = ""
    ):
        self.config = config
        self.logger = logger
        self.model_pathname = model_pathname
        self._proc: Optional[subprocess.Popen] = None
        self._mode: Optional[str] = None
        self._lock = threading.Lock()
        self._log_file = None

        # Log directory for server output
        self._log_dir = Path(config.hdrag_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._proc

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    def stop(self):
        """Terminate the server and free all its resources."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        """Inner stop — caller must hold self._lock."""
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            self._mode = None
            self._close_log()
            return
        self.logger.info(f"Stopping server (mode={self._mode})")
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None
        self._mode = None
        self._close_log()

    def _close_log(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def start(self, mode: str = INFERENCE):
        """Start in requested mode. Idempotent, thread-safe.

        If already running in the requested mode → no-op.
        If running in wrong mode → stop, then start in new mode.
        If not running → start fresh.
        """
        with self._lock:
            if not self.config.llama_cpp_dir or not self.model_pathname:
                return
            # Already running in correct mode
            if self._proc and self._proc.poll() is None and self._mode == mode:
                return
            # Wrong mode or dead process — stop first
            if self._proc and self._proc.poll() is None:
                self._stop_locked()
            self._start_locked(mode)

    def _start_locked(self, mode: str):
        """Build command and start process. Caller must hold self._lock."""
        from urllib.parse import urlparse

        ext = ".exe" if os.name == "nt" else ""
        server_bin = Path(self.config.llama_cpp_dir) / f"llama-server{ext}"
        parsed = urlparse(self.config.llama_server_url)
        cfg = self.config

        cmd = [
            str(server_bin),
            "-m",
            self.model_pathname,
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 8080),
        ]

        if mode == TOKENIZE:
            # CPU only, vocab lookup only. --no-warmup prevents paging
            # in model weights via mmap (only vocab pages get faulted).
            cmd += ["-ngl", "0", "-c", "8", "--no-warmup"]
        else:
            # Full inference — GPU offload, large context, KV cache
            cmd += [
                "-ngl",
                str(cfg.llama_gpu_layers),
                "-c",
                str(cfg.llama_context_size),
                "-b",
                str(cfg.llama_batch_size),
                "-ub",
                str(cfg.llama_ubatch_size),
                "--cache-type-k",
                cfg.llama_cache_type_k,
                "--cache-type-v",
                cfg.llama_cache_type_v,
                "-np",
                str(cfg.llama_parallel),
                "--jinja",
            ]

        self.logger.info(f"Starting ({mode}): {' '.join(cmd)}")

        # Server output → log file for crash diagnostics
        self._close_log()
        log_path = self._log_dir / "llama_server.log"
        self._log_file = open(log_path, "w")

        kwargs = {
            "stdout": self._log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name != "nt":
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            kwargs["preexec_fn"] = lambda: libc.prctl(1, signal.SIGTERM)

        self._proc = subprocess.Popen(cmd, **kwargs)
        self._mode = mode
        if os.name == "nt":
            self._bind_job()
        self._wait_ready()

    def _bind_job(self):
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("_" + str(i), t)
                for i, t in enumerate(
                    [
                        ctypes.c_int64,
                        ctypes.c_int64,
                        wintypes.DWORD,
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        wintypes.DWORD,
                        ctypes.c_size_t,
                        wintypes.DWORD,
                        wintypes.DWORD,
                    ]
                )
            ]

        class IO(ctypes.Structure):
            _fields_ = [(f"_{i}", ctypes.c_uint64) for i in range(6)]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO)] + [
                (f"_{i}", ctypes.c_size_t) for i in range(4)
            ]

        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.Basic._2 = 0x2000
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        k32.AssignProcessToJobObject(job, int(self._proc._handle))

    def _wait_ready(self, timeout=120):
        url = f"{self.config.llama_server_url.rstrip('/')}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp exited with code {self._proc.returncode}"
                )
            try:
                r = requests.get(url, timeout=2)
                if r.ok and r.json().get("status") == "ok":
                    self.logger.info(f"llama.cpp ready ({self._mode})")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama.cpp not ready after {timeout}s")


# ── Inference Engine ──────────────────────────────────────────────


class InferenceEngine:
    """Tokenizer + generation via llama.cpp server.

    Tokenization is delegated to the server's /tokenize endpoint.
    The /tokenize endpoint works in both server modes (tokenize and
    inference), so chat-time token counting doesn't trigger a mode
    switch. Only generate() requires inference mode.

    Static vocabulary metadata (vocab_size, special token IDs, context
    length) is read once from the GGUF at construction time.

    Satisfies the Tokenizer protocol so it can be passed directly
    to HdRAG.
    """

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        server: LlamaServer,
        gguf_path: str = "",
    ):
        self.config = config
        self.logger = logger
        self._server = server
        self.model_pathname = gguf_path
        self._session = requests.Session()
        self._base_url = config.llama_server_url.rstrip("/")
        self._gguf_system_prompt = ""

        # Static metadata from GGUF (immutable, read once)
        self._vocab_size = 0
        self._special_ids: set[int] = set()
        self._context_length = config.llama_context_size

        if gguf_path:
            self._read_gguf_metadata()

    def _read_gguf_metadata(self):
        """Read all static metadata from the GGUF in one pass.

        Single file open for vocab size, special IDs, context length,
        and chat template. These are immutable after the file is written.
        """
        reader = GGUFReader(str(self.model_pathname))

        # Vocab size
        tokens_field = reader.fields.get("tokenizer.ggml.tokens")
        if tokens_field:
            self._vocab_size = sum(1 for _ in gguf_bytes(tokens_field))

        # Special token IDs
        bos = gguf_int(reader, "tokenizer.ggml.bos_token_id")
        eos = gguf_int(reader, "tokenizer.ggml.eos_token_id")
        pad = gguf_int(reader, "tokenizer.ggml.padding_token_id")
        self._special_ids = {x for x in (bos, eos, pad) if x is not None}

        # Context length from model metadata
        arch = gguf_field(reader, "general.architecture", "unknown")
        ctx = gguf_field(reader, f"{arch}.context_length", None)
        if ctx is not None:
            self._context_length = min(ctx, self.config.llama_context_size)

        # Tokenizer identity for diagnostics
        tok_model = gguf_field(reader, "tokenizer.ggml.model", "unknown")
        tok_pre = gguf_field(reader, "tokenizer.ggml.pre", "default")

        self.logger.info(
            f"GGUF metadata: vocab={self._vocab_size:,} "
            f"ctx={self._context_length:,} "
            f"model={tok_model!r} pre={tok_pre!r} "
            f"special={self._special_ids}"
        )

        # Chat template → system prompt
        tmpl = gguf_field(reader, "tokenizer.chat_template", "")
        if tmpl:
            self.logger.info(f"GGUF chat template: {len(tmpl)} chars")
            for pat in (
                r'set\s+(?:default_)?system_message\s*=\s*["\'](.+?)["\']',
                r'system_message\s*=\s*["\'](.+?)["\']',
            ):
                m = re.search(pat, tmpl, re.DOTALL)
                if m:
                    self._gguf_system_prompt = m.group(1).replace("\\n", "\n").strip()
                    self.logger.info(
                        f"GGUF system prompt: {len(self._gguf_system_prompt)} chars"
                    )
                    break

    # ── Server lifecycle ──

    def start_for_indexing(self):
        """Start server in tokenize-only mode (CPU, no GPU memory)."""
        self._server.start(TOKENIZE)

    def stop_server(self):
        """Stop the server entirely. Next call auto-restarts as needed."""
        self._server.stop()

    # ── Tokenizer interface (satisfies Tokenizer protocol) ──

    def _ensure_server(self):
        """Ensure some server is running (any mode).

        /tokenize works in both tokenize and inference modes.
        If already running → no-op.
        If dead → restart in last mode (or inference as default).
        """
        proc = self._server.process
        if proc and proc.poll() is None:
            return  # alive, don't care which mode
        # Dead or never started — restart in last mode or default
        last = self._server.mode or INFERENCE
        if proc and proc.poll() is not None:
            rc = proc.returncode
            self.logger.warning(
                f"Server died (exit={rc}), restarting in {last} mode. "
                f"Check llama_server.log for details."
            )
        self._server.start(last)

    def tokenize(self, text: str) -> list[int]:
        limit = self.config.max_context_tokens
        if len(text) <= limit:
            return self._tokenize_one(text)
        return self._tokenize_long(text, limit)

    def _tokenize_long(self, text: str, limit: int) -> list[int]:
        """Split text into regex-safe pieces, tokenize, concatenate.

        The GPT-4o pre-tokenizer regex has nested lookahead quantifiers
        that cause catastrophic backtracking on long runs of similar
        characters (DNA sequences, base64, repeated patterns). Splitting
        on newlines first keeps each piece naturally short. Lines still
        over the limit get split on whitespace, then hard-cut as last
        resort to keep any single regex invocation under _HARD_LIMIT.
        """
        pieces = []
        buf = []
        buf_len = 0
        hard = min(limit, 2000)  # regex-safe ceiling per piece

        for line in text.split("\n"):
            added = len(line) + (1 if buf else 0)
            if buf and buf_len + added > limit:
                pieces.append("\n".join(buf))
                buf = []
                buf_len = 0

            if len(line) > hard:
                # Flush buffer first
                if buf:
                    pieces.append("\n".join(buf))
                    buf = []
                    buf_len = 0
                # Split long line on whitespace, then hard-cut
                pos = 0
                while pos < len(line):
                    end = pos + hard
                    if end < len(line):
                        split = line.rfind(" ", pos, end)
                        if split <= pos:
                            split = end  # no whitespace — hard cut
                        end = split
                    pieces.append(line[pos:end])
                    pos = end
            else:
                buf.append(line)
                buf_len += added

        if buf:
            pieces.append("\n".join(buf))

        tokens = []
        for piece in pieces:
            tokens.extend(self._tokenize_one(piece))
        return tokens

    def _tokenize_one(self, text: str) -> list[int]:
        """Tokenize a single chunk via server /tokenize endpoint."""
        self._ensure_server()
        for attempt in range(3):
            try:
                r = self._session.post(
                    f"{self._base_url}/tokenize",
                    json={"content": text, "add_special": False},
                    timeout=30,
                )
                if r.status_code == 500:
                    # Server alive but can't tokenize this input
                    self.logger.warning(
                        f"Tokenize 500: {repr(text[:120])}... "
                        f"(len={len(text)}) — skipping"
                    )
                    return []
                r.raise_for_status()
                return r.json()["tokens"]
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                self._ensure_server()
                time.sleep(0.5 + attempt)

    def bulk_tokenize(self, texts: list[str]) -> list[list[int]]:
        return [self.tokenize(t) for t in texts]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def detokenize(self, tokens: list[int]) -> str:
        self._ensure_server()
        r = self._session.post(
            f"{self._base_url}/detokenize",
            json={"tokens": tokens},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["content"]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_ids(self) -> set[int]:
        return self._special_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    # ── System prompt ──

    def build_system_prompt(self, memory: str = "") -> str:
        parts = []
        if self._gguf_system_prompt:
            parts.append(self._gguf_system_prompt)
        parts.append(self.config.system_prompt)
        base = "\n\n".join(parts)
        if memory:
            return f"{base}\n\n<working_memory>\n{memory}\n</working_memory>"
        return base

    # ── Generation ──

    def generate(self, messages: list[dict]) -> str:
        self._server.start(INFERENCE)  # ensure GPU mode
        payload = {
            "model": Path(self.model_pathname).stem,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences
        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        if not resp.ok:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                pass
            self.logger.error(f"[Generate] {resp.status_code} from llama.cpp: {body}")
            if resp.status_code == 400 and "exceed" in body.lower():
                return (
                    "\u26a0\ufe0f Context overflow \u2014 the prompt exceeded "
                    "the model's context window. Try a shorter message, clear "
                    "the conversation, or reduce the token budget."
                )
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Conversation Logger ────────────────────────────────────────────


class ConversationLogger:
    def __init__(self, history_dir: str):
        self.dir = Path(history_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file: Optional[Path] = None

    def log(self, query: str, response: str):
        if not self.file:
            self.file = self.dir / f"conversation_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        with open(self.file, "a") as f:
            f.write(json.dumps({"from": "human", "value": query}) + "\n")
            f.write(json.dumps({"from": "gpt", "value": response}) + "\n")

    def new_conversation(self):
        self.file = None
