"""
hdrag_model - Shared foundation for HdRAG.

Config, GGUF utilities, llama.cpp server management,
inference engine (generation via server HTTP API),
HuggingFace tokenizer (loaded from GGUF), and conversation logging.

Tokenization is handled in-process by HuggingFaceTokenizer, which
extracts the tokenizer directly from the GGUF file via transformers.
The llama.cpp server is used only for text generation.

This module has zero dependency on the HDC memory engine (hdrag.py)
or the Gradio UI (hdrag_gradio.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
import threading
from dataclasses import dataclass, asdict, fields
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Protocol, runtime_checkable

import numpy as np
import yaml
import requests
from gguf import GGUFReader
from transformers import AutoTokenizer


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


def close_gguf_reader(reader: GGUFReader):
    """Close a GGUFReader and release its mmap immediately."""
    try:
        if hasattr(reader, "tensors"):
            reader.tensors.clear()
        if hasattr(reader, "fields"):
            reader.fields.clear()
        if hasattr(reader, "data") and reader.data is not None:
            mm = getattr(reader.data, "_mmap", None)
            if mm is not None:
                mm.close()
            reader.data = None
    except Exception:
        pass


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


@dataclass
class Config:
    chat_history_dir: str
    hdrag_dir: str
    datasets_dir: str
    model_dir: str
    gguf_model: str  # inference model (LLM)
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
    llama_context_size: int = 0  # 0 = auto (--fit decides)
    llama_batch_size: int = 2048
    llama_ubatch_size: int = 512
    llama_cache_type_k: str = "q8_0"
    llama_cache_type_v: str = "q8_0"
    llama_parallel: int = 1
    nthreads: int = 0  # 0 = auto (os.cpu_count)
    llama_cache_reuse: int = 8192  # deprecated, ignored
    llama_no_mmap: bool = True
    llama_flash_attn: bool = True
    llama_sleep_idle: int = 0  # 0 = disabled, N = unload after N seconds idle
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


def _collect_special_ids(tokenizer) -> set[int]:
    """Collect ALL special token IDs from an HF tokenizer."""
    ids: set[int] = set()
    if hasattr(tokenizer, "added_tokens_encoder"):
        ids.update(tokenizer.added_tokens_encoder.values())
    if hasattr(tokenizer, "all_special_ids"):
        ids.update(tokenizer.all_special_ids)
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        ids.update(tokenizer.additional_special_tokens_ids)
    ids.discard(None)
    ids.discard(-1)
    return ids


class HuggingFaceTokenizer:
    """In-process tokenizer via HuggingFace transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        logger: Optional[logging.Logger] = None,
        trust_remote_code: bool = False,
    ):
        """Load tokenizer from HuggingFace model name or local directory."""
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Loading HF tokenizer: {model_name_or_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self._special_ids = _collect_special_ids(self._tokenizer)

        self.logger.info(
            f"HF tokenizer ready: vocab={self.vocab_size:,} "
            f"special={len(self._special_ids)} "
            f"type={type(self._tokenizer).__name__}"
        )

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> "HuggingFaceTokenizer":
        """Load tokenizer directly from a GGUF file."""
        logger = logger or logging.getLogger(__name__)
        logger.info(f"Extracting tokenizer from GGUF: {gguf_path}")

        from transformers.integrations.ggml import convert_gguf_tokenizer

        tokenizer_dict, tokenizer_config, architecture = (
            cls._extract_tokenizer_from_gguf(gguf_path, logger)
        )

        logger.info(
            f"GGUF tokenizer type: {tokenizer_dict.get('tokenizer_type', '?')!r}, "
            f"converter: {architecture!r}, "
            f"vocab tokens: {len(tokenizer_dict.get('tokens', []))}"
        )

        fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(
            architecture, tokenizer_dict
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            fast_tokenizer.save(str(Path(tmp_dir) / "tokenizer.json"))

            config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_type": architecture,
            }
            config.update(additional_kwargs)
            config.update(tokenizer_config)

            with open(Path(tmp_dir) / "tokenizer_config.json", "w") as f:
                json.dump(config, f)

            instance = cls.__new__(cls)
            instance.logger = logger
            instance._tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            instance._special_ids = _collect_special_ids(instance._tokenizer)

        logger.info(
            f"GGUF tokenizer ready: vocab={instance.vocab_size:,} "
            f"special={len(instance._special_ids)} "
            f"type={type(instance._tokenizer).__name__}"
        )

        return instance

    @staticmethod
    def _extract_tokenizer_from_gguf(
        gguf_path: str,
        logger: logging.Logger,
    ) -> tuple[dict, dict, str]:
        """Extract tokenizer dict directly from GGUF metadata."""
        from transformers.modeling_gguf_pytorch_utils import (
            GGUF_TOKENIZER_MAPPING,
            _gguf_parse_value,
        )
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        reader = GGUFReader(str(gguf_path))

        # Parse all tokenizer.* fields from GGUF
        tokenizer_dict: dict = {}
        tokenizer_config: dict = {}
        for gguf_key, field in reader.fields.items():
            if not gguf_key.startswith("tokenizer."):
                continue
            config_key = gguf_key.split(".", 1)[1]  # strip "tokenizer."
            value = [_gguf_parse_value(field.parts[i], field.types) for i in field.data]
            if len(value) == 1:
                value = value[0]

            # Map to tokenizer_dict keys
            for category, renames in GGUF_TOKENIZER_MAPPING.items():
                if config_key in renames:
                    renamed = renames[config_key]
                    if category == "tokenizer":
                        tokenizer_dict[renamed] = value
                    elif category == "tokenizer_config":
                        tokenizer_config[renamed] = value

        # Determine the right converter from tokenizer type + model arch
        arch = gguf_field(reader, "general.architecture", "unknown")
        tok_type = tokenizer_dict.get("tokenizer_type", "")

        # Try model architecture first (exact match or known alias)
        converter = None
        for candidate in [arch, arch.replace("-", "_"), arch.replace("-", "")]:
            if candidate in GGUF_TO_FAST_CONVERTERS:
                converter = candidate
                break

        # Fall back to tokenizer type (gpt2 → gpt2, llama → llama)
        if converter is None and tok_type in GGUF_TO_FAST_CONVERTERS:
            converter = tok_type
            logger.info(
                f"Unknown architecture {arch!r}, "
                f"using {tok_type!r} converter based on tokenizer type"
            )

        if converter is None:
            close_gguf_reader(reader)
            raise ValueError(
                f"Cannot determine tokenizer converter for GGUF with "
                f"architecture={arch!r}, tokenizer_type={tok_type!r}. "
                f"Supported: {sorted(GGUF_TO_FAST_CONVERTERS.keys())}"
            )

        close_gguf_reader(reader)
        del reader
        return tokenizer_dict, tokenizer_config, converter

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def bulk_tokenize(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self._tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def detokenize(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def special_ids(self) -> set[int]:
        return self._special_ids

    def start_for_indexing(self):
        """No-op — no server to start."""

    def stop_server(self):
        """No-op — no server to stop."""


class LlamaServer:
    """Manages a single llama-server process for inference."""

    def __init__(
        self, config: Config, logger: logging.Logger, model_pathname: str = ""
    ):
        self.config = config
        self.logger = logger
        self.model_pathname = model_pathname
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log_file = None
        self._n_ctx: int = 0  # actual context size, set after server ready

        self._log_dir = Path(config.hdrag_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._proc

    @property
    def context_length(self) -> int:
        """Actual context size the server is running with."""
        return self._n_ctx

    def stop(self):
        """Terminate the server and free all its resources."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            self._close_log()
            return
        self.logger.info("Stopping llama-server")
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None
        self._close_log()

    def _close_log(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def start(self):
        """Start the server if not already running. Idempotent, thread-safe."""
        with self._lock:
            if not self.config.llama_cpp_dir or not self.model_pathname:
                return
            if self._proc and self._proc.poll() is None:
                return
            self._start_locked()

    def _start_locked(self):
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
            "-t",
            str(cfg.nthreads or os.cpu_count() or 4),
            "-ngl",
            str(cfg.llama_gpu_layers),
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

        if cfg.llama_context_size > 0:
            cmd.extend(["-c", str(cfg.llama_context_size)])

        if cfg.llama_no_mmap:
            cmd.append("--no-mmap")
        if cfg.llama_flash_attn:
            cmd.extend(["--flash-attn", "on"])
        if cfg.llama_sleep_idle > 0:
            cmd.extend(["--sleep-idle-seconds", str(cfg.llama_sleep_idle)])

        self.logger.info(f"Starting llama-server: {' '.join(cmd)}")

        self._close_log()
        log_path = self._log_dir / "llama_server.log"
        self._log_file = open(log_path, "w")

        kwargs = {"stdout": self._log_file, "stderr": subprocess.STDOUT}
        if os.name != "nt":
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            kwargs["preexec_fn"] = lambda: libc.prctl(1, signal.SIGTERM)

        self._proc = subprocess.Popen(cmd, **kwargs)
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
                    self._query_props()
                    self.logger.info(f"llama.cpp ready (n_ctx={self._n_ctx:,})")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama.cpp not ready after {timeout}s")

    def _query_props(self):
        """Query the server for its actual runtime configuration."""
        base = self.config.llama_server_url.rstrip("/")
        try:
            r = requests.get(f"{base}/props", timeout=5)
            if r.ok:
                props = r.json()
                gen = props.get("default_generation_settings", {})
                n_ctx = gen.get("n_ctx", 0)
                if n_ctx > 0:
                    self._n_ctx = n_ctx
                    return
        except Exception as e:
            self.logger.debug(f"Failed to query /props: {e}")
        # Fallback: use config value or a safe default
        self._n_ctx = self.config.llama_context_size or 4096

    def clear_kv_cache(self):
        """Flush KV cache from all server slots to reclaim VRAM."""
        if self._proc is None or self._proc.poll() is not None:
            return
        base = self.config.llama_server_url.rstrip("/")
        try:
            # Get slot info, erase each active slot's cache
            r = requests.get(f"{base}/slots", timeout=5)
            if r.ok:
                for slot in r.json():
                    slot_id = slot.get("id", 0)
                    requests.post(
                        f"{base}/slots/{slot_id}?action=erase",
                        timeout=5,
                    )
        except (requests.ConnectionError, requests.Timeout, Exception) as e:
            self.logger.debug(f"KV cache clear failed: {e}")


class InferenceEngine:
    """Generation + token counting via llama.cpp server."""

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        server: LlamaServer,
        gguf_path: str = "",
        tokenizer: Tokenizer = None,
    ):
        self.config = config
        self.logger = logger
        self._server = server
        self.model_pathname = gguf_path
        self._session = requests.Session()
        self._base_url = config.llama_server_url.rstrip("/")
        self._gguf_system_prompt = ""
        self._tokenizer = tokenizer

        if gguf_path:
            self._read_gguf_metadata()

    def _read_gguf_metadata(self):
        """Read chat template from the GGUF."""
        reader = GGUFReader(str(self.model_pathname))

        arch = gguf_field(reader, "general.architecture", "unknown")
        self.logger.info(f"GGUF metadata: arch={arch!r}")

        tmpl = gguf_field(reader, "tokenizer.chat_template", "")
        close_gguf_reader(reader)
        del reader

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

    def _ensure_server(self):
        proc = self._server.process
        if proc and proc.poll() is None:
            return
        if proc and proc.poll() is not None:
            self.logger.warning(
                f"Server died (exit={proc.returncode}), restarting. "
                f"Check llama_server.log for details."
            )
        self._server.start()  # re-queries /props, refreshes context_length

    def stop_server(self):
        self._server.stop()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text) // 4
        return self._tokenizer.count_tokens(text)

    @property
    def context_length(self) -> int:
        """Actual context size from the running server."""
        return self._server.context_length

    def build_system_prompt(self, memory: str = "") -> str:
        parts = []
        if self._gguf_system_prompt:
            parts.append(self._gguf_system_prompt)
        parts.append(self.config.system_prompt)
        base = "\n\n".join(parts)
        if memory:
            return f"{base}\n\n<working_memory>\n{memory}\n</working_memory>"
        return base

    def generate(self, messages: list[dict]) -> str:
        """Blocking generation — returns full response."""
        chunks = []
        for chunk in self.generate_stream(messages):
            chunks.append(chunk)
        return "".join(chunks)

    def generate_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Streaming generation — yields token chunks as they arrive."""
        self._server.start()
        payload = {
            "model": Path(self.model_pathname).stem,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
        }
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences
        try:
            resp = self._session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
                stream=True,
            )
        except requests.ConnectionError:
            self._ensure_server()
            resp = self._session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
                stream=True,
            )
        if not resp.ok:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                pass
            self.logger.error(f"[Generate] {resp.status_code} from llama.cpp: {body}")
            if resp.status_code == 400 and "exceed" in body.lower():
                yield (
                    "\u26a0\ufe0f Context overflow \u2014 the prompt exceeded "
                    "the model's context window. Try a shorter message, clear "
                    "the conversation, or reduce the token budget."
                )
                return
            resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


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
