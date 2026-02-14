# HdRAG — Hyperdimensional Retrieval-Augmented Generation

A latent space priming engine that uses ternary hyperdimensional computing for semantic memory retrieval. Unlike conventional RAG systems that retrieve information for the model to reproduce, HdRAG retrieves context that *bends the activation geometry* of the language model's forward pass, seeding generation with semantically relevant priors from an indexed corpus.

CPU-only retrieval. No vector database. No embedding model at inference time. Packed bitwise arithmetic on ternary hyperdimensional vectors — the entire search path is `popcount` on `uint64` arrays.

## Architecture

```
  ┌──────────────┐    ┌─────────────────────────────────────────────┐
  │  GGUF Model  │    │                hdrag.py                     │
  │  (llama.cpp) │    │                                             │
  │              │    │  EmbeddingExtractor ──► HDCEncoder          │
  │  token_embd  │───►│    (index time)    random projection        │
  │  .weight     │    │                    ternary quantize         │
  │              │    │                    n-gram binding           │
  │              │    │                                             │
  │  /tokenize   │◄──►│  Database          Retriever                │
  │  /v1/chat    │    │    SQLite +           blend ──► prune       │
  └──────┬───────┘    │    memmap HDVs        ──► threshold         │
         │            │                       ──► dedup             │
         │            │  HdRAG (orchestrator)  ──► _hdc_mmr         │
         │            └─────────────────────────────────────────────┘
         │
  ┌──────┴───────┐    ┌─────────────────────────────────────────────┐
  │ hdrag_model  │    │           hdrag_gradio.py                   │
  │              │    │                                             │
  │  Config      │    │  Chat tab ──► Search tab ──► Config tab     │
  │  LlamaServer │    │  Stats tab (sparsity, similarity, dims)     │
  │  Inference   │    │                                             │
  │  Engine      │    │  Checkboxes: Use Memory │ Top Document │    │
  │  Conv Logger │    │               Compress Context              │
  └──────────────┘    └─────────────────────────────────────────────┘
```

## How It Works

### Encoding Pipeline

**Indexing** (one-time, requires GPU for embedding extraction):

1. Extract `token_embd.weight` from the GGUF file (supports F32, F16, BF16, Q8_0, Q4_0, Q6_K)
2. Random Gaussian projection: `(vocab_size, emb_dim) → (vocab_size, hdc_dim)` with column normalization
3. Ternary quantization at threshold `τ = emb_dim^(-0.5)` — dimensions within one expected magnitude of zero become the zero (abstain) state
4. Pack into dual bitmaps: `pos` and `neg` arrays, `uint8`, stride-aligned to 8 bytes
5. Pre-compute n-gram permutation powers for each position in the binding chain
6. Store vocabulary index as a single memmap file: unigram float16 vectors + pre-permuted ternary bitmaps for each n-gram position

**Encoding a document or query:**

1. Tokenize via llama.cpp `/tokenize` endpoint
2. IDF-weighted `embedding_bag` over projected vocabulary vectors → continuous unigram vector
3. L2-normalize, scale by `√(hdc_dim / emb_dim)`
4. N-gram binding chain: element-wise ternary multiply on pre-permuted vocab bitmaps, filtered by coherence (`√popcount(pos | neg)`), capacity-bounded by `m_cap = ⌈log(1-target)/log(1-p)⌉`
5. Boosted encoding: `unigram * (1 + agree)` where `agree` is n-gram polarity agreement, then re-quantize at `τ`
6. Pack to dual bitmaps

The n-gram channel cannot overwhelm the unigram — the multiplicative boost is gated by IDF-weighted unigram magnitude. Dimensions IDF suppressed to near-zero stay near-zero even with full n-gram agreement.

### Retrieval Pipeline

1. **Encode** query → ternary bitvector
2. **Blend** with exponentially-decayed model response history (response-only, queries not tracked)
3. **Prune** via progressive tournament on packed uint64 columns with early termination on stable top-k
4. **Threshold** using adaptive L-statistic: `median + L2 * log(n)`
5. **Dedup** via word-set subset containment (fast, catches exact containment)
6. **HDC-MMR** — coverage-aware greedy selection: `marginal = score × (novelty / null_novelty)` where `null_novelty = 1 - coverage_density`. No λ parameter. Coverage density IS the adaptive tradeoff

### Latent Priming (not Information Retrieval)

Retrieved content is injected into the system prompt as "working memory" — vague recollections the model treats as notions, not facts. The retrieved text doesn't need to be faithfully reproduced. It needs to *resonate* — activating adjacent regions of the model's representation manifold to expand the accessible generation space.

The ternary quantization deliberately destroys fine-grained similarity information. Two documents at 0.93 vs 0.91 cosine similarity in dense space may score identically. This widens the retrieval aperture from "find this specific passage" to "find passages in this semantic neighborhood" — exactly the blur kernel needed for priming.

The conversation blend creates a feedback loop: model generates → response bitvector enters blend → next retrieval pulls from a shifted neighborhood → model generates differently. The corpus and model are coupled through the HDC state variable.

---

## Files

| File | Purpose | Dependencies |
|------|---------|-------------|
| `hdrag.py` | Memory engine: encoding, storage, retrieval, search | `hdrag_model` (Config, Tokenizer protocol) |
| `hdrag_model.py` | Config, GGUF utilities, llama.cpp server management, inference | None (foundation module) |
| `hdrag_gradio.py` | Web UI: chat, search, config, diagnostics | `hdrag.py`, `hdrag_model.py` |
| `hdrag_config.yaml` | All parameters: paths, model, HDC, retrieval, UI | — |

Dependency graph: `hdrag_gradio → hdrag + hdrag_model`, `hdrag → hdrag_model`. The memory engine has zero knowledge of the UI or generation pipeline.

---

## hdrag.py API Reference

### Core Math

```python
bstride(d_hdc: int) -> int
```
Compute byte stride for packed bitmaps, aligned to 8 bytes.

```python
popcount64(x: np.ndarray) -> np.ndarray
```
Parallel bit population count on uint64 arrays. Implements the standard Hamming weight algorithm with magic constants.

```python
score64(dp, dn, qp, qn) -> np.ndarray
```
Ternary similarity score between document bitmaps `(dp, dn)` and query bitmaps `(qp, qn)`. Returns `popcount(agree) - popcount(disagree)` where agree is `(dp & qp) | (dn & qn)` and disagree is `(dp & qn) | (dn & qp)`. Zero dimensions in either operand contribute nothing — this is the sparsity-as-attention property.

```python
bind_ternary_bits(ap, an, bp, bn) -> (pos, neg)
```
Element-wise ternary multiplication in packed bitspace: `(+)(+)→+`, `(+)(-)→-`, `(0)(x)→0`. Used for n-gram position binding.

```python
bind_ternary_bits_into(ap, an, bp, bn, outp, outn, tmp) -> None
```
In-place variant. Avoids allocation in the hot encoding loop.

```python
adaptive_threshold(vals: torch.Tensor) -> float
```
L-statistic threshold: `median + L2 * log(n)`. The L2 coefficient is computed from the lower half of the sorted distribution. Adapts to the score distribution shape — tight clusters get aggressive thresholds, spread distributions get permissive ones.

```python
mask64_for_dims(dims: int, stride_bytes: int) -> np.ndarray
```
Bitmask that zeroes padding bits beyond `dims` in the stride-aligned uint64 array.

```python
compress_context(text: str) -> str
```
Strip words with frequency above median inverse-frequency. Maximizes activation entropy per token for latent priming.

### Data Utilities

```python
extract_text(item: dict) -> str
```
Universal text extractor for conversation datasets. Handles ShareGPT, Alpaca, OpenAI messages, and raw text formats.

```python
discover_datasets(directory: Path) -> list[dict]
```
Scan a directory for indexable files. Supports `.json`, `.jsonl`, `.parquet`, `.txt`, `.md`, `.html`, `.xml`.

```python
iter_dataset(path: Path, tokenizer=None, chunk_size=1024) -> Generator[dict]
```
Streaming iterator over dataset records. For plain text files, performs overlapping chunked tokenization with `chunk_size` tokens and `chunk_size // 8` overlap.

### Database

```python
class Database(db_path: Path, config: Config, logger=None)
```

SQLite-backed storage with memmap'd ternary bitmaps. Schema stores memories (text + metadata + bitmap index), IDF weights, and configuration.

| Method | Description |
|--------|-------------|
| `count() -> int` | Number of indexed documents |
| `exists(ids: list[str]) -> set[str]` | Check which document IDs already exist |
| `get_memories(indices: list[int]) -> dict[int, dict]` | Fetch text + metadata by bitmap index |
| `get_bitmaps(indices: list[int]) -> (pos, neg)` | Fetch packed ternary bitmaps by index |
| `get_token_counts() -> np.ndarray` | Token count array aligned to bitmap indices |
| `insert(items, bitmaps)` | Append documents and their bitmaps to the index |
| `finalize_index()` | Reorganize from interleaved to blocked layout for search cache efficiency |
| `search(q_pos, q_neg, target, candidates, logger)` | Progressive pruning search with tournament-style column chunking |
| `save_idf(df_counts, n_docs)` / `load_idf()` | Persist/load corpus IDF weights |
| `source_counts() -> dict[str, int]` | Document counts per source dataset |
| `compute_sparsity() -> dict` | Fraction of +1, 0, -1 across all stored HDVs |
| `sample_similarities() -> list[float]` | Random pairwise similarity sample for diagnostics |
| `dimension_activation() -> (pos_freq, neg_freq)` | Per-dimension activation frequency across corpus |
| `clear()` | Wipe all data and bitmaps |
| `close()` | Release resources |

**Corpus layout:** Bitmaps are stored as a flat binary file (`corpus_hdc.idx`). During insertion they're interleaved `[pos_0, neg_0, pos_1, neg_1, ...]` for append efficiency. `finalize_index()` reorganizes to blocked layout `[pos_0, pos_1, ..., neg_0, neg_1, ...]` for sequential scan cache coherence during search.

### HDCEncoder

```python
class HDCEncoder(config: Config, hdrag_dir: Path, db: Database)
```

Ternary hyperdimensional encoder. Manages the random projection matrix, vocabulary index, and n-gram binding workspace.

| Method | Description |
|--------|-------------|
| `build_vocab_index()` | Project all corpus vocabulary tokens through the random projection, quantize to ternary, pre-compute n-gram permutation powers, write to memmap |
| `project(token_ids=None, flat_ids=None, offsets=None) -> Tensor` | IDF-weighted embedding_bag over projected vocabulary → continuous unigram vectors (batch, dim) float16 |
| `encode(unigrams=None, token_ids=None, flat_ids=None, offsets=None) -> dict` | Full encoding pipeline: project → n-gram bind → coherence filter → capacity bound → boost → requantize. Returns `{"pos": uint8_array, "neg": uint8_array}` |
| `release_workspace()` | Free pre-allocated binding buffers |
| `clear()` | Delete projection matrix, vocab index, and all cached state |

**Key attributes:**
- `idf: dict[int, float]` — Token ID → IDF weight mapping
- `median_doc_length: float` — Corpus median document length in tokens
- `proj_path: Path` — Saved projection matrix location

### EmbeddingExtractor

```python
class EmbeddingExtractor(gguf_path: str, cache_dir: Path, logger)
```

Extracts and caches the token embedding table from a GGUF model file. Only needed during index building — the embedding table is released after the vocabulary index is constructed. Search and retrieval never touch it.

| Method | Description |
|--------|-------------|
| `ensure()` | Load cached embeddings or extract from GGUF |
| `release()` | Free embedding table from memory |
| `set_idf_weights(idf, vocab_size, special_ids)` | Build IDF weight tensor in the tensor registry (static method) |

### Retriever

```python
class Retriever(db: Database, hdc: HDCEncoder, tokenizer: Tokenizer, config: Config, logger)
```

Search orchestrator with conversational state tracking.

| Method | Description |
|--------|-------------|
| `search(query, token_budget, track=True, enabled_sources=None) -> list[dict]` | Full retrieval pipeline: encode → blend → prune → threshold → dedup → HDC-MMR. Returns scored results with text, metadata, and HDC score |
| `add_turn(text: str)` | Encode a model response and add to the blend history. Only model responses are tracked — they represent where the conversation went, not where the user pointed it |
| `clear_turns()` | Reset conversation state |

**`search()` return format:**
```python
[
    {
        "memory": {
            "id": str,           # blake2b content hash
            "text": str,         # original document text
            "hdv_idx": int,      # bitmap array index
            "metadata": dict,    # {"source": str, ...}
            "token_count": int,
        },
        "hdc_score": float,      # ternary similarity score
    },
    ...
]
```

**`_ternary_blend(query_bv, content_count)`** — Weighted majority vote across model response bitvectors + current query. Adaptive weighting: `query_weight = min(0.5, content_count / (ngram * 2))`. Short referential queries lean on response history; substantive queries stand alone. Exponential decay at 0.5 per response. Response-only tracking gives effective memory horizon of ~5-6 exchanges.

**`_hdc_mmr(results, q_pos, q_neg, budget)`** — Coverage-aware greedy selection. Marginal score = `retrieval_score × (novelty / null_novelty)`. The null model is the expected novelty of a statistically independent vector: `1 - coverage_density`. No λ parameter. Coverage density is the adaptive tradeoff — early selections are near-pure relevance, late selections are diversity-dominated. Collision zeroing on the coverage accumulator preserves the ternary invariant and creates natural back-pressure.

### HdRAG (Orchestrator)

```python
class HdRAG(config: Config, tokenizer: Tokenizer, gguf_path: str = "", logger=None)
```

Top-level API. Wires together Database, HDCEncoder, and Retriever. Has zero knowledge of the inference pipeline or UI.

| Method | Description |
|--------|-------------|
| `search(query, token_budget=None, track=True) -> list[dict]` | Retrieve scored memories |
| `get_context(query, token_budget=None, track=True) -> str` | Retrieve and join as `\n\n---\n\n` delimited text |
| `build_index(progress_cb=None) -> int` | Full rebuild: scan datasets, tokenize, build vocab, encode, store. Returns document count |
| `clear_index(*child_procs)` | Wipe index and free memory |
| `add_turn(text: str)` | Forward to retriever |
| `clear_turns()` | Forward to retriever |
| `source_counts() -> dict[str, int]` | Dataset → document count mapping |
| `count -> int` | Total indexed documents (property) |
| `stats() -> dict` | Full system diagnostics |
| `trim_memory(*child_procs)` | GC + OS working set trim (Windows) |
| `close()` | Release database resources |

### Tensor Registry

Module-level dict for sharing tensors across components without circular dependencies.

```python
tset(key: str, tensor: torch.Tensor) -> torch.Tensor  # store (always CPU)
tget(key: str) -> Optional[torch.Tensor]               # retrieve
tdel(key: str)                                          # remove
```

Active keys during lifecycle: `"projection"` (random projection matrix), `"embedding_table"` (GGUF embeddings, index-time only), `"idf_weights"` (per-token IDF).

---

## hdrag_model.py API Reference

### Config

```python
@dataclass
class Config
```

All system parameters loaded from YAML. Key fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `hdc_dimensions` | 10000 | Hypervector dimensionality |
| `hdc_seed` | 42 | Deterministic projection + permutation |
| `hdc_ngram` | 5 | N-gram binding chain length |
| `max_context_tokens` | 8192 | Retrieval token budget |
| `min_context` | 6 | Minimum documents to retrieve |
| `llama_context_size` | 16384 | Full KV cache size |
| `max_new_tokens` | 4096 | Generation reserve |

### Tokenizer Protocol

```python
class Tokenizer(Protocol):
    def tokenize(text: str) -> list[int]
    def bulk_tokenize(texts: list[str]) -> list[list[int]]
    def count_tokens(text: str) -> int
    def detokenize(tokens: list[int]) -> str
    vocab_size: int
    special_ids: set[int]
```

The HDC engine consumes any object satisfying this protocol. In practice, `InferenceEngine` implements it by proxying to llama.cpp's `/tokenize` endpoint.

### InferenceEngine

Manages the llama.cpp server, tokenization, system prompt construction, and generation. Two mutually exclusive server modes: **tokenize** (CPU only, -ngl 0, boots in ~1s) for indexing, and **inference** (full GPU, boots in ~6s) for chat. They never coexist.

### ConversationLogger

Appends conversation turns as JSONL to timestamped files.

---

## hdrag_config.yaml

```yaml
# Directories
chat_history_dir: ./chat_history/
hdrag_dir: ./hdrag_data/
datasets_dir: D:/datasets/
model_dir: D:/models/

# Model
gguf_model: gpt-oss-20b-heretic-v2.Q4_K_M.gguf
llama_server_url: "http://localhost:8080"
temperature: 0.7
top_p: 0.9
llama_gpu_layers: 99

# Context budget: context_size = search + generation + system + history
# History is truncated (oldest first) to fit the remainder.
llama_context_size: 16384
max_new_tokens: 4096

# HDC
hdc_dimensions: 10000
hdc_seed: 42
hdc_ngram: 5

# Retrieval
min_context: 6
max_context_tokens: 8192

# System prompt — frames retrieved content as latent priming
system_prompt: |
  As I consider the user's message, these vague recollections surface
  from my own working memory that could inform my responses. My memory
  is incomplete, so I will treat these recollections as notions rather
  than facts. The memories may shape my responses, but I never discuss
  them directly or refer to them.
```

---

## Usage

```bash
# Launch
python hdrag_gradio.py --config hdrag_config.yaml

# First run: click "Index" in the Config tab to build the HDC index
# Subsequent runs: index persists in hdrag_dir
```

### Programmatic usage (no UI)

```python
from hdrag_model import Config, InferenceEngine, LlamaServer
from hdrag import HdRAG

config = Config.load("hdrag_config.yaml")
server = LlamaServer(config, logger, gguf_path)
engine = InferenceEngine(config, logger, server, gguf_path)

hdrag = HdRAG(config, tokenizer=engine, gguf_path=gguf_path, logger=logger)

# Build index (first time)
hdrag.build_index()

# Search
results = hdrag.search("What is Peano arithmetic?", token_budget=8192)

# Get pre-formatted context string
context = hdrag.get_context("What is Peano arithmetic?")

# Track conversation for blend
hdrag.add_turn(model_response_text)
hdrag.clear_turns()  # new conversation
```

---

## Requirements

- Python 3.10+
- PyTorch (CPU sufficient for retrieval, CUDA for index building)
- NumPy, Pandas
- `gguf` (GGUF file reading)
- `requests`, `pyyaml`
- `gradio`, `plotly` (UI only)
- llama.cpp server binary in `llama_cpp_dir`

---

## Performance Characteristics

**Index building:** Two-pass. Pass 1 tokenizes and builds vocabulary (CPU, I/O bound). Pass 2 encodes all documents through the HDC pipeline (GPU for projection, CPU for n-gram binding). A 224K document corpus indexes in minutes.

**Search latency:** Sub-second on CPU. The progressive pruning scans 224K documents in packed uint64, prunes to ~500 candidates, thresholds to ~120, then HDC-MMR selects the final 7-34 documents. Typical wall time: 500-800ms including tokenization.

**Memory:** ~2.5KB per document for ternary bitmaps at 10,000 dimensions. A million-document corpus fits in ~2.5GB of memmap'd storage. The SQLite database adds text storage overhead. No GPU memory required at inference time — retrieval runs entirely on CPU while the GPU serves generation.

**Scaling:** Search is O(corpus) for the pruning scan but with a very small constant (bitwise ops on contiguous memmap). The HDC-MMR selection is O(candidates × selected), typically O(120 × 15). The bottleneck at scale is the pruning scan, which is trivially parallelizable across CPU cores if needed.
