# HdRAG: Hyperdimensional Retrieval-Augmented Generation

A local-first RAG system using **hyperdimensional computing (HDC)** for fast, memory-efficient document retrieval combined with local LLM generation.

## Why HdRAG?

Traditional RAG systems rely on dense vector databases, HdRAG uses a fundamentally different approach:

- **Ternary Hypervectors**: Documents are encoded as sparse ternary vectors (+1, 0, -1), packed into bitmaps
- **Bitwise Similarity**: Search uses XOR + popcount operations — fast on CPU
- **No Vector DB**: Just SQLite + memory-mapped files
- **Local-First**: Everything runs locally with any HuggingFace model

## Features

### Core
- **HDC Encoding**: Random projection of token embeddings into high-dimensional ternary space
- **Fast Retrieval**: SWAR-optimized bitwise similarity scoring
- **IDF Weighting**: Token importance weighting for both indexing and search
- **Deduplication**: Otsu-threshold near-duplicate removal

### Context Modes
- **Standard**: Full retrieved context injected into prompt
- **Compress**: IDF-filtered unique tokens only — reduces context size while preserving key concepts (experimental, good for small models)

### UI (Gradio)
- **Chat**: Streaming responses with memory toggle and context compression
- **Search**: Direct retrieval testing
- **Config**: HDC dimensions, model settings, dataset selection
- **Stats**: Corpus visualizations (sparsity, similarity distribution, dimension activation)

### Data Management
- **Multi-format**: JSON, JSONL, Parquet, TXT, MD, HTML, XML
- **Dataset Filtering**: Enable/disable specific sources at search time
- **Conversation Logging**: Auto-saved chat history in JSONL format

## Installation

```bash
# Clone repository
git clone https://github.com/youruser/hdrag.git
cd hdrag

# Create environment
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt 
```

## Quick Start

### 1. Edit Config

Create `hdrag_config.yaml`:

```yaml
# Directories
chat_history_dir: ./chat_history/       # Where conversation logs are saved as JSONL
hdrag_dir: ./hdrag_data/                # SQLite DB, HDV binary, and projection matrix
datasets_dir: D:/datasets/              # Source JSON/JSONL files to index
model_dir: D:/models/safetensors/input_models/  # HuggingFace model cache location

# Model 
model_name: "p-e-w/Qwen3-4B-Instruct-2507-heretic"  # HF repo ID or local path
temperature: 0.7                        # Sampling temperature for generation
top_p: 0.9                              # Nucleus sampling threshold
max_new_tokens: 4096                    # Max tokens to generate per response
max_length_tokens: 16384                # Truncation limit for input tokenization

# HDC Encoding
hdc_dimensions: 10000                   # Hypervector dimensionality (higher = more fidelity, more RAM)
hdc_seed: 42                            # RNG seed for reproducible random projection matrix

# .txt file chunking, only used .txt file, not for datasets
text_chunk_size: 1024                   # max chunk size for .txt files in dataset 
text_chunk_overlap: 128                 # text chunk overlap for .txt file in dataset

# Indexing
batch_size: 4                           # Documents per embedding batch (GPU memory bound)
vocab_chunk_multiplier: 8               # Vocab pass processes batch_size * this many docs at once
hash_digest_size: 8                     # Blake2b digest bytes for document IDs (8 = 64-bit)
export_log_interval: 100000             # Log progress every N vectors during HDV export
batch_log_interval: 1000                 # Log progress every N batches during encoding

# Retrieval
max_context_tokens: 8192                # Token budget for retrieved context in prompt
min_context: 6                          # include at least these many dataset items 
hdc_search_mb: 64                       # candidates to score (L3 cache)

# Database
sqlite_max_vars: 900                    # Chunk size for IN clauses (SQLite limit ~999)
sqlite_cache_kb: 64000                  # SQLite page cache size in KB (64MB)
sqlite_mmap_bytes: 2147483648           # SQLite mmap size

# UI
gradio_port: 7860                       # Local port for Gradio web interface

# Prompt
system_prompt: |                        # Injected before retrieved context in chat
  As I consider the user's message, these vague recollections surface from my own working memory that could inform my responses. My memory is incomplete, so I will treat these recollections as notions rather than facts.

```

### 2. Add Data

Place your documents in the `datasets_dir`:
```
datasets/
├── knowledge_base.jsonl
├── papers.parquet
└── notes.txt
```

Supported formats:
- `.json` / `.jsonl`: Objects with `text`, `content`, `conversations`, or message arrays
- `.parquet`: DataFrame with text columns
- `.txt` / `.md` / `.html` / `.xml`: Auto-chunked by token count

### 3. Run

```bash
python hdrag.py --config hdrag_config.yaml
```

Open http://localhost:7860

### 4. Index

Click the **🔄 Index** button in the Config tab to build the HDC index.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         HdRAG                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ ModelManager│  │  HDCEncoder │  │     Retriever       │  │
│  │             │  │             │  │                     │  │
│  │ • Tokenizer │  │ • Random    │  │ • Vocab Index       │  │
│  │ • Embeddings│  │   Projection│  │ • HDC Scoring       │  │
│  │ • Generation│  │ • Ternary   │  │ • Deduplication     │  │
│  │             │  │   Threshold │  │ • Context Assembly  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Database                         │    │
│  │  • SQLite: metadata, IDF weights, config            │    │
│  │  • corpus_hdc.idx: document bitmaps (mmap)          │    │
│  │  • token_hdc.idx: vocabulary bitmaps (mmap)         │    │
│  │  • vocab.idx: inverted index (CSR format)           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

### HDC Encoding Pipeline

1. **Tokenize** → HuggingFace tokenizer
2. **Embed** → Extract from model's embedding table
3. **Pool** → IDF-weighted mean of token embeddings
4. **Project** → Multiply by fixed Gaussian random matrix (R^d → R^D)
5. **Ternarize** → Threshold at ±1/√d to get {-1, 0, +1}
6. **Pack** → Store as two bitmaps (positive, negative)

### Similarity Scoring

```
Score = popcount(q_pos & d_pos) + popcount(q_neg & d_neg)    # agreement
      - popcount(q_pos & d_neg) - popcount(q_neg & d_pos)    # disagreement
```

Normalized by `sqrt(self_similarity)` for length invariance.

### Context Compression (Experimental)

When "Compress Context" is enabled:
1. Retrieve documents normally (fills token budget)
2. Tokenize and extract unique tokens (preserving order)
3. Compute local median IDF of context tokens
4. Keep only tokens with IDF ≥ median
5. Decode back to text

This produces a "concept cloud" rather than prose — useful for:
- Small models that copy context too literally
- Reducing context window usage
- Forcing synthesis over regurgitation

### Deduplication

Three-stage filtering:
1. **Exact**: Remove identical normalized text
2. **Subset**: Remove docs whose token set ⊆ another doc's
3. **Near-duplicate**: Otsu-thresholded bitmap similarity

### Storage Format

| File | Contents |
|------|----------|
| `index.db` | SQLite: memories, IDF weights, config |
| `corpus_hdc.idx` | Document bitmaps (blocked: all pos, then all neg) |
| `token_hdc.idx` | Vocabulary bitmaps for query expansion |
| `vocab.idx` | Inverted index (CSR: row_ptrs + postings) |
| `projection.pt` | Random projection matrix |

## Configuration Reference

| Field | Description | Default |
|-------|-------------|---------|
| `hdc_dimensions` | Hypervector dimensionality | 10000 |
| `hdc_seed` | Random projection seed | 42 |
| `max_context_tokens` | Token budget for retrieval | 4000 |
| `min_context` | Minimum documents to retrieve | 3 |
| `hdc_search_mb` | Memory budget for candidate scoring | 64 |
| `text_chunk_size` | Tokens per chunk for .txt files | 1024 |
| `text_chunk_overlap` | Overlap between chunks | 128 |

## UI Guide

### Chat Tab
- **Token Budget**: Max tokens for retrieved context
- **Use Memory**: Toggle retrieval on/off
- **Compress Context**: Enable IDF-filtered concept extraction
- **Debug Viewer**: See full prompt including working memory

### Config Tab
- **HDC Settings**: Dimensions and seed (requires reindex)
- **Retrieval Settings**: Context token budget
- **Model Settings**: Temperature, top_p, max output tokens
- **Datasets**: Enable/disable specific sources for search

### Stats Tab
- **Sparsity**: Distribution of +1/0/-1 across all HDVs
- **Similarity**: Pairwise document similarity histogram
- **Dimension Activation**: Per-dimension +/- frequency

## Performance Notes

- **Indexing**: ~1-5 docs/sec depending on model and batch size
- **Search**: <100ms for 100k+ documents (CPU only)
- **Memory**: ~2 bits per dimension per document
- **GPU**: Used for embedding extraction only; retrieval is CPU

## Limitations

- Assumes spherical similarity (no learned metric)
- Random projection loses some semantic precision vs. dense vectors
- Context compression is experimental — may hurt coherence for some tasks

## License

MIT

## Acknowledgments

- Hyperdimensional computing concepts from Kanerva, Rahimi, et al.
- Built with HuggingFace Transformers, Gradio, PyTorch
