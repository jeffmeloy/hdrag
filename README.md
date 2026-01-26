# HdRAG: Hyperdimensional Retrieval-Augmented Generation

A local-first RAG system using **hyperdimensional computing (HDC)** for fast, memory-efficient document retrieval combined with local LLM generation.

## Why HdRAG?

Traditional RAG systems rely on dense vector databases with approximate nearest neighbor search. HdRAG takes a fundamentally different approach:

- **Ternary Hypervectors**: Documents encoded as sparse {+1, 0, -1} vectors, packed into dual bitmaps
- **Bitwise Similarity**: Search uses AND + popcount — integer ops, no floating point in the hot path
- **Progressive Pruning**: Coarse-to-fine search reads ~1/6th of total data by making early low-fidelity cuts
- **Data-Driven Thresholds**: Otsu's method finds natural cluster boundaries per query — no magic numbers
- **No Vector DB**: Just SQLite + memory-mapped files
- **Local-First**: Everything runs locally with any HuggingFace model

## How It Works

### The Ternary Encoding Insight

Dense embeddings are float32 vectors — 768-4096 dimensions × 4 bytes = 3-16KB per document. HdRAG projects these into ternary space: each dimension becomes +1, -1, or 0.

The zeros aren't wasted bits. They encode **uncertainty**. When a random projection lands near zero, the system abstains from voting on that dimension. A typical corpus shows ~68% zeros — that's the embedding's confidence interval made geometric. Only 32% of dimensions actually vote; the rest say "I don't have strong evidence either way."

Storage: two bitmaps per document (positive bits, negative bits). A dimension is +1 if pos is set, -1 if neg is set, 0 if neither. 10k dimensions = 2.5KB per document instead of 40KB for float32.

### Why Random Projections Work

In 3D, projecting onto random directions loses information. In 4096D, random projections preserve nearly all pairwise structure. Johnson-Lindenstrauss guarantees it: O(log n / ε²) dimensions suffice to preserve distances within (1±ε) for n points.

The deeper weirdness is concentration of measure. In high dimensions, random vectors are almost orthogonal with probability approaching 1. The projection matrix isn't carefully designed — it's sampled from a geometry where careful design is unnecessary.

### Similarity as Bit Counting
```
agree    = popcount(q_pos & d_pos) + popcount(q_neg & d_neg)
disagree = popcount(q_pos & d_neg) + popcount(q_neg & d_pos)
score    = agree - disagree
```

Four ANDs, four popcounts, one subtract. All integer ops. Popcount is a single CPU instruction. The AND operations process 64 dimensions per instruction. A 10k dimension dot product that would be 10k multiplies + 10k adds in float32 becomes ~600 integer ops total.

### IDF-Weighted Embeddings

Raw token embeddings are dominated by high-frequency tokens. "The," "is," "of" have massive embedding norms from training — they've seen the most gradient updates. A naive mean pool is mostly stopword soup.

IDF inverts that. Rare tokens get amplified, common tokens get suppressed. The embedding shifts toward the document's **distinctive** semantic signature. When that hits the ternary projection, you're binarizing discriminative content, not boilerplate.

Information-theoretically: IDF is the self-information of the token, `-log(P(token))`. Weighting by IDF means weighting by bits of information. A document's embedding becomes its entropy-weighted centroid.

The IDF is computed from **your** corpus, not pretrained. The embedding table is frozen generic semantics. IDF is where your corpus statistics enter the representation — adapting a general-purpose embedding space to your specific information distribution.

### Progressive Pruning

Scoring all 223k documents against all 10k dimensions would read ~4.5GB. Instead, the search runs in passes:

1. Score all candidates on dimensions 0-833 (1/12 of total)
2. Keep top 50% by cumulative score
3. Score survivors on dimensions 834-1666
4. Keep top 50%
5. Repeat for 12 passes

By the final pass, you're scoring ~100 survivors on the last dimension chunk. Total bytes read: ~750MB instead of 4.5GB. The early passes are deliberately low-fidelity — they're only good enough to eliminate obvious non-matches.

The key insight: partial information enables early decisions. You don't need full precision to know that a document about cooking is irrelevant to a query about polynomials.

### Otsu Thresholding

Most retrieval systems use top-k (arbitrary) or similarity cutoffs (tuned on dev set, drifts over time). HdRAG uses Otsu's method: find the threshold that maximizes inter-class variance in the score distribution.

This is just Bayesian decision theory. You have a mixture of two distributions (relevant, irrelevant). Otsu finds where the posterior odds flip — the point where P(relevant|score) = P(irrelevant|score). The threshold recalibrates per query from the actual score distribution.

The assumption is bimodality. Real corpora have it — documents cluster by topic, queries land near some clusters and far from others. If no valley exists (everything equally irrelevant), Otsu returns 0 and nothing filters. Graceful degradation.

## Features

### Core
- **HDC Encoding**: Random projection of IDF-weighted embeddings into high-dimensional ternary space
- **Progressive Retrieval**: Coarse-to-fine pruning reads fraction of total corpus data
- **Adaptive Thresholds**: Otsu-based relevance cutoff and deduplication — no hyperparameters
- **Pure Semantic**: No lexical/BM25 hybrid — single representation end-to-end

### Context Modes
- **Standard**: Full retrieved context injected into prompt
- **Compress**: IDF-filtered keywords only — reduces context while preserving signal (experimental)

### UI (Gradio)
- **Chat**: Streaming responses with memory toggle and context compression
- **Search**: Direct retrieval testing with score visualization
- **Config**: HDC dimensions, model settings, dataset selection
- **Stats**: Corpus visualizations (sparsity, similarity distribution, dimension activation)

## Installation
```bash
git clone https://github.com/jeffmeloy/hdrag.git
cd hdrag
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows
pip install -r requirements.txt 
```

## Quick Start

### 1. Edit Config

Create `hdrag_config.yaml`:
```yaml
# Directories
chat_history_dir: ./chat_history/
hdrag_dir: ./hdrag_data/
datasets_dir: D:/datasets/
model_dir: D:/models/

# Model 
model_name: "p-e-w/Qwen3-4B-Instruct-2507-heretic"
temperature: 0.7
top_p: 0.9
max_new_tokens: 4096
max_length_tokens: 16384

# HDC Encoding
hdc_dimensions: 10000
hdc_seed: 42

# Retrieval
max_context_tokens: 8192
min_context: 6

# Database
sqlite_cache_kb: 64000

# UI
gradio_port: 7860

system_prompt: |
  As I consider the user's message, these vague recollections surface from my own working memory that could inform my responses. My memory is incomplete, so I will treat these recollections as notions rather than facts.
```

### 2. Add Data

Place documents in `datasets_dir`. Supported formats:
- `.json` / `.jsonl`: Objects with `text`, `content`, `conversations`, or message arrays
- `.parquet`: DataFrame with text columns
- `.txt` / `.md` / `.html` / `.xml`: Auto-chunked by token count

### 3. Run
```bash
python hdrag.py --config hdrag_config.yaml
```

Open http://localhost:7860 and click **🔄 Index** to build the HDC index.

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                         HdRAG                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ ModelManager│  │  HDCEncoder │  │     Retriever       │  │
│  │             │  │             │  │                     │  │
│  │ • Tokenizer │  │ • Random    │  │ • Progressive Prune │  │
│  │ • Embeddings│  │   Projection│  │ • Otsu Threshold    │  │
│  │ • Generation│  │ • Ternary   │  │ • Deduplication     │  │
│  │ • IDF Weights│ │   Packing   │  │ • Budget Assembly   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Database                           │  │
│  │  • SQLite: metadata, IDF weights, token counts        │  │
│  │  • corpus_hdc.idx: document bitmaps (mmap)            │  │
│  │  • projection.pt: random projection matrix            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

### HDC Encoding Pipeline

1. **Tokenize** → HuggingFace tokenizer
2. **Embed** → Lookup from model's embedding table
3. **Pool** → IDF-weighted mean (corpus-specific term importance)
4. **Project** → Gaussian random matrix (R^d → R^D)
5. **Ternarize** → Threshold at ±1/√d to get {-1, 0, +1}
6. **Pack** → Store as dual bitmaps (pos, neg)

### Search Pipeline

1. **Whale Filter** → Remove docs that can't fit in token budget (O(1) metadata check)
2. **Progressive Prune** → 12-pass coarse-to-fine scoring, halving survivors each pass
3. **Otsu Relevance** → Data-driven score threshold separates relevant from irrelevant
4. **Otsu Dedup** → Data-driven similarity threshold removes near-duplicates
5. **Budget Fill** → Greedily pack highest-scoring survivors into context window

### Context Compression (Experimental)

When enabled, retrieved context is reduced to high-IDF tokens only:
1. Compute median IDF across context tokens
2. Keep only words with IDF ≥ median
3. Result is a "concept cloud" — keywords without grammatical mortar

The LLM reconstructs coherence from its pretrained priors. Works well for semantic priming; may hurt tasks requiring precise quotes or structure.

### Storage Format

| File | Contents | Size (300k docs) |
|------|----------|------------------|
| `index.db` | SQLite: text, metadata, IDF, config | ~500MB |
| `corpus_hdc.idx` | Document bitmaps (blocked layout) | ~750MB |
| `projection.pt` | Random matrix | ~150MB |

## Performance

Tested on 301k documents, 10k dimensions:

| Stage | Time | Reduction |
|-------|------|-----------|
| Whale filter | <1ms | 301k → 223k |
| Progressive prune (12 passes) | ~350ms | 223k → 109 |
| Otsu relevance | <1ms | 109 → 34 |
| Otsu dedup | <1ms | 34 → 15 |
| Budget fill | <1ms | 15 → 14 |
| **Total** | **~400ms** | **301k → 14 docs** |

Memory: ~2 bits per dimension per document. GPU used only for embedding extraction; retrieval is pure CPU.

## License

MIT

## Acknowledgments

- Hyperdimensional computing: Kanerva, Rahimi, et al.
- Built with HuggingFace Transformers, Gradio, PyTorch
