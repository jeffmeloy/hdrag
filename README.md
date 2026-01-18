# HdRAG: Hyperdimensional Retrieval-Augmented Generation 
HdRAG is a hyperdimensional computing (HDC) based retrieval-augmented generation system that implements a local-first pipeline using ternary hypervectors for document indexing and retrieval.

## System Architecture

The system consists of several integrated components:
*   **HDCEncoder:** Projects dense Transformer embeddings into a high-dimensional ternary space.
*   **Database:** A dual-storage engine using SQLite for metadata and a custom memory-mapped (mmap) format for hypervectors.
*   **Retriever:** Orchestrates search using bitwise similarity scoring and adaptive filtering.
*   **Deduplicator:** A multi-stage filtering system for result refinement.
*   **ModelManager:** Interface for HuggingFace models for embedding extraction and text generation.

## Technical Implementation

### 1. Vector Encoding Pipeline
Document representation is generated through the following sequence:
1.  **Tokenization:** Text is processed via a HuggingFace tokenizer.
2.  **Embedding Extraction:** Hidden states are extracted from the model's embedding table.
3.  **IDF-Weighted Pooling:** Token embeddings are aggregated using an Inverse Document Frequency (IDF) weighted mean pool.
4.  **Random Projection:** The resulting vector is projected via a fixed Gaussian random matrix ($\mathbb{R}^d \rightarrow \mathbb{R}^D$), where $D$ is typically $\geq 10,000$.
5.  **Ternarization:** Values are thresholded at $\pm 1/\sqrt{d}$ to produce a ternary hypervector (HDV) containing elements in $\{-1, 0, +1\}$.
6.  **Bitmap Packing:** The HDV is stored as two bitmasks (Positive and Negative) packed into `uint64` arrays.

### 2. Search and Retrieval
The system uses a two-pass search algorithm over the memory-mapped bitmasks:
*   **Coarse Pass:** The system performs a linear scan using SIMD Within A Register (SWAR) popcount logic. It samples 10% of the dimensions across two random contiguous slices to generate initial candidates.
*   **Refinement Pass:** The full dimensions of the candidate hypervectors are scored using the Hamming-based agreement/disagreement formula:
    `Score = popcount(d_pos & q_pos | d_neg & q_neg) - popcount(d_pos & q_neg | d_neg & q_pos)`
*   **Length Normalization:** Scores are normalized by the square root of the document's token count.

### 3. Deduplication Logic
Retrieved results undergo three stages of filtering:
1.  **Exact Match:** Removal of identical string content.
2.  **Token Subset:** Removal of documents where the token set is a subset of another retrieved document.
3.  **Bitmap Near-Duplicate:** Calculation of pairwise similarity between document bitmaps. The system uses **Otsu's Method** to dynamically determine a similarity threshold for exclusion.

### 4. Conversation and Context Management
*   **Recency-Weighted Blending:** Query vectors are blended with previous conversation turns. Weights are assigned as $1/n$ based on the distance from the current turn.
*   **Whale Filtering:** Documents exceeding a calculated token threshold (based on budget and context requirements) are excluded.

### 5. Storage Layout
*   **Metadata:** SQLite stores document IDs, raw text, JSON metadata, and token counts.
*   **HDC Index:** A `.idx` file stores bitmaps. Upon finalization, the file is reorganized into a blocked layout where all positive bitmasks are contiguous, followed by all negative bitmasks, to facilitate efficient memory-mapped access.

## Data Structures

| Component | Format |
| :--- | :--- |
| **HDV Dimensions** | Default 10,000+ (User configurable) |
| **HDV Type** | Ternary ($\pm 1, 0$) |
| **Bitmap Storage** | `uint8` packed, `uint64` aligned for SWAR |
| **IDF Storage** | SQLite `idf` table (token_id, weight) |
| **Mmap Access** | `PROT_READ` / `ACCESS_READ` |

## Requirements and Environment
*   **Hardware Acceleration:** Supports CUDA for embedding extraction; retrieval scoring is CPU-bound using NumPy/SWAR.
*   **Persistence:** All indices and configurations are stored locally in the directory defined by the `Config`.
*   **Model Compatibility:** Compatible with any HuggingFace `AutoModelForCausalLM` and `AutoTokenizer` that supports safe-tensors.

Usage
-----
    python hdrag.py --config hdrag_config.yaml [--debug]
