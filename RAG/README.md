# RAG — Retrieval-Augmented Generation

RAG is an architecture that enhances LLMs by retrieving relevant documents from a knowledge base at inference time. Instead of relying solely on parametric knowledge baked into model weights, RAG grounds responses in up-to-date, domain-specific context.

---

## 📖 **Sections**

- [Why RAG?](#why-rag)
- [Core Architecture](#core-architecture)
- [Building a RAG Pipeline](#building-a-rag-pipeline)
- [Chunking Strategies](#chunking-strategies)
- [Retrieval Techniques](#retrieval-techniques)
- [Advanced RAG Patterns](#advanced-rag-patterns)
- [Evaluation](#evaluation)

---

## 🤔 **Why RAG?**

| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| Knowledge cutoff | LLM only knows training data | Access to real-time/custom docs |
| Hallucination | Model invents facts | Grounded in retrieved evidence |
| Domain specificity | Generic responses | Company/domain-specific answers |
| Source attribution | Can't cite sources | Direct document references |
| Cost | Fine-tuning = expensive | Dynamic retrieval = cheap updates |

---

## 🏗️ **Core Architecture**

```
User Query
    │
    ▼
[Query Encoder] ──► Query Embedding
    │
    ▼
[Vector Store] ──► Top-K Relevant Chunks
    │
    ▼
[Context Assembly] ──► [LLM] ──► Final Answer
```

**Two main phases:**
1. **Indexing** (offline): Chunk documents → Embed → Store in vector DB
2. **Retrieval** (online): Embed query → Search vector DB → Augment prompt → Generate

---

## 🔨 **Building a RAG Pipeline**

### Step 1: Document Ingestion

```python
from pathlib import Path
import anthropic

# Load documents
def load_documents(directory: str) -> list[dict]:
    documents = []
    for file_path in Path(directory).rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
                "content": f.read(),
                "source": str(file_path),
                "filename": file_path.name
            })
    return documents

docs = load_documents("./knowledge_base")
print(f"Loaded {len(docs)} documents")
```

### Step 2: Chunking

```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

# Chunk all documents
all_chunks = []
for doc in docs:
    chunks = chunk_text(doc["content"])
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "source": doc["source"],
            "chunk_id": f"{doc['filename']}_{i}"
        })

print(f"Created {len(all_chunks)} chunks")
```

### Step 3: Embedding & Indexing

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, good quality

def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatL2, list[dict]]:
    """Build FAISS index from chunks."""
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings (batch for efficiency)
    print("Generating embeddings...")
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim after normalization
    index.add(embeddings)

    print(f"Indexed {index.ntotal} vectors of dimension {dimension}")
    return index, chunks

index, metadata = build_index(all_chunks)

# Save for reuse
faiss.write_index(index, "knowledge_base.index")
```

### Step 4: Retrieval

```python
def retrieve(query: str, index, metadata: list[dict], top_k: int = 5) -> list[dict]:
    """Retrieve top-K relevant chunks for a query."""
    # Embed query
    query_embedding = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                **metadata[idx],
                "relevance_score": float(score)
            })

    return results

results = retrieve("What are the refund policies?", index, metadata)
for r in results:
    print(f"Score: {r['relevance_score']:.3f} | Source: {r['source']}")
    print(f"  {r['text'][:200]}...\n")
```

### Step 5: Generation

```python
import anthropic

client = anthropic.Anthropic()

def rag_answer(query: str, index, metadata: list[dict], top_k: int = 5) -> dict:
    """Full RAG pipeline: retrieve + generate."""
    # Retrieve relevant context
    retrieved = retrieve(query, index, metadata, top_k=top_k)

    if not retrieved:
        return {"answer": "No relevant information found.", "sources": []}

    # Build context string
    context_parts = []
    for i, chunk in enumerate(retrieved, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Generate answer
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite which source(s) you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.content[0].text,
        "sources": [r["source"] for r in retrieved],
        "retrieved_chunks": retrieved
    }

result = rag_answer("What is the return policy for electronics?", index, metadata)
print(result["answer"])
print("\nSources used:", result["sources"])
```

---

## ✂️ **Chunking Strategies**

### Fixed-Size Chunking
Simple but may cut sentences mid-way.

```python
def fixed_size_chunks(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]
```

### Sentence-Aware Chunking
Respects sentence boundaries.

```python
import nltk
nltk.download("punkt", quiet=True)

def sentence_chunks(text: str, max_sentences: int = 5, overlap: int = 1) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks
```

### Semantic Chunking
Group sentences by semantic similarity.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_chunks(text: str, threshold: float = 0.5) -> list[str]:
    """Split text at semantic boundaries."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

        if sim < threshold:  # Semantic break detected
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

---

## 🔍 **Retrieval Techniques**

### Hybrid Search (Dense + Sparse)

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, chunks: list[str], alpha: float = 0.5):
        """
        alpha: weight for dense search (1-alpha for sparse)
        """
        self.chunks = chunks
        self.alpha = alpha

        # Sparse: BM25
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Dense: embeddings
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embed_model.encode(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        # Sparse scores
        sparse_scores = self.bm25.get_scores(query.lower().split())

        # Dense scores
        q_emb = self.embed_model.encode([query])
        dense_scores = cosine_similarity(q_emb, self.embeddings)[0]

        # Normalize and combine
        sparse_norm = sparse_scores / (sparse_scores.max() + 1e-9)
        dense_norm = dense_scores / (dense_scores.max() + 1e-9)
        combined = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        top_indices = combined.argsort()[-top_k:][::-1]
        return [(int(i), float(combined[i])) for i in top_indices]
```

### Re-Ranking with Cross-Encoder

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Re-rank candidates using a cross-encoder (more accurate but slower)."""
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

---

## 🚀 **Advanced RAG Patterns**

### Hypothetical Document Embeddings (HyDE)

```python
def hyde_retrieve(query: str, index, metadata, top_k: int = 5) -> list[dict]:
    """Generate a hypothetical answer first, then retrieve with it."""
    # Step 1: Generate a hypothetical document
    hypothesis_prompt = f"""Write a detailed paragraph that would answer this question:
{query}

Write as if you are an expert writing a factual document. Be specific and detailed."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": hypothesis_prompt}]
    )
    hypothetical_doc = response.content[0].text

    # Step 2: Retrieve using the hypothetical document (not the original query)
    return retrieve(hypothetical_doc, index, metadata, top_k=top_k)
```

### Multi-Query RAG

```python
def multi_query_retrieve(query: str, index, metadata, top_k: int = 3) -> list[dict]:
    """Generate multiple query variations for better coverage."""
    # Generate query variations
    expansion_prompt = f"""Generate 3 different phrasings of this question for document retrieval.
Output one per line, no numbering.

Question: {query}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": expansion_prompt}]
    )

    queries = [query] + response.content[0].text.strip().split("\n")

    # Retrieve for each query and merge
    seen_ids = set()
    all_results = []

    for q in queries:
        results = retrieve(q, index, metadata, top_k=top_k)
        for r in results:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                all_results.append(r)

    # Sort by relevance and return top results
    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return all_results[:top_k * 2]
```

### Contextual Compression

```python
def compress_context(query: str, retrieved_chunks: list[dict]) -> str:
    """Extract only the relevant parts from retrieved chunks."""
    full_context = "\n\n".join([c["text"] for c in retrieved_chunks])

    compression_prompt = f"""Extract ONLY the sentences or phrases directly relevant to answering this question.
Remove irrelevant content. Keep exact wording from the original.

Question: {query}

Context:
{full_context}

Relevant excerpts:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": compression_prompt}]
    )
    return response.content[0].text
```

---

## 📏 **Evaluation**

### RAGAS Metrics

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is the return policy?", "How do I contact support?"],
    "answer": ["Returns are accepted within 30 days...", "Contact support via email..."],
    "contexts": [["Our return policy allows...", "Items must be..."], ["Support team can be reached..."]],
    "ground_truth": ["Items can be returned within 30 days of purchase.", "Support is available via email."]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[answer_relevancy, faithfulness, context_recall, context_precision])
print(results)
```

---

## 💡 **Tips & Tricks**

1. **Chunk size matters**: 256-512 tokens usually works best. Too small = no context; too large = noise
2. **Overlap is crucial**: 10-15% overlap prevents losing information at chunk boundaries
3. **Metadata filtering**: Always store document metadata to enable pre-filtering before vector search
4. **Cache embeddings**: Embedding generation is expensive — cache and reuse
5. **Monitor retrieval quality**: Log what gets retrieved; poor retrieval = poor answers regardless of LLM quality
6. **Use re-ranking for critical applications**: Cross-encoders are 10-20x more accurate but slower

---

## 🔗 **Related Topics**

- [Vector Databases & Embeddings](../Vector-Databases/README.md)
- [Prompt Engineering](../Prompt-Engineering/README.md)
- [LLM Agents](../LLM/Agents.md)
- [Fine-Tuning LLMs](../Fine-Tuning/README.md)
