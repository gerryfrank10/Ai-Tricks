# Vector Databases & Embeddings

Vector databases store high-dimensional numerical representations (embeddings) of data and enable lightning-fast semantic similarity search. They are the backbone of RAG systems, recommendation engines, duplicate detection, and semantic search.

---

## 📖 **Sections**

- [What Are Embeddings?](#what-are-embeddings)
- [Embedding Models](#embedding-models)
- [Vector Database Comparison](#vector-database-comparison)
- [Pinecone](#pinecone)
- [ChromaDB](#chromadb)
- [Weaviate](#weaviate)
- [pgvector (Postgres)](#pgvector-postgres)
- [Similarity Search Deep Dive](#similarity-search-deep-dive)
- [Real-World Patterns](#real-world-patterns)

---

## 🧮 **What Are Embeddings?**

Embeddings are dense vector representations that capture semantic meaning. Similar concepts cluster together in the embedding space.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",           # Similar to next
    "A feline rested on a rug.",          # Similar to previous
    "Machine learning is powerful.",      # Different topic
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (3, 384)

# Semantic similarity via cosine
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)

print(f"cat/mat vs feline/rug:   {sim_matrix[0][1]:.3f}")  # ~0.85 (similar)
print(f"cat/mat vs ML:           {sim_matrix[0][2]:.3f}")  # ~0.10 (different)
```

---

## 🤖 **Embedding Models**

### Text Embeddings

```python
# Option 1: Sentence Transformers (local, free)
from sentence_transformers import SentenceTransformer

models = {
    "fast":    "all-MiniLM-L6-v2",       # 384 dims, very fast
    "balanced": "all-mpnet-base-v2",      # 768 dims, great quality
    "multilingual": "paraphrase-multilingual-mpnet-base-v2",  # 50+ languages
    "code":    "flax-sentence-embeddings/st-codesearch-distilroberta-base",
}

model = SentenceTransformer(models["balanced"])
embedding = model.encode("Hello world")

# Option 2: OpenAI Embeddings (API, paid)
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",  # 1536 dims
    input="Hello world"
)
embedding = response.data[0].embedding

# Option 3: Cohere Embeddings
import cohere
co = cohere.Client("API_KEY")
response = co.embed(texts=["Hello world"], model="embed-english-v3.0")
embedding = response.embeddings[0]
```

### Image Embeddings

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# CLIP: embeds images and text in the same space (cross-modal search!)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed image
image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_embedding = model.get_image_features(**inputs)

# Embed text
text_inputs = processor(text=["a cat", "a dog", "a car"], return_tensors="pt", padding=True)
with torch.no_grad():
    text_embeddings = model.get_text_features(**text_inputs)

# Find which text matches the image best
from torch.nn.functional import cosine_similarity
scores = cosine_similarity(image_embedding, text_embeddings)
print(scores)  # cat should score highest
```

---

## ⚖️ **Vector Database Comparison**

| Feature | Pinecone | ChromaDB | Weaviate | pgvector | Qdrant |
|---------|----------|----------|----------|----------|--------|
| Hosting | Cloud only | Local/Cloud | Local/Cloud | Self-hosted | Local/Cloud |
| Free tier | Yes (1 index) | Yes (unlimited) | Yes | Yes | Yes |
| Filtering | Yes | Yes | Yes | SQL | Yes |
| Hybrid search | Yes | No | Yes | Partial | Yes |
| Scale | Massive | Medium | Large | Medium | Large |
| Best for | Production SaaS | Prototyping | Full-featured | Existing Postgres | Performance |

---

## 📌 **Pinecone**

```python
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid

pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

# Create index
pc.create_index(
    name="knowledge-base",
    dimension=384,           # Must match embedding model output
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("knowledge-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Upsert vectors with metadata
documents = [
    {"text": "Python is a high-level programming language.", "category": "programming"},
    {"text": "TensorFlow is a machine learning framework.", "category": "ml"},
    {"text": "PostgreSQL is a relational database.", "category": "databases"},
]

vectors = []
for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    vectors.append({
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {"text": doc["text"], "category": doc["category"]}
    })

index.upsert(vectors=vectors)
print(f"Index stats: {index.describe_index_stats()}")

# Query with metadata filtering
query = "deep learning frameworks"
query_embedding = model.encode(query).tolist()

results = index.query(
    vector=query_embedding,
    top_k=3,
    filter={"category": {"$in": ["ml", "programming"]}},  # Pre-filter
    include_metadata=True
)

for match in results["matches"]:
    print(f"Score: {match['score']:.3f} | {match['metadata']['text']}")
```

---

## 🌈 **ChromaDB**

Best for local development and prototyping.

```python
import chromadb
from chromadb.utils import embedding_functions

# Local persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Use sentence transformers for embedding (handled automatically)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents (embeddings generated automatically)
collection.add(
    documents=[
        "Neural networks are inspired by the human brain.",
        "Gradient descent minimizes the loss function.",
        "Overfitting occurs when a model memorizes training data.",
        "Cross-validation prevents overfitting.",
        "The transformer architecture uses attention mechanisms.",
    ],
    metadatas=[
        {"topic": "neural_networks", "difficulty": "beginner"},
        {"topic": "optimization", "difficulty": "intermediate"},
        {"topic": "regularization", "difficulty": "beginner"},
        {"topic": "evaluation", "difficulty": "beginner"},
        {"topic": "architectures", "difficulty": "advanced"},
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

# Query
results = collection.query(
    query_texts=["how to prevent model from memorizing?"],
    n_results=2,
    where={"difficulty": "beginner"},  # Metadata filter
)

for doc, meta, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"Distance: {distance:.3f} | Topic: {meta['topic']}")
    print(f"  {doc}\n")
```

---

## 🔷 **Weaviate**

Full-featured with hybrid search and generative search.

```python
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc

# Connect to local instance
client = weaviate.connect_to_local()

# Or connect to Weaviate Cloud
# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url="YOUR_CLUSTER_URL",
#     auth_credentials=Auth.api_key("YOUR_API_KEY"),
# )

# Define collection schema
client.collections.create(
    name="Article",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
    properties=[
        wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
    ]
)

articles = client.collections.get("Article")

# Insert objects
with articles.batch.dynamic() as batch:
    for article in my_articles:
        batch.add_object({
            "title": article["title"],
            "content": article["content"],
            "category": article["category"],
        })

# Hybrid search (dense + sparse BM25)
results = articles.query.hybrid(
    query="machine learning optimization techniques",
    alpha=0.75,  # 0 = pure BM25, 1 = pure vector
    limit=5,
    filters=wvc.query.Filter.by_property("category").equal("tutorial")
)

for obj in results.objects:
    print(obj.properties["title"])

client.close()
```

---

## 🐘 **pgvector (Postgres)**

Add vector search to your existing Postgres database.

```bash
# Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

```python
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg2.connect("postgresql://user:password@localhost/mydb")
cur = conn.cursor()

# Create table with vector column
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        category VARCHAR(100),
        embedding vector(384),
        created_at TIMESTAMP DEFAULT NOW()
    )
""")

# Create HNSW index for fast approximate search
cur.execute("""
    CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")

conn.commit()

# Insert with embeddings
def insert_documents(documents: list[dict]):
    records = []
    for doc in documents:
        embedding = model.encode(doc["content"]).tolist()
        records.append((doc["content"], doc.get("category"), embedding))

    execute_values(cur, """
        INSERT INTO documents (content, category, embedding)
        VALUES %s
    """, records, template="(%s, %s, %s::vector)")
    conn.commit()

# Semantic search with SQL
def semantic_search(query: str, category: str = None, top_k: int = 5) -> list[dict]:
    query_embedding = model.encode(query).tolist()

    sql = """
        SELECT id, content, category,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE (%s IS NULL OR category = %s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    cur.execute(sql, (query_embedding, category, category, query_embedding, top_k))

    return [
        {"id": r[0], "content": r[1], "category": r[2], "similarity": r[3]}
        for r in cur.fetchall()
    ]

results = semantic_search("neural network training tricks", top_k=3)
for r in results:
    print(f"Similarity: {r['similarity']:.3f} | {r['content'][:100]}")
```

---

## 🔬 **Similarity Search Deep Dive**

### Distance Metrics

```python
import numpy as np

def cosine_similarity(a, b):
    """Best for text embeddings — scale invariant."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Good for normalized embeddings, geometric distance."""
    return np.linalg.norm(a - b)

def dot_product(a, b):
    """Used when magnitude matters (e.g., document importance)."""
    return np.dot(a, b)

a = np.array([0.1, 0.9, 0.3])
b = np.array([0.2, 0.8, 0.4])
print(f"Cosine:     {cosine_similarity(a, b):.4f}")
print(f"Euclidean:  {euclidean_distance(a, b):.4f}")
print(f"Dot Product:{dot_product(a, b):.4f}")
```

### HNSW vs FLAT Index

```python
import faiss
import numpy as np
import time

n_vectors = 100_000
dimension = 384
data = np.random.rand(n_vectors, dimension).astype("float32")
query = np.random.rand(1, dimension).astype("float32")

# Flat (exact, slow for large datasets)
flat_index = faiss.IndexFlatL2(dimension)
flat_index.add(data)

start = time.time()
flat_index.search(query, 10)
print(f"Flat search: {(time.time()-start)*1000:.1f}ms")

# HNSW (approximate, much faster)
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
hnsw_index.add(data)

start = time.time()
hnsw_index.search(query, 10)
print(f"HNSW search: {(time.time()-start)*1000:.1f}ms")
# HNSW is 10-100x faster with ~99% accuracy
```

---

## 🏗️ **Real-World Patterns**

### Semantic Deduplication

```python
def deduplicate_documents(docs: list[str], threshold: float = 0.95) -> list[str]:
    """Remove near-duplicate documents."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs)

    unique_docs = [docs[0]]
    unique_embeddings = [embeddings[0]]

    for i in range(1, len(docs)):
        sims = cosine_similarity([embeddings[i]], unique_embeddings)[0]
        if max(sims) < threshold:
            unique_docs.append(docs[i])
            unique_embeddings.append(embeddings[i])

    print(f"Reduced {len(docs)} → {len(unique_docs)} unique documents")
    return unique_docs
```

### Multi-Modal Search (Text → Image)

```python
def search_images_by_text(query_text: str, image_embeddings: np.ndarray,
                           image_paths: list[str], top_k: int = 5):
    """Find images matching a text description using CLIP."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Embed query text
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).numpy()

    # Find similar images
    sims = cosine_similarity(text_embedding, image_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]

    return [(image_paths[i], float(sims[i])) for i in top_indices]
```

---

## 💡 **Tips & Tricks**

1. **Normalize embeddings** before storing: enables faster cosine similarity with dot product
2. **Batch embed**: Always embed in batches (32-256 items) for 10-50x speedup
3. **Dimension reduction**: PCA to 128 dims can reduce storage 3x with minimal accuracy loss
4. **Metadata pre-filtering**: Filter by category/date before vector search to dramatically reduce search space
5. **HNSW params**: `ef_construction=200, m=32` for high-recall; `ef=64, m=16` for speed
6. **Version your embedding models**: Changing models requires re-embedding all data

---

## 🔗 **Related Topics**

- [RAG - Retrieval Augmented Generation](../RAG/README.md)
- [Natural Language Processing](../Natural%20Language%20Processing/Text-Preprocessing.md)
- [MLOps & Deployment](../MLOps/README.md)
