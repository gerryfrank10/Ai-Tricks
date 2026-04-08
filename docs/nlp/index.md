# Natural Language Processing

NLP enables machines to understand, process, and generate human language. In 2025, most NLP tasks are solved with transformer-based models from the Hugging Face Hub.

---

## NLP Task Taxonomy

| Task | Description | Best Models (2025) |
|------|-------------|-------------------|
| Classification | Sentiment, topic, intent | DeBERTa-v3, RoBERTa |
| NER | Named entity recognition | bert-base-NER, Flair |
| Question Answering | Extractive / generative QA | DeBERTa, LLMs |
| Summarization | Abstractive / extractive | BART, Pegasus, LLMs |
| Translation | Cross-lingual | NLLB-200, M2M-100 |
| Text Generation | Open-ended generation | LLaMA, Mistral, Claude |
| Semantic Search | Dense retrieval | all-mpnet, E5, GTE |
| NLI | Textual entailment | DeBERTa-v3-large |

---

## Modern NLP Stack

```python
# Hugging Face — the center of the NLP universe
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Zero-shot classification (no training needed!)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This movie was absolutely fantastic!",
    candidate_labels=["positive", "negative", "neutral"],
)
print(result)  # {'labels': ['positive', ...], 'scores': [0.97, ...]}

# Named Entity Recognition
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
entities = ner("Apple Inc. was founded by Steve Jobs in Cupertino, California.")
# [{'entity_group': 'ORG', 'word': 'Apple Inc.'}, {'entity_group': 'PER', 'word': 'Steve Jobs'}, ...]

# Semantic similarity
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-mpnet-base-v2")
sentences = ["The cat sat on the mat.", "A feline rested on a rug."]
embeddings = model.encode(sentences)
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")   # ~0.85
```

---

## Topics

| Page | Description |
|------|-------------|
| [Text Preprocessing](text-preprocessing.md) | Tokenization, cleaning, vectorization (TF-IDF, embeddings) |
| [Sentiment Analysis](sentiment-analysis.md) | Rule-based, ML, transformer approaches |

---

## spaCy — Industrial-Strength NLP

```python
import spacy

nlp = spacy.load("en_core_web_trf")   # Transformer-based

doc = nlp("The quick brown fox jumps over the lazy dog in New York.")

# Tokens
print([token.text for token in doc])

# Named entities
for ent in doc.ents:
    print(f"{ent.text:20s} → {ent.label_}")

# Dependency parse
for token in doc:
    print(f"{token.text:10s} {token.dep_:10s} ← {token.head.text}")
```

---

## Related Topics

- [LLMs & Agents](../llm/index.md)
- [RAG](../llm/rag.md)
- [Sentiment Analysis](sentiment-analysis.md)
