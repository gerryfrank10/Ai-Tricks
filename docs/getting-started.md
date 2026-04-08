# Getting Started with AI Tricks

Welcome to **AI Tricks** — a practical, code-first reference for everything AI. Think of it as HackTricks, but for Machine Learning, LLMs, Agents, and beyond.

---

## How This Site Is Organized

```
AI Tricks
├── Foundations        — Math, stats, probability, Python
├── Machine Learning   — Algorithms, pipelines, evaluation
├── Deep Learning      — Neural nets, CNNs, RNNs, Transformers
├── LLMs & Agents      — Prompt engineering, RAG, fine-tuning, agents
├── Generative AI      — Diffusion models, multimodal, GANs
├── NLP                — Text processing, sentiment, transformers
├── Computer Vision    — Classification, detection, segmentation
├── Data               — EDA, feature engineering, pipelines
├── MLOps              — Deployment, monitoring, CI/CD
├── Security           — Adversarial attacks, red teaming
├── Optimization       — Gradient descent, genetic algorithms
├── Specialized        — RL, time series, neuronal analysis
├── Industry & Ethics  — Finance, healthcare, retail, ethics
└── Research           — Papers, conferences, reading guides
```

---

## Recommended Learning Paths

=== "New to AI"
    1. [Foundations →](foundations/index.md) — Start with math and Python
    2. [Machine Learning →](machine-learning/index.md) — Learn the core algorithms
    3. [Deep Learning →](deep-learning/index.md) — Understand neural networks
    4. [NLP →](nlp/index.md) — Work with text
    5. [MLOps →](mlops/index.md) — Deploy your first model

=== "Working with LLMs"
    1. [Prompt Engineering →](llm/prompt-engineering.md) — Master prompting
    2. [RAG →](llm/rag.md) — Ground models in your data
    3. [AI Agents →](llm/agents.md) — Build autonomous systems
    4. [Fine-Tuning →](llm/fine-tuning.md) — Customize models
    5. [Vector Databases →](llm/vector-databases.md) — Scale semantic search

=== "Building AI Apps"
    1. [AI Agents →](llm/agents.md) — Agentic architecture
    2. [MLOps →](mlops/index.md) — Productionize
    3. [AI Security →](security/ai-security.md) — Secure your system
    4. [RAG →](llm/rag.md) — Knowledge retrieval
    5. [Cloud Platforms →](mlops/cloud.md) — Deploy at scale

=== "Security / Red Teaming"
    1. [AI Security →](security/ai-security.md) — Attack surfaces
    2. [Prompt Injection →](llm/prompt-engineering.md#prompt-injection--security) — LLM-specific attacks
    3. [AI for Cybersecurity →](security/cybersecurity.md) — Defensive AI

---

## Using Ask AI Docs

Every page has a **🤖 button** in the bottom-right corner.

1. Click it to open the chat panel
2. Enter your [Anthropic API key](https://console.anthropic.com/) when prompted (stored in your browser session only — never sent anywhere except `api.anthropic.com`)
3. Ask anything about the current page — the AI answers with the page content as context

!!! tip "Free tier available"
    Anthropic offers a free tier. The widget uses `claude-haiku-4-5-20251001` by default, which is fast and cheap (~$0.001 per question).

---

## Quick Reference

| Topic | Key Tool | Code Example |
|-------|----------|--------------|
| Train a model | `scikit-learn` | `model.fit(X_train, y_train)` |
| Deep learning | `PyTorch` | `loss.backward(); optimizer.step()` |
| LLM inference | `anthropic` | `client.messages.create(...)` |
| Embeddings | `sentence-transformers` | `model.encode(texts)` |
| Vector search | `chromadb` | `collection.query(query_texts=[...])` |
| Experiment tracking | `mlflow` | `mlflow.log_metric("acc", 0.95)` |
| Fine-tuning | `peft` | `get_peft_model(model, lora_config)` |
| Data pipelines | `pandas` / `polars` | `df.pipe(...).groupby(...)` |

---

## Contributing

1. Fork [gerryfrank10/Ai-Tricks](https://github.com/gerryfrank10/Ai-Tricks)
2. Add or improve a topic in `docs/`
3. Submit a pull request

All contributions welcome — code examples, corrections, new topics.
