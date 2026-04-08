# LLMs & Agents

Large Language Models have fundamentally changed what software can do. This section covers the practical stack for building with LLMs — from prompting to agents to production deployment.

---

## The 2025 LLM Landscape

| Model | Organization | Context | Strengths |
|-------|-------------|---------|-----------|
| Claude Opus 4.6 | Anthropic | 200K | Reasoning, coding, safety |
| Claude Sonnet 4.6 | Anthropic | 200K | Best speed/quality balance |
| GPT-4o | OpenAI | 128K | Multimodal, broad capability |
| Gemini 2.0 Pro | Google | 1M | Long context, multimodal |
| Llama 3.3 70B | Meta | 128K | Open, self-hostable |
| Mistral Large 2 | Mistral | 128K | Multilingual, efficient |
| Qwen 2.5 72B | Alibaba | 128K | Open, strong coding |
| DeepSeek-R1 | DeepSeek | 64K | Open reasoning model |

**Open vs Closed:**
- **Closed APIs** (Claude, GPT, Gemini): easier to start, no infra, per-token cost
- **Open models** (Llama, Mistral, Qwen): self-hostable, one-time cost, full control, privacy

---

## Key Concepts

| Concept | What It Is |
|---------|-----------|
| **Context window** | Max tokens model can process at once |
| **Temperature** | Randomness (0=deterministic, 1=creative) |
| **Top-p / Top-k** | Sampling strategies for generation |
| **System prompt** | Instructions that persist across the conversation |
| **Tool use** | Model calls functions to take actions |
| **RAG** | Retrieve documents to ground responses |
| **Fine-tuning** | Adapt model weights on custom data |
| **Quantization** | Reduce precision (fp32→int4) for efficiency |

---

## Topics in This Section

| Page | Description |
|------|-------------|
| [Prompt Engineering](prompt-engineering.md) | Zero-shot, few-shot, CoT, ReAct, injection defenses |
| [RAG](rag.md) | Retrieval-Augmented Generation pipeline |
| [Fine-Tuning](fine-tuning.md) | LoRA, QLoRA, SFT, dataset prep |
| [AI Agents](agents.md) | Tool use, ReAct loop, multi-agent systems |
| [Vector Databases](vector-databases.md) | Pinecone, ChromaDB, pgvector, Weaviate |

---

## Quick API Reference

```python
import anthropic

client = anthropic.Anthropic()

# Basic message
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)

# Streaming
with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## Related Topics

- [Generative AI](../generative-ai/index.md)
- [MLOps](../mlops/index.md)
- [AI Security](../security/index.md)
