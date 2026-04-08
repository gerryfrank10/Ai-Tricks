# Cutting-Edge AI Research (2024-2025)

A curated overview of the most important recent developments. Use this as a compass for where the field is heading.

---

## Major Themes (2024-2025)

### 1. Test-Time Compute Scaling
The insight that letting models "think longer" at inference time dramatically improves reasoning — even on tasks where larger models plateau.

**Key papers:**
- **DeepSeek-R1** (2025) — Open reasoning model using GRPO (no supervised data needed for RL)
- **OpenAI o1/o3** (2024-2025) — Chain-of-thought reasoning scaled at test time
- **s1: Simple Test-Time Scaling** (Stanford, 2025) — 1000 reasoning examples + budget forcing = o1-competitive

```python
# The core idea: force the model to think before answering
prompt = """Think through this step by step, considering multiple approaches
before giving your final answer. Use <thinking> tags for your reasoning.

Problem: {problem}
"""
# Models trained with RL on verifiable rewards (math, code) learn to use
# longer thinking chains to solve harder problems
```

### 2. Long-Context Models
Context windows have grown from 4K (2020) to 1M+ tokens (2024).

| Model | Context | Notes |
|-------|---------|-------|
| Claude 3.5 Sonnet | 200K | Best retrieval accuracy in long context |
| Gemini 1.5 Pro | 1M | 1hr video, entire codebases |
| GPT-4o | 128K | General purpose |
| Llama 3.1 | 128K | Open, self-hostable |

**Research insight:** Needle-in-a-haystack (NIAH) tests show most models lose information in the "lost in the middle" problem — content at the beginning and end is retrieved better than the middle.

### 3. Mixture of Experts (MoE)
Activate only a fraction of parameters per token — scaling model capacity without proportional compute cost.

```
Dense model: ALL 70B parameters activated for every token
MoE model:   8 experts, 2 activated → effectively 70B capacity, 14B compute
```

**Key models:** Mixtral 8×7B, Mixtral 8×22B, DeepSeek-V3 (671B params, 37B active), GPT-4o (rumored MoE)

### 4. Multimodal Foundation Models
Models that natively handle text, images, audio, and video in a single architecture.

- **GPT-4o** — "omni" model, native audio/video
- **Gemini 2.0** — Deep research, code execution, real-time audio/video
- **Claude 3.x** — Vision + extended context
- **Llama 3.2** — First open multimodal LLM (11B, 90B)

### 5. Autonomous Agents & Computer Use
LLMs that can browse the web, write and execute code, and control computers.

- **Claude Computer Use** (Anthropic, 2024) — Direct GUI control
- **OpenAI Operator** (2025) — Web browsing agents
- **Devin / SWE-Agent** — Autonomous software engineering
- **WebVoyager, WebArena** — Benchmarks for web agents

---

## Key Papers to Read in 2025

### LLMs
```
1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (2025)
   Key insight: Pure RL (GRPO) from base model beats SFT → RL pipeline
   ArXiv: 2501.12948

2. Scaling LLM Test-Time Compute Optimally (2024, Google DeepMind)
   Key insight: How to allocate test-time compute for best performance
   ArXiv: 2408.03314

3. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (2024)
   Key insight: Ternary weights {-1, 0, 1} with minimal quality loss
   ArXiv: 2402.17764

4. Transformer²: Self-Adaptive LLMs (2025)
   Key insight: Singular value decomposition for dynamic weight adaption
   ArXiv: 2501.06252
```

### Agents & Reasoning
```
5. SWE-bench Verified (2024) — Software engineering benchmark
   State of play: Claude 3.5 Sonnet ~49%, vs ~2% in early 2024
   ArXiv: 2310.06770

6. AgentBench: Evaluating LLMs as Agents (2023, updated 2024)
   ArXiv: 2308.03688
```

### Efficiency
```
7. FlashAttention-3 (2024) — 1.5-2x faster attention on H100s
   ArXiv: 2407.08608

8. MobileLLM (2024, Meta) — Sub-1B models for mobile deployment
   ArXiv: 2402.14905
```

### Vision & Multimodal
```
9. SAM 2: Segment Anything in Images and Videos (2024, Meta)
   ArXiv: 2408.00714

10. Depth Anything V2 (2024) — State-of-the-art monocular depth
    ArXiv: 2406.09414

11. FLUX.1 [dev] (2024, Black Forest Labs)
    Technical report: blackforestlabs.ai
```

---

## Benchmark Leaderboards to Watch

```python
# Track model progress on key benchmarks
BENCHMARKS = {
    "MMLU": "General knowledge (57 subjects)",
    "HumanEval": "Code generation (Python functions)",
    "SWE-bench": "Real GitHub issue resolution",
    "MATH": "Competition mathematics",
    "GPQA": "Expert-level science questions",
    "MMMU": "Multimodal understanding",
    "LongBench": "Long-context understanding",
    "ArenaHard": "Hard instruction following (vs GPT-4)",
}

# Top leaderboards:
# chatbot-arena.lmsys.org    — Human preference votes
# huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
# paperswithcode.com/sota     — Per-benchmark SOTA
```

---

## Emerging Topics to Watch

| Topic | Why It Matters | Status |
|-------|---------------|--------|
| **Embodied AI** | Robots that learn from human video | Active (π₀, GROOT, RT-2) |
| **World Models** | Models that simulate physics | Early (DIAMOND, Genie 2) |
| **Mechanistic Interpretability** | Understanding what's inside LLMs | Growing (Anthropic, DeepMind) |
| **Synthetic Data** | LLMs generating training data | Scaling now (Phi-4, s1) |
| **Continual Learning** | Learning without forgetting | Research stage |
| **Neuromorphic Computing** | Brain-inspired hardware | Early commercial (Intel Loihi) |
| **Speculative Decoding** | 2-3x inference speedup | Deployed (Medusa, EAGLE) |

---

## How to Stay Current

```python
weekly_routine = [
    "Monday: ArXiv cs.LG + cs.CL weekend papers",
    "Tuesday: Hugging Face Papers Daily digest",
    "Wednesday: Implement 1 key idea from a recent paper",
    "Thursday: Read comments/discussion on OpenReview",
    "Friday: Check benchmark leaderboards for updates",
]

tools = {
    "Daily alerts": "arxiv-sanity.com or RSS feed for cs.LG",
    "Paper discussion": "huggingface.co/papers (community comments)",
    "Implementations": "paperswithcode.com",
    "Industry blogs": [
        "anthropic.com/research",
        "ai.meta.com/research",
        "deepmind.google/research",
        "openai.com/research",
    ]
}
```

---

## Related Topics

- [Reading Papers](reading-papers.md)
- [Journals & Conferences](journals-conferences.md)
- [Deep Learning](../deep-learning/index.md)
- [LLMs & Agents](../llm/index.md)
