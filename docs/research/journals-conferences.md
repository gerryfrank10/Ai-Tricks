# AI Journals & Conferences

Knowing where research is published tells you how to weigh it. Tier-1 venues have rigorous peer review; preprints are faster but unvetted.

---

## Conference Tier List

### Tier 1 — Top Venues (most competitive)

| Venue | Full Name | Focus | Typical Deadline |
|-------|-----------|-------|-----------------|
| **NeurIPS** | Conference on Neural Information Processing Systems | General ML/DL | May |
| **ICML** | International Conference on Machine Learning | ML theory + applications | January |
| **ICLR** | International Conference on Learning Representations | Deep learning | October |
| **ACL** | Annual Meeting of the ACL | NLP | February |
| **CVPR** | Conference on Computer Vision and Pattern Recognition | Vision | November |
| **EMNLP** | Empirical Methods in NLP | NLP | June |

### Tier 2 — Strong Venues

| Venue | Focus |
|-------|-------|
| ECCV / ICCV | Computer vision (alternating years) |
| NAACL | NLP (North American) |
| AAAI | General AI |
| UAI | Probabilistic inference |
| AISTATS | Statistics + ML |
| KDD | Data mining + applied ML |
| SIGIR | Information retrieval |
| COLING | Computational linguistics |

### Workshops
Many influential papers first appear at workshops (NeurIPS workshops, ICML workshops). Don't dismiss them.

---

## Journals

| Journal | Focus | Impact |
|---------|-------|--------|
| Journal of Machine Learning Research (JMLR) | Broad ML | Open access, high quality |
| Transactions on Machine Learning Research (TMLR) | Broad ML | Rolling review (no deadlines) |
| Nature Machine Intelligence | Applied ML + science | High impact, broad audience |
| IEEE TPAMI | Vision + pattern recognition | Established, rigorous |
| Artificial Intelligence Journal | General AI | Oldest AI journal |

---

## Preprint Servers

```
ArXiv (cs.LG, cs.AI, cs.CL, cs.CV, stat.ML)
├── Pros: Immediate, free, indexed
├── Cons: Not peer-reviewed — evaluate critically
└── Tip: Look for papers that later appear at top venues

OpenReview (openreview.net)
├── Pros: Peer review is PUBLIC — you can read reviewer scores
├── Houses: ICLR, NeurIPS, TMLR reviews
└── Tip: Filter by "Accept" to find accepted papers pre-conference
```

---

## Landmark Papers (2017-2025)

### Foundations
| Year | Paper | Contribution |
|------|-------|-------------|
| 2017 | Attention Is All You Need | Transformer architecture |
| 2018 | BERT | Bidirectional pre-training for NLP |
| 2020 | GPT-3 | Scale + few-shot learning |
| 2020 | DDPM | Modern diffusion models |
| 2021 | CLIP | Cross-modal image-text embeddings |
| 2021 | AlphaFold 2 | Protein structure prediction |
| 2022 | InstructGPT | RLHF for instruction following |
| 2022 | ChatGPT | (Technical report, Dec 2022) |

### 2023-2024 Breakthroughs
| Year | Paper | Contribution |
|------|-------|-------------|
| 2023 | LLaMA (1 + 2) | Open foundation models |
| 2023 | Segment Anything (SAM) | Zero-shot segmentation |
| 2023 | Mamba | State space models as Transformer alternative |
| 2023 | Mixtral | Mixture of Experts for open models |
| 2024 | Llama 3 | 70B open model matching GPT-4 |
| 2024 | SAM 2 | Segmentation for images + video |
| 2024 | DeepSeek-R1 | Open reasoning model (o1-level) |
| 2024 | Gemini 1.5 Pro | 1M context window |
| 2024 | Flux.1 | State-of-the-art image generation |
| 2024 | π₀ (pi-zero) | Robot foundation model |

### 2025 Highlights
| Paper | Contribution |
|-------|-------------|
| Gemini 2.0 | Native multimodal agents |
| Claude 4.x | Extended context, computer use |
| GPT-4.5 / o3 | Reasoning + tool use |
| Llama 4 | Multimodal open model |
| Qwen 2.5-Max | Open model competitive with GPT-4 |

---

## How to Evaluate a Paper's Credibility

```python
# Quick credibility checklist
def evaluate_paper_credibility(paper: dict) -> dict:
    """
    paper = {
        "venue": "NeurIPS 2024",
        "citations": 847,
        "has_code": True,
        "replicated_by_others": True,
        "industry_adoption": True,
    }
    """
    score = 0
    reasons = []

    venue_scores = {
        "NeurIPS": 3, "ICML": 3, "ICLR": 3, "CVPR": 3,
        "ACL": 3, "EMNLP": 2, "AAAI": 2, "KDD": 2,
        "arxiv": 0, "workshop": 1,
    }
    for v, s in venue_scores.items():
        if v.lower() in paper["venue"].lower():
            score += s
            reasons.append(f"+{s}: Published at {v}")
            break

    if paper["citations"] > 100:
        score += 2
        reasons.append(f"+2: {paper['citations']} citations")
    if paper["has_code"]:
        score += 1
        reasons.append("+1: Official code available")
    if paper.get("replicated_by_others"):
        score += 2
        reasons.append("+2: Independently replicated")
    if paper.get("industry_adoption"):
        score += 2
        reasons.append("+2: Industry adoption")

    return {
        "score": score,
        "max_score": 10,
        "credibility": "High" if score >= 7 else "Medium" if score >= 4 else "Low",
        "reasons": reasons
    }
```

---

## Related Topics

- [Reading Papers](reading-papers.md)
- [Cutting-Edge Research](cutting-edge.md)
