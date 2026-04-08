# How to Read AI Research Papers

Reading papers efficiently is a skill. Most people spend too long on papers that won't matter and not long enough on the ones that do.

---

## The Three-Pass Method

### Pass 1 — Triage (5 minutes)
Decide if the paper is worth reading in depth.

1. **Title & Abstract**: Is the problem relevant to you?
2. **Introduction (last 2 paragraphs)**: What do they claim to contribute?
3. **Section headers**: What's the structure?
4. **Conclusion**: Did they achieve what they claimed?
5. **Figures**: Can you understand the main result from the figures alone?

✅ Proceed if: directly relevant, strong venue, novel claim
❌ Skip if: marginal improvement, irrelevant domain, weak venue

### Pass 2 — Understanding (30-60 minutes)
Read carefully, skip proofs.

1. **Introduction**: Full read — sets up the problem and prior work
2. **Method section**: How does it work? Draw a diagram if needed
3. **Experiments**: What benchmarks? What baselines? What ablations?
4. **Results**: Do the numbers support the claims?
5. **Related work**: Who are the key prior works? (follow these)

### Pass 3 — Critical Reading (2-4 hours, for important papers)
Reproduce the core result or implementation.

1. **Read the proofs** and derivations carefully
2. **Implement the key idea** (even a toy version)
3. **Question the baselines**: Are they fair? Up to date?
4. **Check the appendix**: Often has important details
5. **Read code** (if available on GitHub/Papers With Code)

---

## Key Questions to Ask

```
For any claim:
├── Is the improvement statistically significant?
├── What's the baseline? Is it a fair comparison?
├── Does it generalize beyond the reported benchmarks?
├── What are the compute requirements? (can you reproduce it?)
└── Are there cherry-picked examples or is it consistent?

For architectures:
├── What is the inductive bias? Why would this work?
├── How does it scale with data / model size?
└── What's the ablation study showing the key component?

For new datasets/benchmarks:
├── Does it actually measure what it claims?
├── Is there train/test contamination risk?
└── Does it correlate with real-world performance?
```

---

## ArXiv Workflow

```bash
# Subscribe to ArXiv daily digest for your areas:
# cs.LG  — Machine Learning
# cs.AI  — Artificial Intelligence
# cs.CL  — Computation and Language (NLP)
# cs.CV  — Computer Vision
# cs.RO  — Robotics
# stat.ML — Statistics and Machine Learning

# Tools:
# arxiv-sanity.com — Andrej Karpathy's recommendation engine
# huggingface.co/papers — Community curated + discussion
# paperdigest.org — Auto-generated summaries

# Download and organize papers locally
pip install arxiv

import arxiv

def download_paper(arxiv_id: str, save_dir: str = "papers/"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    paper.download_pdf(dirpath=save_dir, filename=f"{arxiv_id}_{paper.title[:50]}.pdf")
    return f"{paper.title} by {paper.authors[0]}"

# E.g., download "Attention Is All You Need"
result = download_paper("1706.03762")
print(result)
```

---

## Finding Related Work

```python
# Semantic Scholar API — free, AI-powered
import requests

def find_related_papers(title: str, limit: int = 10) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": limit,
        "fields": "title,authors,year,citationCount,tldr,openAccessPdf"
    }
    resp = requests.get(url, params=params)
    papers = resp.json().get("data", [])

    return [{
        "title": p["title"],
        "authors": [a["name"] for a in p["authors"][:3]],
        "year": p["year"],
        "citations": p.get("citationCount", 0),
        "tldr": p.get("tldr", {}).get("text", "N/A"),
        "pdf": p.get("openAccessPdf", {}).get("url", "N/A"),
    } for p in papers]

results = find_related_papers("attention mechanism transformer")
for r in sorted(results, key=lambda x: x["citations"], reverse=True):
    print(f"{r['year']} | {r['citations']:5d} citations | {r['title'][:60]}")
```

---

## Paper Summary Template

Use this when writing notes on papers:

```markdown
## Paper: [Title]
**ArXiv**: [ID] | **Venue**: [NeurIPS/ICML/etc.] **Year**: [Year]
**Authors**: [Author list]

### Problem
[One sentence: what problem does this solve?]

### Key Idea
[2-3 sentences: the core technical contribution]

### Method
[Brief description with the key equation or figure]

### Results
| Benchmark | This Work | Previous SOTA | Gain |
|-----------|-----------|--------------|------|
| ...       | ...       | ...          | ...  |

### Strengths
- [Point 1]
- [Point 2]

### Weaknesses / Limitations
- [Point 1]
- [Point 2]

### Code
[Link to official implementation]

### My Take
[What does this mean for your work? Would you use it?]

### Follow-up Papers to Read
- [Related work 1]
- [Related work 2]
```

---

## Tips & Tricks

| Situation | Action |
|-----------|--------|
| Too many papers to read | Use the 3-pass method; discard after Pass 1 |
| Can't understand the math | Read the intuition first, then formalize |
| Paper claims are unclear | Check the code — implementations don't lie |
| Paper unavailable | Search "[paper title] arxiv" — usually there |
| Want implementation | Check paperswithcode.com first |
| Building a literature review | Use Elicit or Semantic Scholar Connected Papers |

---

## Related Topics

- [Journals & Conferences](journals-conferences.md)
- [Cutting-Edge Research](cutting-edge.md)
