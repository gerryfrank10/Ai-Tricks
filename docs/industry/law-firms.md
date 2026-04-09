# AI in Law Firms

AI is reshaping legal practice at every layer — from first-pass contract review taking minutes instead of days, to RAG-powered research assistants that surface the right precedent in seconds. This guide covers production-grade techniques, real tools, and working code for deploying AI in legal environments in 2025.

---

## Overview

The legal industry is one of the highest-stakes, highest-margin sectors adopting AI. Law firms generate enormous volumes of unstructured text — briefs, contracts, discovery documents, memos — making them ideal candidates for NLP and LLM solutions.

| Metric | Figure |
|--------|--------|
| Global legal AI market size (2025) | ~$1.2B, growing at 30% CAGR |
| Time saved on contract review with AI | 60–85% reduction |
| E-discovery cost reduction | Up to 70% |
| Associates using AI tools (Am Law 100) | >75% as of 2025 |
| Hallucination-related case sanctions | 10+ reported incidents in 2023–2024 |

Key transformation areas:

- **Research acceleration** — surfacing case law in seconds vs. hours
- **Contract intelligence** — automated clause extraction and risk scoring
- **Discovery efficiency** — semantic search over millions of documents
- **Drafting augmentation** — LLM-generated first drafts from templates and facts
- **Client access** — 24/7 intake bots and status chatbots

---

## Key AI Use Cases

### Legal Research

Traditional legal research (Westlaw, Lexis) relies on Boolean keyword search. AI upgrades this with semantic retrieval, citation graph analysis, and LLM-generated research memos.

**What AI does:**

- Retrieves relevant cases, statutes, and secondary sources via embedding similarity
- Summarizes holdings and identifies contradictory precedents
- Generates structured research memos with citations
- Tracks treatment history (how subsequent courts cited a case)

**Technical approach:** RAG (Retrieval-Augmented Generation) over a legal corpus. Embeddings are created for case law chunks; user queries are embedded and matched via cosine similarity; retrieved passages are sent to an LLM for synthesis.

**Example tools:** Harvey AI, Westlaw Precision, LexisNexis+ AI, Casetext CoCounsel

---

### Contract Analysis & Review

Contract review is the highest-ROI AI use case in law. AI tools can read hundreds of pages of contracts, extract defined terms, flag non-standard clauses, and compare against playbooks in minutes.

**What AI does:**

- Clause extraction (indemnification, limitation of liability, termination, IP ownership)
- Deviation detection against standard playbooks
- Risk scoring per clause
- Side-by-side comparison of multiple contract versions
- Missing clause detection

**Technical approach:** Fine-tuned or prompted LLMs with structured output (JSON extraction). Clause classification with BERT-based models. RAG against clause libraries.

**Example tools:** Kira Systems, Ironclad, Spellbook, Harvey AI

---

### Document Drafting & Generation

AI generates first drafts of routine legal documents — NDAs, engagement letters, demand letters, motions — based on a set of facts and a template library.

**What AI does:**

- Populates templates with party names, dates, deal terms
- Generates bespoke clause language from natural-language instructions
- Adapts jurisdiction-specific language
- Suggests alternative formulations

**Technical approach:** Prompt chaining with structured fact extraction followed by template-conditioned LLM generation. Few-shot examples from firm's own document history improve output quality significantly.

**Example tools:** Spellbook (for contracts in Word), Harvey AI, GPT-4 via API, Claude API

---

### E-Discovery

E-discovery involves reviewing tens of thousands to millions of documents for relevance, privilege, and responsiveness to litigation requests. AI dramatically reduces the review burden.

**What AI does:**

- Technology-Assisted Review (TAR) — active learning to prioritize documents
- Predictive coding — trains on attorney-coded seed set, predicts relevance
- Privilege log generation — identifies attorney-client privileged documents
- Near-duplicate detection
- Concept clustering

**Technical approach:** Embedding-based similarity search (FAISS or pgvector), active learning loops, LLM-based privilege analysis.

**Example tools:** Relativity (RelativityOne), Reveal AI, Logikcull, Exterro

---

### Due Diligence Automation

M&A due diligence requires reviewing data rooms with thousands of documents across corporate, IP, employment, and regulatory categories. AI compresses weeks of work into hours.

**What AI does:**

- Auto-categorizes data room documents by type
- Extracts key terms from all contracts simultaneously
- Flags change of control provisions, assignment restrictions, and expiry dates
- Generates diligence summary reports by category

**Technical approach:** Multi-document RAG with structured extraction. Each document is chunked, embedded, and stored with metadata (document type, party names, dates). A diligence agent queries across the corpus.

**Example tools:** Kira Systems, Harvey AI, Luminance, Litera

---

### Billing & Time Tracking

Lawyers are notoriously inconsistent at recording time. AI reconstructs and narrates billable time from emails, documents, and calendar events.

**What AI does:**

- Infers billable activities from email threads, document edits, and calendar
- Suggests time entries with narrative descriptions in proper billing format
- Flags potential write-downs or write-offs
- Detects billing guideline violations

**Technical approach:** Activity classification from metadata, LLM-generated billing narratives, rules-based compliance checking against client billing guidelines.

**Example tools:** Clocktimizer, BillBlast, BigHand, TimeSolv AI

---

### Client Intake & Chatbots

Law firms use AI chatbots to handle first contact, screen for conflicts, collect matter information, and answer basic FAQs — freeing attorneys for substantive work.

**What AI does:**

- Collects party names and adverse parties for conflict checks
- Qualifies matters (area of law, urgency, estimated value)
- Answers FAQ about firm services, fee structures, process
- Routes to the right practice group
- Schedules consultations

**Technical approach:** Conversational agents with structured extraction. Conflict check integration via API. RAG over firm FAQs and practice area content.

**Example tools:** Clio Grow, Josef, Gideon, custom Claude/GPT agents

---

## Top AI Tools & Platforms

| Tool | Provider | Primary Use Case | Pricing Tier |
|------|----------|-----------------|--------------|
| Harvey AI | Harvey | Legal research, drafting, Q&A | Enterprise (custom) |
| CoCounsel (Casetext) | Thomson Reuters | Research, contract review, deposition prep | $100–$500/mo per user |
| Westlaw Precision | Thomson Reuters | Case law search with AI filters | Subscription (firm deal) |
| LexisNexis+ AI | LexisNexis | Research, summarization, drafting | Subscription (firm deal) |
| Ironclad | Ironclad | Contract lifecycle management | $500–$2,000+/mo |
| Relativity (RelativityOne) | Relativity | E-discovery, document review | Per-GB or per-user |
| Kira Systems | Litera | Contract analysis, due diligence | Enterprise (custom) |
| Spellbook | Rally | Contract drafting in Word | $99–$299/mo per user |
| Claude API | Anthropic | Custom legal AI apps, RAG, drafting | $3–$15 per million tokens |
| GPT-4o | OpenAI | General legal tasks via API or ChatGPT | $2.50–$10 per million tokens |
| Luminance | Luminance | Due diligence, contract review | Enterprise (custom) |
| Logikcull | Logikcull | E-discovery, TAR | Per-GB ingestion |

---

## Technology Stack

### RAG on Legal Databases

The foundational architecture for legal AI: embed a corpus of case law or firm documents, store in a vector database, retrieve relevant chunks at query time, and synthesize with an LLM.

```python
# Legal RAG system using LangChain + Claude
# Requirements: anthropic, langchain, langchain-anthropic,
#               langchain-community, faiss-cpu, tiktoken

import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import json

# ── 1. Ingest legal documents ──────────────────────────────────────────────

def load_legal_corpus(file_paths: List[str]) -> List[Document]:
    """Load case law, statutes, or firm memos into LangChain Documents."""
    docs = []
    for path in file_paths:
        with open(path, "r") as f:
            text = f.read()
        # Attach metadata (citation, jurisdiction, date) parsed from filename
        # In production: parse PACER XML, Westlaw exports, or LexisNexis bulk
        docs.append(Document(
            page_content=text,
            metadata={"source": path, "jurisdiction": "federal"}
        ))
    return docs

def build_legal_index(docs: List[Document]) -> FAISS:
    """Chunk documents and build FAISS vector index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],  # respect sentence boundaries
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")

    # Use a lightweight embedding model — swap for text-embedding-3-large
    # or voyage-law-2 (Voyage AI's legal-optimized model) in production
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local("legal_index")
    return index

# ── 2. Retrieve + Synthesize with Claude ───────────────────────────────────

def legal_research_query(
    question: str,
    index: FAISS,
    client: anthropic.Anthropic,
    top_k: int = 6,
) -> dict:
    """
    Run a legal research query:
    1. Embed the question
    2. Retrieve top-k relevant chunks
    3. Ask Claude to synthesize a research memo
    """
    # Retrieve relevant passages
    retrieved = index.similarity_search_with_score(question, k=top_k)

    # Build context block with citations
    context_blocks = []
    for i, (doc, score) in enumerate(retrieved):
        source = doc.metadata.get("source", "Unknown")
        context_blocks.append(
            f"[Source {i+1}: {source} | Relevance: {1 - score:.2f}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    # System prompt for legal research memo
    system_prompt = """You are a senior legal research attorney. Given retrieved
legal sources, produce a concise research memo that:
1. Directly answers the legal question
2. Identifies the controlling rule or standard
3. Cites specific sources (use [Source N] references)
4. Notes any circuit splits, contrary authority, or open questions
5. Flags if the retrieved sources are insufficient to answer confidently

Always hedge appropriately — this is research, not legal advice."""

    # Query Claude
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Legal Question: {question}\n\n"
                    f"Retrieved Sources:\n\n{context}\n\n"
                    f"Please produce a research memo."
                ),
            }
        ],
    )

    return {
        "question": question,
        "memo": response.content[0].text,
        "sources_used": len(retrieved),
        "top_source": retrieved[0][0].metadata.get("source") if retrieved else None,
    }


# ── Example usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    # Build index from your legal corpus (replace with real file paths)
    docs = load_legal_corpus(["cases/smith_v_jones.txt", "statutes/ucc_2.txt"])
    index = build_legal_index(docs)

    result = legal_research_query(
        question=(
            "What is the standard for preliminary injunctions in the "
            "Ninth Circuit for trade secret misappropriation claims?"
        ),
        index=index,
        client=client,
    )

    print("=== LEGAL RESEARCH MEMO ===")
    print(result["memo"])
    print(f"\nBased on {result['sources_used']} retrieved sources")
```

---

### Contract Review with NLP

Extract structured data from contracts, classify clauses, and flag non-standard terms using Claude's structured output capabilities.

```python
# Contract clause extraction and risk flagging with Claude
# Requirements: anthropic, pydantic

import anthropic
import json
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

client = anthropic.Anthropic()

# ── Data models ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedClause(BaseModel):
    clause_type: str           # e.g. "Indemnification", "Limitation of Liability"
    text: str                  # verbatim clause text
    risk_level: RiskLevel
    risk_rationale: str        # why this is flagged
    is_standard: bool          # matches firm playbook?
    recommended_action: str    # "Accept" | "Negotiate" | "Reject" | "Escalate"

class ContractReviewResult(BaseModel):
    contract_title: str
    governing_law: Optional[str]
    effective_date: Optional[str]
    parties: List[str]
    clauses: List[ExtractedClause]
    overall_risk: RiskLevel
    executive_summary: str

# ── Clause extraction prompt ─────────────────────────────────────────────────

REVIEW_SYSTEM_PROMPT = """You are a senior contract attorney specializing in
commercial agreements. Review the provided contract and extract key clauses.

For each clause, assess:
- Whether it deviates from market-standard terms
- Risk to our client (the party receiving this contract)
- Recommended negotiation position

Focus on: Indemnification, Limitation of Liability, IP Ownership,
Termination Rights, Change of Control, Non-Compete/Non-Solicit,
Payment Terms, Governing Law, Dispute Resolution, Confidentiality.

Return your analysis as valid JSON matching the ContractReviewResult schema."""

def review_contract(contract_text: str, client_party: str = "our client") -> ContractReviewResult:
    """
    Extract clauses and risk-score a contract using Claude.

    Args:
        contract_text: Full contract text
        client_party: Name of the party we represent

    Returns:
        Structured ContractReviewResult with all extracted clauses
    """
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=REVIEW_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Please review this contract. Our client is: {client_party}\n\n"
                    f"CONTRACT:\n{contract_text}\n\n"
                    f"Return your analysis as JSON matching this schema:\n"
                    f"{ContractReviewResult.model_json_schema()}"
                ),
            }
        ],
    )

    raw = response.content[0].text

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = ContractReviewResult.model_validate_json(raw)
    return result


def format_review_report(result: ContractReviewResult) -> str:
    """Format extracted clauses into a human-readable review report."""
    lines = [
        f"CONTRACT REVIEW REPORT",
        f"=" * 50,
        f"Contract:       {result.contract_title}",
        f"Parties:        {', '.join(result.parties)}",
        f"Governing Law:  {result.governing_law or 'Not specified'}",
        f"Effective Date: {result.effective_date or 'Not specified'}",
        f"Overall Risk:   {result.overall_risk.upper()}",
        f"",
        f"EXECUTIVE SUMMARY",
        f"-" * 30,
        result.executive_summary,
        f"",
        f"CLAUSE ANALYSIS",
        f"-" * 30,
    ]

    # Group by risk level — critical first
    risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
    for risk in risk_order:
        flagged = [c for c in result.clauses if c.risk_level == risk]
        if not flagged:
            continue
        lines.append(f"\n[{risk.upper()} RISK]")
        for clause in flagged:
            lines.extend([
                f"  Clause Type:  {clause.clause_type}",
                f"  Standard:     {'Yes' if clause.is_standard else 'NO - DEVIATION'}",
                f"  Action:       {clause.recommended_action}",
                f"  Rationale:    {clause.risk_rationale}",
                f"  Text:         {clause.text[:200]}...",
                "",
            ])

    return "\n".join(lines)


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_contract = """
    SOFTWARE LICENSE AGREEMENT
    This Agreement is between Acme Corp ("Licensor") and Beta Inc ("Licensee").

    1. INDEMNIFICATION. Licensee shall indemnify, defend, and hold harmless
    Licensor and its officers, directors, employees from any and all claims,
    damages, losses, costs, including attorneys' fees arising out of or related
    to Licensee's use of the Software, including claims by third parties.

    2. LIMITATION OF LIABILITY. IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY
    INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES. LICENSOR'S TOTAL
    LIABILITY SHALL NOT EXCEED $100.

    3. IP OWNERSHIP. All modifications, improvements, or derivative works created
    by Licensee using the Software shall be the sole property of Licensor.

    4. GOVERNING LAW. This Agreement shall be governed by the laws of Delaware.
    """

    result = review_contract(sample_contract, client_party="Beta Inc (Licensee)")
    print(format_review_report(result))
```

---

### E-Discovery Semantic Search

Find conceptually relevant documents across a massive document set without exact keyword matches — a critical capability for modern discovery.

```python
# E-discovery semantic search with embeddings
# Requirements: anthropic, numpy, faiss-cpu, sentence-transformers

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Tuple
import json

@dataclass
class DiscoveryDocument:
    doc_id: str
    filename: str
    text: str
    custodian: str
    date: str
    metadata: dict = field(default_factory=dict)
    is_privileged: bool = False

@dataclass
class SearchResult:
    doc: DiscoveryDocument
    score: float
    matched_concepts: List[str]

class EDiscoverySearchEngine:
    """
    Semantic search engine for e-discovery document review.

    Uses dense embeddings to find conceptually relevant documents
    even when they don't share exact keywords with the query.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        # In production: use voyage-2, OpenAI text-embedding-3-large,
        # or a legal-domain fine-tuned model
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.documents: List[DiscoveryDocument] = []
        self.embeddings: np.ndarray | None = None

    def ingest(self, documents: List[DiscoveryDocument]) -> None:
        """Embed all documents and build FAISS index."""
        self.documents = documents
        texts = [f"{doc.filename}\n{doc.text}" for doc in documents]

        print(f"Embedding {len(texts)} documents...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # for cosine similarity via dot product
        )
        self.embeddings = embeddings.astype(np.float32)

        # Build flat inner-product index (= cosine sim on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"Index built. {self.index.ntotal} documents indexed.")

    def search(
        self,
        query: str,
        top_k: int = 20,
        exclude_privileged: bool = True,
    ) -> List[SearchResult]:
        """
        Search for documents relevant to a legal concept or issue.

        Args:
            query: Natural language query (e.g. "communications about pricing strategy")
            top_k: Number of results to return
            exclude_privileged: Skip attorney-client privileged documents

        Returns:
            Ranked list of SearchResult objects
        """
        if self.index is None:
            raise RuntimeError("Call ingest() before search()")

        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Retrieve more to account for filtered results
        fetch_k = top_k * 3 if exclude_privileged else top_k
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            if exclude_privileged and doc.is_privileged:
                continue
            results.append(SearchResult(
                doc=doc,
                score=float(score),
                matched_concepts=[query],  # extend with concept expansion in prod
            ))
            if len(results) >= top_k:
                break

        return results

    def find_near_duplicates(
        self,
        threshold: float = 0.95,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify near-duplicate documents for deduplication.
        Returns list of (doc_id_1, doc_id_2, similarity_score) tuples.
        """
        if self.embeddings is None:
            raise RuntimeError("Call ingest() first")

        duplicates = []
        n = len(self.documents)
        # Batch pairwise search — for large sets use approximate methods
        scores_matrix = np.dot(self.embeddings, self.embeddings.T)

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(scores_matrix[i, j])
                if sim >= threshold:
                    duplicates.append((
                        self.documents[i].doc_id,
                        self.documents[j].doc_id,
                        sim,
                    ))

        return sorted(duplicates, key=lambda x: -x[2])

    def generate_privilege_log(
        self,
        privileged_docs: List[DiscoveryDocument],
    ) -> str:
        """Generate a privilege log entry for each privileged document."""
        lines = ["PRIVILEGE LOG", "=" * 60]
        for doc in privileged_docs:
            lines.extend([
                f"Document ID:   {doc.doc_id}",
                f"Date:          {doc.date}",
                f"Custodian:     {doc.custodian}",
                f"Filename:      {doc.filename}",
                f"Basis:         Attorney-Client Privilege / Work Product",
                f"Description:   [Attorney review required]",
                "-" * 40,
            ])
        return "\n".join(lines)


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = EDiscoverySearchEngine()

    # Simulate a document corpus
    sample_docs = [
        DiscoveryDocument(
            doc_id="DOC001",
            filename="email_pricing_meeting.eml",
            text="Let's discuss the new pricing strategy before the board meeting next week.",
            custodian="john.smith@acme.com",
            date="2024-03-15",
        ),
        DiscoveryDocument(
            doc_id="DOC002",
            filename="memo_legal_advice.docx",
            text="Per our attorney's advice regarding the antitrust exposure...",
            custodian="jane.doe@acme.com",
            date="2024-03-16",
            is_privileged=True,
        ),
        DiscoveryDocument(
            doc_id="DOC003",
            filename="slides_market_analysis.pptx",
            text="Competitive pricing analysis shows we can increase margins by 15% in Q3.",
            custodian="cfo@acme.com",
            date="2024-03-17",
        ),
    ]

    engine.ingest(sample_docs)

    results = engine.search("pricing strategy and competitive analysis", top_k=5)
    print(f"\nTop results for pricing query:")
    for r in results:
        print(f"  [{r.score:.3f}] {r.doc.filename} ({r.doc.custodian})")
```

---

## Best Workflow

End-to-end AI-augmented matter workflow from client intake to final billing:

```
                     AI-AUGMENTED LEGAL MATTER WORKFLOW
                     ====================================

  ┌──────────────────────────────────────────────────────────────────┐
  │  CLIENT INTAKE                                                   │
  │  ▸ AI chatbot collects: party names, matter type, urgency        │
  │  ▸ Automated conflict check against client/adverse party DB      │
  │  ▸ AI routes to correct practice group                           │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  LEGAL RESEARCH                                                  │
  │  ▸ RAG search over Westlaw / LexisNexis / firm knowledge base    │
  │  ▸ LLM synthesizes research memo with citations                  │
  │  ▸ Attorney validates memo and identifies gaps                   │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  DOCUMENT DRAFTING                                               │
  │  ▸ LLM generates first draft from template + matter facts        │
  │  ▸ Jurisdiction-specific language inserted automatically         │
  │  ▸ Attorney reviews, edits, and approves                         │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  CONTRACT / DOCUMENT REVIEW                                      │
  │  ▸ AI extracts all key clauses and risk scores them              │
  │  ▸ Deviations from playbook flagged for negotiation              │
  │  ▸ E-discovery: semantic search surfaces relevant documents      │
  │  ▸ Near-duplicate detection + privilege screening                │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  CLIENT DELIVERY                                                 │
  │  ▸ AI-generated summary of key findings and recommendations      │
  │  ▸ Structured report with risk matrix                            │
  │  ▸ Client-facing chatbot for status queries                      │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  BILLING & CLOSURE                                               │
  │  ▸ AI reconstructs time entries from activity logs               │
  │  ▸ Billing narratives generated in proper format                 │
  │  ▸ Guideline compliance check against client billing rules       │
  │  ▸ Invoice generated and delivered                               │
  └──────────────────────────────────────────────────────────────────┘

  AI TOUCHPOINTS:  Intake  Research  Drafting  Review  Summary  Billing
                     ●        ●         ●        ●        ●        ●
```

---

## Building Your Own Legal AI Assistant

A production-ready streaming RAG-based legal Q&A bot using Claude API. This serves as the foundation for a law firm knowledge base assistant.

```python
# Streaming legal Q&A assistant with RAG + Claude
# Full implementation with conversation history and source citations
# Requirements: anthropic, faiss-cpu, sentence-transformers, rich

import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Generator
import textwrap

# ── Configuration ─────────────────────────────────────────────────────────────

LEGAL_ASSISTANT_SYSTEM = """You are a legal research assistant for a law firm.
You help attorneys find relevant case law, statutes, and legal principles.

Rules you ALWAYS follow:
1. Base your answers ONLY on the retrieved sources provided. Do not invent cases.
2. Always include citations like [Source N: case name/document].
3. If sources are insufficient, say so explicitly — never hallucinate precedent.
4. Distinguish between controlling authority and persuasive authority.
5. Note if an area of law is evolving or unsettled.
6. This is legal research, not legal advice to a client.
7. If asked about current events after your knowledge cutoff, say so.

Format: Start with a direct answer, then supporting analysis, then citations."""

@dataclass
class LegalDocument:
    doc_id: str
    title: str
    text: str
    citation: str          # e.g., "Smith v. Jones, 123 F.3d 456 (9th Cir. 2023)"
    doc_type: str          # "case", "statute", "regulation", "memo"
    jurisdiction: str

class LegalAssistant:
    """
    Streaming RAG-based legal Q&A assistant.

    Maintains conversation history for multi-turn research sessions.
    """

    def __init__(self, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.documents: List[LegalDocument] = []
        self.index: faiss.IndexFlatIP | None = None
        self.conversation_history: List[dict] = []

    def load_documents(self, docs: List[LegalDocument]) -> None:
        """Embed and index a legal corpus."""
        self.documents = docs
        texts = [f"{d.title}\n{d.citation}\n{d.text}" for d in docs]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Loaded {len(docs)} legal documents into knowledge base.")

    def _retrieve(self, query: str, k: int = 5) -> List[tuple[LegalDocument, float]]:
        """Retrieve top-k relevant documents for the query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_vec, k)
        return [
            (self.documents[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def _build_context(self, retrieved: List[tuple[LegalDocument, float]]) -> str:
        """Format retrieved documents as numbered context block."""
        blocks = []
        for i, (doc, score) in enumerate(retrieved, 1):
            blocks.append(
                f"[Source {i}: {doc.title}]\n"
                f"Citation: {doc.citation}\n"
                f"Type: {doc.doc_type} | Jurisdiction: {doc.jurisdiction}\n"
                f"Relevance: {score:.2f}\n\n"
                f"{doc.text[:800]}"  # truncate for context window
            )
        return "\n\n" + ("─" * 40 + "\n\n").join(blocks)

    def ask(self, question: str, stream: bool = True) -> Generator[str, None, None] | str:
        """
        Ask a legal question. Returns streamed response or full string.

        Maintains conversation history for follow-up questions.
        """
        # Retrieve relevant documents
        retrieved = self._retrieve(question, k=5)
        context = self._build_context(retrieved) if retrieved else "No documents loaded."

        # Build user message with retrieved context
        user_message = (
            f"RETRIEVED LEGAL SOURCES:\n{context}\n\n"
            f"LEGAL QUESTION: {question}"
        )

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        if stream:
            return self._stream_response()
        else:
            return self._full_response()

    def _stream_response(self) -> Generator[str, None, None]:
        """Stream the assistant response token by token."""
        full_response = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=LEGAL_ASSISTANT_SYSTEM,
            messages=self.conversation_history,
        ) as stream:
            for text_chunk in stream.text_stream:
                full_response += text_chunk
                yield text_chunk

        # Store the complete response in history for multi-turn context
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

    def _full_response(self) -> str:
        """Get complete response (non-streaming)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=LEGAL_ASSISTANT_SYSTEM,
            messages=self.conversation_history,
        )
        answer = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
        })
        return answer

    def reset_conversation(self) -> None:
        """Clear conversation history to start a new research session."""
        self.conversation_history = []
        print("Conversation history cleared.")


# ── Interactive CLI session ────────────────────────────────────────────────────

def run_interactive_session():
    """Run an interactive legal research session in the terminal."""
    assistant = LegalAssistant()

    # Load sample legal corpus (replace with real documents)
    sample_docs = [
        LegalDocument(
            doc_id="1",
            title="Winter v. Natural Resources Defense Council",
            text=(
                "A plaintiff seeking a preliminary injunction must establish that "
                "he is likely to succeed on the merits, that he is likely to suffer "
                "irreparable harm in the absence of preliminary relief, that the "
                "balance of equities tips in his favor, and that an injunction is "
                "in the public interest."
            ),
            citation="555 U.S. 7 (2008)",
            doc_type="case",
            jurisdiction="U.S. Supreme Court",
        ),
        LegalDocument(
            doc_id="2",
            title="eBay Inc. v. MercExchange, L.L.C.",
            text=(
                "A plaintiff must demonstrate: (1) that it has suffered an "
                "irreparable injury; (2) that remedies available at law are "
                "inadequate; (3) that, considering the balance of hardships, "
                "a remedy in equity is warranted; and (4) that the public interest "
                "would not be disserved by a permanent injunction."
            ),
            citation="547 U.S. 388 (2006)",
            doc_type="case",
            jurisdiction="U.S. Supreme Court",
        ),
    ]
    assistant.load_documents(sample_docs)

    print("\n" + "=" * 60)
    print("  LEGAL RESEARCH ASSISTANT  (powered by Claude)")
    print("  Type 'quit' to exit | 'reset' to clear history")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            assistant.reset_conversation()
            continue

        print("\nAssistant: ", end="", flush=True)
        for chunk in assistant.ask(question, stream=True):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    run_interactive_session()
```

---

## ROI & Metrics

Measured time savings and cost reductions reported by Am Law 100 firms and legal tech vendors in 2024–2025:

| Task | Manual Time | AI-Assisted Time | Time Saved | Cost Reduction |
|------|-------------|-----------------|------------|----------------|
| NDA review (20 pages) | 2–3 hours | 10–20 minutes | ~85% | 80–85% |
| Due diligence (100-doc data room) | 3–5 days | 4–8 hours | ~80% | 70–80% |
| Legal research memo (1 issue) | 4–6 hours | 30–60 minutes | ~85% | 75–85% |
| Contract abstraction (50 leases) | 2–3 weeks | 1–2 days | ~85% | 80% |
| E-discovery review (100k docs) | 6–8 weeks | 1–2 weeks | ~75% | 60–70% |
| First draft NDA | 1–2 hours | 5–10 minutes | ~90% | 85–90% |
| Billing narrative generation | 30–60 min/day | 5 min/day | ~90% | N/A |
| Deposition prep (1 witness) | 8–12 hours | 2–3 hours | ~70% | 65–70% |

**Economic impact at a 50-attorney firm (illustrative):**

```
Annual associate billing rate:        $400/hr × 1,800 hrs = $720,000/attorney
AI tools reduce non-billable prep:    ~200 hrs/attorney/year recaptured
Revenue recovered per attorney:       $80,000/year
50-attorney firm total:               $4,000,000/year additional capacity
AI tooling cost (all tools):          ~$150,000/year
Net ROI:                              ~26x
```

---

## Compliance & Ethics

### Confidentiality & Data Security

Legal AI deployments face unique confidentiality obligations that consumer AI tools do not address by default.

**Key risks:**

- Sending client documents to third-party LLM APIs may violate attorney-client privilege
- Training data retention policies of AI vendors may expose confidential information
- Cloud-hosted AI tools require data processing agreements (DPAs)

**Mitigations:**

- Use enterprise API tiers with zero data retention (Anthropic, OpenAI both offer this)
- Deploy on-premise or in private VPC for highly sensitive matters
- Implement client consent frameworks for AI-assisted work
- Audit all data flows — know exactly what leaves your perimeter

### Attorney-Client Privilege

AI-generated work product can still be privileged, but the analysis is fact-specific.

- Privilege attaches to attorney's use of AI to assist legal work, not to AI output itself
- Counsel must review and exercise independent judgment — AI does not substitute
- Metadata from AI tools (prompts, retrieved documents) may be discoverable
- Some courts have required disclosure that AI was used in brief preparation

### Hallucination Risk

**This is the highest-risk issue in legal AI.** LLMs can fabricate case citations that appear real but do not exist.

Notable incidents:

- Mata v. Avianca (S.D.N.Y. 2023): Attorneys sanctioned for submitting AI-hallucinated case citations
- Multiple subsequent sanctions in 2024 across federal and state courts

**Prevention checklist:**

- [ ] Always verify every case citation in Westlaw or Lexis before filing
- [ ] Use RAG over verified legal databases — do not rely on LLM parametric memory for citations
- [ ] Implement a citation verification step (automated Westlaw/Nexis API check)
- [ ] Brief all attorneys on hallucination risk before AI tool deployment
- [ ] Never use AI output directly in court filings without attorney verification

### Human Oversight Requirements

| Activity | AI Role | Required Human Oversight |
|----------|---------|--------------------------|
| Legal research | First-pass retrieval and synthesis | Attorney must verify all cited authority |
| Contract review | Clause extraction and risk flagging | Attorney must validate each flag |
| Document drafting | First draft generation | Attorney edits and approves final |
| E-discovery review | Relevance ranking and privilege screening | Attorney review of privileged calls |
| Court filings | Drafting assistance only | Attorney reviews every word before filing |
| Client advice | None — AI must not advise clients | All advice from licensed attorney |

### Model Selection Considerations

| Factor | Consideration |
|--------|--------------|
| Data residency | Where are prompts and responses stored? |
| Training opt-out | Does the vendor use your data to train? |
| SOC 2 Type II | Is the vendor certified? |
| BAA/DPA | Can you sign a data processing agreement? |
| On-premise option | Can you self-host the model? |
| Audit logs | Are all queries logged for review? |

---

## Tips & Tricks

| Tip | Category | Detail |
|-----|----------|--------|
| Use voyage-law-2 embeddings | Retrieval | Voyage AI's legal-domain embedding model outperforms general models on legal retrieval benchmarks |
| Chunk by paragraph, not fixed tokens | Indexing | Legal text has natural paragraph boundaries — respect them to avoid splitting holdings mid-sentence |
| Add citation metadata to every chunk | Indexing | Store case name, court, year as chunk metadata — retrieve the full citation without re-reading the document |
| System prompt the playbook | Contract review | Embed your firm's standard positions in the system prompt for consistent risk scoring |
| Two-pass review | Contract review | First pass: extract all clauses. Second pass: score each against playbook. Fewer tokens per call, better accuracy |
| Always verify citations with Westlaw API | Hallucination prevention | Automate a post-generation step that checks every case name against Westlaw's citation API |
| Use streaming for long documents | UX | Stream Claude responses so attorneys see output immediately rather than waiting 30+ seconds |
| Jurisdiction-specific prompting | Drafting | Always include the governing law jurisdiction in the system prompt — Delaware vs. NY vs. California contract law differ significantly |
| Privilege screen before sending to API | Confidentiality | Run a keyword/regex pre-filter to catch obvious privilege markers before any document goes to an external API |
| Implement human-in-the-loop gates | Ethics | Require attorney approval at defined checkpoints — never let AI output flow directly to client or court |
| Log all queries and responses | Audit | Maintain immutable logs of all AI interactions on client matters for conflict checks and malpractice defense |
| Evaluate on your own documents | Accuracy | Before deploying any AI tool firm-wide, run it on 50–100 documents you've already manually reviewed and measure accuracy |

---

## Related Topics

- [RAG (Retrieval-Augmented Generation)](../llm/rag.md) — the core architecture powering legal research AI
- [LLM Agents](../llm/agents.md) — building multi-step legal AI workflows
- [Vector Databases](../llm/vector-databases.md) — storing and retrieving legal document embeddings
- [NLP: Text Preprocessing](../nlp/text-preprocessing.md) — cleaning legal text for ML pipelines
- [AI Ethics](ethics.md) — fairness, accountability, and transparency in AI systems
- [AI in Finance](finance.md) — adjacent regulated industry with similar compliance challenges
- [AI in Healthcare](healthcare.md) — another high-stakes regulated domain
- [AI Security](../security/ai-security.md) — protecting sensitive client data in AI systems
- [Prompt Engineering](../llm/prompt-engineering.md) — crafting effective prompts for legal tasks
