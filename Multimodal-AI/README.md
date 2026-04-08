# Multimodal AI

Multimodal AI processes and generates content across multiple modalities — text, images, audio, and video — simultaneously. Models like GPT-4V, Claude, and Gemini can see, read, and reason across different data types in a single conversation.

---

## 📖 **Sections**

- [Multimodal Architectures](#multimodal-architectures)
- [Vision-Language Models](#vision-language-models)
- [CLIP & Cross-Modal Embeddings](#clip--cross-modal-embeddings)
- [Document Understanding](#document-understanding)
- [Video Understanding](#video-understanding)
- [Audio + Text](#audio--text)
- [Building Multimodal Apps](#building-multimodal-apps)

---

## 🏗️ **Multimodal Architectures**

```
FUSION STRATEGIES:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Early Fusion         Late Fusion    Cross-Attention│
│  ┌──┐ ┌──┐           ┌──┐  ┌──┐    ┌──────────┐   │
│  │I │+│T │→[Model]   │I │  │T │    │ Attn(Q,K,V│  │
│  └──┘ └──┘           └┬─┘  └┬─┘    └──────────┘   │
│  (concat features)    └──+──┘    (cross attend)    │
│                       (merge outputs)               │
└─────────────────────────────────────────────────────┘

Modern VLMs typically use:
Image → [Vision Encoder (ViT)] → Patch Embeddings
                                       ↓
Text Tokens ────────────────→ [LLM + Cross-Attention] → Output
```

---

## 👁️ **Vision-Language Models**

### Using Claude's Vision

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 for API."""
    path = Path(image_path)
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_types.get(path.suffix.lower(), "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type

# Image analysis
def analyze_image(image_path: str, question: str) -> str:
    image_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }]
    )
    return response.content[0].text

# Example uses
description = analyze_image("product.jpg", "Describe this product in detail for an e-commerce listing.")
print(description)

chart_analysis = analyze_image("sales_chart.png",
    "Analyze this chart. What are the key trends? Any anomalies?")
print(chart_analysis)

code_from_image = analyze_image("whiteboard.jpg",
    "Extract and format the code/algorithm written on this whiteboard.")
print(code_from_image)
```

### Multi-Image Analysis

```python
def compare_images(image_paths: list[str], comparison_prompt: str) -> str:
    """Analyze and compare multiple images."""
    content = []

    for i, path in enumerate(image_paths):
        image_data, media_type = encode_image(path)
        content.append({
            "type": "text",
            "text": f"Image {i+1} ({Path(path).name}):"
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_data}
        })

    content.append({"type": "text", "text": comparison_prompt})

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text

# Compare product designs, before/after images, etc.
result = compare_images(
    ["design_v1.png", "design_v2.png"],
    "Compare these two UI designs. What changed? Which is more user-friendly and why?"
)
```

### Vision with Tool Use (Agentic)

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "extract_table_data",
        "description": "Extract structured data from a table in an image",
        "input_schema": {
            "type": "object",
            "properties": {
                "headers": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array", "items": {"type": "array"}},
                "caption": {"type": "string"}
            },
            "required": ["headers", "rows"]
        }
    },
    {
        "name": "detect_objects",
        "description": "Identify objects in an image with their approximate positions",
        "input_schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "confidence": {"type": "number"},
                            "position": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
]

def structured_image_analysis(image_path: str) -> dict:
    """Extract structured data from image using tool use."""
    image_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        tools=tools,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data}
                },
                {
                    "type": "text",
                    "text": "Analyze this image. If it contains a table, extract the data. If it contains objects, detect them."
                }
            ]
        }]
    )

    # Process tool calls
    results = {}
    for block in response.content:
        if block.type == "tool_use":
            results[block.name] = block.input

    return results
```

---

## 🎯 **CLIP & Cross-Modal Embeddings**

CLIP (Contrastive Language-Image Pre-training) embeds images and text in the same vector space.

```python
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

# Zero-shot image classification
def classify_image(image_path: str, candidate_labels: list[str]) -> dict:
    """Classify image without any training examples."""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Compute probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]

    return {label: float(prob) for label, prob in zip(candidate_labels, probs)}

# Zero-shot classification — no training needed!
labels = ["a cat", "a dog", "a bird", "a car", "a house"]
results = classify_image("mystery.jpg", labels)

for label, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{label:20s}: {prob:.2%}")

# Image-text retrieval
def find_matching_images(query_text: str, image_paths: list[str], top_k: int = 3):
    """Find images that best match a text description."""
    images = [Image.open(p).convert("RGB") for p in image_paths]

    inputs = processor(
        text=[query_text],
        images=images,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    similarities = outputs.logits_per_text[0]
    top_indices = similarities.topk(top_k).indices.tolist()

    return [(image_paths[i], float(similarities[i])) for i in top_indices]

matches = find_matching_images(
    "a sunset over the ocean",
    ["photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg"],
    top_k=2
)
for path, score in matches:
    print(f"Score: {score:.3f} | {path}")
```

---

## 📄 **Document Understanding**

### PDF/Document Analysis Pipeline

```python
import anthropic
import base64
from pdf2image import convert_from_path
import io

client = anthropic.Anthropic()

def analyze_pdf(pdf_path: str, questions: list[str]) -> dict:
    """Extract insights from PDF by converting pages to images."""
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=200)
    print(f"Processing {len(pages)} pages...")

    content = []

    # Add all pages
    for i, page in enumerate(pages[:10]):  # Limit to first 10 pages
        img_buffer = io.BytesIO()
        page.save(img_buffer, format="PNG")
        img_data = base64.standard_b64encode(img_buffer.getvalue()).decode()

        content.append({"type": "text", "text": f"--- Page {i+1} ---"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_data}
        })

    # Add questions
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    content.append({
        "type": "text",
        "text": f"Please answer these questions based on the document:\n{questions_text}"
    })

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    return {
        "analysis": response.content[0].text,
        "pages_processed": len(pages)
    }

# Usage
result = analyze_pdf("contract.pdf", [
    "What are the key obligations of each party?",
    "What are the payment terms?",
    "What are the termination conditions?",
    "Are there any unusual clauses I should flag?"
])
print(result["analysis"])
```

### Form Data Extraction

```python
def extract_form_data(image_path: str) -> dict:
    """Extract key-value pairs from a form image."""
    image_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data}
                },
                {
                    "type": "text",
                    "text": """Extract all form fields and their values from this form image.
Return as JSON with field names as keys and entered values as values.
For unchecked checkboxes use false, checked use true.
For empty fields use null.
Return ONLY valid JSON."""
                }
            ]
        }]
    )

    import json
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"raw": response.content[0].text}
```

---

## 🎬 **Video Understanding**

```python
import cv2
import base64
from pathlib import Path

def sample_video_frames(video_path: str, n_frames: int = 10) -> list[str]:
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    frame_indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    frames_b64 = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.standard_b64encode(buffer).decode()
            frames_b64.append(b64)

    cap.release()
    print(f"Video: {duration:.1f}s, sampled {len(frames_b64)} frames")
    return frames_b64

def analyze_video(video_path: str, question: str) -> str:
    """Analyze video by sampling frames and asking questions."""
    frames = sample_video_frames(video_path, n_frames=10)

    content = [{"type": "text", "text": f"The following are {len(frames)} frames sampled from a video:"}]

    for i, frame_b64 in enumerate(frames):
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}
        })

    content.append({"type": "text", "text": f"Based on these video frames: {question}"})

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text

# Usage
summary = analyze_video("tutorial.mp4",
    "Summarize what happens in this video. What is being demonstrated?")
print(summary)
```

---

## 🔊 **Audio + Text**

### Speech-to-Text with Whisper

```python
import whisper
import torch

model = whisper.load_model("large-v3")  # Options: tiny, base, small, medium, large

def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """Transcribe audio with timestamps."""
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False
    )

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]
    }

# Transcribe and then analyze with LLM
transcript = transcribe_audio("meeting.mp3")

# Now analyze the transcript
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""Analyze this meeting transcript:

{transcript['text']}

Provide:
1. Key decisions made
2. Action items with owners
3. Main topics discussed
4. Sentiment/tone of the meeting"""
    }]
)
print(response.content[0].text)
```

---

## 🏗️ **Building Multimodal Apps**

### Multimodal RAG Pipeline

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class MultimodalRAG:
    """RAG system that handles both text and image queries."""

    def __init__(self):
        self.text_client = anthropic.Anthropic()
        self.db = chromadb.PersistentClient("./multimodal_db")

        # Separate collections for text and images
        self.text_collection = self.db.get_or_create_collection(
            "text_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-mpnet-base-v2")
        )

    def add_document(self, text: str, doc_id: str, metadata: dict = None):
        self.text_collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata or {}]
        )

    def query(self, user_message: str, image_path: str = None) -> str:
        # Retrieve relevant context
        text_results = self.text_collection.query(
            query_texts=[user_message],
            n_results=3
        )
        context = "\n\n".join(text_results["documents"][0])

        # Build multimodal prompt
        content = []

        if image_path:
            image_data, media_type = encode_image(image_path)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_data}
            })

        content.append({
            "type": "text",
            "text": f"""Answer based on the provided context and image (if any).

CONTEXT:
{context}

QUESTION: {user_message}"""
        })

        response = self.text_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text
```

---

## 💡 **Tips & Tricks**

1. **Image resolution matters**: Send images at the resolution you actually need — larger = more tokens = more cost
2. **Guide attention with text**: Tell the model exactly what part of the image to focus on
3. **Chain modalities**: Transcribe audio → analyze transcript → generate summary — each step uses the best model for that task
4. **CLIP for zero-shot classification**: Cheaper and faster than full VLM for simple classification
5. **Frame sampling for video**: 1 frame per 2-5 seconds is usually sufficient for understanding; sample more for action-dense content
6. **Cache visual features**: If asking multiple questions about the same image, structure as a multi-turn conversation

---

## 🔗 **Related Topics**

- [Computer Vision](../Computer%20Vision/Image-Classification.md)
- [Generative AI & Diffusion Models](../Generative-AI/README.md)
- [Natural Language Processing](../Natural%20Language%20Processing/Text-Preprocessing.md)
- [LLM Agents](../LLM/Agents.md)
