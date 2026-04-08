# Transfer Learning & Foundation Models

Transfer learning repurposes a model trained on a large task for a smaller, related one. In 2025, this means fine-tuning foundation models — pretrained on trillions of tokens or billions of images.

---

## Why Transfer Learning Works

```
Pretrained on 1B images          Fine-tune on 1000 medical images
━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1: edges, corners    →    Still useful (low-level features)
Layer 2: textures, shapes  →    Still useful
Layer 3: objects, parts    →    Partially useful
Layer 4: high-level repr   →    Needs adapting for medical domain
Output: ImageNet classes   →    Replace with tumor/no-tumor
```

Low-level features are **universal**. High-level features are **task-specific**.

---

## Feature Extraction vs Fine-Tuning

| Strategy | Frozen Layers | When to Use | Data Needed |
|----------|--------------|------------|-------------|
| Feature extraction | All | Small dataset (<1000), similar domain | <500 |
| Partial fine-tune | First N layers | Medium dataset, somewhat different | 1k–10k |
| Full fine-tune | None | Large dataset or very different domain | >10k |
| Linear probe | All (train only head) | Evaluation, quick baselines | Any |

---

## torchvision Pretrained Models

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

# ── Strategy 1: Feature Extraction (freeze backbone) ──────────────
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Freeze ALL parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer only
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Only model.fc is trainable now

# ── Strategy 2: Partial Fine-Tuning ───────────────────────────────
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Freeze early layers, unfreeze later ones
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True    # Unfreeze last block + head
    else:
        param.requires_grad = False

# ── Strategy 3: Full Fine-Tuning with Differential LR ─────────────
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Use lower LR for pretrained layers, higher for new head
optimizer = torch.optim.AdamW([
    {"params": model.layer1.parameters(), "lr": 1e-5},
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 2e-5},
    {"params": model.layer4.parameters(), "lr": 5e-5},
    {"params": model.fc.parameters(),     "lr": 1e-3},  # New head gets high LR
], weight_decay=0.01)
```

### Vision Transformer (ViT)

```python
# ViT — currently state-of-the-art for image classification
model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# ViT head is model.heads.head
in_features = model.heads.head.in_features
model.heads.head = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes)
)

# ViT preprocessing — MUST use specific transforms
transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
```

---

## Hugging Face Transformers — Text Fine-Tuning

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load pretrained model + tokenizer
model_name = "microsoft/deberta-v3-base"   # Top performer on NLP benchmarks
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare dataset
dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.rename_column("label", "labels")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }

# Training arguments — optimized for text classification
args = TrainingArguments(
    output_dir="./deberta-imdb",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
```

---

## PEFT — Parameter-Efficient Fine-Tuning

Only train a tiny fraction of weights. Critical for large models.

```python
from peft import (
    LoraConfig, get_peft_model,
    TaskType, IA3Config,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,               # Rank
    lora_alpha=32,      # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 7,283,556,352 || trainable%: 0.58%

# IA³ — even fewer parameters (only 0.01%)
ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
```

---

## Domain Adaptation

Fine-tune on domain text before task-specific fine-tuning (two-stage).

```python
from transformers import DataCollatorForLanguageModeling

# Stage 1: Continue pretraining on domain corpus
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

domain_corpus = load_dataset("text", data_files={"train": "legal_texts.txt"})
domain_corpus = domain_corpus.map(tokenize, batched=True)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

stage1_args = TrainingArguments(
    output_dir="./llama-legal-pretrain",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,     # Higher LR for continual pretraining
    num_train_epochs=1,
    bf16=True,
    max_steps=10000,
)

trainer = Trainer(
    model=model,
    args=stage1_args,
    train_dataset=domain_corpus["train"],
    data_collator=collator,
)
trainer.train()

# Stage 2: Fine-tune adapted model on task
# Load the domain-adapted checkpoint and fine-tune on labeled data
```

---

## Few-Shot Learning

```python
# Few-shot with frozen model + prompt engineering (no gradient)
import anthropic

client = anthropic.Anthropic()

def few_shot_classify(text: str, examples: list[dict]) -> str:
    """Use few-shot prompting as lightweight transfer learning."""
    examples_str = "\n".join(
        f"Text: {ex['text']}\nLabel: {ex['label']}" for ex in examples
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"{examples_str}\n\nText: {text}\nLabel:"
        }]
    )
    return response.content[0].text.strip()

# Only 3 examples needed vs hundreds for fine-tuning
examples = [
    {"text": "The drug significantly reduced tumor size", "label": "positive"},
    {"text": "Adverse effects were observed in 40% of patients", "label": "negative"},
    {"text": "No statistically significant difference was found", "label": "neutral"},
]
result = few_shot_classify("Treatment led to complete remission in 78% of cases", examples)
```

---

## Tips & Tricks

| Situation | Best Approach |
|-----------|--------------|
| < 500 labeled samples | Feature extraction + linear probe |
| 500–5000 samples | LoRA fine-tuning |
| > 5000 samples | Full fine-tuning with differential LR |
| Very different domain | Two-stage: domain adapt → task fine-tune |
| Limited GPU | QLoRA (4-bit quantization + LoRA) |
| Catastrophic forgetting | Use smaller LR, regularize with EWC |
| Evaluation during FT | Monitor original benchmark (don't regress!) |

---

## Related Topics

- [Fine-Tuning LLMs](../../llm/fine-tuning.md)
- [PEFT & QLoRA](../../llm/fine-tuning.md)
- [Computer Vision](../../computer-vision/index.md)
- [Deep Learning Index](index.md)
