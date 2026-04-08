# Fine-Tuning LLMs

Fine-tuning adapts a pre-trained model to a specific task or domain by continuing training on custom data. It's the bridge between a general-purpose model and a specialist that speaks your domain's language.

---

## 📖 **Sections**

- [When to Fine-Tune vs RAG](#when-to-fine-tune-vs-rag)
- [Fine-Tuning Methods](#fine-tuning-methods)
- [Data Preparation](#data-preparation)
- [LoRA / QLoRA Training](#lora--qlora-training)
- [Full Fine-Tuning](#full-fine-tuning)
- [Evaluation](#evaluation)
- [Serving Fine-Tuned Models](#serving-fine-tuned-models)

---

## 🤔 **When to Fine-Tune vs RAG**

| Factor | Fine-Tuning | RAG |
|--------|-------------|-----|
| Knowledge updates | Requires retraining | Real-time |
| Latency | Fast (no retrieval) | Slower (retrieval step) |
| Style/tone adaptation | Excellent | Limited |
| Factual grounding | Hallucination risk | Source-grounded |
| Data needed | 100s–1000s examples | Just documents |
| Cost | High (training) | Low (inference only) |
| Best for | Behavior, format, style | Knowledge, facts, search |

**Rule of thumb**: Use RAG for knowledge, fine-tuning for behavior.

---

## 🔧 **Fine-Tuning Methods**

### Full Fine-Tuning
Update all model weights. Maximum performance but requires significant GPU memory.

### LoRA (Low-Rank Adaptation)
Inject small trainable matrices into attention layers. 10-100x fewer trainable params.

```
Original: W (frozen)
LoRA:     W + A×B (A and B are small trainable matrices)
```

### QLoRA
LoRA + 4-bit quantization. Fine-tune 70B models on a single 48GB GPU.

### PEFT (Parameter-Efficient Fine-Tuning)
Umbrella term for LoRA, prefix tuning, prompt tuning, adapters, etc.

---

## 📦 **Data Preparation**

### Dataset Format (Instruction Tuning)

```python
import json

# Standard instruction-following format
training_data = [
    {
        "instruction": "Classify the sentiment of this review.",
        "input": "The product quality is outstanding but delivery was slow.",
        "output": "Mixed - Positive sentiment about product quality, negative about delivery."
    },
    {
        "instruction": "Summarize the following legal clause in plain English.",
        "input": "The licensee shall indemnify and hold harmless the licensor...",
        "output": "You agree to protect us from any legal costs or damages caused by your use of the software."
    }
]

# Save as JSONL
with open("train.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")
```

### Chat Format (for instruction models)

```python
# Alpaca / ChatML format
chat_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a medical triage assistant. Provide guidance based on symptoms described."},
            {"role": "user", "content": "Patient reports fever of 103°F, severe headache, and stiff neck."},
            {"role": "assistant", "content": "These symptoms — high fever, severe headache, and neck stiffness — are classic signs of meningitis. This is a medical emergency. Go to the emergency room immediately. Do not wait."}
        ]
    }
]
```

### Data Quality Checklist

```python
import re
from collections import Counter

def audit_dataset(data: list[dict]) -> dict:
    """Check dataset quality before fine-tuning."""
    issues = []

    # Check lengths
    output_lengths = [len(d.get("output", "").split()) for d in data]

    if min(output_lengths) < 5:
        issues.append(f"WARNING: {sum(l < 5 for l in output_lengths)} examples have very short outputs")

    # Check for duplicates
    outputs = [d.get("output", "") for d in data]
    dupes = len(outputs) - len(set(outputs))
    if dupes > 0:
        issues.append(f"WARNING: {dupes} duplicate outputs found")

    # Check for empty fields
    empty_inputs = sum(1 for d in data if not d.get("input", "").strip())

    stats = {
        "total_examples": len(data),
        "avg_output_length": sum(output_lengths) / len(output_lengths),
        "min_output_length": min(output_lengths),
        "max_output_length": max(output_lengths),
        "duplicates": dupes,
        "empty_inputs": empty_inputs,
        "issues": issues
    }

    return stats

stats = audit_dataset(training_data)
print(json.dumps(stats, indent=2))
```

---

## 🔥 **LoRA / QLoRA Training**

### Setup

```bash
pip install transformers peft bitsandbytes datasets accelerate trl
```

### QLoRA Fine-Tuning with TRL

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    r=16,              # Rank — higher = more capacity, more params
    lora_alpha=32,     # Scaling factor (usually 2x rank)
    target_modules=[   # Which layers to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 20,971,520 || all params: 3,773,063,168 || trainable%: 0.5559

# Load and format dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    formatting_func=format_instruction,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model("./fine-tuned-model/final")
```

### Merge LoRA Weights (for deployment)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model (in full precision for merging)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load and merge LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "./fine-tuned-model/final")
merged_model = peft_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
print("Model merged and saved!")
```

---

## 🏋️ **Full Fine-Tuning**

For smaller models (< 3B params) where you have sufficient GPU memory.

```python
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

# For sequence-to-sequence models (T5, BART)
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def preprocess(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=True)
    labels = tokenizer(examples["highlights"], max_length=256, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./t5-summarizer",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

trainer.train()
```

---

## 📊 **Evaluation**

### Perplexity (Language Modeling Quality)

```python
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text: str) -> float:
    """Lower perplexity = model is less surprised by the text = better fit."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    return math.exp(loss.item())

# Compare base vs fine-tuned
base_ppl = calculate_perplexity(base_model, tokenizer, domain_text_sample)
finetuned_ppl = calculate_perplexity(finetuned_model, tokenizer, domain_text_sample)
print(f"Base model perplexity: {base_ppl:.2f}")
print(f"Fine-tuned perplexity: {finetuned_ppl:.2f}")
```

### Task-Specific Metrics

```python
from evaluate import load

# ROUGE for summarization
rouge = load("rouge")
results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
print(results)  # {'rouge1': 0.45, 'rouge2': 0.22, 'rougeL': 0.38}

# BERTScore for semantic similarity
bertscore = load("bertscore")
results = bertscore.compute(
    predictions=outputs,
    references=targets,
    lang="en",
    model_type="distilbert-base-uncased"
)
print(f"BERTScore F1: {sum(results['f1']) / len(results['f1']):.4f}")
```

---

## 🚢 **Serving Fine-Tuned Models**

### vLLM (High-Throughput Inference)

```bash
pip install vllm

# Start OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model ./merged-model \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="./merged-model",
    messages=[{"role": "user", "content": "Summarize this contract..."}]
)
print(response.choices[0].message.content)
```

### Ollama (Local Deployment)

```bash
# Convert to GGUF format first
pip install llama-cpp-python

# Create Modelfile
cat > Modelfile << EOF
FROM ./merged-model-q4.gguf
SYSTEM "You are a specialized legal document analyzer."
PARAMETER temperature 0.1
EOF

ollama create my-legal-model -f Modelfile
ollama run my-legal-model "Analyze this contract clause..."
```

---

## 💡 **Tips & Tricks**

1. **Start with LoRA rank 8-16**: Increase only if performance plateaus
2. **Learning rate**: 1e-4 to 3e-4 for LoRA, 1e-5 to 5e-5 for full fine-tuning
3. **Watch for catastrophic forgetting**: Validate on general benchmarks, not just your task
4. **Data quality > quantity**: 500 high-quality examples beats 10,000 noisy ones
5. **Use gradient checkpointing**: Trades compute for memory (`model.gradient_checkpointing_enable()`)
6. **Early stopping**: Monitor validation loss; stop when it starts increasing
7. **Flash Attention 2**: Significant speedup for long context (`attn_implementation="flash_attention_2"`)

---

## 🔗 **Related Topics**

- [RAG - Retrieval Augmented Generation](../RAG/README.md)
- [LLM Agents](../LLM/Agents.md)
- [MLOps & Deployment](../MLOps/README.md)
- [Deep Learning](../Deep%20Learning/Introduction.md)
