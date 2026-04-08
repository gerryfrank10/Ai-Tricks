# Image Classification

Image classification is the task of assigning a label to an entire image. It's the "hello world" of computer vision and the foundation for more complex tasks.

---

## Quick Start — Fine-Tune ResNet in 30 Lines

```python
import torch
import torch.nn as nn
from torchvision import models, datasets
from torchvision.transforms import v2 as T
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
train_tfm = T.Compose([
    T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tfm = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Dataset (ImageFolder structure: root/class_name/img.jpg)
train_ds = datasets.ImageFolder("data/train", transform=train_tfm)
val_ds   = datasets.ImageFolder("data/val",   transform=val_tfm)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=8)

# Model — ResNet50 with pretrained weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(device)
model = torch.compile(model)    # PyTorch 2.x speed boost

# Optimizer + Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=20, steps_per_epoch=len(train_loader)
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            pred = model(images.to(device)).argmax(1)
            correct += (pred.cpu() == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch+1:2d} | Val Acc: {correct/total:.4f}")
```

---

## Architecture Comparison (2025)

| Model | Top-1 (ImageNet) | Params | Speed (A100) | Best For |
|-------|-----------------|--------|-------------|---------|
| ViT-B/16 | 81.9% | 86M | 165 img/s | Transfer learning |
| ViT-L/16 | 85.2% | 307M | 55 img/s | High accuracy |
| ConvNeXt V2-B | 87.5% | 89M | 200 img/s | Balanced |
| EfficientNet V2-M | 85.5% | 54M | 280 img/s | Edge/mobile |
| ResNet-50 | 80.9% | 25M | 500 img/s | Baseline |
| DINOv2-ViT-L | 86.3% | 307M | 55 img/s | Self-supervised features |

---

## Vision Transformer (ViT) — Modern Standard

```python
import timm
import torch

# ViT-B/16 with register tokens (state-of-the-art 2024)
model = timm.create_model(
    "vit_base_patch16_reg4_gap_256",  # register tokens reduce artifacts
    pretrained=True,
    num_classes=10,
)

# DINOv2 — self-supervised features, excellent without fine-tuning
model = timm.create_model("vit_large_patch14_dinov2", pretrained=True, num_classes=0)

# Get transforms automatically
from timm.data import create_transform, resolve_model_data_config

config = resolve_model_data_config(model)
transform = create_transform(**config, is_training=False)

# Feature extraction for downstream tasks
model.eval()
with torch.no_grad():
    features = model(transform(image).unsqueeze(0))   # (1, 1024)
```

---

## Data Augmentation Strategies

```python
from torchvision.transforms import v2 as T
import torch

# Standard (good baseline)
standard = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Strong (TrivialAugment — AutoAugment without search cost)
strong = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
])

# MixUp (crucial for >85% accuracy)
from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice
mixup = RandomChoice([MixUp(num_classes=N, alpha=0.2), CutMix(num_classes=N, alpha=1.0)])

# In training loop:
images, labels = mixup(images, labels)
# labels is now soft (one-hot weighted) — cross_entropy handles this automatically
```

---

## Model Evaluation & Diagnostics

```python
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, loader, class_names, device="cuda"):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = logits.argmax(1).cpu()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()

    # Classification report
    print(classification_report(labels, preds, target_names=class_names))

    # Top-5 accuracy
    top5 = sum(
        labels[i] in np.argsort(probs[i])[-5:]
        for i in range(len(labels))
    ) / len(labels)
    print(f"Top-5 Accuracy: {top5:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, ax=ax, cmap="Blues")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    return {"top1": (preds == labels).mean(), "top5": top5}
```

---

## Grad-CAM — Interpretability

```python
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2, numpy as np

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

# Target the last convolutional layer
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

image = val_tfm(Image.open("cat.jpg")).unsqueeze(0)
rgb   = np.array(Image.open("cat.jpg").resize((224, 224))) / 255.0

# Generate heatmap for predicted class
grayscale_cam = cam(input_tensor=image, targets=None)   # None = predicted class
visualization = show_cam_on_image(rgb.astype(np.float32), grayscale_cam[0])
cv2.imwrite("gradcam.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
```

---

## ONNX Export & Deployment

```python
import torch
import onnxruntime as ort

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.cpu().eval(),
    dummy_input,
    "resnet50.onnx",
    opset_version=18,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
)

# Run with ONNX Runtime (CPU inference, often faster than PyTorch)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession("resnet50.onnx", providers=providers)

def predict_onnx(image_tensor: np.ndarray) -> np.ndarray:
    return sess.run(["logits"], {"image": image_tensor})[0]
```

---

## Tips & Tricks

| Issue | Fix |
|-------|-----|
| Low accuracy | More data augmentation, unfreeze more layers |
| Slow convergence | OneCycleLR, warmup |
| Overfitting | Label smoothing=0.1, Dropout, MixUp/CutMix |
| Class imbalance | WeightedRandomSampler or class weights in loss |
| Small dataset (<1k) | Feature extraction only, DINOv2 features |
| Production serving | ONNX → TensorRT for 5-10x GPU speedup |

---

## Related Topics

- [Computer Vision Overview](index.md)
- [Transfer Learning](../deep-learning/transfer-learning.md)
- [Data Augmentation](index.md#data-augmentation-torchvision-v2)
- [Generative AI](../generative-ai/index.md)
