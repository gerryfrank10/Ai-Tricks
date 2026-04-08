# Computer Vision

Computer vision teaches machines to interpret visual data — images, video, and 3D scenes. In 2025, the field is dominated by transformer-based architectures and foundation models trained on billions of images.

---

## Task Taxonomy

| Task | Description | Example Models (2025) |
|------|-------------|----------------------|
| Image Classification | Assign label to whole image | ViT, ConvNeXt V2, EfficientNet V2 |
| Object Detection | Locate + classify multiple objects | YOLOv11, RT-DETR, Grounding DINO |
| Semantic Segmentation | Per-pixel class label | Mask2Former, SegFormer |
| Instance Segmentation | Per-pixel + per-instance | SAM 2, Mask R-CNN |
| Image Generation | Create new images | Stable Diffusion 3, Flux, DALL-E 3 |
| Depth Estimation | Per-pixel depth | Depth Anything V2 |
| Video Understanding | Temporal reasoning | VideoMAE, Intern-Video2 |
| 3D Vision | Point clouds, NeRF | Gaussian Splatting, PointNet++ |

---

## The Modern CV Stack (2025)

```python
# Core libraries
import torch
import torchvision
from torchvision.transforms import v2 as T   # v2 is the new standard
import cv2
import numpy as np
from PIL import Image

# Foundation model hubs
# - torchvision: ResNet, ViT, ConvNeXt (pretrained)
# - Ultralytics: YOLO v8/v11 (detection, segmentation)
# - Hugging Face: Transformers (ViT, CLIP, SAM, Grounding DINO)
# - timm: 900+ pretrained CV models (pip install timm)
```

### timm — 900+ Pretrained Models

```python
import timm

# Browse available models
models = timm.list_models("*vit*", pretrained=True)[:10]
print(models)
# ['vit_base_patch16_224', 'vit_large_patch16_224', ...]

# Load any model
model = timm.create_model(
    "convnextv2_base",
    pretrained=True,
    num_classes=0,          # Remove head (feature extractor)
)
model.eval()

# Get model-specific transforms
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Extract features
img = Image.open("photo.jpg")
tensor = transforms(img).unsqueeze(0)
with torch.no_grad():
    features = model(tensor)    # (1, 1024) feature vector
```

---

## YOLO v11 — Real-Time Detection

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")   # nano: fastest
model = YOLO("yolo11x.pt")   # extra-large: most accurate

# Inference
results = model.predict(
    source="street.jpg",    # path, URL, webcam, video
    conf=0.5,               # confidence threshold
    iou=0.45,               # NMS IoU threshold
    classes=[0, 2, 7],      # filter: person, car, truck
    device="cuda",
    half=True,              # FP16 for speed
)

for r in results:
    boxes = r.boxes.xyxy    # (N, 4) bounding boxes
    confs = r.boxes.conf    # (N,) confidence scores
    cls   = r.boxes.cls     # (N,) class ids
    print(r.verbose())      # "2 persons, 1 car"

# Fine-tune on custom data
model.train(
    data="my_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda",
    pretrained=True,
    lr0=0.01,
    patience=50,    # Early stopping
)
model.export(format="onnx")   # Export for deployment
```

---

## SAM 2 — Segment Anything Model

```python
# SAM 2 (Meta, 2024): Zero-shot segmentation for images AND video
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2 = build_sam2("sam2_hiera_large.yaml", "sam2_hiera_large.pt", device="cuda")
predictor = SAM2ImagePredictor(sam2)

image = np.array(Image.open("photo.jpg"))
predictor.set_image(image)

# Segment from point prompts
masks, scores, _ = predictor.predict(
    point_coords=np.array([[500, 375]]),   # Click coordinates
    point_labels=np.array([1]),             # 1=foreground, 0=background
    multimask_output=True,
)
best_mask = masks[scores.argmax()]  # (H, W) boolean array
```

---

## Grounding DINO — Open-Vocabulary Detection

```python
# Detect ANY object described in natural language
from groundingdino.util.inference import load_model, predict, load_image

model = load_model("groundingdino_swint_ogc.py", "groundingdino_swint_ogc.pth")
image_source, image = load_image("street.jpg")

# Open-vocabulary detection — no predefined categories!
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="a red bicycle . a white van . person wearing helmet",
    box_threshold=0.35,
    text_threshold=0.25,
)
print(phrases)  # ['a red bicycle', 'person wearing helmet', ...]
```

---

## Data Augmentation (torchvision v2)

```python
from torchvision.transforms import v2 as T
import torch

# Training transforms — aggressive augmentation
train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.TrivialAugmentWide(),     # Auto-augment policy
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25),    # Cutout regularization
])

# Validation transforms — minimal
val_transforms = T.Compose([
    T.Resize(256, antialias=True),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MixUp + CutMix (state-of-the-art regularization)
mixup   = T.MixUp(num_classes=num_classes, alpha=0.2)
cutmix  = T.CutMix(num_classes=num_classes, alpha=1.0)
cutmix_or_mixup = T.RandomChoice([cutmix, mixup])

# Apply in training loop
for images, labels in train_loader:
    images, labels = cutmix_or_mixup(images, labels)
    outputs = model(images)
    loss = criterion(outputs, labels)   # labels are now soft targets
```

---

## Topics

- [Image Classification](image-classification.md) — Full CNN/ViT classification pipeline
- [Object Detection](../generative-ai/index.md) — Coming soon
- [Semantic Segmentation](../generative-ai/index.md) — Coming soon

---

## Related Topics

- [Deep Learning](../deep-learning/index.md)
- [Transfer Learning](../deep-learning/transfer-learning.md)
- [Multimodal AI](../generative-ai/multimodal.md)
- [Generative AI](../generative-ai/index.md)
