# 🏥 AI in Healthcare

Healthcare AI is moving from pilot to production — FDA-cleared algorithms now number in the hundreds, and LLMs are being deployed for clinical documentation at scale. This guide covers the core technical patterns alongside the regulatory realities.

---

## 🗺️ Overview

| Domain | Techniques | Maturity |
|---|---|---|
| Medical Imaging | CNN, ViT, SAM | Production (FDA cleared) |
| EHR Analysis | Tabular ML, LSTM, transformers | Production |
| Drug Discovery | GNN, diffusion models, docking | Research → Production |
| Clinical NLP | NER, relation extraction, LLMs | Production |
| Patient Notes | LLM summarization, SOAP generation | Rapid deployment |
| Federated Learning | FL for privacy-preserving training | Growing |

---

## 🩻 Medical Imaging — CNN for Chest X-Ray Classification

### Model Setup with Pretrained ResNet

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image

# CheXpert / NIH Chest X-Ray 14 labels
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Emphysema", "Fibrosis",
    "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir: Path, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.img_dir / row["Image Index"]).convert("RGB")
        label = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes: int = 14, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.densenet121(weights=weights)
    # Replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
        # No sigmoid — use BCEWithLogitsLoss
    )
    return model


# Transforms (CheXpert-style)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
```

### Training Loop

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import numpy as np

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs_arr  = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    aucs = [roc_auc_score(labels_arr[:, i], probs_arr[:, i])
            for i in range(labels_arr.shape[1])]
    return dict(zip(LABELS, [round(a, 3) for a in aucs])), np.mean(aucs)

# Training setup
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = build_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
```

---

## 📋 EHR Analysis — Structured Clinical Data

### Mortality Prediction (MIMIC-IV style)

```python
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Typical MIMIC-IV feature set for 24-hour ICU mortality
VITAL_COLS  = ["heart_rate", "sbp", "dbp", "spo2", "temp", "resp_rate"]
LAB_COLS    = ["wbc", "sodium", "potassium", "creatinine", "bun", "lactate", "ph"]
STATIC_COLS = ["age", "gender", "admission_type", "insurance", "sofa_score"]

def build_ehr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly vitals/labs into summary stats."""
    agg_fns = ["mean", "std", "min", "max", "last"]
    vitals_agg = df.groupby("stay_id")[VITAL_COLS].agg(agg_fns)
    vitals_agg.columns = ["_".join(c) for c in vitals_agg.columns]

    labs_agg = df.groupby("stay_id")[LAB_COLS].agg(["mean", "min", "max", "last"])
    labs_agg.columns = ["_".join(c) for c in labs_agg.columns]

    static = df.groupby("stay_id")[STATIC_COLS].first()
    return pd.concat([static, vitals_agg, labs_agg], axis=1)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model",   lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )),
])

# 5-fold CV
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    auc  = roc_auc_score(y.iloc[val_idx], prob)
    aucs.append(auc)
    print(f"Fold {fold+1}: AUC = {auc:.4f}")
print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
```

---

## 💊 Drug Discovery — ML on Molecular Data

### Molecular Property Prediction with RDKit + sklearn

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def smiles_to_ecfp4(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES string to Morgan (ECFP4) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def compute_rdkit_descriptors(smiles: str) -> dict:
    """Compute a subset of RDKit 2D descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MolWt":        Descriptors.MolWt(mol),
        "LogP":         Descriptors.MolLogP(mol),
        "TPSA":         Descriptors.TPSA(mol),
        "NumHDonors":   Descriptors.NumHDonors(mol),
        "NumHAcceptors":Descriptors.NumHAcceptors(mol),
        "RotatableBonds":Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
    }

# Example: predict aqueous solubility (ESOL dataset)
smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CC(O)=O"]  # aspirin, benzene, acetic acid
X_fps = np.array([smiles_to_ecfp4(s) for s in smiles_list])

# Combine fingerprints + descriptors
desc_df = pd.DataFrame([compute_rdkit_descriptors(s) for s in smiles_list])
X_combined = np.hstack([X_fps, desc_df.values])

# Train solubility model (illustrative)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
# rf.fit(X_combined, y_logS)
```

---

## 📝 Clinical NLP — Summarizing Patient Notes with LLMs

```python
from openai import OpenAI

client = OpenAI()

CLINICAL_SYSTEM = """You are a clinical documentation specialist.
Summarize the clinical note into a structured SOAP note:
- Subjective: patient-reported symptoms and history
- Objective: vitals, exam findings, lab results
- Assessment: diagnoses / clinical impressions
- Plan: medications, procedures, follow-up

Be concise. Use medical abbreviations. Highlight abnormal findings.
IMPORTANT: Do not invent information not present in the note."""

def generate_soap_note(raw_note: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM},
            {"role": "user",   "content": raw_note},
        ],
        temperature=0.0,   # deterministic for clinical safety
        max_tokens=600,
    )
    return response.choices[0].message.content

# Named Entity Recognition with scispaCy
import spacy

# pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
nlp = spacy.load("en_core_sci_lg")

def extract_clinical_entities(text: str) -> list[dict]:
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
```

---

## 🔒 Federated Learning for Privacy

### Basic FL Simulation with Flower

```python
import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class HealthcareClient(fl.client.NumPyClient):
    """Each hospital is a separate FL client."""
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model   = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_val   = torch.tensor(X_val,   dtype=torch.float32)
        self.y_val   = torch.tensor(y_val,   dtype=torch.long)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(config.get("local_epochs", 5)):
            optimizer.zero_grad()
            loss = criterion(self.model(self.X_train), self.y_train)
            loss.backward()
            optimizer.step()
        return self.get_parameters({}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_val)
            loss   = nn.CrossEntropyLoss()(logits, self.y_val)
            acc    = (logits.argmax(1) == self.y_val).float().mean()
        return float(loss), len(self.X_val), {"accuracy": float(acc)}
```

---

## ⚖️ Regulatory Landscape

| Regulation | Scope | Key Requirement |
|---|---|---|
| FDA 510(k) / De Novo | US medical devices | Pre-market clearance, performance testing |
| FDA SaMD Guidance (2021) | AI/ML software | Predetermined change control plan |
| HIPAA | US patient data | De-identification, BAA with vendors |
| EU MDR 2017/745 | EU medical devices | CE marking, post-market surveillance |
| EU AI Act (High-Risk) | Clinical decision support | Conformity assessment, transparency |
| 21 CFR Part 11 | Electronic records | Audit trails, validation |

> **HIPAA Tip:** When using LLM APIs for clinical data, ensure a Business Associate Agreement (BAA) is in place with the provider. Azure OpenAI, AWS Bedrock, and Google Vertex AI all offer BAAs for healthcare customers.

---

## 💡 Healthcare AI Best Practices

| Practice | Detail |
|---|---|
| Use AUROC + specificity/sensitivity | Accuracy is misleading on imbalanced clinical datasets |
| Prospective validation | Retrospective AUC often drops 5-15 pts in prospective trials |
| Subgroup analysis | Performance across age, sex, race, hospital site |
| Uncertainty quantification | Monte Carlo Dropout or conformal prediction for confidence |
| Human-in-the-loop | FDA expects clinician oversight for high-risk AI |
| Model cards | Document training data, intended use, known limitations |
| Differential privacy | Add noise during FL to prevent gradient inversion attacks |

---

## 📚 Further Reading

- [CheXNet paper (Stanford)](https://arxiv.org/abs/1711.05225)
- [Med-PaLM 2 (Google, 2023)](https://arxiv.org/abs/2305.09617)
- [MIMIC-IV dataset](https://physionet.org/content/mimiciv/2.2/)
- [Flower FL framework](https://flower.ai/)
- [FDA AI/ML Action Plan](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)
- [AlphaFold 3 (2024)](https://www.nature.com/articles/s41586-024-07487-w)
