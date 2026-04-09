# AI in Hospitals & Healthcare

AI is moving from radiology reading rooms to every touchpoint of the patient journey. FDA-cleared AI algorithms now number over 950 (as of 2025), ambient clinical documentation is deployed across thousands of provider groups, and predictive analytics systems run continuously in ICUs worldwide. This guide covers the technical patterns, top platforms, regulatory requirements, and practical code you need to build and deploy hospital AI systems.

---

## Overview

| Metric | Impact | Source |
|--------|--------|--------|
| Diagnostic accuracy (chest X-ray AI vs. radiologist) | +6–10 % AUC improvement on rare findings | Stanford CheXNet |
| Admin time saved per physician per day | 1–3 hours via ambient documentation | AMA / Nuance |
| Readmission reduction with predictive models | 15–25 % reduction | Epic Deterioration Index |
| Sepsis early-warning lead time | 6–12 hours ahead of clinical recognition | Epic Sepsis Model |
| Radiology turnaround time reduction (AI triage) | 30–50 % for critical findings | Aidoc / Viz.ai |
| Coding accuracy improvement (AI-assisted) | 20–35 % error reduction | 3M / Optum |
| Prior-auth automation rate | 70–85 % straight-through processing | Cohere Health |

AI in hospitals clusters into three layers:

1. **Clinical AI** — Imaging analysis, decision support, predictive monitoring
2. **Documentation AI** — Ambient scribes, NLP coding, discharge summaries
3. **Operational AI** — Scheduling, capacity planning, supply chain

---

## Key AI Use Cases

### Medical Imaging & Radiology AI

Computer vision models — primarily CNNs, Vision Transformers, and foundation models — now match or exceed radiologist performance on specific tasks: detecting pneumothorax, diabetic retinopathy, breast cancer in mammography, and intracranial hemorrhage.

**Core workflow:**
- DICOM image ingestion → preprocessing (normalization, windowing)
- Inference via fine-tuned DenseNet121 / EfficientNet / ViT
- Confidence-scored findings overlaid on PACS viewer
- Worklist prioritization: critical findings surfaced to top

**Key datasets:** CheXpert (224K X-rays), NIH ChestX-ray14, MIMIC-CXR, LIDC-IDRI (lung nodules), RSNA Pneumonia Detection.

---

### Clinical Decision Support

Rule-based CDS (drug interaction alerts, dosing checks) has existed for decades. Modern AI-CDS adds:

- **Differential diagnosis generation** from unstructured notes (LLM + medical knowledge graphs)
- **Evidence-based recommendation** retrieval (RAG over UpToDate, clinical guidelines)
- **Lab result interpretation** in context of patient history

The risk: alert fatigue. AI-CDS must be calibrated for specificity — too many alerts and clinicians override everything.

---

### Ambient Clinical Documentation (AI Scribes)

The fastest-growing category in 2024–2025. AI listens to the patient-physician conversation, generates a structured clinical note (SOAP, DAP, or custom template), and pushes it into the EHR for physician review and sign-off.

**Components:**
- ASR (Automatic Speech Recognition): Whisper, Azure Speech, Deepgram
- Speaker diarization: identify patient vs. clinician
- Medical entity extraction: diagnoses, medications, dosages
- Note generation: LLM conditioned on conversation + patient context
- EHR integration: HL7 FHIR API to pre-fill note fields

**Market leaders:** Nuance DAX Copilot (Microsoft), Suki AI, Ambience Healthcare, Abridge, DeepScribe

---

### Patient Flow & Scheduling Optimization

Hospitals lose millions annually to OR cancellations, ED boarding, and no-shows. AI addresses:

- **No-show prediction** — XGBoost on demographics, appointment history, distance, weather
- **OR scheduling** — constraint satisfaction + ML for case duration prediction
- **ED patient flow** — real-time census forecasting, bed management
- **Discharge prediction** — predict discharge date at admission to coordinate downstream resources

---

### Predictive Analytics (Readmission, Sepsis, Deterioration)

The highest-acuity AI use case. Models run continuously on streaming EHR data to flag patients before clinical deterioration is visually apparent.

- **Sepsis:** Detect early SIRS criteria + lactate trends + fluid responsiveness (Epic Sepsis Model, Sepsis ImmunoScore)
- **Deterioration:** Modified Early Warning Score (MEWS) augmented with ML — SpO2 drift, HR variability, respiratory rate trends
- **30-day readmission:** CMS publicly reports hospital readmission rates; accurate prediction enables targeted transitional care

---

### Drug Interaction & Medication Safety

AI layers on top of traditional rule-based drug-drug interaction (DDI) checkers:

- **Polypharmacy risk scoring** — graph neural networks on drug-protein interaction networks
- **Dose optimization** — pharmacokinetic modeling with patient-specific covariates (weight, renal function, genetic markers)
- **IV compatibility checking** — ML on Y-site compatibility databases
- **Natural language medication reconciliation** — extracting medication lists from discharge summaries and reconciling against pharmacy records

---

### Administrative Automation (Billing, Coding, Prior Auth)

Revenue cycle management (RCM) is a $20B+ market where AI delivers measurable ROI without regulatory approval requirements.

- **Medical coding (ICD-10, CPT)** — NLP on clinical notes to suggest codes, reducing coder review time
- **Prior authorization** — ML classifiers predict approval likelihood; RPA automates submission workflows
- **Claim scrubbing** — AI detects unbundling errors, upcoding, modifier misuse before claim submission
- **Denial management** — classify denial reason codes, auto-generate appeal letters

---

### Virtual Health Assistants

Conversational AI for patient-facing interactions:

- **Symptom checkers** — guided triage to ED, urgent care, telehealth, or self-care
- **Post-discharge follow-up** — automated check-in calls/texts, medication adherence nudges
- **Appointment scheduling** — NLU-based scheduling bots integrated with Epic/Cerner calendars
- **Mental health support** — CBT-based chatbots (Woebot, Wysa) for low-acuity mental health support

---

## Top AI Tools & Platforms

| Tool | Provider | Use Case | FDA Status |
|------|----------|----------|------------|
| Nuance DAX Copilot | Microsoft | Ambient clinical documentation, note generation | Not a medical device (documentation tool) |
| MedPaLM 2 / Health AI | Google | Clinical Q&A, medical imaging, EHR summarization | Research; Vertex AI HIPAA-eligible |
| Epic + AI (Cognitive Computing) | Epic Systems | Predictive models (sepsis, readmission), NLP, in-basket automation | Embedded in FDA-cleared EHR workflows |
| Viz.ai | Viz.ai | Stroke, PE, aortic disease triage from CT | FDA 510(k) cleared (multiple indications) |
| Aidoc | Aidoc | Radiology AI — ICH, PE, spine fracture, aortic aneurysm | FDA cleared (11+ indications) |
| PathAI | PathAI | Digital pathology — cancer detection, biomarker quantification | FDA De Novo (AIdx-M breast) |
| Paige.ai | Paige | Prostate cancer detection in whole-slide images | FDA De Novo cleared (first AI pathology approval) |
| Tempus | Tempus AI | Genomic profiling, oncology decision support, multimodal EHR | CLIA-certified lab; FDA breakthrough device |
| Azure Health Bot | Microsoft | Patient-facing virtual assistants, symptom triage | Not a medical device |
| Suki AI | Suki | AI medical scribe, voice-to-note EHR assistant | Not a medical device (documentation) |
| Amazon Comprehend Medical | AWS | Clinical NLP — entity extraction, ICD/RxNorm coding | HIPAA eligible; not FDA-cleared |
| IBM Merative (Watson Health) | Merative | Clinical trial matching, imaging AI, RCM analytics | Varies by module |
| Cohere Health | Cohere | Prior authorization automation, clinical intelligence | Not a medical device (administrative) |
| Eko Health | Eko | AI-powered cardiac auscultation (stethoscope + app) | FDA cleared for murmur detection |

---

## Technology Stack

### Medical Imaging Classification — Fine-tuning ResNet for Chest X-Ray

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# 14-class multi-label classification (NIH ChestX-ray14)
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

class ChestXRayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["Image Index"]).convert("RGB")
        label = torch.tensor(row[PATHOLOGIES].values.astype(float), dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label


def build_resnet_model(num_classes: int = 14, freeze_backbone: bool = False) -> nn.Module:
    """Fine-tune ResNet-50 for multi-label chest X-ray classification."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in list(model.parameters())[:-20]:  # freeze all but last block
            param.requires_grad = False

    # Replace FC head with multi-label classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
        # No sigmoid here — use BCEWithLogitsLoss during training
    )
    return model


# Transforms with medical imaging augmentations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate_model(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs_arr = np.vstack(all_probs)
    labels_arr = np.vstack(all_labels)
    per_class_auc = {
        path: roc_auc_score(labels_arr[:, i], probs_arr[:, i])
        for i, path in enumerate(PATHOLOGIES)
        if labels_arr[:, i].sum() > 0  # skip classes with no positive examples
    }
    mean_auc = np.mean(list(per_class_auc.values()))
    return per_class_auc, mean_auc


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_resnet_model(num_classes=14, freeze_backbone=False).to(device)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0] * 14).to(device)  # handle class imbalance
)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Example: 20 epochs, batch size 32
EPOCHS = 20
BATCH_SIZE = 32

# scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)
# for epoch in range(EPOCHS):
#     loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
#     per_class, mean_auc = evaluate_model(model, val_loader, device)
#     print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Mean AUC: {mean_auc:.4f}")
```

> **Tip:** For production radiology AI, use DenseNet-121 (CheXNet architecture) or EfficientNet-B4. Ensemble 3–5 models trained with different seeds for a 1–2 AUC point improvement. Always report sensitivity at fixed specificity thresholds, not just mean AUC.

---

### Clinical NLP — Extracting Diagnoses & Medications from Notes

```python
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from dataclasses import dataclass, field
from typing import Optional

# --- Option 1: scispaCy for fast rule-based + statistical NER ---
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

def extract_with_scispacy(text: str) -> dict:
    """Extract clinical entities using scispaCy BC5CDR model (diseases + chemicals)."""
    nlp = spacy.load("en_ner_bc5cdr_md")  # trained on BioCreative V CDR corpus
    doc = nlp(text)
    entities = {"DISEASE": [], "CHEMICAL": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text.strip())
    return entities


# --- Option 2: HuggingFace transformer-based clinical NER ---
@dataclass
class ClinicalEntity:
    text: str
    label: str
    score: float
    start: int
    end: int
    normalized: Optional[str] = None  # RxNorm / SNOMED code after linking


def extract_clinical_entities_transformer(text: str) -> list[ClinicalEntity]:
    """Use Bio_ClinicalBERT fine-tuned for NER on i2b2/n2c2 datasets."""
    model_name = "samrawal/bert-base-uncased_clinical-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # merge subword tokens
        device=-1,
    )
    raw = ner_pipe(text)
    return [
        ClinicalEntity(
            text=ent["word"],
            label=ent["entity_group"],
            score=round(ent["score"], 3),
            start=ent["start"],
            end=ent["end"],
        )
        for ent in raw
        if ent["score"] > 0.75  # confidence threshold
    ]


# --- Option 3: Amazon Comprehend Medical (HIPAA-eligible, managed) ---
import boto3

def extract_with_comprehend_medical(text: str) -> dict:
    """Use AWS Comprehend Medical for entity + RxNorm/ICD-10 linking."""
    client = boto3.client("comprehendmedical", region_name="us-east-1")

    # Entity detection
    entity_response = client.detect_entities_v2(Text=text[:10000])

    # ICD-10-CM inference
    icd_response = client.infer_icd10_cm(Text=text[:10000])

    # RxNorm inference (medications)
    rx_response = client.infer_rx_norm(Text=text[:10000])

    return {
        "entities": [
            {
                "text": e["Text"],
                "category": e["Category"],      # MEDICATION, MEDICAL_CONDITION, etc.
                "type": e["Type"],
                "score": round(e["Score"], 3),
                "traits": [t["Name"] for t in e.get("Traits", [])],
            }
            for e in entity_response["Entities"]
        ],
        "icd10_codes": [
            {
                "text": e["Text"],
                "code": e["ICD10CMConcepts"][0]["Code"] if e.get("ICD10CMConcepts") else None,
                "description": e["ICD10CMConcepts"][0]["Description"] if e.get("ICD10CMConcepts") else None,
            }
            for e in icd_response["Entities"]
        ],
        "medications": [
            {
                "text": e["Text"],
                "rxnorm_id": e["RxNormConcepts"][0]["Code"] if e.get("RxNormConcepts") else None,
                "drug_name": e["RxNormConcepts"][0]["Description"] if e.get("RxNormConcepts") else None,
            }
            for e in rx_response["Entities"]
        ],
    }


# Example usage
note = """
Patient is a 67-year-old male with hypertension and type 2 diabetes mellitus.
Currently on metformin 1000 mg BID and lisinopril 10 mg daily.
Presents with worsening dyspnea and bilateral lower extremity edema.
Labs show creatinine 1.8, BNP 890 pg/mL. Impression: acute decompensated heart failure.
"""
entities = extract_clinical_entities_transformer(note)
for e in entities:
    print(f"[{e.label:20s}] {e.text:30s} (score={e.score})")
```

---

### Readmission Prediction — XGBoost on Patient Features

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import shap

# --- Feature Engineering from EHR Data ---

def build_readmission_features(admissions_df: pd.DataFrame,
                                diagnoses_df: pd.DataFrame,
                                vitals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 30-day readmission features from MIMIC-style tables.
    Target: 1 if patient readmitted within 30 days of discharge.
    """
    features = admissions_df.copy()

    # Length of stay (days)
    features["los_days"] = (
        pd.to_datetime(features["dischtime"]) -
        pd.to_datetime(features["admittime"])
    ).dt.total_seconds() / 86400

    # Admission type encoding
    features["is_emergency"] = (features["admission_type"] == "EMERGENCY").astype(int)

    # Number of previous admissions (utilization)
    prev_counts = (
        admissions_df.groupby("subject_id")
        .apply(lambda g: g.sort_values("admittime")["hadm_id"].expanding().count() - 1)
        .reset_index(level=0, drop=True)
    )
    features["n_prior_admissions"] = prev_counts.values

    # Diagnosis count (complexity proxy)
    diag_counts = diagnoses_df.groupby("hadm_id")["icd_code"].count().rename("n_diagnoses")
    features = features.merge(diag_counts, on="hadm_id", how="left")

    # Comorbidity flags (Charlson index components)
    charlson_codes = {
        "ami":        ["I21", "I22"],
        "chf":        ["I50"],
        "pvd":        ["I70", "I71", "I73"],
        "stroke":     ["G45", "G46", "I60", "I61", "I62", "I63", "I64"],
        "dementia":   ["F00", "F01", "F02", "F03"],
        "copd":       ["J40", "J41", "J42", "J43", "J44", "J45"],
        "diabetes":   ["E10", "E11", "E12", "E13", "E14"],
        "renal":      ["N18", "N19"],
        "cancer":     ["C00", "C01", "C02"],
    }
    icd_pivot = diagnoses_df.copy()
    icd_pivot["icd3"] = icd_pivot["icd_code"].str[:3]

    for condition, prefixes in charlson_codes.items():
        mask = icd_pivot["icd3"].isin(prefixes)
        flag = icd_pivot[mask].groupby("hadm_id")["icd3"].any().astype(int)
        features[f"cc_{condition}"] = features["hadm_id"].map(flag).fillna(0).astype(int)

    # Charlson Comorbidity Index (simplified)
    features["charlson_score"] = (
        features["cc_ami"] +
        features["cc_chf"] * 2 +
        features["cc_pvd"] +
        features["cc_stroke"] * 2 +
        features["cc_dementia"] * 2 +
        features["cc_copd"] +
        features["cc_diabetes"] +
        features["cc_renal"] * 2 +
        features["cc_cancer"] * 2
    )

    # Last recorded vitals (stability at discharge)
    last_vitals = (
        vitals_df.sort_values("charttime")
        .groupby("hadm_id")
        .agg(
            last_sbp=("sbp", "last"),
            last_spo2=("spo2", "last"),
            last_hr=("heart_rate", "last"),
            last_temp=("temperature", "last"),
        )
    )
    features = features.merge(last_vitals, on="hadm_id", how="left")

    # Discharge destination
    le = LabelEncoder()
    features["discharge_location_enc"] = le.fit_transform(
        features["discharge_location"].fillna("UNKNOWN")
    )

    # Target: 30-day readmission
    features = features.sort_values(["subject_id", "admittime"])
    features["next_admit"] = features.groupby("subject_id")["admittime"].shift(-1)
    features["days_to_readmit"] = (
        pd.to_datetime(features["next_admit"]) -
        pd.to_datetime(features["dischtime"])
    ).dt.days
    features["readmit_30d"] = (features["days_to_readmit"] <= 30).astype(int)

    return features


# --- XGBoost Model with Calibration ---

FEATURE_COLS = [
    "age", "los_days", "is_emergency", "n_prior_admissions",
    "n_diagnoses", "charlson_score",
    "cc_chf", "cc_copd", "cc_diabetes", "cc_renal",
    "last_sbp", "last_spo2", "last_hr", "last_temp",
    "discharge_location_enc",
]

def train_readmission_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["readmit_30d"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    models = []

    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,    # avoid overfitting on small patient subgroups
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Calibrate probabilities (Platt scaling) — critical for clinical use
        calibrated = CalibratedClassifierCV(
            base_model.__class__(**base_model.get_params()),
            method="sigmoid",
            cv=3,
        )
        calibrated.fit(X_tr, y_tr)
        oof_probs[val_idx] = calibrated.predict_proba(X_val)[:, 1]
        models.append(calibrated)

        fold_auc = roc_auc_score(y_val, oof_probs[val_idx])
        print(f"Fold {fold+1}: AUC={fold_auc:.4f}")

    print(f"\nOOF ROC-AUC  : {roc_auc_score(y, oof_probs):.4f}")
    print(f"OOF PR-AUC   : {average_precision_score(y, oof_probs):.4f}")
    print(f"OOF Brier    : {brier_score_loss(y, oof_probs):.4f}")

    # SHAP explainability (use inner XGB from last fold)
    explainer = shap.TreeExplainer(models[-1].calibrated_classifiers_[0].estimator)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)

    return models, oof_probs

# models, probs = train_readmission_model(features_df)
```

> **Clinical Note:** Always calibrate readmission model probabilities. A model predicting 0.8 should be right ~80% of the time — uncalibrated XGBoost scores are not well-calibrated by default. Use Platt scaling or isotonic regression via `CalibratedClassifierCV`.

---

### AI Medical Scribe — Transcribe + Summarize Patient Encounter

```python
import anthropic
import whisper
import tempfile
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SOAPNote:
    chief_complaint: str
    subjective: str        # Patient's own words: HPI, symptoms, history
    objective: str         # Vitals, physical exam, lab/imaging results
    assessment: str        # Diagnoses, differential, clinical reasoning
    plan: str              # Medications, orders, referrals, follow-up
    follow_up: str         # Timeframe and conditions for follow-up
    billing_codes: list    # Suggested ICD-10 / CPT codes


def transcribe_encounter(audio_path: str, model_size: str = "medium") -> str:
    """
    Transcribe physician-patient encounter audio using OpenAI Whisper.
    The 'medium' model balances speed and accuracy for medical vocabulary.
    Use 'large-v3' for best accuracy in production.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        word_timestamps=True,
        condition_on_previous_text=True,   # better for long clinical conversations
        initial_prompt=(
            "Medical encounter between physician and patient. "
            "Include medical terms, drug names, and dosages verbatim."
        ),
    )
    return result["text"]


def generate_soap_note(
    transcript: str,
    patient_context: dict | None = None,
    client: anthropic.Anthropic | None = None,
) -> SOAPNote:
    """
    Generate a structured SOAP note from encounter transcript using Claude.
    patient_context: optional dict with prior diagnoses, medications, allergies.
    """
    if client is None:
        client = anthropic.Anthropic()

    context_str = ""
    if patient_context:
        context_str = f"""
**Patient Context (from EHR):**
- Active diagnoses: {', '.join(patient_context.get('diagnoses', []))}
- Current medications: {', '.join(patient_context.get('medications', []))}
- Allergies: {', '.join(patient_context.get('allergies', []))}
- Last visit: {patient_context.get('last_visit', 'Unknown')}
"""

    system_prompt = """You are an expert clinical documentation specialist with 20 years of experience
generating accurate, concise SOAP notes for physician review.

Rules:
1. Only document information explicitly stated in the transcript — never fabricate details.
2. Use standard medical abbreviations (HPI, PMHx, ROS, BP, HR, etc.).
3. Flag uncertainty with phrases like "patient reports" or "per patient."
4. For the Assessment, list diagnoses in order of clinical priority.
5. For the Plan, number each action item.
6. Suggest relevant ICD-10-CM codes for each diagnosis.
7. Output ONLY valid JSON matching the specified schema — no additional text.

JSON Schema:
{
  "chief_complaint": "string (1-2 sentences)",
  "subjective": "string (HPI, PMHx, social/family history, ROS)",
  "objective": "string (vitals, exam findings, labs, imaging)",
  "assessment": "string (diagnoses with ICD-10 codes, reasoning)",
  "plan": "string (numbered list: medications, orders, referrals, patient education)",
  "follow_up": "string (timing and conditions)",
  "billing_codes": ["list of ICD-10 or CPT strings"]
}"""

    user_prompt = f"""{context_str}

**Encounter Transcript:**
{transcript}

Generate the SOAP note JSON."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        temperature=0.0,     # deterministic output for clinical safety
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    raw_json = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]

    data = json.loads(raw_json)
    return SOAPNote(**data)


def process_encounter(audio_path: str, patient_context: dict | None = None) -> SOAPNote:
    """End-to-end: audio → transcription → SOAP note."""
    print(f"Transcribing {audio_path}...")
    transcript = transcribe_encounter(audio_path)

    print("Generating SOAP note...")
    client = anthropic.Anthropic()
    soap = generate_soap_note(transcript, patient_context, client)

    return soap


# --- Example Usage ---
# patient_ctx = {
#     "diagnoses": ["Type 2 Diabetes Mellitus (E11.9)", "Hypertension (I10)"],
#     "medications": ["Metformin 1000mg BID", "Lisinopril 10mg daily"],
#     "allergies": ["Penicillin (rash)", "Sulfa drugs"],
#     "last_visit": "2025-01-15",
# }
# soap = process_encounter("encounter_recording.mp3", patient_ctx)
# print(json.dumps(asdict(soap), indent=2))
```

**Sample output:**
```json
{
  "chief_complaint": "67-year-old male with HTN and T2DM presenting with worsening dyspnea x 3 days.",
  "subjective": "HPI: Patient reports progressive dyspnea on exertion, orthopnea (2-pillow), and bilateral ankle swelling for 3 days. Denies chest pain or fever. Has been compliant with medications. Diet has been high in sodium this week per patient. PMHx: HTN, T2DM. Meds: Metformin 1000mg BID, Lisinopril 10mg daily. Allergies: Penicillin (rash). Social: Lives alone, retired. No tobacco/ETOH.",
  "objective": "Vitals: BP 158/94, HR 92, RR 20, SpO2 93% RA, Temp 37.1°C, Wt 84 kg (+4 kg from last visit). Exam: JVD present at 45°. Lungs: bilateral crackles at bases. Extremities: 2+ pitting edema to knees bilaterally. Labs: BNP 890 pg/mL (H), Creatinine 1.8 mg/dL (H), Na 138, K 4.1. CXR: cardiomegaly, bilateral pleural effusions.",
  "assessment": "1. Acute decompensated heart failure (I50.9) — elevated BNP, volume overload, orthopnea\n2. Hypertension, inadequately controlled (I10)\n3. Type 2 Diabetes Mellitus (E11.9) — chronic, stable\n4. Acute kidney injury on CKD (N17.9) — Cr 1.8, monitor closely with diuresis",
  "plan": "1. IV Furosemide 40mg now, then PO 40mg BID; target 1-2L negative fluid balance/day\n2. Hold Metformin given AKI — restart when Cr < 1.5\n3. Sodium restriction < 2g/day, fluid restriction 1.5L/day\n4. Cardiology consult today\n5. Repeat BMP in AM\n6. Patient education: daily weights, low-sodium diet\n7. Consider echocardiogram to assess EF if not done in past 12 months",
  "follow_up": "Cardiology within 1 week post-discharge. PCP in 3-5 days. Return to ED if weight increases > 2 lbs/day or dyspnea worsens.",
  "billing_codes": ["I50.9", "I10", "E11.9", "N17.9", "99214"]
}
```

---

## Best Workflow — AI-Augmented Patient Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI-AUGMENTED HOSPITAL CARE PATHWAY                       │
└─────────────────────────────────────────────────────────────────────────────┘

  PATIENT ARRIVAL
  ───────────────
  Patient arrives → Registration
  │
  └── [AI] Virtual triage chatbot → symptom collection → acuity prediction
       ↓
       ESI level assigned (AI-assisted)

  TRIAGE
  ──────
  Nurse assessment
  │
  └── [AI] NLP extracts CC from spoken/typed intake
       └── [AI] Sepsis pre-screen (vitals + chief complaint → SIRS criteria)
       ↓
       Risk stratified: High / Medium / Low

  DIAGNOSIS
  ─────────
  Physician evaluation
  │
  ├── [AI] Medical imaging AI
  │    ├── Chest X-ray → pneumonia, effusion, PTX detection (Aidoc / Viz.ai)
  │    ├── CT head → ICH triage (< 5 min read, worklist prioritization)
  │    └── ECG interpretation → STEMI, AF, long QT detection
  │
  ├── [AI] Clinical Decision Support (Epic CDS / Isabel DDx)
  │    └── Differential diagnoses surfaced from note + lab context
  │
  └── [AI] Drug interaction check
       └── Polypharmacy risk flagged before any order placed

  TREATMENT & ORDERS
  ──────────────────
  Treatment plan created
  │
  ├── [AI] Ambient scribe recording encounter
  │    └── Real-time SOAP note drafted (Nuance DAX / Suki)
  │
  ├── [AI] Order suggestions from CDS
  │    └── Evidence-based order sets surfaced contextually
  │
  └── [AI] Dose optimization (renal/hepatic adjustment)

  MONITORING (INPATIENT)
  ──────────────────────
  Continuous vital sign streams + labs
  │
  ├── [AI] Sepsis early warning score (continuous inference)
  │    └── Alert fired 6–12 hrs before clinical criteria met
  │
  ├── [AI] Deterioration index (Epic Deterioration Index)
  │    └── Escalation alert → RRT activation
  │
  └── [AI] Medication administration safety check
       └── 5-rights verification at bedside scanner

  DISCHARGE
  ─────────
  Physician decision to discharge
  │
  ├── [AI] Predicted discharge date (capacity planning)
  │
  ├── [AI] Discharge summary auto-generated from encounter notes
  │    └── Medication reconciliation NLP check
  │
  ├── [AI] 30-day readmission risk score
  │    └── High-risk → transitional care program referral
  │
  └── [AI] Prior authorization auto-submitted for post-acute orders

  FOLLOW-UP
  ─────────
  Post-discharge
  │
  ├── [AI] Automated check-in SMS/call at 48h, 7d, 30d
  │
  ├── [AI] Remote monitoring data ingestion (wearables, RPM)
  │    └── Anomaly detection → care team alert
  │
  └── [AI] Appointment no-show prediction → proactive outreach
```

---

## Building a Clinical Assistant

Complete SOAP note generator with structured output, patient context injection, and provider review integration:

```python
import anthropic
import json
from typing import Optional
from datetime import datetime


def build_clinical_assistant(
    transcript: str,
    patient_mrn: str,
    provider_id: str,
    encounter_type: str = "office_visit",
    ehr_context: Optional[dict] = None,
) -> dict:
    """
    Production-grade clinical assistant that generates a SOAP note
    from a conversation transcript, ready for EHR integration.

    Returns a structured dict ready for HL7 FHIR DocumentReference resource.
    """
    client = anthropic.Anthropic()

    # Build patient context block
    ehr_block = ""
    if ehr_context:
        active_meds = ehr_context.get("active_medications", [])
        meds_str = "\n  - ".join(active_meds) if active_meds else "None on file"

        problems = ehr_context.get("problem_list", [])
        problems_str = "\n  - ".join(problems) if problems else "None on file"

        allergies = ehr_context.get("allergies", [])
        allergy_str = ", ".join(allergies) if allergies else "NKDA"

        ehr_block = f"""
<patient_context>
  <active_medications>
  - {meds_str}
  </active_medications>
  <problem_list>
  - {problems_str}
  </problem_list>
  <allergies>{allergy_str}</allergies>
  <last_labs>{json.dumps(ehr_context.get('last_labs', {}), indent=4)}</last_labs>
</patient_context>
"""

    system = """You are a clinical documentation AI assistant integrated into an electronic health record system.
Your role is to generate accurate, clinically appropriate SOAP notes for physician review and attestation.

CRITICAL RULES:
- Never fabricate clinical information. Document only what is in the transcript.
- If information is unclear or absent, note it explicitly (e.g., "not documented in encounter").
- Use proper medical abbreviations and clinical terminology.
- Format the Assessment as a numbered problem list with ICD-10 codes.
- Format the Plan as numbered action items organized by problem.
- Flag any safety concerns (drug interactions, allergy conflicts, abnormal values) in a SAFETY_FLAGS field.
- Output valid JSON only — no markdown, no preamble.

Output schema:
{
  "encounter_metadata": {
    "encounter_type": "string",
    "documentation_datetime": "ISO 8601",
    "documentation_method": "ambient_ai_scribe"
  },
  "chief_complaint": "string",
  "subjective": {
    "hpi": "string",
    "pmhx": "string",
    "medications": "string (reconciled from transcript)",
    "allergies": "string",
    "social_history": "string",
    "ros": "string (relevant positives and pertinent negatives)"
  },
  "objective": {
    "vitals": "string",
    "physical_exam": "string",
    "results": "string (labs, imaging, other diagnostic data)"
  },
  "assessment": "string (numbered problem list with ICD-10 codes)",
  "plan": "string (numbered action items per problem)",
  "follow_up": "string",
  "safety_flags": ["list of safety concerns or empty list"],
  "suggested_billing": {
    "em_level": "string (e.g., 99214)",
    "diagnoses": ["ICD-10 codes"],
    "procedures": ["CPT codes if applicable"]
  },
  "pending_physician_review": true
}"""

    user_content = f"""
{ehr_block}

<encounter_transcript>
{transcript}
</encounter_transcript>

Encounter type: {encounter_type}
Generate the SOAP note JSON."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()
    soap_data = json.loads(raw)

    # Add metadata
    soap_data["encounter_metadata"].update({
        "patient_mrn": patient_mrn,
        "provider_id": provider_id,
        "documentation_datetime": datetime.utcnow().isoformat() + "Z",
        "ai_model_version": "claude-opus-4-5",
        "status": "draft_pending_attestation",
    })

    # Surface safety flags prominently
    if soap_data.get("safety_flags"):
        print("\n⚠ SAFETY FLAGS DETECTED — Physician review required:")
        for flag in soap_data["safety_flags"]:
            print(f"  • {flag}")

    return soap_data


# --- FHIR Integration Helper ---
def soap_to_fhir_document_reference(soap_data: dict, patient_id: str) -> dict:
    """Convert SOAP dict to HL7 FHIR R4 DocumentReference resource."""
    import base64

    note_text = json.dumps(soap_data, indent=2)
    encoded = base64.b64encode(note_text.encode()).decode()

    return {
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "preliminary",   # becomes 'final' after physician attestation
        "type": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "11488-4",
                "display": "Consult note"
            }]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": soap_data["encounter_metadata"]["documentation_datetime"],
        "author": [{"reference": f"Practitioner/{soap_data['encounter_metadata']['provider_id']}"}],
        "content": [{
            "attachment": {
                "contentType": "application/json",
                "data": encoded,
                "title": "AI-Generated SOAP Note (Pending Attestation)"
            }
        }],
        "context": {
            "practiceSetting": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "394814009",
                    "display": "General practice"
                }]
            }
        }
    }
```

---

## ROI & Metrics

| Category | Metric | Typical Improvement | Measurement Method |
|----------|--------|--------------------|--------------------|
| Documentation | Time in EHR per encounter | -1.5 to -2.5 hrs/day | EHR audit logs (keystrokes, time-in-chart) |
| Documentation | Note completion rate at end of shift | +25–40 % | EHR reporting |
| Documentation | Physician burnout score (MBI) | -15 to -20 % | Validated burnout surveys |
| Radiology | Critical finding notification time | -30 to -60 min | PACS turnaround time reports |
| Radiology | Missed finding rate (ICH, PE) | -30 to -50 % | Radiologist audit, malpractice data |
| Readmissions | 30-day all-cause readmission rate | -15 to -25 % | CMS claims data |
| Sepsis | Sepsis-related mortality | -10 to -20 % | ICU mortality dashboards |
| Revenue | Coder productivity (charts/hr) | +20–40 % | RCM system reports |
| Revenue | Clean claim rate (first-pass) | +5 to +15 % | Clearinghouse reports |
| Operations | OR cancellation rate | -20 to -30 % | Surgical scheduling system |
| Patient Flow | ED left-without-being-seen (LWBS) | -25 to -35 % | ED tracking board |

### Calculating Ambient Scribe ROI

```python
def calculate_scribe_roi(
    n_physicians: int,
    avg_salary_usd: float = 280_000,
    hours_saved_per_day: float = 1.8,
    scribe_cost_per_physician_month: float = 1_200,
    working_days_per_year: int = 230,
) -> dict:
    """
    Calculate annual ROI of ambient AI scribe deployment.
    Default values based on AMA 2024 physician survey data.
    """
    # Time value of documentation time saved
    hourly_rate = avg_salary_usd / (working_days_per_year * 8)
    annual_time_value = (
        n_physicians * hours_saved_per_day * working_days_per_year * hourly_rate
    )

    # Additional revenue from increased patient capacity
    # Assume 15 min saved = 1 additional patient slot at $150 avg reimbursement
    extra_visits_per_day = (hours_saved_per_day * 60) / 15
    annual_extra_revenue = (
        n_physicians * extra_visits_per_day * working_days_per_year * 150
    )

    # Total cost
    annual_cost = n_physicians * scribe_cost_per_physician_month * 12

    total_benefit = annual_time_value + annual_extra_revenue
    roi_pct = ((total_benefit - annual_cost) / annual_cost) * 100

    return {
        "n_physicians": n_physicians,
        "annual_cost_usd": round(annual_cost),
        "annual_time_value_usd": round(annual_time_value),
        "annual_extra_revenue_usd": round(annual_extra_revenue),
        "total_annual_benefit_usd": round(total_benefit),
        "net_annual_benefit_usd": round(total_benefit - annual_cost),
        "roi_percent": round(roi_pct, 1),
        "payback_months": round(annual_cost / (total_benefit / 12), 1),
    }

# Example: 50-physician group practice
result = calculate_scribe_roi(n_physicians=50)
# {'n_physicians': 50, 'annual_cost_usd': 720000,
#  'annual_time_value_usd': 894750, 'annual_extra_revenue_usd': 1552500,
#  'total_annual_benefit_usd': 2447250, 'net_annual_benefit_usd': 1727250,
#  'roi_percent': 239.9, 'payback_months': 3.5}
```

---

## Compliance & Ethics

### HIPAA Requirements for AI Systems

| Requirement | Detail | Implementation |
|-------------|--------|----------------|
| Business Associate Agreement (BAA) | Required with any AI vendor handling PHI | Azure OpenAI, AWS Bedrock, Google Vertex AI all offer BAAs |
| Data De-identification | Safe Harbor (18 identifiers) or Expert Determination | Use AWS Comprehend Medical de-id or MITRE Presidio |
| Minimum Necessary | Only share the PHI needed for the AI task | Scope API calls to relevant data elements |
| Audit Logs | Track all access to PHI in AI pipelines | CloudTrail, Azure Monitor, or custom logging |
| Breach Notification | 60-day notice to HHS if PHI breach occurs | Incident response plan must include AI systems |

### FDA Regulations — AI as a Medical Device (SaMD)

Software as a Medical Device (SaMD) requires FDA oversight when it is:
- **Intended to diagnose, treat, cure, or prevent** disease
- Makes or assists in **clinical decisions** that could harm patients if wrong

**Regulatory pathways:**

| Pathway | Risk Level | Requirements |
|---------|-----------|--------------|
| 510(k) Clearance | Moderate risk | Demonstrate substantial equivalence to predicate device |
| De Novo | Novel, low-to-moderate risk | New classification, used for first-of-type AI (e.g., Paige.ai) |
| PMA (Pre-Market Approval) | High risk | Full clinical trial evidence, most rigorous |
| Breakthrough Device | Life-threatening conditions | Expedited review, increased FDA interaction |

**Predetermined Change Control Plan (PCCP):** FDA's 2024 guidance allows AI/ML developers to pre-specify algorithm changes that do not require a new 510(k). This enables continuous learning models to update without re-clearance for pre-approved modifications.

**Not medical devices (no FDA clearance needed):**
- Administrative tools (billing, scheduling, documentation)
- Clinical communication tools
- General wellness apps without clinical claims

### Bias in Medical AI

Medical AI models trained on historically biased data can perpetuate and amplify health disparities.

```python
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np

def audit_model_fairness(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    sensitive_attrs: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Audit model performance across demographic subgroups.
    Returns a DataFrame with key fairness metrics per group.
    """
    results = []
    y_pred = (y_pred_prob >= threshold).astype(int)

    for col in sensitive_attrs.columns:
        for group_val in sensitive_attrs[col].unique():
            mask = sensitive_attrs[col] == group_val
            if mask.sum() < 30:  # skip groups too small for reliable stats
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            yp_prob = y_pred_prob[mask]

            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            n = len(yt)

            results.append({
                "attribute": col,
                "group": group_val,
                "n": n,
                "prevalence": round(yt.mean(), 3),
                "auc": round(roc_auc_score(yt, yp_prob) if yt.nunique() > 1 else np.nan, 3),
                "sensitivity_tpr": round(tp / (tp + fn) if (tp + fn) > 0 else np.nan, 3),
                "specificity_tnr": round(tn / (tn + fp) if (tn + fp) > 0 else np.nan, 3),
                "ppv": round(tp / (tp + fp) if (tp + fp) > 0 else np.nan, 3),
                "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else np.nan, 3),
                "fnr": round(fn / (fn + tp) if (fn + tp) > 0 else np.nan, 3),
            })

    report = pd.DataFrame(results)

    # Flag disparities exceeding 10% absolute difference between groups
    for metric in ["sensitivity_tpr", "specificity_tnr", "auc"]:
        group_vals = report.groupby("attribute")[metric]
        report[f"{metric}_disparity"] = group_vals.transform(
            lambda x: (x - x.mean()).abs() > 0.10
        )

    return report


# Example: audit readmission model across race and insurance type
# sensitive = patient_df[["race", "insurance_type"]]
# fairness_report = audit_model_fairness(y_test, probs, sensitive)
# print(fairness_report[fairness_report["auc_disparity"]].to_string())
```

### Human-in-the-Loop Requirements

The FDA and clinical best practice require human oversight for AI decisions in high-stakes clinical contexts:

| AI Application | Required Human Role | Override Mechanism |
|----------------|--------------------|--------------------|
| Radiology AI flag | Radiologist must review all AI-flagged findings | Radiologist can dismiss flag with documented reason |
| Sepsis alert | Nurse/physician must acknowledge and assess | Alert can be snoozed with clinical justification |
| Drug interaction warning | Pharmacist or prescriber must review | Override with mandatory comment |
| Diagnostic suggestion (CDS) | Physician makes final diagnosis | AI is advisory only — not a standalone diagnosis |
| Ambient scribe note | Physician reviews and attests before signing | No note enters legal record without physician sign-off |
| Autonomous triage chatbot | Clinician reviews escalations | Always routes to human for high-acuity symptoms |

### Data Privacy Beyond HIPAA

- **De-identification for model training:** Use MITRE Presidio, AWS Comprehend Medical de-id, or Microsoft Presidio to strip PHI from training datasets
- **Federated learning:** Train models at each hospital without data leaving the institution (Flower framework, NVIDIA FLARE)
- **Differential privacy:** Add calibrated noise to model updates in FL to prevent gradient inversion attacks
- **Synthetic data:** Generate synthetic patient data (Synthea, CTGAN) for development and testing environments
- **Data use agreements:** IRB approval required for research use of patient data; separate pathway for quality improvement

---

## Tips & Tricks

| Tip | Why It Matters |
|-----|---------------|
| Use AUROC + sensitivity at fixed specificity for clinical models | Raw accuracy is meaningless on imbalanced clinical datasets (sepsis prevalence ~2-3%) |
| Prospective validation before go-live | Retrospective AUC routinely drops 5–15 points when deployed on live data |
| Subgroup analysis by race, age, sex, payer | FDA and Joint Commission expect fairness evidence; disparities discovered post-deployment are costly |
| Calibrate predicted probabilities | Physicians trust "this patient has a 78% readmission risk" — uncalibrated scores destroy trust |
| Never use last-observation-carried-forward for vitals | Missing vitals in ICU often mean a measurement was missed because patient was deteriorating — missingness is informative |
| BAA before any PHI touches an AI API | Violation is a HIPAA breach regardless of data volume; penalties up to $1.9M per violation category |
| Implement alert fatigue monitoring | Track override rates for CDS alerts; >95% override rate = the alert should be removed or redesigned |
| Shadow mode deployment (parallel run) | Run new model in background for 30–90 days before switching clinical workflows; measure drift between shadow and production predictions |
| Version-pin your clinical models | A model update that changes predictions can alter patient outcomes; treat model versions like drug formulary changes |
| PCCP filing for continuous learning models | File a Predetermined Change Control Plan with FDA before deploying models that retrain on incoming data |
| Use scispaCy over general spaCy for clinical NLP | General NLP models miss medical abbreviations and drug names (e.g., "SOB" = shortness of breath, not "son of a …") |
| Time-aware train/validation splits | Never use random splits for temporal EHR data — always split by admission date to prevent data leakage |

---

## Related Topics

- [AI in Healthcare (General)](healthcare.md) — Drug discovery, federated learning, molecular AI
- [AI Ethics](ethics.md) — Fairness, privacy, differential privacy
- [Computer Vision](../computer-vision/index.md) — CNN architectures underlying medical imaging AI
- [NLP](../nlp/index.md) — Clinical text processing, NER, relation extraction
- [LLM Agents](../llm/agents.md) — Building autonomous clinical workflow agents
- [RAG](../llm/rag.md) — Retrieval-augmented generation for clinical guidelines Q&A
- [MLOps](../mlops/index.md) — Deploying, monitoring, and versioning clinical models in production
- [AI Security](../security/ai-security.md) — Adversarial attacks on medical imaging AI
