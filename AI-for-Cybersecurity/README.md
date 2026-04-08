# AI for Cybersecurity

AI is transforming cybersecurity on both sides of the equation — defenders use ML to detect threats faster than any human analyst, while attackers leverage AI to scale and automate attacks. This section covers practical AI techniques for blue team (defense), threat intelligence, and security automation.

---

## 📖 **Sections**

- [AI in the Security Stack](#ai-in-the-security-stack)
- [Anomaly Detection](#anomaly-detection)
- [Malware Classification](#malware-classification)
- [Network Intrusion Detection](#network-intrusion-detection)
- [Log Analysis with LLMs](#log-analysis-with-llms)
- [Threat Intelligence Automation](#threat-intelligence-automation)
- [Vulnerability Triage](#vulnerability-triage)
- [AI-Powered SIEM](#ai-powered-siem)

---

## 🗺️ **AI in the Security Stack**

```
         ATTACKER                    DEFENDER (AI-Enhanced)
    ─────────────────               ─────────────────────────────
    Recon & Scanning        ──►     Asset Discovery & Exposure Mgmt
    Phishing / Social Eng   ──►     Email Filtering (NLP classifiers)
    Exploit & Intrude       ──►     IDS/IPS (ML anomaly detection)
    Lateral Movement        ──►     User/Entity Behavior Analytics (UEBA)
    Exfiltration            ──►     DLP (data pattern detection)
    Persistence             ──►     Endpoint Detection (sequence models)
    C2 Communication        ──►     Network Traffic Analysis (DNN)
```

---

## 🚨 **Anomaly Detection**

### User & Entity Behavior Analytics (UEBA)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulate user login data
def generate_user_logs(n_users: int = 100, n_days: int = 30) -> pd.DataFrame:
    records = []
    for user_id in range(n_users):
        # Normal behavior
        typical_hour = np.random.randint(8, 18)  # Business hours
        typical_country = "US"

        for day in range(n_days):
            n_logins = np.random.poisson(3)
            for _ in range(n_logins):
                hour = int(np.clip(np.random.normal(typical_hour, 2), 0, 23))
                records.append({
                    "user_id": user_id,
                    "day": day,
                    "hour": hour,
                    "country": typical_country,
                    "failed_attempts": np.random.poisson(0.1),
                    "data_downloaded_mb": np.random.exponential(10),
                    "is_anomaly": False
                })

    # Inject anomalies for user 5
    for _ in range(5):
        records.append({
            "user_id": 5,
            "day": 28,
            "hour": 3,           # Unusual hour
            "country": "RU",     # Unusual country
            "failed_attempts": 8, # Many failures
            "data_downloaded_mb": 500,  # Unusual volume
            "is_anomaly": True
        })

    return pd.DataFrame(records)

df = generate_user_logs()

# Feature engineering per user
def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.groupby("user_id").agg(
        avg_login_hour=("hour", "mean"),
        std_login_hour=("hour", "std"),
        avg_failed_attempts=("failed_attempts", "mean"),
        max_failed_attempts=("failed_attempts", "max"),
        avg_data_downloaded=("data_downloaded_mb", "mean"),
        max_data_downloaded=("data_downloaded_mb", "max"),
        unique_countries=("country", "nunique"),
        login_frequency=("day", "count"),
    ).fillna(0)
    return features

user_features = extract_user_features(df)
scaler = StandardScaler()
X = scaler.fit_transform(user_features)

# Isolation Forest: no labels needed (unsupervised)
clf = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # Expected % of anomalies
    random_state=42
)
clf.fit(X)

scores = clf.decision_function(X)
predictions = clf.predict(X)  # -1 = anomaly, 1 = normal

user_features["anomaly_score"] = scores
user_features["is_anomaly"] = predictions == -1

# Alert on suspicious users
anomalous_users = user_features[user_features["is_anomaly"]].sort_values("anomaly_score")
print("Suspicious users detected:")
print(anomalous_users[["avg_login_hour", "max_failed_attempts", "max_data_downloaded", "unique_countries"]].to_string())
```

### Time-Series Anomaly Detection (Network Traffic)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

class LSTMAnomalyDetector(nn.Module):
    """
    LSTM autoencoder for network traffic anomaly detection.
    High reconstruction error = anomalous traffic pattern.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 64, seq_len: int = 60):
        super().__init__()
        self.seq_len = seq_len

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Decode
        decoder_input = hidden.permute(1, 0, 2).repeat(1, x.size(1), 1)
        output, _ = self.decoder(decoder_input)
        return self.output_layer(output)

def detect_traffic_anomalies(traffic_series: np.ndarray,
                              seq_len: int = 60,
                              threshold_std: float = 3.0) -> dict:
    """Detect anomalous periods in network traffic using LSTM autoencoder."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(traffic_series.reshape(-1, 1))

    # Create sequences
    sequences = []
    for i in range(len(scaled) - seq_len):
        sequences.append(scaled[i:i + seq_len])
    X = torch.FloatTensor(np.array(sequences))

    # Train (in practice, train only on known-good traffic)
    model = LSTMAnomalyDetector(seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for seq in X:
            optimizer.zero_grad()
            reconstructed = model(seq.unsqueeze(0))
            loss = nn.MSELoss()(reconstructed, seq.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Calculate reconstruction errors
    model.eval()
    errors = []
    with torch.no_grad():
        for seq in X:
            reconstructed = model(seq.unsqueeze(0))
            error = nn.MSELoss()(reconstructed, seq.unsqueeze(0)).item()
            errors.append(error)

    errors = np.array(errors)
    threshold = errors.mean() + threshold_std * errors.std()

    anomaly_indices = np.where(errors > threshold)[0]
    print(f"Threshold: {threshold:.6f}")
    print(f"Anomalies detected at {len(anomaly_indices)} time windows")

    return {
        "errors": errors,
        "threshold": threshold,
        "anomaly_indices": anomaly_indices.tolist()
    }
```

---

## 🦠 **Malware Classification**

### Static Analysis with ML

```python
import pefile
import hashlib
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def extract_pe_features(file_path: str) -> dict:
    """
    Extract static features from a PE (Windows executable) file.
    No execution needed — safe to analyze malware samples.
    """
    features = {}

    try:
        pe = pefile.PE(file_path)

        # Header features
        features["machine_type"] = pe.FILE_HEADER.Machine
        features["num_sections"] = pe.FILE_HEADER.NumberOfSections
        features["timestamp"] = pe.FILE_HEADER.TimeDateStamp
        features["characteristics"] = pe.FILE_HEADER.Characteristics

        # Section features
        section_sizes = [s.SizeOfRawData for s in pe.sections]
        features["avg_section_size"] = np.mean(section_sizes) if section_sizes else 0
        features["max_section_size"] = max(section_sizes) if section_sizes else 0
        features["num_sections"] = len(section_sizes)

        # Import features
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            imports = []
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        imports.append(imp.name.decode("utf-8", errors="ignore"))
            features["num_imports"] = len(imports)

            # Suspicious imports
            suspicious = ["CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory",
                          "SetWindowsHookEx", "OpenProcess", "NtCreateThreadEx"]
            features["suspicious_imports"] = sum(1 for i in imports if i in suspicious)
        else:
            features["num_imports"] = 0
            features["suspicious_imports"] = 0

        # Entropy (high entropy in code section = possible packing/encryption)
        for section in pe.sections:
            name = section.Name.decode("utf-8", errors="ignore").rstrip("\x00")
            if ".text" in name or "CODE" in name:
                data = section.get_data()
                features["code_entropy"] = calculate_entropy(data)

    except Exception as e:
        features["parse_error"] = 1

    return features

def calculate_entropy(data: bytes) -> float:
    """Shannon entropy of byte data."""
    if not data:
        return 0
    freq = np.zeros(256)
    for byte in data:
        freq[byte] += 1
    freq = freq[freq > 0] / len(data)
    return -np.sum(freq * np.log2(freq))

# Train classifier on labeled PE files
# Features extracted from known malware/benign samples
def train_malware_classifier(malware_files: list[str], benign_files: list[str]):
    X, y = [], []

    for f in malware_files:
        feats = extract_pe_features(f)
        X.append(list(feats.values()))
        y.append(1)  # malware

    for f in benign_files:
        feats = extract_pe_features(f)
        X.append(list(feats.values()))
        y.append(0)  # benign

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    return clf
```

---

## 🌐 **Network Intrusion Detection**

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def build_nids_model(network_logs_path: str):
    """
    Network Intrusion Detection System using network flow features.
    Works with NSL-KDD, CICIDS2017, or similar datasets.
    """
    df = pd.read_csv(network_logs_path)

    # Common network flow features
    feature_cols = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells",
        "num_access_files", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate"
    ]

    # Encode categorical features
    le = LabelEncoder()
    for col in ["protocol_type", "service", "flag"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Binary classification: normal vs attack
    df["is_attack"] = (df["label"] != "normal").astype(int)

    X = df[[c for c in feature_cols if c in df.columns]]
    y = df["is_attack"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop 10 most important features:")
    print(importances.nlargest(10).to_string())

    return model
```

---

## 📋 **Log Analysis with LLMs**

```python
import anthropic
import re
from datetime import datetime

client = anthropic.Anthropic()

def analyze_security_logs(log_lines: list[str], context: str = "") -> dict:
    """Use LLM to analyze security logs for threats."""

    # Pre-filter to reduce noise
    relevant_patterns = [
        r"failed|failure|error|denied|blocked",
        r"admin|root|sudo|privilege",
        r"login|auth|session",
        r"sql|injection|xss|traversal",
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
    ]

    filtered = [
        line for line in log_lines
        if any(re.search(p, line, re.IGNORECASE) for p in relevant_patterns)
    ]

    if not filtered:
        return {"status": "no_relevant_events", "threats": []}

    logs_text = "\n".join(filtered[:200])  # Limit tokens

    prompt = f"""You are a senior security analyst reviewing logs.
Analyze these logs and identify:
1. Security threats or incidents (list each with severity: CRITICAL/HIGH/MEDIUM/LOW)
2. Attack patterns (brute force, SQLi, XSS, privilege escalation, etc.)
3. Suspicious IP addresses or user accounts
4. Recommended immediate actions

{f"Context: {context}" if context else ""}

LOGS:
{logs_text}

Respond as JSON:
{{
  "threats": [{{"severity": "HIGH", "type": "brute_force", "description": "...", "indicators": ["ip", "user"]}}],
  "attack_patterns": ["..."],
  "suspicious_entities": {{"ips": [], "users": []}},
  "immediate_actions": ["..."],
  "summary": "..."
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"raw_analysis": response.content[0].text}

# Example usage
sample_logs = [
    "2025-01-15 03:42:11 Failed password for root from 192.168.1.55 port 22 ssh2",
    "2025-01-15 03:42:12 Failed password for root from 192.168.1.55 port 22 ssh2",
    "2025-01-15 03:42:13 Failed password for root from 192.168.1.55 port 22 ssh2",
    "2025-01-15 03:42:50 Failed password for admin from 192.168.1.55 port 22 ssh2",
    "2025-01-15 03:43:01 Accepted password for admin from 192.168.1.55 port 22 ssh2",
    "2025-01-15 03:45:00 sudo: admin : TTY=pts/0 ; USER=root ; COMMAND=/bin/bash",
    "2025-01-15 03:47:33 scp from 192.168.1.55: /etc/shadow /etc/passwd",
]

result = analyze_security_logs(sample_logs, context="Production Linux server")
print(f"Threats found: {len(result.get('threats', []))}")
for threat in result.get("threats", []):
    print(f"  [{threat['severity']}] {threat['type']}: {threat['description']}")
```

---

## 🔍 **Threat Intelligence Automation**

```python
import anthropic
import requests
from dataclasses import dataclass

client = anthropic.Anthropic()

@dataclass
class ThreatIndicator:
    type: str       # ip, domain, hash, url, email
    value: str
    severity: str   # critical, high, medium, low
    tags: list[str]
    description: str

def enrich_ioc(indicator_type: str, indicator_value: str) -> dict:
    """
    Enrich an Indicator of Compromise (IOC) with threat intelligence.
    In production, integrate with VirusTotal, Shodan, AbuseIPDB, etc.
    """
    # Simulate threat intel lookup
    mock_intel = {
        "reputation_score": 85,  # 0-100, higher = more malicious
        "first_seen": "2024-06-15",
        "last_seen": "2025-01-10",
        "reported_by": ["AbuseIPDB", "Spamhaus"],
        "associated_malware": ["Emotet", "TrickBot"],
        "geolocation": {"country": "RU", "city": "Moscow"},
        "asn": "AS12345 - ShadyHost",
        "open_ports": [22, 80, 443, 8080],
        "abuse_categories": ["port_scan", "brute_force", "malware_c2"]
    }

    return mock_intel

def generate_threat_report(indicators: list[ThreatIndicator]) -> str:
    """Use LLM to generate human-readable threat report."""
    ioc_details = []
    for ioc in indicators:
        intel = enrich_ioc(ioc.type, ioc.value)
        ioc_details.append(f"""
IOC: {ioc.value} (Type: {ioc.type})
- Severity: {ioc.severity}
- Reputation Score: {intel['reputation_score']}/100
- Associated Malware: {', '.join(intel['associated_malware'])}
- Abuse Categories: {', '.join(intel['abuse_categories'])}
- Geolocation: {intel['geolocation']['country']}
""")

    prompt = f"""You are a threat intelligence analyst. Based on these IOCs,
write an executive threat intelligence brief that includes:
1. Threat actor attribution (if possible)
2. Attack campaign assessment
3. Business risk level
4. Specific mitigation recommendations
5. Detection rules summary

IOC Details:
{''.join(ioc_details)}

Write for a CISO audience — clear, actionable, jargon-minimal."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Example
iocs = [
    ThreatIndicator("ip", "192.168.1.55", "high", ["brute_force", "c2"], "Repeated SSH failures"),
    ThreatIndicator("hash", "d41d8cd98f00b204e9800998ecf8427e", "critical", ["ransomware"], "Ransomware payload"),
]

report = generate_threat_report(iocs)
print(report)
```

---

## 🐛 **Vulnerability Triage**

```python
def triage_vulnerabilities(cve_list: list[str], system_context: str) -> list[dict]:
    """
    Prioritize CVEs based on context using LLM reasoning.
    Goes beyond CVSS score to consider actual exploitability in your environment.
    """
    cve_info = "\n".join(f"- {cve}" for cve in cve_list)

    prompt = f"""You are a vulnerability management specialist.
Triage these CVEs for the following system context.

System Context: {system_context}

CVEs to triage: {cve_info}

For each CVE, assess:
1. Priority (P1-Critical / P2-High / P3-Medium / P4-Low)
2. Exploitability in this specific context
3. Business impact if exploited
4. Patch urgency (immediate/within 7 days/within 30 days/next cycle)
5. Compensating controls that reduce risk

Return as JSON list of objects with fields:
cve_id, priority, exploitability_score (0-10), business_impact, patch_urgency, compensating_controls, reasoning"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return [{"raw": response.content[0].text}]

context = """
Production web application running:
- Apache 2.4.50 on Ubuntu 22.04
- PostgreSQL 14.2
- Python 3.10 Flask API
- Internet-facing, handles PII
- WAF in front: Cloudflare
- No direct OS access from the app
"""

vulnerabilities = triage_vulnerabilities(
    ["CVE-2021-41773", "CVE-2022-0847", "CVE-2023-44487"],
    system_context=context
)

for vuln in vulnerabilities:
    print(f"\n{vuln.get('cve_id', 'N/A')} — Priority: {vuln.get('priority', 'N/A')}")
    print(f"  Patch urgency: {vuln.get('patch_urgency', 'N/A')}")
    print(f"  Reasoning: {vuln.get('reasoning', '')[:200]}")
```

---

## 🖥️ **AI-Powered SIEM**

```python
from collections import defaultdict
from datetime import datetime, timedelta
import anthropic

class AIEnhancedSIEM:
    """
    Simple AI-enhanced SIEM that correlates events and generates alerts.
    """

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.event_store = defaultdict(list)
        self.alerts = []

    def ingest_event(self, event: dict):
        """Ingest a security event."""
        source_ip = event.get("source_ip", "unknown")
        self.event_store[source_ip].append({
            **event,
            "timestamp": datetime.now().isoformat()
        })

        # Trigger correlation rules
        self._check_rules(source_ip)

    def _check_rules(self, source_ip: str):
        """Check correlation rules and trigger AI analysis if needed."""
        events = self.event_store[source_ip]

        # Rule: 5+ failed logins in 2 minutes
        recent_failures = [
            e for e in events
            if e.get("event_type") == "login_failure"
            and (datetime.now() - datetime.fromisoformat(e["timestamp"])).seconds < 120
        ]

        if len(recent_failures) >= 5:
            self._create_ai_alert(source_ip, events[-20:], "brute_force_suspected")

        # Rule: Login after multiple failures
        if (events and events[-1].get("event_type") == "login_success"
                and len(recent_failures) >= 3):
            self._create_ai_alert(source_ip, events[-10:], "account_compromise_suspected")

    def _create_ai_alert(self, source_ip: str, events: list, rule_triggered: str):
        """Use LLM to analyze events and create enriched alert."""
        events_summary = "\n".join(
            f"[{e['timestamp']}] {e.get('event_type')} - {e.get('details', '')}"
            for e in events
        )

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Analyze these security events from IP {source_ip}.
Rule triggered: {rule_triggered}

Events:
{events_summary}

Provide:
1. Threat assessment (1-2 sentences)
2. Confidence level (high/medium/low)
3. Recommended action (block/investigate/monitor/false_positive)
4. Key evidence

Be concise and direct."""
            }]
        )

        alert = {
            "alert_id": f"ALERT-{len(self.alerts)+1:04d}",
            "source_ip": source_ip,
            "rule": rule_triggered,
            "ai_analysis": response.content[0].text,
            "event_count": len(events),
            "timestamp": datetime.now().isoformat(),
            "status": "open"
        }

        self.alerts.append(alert)
        print(f"\n[ALERT] {alert['alert_id']} - {rule_triggered} from {source_ip}")
        print(f"AI Analysis: {response.content[0].text[:300]}")
        return alert
```

---

## 💡 **Tips & Tricks**

1. **False positive reduction**: Use anomaly detection as a pre-filter, then LLM for final judgment — this dramatically reduces analyst fatigue
2. **Context is everything**: The same CVE can be critical or low risk depending on your stack — always contextualize
3. **Automate the easy 80%**: Automate triage and enrichment; reserve human analysts for novel/complex incidents
4. **Feature engineering for NIDS**: Packet-level features (packet size distribution, inter-arrival times) catch encrypted malware better than payload inspection
5. **Log everything, analyze selectively**: Collect all logs but use AI to highlight what actually matters
6. **Adversarial robustness**: Security ML models face adversarial ML attacks — test with evasion techniques
7. **Chain of custody**: Any AI decision that triggers a response (block IP, kill process) needs an immutable audit trail

---

## 🔗 **Related Topics**

- [AI Security & Red Teaming](../AI-Security/README.md)
- [Anomaly Detection](../Machine%20Learning/Unsupervised-Learning.md)
- [Natural Language Processing](../Natural%20Language%20Processing/Text-Preprocessing.md)
- [MLOps & Deployment](../MLOps/README.md)
