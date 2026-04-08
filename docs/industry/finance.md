# 💰 AI in Finance

AI is transforming every layer of financial services — from catching fraudulent card swipes in milliseconds to generating analyst-grade earnings summaries with LLMs. This guide covers production-grade techniques used by quant desks, fintechs, and banks in 2025.

---

## 🗺️ Overview

| Domain | Key Techniques | Typical Stack |
|---|---|---|
| Fraud Detection | XGBoost, Isolation Forest, GNN | scikit-learn, LightGBM, PyG |
| Algorithmic Trading | RL, time-series models, factor models | backtrader, vectorbt, zipline-reloaded |
| Credit Scoring | Logistic Regression, XGBoost, SHAP | sklearn, SHAP, Optuna |
| Risk Modeling | Monte Carlo, CVaR, deep learning | numpy, scipy, PyTorch |
| NLP for Finance | FinBERT, LLMs, sentiment analysis | HuggingFace, LangChain |
| Compliance / RegTech | Entity extraction, document parsing | spaCy, LLM APIs |

---

## 🛡️ Fraud Detection with XGBoost

### Dataset Setup

The classic benchmark is the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, only 0.17 % fraudulent (severe class imbalance).

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import xgboost as xgb

# Load data
df = pd.read_csv("creditcard.csv")

# Separate features / target
X = df.drop("Class", axis=1)
y = df["Class"]

# Time & Amount need scaling; V1-V28 are already PCA-transformed
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Class imbalance: use scale_pos_weight
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale = neg / pos  # ~578

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,        # handles imbalance
    use_label_encoder=False,
    eval_metric="aucpr",           # area under PR curve is better than ROC for imbalanced
    tree_method="hist",            # fast histogram-based
    random_state=42,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

y_prob = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, y_prob):.4f}")
print(classification_report(y_test, model.predict(X_test)))
```

### Feature Engineering for Transactions

```python
def engineer_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavioral features from raw card transactions."""
    df = df.sort_values(["card_id", "transaction_time"]).copy()

    # Rolling spend velocity
    df["spend_7d"] = (
        df.groupby("card_id")["amount"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).sum())
    )

    # Transaction frequency in last hour
    df["txn_count_1h"] = (
        df.groupby("card_id")["transaction_time"]
        .transform(lambda t: t.expanding().count())
    )

    # Deviation from personal average
    personal_mean = df.groupby("card_id")["amount"].transform("mean")
    personal_std  = df.groupby("card_id")["amount"].transform("std").fillna(1)
    df["amount_zscore"] = (df["amount"] - personal_mean) / personal_std

    # New merchant category flag
    seen = df.groupby("card_id")["mcc"].transform(
        lambda s: s.duplicated(keep="first").astype(int)
    )
    df["new_mcc"] = 1 - seen

    return df
```

> **Tip:** In production, features must be computed from a streaming window (Flink, Kafka Streams) to avoid look-ahead bias. Offline backtests should replay events in strict chronological order.

---

## 📈 Algorithmic Trading Basics

### Factor-Based Strategy with vectorbt

```python
import vectorbt as vbt
import pandas as pd
import yfinance as yf

# Download OHLCV data
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
raw = yf.download(tickers, start="2020-01-01", end="2024-12-31", auto_adjust=True)
close = raw["Close"]

# Momentum signal: 12-1 month cross-sectional momentum
returns_12m = close.pct_change(252)
returns_1m  = close.pct_change(21)
momentum    = returns_12m - returns_1m  # skip most-recent month

# Long top-2, short bottom-2 each month (rebalance monthly)
entries = momentum.rank(axis=1, ascending=False) <= 2   # top 2
exits   = momentum.rank(axis=1, ascending=True)  <= 2   # bottom 2

portfolio = vbt.Portfolio.from_signals(
    close,
    entries=entries,
    exits=exits,
    freq="D",
    init_cash=100_000,
    fees=0.001,          # 10 bps per trade
    slippage=0.001,
)

print(portfolio.stats())
portfolio.plot().show()
```

### Key Risk Metrics

```python
import numpy as np

def compute_risk_metrics(returns: np.ndarray, risk_free: float = 0.05) -> dict:
    """Compute standard portfolio risk metrics."""
    daily_rf = risk_free / 252
    excess   = returns - daily_rf

    sharpe   = (excess.mean() / excess.std()) * np.sqrt(252)
    sortino  = (excess.mean() / excess[excess < 0].std()) * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # CVaR (Expected Shortfall) at 95 %
    var_95  = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    return {
        "sharpe":   round(sharpe, 3),
        "sortino":  round(sortino, 3),
        "max_dd":   round(max_dd, 4),
        "var_95":   round(var_95, 4),
        "cvar_95":  round(cvar_95, 4),
        "ann_ret":  round(returns.mean() * 252, 4),
        "ann_vol":  round(returns.std() * np.sqrt(252), 4),
    }
```

---

## 🏦 Credit Scoring

### Logistic Regression Baseline + SHAP Explainability

```python
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Pipeline ensures no data leakage
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )),
])
pipe.fit(X_train, y_train)

# SHAP explanations (TreeExplainer works on the inner GBM)
explainer    = shap.TreeExplainer(pipe.named_steps["model"])
X_test_scaled = pipe.named_steps["scaler"].transform(X_test)
shap_values  = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test_scaled, feature_names=X_test.columns.tolist())
```

> **Regulatory Note:** In the EU (CRD/CRR) and US (ECOA), lenders must provide adverse-action notices. SHAP values help generate human-readable reasons ("Your debt-to-income ratio of 0.62 exceeds our threshold").

---

## 📰 NLP for Earnings Calls & Financial Documents

### FinBERT Sentiment on 10-K / Earnings Transcripts

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "ProsusAI/finbert"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
)

sentences = [
    "Revenue grew 18 % year-over-year driven by strong enterprise demand.",
    "We expect significant headwinds from rising interest rates in the next quarter.",
    "The Board declared a quarterly dividend of $0.42 per share.",
]

for sent in sentences:
    result = sentiment_pipe(sent[:512])[0]
    scores = {r["label"]: round(r["score"], 3) for r in result}
    dominant = max(scores, key=scores.get)
    print(f"[{dominant:>8}] {sent[:60]}...")
```

### LLM-Powered Earnings Call Summarizer

```python
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a senior equity analyst. Given an earnings call transcript,
produce a structured summary with:
1. KEY METRICS (revenue, EPS, guidance — actual vs. consensus)
2. MANAGEMENT TONE (bullish/neutral/cautious, supporting quotes)
3. RISKS MENTIONED
4. ANALYST CONCERNS from Q&A
5. ONE-LINE VERDICT

Be concise. Use numbers. Flag any discrepancies between guidance and analyst expectations."""

def summarize_earnings_call(transcript: str, ticker: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Ticker: {ticker}\n\nTranscript:\n{transcript}"},
        ],
        temperature=0.1,   # low temp for factual extraction
        max_tokens=800,
    )
    return response.choices[0].message.content

# Usage
# summary = summarize_earnings_call(open("NVDA_Q4_2024.txt").read(), "NVDA")
```

---

## ⚙️ Risk Models

### Monte Carlo VaR Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_var(
    portfolio_value: float,
    daily_returns: np.ndarray,
    n_simulations: int = 10_000,
    horizon_days: int = 10,
    confidence: float = 0.95,
) -> dict:
    mu    = daily_returns.mean()
    sigma = daily_returns.std()

    # Simulate correlated paths
    rng       = np.random.default_rng(42)
    sim_rets  = rng.normal(mu, sigma, (n_simulations, horizon_days))
    sim_paths = portfolio_value * np.cumprod(1 + sim_rets, axis=1)
    final_pnl = sim_paths[:, -1] - portfolio_value

    var  = np.percentile(final_pnl, (1 - confidence) * 100)
    cvar = final_pnl[final_pnl <= var].mean()

    return {
        "var":  round(var, 2),    # negative = loss
        "cvar": round(cvar, 2),
        "pct_var":  round(var / portfolio_value * 100, 2),
    }

# Example: $1M portfolio
result = monte_carlo_var(1_000_000, np.random.normal(0.0005, 0.012, 1000))
print(result)  # e.g., {'var': -32145.0, 'cvar': -44210.0, 'pct_var': -3.21}
```

---

## ✅ Compliance & Regulatory Considerations

| Regulation | Jurisdiction | AI Impact |
|---|---|---|
| SR 11-7 (Model Risk Mgmt) | US (Fed/OCC) | Model validation, documentation |
| GDPR / AI Act | EU | Explainability, right to contest automated decisions |
| ECOA / Fair Lending | US | Bias testing, disparate impact analysis |
| MiFID II | EU | Algorithmic trading registration, kill switches |
| Basel III / FRTB | Global | VaR models, backtesting requirements |
| DORA (2025) | EU | Operational resilience, third-party AI risk |

### Bias Audit with Fairlearn

```python
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate
from sklearn.metrics import accuracy_score

sensitive_feature = X_test["applicant_race"]  # protected attribute

mf = MetricFrame(
    metrics={
        "accuracy":    accuracy_score,
        "selection_rate": selection_rate,
        "fpr":         false_positive_rate,
    },
    y_true=y_test,
    y_pred=model.predict(X_test),
    sensitive_features=sensitive_feature,
)

print(mf.by_group)
print("\nDisparate impact ratio:", mf.ratio(method="between_groups"))
```

---

## 💡 Pro Tips

| Tip | Why It Matters |
|---|---|
| Use PR-AUC not ROC-AUC for fraud | Class imbalance makes ROC-AUC misleadingly optimistic |
| Feature stores (Feast, Hopsworks) | Share features between training and real-time inference |
| Walk-forward validation | Time-series splits prevent look-ahead bias in trading |
| SHAP for adverse action notices | Regulatory compliance + customer trust |
| Stress-test models on 2008, 2020 | Distribution shifts during crises are severe |
| Shadow mode before prod rollout | Run new fraud model in parallel before switching traffic |
| Separate alpha from execution | Strategy signal quality != live trading performance |

---

## 📚 Further Reading

- [FinRL: Deep RL for trading](https://github.com/AI4Finance-Foundation/FinRL)
- [FinBERT paper (Araci 2019)](https://arxiv.org/abs/1908.10063)
- [Advances in Financial Machine Learning — de Prado](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [BloombergGPT (2023)](https://arxiv.org/abs/2303.17564)
- [SEC EDGAR full-text search API](https://efts.sec.gov/LATEST/search-index?q=%22artificial+intelligence%22&dateRange=custom&startdt=2024-01-01)
