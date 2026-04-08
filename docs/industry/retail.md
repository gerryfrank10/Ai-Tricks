# 🛒 AI in Retail & E-commerce

From personalized recommendations to real-time inventory optimization, AI drives competitive advantage in retail. This guide covers the full stack — from classic collaborative filtering to modern two-tower neural models and LLM-powered product content.

---

## 🗺️ Overview

| Capability | Technique | Business Impact |
|---|---|---|
| Recommendations | Collaborative filtering, two-tower, LLMs | +15-30 % GMV |
| Demand Forecasting | XGBoost, LightGBM, N-BEATS, TimeMixer | Reduce overstock/stockout |
| Price Optimization | Bandit algorithms, RL | +3-8 % margin |
| Customer Segmentation | K-means, DBSCAN, RFM | Targeted marketing |
| Visual Search | CLIP, ResNet embeddings | Discovery lift |
| Product Descriptions | LLMs (GPT-4o, Claude) | Content velocity |
| Fraud / Returns Abuse | XGBoost, anomaly detection | Loss prevention |

---

## ⭐ Recommendation Systems

### 1. Collaborative Filtering — Matrix Factorization

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(df: pd.DataFrame) -> tuple[csr_matrix, list, list]:
    """Build sparse user-item interaction matrix from purchase/view log."""
    users   = df["user_id"].astype("category")
    items   = df["item_id"].astype("category")
    ratings = df["rating"].values  # implicit: 1=clicked, 5=purchased

    matrix = csr_matrix(
        (ratings, (users.cat.codes, items.cat.codes)),
        shape=(users.cat.categories.size, items.cat.categories.size),
    )
    return matrix, users.cat.categories.tolist(), items.cat.categories.tolist()


def svd_recommendations(
    matrix: csr_matrix,
    user_idx: int,
    n_factors: int = 50,
    top_k: int = 10,
) -> list[int]:
    """SVD-based collaborative filtering."""
    # Subtract row means for mean-centering
    dense = matrix.toarray().astype(float)
    user_ratings_mean = dense.mean(axis=1, keepdims=True)
    centered = dense - user_ratings_mean

    U, sigma, Vt = svds(centered, k=n_factors)
    Sigma = np.diag(sigma)

    # Predict scores for all items
    predicted = U @ Sigma @ Vt + user_ratings_mean
    user_preds = predicted[user_idx]

    # Mask already-interacted items
    interacted = matrix[user_idx].nonzero()[1]
    user_preds[interacted] = -np.inf

    return np.argsort(user_preds)[::-1][:top_k].tolist()
```

### 2. Implicit Feedback with ALS (Alternating Least Squares)

```python
import implicit
from scipy.sparse import csr_matrix

def train_als_model(user_item_matrix: csr_matrix) -> implicit.als.AlternatingLeastSquares:
    """Train ALS model for implicit feedback (views, clicks, purchases)."""
    model = implicit.als.AlternatingLeastSquares(
        factors=128,
        regularization=0.05,
        iterations=30,
        calculate_training_loss=True,
        use_gpu=False,
    )
    # ALS expects item-user matrix
    model.fit(user_item_matrix.T)
    return model

def get_recommendations(
    model: implicit.als.AlternatingLeastSquares,
    user_item_matrix: csr_matrix,
    user_id: int,
    n: int = 10,
) -> list[tuple[int, float]]:
    ids, scores = model.recommend(
        user_id, user_item_matrix[user_id], N=n, filter_already_liked_items=True
    )
    return list(zip(ids.tolist(), scores.tolist()))
```

### 3. Two-Tower Neural Model (Modern Industry Standard)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, n_users: int, n_cats: int, embed_dim: int = 64):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.cat_embed  = nn.Embedding(n_cats,  16)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 16 + 4, 128), nn.ReLU(),  # +4 for age, gender, tenure, device
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, user_ids, cat_ids, user_feats):
        u = self.user_embed(user_ids)
        c = self.cat_embed(cat_ids)
        x = torch.cat([u, c, user_feats], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items: int, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.item_embed  = nn.Embedding(n_items, embed_dim)
        self.title_embed = nn.EmbeddingBag(vocab_size, 32, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 32 + 5, 128), nn.ReLU(),  # +5: price,rating,reviews,age,stock
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, item_ids, title_tokens, item_feats):
        i = self.item_embed(item_ids)
        t = self.title_embed(title_tokens)
        x = torch.cat([i, t, item_feats], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower, temperature: float = 0.07):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, user_ids, cat_ids, user_feats, item_ids, title_tokens, item_feats):
        u_emb = self.user_tower(user_ids, cat_ids, user_feats)
        i_emb = self.item_tower(item_ids, title_tokens, item_feats)
        # Contrastive logits (in-batch negatives)
        logits = (u_emb @ i_emb.T) / self.temperature
        return logits, u_emb, i_emb
```

---

## 📦 Demand Forecasting

### XGBoost with Lag Features

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error

def create_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    # Rolling stats
    df["rolling_mean_7"]  = df[target].shift(1).rolling(7).mean()
    df["rolling_std_7"]   = df[target].shift(1).rolling(7).std()
    df["rolling_mean_28"] = df[target].shift(1).rolling(28).mean()
    # Calendar features
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    return df.dropna()

# Train/test split (time-aware)
def time_split(df: pd.DataFrame, test_weeks: int = 8):
    cutoff = df.index.max() - pd.Timedelta(weeks=test_weeks)
    return df.loc[:cutoff], df.loc[cutoff:]

# Model
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=42,
)
# model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
# mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
```

---

## 💲 Price Optimization

### Thompson Sampling for Dynamic Pricing

```python
import numpy as np

class ThompsonSamplingPricer:
    """Multi-armed bandit for discrete price tiers."""
    def __init__(self, prices: list[float]):
        self.prices  = prices
        self.n_arms  = len(prices)
        # Beta distribution parameters (successes, failures = conversions / non-conversions)
        self.alpha   = np.ones(self.n_arms)
        self.beta    = np.ones(self.n_arms)
        self.revenue = np.zeros(self.n_arms)
        self.trials  = np.zeros(self.n_arms)

    def select_price(self) -> tuple[int, float]:
        """Sample conversion rate from Beta posterior, pick price maximizing E[revenue]."""
        sampled_rates = np.random.beta(self.alpha, self.beta)
        expected_rev  = sampled_rates * np.array(self.prices)
        arm = int(np.argmax(expected_rev))
        return arm, self.prices[arm]

    def update(self, arm: int, price: float, converted: bool, quantity: int = 1):
        if converted:
            self.alpha[arm]   += quantity
            self.revenue[arm] += price * quantity
        else:
            self.beta[arm] += 1
        self.trials[arm] += 1

    def get_stats(self) -> dict:
        conv_rates = self.alpha / (self.alpha + self.beta)
        return {
            p: {"conv_rate": round(cr, 3), "trials": int(t), "revenue": round(r, 2)}
            for p, cr, t, r in zip(self.prices, conv_rates, self.trials, self.revenue)
        }
```

---

## 👥 Customer Segmentation (RFM + K-Means)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

def compute_rfm(orders: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary value per customer."""
    rfm = orders.groupby("customer_id").agg(
        recency   = ("order_date", lambda d: (reference_date - d.max()).days),
        frequency = ("order_id",   "count"),
        monetary  = ("total_amount", "sum"),
    ).reset_index()
    return rfm

def segment_customers(rfm: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    scaler   = StandardScaler()
    features = rfm[["recency", "frequency", "monetary"]]
    scaled   = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["segment"] = km.fit_predict(scaled)

    # Label segments by centroid characteristics
    centers = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=["recency", "frequency", "monetary"],
    )
    labels = {
        centers["monetary"].idxmax(): "Champions",
        centers["frequency"].idxmax(): "Loyal Customers",
        centers["recency"].idxmin():   "Recent Customers",
        centers["recency"].idxmax():   "At Risk",
    }
    rfm["segment_name"] = rfm["segment"].map(labels).fillna("Potential Loyalists")
    return rfm
```

---

## 🤖 LLMs for Product Descriptions

```python
from openai import OpenAI
import json

client = OpenAI()

PRODUCT_PROMPT = """Generate a compelling product description for an e-commerce listing.

Product data:
{product_json}

Requirements:
- 2-3 sentence headline description (engaging, benefit-focused)
- 5 bullet points (key features, materials, sizing, care)
- SEO meta description (155 chars max)
- Tone: {tone}

Output JSON with keys: headline, bullets (list), meta_description"""

def generate_product_content(
    product: dict,
    tone: str = "professional yet friendly",
) -> dict:
    prompt = PRODUCT_PROMPT.format(
        product_json=json.dumps(product, indent=2),
        tone=tone,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(response.choices[0].message.content)

# Example
product = {
    "name": "Merino Wool Crew Neck Sweater",
    "material": "100% Merino Wool",
    "colors": ["Navy", "Charcoal", "Ivory"],
    "sizes": ["XS", "S", "M", "L", "XL"],
    "price": 129.99,
    "category": "Men's Knitwear",
}
# content = generate_product_content(product)
```

---

## 🔍 Visual Search with CLIP

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path

class VisualSearchEngine:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index: np.ndarray | None = None
        self.item_ids:  list = []

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb    = self.model.get_image_features(**inputs)
        return emb.cpu().numpy() / np.linalg.norm(emb.cpu().numpy())

    def build_index(self, image_paths: list[Path], item_ids: list[str]):
        embeddings = [self.embed_image(Image.open(p).convert("RGB")) for p in image_paths]
        self.index    = np.vstack(embeddings)
        self.item_ids = item_ids

    def search(self, query_image: Image.Image, top_k: int = 10) -> list[dict]:
        q_emb     = self.embed_image(query_image)
        scores    = self.index @ q_emb.T
        top_idxs  = np.argsort(scores.ravel())[::-1][:top_k]
        return [
            {"item_id": self.item_ids[i], "score": float(scores[i])}
            for i in top_idxs
        ]
```

---

## 💡 Retail AI Tips

| Tip | Context |
|---|---|
| Separate explore/exploit in recs | Pure exploitation kills long-tail discovery |
| Use session-based recs for cold start | New users have no history — use current session context |
| A/B test with holdout groups | Never roll out rec changes without an experiment |
| Feature freshness matters | Stale item embeddings hurt sales of new products |
| Demand forecast by (SKU, store, channel) | Hierarchy matters for inventory placement |
| Multi-touch attribution | Last-click inflates direct channel contribution |
| Privacy: differential privacy for user data | GDPR/CCPA compliance for personalization |

---

## 📚 Further Reading

- [RecSys conference proceedings](https://recsys.acm.org/)
- [Recommender Systems Handbook (Ricci et al.)](https://link.springer.com/book/10.1007/978-1-0716-2197-4)
- [Meta's Two-Tower paper (2022)](https://arxiv.org/abs/2208.09257)
- [N-BEATS for forecasting](https://arxiv.org/abs/1905.10437)
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [Implicit library (ALS)](https://github.com/benfred/implicit)
