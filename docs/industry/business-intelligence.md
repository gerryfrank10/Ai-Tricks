# Business Intelligence with AI

AI-powered BI moves beyond static dashboards to automated insights, natural language queries, and proactive anomaly alerts.

---

## The AI-Enhanced BI Stack (2025)

```
Data Warehouse (Snowflake / BigQuery / DuckDB)
        │
        ▼
dbt (Transform) + Great Expectations (Validate)
        │
        ▼
Semantic Layer (dbt Metrics / Cube.dev)
        │
   ┌────┴─────────────┐
   ▼                  ▼
Traditional BI     AI-Enhanced Layer
(Metabase, Superset) (Text2SQL, Anomaly Detection, LLM Insights)
```

---

## Text2SQL — Natural Language to SQL

```python
import anthropic
import duckdb

client = anthropic.Anthropic()

# Set up DuckDB with sample data
conn = duckdb.connect()
conn.execute("""
    CREATE TABLE sales AS
    SELECT * FROM read_parquet('sales_data.parquet')
""")

# Get schema for context
schema = conn.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'sales'
""").fetchall()

schema_str = "\n".join(f"  {col}: {dtype}" for col, dtype in schema)

def natural_language_to_sql(question: str) -> tuple[str, any]:
    """Convert a natural language question to SQL and execute it."""

    prompt = f"""You are a SQL expert. Convert the user's question to a DuckDB SQL query.

Table: sales
Schema:
{schema_str}

Rules:
- Return ONLY the SQL query, no explanation
- Use DuckDB syntax (strftime for dates, etc.)
- Always include a LIMIT clause (max 1000 rows)
- Prefer readable column aliases

Question: {question}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    sql = response.content[0].text.strip()
    # Strip markdown code fences if present
    sql = sql.replace("```sql", "").replace("```", "").strip()

    try:
        result = conn.execute(sql).df()
        return sql, result
    except Exception as e:
        return sql, f"SQL Error: {e}"

# Usage
questions = [
    "What were the top 5 products by revenue last month?",
    "Show me daily sales trend for Q1 2025",
    "Which regions have declining sales vs last quarter?",
]

for q in questions:
    sql, result = natural_language_to_sql(q)
    print(f"Q: {q}")
    print(f"SQL: {sql}")
    print(f"Result:\n{result}\n{'='*60}")
```

---

## KPI Anomaly Detection

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats

def detect_kpi_anomalies(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str,
    sensitivity: float = 0.05   # Expected anomaly rate
) -> pd.DataFrame:
    """Detect anomalies in time-series KPI data."""

    df = df.sort_values(date_col).copy()
    values = df[metric_col].values

    # Method 1: Z-score (good for stationary series)
    z_scores = np.abs(stats.zscore(values))
    df["zscore_anomaly"] = z_scores > 3

    # Method 2: Isolation Forest (handles non-stationarity)
    clf = IsolationForest(contamination=sensitivity, random_state=42)
    df["if_anomaly"] = clf.fit_predict(values.reshape(-1, 1)) == -1

    # Method 3: Rolling statistics (detects trend breaks)
    window = 7
    rolling_mean = df[metric_col].rolling(window, center=True).mean()
    rolling_std  = df[metric_col].rolling(window, center=True).std()
    df["rolling_anomaly"] = np.abs(df[metric_col] - rolling_mean) > 2.5 * rolling_std

    # Consensus: flag if 2+ methods agree
    df["is_anomaly"] = (
        df["zscore_anomaly"].astype(int) +
        df["if_anomaly"].astype(int) +
        df["rolling_anomaly"].astype(int)
    ) >= 2

    # Add context
    df["anomaly_direction"] = np.where(
        df[metric_col] > rolling_mean, "spike", "dip"
    )
    df["pct_deviation"] = (df[metric_col] - rolling_mean) / rolling_mean * 100

    return df[df["is_anomaly"]][
        [date_col, metric_col, "pct_deviation", "anomaly_direction"]
    ]

# Example
df = pd.read_csv("daily_revenue.csv", parse_dates=["date"])
anomalies = detect_kpi_anomalies(df, "revenue", "date")
print(f"Found {len(anomalies)} anomalies:")
print(anomalies.to_string(index=False))
```

---

## AI-Generated Insights Report

```python
import anthropic
import pandas as pd

client = anthropic.Anthropic()

def generate_executive_report(data: pd.DataFrame, period: str) -> str:
    """Generate an executive BI report using Claude."""

    # Compute key metrics
    stats = {
        "total_revenue": data["revenue"].sum(),
        "growth_vs_prior": (data["revenue"].iloc[-7:].mean() / data["revenue"].iloc[-14:-7].mean() - 1) * 100,
        "top_products": data.groupby("product")["revenue"].sum().nlargest(3).to_dict(),
        "anomalies_detected": int((data["is_anomaly"] == True).sum()),
        "avg_order_value": data["revenue"].mean(),
    }

    prompt = f"""You are a senior business analyst. Write an executive summary for {period}.

Key Metrics:
- Total Revenue: ${stats['total_revenue']:,.0f}
- Growth vs Prior Week: {stats['growth_vs_prior']:+.1f}%
- Top 3 Products: {stats['top_products']}
- KPI Anomalies Detected: {stats['anomalies_detected']}
- Average Order Value: ${stats['avg_order_value']:.2f}

Write a 3-paragraph executive summary that:
1. Opens with the headline number and trend
2. Highlights key drivers and anomalies
3. Recommends 2-3 specific actions

Use a direct, confident tone suitable for a C-suite audience."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

report = generate_executive_report(df, "Week of April 7, 2025")
print(report)
```

---

## Automated Dashboard with Python

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def build_revenue_dashboard(df: pd.DataFrame) -> go.Figure:
    """Build interactive revenue dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Daily Revenue", "Revenue by Category",
                        "Top 10 Products", "Weekly Trend"],
        specs=[[{"type": "scatter"}, {"type": "pie"}],
               [{"type": "bar"},    {"type": "bar"}]]
    )

    # 1. Daily revenue with anomaly markers
    daily = df.groupby("date")["revenue"].sum().reset_index()
    anomalies_mask = df.groupby("date")["is_anomaly"].any()

    fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"],
                             mode="lines", name="Revenue", line_color="#7c4dff"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=daily["date"][anomalies_mask], y=daily["revenue"][anomalies_mask],
        mode="markers", marker=dict(color="red", size=10, symbol="x"),
        name="Anomaly"
    ), row=1, col=1)

    # 2. Category pie
    cat_rev = df.groupby("category")["revenue"].sum()
    fig.add_trace(go.Pie(labels=cat_rev.index, values=cat_rev.values,
                         hole=0.4, name="Category"), row=1, col=2)

    # 3. Top products
    top_prod = df.groupby("product")["revenue"].sum().nlargest(10)
    fig.add_trace(go.Bar(y=top_prod.index, x=top_prod.values,
                         orientation='h', marker_color="#00bcd4"), row=2, col=1)

    # 4. Weekly trend with forecast
    weekly = df.resample("W", on="date")["revenue"].sum().reset_index()
    fig.add_trace(go.Bar(x=weekly["date"], y=weekly["revenue"],
                         marker_color="#7c4dff", name="Weekly"), row=2, col=2)

    fig.update_layout(
        title="Revenue Intelligence Dashboard",
        height=800, showlegend=False,
        template="plotly_dark",
    )

    return fig

fig = build_revenue_dashboard(df)
fig.write_html("dashboard.html")   # Share as self-contained HTML
```

---

## Tips & Tricks

| BI Challenge | AI Solution |
|-------------|------------|
| Analysts stuck writing SQL | Text2SQL with Claude/GPT |
| Too many dashboards, no insight | LLM-generated narrative summaries |
| Manual anomaly review | Automated detection + Slack alerts |
| Ad-hoc analysis requests | Conversational BI chatbot |
| Report generation | Template + LLM variable filling |

---

## Related Topics

- [Data Pipelines](../data/pipelines.md)
- [AI Agents](../llm/agents.md)
- [Data Visualization](../data/visualization.md)
