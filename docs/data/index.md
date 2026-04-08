# Data

Data is the foundation of every AI system. This section covers everything from exploration to production pipelines.

---

## Topics

| Page | Description |
|------|-------------|
| [EDA](eda.md) | Exploratory Data Analysis — profiling, distributions, correlations |
| [Feature Engineering](feature-engineering.md) | Encoding, scaling, selection, derived features |
| [Data Visualization](visualization.md) | Matplotlib, Seaborn, Plotly, Bokeh |
| [Data Engineering](engineering.md) | Spark, Dask, Flink, streaming |
| [Data Pipelines](pipelines.md) | Airflow, Prefect, dbt, medallion architecture |
| [Databases & Storage](storage.md) | SQL tricks, NoSQL, cloud storage |
| [Time Series](time-series.md) | ARIMA, Prophet, LSTM forecasting |

---

## Data Quality Checklist

```python
import pandas as pd
import numpy as np

def audit_dataframe(df: pd.DataFrame) -> dict:
    """Quick data quality audit."""
    return {
        "shape": df.shape,
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_stats": df.describe().to_dict(),
        "cardinality": {c: df[c].nunique() for c in df.select_dtypes("object").columns},
    }
```

---

## Related Topics

- [Machine Learning](../machine-learning/index.md)
- [MLOps](../mlops/index.md)
