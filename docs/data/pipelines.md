# Data Pipelines

A data pipeline moves data from source to destination, applying transformations, validations, and enrichments along the way. In 2025, the best pipelines are declarative, observable, and fault-tolerant.

---

## Pipeline Architecture: Medallion

```
Source Systems
    │
    ▼
Bronze (Raw)       — Exact copy of source, no transforms, immutable
    │
    ▼
Silver (Cleaned)   — Deduplication, type casting, basic validation
    │
    ▼
Gold (Curated)     — Business logic, aggregations, ML-ready features
    │
    ▼
Serving Layer      — Feature store, analytics DB, API
```

---

## Prefect 3.x — Modern Workflow Orchestration

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import polars as pl

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract(source_url: str) -> pl.DataFrame:
    """Extract raw data from source."""
    return pl.read_csv(source_url)

@task
def validate(df: pl.DataFrame) -> pl.DataFrame:
    """Validate schema and data quality."""
    assert "user_id" in df.columns, "Missing user_id column"
    assert df["user_id"].null_count() == 0, "Null user_ids found"
    n_dups = df.filter(df.is_duplicated()).height
    if n_dups > 0:
        print(f"Warning: {n_dups} duplicate rows")
    return df

@task
def transform_silver(df: pl.DataFrame) -> pl.DataFrame:
    """Clean and standardize data."""
    return (
        df
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("amount").cast(pl.Float64),
            pl.col("category").str.to_lowercase().str.strip_chars(),
        ])
        .drop_nulls(subset=["user_id", "amount"])
        .unique(subset=["event_id"])
    )

@task
def transform_gold(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to business-level metrics."""
    return (
        df
        .group_by(["user_id", pl.col("timestamp").dt.date().alias("date")])
        .agg([
            pl.col("amount").sum().alias("daily_spend"),
            pl.col("amount").mean().alias("avg_transaction"),
            pl.len().alias("transaction_count"),
            pl.col("category").n_unique().alias("unique_categories"),
        ])
        .with_columns(
            pl.col("daily_spend").log1p().alias("log_spend")
        )
    )

@task
def load(df: pl.DataFrame, output_path: str) -> None:
    """Write to destination."""
    df.write_parquet(output_path, compression="zstd")
    print(f"Written {len(df):,} rows to {output_path}")

@flow(name="etl-pipeline", log_prints=True)
def etl_pipeline(source_url: str, output_path: str):
    raw = extract(source_url)
    validated = validate(raw)
    silver = transform_silver(validated)
    gold = transform_gold(silver)
    load(gold, output_path)
    return gold

if __name__ == "__main__":
    etl_pipeline.serve(name="daily-etl", cron="0 2 * * *")  # Run at 2am daily
```

---

## Apache Airflow DAGs

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    dag_id="feature_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "features"],
) as dag:

    def extract_raw(**ctx):
        ds = ctx["ds"]   # Execution date: "2025-01-15"
        df = pd.read_sql(
            f"SELECT * FROM events WHERE DATE(created_at) = '{ds}'",
            con=get_db_connection()
        )
        df.to_parquet(f"/tmp/raw_{ds}.parquet")
        return df.shape[0]

    def compute_features(**ctx):
        ds = ctx["ds"]
        df = pd.read_parquet(f"/tmp/raw_{ds}.parquet")
        features = compute_user_features(df)
        features.to_parquet(f"/tmp/features_{ds}.parquet")

    def load_to_feature_store(**ctx):
        ds = ctx["ds"]
        features = pd.read_parquet(f"/tmp/features_{ds}.parquet")
        write_to_feature_store(features, partition_date=ds)

    t1 = PythonOperator(task_id="extract_raw",           python_callable=extract_raw)
    t2 = PythonOperator(task_id="compute_features",      python_callable=compute_features)
    t3 = PythonOperator(task_id="load_to_feature_store", python_callable=load_to_feature_store)
    t4 = PostgresOperator(
        task_id="update_metadata",
        postgres_conn_id="postgres_default",
        sql="UPDATE pipeline_runs SET status='done', rows={{ ti.xcom_pull('extract_raw') }} WHERE date='{{ ds }}'",
    )

    t1 >> t2 >> t3 >> t4
```

---

## dbt — Data Transformations in SQL

```sql
-- models/silver/users_cleaned.sql
{{ config(materialized='incremental', unique_key='user_id') }}

SELECT
    user_id,
    LOWER(TRIM(email))         AS email,
    COALESCE(country, 'unknown') AS country,
    created_at::TIMESTAMP      AS created_at,
    age::INTEGER               AS age

FROM {{ source('raw', 'users') }}

WHERE email IS NOT NULL
  AND email LIKE '%@%'

{% if is_incremental() %}
  AND created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

```yaml
# models/schema.yml
models:
  - name: users_cleaned
    description: "Cleaned and validated user records"
    tests:
      - dbt_utils.recency:
          datepart: day
          field: created_at
          interval: 1
    columns:
      - name: user_id
        tests:
          - unique
          - not_null
      - name: email
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "email LIKE '%@%'"
```

```bash
dbt run --models silver+          # Run silver layer and downstream
dbt test --models users_cleaned   # Run data tests
dbt docs generate && dbt docs serve  # Browse lineage graph
```

---

## Great Expectations — Data Validation

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations
suite = context.add_expectation_suite("transactions_suite")

validator = context.get_validator(
    batch_request=...,
    expectation_suite_name="transactions_suite"
)

# Schema
validator.expect_column_to_exist("user_id")
validator.expect_column_to_exist("amount")
validator.expect_column_to_exist("timestamp")

# Values
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=100000)
validator.expect_column_values_to_be_of_type("amount", "float")
validator.expect_column_values_to_match_strftime_format("timestamp", "%Y-%m-%d %H:%M:%S")

# Freshness
validator.expect_column_max_to_be_between(
    "timestamp",
    min_value="2025-01-01",  # Data should be recent
)

# Distribution check (catches data drift!)
validator.expect_column_mean_to_be_between("amount", min_value=40, max_value=80)

validator.save_expectation_suite()

# Run checkpoint in CI/CD
results = context.run_checkpoint(checkpoint_name="daily_check")
assert results.success, f"Data quality failed: {results}"
```

---

## Streaming Pipelines with Kafka + Python

```python
from confluent_kafka import Producer, Consumer, KafkaException
import json, time

# Producer — publish events
producer = Producer({"bootstrap.servers": "localhost:9092"})

def produce_event(topic: str, key: str, value: dict):
    producer.produce(
        topic=topic,
        key=key.encode(),
        value=json.dumps(value).encode(),
        on_delivery=lambda err, msg: print(f"Delivered: {msg.offset()}" if not err else f"Error: {err}")
    )
    producer.poll(0)   # Trigger callbacks

# Publish user events
for i in range(1000):
    produce_event("user-events", f"user_{i}", {
        "user_id": i,
        "event": "purchase",
        "amount": round(20 + i * 0.5, 2),
        "timestamp": time.time()
    })

producer.flush()

# Consumer — process events in real-time
consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "feature-pipeline",
    "auto.offset.reset": "latest",
    "enable.auto.commit": False,   # Manual commit for exactly-once
})
consumer.subscribe(["user-events"])

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())

        event = json.loads(msg.value())
        process_event(event)          # Your processing logic
        consumer.commit(msg)          # Commit after successful processing
finally:
    consumer.close()
```

---

## Polars — Fast DataFrame Transforms

```python
import polars as pl

# Lazy evaluation — build query plan, execute optimally
df = (
    pl.scan_parquet("s3://bucket/data/*.parquet")  # Lazy, no data loaded yet
    .filter(pl.col("amount") > 0)
    .with_columns([
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        pl.col("amount").log1p().alias("log_amount"),
        (pl.col("amount") / pl.col("amount").mean().over("user_id")).alias("normalized_amount"),
    ])
    .group_by(["user_id", pl.col("timestamp").dt.month().alias("month")])
    .agg([
        pl.col("amount").sum().alias("monthly_spend"),
        pl.col("amount").mean().alias("avg_amount"),
        pl.len().alias("tx_count"),
    ])
    .sort(["user_id", "month"])
    .collect(streaming=True)   # Execute — streams data for low memory
)

# Write partitioned output
df.write_parquet(
    "output/",
    partition_by=["month"],
    compression="zstd",
)
```

---

## Tips & Tricks

| Issue | Solution |
|-------|---------|
| Pipeline failures mid-run | Checkpoint outputs at each stage |
| Slow Airflow tasks | Use `KubernetesPodOperator` for isolation |
| dbt slow models | Add `materialized='table'` for hot paths |
| Data validation in CI | Run `great_expectations` in GitHub Actions |
| Kafka consumer lag | Scale consumer group instances |
| Large parquet files | Partition by date/region for faster reads |

---

## Related Topics

- [Data Engineering (Big Data)](engineering.md)
- [Feature Engineering](feature-engineering.md)
- [MLOps](../mlops/index.md)
