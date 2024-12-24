# Big Data

Big Data refers to datasets that are too large or complex to be processed using traditional data processing tools. This guide provides practical steps, libraries, and code snippets to efficiently handle and work with massive datasets.

---

## 1. **Key Libraries & Frameworks for Big Data**

Big Data processing involves distributed computing and scalable storage solutions. Below are the most important tools and frameworks:

### **Distributed Data Processing**:
- **Apache Spark**:
  - Unified framework for distributed data analysis.
  - Works with Python (`pyspark`), R, Scala, and Java.
- **Hadoop**:
  - Batch-processing tool using HDFS (Hadoop Distributed File System) and MapReduce Algorithm.
- **Dask**:
  - Scalable Python library for parallel and distributed computations.
- **Apache Flink**:
  - Framework for real-time stream processing.
- **Beam**:
  - Allows building unified batch and stream data pipelines.

### **Big Data Storage**:
- **HDFS** (Hadoop Distributed File System): Default distributed storage of Hadoop.
- **Amazon S3**: Cloud-based massively scalable object storage.
- **GCS (Google Cloud Storage)**: For serverless Big Data storage.
- **NoSQL Databases**:
  - MongoDB, Cassandra, or HBase for semi-structured or unstructured data.
  
### **SQL on Big Data**:
- **Presto/Trino**: SQL query engine for big datasets in distributed systems.
- **Google BigQuery**: SQL-compatible data warehouse for big datasets.

### **Visualization Tools**:
- Tools like **Databricks**, **Tableau**, and **Power BI** enable consuming Big Data for insights.

---

## 2. **Processing Big Data with Apache Spark**

Apache Spark is one of the most popular Big Data frameworks.

### **PySpark Setup**:
```python
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Big Data Processing") \
    .getOrCreate()

# Reading Data
df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)

# Displaying Schema
df.printSchema()

# Basic Operations
df.select(['column1', 'column2']).show()
df.filter(df['column1'] > 100).show()

# Writing Output
df.write.csv("output_directory/")
```

### **Common Transformations and Actions**:
- Transformations (lazy execution):
  - `filter()`, `select()`, `groupBy()`, `join()`
- Actions (trigger computation):
  - `count()`, `collect()`, `show()`

```python
# Counting rows
print("Total rows:", df.count())

# Calculating aggregates
df.groupBy('country').agg({'sales': 'sum'}).show()
```

---

## 3. **Working with Dask**

Dask scales Python's functionality to handle operations on Big Data.

### **Setup and Basic Operations**:
```python
import dask.dataframe as dd

# Reading a large CSV file
df = dd.read_csv("large_dataset.csv")

# Compute statistics
print(df.describe().compute())

# Filtering data
filtered = df[df['column_name'] > 100]
print(filtered.head())
```

### **Parallel Processing**:
Dask automatically partitions data across CPU cores or distributed clusters:
```python
from dask.distributed import Client

client = Client()
df = dd.read_csv("large_file.csv")
result = df.groupby('category').sum().compute()
```

---

## 4. **Hadoop Basics**

Hadoop processes Big Data using MapReduce and stores it on HDFS.

### **HDFS Commands**:
```bash
# List directories
hdfs dfs -ls /

# Create a directory
hdfs dfs -mkdir /input_data

# Upload a file
hdfs dfs -put local_file.csv /input_data

# Read a file
hdfs dfs -cat /input_data/local_file.csv
```

### **MapReduce in Hadoop**:
Hadoop MapReduce involves splitting data into key-value pairs and processing in parallel.
```java
public class WordCountMapper extends Mapper<Object, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        StringTokenizer itr = new StringTokenizer(value.toString());
        while (itr.hasMoreTokens()) {
            word.set(itr.nextToken());
            context.write(word, one);
        }
    }
}
```

---

## 5. **SQL on Big Data**

SQL-based tools allow querying massive datasets without complex programming.

### **Presto/Trino**:
```sql
SELECT country, SUM(sales)
FROM big_data_table
GROUP BY country
ORDER BY SUM(sales) DESC
LIMIT 10;
```

### **Google BigQuery**:
```python
from google.cloud import bigquery

client = bigquery.Client()

# SQL Query
query = """
    SELECT customer, SUM(order_value) AS total_value
    FROM `project.dataset.orders`
    GROUP BY customer
    ORDER BY total_value DESC
    LIMIT 10
"""
query_job = client.query(query)
results = query_job.to_dataframe()
print(results)
```

---

## 6. **Streaming Big Data**

Real-time processing frameworks are essential for streaming data pipelines.

### **Apache Kafka Basics**:
Kafka is a distributed streaming platform.

- **Producers**: Send data to topics.
- **Consumers**: Read data from topics.

```python
from kafka import KafkaProducer, KafkaConsumer

# Producer Example
producer = KafkaProducer(bootstrap_servers='localhost:9092')
producer.send('big_data_topic', b'message content')

# Consumer Example
consumer = KafkaConsumer('big_data_topic', bootstrap_servers='localhost:9092')
for message in consumer:
    print(message.value)
```

### **Apache Spark Streaming**:
```python
from pyspark.streaming import StreamingContext

# Batch interval of 2 seconds
ssc = StreamingContext(spark.sparkContext, 2)

# Create a DStream
lines = ssc.socketTextStream("localhost", 9999)

# Word count
counts = lines.flatMap(lambda line: line.split(" ")) \
              .map(lambda word: (word, 1)) \
              .reduceByKey(lambda a, b: a + b)

counts.pprint()
ssc.start()
ssc.awaitTermination()
```

---

## 7. **Machine Learning on Big Data**

Machine Learning on Big Data requires distributed frameworks for scalability.

### **ML with PySpark**:
```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Prepare features
assembler = VectorAssembler(inputCols=['feature1', 'feature2'], outputCol='features')
df = assembler.transform(df)

# Train/Test split
train, test = df.randomSplit([0.8, 0.2], seed=123)

# Linear Regression Model
lr = LinearRegression(featuresCol='features', labelCol='target')
model = lr.fit(train)

# Evaluate
results = model.evaluate(test)
print("RMSE:", results.rootMeanSquaredError)
```

---

## 8. **Data Cleaning at Scale**

Handling and cleaning messy data at scale is essential for Big Data processing.

### Handle Missing Values (PySpark Example):
```python
from pyspark.sql.functions import col

# Fill missing values
df = df.fillna({'column1': 0, 'column2': 'unknown'})

# Drop rows with missing values
df = df.na.drop()
```

### Removing Duplicates:
```python
df = df.dropDuplicates(['unique_column'])
```

### Filtering Outliers:
```python
from pyspark.sql.functions import mean, stddev

# Calculate Z-score
stats = df.select(mean(col('sales')).alias('mean'), stddev(col('sales')).alias('stddev')).collect()[0]
mean_sales = stats['mean']
stddev_sales = stats['stddev']

df = df.filter((col('sales') <= mean_sales + 3 * stddev_sales) & (col('sales') >= mean_sales - 3 * stddev_sales))
```

---

## 9. **Visualization of Big Data**

### PySpark Integration with Pandas:
```python
# Convert Spark DataFrame to Pandas
pdf = df.limit(1000).toPandas()

# Visualization
import seaborn as sns
sns.pairplot(pdf)
```

### Interactive Visualization (Plotly):
```python
import plotly.express as px

px.scatter(pdf, x='column1', y='column2', size='column3', color='category')
```

---

By combining tools like Spark, Dask, and SQL-based engines with distributed storage systems such as HDFS or S3, you can efficiently process massive datasets. This approach ensures scalability, speed, and actionable insights when handling Big Data systems.