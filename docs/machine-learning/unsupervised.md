# Unsupervised Learning AITricks

Unsupervised learning is a type of machine learning where the model is trained on unlabeled data. Instead of predicting outputs, the goal is to discover patterns, structures, or relationships in the dataset. This type of learning is used for clustering, dimensionality reduction, anomaly detection, and more.

---

## 📄 **Sections in Unsupervised Learning**

---

### 1. **Key Concepts**

Key characteristics of unsupervised learning include:
1. **No Labeled Data**: The model relies only on the features of the input data to uncover patterns.
2. **Common Applications**:
   - Clustering (e.g., grouping customers based on behavior).
   - Dimensionality Reduction (e.g., feature selection or visualization).
   - Association Rule Learning (e.g., market basket analysis).
   - Anomaly Detection (e.g., identifying fraudulent transactions).
3. **Types of Unsupervised Learning**:
   - **Clustering**: Divides data into groups with similar characteristics.
   - **Dimensionality Reduction**: Reduces the number of input features while retaining essential information.

---

### 2. **Applications of Unsupervised Learning**

Some practical examples include:
- **Customer Segmentation**: Grouping customers with similar purchase behaviors.
- **Fraud Detection**: Identifying unusual or anomalous behavior in transactions.
- **Document Clustering**: Grouping similar documents for topic identification.
- **Data Visualization**: Reducing high-dimensional data into 2D or 3D representations.

---

### 3. **Important Python Libraries**

- **Clustering**: `scikit-learn`
- **Dimensionality Reduction**: `scikit-learn`, `PCA`, `TSNE`, `UMAP`.
- **Visualization**: `matplotlib`, `seaborn`.
- **Anomaly Detection**: `scikit-learn`, `pyod`.

---

### 4. **Steps in Unsupervised Learning**

---

#### a. Data Preprocessing

Before applying unsupervised learning methods, preprocess the data to ensure better results.

**Key Tasks**:
1. Handle missing values.
2. Standardize/Scale features.
3. Optionally, reduce dimensionality for speed and interpretability.

**Code Example**:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data.csv")

# Handle missing values
data = data.fillna(method="ffill")

# Standardizing the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
```

---

#### b. Clustering

Clustering is the process of grouping data points such that those within the same group (cluster) are more similar to each other than to those in other groups.

##### i. K-Means Clustering
K-Means is one of the most popular clustering methods.

**Code Example**:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define the model
kmeans = KMeans(n_clusters=3, random_state=42)

# Train the model and get cluster labels
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.show()
```

##### How to Determine the Number of Clusters:
Use the **Elbow Method** to find the optimal number of clusters:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Elbow method to find the optimal number of clusters
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
```

##### ii. Hierarchical Clustering
Clusters are created in a tree structure.
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform hierarchical clustering
hclust = linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(hclust)
plt.show()
```

##### iii. DBSCAN Clustering
**Density-based clustering** identifies clusters of varying shapes and sizes.
```python
from sklearn.cluster import DBSCAN

# Train DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(scaled_data)

# Visualize DBSCAN Clustering
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='plasma')
plt.show()
```

---

#### c. Dimensionality Reduction

Dimensionality reduction algorithms reduce the number of features in the dataset while preserving meaningful patterns.

##### i. PCA (Principal Component Analysis)
PCA is used to project the data into fewer dimensions.
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Visualize PCA results
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], cmap='rainbow')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

##### ii. t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE is commonly used for visualizing high-dimensional data in 2D/3D space.
```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_transformed = tsne.fit_transform(scaled_data)

# Visualize t-SNE results
plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], cmap='coolwarm')
plt.show()
```

##### iii. UMAP (Uniform Manifold Approximation and Projection)
UMAP provides an alternative to t-SNE for dimensionality reduction.
```python
import umap

# Apply UMAP
embedding = umap.UMAP(n_components=2, random_state=42)
umap_data = embedding.fit_transform(scaled_data)

# Visualize UMAP results
plt.scatter(umap_data[:, 0], umap_data[:, 1], cmap='Spectral')
plt.show()
```

---

#### d. Anomaly Detection

Unsupervised learning can identify anomalies or outliers in the dataset.

**Example using Isolation Forest**:
```python
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(scaled_data)

# Visualize results
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=anomalies, cmap='cool')
plt.title("Anomaly Detection")
plt.show()
```

---

### 5. **Challenges in Unsupervised Learning**

1. **Evaluation**: Without labeled data, evaluating clusters or reduced dimensions can be challenging.
   - Solutions:
     - Use metrics like **Silhouette Score** or **Dunn Index** for clustering.
     ```python
     from sklearn.metrics import silhouette_score
     print("Silhouette Score:", silhouette_score(scaled_data, labels))
     ```
2. **Choosing the Right Approach**: Determining the optimal number of clusters or dimensions requires experimentation.
3. **Interpretability**: Understanding the clusters or reduced dimensions isn’t always straightforward.

---

### 6. **Popular Libraries**

- **`scikit-learn`**: Wide range of clustering and dimensionality reduction algorithms (K-Means, DBSCAN, PCA, IsolationForest, etc.).
- **`umap-learn`**: UMAP for dimensionality reduction.
- **`matplotlib`, `seaborn`**: Visualization.
- **`pyod`**: Toolkit for outlier detection.

---

### 7. **References and Resources**
1. [scikit-learn Official Documentation](https://scikit-learn.org)  
2. [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/)  
3. [t-SNE Explained](https://distill.pub/2016/misread-tsne/)  
4. [Clustering Metrics - Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)  

---

By mastering unsupervised learning techniques such as clustering, dimensionality reduction, and anomaly detection, you can extract valuable insights from unlabeled data and solve a wide range of practical problems. This guide provides the foundational steps and code for applying unsupervised learning effectively.