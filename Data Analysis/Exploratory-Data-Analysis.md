# Exploratory Data Analysis AITricks

Exploratory Data Analysis (EDA) is the process of understanding the structure, patterns, and anomalies in the data prior to modeling. Below are **tricks, shortcuts, and strategies** to streamline your EDA workflow effectively.

---

## 1. **Libraries for EDA**
Here are the most robust libraries for EDA and their usage:
- **Pandas**: Data manipulation and descriptive stats.
- **NumPy**: Numerical operations.
- **Seaborn** & **Matplotlib**: Data visualization.
- **Plotly**: Interactive visualizations.
- **Sweetviz & Pandas Profiling**: Quick automated EDA reports.
- **SciPy**: Statistical analysis.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
```

---

## 2. **Quick Data Overview**
- **Basic Dataset Information**:
    ```python
    df.info()       # Overview of data structure (column types, missing values)
    df.describe()   # Summary statistics for numerical columns
    df.head(10)     # View the first 10 rows
    df.tail(10)     # View the last 10 rows
    ```

- **Data Types Distribution**:
    ```python
    df.dtypes.value_counts()
    ```

- **Identify Duplicates**:
    ```python
    print(f"Duplicates in dataset: {df.duplicated().sum()}")  # Check duplicates
    ```

- **Summary Using Automated Tools** (for large datasets):
    ```python
    import pandas_profiling as pp

    profile = pp.ProfileReport(df)
    profile.to_widgets()
    ```

---

## 3. **Data Cleaning**
Cleaning data ensures quality before EDA:
- **Fix Inconsistent Data Entry (e.g., Categorical Labels)**:
    ```python
    df['Category'] = df['Category'].str.lower().str.strip()  # Standardizing text entries
    ```

- **Remove Outliers** (using interquartile range):
    ```python
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR)))]
    ```

- **Handle Missing Values**:
    - Numerical columns:
        ```python
        df['Column'] = df['Column'].fillna(df['Column'].mean())  # Or median
        ```
    - Categorical columns:
        ```python
        df['Category'] = df['Category'].fillna('Unknown')
        ```

---

## 4. **Target Variable Understanding**
Always start your analysis with the **target variable**:
- **Distribution of the Target**:
    ```python
    sns.histplot(df['Target'], kde=True, bins=30)
    plt.title("Target Distribution")
    plt.show()
    ```

- **Class Imbalance**:
    ```python
    df['Target'].value_counts(normalize=True).plot(kind='bar', title="Class Distribution")
    ```

---

## 5. **Univariate Analysis**
Analyze each feature independently.

### **Numerical Features**:
- **Visualizing Distributions**:
    ```python
    sns.histplot(df['Age'], kde=True, bins=30)
    px.box(df, y='Age', title='Age Boxplot').show()  # Interactive version
    ```

- **Statistical Summary**:
    ```python
    print("Skewness: ", df['Age'].skew())          # Skewness of distribution
    print("Kurtosis: ", df['Age'].kurt())          # Measure of outliers in data
    ```

### **Categorical Features**:
- **Visualizing Occurrences**:
    ```python
    sns.countplot(x='Gender', data=df, palette='viridis')
    px.pie(df, names='Gender', title='Gender Distribution').show()  # Interactive pie chart
    ```

- **Check Unique Values**:
    ```python
    print(df['Category'].nunique(), "unique categories")
    print(df['Category'].value_counts())  # Frequency count of each label
    ```

---

## 6. **Bivariate Analysis**
Analyze pairwise relationships between target and features.

### **Numerical vs Numerical**:
- **Correlation Heatmap**:
    ```python
    corr = df.corr()  # Pearson Correlation Matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    ```

- **Pairplot Analysis**:
    ```python
    sns.pairplot(df, diag_kind='kde', kind='scatter')
    ```

### **Categorical vs Numerical**:
- **Boxplots for Target vs Categories**:
    ```python
    sns.boxplot(x='Gender', y='Income', data=df, palette='viridis')
    ```

- **Violin Plot**:
    ```python
    sns.violinplot(x='Education', y='Salary', data=df)
    ```

### **Categorical vs Categorical**:
- **Cross Tab Analysis**:
    ```python
    pd.crosstab(df['Gender'], df['Purchased'], normalize='index').plot(kind='bar', stacked=True)
    ```

---

## 7. **Multivariate Analysis**
Examine interactions between three or more features.
- **Pairwise Correlations with Target** (via heatmap):
    ```python
    sns.heatmap(df.corr()[['Target']].sort_values('Target', ascending=False), annot=True)
    ```

- **Visualizing Clusters (e.g., PCA/TSNE)**:
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit_transform(df.iloc[:, :-1])
    plt.scatter(pca[:, 0], pca[:, 1], c=df['Target'], cmap='viridis')
    ```

---

## 8. **Outlier Detection**
Outliers can skew statistics or introduce noise.

- **Visualizing Outliers with Boxplots**:
    ```python
    sns.boxplot(x=df['Salary'])
    ```

- **Z-Score Method**:
    ```python
    from scipy.stats import zscore
    df['Z_Score'] = zscore(df['Salary'])
    outliers = df[df['Z_Score'].abs() > 3]
    ```

- **Isolation Forest (for Multivariate Outliers)**:
    ```python
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.1)
    df['Scores'] = iso.fit_predict(df[['Age', 'Salary']])  # -1 indicates outlier
    ```

---

## 9. **Statistical Hypothesis Testing**
Statistical methods to verify feature relationships.

- **Chi-Square Test (for categorical data)**:
    ```python
    from scipy.stats import chi2_contingency

    chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Gender'], df['Purchased']))
    print("Chi-Square statistic:", chi2, "p-value:", p)
    ```

- **T-Test (for numerical features)**:
    ```python
    from scipy.stats import ttest_ind

    t_stat, p_val = ttest_ind(df['Salary'][df['Gender'] == 'Male'], df['Salary'][df['Gender'] == 'Female'])
    print(f"T-statistic: {t_stat}, P-value: {p_val}")
    ```

---

## 10. **EDA Automation Tools**
Shortcuts for auto-EDA reports:
- **Sweetviz**:
    ```python
    import sweetviz as sv
    report = sv.analyze(df)
    report.show_html('EDA_Report.html')
    ```

- **D-Tale**:
    ```python
    import dtale
    dtale.show(df)
    ```

---

## 11. Hidden Gems and Tips for Advanced EDA

1. **Find Highly Correlated Features (> 0.8)**:
    ```python
    correlated_features = corr[corr > 0.8].stack().index.tolist()
    print("Highly correlated features:", correlated_features)
    ```

2. **Show Missingness Heatmap**:
    ```python
    import missingno as msno
    msno.heatmap(df, cmap='coolwarm')
    ```

3. **Log Transform for Skewed Data**:
    ```python
    df['Log_Salary'] = np.log1p(df['Salary'])
    ```

4. **Categorical Heatmap (with Target)**:
    ```python
    sns.heatmap(pd.crosstab(df['Category'], df['Target']), annot=True)
    ```

5. **Interactive Correlation Matrix**:
    ```python
    import plotly.figure_factory as ff
    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist())
    fig.show()
    ```

---

Begin with basic data sanity checks, progress into deeper analysis using various features, visualizations, and tests, and conclude by automating insights for efficiency. Proper EDA ensures the model development foundation is rock-solid!