# Feature Engineering

Feature engineering is a critical step in the machine learning pipeline, aimed at improving the dataset for better model performance. Below are essential techniques, tricks, and tips for handling various scenarios in feature engineering.

---

## Key Steps in Feature Engineering

1. **Data Cleaning**: Handle missing, inconsistent, or irrelevant data.
2. **Feature Transformation**: Modify features to make them better suited for models.
3. **Feature Selection**: Identify only the most relevant features.
4. **Feature Generation**: Create new features from existing data.

---

## 1. Handling Missing Data

### **Tricks to Handle Missing Data**
- **Dropping Rows/Columns**:
    - Remove rows or columns with excessive missing values when they don’t contain critical information.
    ```python
    df.dropna(how='any', inplace=True)  # Drop rows with any missing value
    ```

- **Imputation**:
    - Fill missing values with statistical metrics like mean, median, or a custom value.
    ```python
    # Fill numerical values with mean
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # Fill categorical values with mode
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    ```

- **Predictive Imputation**:
    - Use models to predict missing feature values based on other features.
    ```python
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(random_state=0)
    df_filled = imputer.fit_transform(df)
    ```

- **Indicator Variables**:
    - Add binary columns indicating whether a value is missing.
    ```python
    df['Age_missing'] = df['Age'].isnull().astype(int)
    ```

---

## 2. Encoding Categorical Variables

### **Tricks for Encoding**:
- **One-Hot Encoding**:
    - Convert categorical values into multiple binary columns.
    ```python
    pd.get_dummies(df, columns=['City'], drop_first=True)
    ```

- **Label Encoding**:
    - Assign integers to categorical values (useful for tree-based models like XGBoost).
    ```python
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    ```

- **Ordinal Encoding**:
    - Use a custom order for encoding ordinal categories.
    ```python
    from sklearn.preprocessing import OrdinalEncoder

    categories_order = [['Low', 'Medium', 'High']]
    encoder = OrdinalEncoder(categories=categories_order)
    df['RiskLevel'] = encoder.fit_transform(df[['RiskLevel']])
    ```

- **Target Encoding**:
    - Encode categories based on their relationship to the target variable.
    ```python
    df['EncodedCategory'] = df.groupby('Category')['Target'].transform('mean')
    ```

---

## 3. Scaling and Normalization

### **Tricks for Numerical Transformation**:
- **Standard Scaling**:
    - Centers data with zero mean and unit variance (best for distance-based models like SVM and kNN).
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Age', 'Salary']])
    ```

- **Min-Max Scaling**:
    - Rescales data to the range [0, 1] (useful for neural networks).
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Age', 'Salary']])
    ```

- **Log Transformation**:
    - Compresses features with heavy skewness.
    ```python
    df['Log_Salary'] = np.log1p(df['Salary'])  # log(x + 1) to handle zeroes
    ```

- **Robust Scaling**:
    - Scales data based on statistics robust to outliers.
    ```python
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df[['Income']])
    ```

---

## 4. Feature Selection

Feature selection is the process of removing irrelevant or redundant features to simplify the model.

### **Tricks for Feature Selection**:
- **Correlation Threshold**:
    - Remove features with high correlation (e.g., correlation > 0.9).
    ```python
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(columns=to_drop, inplace=True)
    ```

- **Univariate Selection**:
    - Use statistical tests to select the best features.
    ```python
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(score_func=f_classif, k=10)
    selected_features = selector.fit_transform(X, y)
    ```

- **Feature Importance** (Tree Models like Random Forest or Gradient Boosting):
    ```python
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X, y)
    feature_importances = model.feature_importances_
    ```

- **LASSO (L1 Regularization)**:
    - Shrinks coefficients of less important features to zero.
    ```python
    from sklearn.linear_model import Lasso

    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    selected_features = X.columns[(lasso.coef_ != 0)]
    ```

- **Recursive Feature Elimination (RFE)**:
    - Recursively removes less important features.
    ```python
    from sklearn.feature_selection import RFE

    from sklearn.ensemble import GradientBoostingClassifier
    selector = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=5)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    ```

---

## 5. Feature Engineering Tricks for Time Series Data

1. **Lagged Features**:
    - Create lagged versions of your target or predictors.
    ```python
    df['Sales_Lag1'] = df['Sales'].shift(1)
    ```

2. **Rolling Window Features**:
    - Calculate rolling averages, sums, or standard deviations.
    ```python
    df['Sales_MA3'] = df['Sales'].rolling(window=3).mean()
    ```

3. **Datetime Features**:
    - Extract components such as day of the week, month, or season.
    ```python
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    ```

---

## 6. Derived Features

Generate new, meaningful features:
1. **Interaction Features**:
    - Create features by multiplying two other features.
    ```python
    df['Income_per_Age'] = df['Income'] / df['Age']
    ```

2. **Polynomials**:
    - Add polynomial terms for non-linear relationships.
    ```python
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False)
    df_poly = poly.fit_transform(df[['Experience', 'Age']])
    ```

3. **Clustering Features**:
    - Use unsupervised learning methods like K-Means to assign cluster labels as new features.
    ```python
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df[['Age', 'Income']])
    ```

---

## 7. Tips for Automated Feature Engineering
1. **FeatureTools**: An auto feature-engineering library to create and manage features efficiently.
    ```python
    import featuretools as ft

    es = ft.EntitySet(id="example")
    es = es.entity_from_dataframe(entity_id="data", dataframe=df, index="id")
    features, feature_defs = ft.dfs(entityset=es, target_entity="data")
    ```

2. **Auto-Sklearn**: Automatically selects important features during model training.
    ```python
    from autosklearn.experimental.askl2 import AutoSklearn2Classifier

    automl = AutoSklearn2Classifier()
    automl.fit(X, y)
    ```

---

## 8. Hidden Gems for Feature Engineering

1. **Text Features**:
    - Use TF-IDF or Word2Vec to convert text into numerical features.
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=100)
    tfidf_matrix = vec.fit_transform(df['Text'])
    ```

2. **Feature Hashing**:
    - Quickly encode high-cardinality categorical data.
    ```python
    from sklearn.feature_extraction import FeatureHasher

    hasher = FeatureHasher(n_features=10, input_type="string")
    hashed_features = hasher.transform(df['Category'])
    ```

---

By systematically applying these tricks, you’ll increase predictive power and efficiency in your machine learning pipeline.