# Support Vector Machine (SVM)

A **Support Vector Machine (SVM)** is a supervised learning algorithm used for classification and regression tasks. It finds the **optimal decision boundary (hyperplane)** that best separates the classes in a dataset. 

---

## 1. How SVM Works

SVM tries to find a hyperplane that:
1. **Maximally separates data points** from different classes.
2. Maximizes the **distance (margin)** between data points and the hyperplane.

Key concepts:
1. **Support Vectors**:
   - Points closest to the hyperplane that influence its position.
2. **Kernel Functions**:
   - Allow SVMs to work efficiently in non-linear data spaces by mapping inputs to higher dimensions.
3. **Margin**:
   - Distance between the hyperplane and the data points. Larger margins provide better separation and generalization.

### SVM Objective - Classification:
For a binary classification, the decision function is:
```txt
f(x) = w · x + b
```
Where:
- `w`: Weights (hyperplane parameters)
- `x`: Feature vector
- `b`: Bias (offset)

SVM tries to minimize:
```txt
Loss = 1/2 ||w||^2 + C Σ max(0, 1 - y_i (w · x_i + b))
```
Where:
- `C`: Regularization parameter (tradeoff between margin width and errors)

---

## 2. Key Libraries for SVM

1. **NumPy**:
   For data manipulation and mathematical operations.
   ```bash
   pip install numpy
   ```

2. **scikit-learn**:
   Simplifies SVM implementation with built-in models.
   ```bash
   pip install scikit-learn
   ```

3. **Matplotlib and Seaborn**:
   For visualizing SVM results.
   ```bash
   pip install matplotlib seaborn
   ```

---

## 3. Support Vector Machine with scikit-learn

### 3.1 Example: Classifying Points with Linear SVM
```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.datasets import make_blobs  # Dummy dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Generate a dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)  # Two classes
# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the SVM model
model = SVC(kernel='linear', C=1)  # Linear kernel SVM
model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Visualize the decision boundary
def plot_svm_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired, edgecolors='k')
    # Create a mesh grid
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5,
               linestyles=['-'])

plot_svm_boundary(X, y, model)
plt.title("Linear SVM Decision Boundary")
plt.show()
```

---

### 3.2 Non-Linear SVM (Kernel Trick)

For non-linear data, SVM uses **kernel tricks** to map input data into higher dimensions where it becomes linearly separable.

Popular kernels:
1. **Polynomial Kernel**:
   ```txt
   K(x, y) = (x · y + c)^d
   ```

2. **Radial Basis Function (RBF)** (default in scikit-learn):
   ```txt
   K(x, y) = exp(-γ ||x - y||²)
   ```

Example with RBF Kernel:
```python
# Non-linear kernel SVM
model = SVC(kernel='rbf', gamma=0.5, C=1)  # Use RBF kernel
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with RBF Kernel: {accuracy * 100:.2f}%")
```

---

## 4. Parameter Tuning in SVM

1. **C (Regularization Parameter)**:
   - Larger `C` values aim to classify **all points correctly** but may overfit.
   - Lower values allow more misclassifications for a wider margin (better generalization).

2. **Gamma (for RBF Kernel)**:
   - Controls how far influence of a single data point reaches.
   - Smaller γ values mean more smoothing; larger values lead to tight boundaries.

3. Use **GridSearchCV** for hyperparameter tuning.
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
model = grid.best_estimator_
```

---

## 5. Build SVM **from Scratch (Template)**

To construct an SVM for linear classification, we need:
1. **Hinge Loss Function** to calculate the margin.
2. **Gradient Descent** to iteratively optimize weights `w` and `b`.

### Implementation
```python
import numpy as np

# Generate synthetic data
X = np.array([[2, 3], [1, 1], [2, 1], [3, 3], [4, 1]])  # Features
y = np.array([-1, -1, -1, 1, 1])                       # Labels (-1 or 1)

# Function for Hinge Loss calculation
def hinge_loss(w, b, X, y, lambda_):
    n = len(y)
    distances = 1 - y * (np.dot(X, w) + b)
    distances = np.maximum(0, distances)  # Maximum of 0 and hinge loss
    loss = 0.5 * lambda_ * np.dot(w, w) + (1 / n) * np.sum(distances)
    return loss

# Gradient Descent Function
def gradient_descent(X, y, learning_rate=0.01, lambda_=0.01, epochs=1000):
    n_features = X.shape[1]
    w = np.zeros(n_features)   # Initialize weights
    b = 0                      # Initialize bias
    
    # Iterate for epochs
    for _ in range(epochs):
        # Calculate gradients
        for i, x_i in enumerate(X):
            condition = y[i] * (np.dot(x_i, w) + b) >= 1
            if condition:
                dw = lambda_ * w
                db = 0
            else:
                dw = lambda_ * w - y[i] * x_i
                db = -y[i]
            # Update weights and bias
            w = w - learning_rate * dw
            b = b - learning_rate * db
    return w, b

# Train the model
final_w, final_b = gradient_descent(X, y)
print("Weights:", final_w)
print("Bias:", final_b)

# Make predictions
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

preds = predict(X, final_w, final_b)
print("Predictions:", preds)
```

---

## 6. Practical Tips for SVM

1. Use **feature scaling** (e.g., Standardization) before SVM to ensure smooth convergence:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. For large datasets, consider **LinearSVC**, which scales better to more samples.

3. **Kernel selection** for different data:
   - Use `linear` kernel for linearly separable data.
   - Use `rbf` kernel for non-linear patterns.

---

This detailed breakdown includes theoretical knowledge, library use, and a from-scratch implementation of Support Vector Machines, covering everything from basic concepts to advanced customization!