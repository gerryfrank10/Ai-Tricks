# Linear Regression

## 1. What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used for predicting continuous outcomes. It models a relationship between:
- A **dependent variable (`y`)** (target) and
- One or more **independent variables (`x`)** (features).

The goal is to fit a **straight line** (or hyperplane in higher dimensions) that minimizes the **error** (the difference between the predicted and actual values).

### Mathematical Equation:
For a single-variable linear regression:
```txt
y = mx + c
```
Where:
- `y`: Dependent variable
- `m`: Slope (weight)
- `x`: Independent variable
- `c`: Intercept (bias)

For multiple variables:
```txt
y = w1x1 + w2x2 + ... + wnxn + c
```
Where:
- `w`: Coefficients (weights)

---

## 2. Key Libraries Required

We’ll use Python libraries to simplify regression workflow:

1. **NumPy**:
   For mathematical calculations (like dot products, gradient descent, etc.).
   ```bash
   pip install numpy
   ```

2. **pandas**:
   For data preprocessing and manipulation.
   ```bash
   pip install pandas
   ```

3. **Matplotlib & Seaborn**:
   For visualizing data and model outputs.
   ```bash
   pip install matplotlib seaborn
   ```

4. **scikit-learn**:
   Provides pre-built implementations of linear regression.
   ```bash
   pip install scikit-learn
   ```

---

## 3. Workflow of Linear Regression

1. Understand your data.
2. Prepare the data (handling missing values, scaling, etc.).
3. Train a linear regression model.
4. Evaluate the model's performance (visualizations, metrics).
5. Make predictions.

---

## 4. Linear Regression with scikit-learn

### 4.1 Example: Predicting house prices
```python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate dummy data (or load a dataset)
# Feature: Size in square feet, Target: Price in thousands of dollars
data = pd.DataFrame({
    'Size': [500, 800, 1000, 1200, 1500],
    'Price': [150, 240, 300, 360, 450]
})

# Step 2: Split data into Features (X) and Target (y)
X = data[['Size']]  # Predictor (independent)
y = data['Price']   # Target (dependent)

# Step 3: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Step 7: Visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')                   # Actual data
plt.plot(X, model.predict(X), color='red', label='Regression Line')   # Regression line
plt.title("Linear Regression")
plt.xlabel("Size of the House (sq ft)")
plt.ylabel("Price (in thousands)")
plt.legend()
plt.show()
```

---

### 4.2 Key Outputs
- **`model.coef_`**: Coefficients (slopes for each feature).
- **`model.intercept_`**: Intercept (bias term).

```python
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
```

---

## 5. Evaluating Regression Models

### Common Metrics:
1. **Mean Squared Error (MSE)**:
   Measures average squared difference between actual and predicted values.
   ```txt
   MSE = (1/n) * Σ(actual - predicted)²
   ```

2. **R-squared (R²)**:
   Proportion of variance explained by the model.
   ```txt
   R² = 1 - (sum of squared residuals / total sum of squares)
   ```

---

## 6. Building Linear Regression from Scratch

### Example: Single Variable Regression

1. Use **Gradient Descent** to minimize the cost function (**MSE**).
2. Update weights (`m`) and intercept (`c`) iteratively.

#### Step-by-step Implementation:
```python
import numpy as np

# Step 1: Define the cost function
def compute_cost(X, y, w, b):
    n = len(y)
    predictions = w * X + b
    cost = (1/(2*n)) * np.sum((predictions - y)**2)
    return cost

# Step 2: Gradient Descent Algorithm
def gradient_descent(X, y, w, b, learning_rate, iterations):
    n = len(y)
    for _ in range(iterations):
        predictions = w * X + b
        dw = (1/n) * np.sum((predictions - y) * X)  # Derivative w.r.t weight
        db = (1/n) * np.sum(predictions - y)       # Derivative w.r.t bias
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b

# Step 3: Initialize variables
X = np.array([1, 2, 3, 4, 5])    # Features
y = np.array([3, 7, 11, 15, 19]) # Target values
w, b = 0, 0                      # Initial weight and bias
learning_rate = 0.01
iterations = 1000

# Step 4: Run Gradient Descent
final_w, final_b = gradient_descent(X, y, w, b, learning_rate, iterations)
print("Final Weight:", final_w)
print("Final Bias:", final_b)

# Step 5: Make predictions
predictions = final_w * X + final_b
print("Predicted Values:", predictions)

# Step 6: Visualize
import matplotlib.pyplot as plt
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, predictions, color="red", label="Fitted Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

---

## 7. Multiple Linear Regression

In multiple linear regression, the model considers multiple independent variables.

### Building from Scratch Template:

```python
# Predicts y = w1*x1 + w2*x2 + ... + wn*xn + b
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

def gradient_descent_multi(X, y, weights, bias, learning_rate, iterations):
    n = len(y)
    for _ in range(iterations):
        predictions = predict(X, weights, bias)
        residuals = predictions - y
        
        # Update weights and bias
        dw = (1/n) * np.dot(X.T, residuals)    # Gradient of weights
        db = (1/n) * np.sum(residuals)         # Gradient of bias
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

# Example Inputs
X = np.array([[1, 1], [2, 2], [3, 3]])  # Features
y = np.array([5, 8, 11])                # Targets
weights = np.zeros(X.shape[1])          # Initial weights
bias = 0                                # Initial bias
learning_rate = 0.01
iterations = 1000

# Run Gradient Descent
final_weights, final_bias = gradient_descent_multi(X, y, weights, bias, learning_rate, iterations)
print("Final Weights:", final_weights)
print("Final Bias:", final_bias)
```

---

## 8. Practical Tips

1. **Feature Scaling**:
   Standardize input features for faster convergence.
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Train-Test Splitting**:
   Always split your data into training and testing datasets.

3. **Regularization**:
   Use **L1 Regularization (Lasso)** or **L2 Regularization (Ridge)** to reduce overfitting.

---

This breakdown starts from simple concepts to advanced implementation, empowering you to train, evaluate, and tune Linear Regression models with Python step-by-step!