---
tags:
  - machine-learning/algorithm
  - supervised/regression
  - status/complete
aliases:
  - Linear Regression
  - Ordinary Least Squares
  - OLS
type: algorithm
paradigm: Supervised
task: Regression
difficulty: Beginner
prerequisites:
  - "[[Linear Algebra for ML]]"
  - "[[Multivariate Calculus & Gradients]]"
  - "[[Loss Functions & Cost Functions]]"
created: 2026-08-17
---

# 📈 Linear Regression & Ordinary Least Squares (OLS)

> [!summary] **Algorithm at a Glance**
> **Goal**: Model the linear relationship between continuous dependent target variable $y \in \mathbb{R}$ and one or more explanatory feature vectors $\mathbf{x} \in \mathbb{R}^d$.  
> **Key Equation**: $\hat{y} = \mathbf{w}^T \mathbf{x} + b$  
> **Optimization**: Minimizes the sum of squared differences between predictions and true labels (OLS).

---

## 🎯 Intuition & Geometric Picture

Imagine a scatter plot of house sizes vs prices. Linear Regression draws the single straight line (or hyper-plane in $>2$ dimensions) that minimizes the vertical distances (residuals) from every data point to the line.

```
Price (y)
  ^
  |          * (y_i)
  |         /|
  |        / |  <-- Residual e_i = y_i - y_hat_i
  |       /  * (y_hat_i on line)
  |      /       *
  |     /  *
  |    /
  +---+------------------> Square Footage (x)
```

---

## 📐 Mathematical Formulation

### 1. The Hypothesis Function
In matrix notation (absorbing bias $b$ as $w_0$ with dummy feature $x_0 = 1$):
$$
\mathbf{\hat{y}} = \mathbf{X} \mathbf{w}
$$
where $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ is the design matrix and $\mathbf{w} \in \mathbb{R}^{d+1}$ is the weight vector.

---

### 2. The Cost Function (Ordinary Least Squares)
$$
J(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = \frac{1}{2n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})
$$

---

### 3. Solving for Optimal Weights $\mathbf{w}^*$

#### Method A: Analytical Solution (The Normal Equation)
Taking the matrix derivative with respect to $\mathbf{w}$ and setting it to zero:
$$
\nabla_{\mathbf{w}} J(\mathbf{w}) = -\frac{1}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0} \implies \mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y}
$$

> [!math] **The Normal Equation**
> $$
> \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
> $$
> - **Time Complexity**: $O(d^3)$ due to matrix inversion.
> - **Best for**: Small to medium feature spaces ($d < 10,000$).

#### Method B: Iterative Solution (Gradient Descent)
For large-scale datasets ($n \gg 100,000$ or $d \gg 10,000$):
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{n} \mathbf{X}^T (\mathbf{X}\mathbf{w}_t - \mathbf{y})
$$

---

## 📋 The 4 Classical Statistical Assumptions (L.I.N.E.)
To trust the $p$-values and confidence intervals of linear regression:
1. **L — Linearity**: The true relationship between features and target is linear.
2. **I — Independence**: Residuals are independent of each other (no autocorrelation in time series).
3. **N — Normality**: Residuals $\epsilon = y - \hat{y}$ are normally distributed ($\epsilon \sim \mathcal{N}(0, \sigma^2)$).
4. **E — Equal Variance (Homoscedasticity)**: Variance of residuals is constant across all predicted values.

---

## ⚖️ Strengths, Weaknesses & Variations

| Strengths (Pros) | Weaknesses (Cons) |
| :--- | :--- |
| • Highly interpretable coefficients ($w_j = \Delta y / \Delta x_j$)<br>• Fast to train and evaluate ($O(1)$ inference per sample)<br>• Does not overfit easily on simple datasets | • Cannot capture non-linear relationships without polynomial expansion<br>• Highly sensitive to outliers (due to squared loss)<br>• Fails with multicollinearity unless regularized |

### 🔹 Regularized Variants
- **Ridge Regression ($L_2$)**: Adds $\lambda \|\mathbf{w}\|_2^2 \implies \mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$. Handles collinearity.
- **Lasso Regression ($L_1$)**: Adds $\lambda \|\mathbf{w}\|_1$. Zeroes out uninformative features.
- See [[Regularization Techniques]] for deep-dive.

---

## 💻 Python Implementation (From Scratch & Scikit-Learn)

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Normal Equation from Scratch
class NormalEquationLinearRegression:
    def fit(self, X, y):
        # Add bias column of ones
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # w = (X^T X)^-1 X^T y
        self.w_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = self.w_[0]
        self.coef_ = self.w_[1:]
        return self
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.w_

# Synthetic test
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.ravel() + np.random.randn(100) # True w0=4, w1=3

# Compare Scratch vs Scikit-learn
scratch_model = NormalEquationLinearRegression().fit(X, y)
sklearn_model = LinearRegression().fit(X, y)

print(f"Scratch: Intercept = {scratch_model.intercept_:.4f}, Coef = {scratch_model.coef_[0]:.4f}")
print(f"Sklearn: Intercept = {sklearn_model.intercept_:.4f}, Coef = {sklearn_model.coef_[0]:.4f}")
```

---

## 🔗 Related Notes & Graph Connections
- **Underlying Math**: [[Linear Algebra for ML]], [[Multivariate Calculus & Gradients]]
- **Extensions**:
  - [[Logistic Regression]] (Classification counterpart using Sigmoid)
  - [[Regularization Techniques]] (Ridge, Lasso, ElasticNet)
  - [[Evaluation Metrics for ML]] (MSE, RMSE, $R^2$)
- **Parent Hub**: [[Supervised Learning MOC]]
