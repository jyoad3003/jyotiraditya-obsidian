---
tags:
  - machine-learning/algorithm
  - supervised/classification
  - status/complete
aliases:
  - Logistic Regression
  - Logit
  - Sigmoid Classifier
type: algorithm
paradigm: Supervised
task: Classification
difficulty: Beginner
prerequisites:
  - "[[Linear Regression]]"
  - "[[Loss Functions & Cost Functions]]"
  - "[[Gradient Descent & Optimizers]]"
created: 2026-08-17
---

# 🔘 Logistic Regression

> [!summary] **Algorithm at a Glance**
> **Goal**: Predict the posterior probability $P(y=1 \mid \mathbf{x})$ that an instance belongs to a binary class ($0$ or $1$) using a linear combination of features squashed through the **Sigmoid function**.  
> **Key Equation**: $\hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$  
> **Decision Boundary**: $\mathbf{w}^T \mathbf{x} + b = 0$ (Linear hyperplane).

---

## 🎯 Intuition & Mental Model

Why can't we just use [[Linear Regression]] for classification?
1. **Unbounded Output**: Linear regression produces values from $-\infty$ to $+\infty$, which cannot represent probabilities ($[0, 1]$).
2. **Extreme Outliers**: A distant point pulls the regression line and shifts the classification threshold unpredictably.

```
Probability P(y=1)
  1.0 +                      .---'""  <-- Sigmoid "S-curve" squashes any real
      |                    .'             number into a valid probability (0 to 1)
  0.5 + - - - - - - - - - / - - - - - - Threshold = 0.5
      |                 .'
  0.0 +__________....--'----------------> Linear score z = w^T x + b
              Negative        Positive
```

---

## 📐 Mathematical Formulation

### 1. The Sigmoid (Logistic) Function
$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$
- **Key Derivative Property**:
  $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$

### 2. Odds and Log-Odds (The Logit Transform)
Let $p = P(y=1 \mid \mathbf{x})$. The **Odds** is $\frac{p}{1-p}$.
Taking the natural logarithm yields the **Logit**:

> [!math] **Linear Log-Odds**
> $$
> \log\left(\frac{p}{1-p}\right) = \mathbf{w}^T \mathbf{x} + b
> $$
> **Interpretation**: Logistic regression is a linear model for the *log-odds* of the positive outcome. A unit increase in $x_j$ multiplies the odds of $y=1$ by $e^{w_j}$.

---

### 3. Cost Function: Binary Cross-Entropy (Log-Loss)

> [!warning] **Why MSE Fails for Logistic Regression**
> Using Mean Squared Error with a Sigmoid produces a **non-convex** cost function with numerous poor local minima and flat vanishing gradients. We must use **Binary Cross-Entropy**, which is strictly convex!

$$
J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

### 4. Gradient of the Cost Function
Using the chain rule with $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$:

> [!math] **The Gradient Vector**
> $$
> \nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{n} \mathbf{X}^T (\mathbf{\hat{p}} - \mathbf{y})
> $$
> The parameter update rule for Gradient Descent is:
> $$
> \mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{n} \mathbf{X}^T (\sigma(\mathbf{X}\mathbf{w}) - \mathbf{y})
> $$

---

## 🌐 Multi-Class Extension: Softmax Regression

For $K > 2$ classes, we replace the Sigmoid with the **Softmax Function**:

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}
$$

---

## 💻 Python Implementation (From Scratch & Scikit-Learn)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Logistic Regression from Scratch
class ScratchLogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            
            # Gradients
            dw = (1 / n_samples) * (X.T @ (p - y))
            db = (1 / n_samples) * np.sum(p - y)
            
            # Gradient Descent Update
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self
        
    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)
        
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Quick verification on synthetic classification data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

clf = ScratchLogisticRegression(lr=0.5, epochs=500).fit(X, y)
preds = clf.predict(X)
print(f"Scratch Accuracy: {accuracy_score(y, preds) * 100:.2f}%")
```

---

## 🔗 Related Notes & Graph Connections
- **Foundations**: [[Probability & Statistics for ML]], [[Loss Functions & Cost Functions]]
- **Extensions**:
  - [[Decision Trees]] (Non-linear alternatives)
  - [[Support Vector Machines (SVM)]] (Max-margin linear alternative)
  - [[Evaluation Metrics for ML]] (ROC-AUC, Precision/Recall, Log-Loss)
- **Parent Hub**: [[Supervised Learning MOC]]
