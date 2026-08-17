---
tags:
  - machine-learning/algorithm
  - supervised/boosting
  - status/complete
aliases:
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
  - GBDT
type: algorithm
paradigm: Supervised
task: Classification # and Regression
difficulty: Intermediate
prerequisites:
  - "[[Decision Trees]]"
  - "[[Multivariate Calculus & Gradients]]"
  - "[[Loss Functions & Cost Functions]]"
created: 2026-08-17
---

# 🚀 Gradient Boosting & XGBoost

> [!summary] **Algorithm at a Glance**
> **Goal**: An ensemble technique that constructs trees **sequentially**, where each new tree is trained to predict the **pseudo-residuals (negative gradients)** of the previous ensemble.  
> **Key Equation**: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$  
> **Status**: The reigning champion for tabular data competitions (Kaggle) and production tabular systems.

---

## 🧭 Bagging vs Boosting: The Mental Shift

```mermaid
flowchart TD
    subgraph Bagging [Random Forests: Bagging (Parallel)]
        Data1[Data] --> T1[Tree 1 (Independent)]
        Data1 --> T2[Tree 2 (Independent)]
        Data1 --> T3[Tree 3 (Independent)]
        T1 & T2 & T3 --> Avg[Average / Majority Vote]
        Note1[Reduces Variance]
    end

    subgraph Boosting [Gradient Boosting: Boosting (Sequential)]
        Data2[Data] --> B1[Tree 1: Makes Predictions]
        B1 --> Res1["Calculate Residuals (Errors)"]
        Res1 --> B2["Tree 2: Fits Residuals"]
        B2 --> Res2["Calculate Remaining Residuals"]
        Res2 --> B3["Tree 3: Fits Remaining Residuals"]
        Note2[Reduces Bias & Variance]
    end
```

---

## 📐 Mathematical Formulation (Gradient Boosted Trees)

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ and differentiable loss function $\mathcal{L}(y, \hat{y})$:

### Step 1: Initialize with a Constant Baseline
$$
F_0(\mathbf{x}) = \arg\min_{\gamma} \sum_{i=1}^n \mathcal{L}(y_i, \gamma)
$$
*(For MSE loss, $F_0(\mathbf{x}) = \bar{y}$ is simply the mean of all targets).*

---

### Step 2: For iterations $m = 1$ to $M$:
1. **Compute Pseudo-Residuals** (the negative gradient of the loss surface with respect to current predictions):
   > [!math] **Negative Gradient (Residual)**
   > $$
   > r_{im} = - \left[ \frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F(\mathbf{x}) = F_{m-1}(\mathbf{x})}
   > $$
   *(For MSE loss, $r_{im} = y_i - F_{m-1}(\mathbf{x}_i)$ is the exact physical residual!)*

2. **Fit a regression tree** $h_m(\mathbf{x})$ to the pseudo-residuals $\{(\mathbf{x}_i, r_{im})\}_{i=1}^n$.

3. **Update the ensemble model** with learning rate (shrinkage) $\eta \in (0, 1]$:
   $$
   F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})
   $$

---

## ⚡️ The Modern Holy Trinity: XGBoost vs LightGBM vs CatBoost

| Feature | **XGBoost** (Chen & Guestrin, 2016) | **LightGBM** (Microsoft, 2017) | **CatBoost** (Yandex, 2018) |
| :--- | :--- | :--- | :--- |
| **Splitting Strategy** | Level-wise (depth-first) | **Leaf-wise** (best-first) | Symmetric / Oblivious trees |
| **Math Optimization** | 2nd-order Taylor (Hessians $h_i$) | 1st & 2nd order + Histograms | Ordered boosting (prevents target leakage) |
| **Categorical Data** | One-hot / Partitioning | Integer binning | **Native, state-of-the-art target encoding** |
| **Speed & Memory** | ⚡️ Fast | ⚡️⚡️ **Ultra Fast & Low RAM** | ⚡️ Fast GPU training |

---

## 🛠️ Practical Hyperparameter Tuning Guide

- `learning_rate` ($\eta$): Controls the step size shrinkage ($0.01 \sim 0.1$). Lower $\eta$ with higher `n_estimators` generalizes better!
- `n_estimators`: Total number of boosting rounds (use with Early Stopping).
- `max_depth`: Depth of each tree ($3 \sim 6$). Shallow trees prevent overfitting!
- `subsample`: Fraction of rows randomly sampled per tree ($0.7 \sim 0.9$).
- `colsample_bytree`: Fraction of features randomly sampled per split ($0.7 \sim 0.9$).
- `reg_lambda` ($L_2$) & `reg_alpha` ($L_1$): Regularization penalties on leaf weights.

---

## 💻 Python Implementation (Scikit-Learn / XGBoost)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting with early stopping
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=10, # Early stopping patience
    random_state=42
)

gbm.fit(X_train, y_train)

y_probs = gbm.predict_proba(X_test)[:, 1]
print(f"Optimal Trees Used: {gbm.n_estimators_}")
print(f"Test ROC-AUC:      {roc_auc_score(y_test, y_probs):.4f}")
```

---

## 🔗 Related Notes & Graph Connections
- **Building Blocks**: [[Decision Trees]], [[Multivariate Calculus & Gradients]]
- **Comparison**: [[Random Forests]]
- **Parent Hub**: [[Supervised Learning MOC]]
