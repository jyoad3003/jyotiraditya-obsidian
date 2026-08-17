---
tags:
  - algorithm
  - supervised/classification
  - status/complete
aliases:
  - Logistic Regression
created: 2026-08-17
---
# Logistic Regression

> [!tip] **Note Relocation**
> This note is part of the comprehensive **Machine Learning Knowledge Base**. The primary detailed deep-dive note is located at:
> ➡️ **[[Machine Learning/03 - Supervised Learning/Logistic Regression|Logistic Regression (Deep Dive)]]**

---

## 📖 Overview
- **Purpose**: Binary and multi-class probabilistic classification mapping $\mathbf{x} \in \mathbb{R}^d \rightarrow P(y=1 \mid \mathbf{x}) \in [0, 1]$.
- **Input / Output**: Feature matrix $\mathbf{X} \in \mathbb{R}^{n \times d} \rightarrow$ Probabilities $\hat{\mathbf{p}} \in [0, 1]^n$ / Class labels $\hat{y} \in \{0, 1\}$.

---

## 📝 Key Equations & Concepts
1. **Hypothesis (Sigmoid activation)**:
   $$
   \hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
   $$
2. **Log-Odds (Logit)**:
   $$
   \log\left(\frac{p}{1-p}\right) = \mathbf{w}^T \mathbf{x} + b
   $$
3. **Loss Function (Binary Cross-Entropy)**:
   $$
   J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
   $$
4. **Gradient Update**:
   $$
   \mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{n}\mathbf{X}^T(\mathbf{\hat{p}} - \mathbf{y})
   $$

---

## 💻 Implementation Steps
1. **Initialize** weights $\mathbf{w} = \mathbf{0}$ and bias $b = 0$.
2. **Forward pass**: Compute linear combinations $z = \mathbf{X}\mathbf{w} + b$ and apply Sigmoid $\sigma(z)$ to get probabilities $\hat{p}$.
3. **Compute Loss**: Calculate binary cross-entropy.
4. **Backward pass**: Calculate gradients $\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n}\mathbf{X}^T(\hat{p} - y)$ and $\frac{\partial J}{\partial b} = \frac{1}{n}\sum(\hat{p}-y)$.
5. **Update parameters** using Gradient Descent.
6. **Inference**: Predict class 1 if $\hat{p} \ge 0.5$, else 0.

---

## 🔗 Related Notes
- [[Linear Regression]]
- [[Loss Functions & Cost Functions]]
- [[Gradient Descent & Optimizers]]
- [[Supervised Learning MOC]]
