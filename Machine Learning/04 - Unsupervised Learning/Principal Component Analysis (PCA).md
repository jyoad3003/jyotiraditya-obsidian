---
tags:
  - machine-learning/algorithm
  - unsupervised/dimensionality-reduction
  - status/complete
aliases:
  - PCA
  - Principal Component Analysis
  - Dimensionality Reduction
type: algorithm
paradigm: Unsupervised
task: Dimensionality Reduction
difficulty: Intermediate
prerequisites:
  - "[[Linear Algebra for ML]]"
created: 2026-08-17
---

# 📉 Principal Component Analysis (PCA)

> [!summary] **Algorithm at a Glance**
> **Goal**: Unsupervised linear dimensionality reduction technique that projects $d$-dimensional data onto a lower $k$-dimensional subspace ($k < d$) while **maximizing the variance** of the projected data (or equivalently, minimizing reconstruction error).  
> **Key Mathematical Engine**: Eigendecomposition of the Sample Covariance Matrix or **Singular Value Decomposition (SVD)**.

---

## 🎯 Intuition: Finding the Best Camera Angle

Imagine a 3D flock of birds in the sky. If you take a 2D photograph from the side, you might see a wide, spread-out pattern with lots of detail (High Variance = High Information). If you take a photo from directly underneath, all the birds might overlap into an unreadable clump (Low Variance).

PCA calculates the exact **optimal camera angle (principal component axes)** to capture maximum information!

```
Feature 2
   ^           /  <-- PC1: Direction of MAXIMUM variance (Largest eigenvalue λ1)
   |       *  /
   |     *   *
   |   *   * /
   |  *  *  /
   | *  *  /
   | \    /
   |  \  /     <-- PC2: Orthogonal direction of 2nd greatest variance (λ2)
   +---+----------------------> Feature 1
```

---

## 📐 Step-by-Step Mathematical Derivation

### Step 1: Standardize & Mean-Center Data
$$
\mathbf{X}_c = \mathbf{X} - \mathbf{\mu}
$$

---

### Step 2: Compute the Sample Covariance Matrix $\mathbf{\Sigma}$
$$
\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{d \times d}
$$

---

### Step 3: Compute Eigenvalues & Eigenvectors
Solve the characteristic equation:
$$
\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j
$$
- Sort eigenvalues in descending order: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$.
- The corresponding eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_k$ form the projection matrix $\mathbf{W}_k \in \mathbb{R}^{d \times k}$.

---

### Step 4: Project onto the $k$-Dimensional Subspace
$$
\mathbf{Z} = \mathbf{X}_c \mathbf{W}_k \in \mathbb{R}^{n \times k}
$$

---

## 📊 Explained Variance Ratio & Scree Plot

The fraction of total dataset variance explained by the $j$-th principal component is:

> [!math] **Explained Variance Ratio**
> $$
> \text{EVR}_j = \frac{\lambda_j}{\sum_{i=1}^d \lambda_i}
> $$

```
Cumulative Variance
  100% +                    .---- 95% Cutoff Threshold (e.g. k=10 components)
       |                .--'
   80% +             .-'
       |          .-'
   50% +       .-'
       +-------+----+----+----+----> Number of Components (k)
               1    2    3    10
```

---

## 💻 Python Implementation (Scikit-Learn)

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load 64-dimensional hand-written digits (8x8 pixel images)
digits = load_digits()
X = digits.data # Shape: (1797, 64)

# 1. Mandatory Scaling
X_scaled = StandardScaler().fit_transform(X)

# 2. PCA preserving 95% of total variance
pca_95 = PCA(n_components=0.95, random_state=42)
X_reduced = pca_95.fit_transform(X_scaled)

print(f"Original Dimensionality: {X.shape[1]} features")
print(f"Reduced Dimensionality:  {X_reduced.shape[1]} components (Preserving 95% variance)")
print(f"Variance explained by first 2 PCs: {np.sum(pca_95.explained_variance_ratio_[:2]) * 100:.2f}%")
```

---

## 🔗 Related Notes & Graph Connections
- **Foundations**: [[Linear Algebra for ML]] (SVD & Eigendecomposition)
- **Non-Linear Alternatives**: [[t-SNE & UMAP]]
- **Parent Hub**: [[Unsupervised Learning MOC]]
