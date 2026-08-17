---
tags:
  - machine-learning/algorithm
  - deep-learning/vision
  - status/complete
aliases:
  - CNN
  - Convolutional Neural Networks
  - Computer Vision
  - ResNet
type: algorithm
paradigm: Supervised # Deep Learning
task: Computer Vision
difficulty: Intermediate
prerequisites:
  - "[[Perceptrons & Multi-Layer Perceptrons]]"
  - "[[Activation Functions]]"
created: 2026-08-17
---

# 👁️ Convolutional Neural Networks (CNN)

> [!summary] **Algorithm at a Glance**
> **Goal**: Specialized deep neural network architecture designed for grid-structured spatial data (images, video, spectrograms).  
> **Key Innovations**: **Local receptive fields**, **parameter sharing** (reusable sliding filters), and **translation equivariance** (detecting features regardless of where they appear in the image).

---

## 🎯 Why Standard MLPs Fail on Images

If you flatten a $1000 \times 1000$ RGB color image into an MLP:
- **Input Dimension**: $1000 \times 1000 \times 3 = 3,000,000$ features.
- A single hidden layer with 1,000 neurons requires **$3 \text{ billion}$ parameters**!
- Flattening completely destroys the 2D spatial relationships between neighboring pixels.

```mermaid
flowchart LR
    Img["2D Image (H x W x C)"] --> Conv["Conv2D Filters (Feature Maps)"]
    Conv --> Pool["Max Pooling (Downsampling)"]
    Pool --> Flatten["Flatten"]
    Flatten --> FC["Dense Classifier Head"]
    FC --> Out["Probabilities (e.g. Dog, Cat, Car)"]
```

---

## 📐 The 3 Core Operations of a CNN

### 1. The 2D Convolution Operation (Feature Extraction)
A learnable small filter (kernel) of size $K \times K$ (e.g., $3 \times 3$) slides across the image computing dot products:

> [!math] **Discrete 2D Cross-Correlation**
> $$
> S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n)
> $$
> - **Early Layers**: Learn low-level edges, textures, and color gradients.
> - **Deeper Layers**: Compose low-level edges into eyes, wheels, and complex objects.

---

### 2. Spatial Dimension Output Formula (Stride & Padding)
For an input of width $W$, kernel size $K$, padding $P$, and stride $S$:

> [!math] **Output Dimension Equation**
> $$
> W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1
> $$
> - **Padding ($P$)**: Adding zeros around the image border (*'same'* padding preserves image size).
> - **Stride ($S$)**: Step size of the sliding filter ($S=2$ halves spatial resolution).

---

### 3. Max Pooling (Dimensionality Reduction & Invariance)
Slides a window (e.g., $2 \times 2$ with stride 2) and keeps only the **maximum value**:
- Reduces spatial dimension by $50\%$, cutting computational cost.
- Provides slight translation and rotation invariance.

---

## 🏛️ The ResNet Breakthrough: Residual Skip Connections

As networks grew deeper (e.g., 20+ layers), training accuracy degraded because gradients vanished.  
He et al. (2015) introduced **Residual Connections**:

> [!math] **Residual Block**
> $$
> \mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}
> $$

```
        Input x
          | \
          |  \---> [ Weight Layer ] ---> [ ReLU ] ---> [ Weight Layer ]
          |                                                   |
          | (Identity Shortcut / Skip Connection)            |
          +---------------------------------------------(+)--+
                                                         |
                                                      Output y
```

- If a layer is unhelpful, the network can easily learn $\mathcal{F}(\mathbf{x}) = 0$, allowing information and gradients to flow unimpeded directly through the identity shortcut $\mathbf{x}$!

---

## 💻 Python Implementation (PyTorch)

```python
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, 3, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (B, 32, 16, 16)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                     # Output: (B, 64, 8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Verify tensor dimensions
model = ConvNet()
dummy_img = torch.randn(4, 3, 32, 32) # Batch 4, RGB 32x32 images
logits = model(dummy_img)
print(f"Output Batch Logits Shape: {logits.shape}") # (4, 10)
```

---

## 🔗 Related Notes & Graph Connections
- **Foundations**: [[Perceptrons & Multi-Layer Perceptrons]], [[Activation Functions]]
- **Alternative Spatial/Sequential Architectures**:
  - [[Transformers & Attention]] (Vision Transformers / ViT)
  - [[Recurrent Neural Networks & LSTM]]
- **Parent Hub**: [[Deep Learning MOC]]
