---
tags:
  - machine-learning/concept
  - fundamentals
  - status/complete
aliases:
  - Traditional Programming vs ML
  - Rules vs Models
  - Classification vs Clustering
created: 2026-08-17
---

# Traditional Programming VS Machine Learning

> [!tip] **Knowledge Base Connections**
> This foundational note connects to:
> - [[Machine Learning Paradigms]] (Supervised vs Unsupervised frameworks)
> - [[Linear Regression]] (Continuous Estimation)
> - [[Logistic Regression]] & [[k-Nearest Neighbors (KNN)]] (Classification)
> - [[K-Means & K-Means++]] (Centroid-based clustering & smart initialization)
> - [[Machine Learning MOC]] (Central Index)

---

> “If you look at machine learning (ML) and data mining (DM) development, you have a bunch of outputs, you have a bunch of inputs, and you develop the program in the middle. Instead of predicting the output with explicitly coded rules, you try to come up with a probabilistic model that maps the input to the output.”

### What This Means

1. **Traditional Programming vs. Machine Learning:**
    
    - **Traditional Programming:** You manually write code (if-else conditions, loops, etc.) to get from an input to an output. The logic is explicitly defined by a human programmer.
        
    - **Machine Learning (ML):** Instead of hand-coding rules, you let the data and an algorithm “learn” those rules automatically. You have inputs (features $\mathbf{X}$) and outputs (labels or target $\mathbf{y}$), and your goal is to discover a function (or model) that best maps $\mathbf{X}$ to $\mathbf{y}$. (See [[Machine Learning Paradigms]]).
        
2. **Probabilistic Model:**
    
    - ML models often operate in a probabilistic manner. Rather than giving you a definitive rule (“IF color = red THEN it’s an apple”), they learn probabilities (“There’s an 80% chance this fruit is an apple given its color and other features”).
        
    - This probabilistic view captures the idea that real-world data can have noise or uncertainty, and the best we can do is find a likely or most probable mapping. (See [[Probability & Statistics for ML]]).
        
3. **Why It’s Important:**
    
    - Because real-world data is rarely perfectly clean or deterministic, being able to estimate “how likely” an outcome is can be much more powerful and realistic than a single hard-coded rule. This approach lets the model adapt as more data arrives, rather than requiring a developer to rewrite code for every new scenario.
        

---

## Statement 2

> “If the output is continuous, then we call that estimation. In other words, if I have a bunch of XXX and there’s a bunch of YYY values that are continuous, and I can find a way to map the record XXX to an outcome YYY that is continuous, we call that estimation.”

### Breaking It Down

1. **Continuous Output = Regression/Estimation:**
    
    - In machine learning terminology, when the target $\mathbf{y}$ is a continuous numeric value (like temperature, price, height, etc.), we typically refer to the task as **[[Linear Regression|regression]]** or **estimation**.
        
    - **Example:** Predicting house prices from features like number of bedrooms, location, and square footage is a regression/estimation problem because the predicted price is a continuous number.
        
2. **Why “Estimation” Instead of “Prediction?”**
    
    - The word “estimation” emphasizes that we are approximating a continuous quantity, often recognizing there is uncertainty. We are “estimating” the likely value within some range, rather than deciding among discrete categories.
        
    - In contrast, if the output $\mathbf{y}$ were categorical (e.g., “spam” vs. “not spam”), we’d call it **[[Logistic Regression|classification]]** or **categorical prediction**.
        
3. **Mapping $\mathbf{X}$ to $\mathbf{y}$:**
    
    - You’re taking a set of features $\mathbf{X} = (x_1, x_2, \dots, x_d)$ and training a model $f(\mathbf{X})$ such that $f(\mathbf{X})$ approximates the continuous value $\mathbf{y}$.
        
    - The model learns from many examples (pairs of $(\mathbf{x}, y)$ in your training data) and then generalizes to predict a new $y$ for unseen $\mathbf{x}$.
        

---

## Putting It All Together

1. **Inputs and Outputs in ML/DM:**
    
    - You have **inputs** (feature vectors $\mathbf{X}$) and **outputs** (labels or targets $\mathbf{y}$).
        
    - The “program in the middle” is not hand-coded logic but a learned model that captures patterns in the data.
        
2. **Probabilistic Model vs. Deterministic Rules:**
    
    - Real-world data is noisy or has inherent variability, so using a probabilistic approach lets you handle uncertainty more naturally.
        
    - This approach is more flexible and can adapt when more data becomes available or when the data distribution changes over time.
        
3. **Continuous vs. Discrete Outputs:**
    
    - **Continuous (Estimation/[[Linear Regression|Regression]]):** If $\mathbf{y}$ is a real-valued variable (e.g., 25.4 °C, $300, etc.).
        
    - **Discrete ([[Logistic Regression|Classification]]):** If $\mathbf{y}$ is a category (e.g., “cat” or “dog,” “spam” or “not spam,” “apple,” “banana,” or “orange”).
        
4. **Why Call It ‘Estimation?’**
    
    - When dealing with continuous outputs, the model is essentially estimating a numeric value—often with an associated level of uncertainty.
        
    - This underscores the idea that we don’t have a perfectly correct or exact answer every time, but rather a best guess given the existing data and learned parameters.
        

---

## Conclusion

Your professor is emphasizing two core ideas:

1. **Modern ML Approaches**: We rely on data-driven models instead of hard-coded rules. These models often produce probabilistic outputs, reflecting real-world uncertainty.
    
2. **Types of Outputs**: How we name the task (estimation vs. classification) depends on whether we’re predicting a continuous value or a categorical label.
    

Together, these statements lay the groundwork for understanding how machine learning shifts from explicit programming to data-driven model building, and how tasks differ based on the nature of the outcome variable.

---

# Classification vs Clustering 
Here’s a **concise, all‐in‐one definition** that ties together every aspect we’ve discussed—including your key insight about “closeness” in classification vs. clustering:

---

> **[[Supervised Learning MOC|Classification]]** is a **supervised** learning task in which we assign each new example to one of several **predefined classes** by measuring its **distance** (or similarity) to **labeled training points**.
> 
> - **Closeness is to labeled examples**: e.g. in **[[k-Nearest Neighbors (KNN)]]**, a fruit is classified as an “Apple” because it lies nearest to many previously labeled “Apple” points in feature space, and the majority vote of those neighbors determines its class.
>     
> - The model learns decision boundaries (often probabilistic) from the known labels, then uses those boundaries to predict unseen data.
>     

> **[[Unsupervised Learning MOC|Clustering]]** is an **unsupervised** learning task in which we discover **natural groups** in the data by measuring each point’s **distance** (or similarity) to a set of **cluster centroids**.
> 
> - **Closeness is to centroids**: e.g. in **[[K-Means & K-Means++]]**, you choose K clusters and (optionally via K-Means++) initialize K centroids. Each fruit is assigned to the cluster whose centroid it is closest to; then centroids are recomputed as the mean of their cluster’s points, and the process repeats until stable.
>     
> - No labels are used or learned—clusters simply represent regions of high feature similarity.
>     

---

### Key Distinction in One Sentence

- **Classification** groups by distance to **known labels** (and then votes or applies a probabilistic rule).
    
- **Clustering** groups by distance to **learnt centroids** (and then refines those centroids iteratively).
    

Keep this definition handy, and you’ll see at a glance why classification needs labeled examples (to “vote” on or learn boundaries from) and clustering needs centroids (to “gather” similar points around).

You're thinking **exactly right**, and you're asking a **super important question** that even professionals struggle with when starting with clustering:

> **How does K-Means choose those initial 3 centroids? And what if they’re all close together? Won’t that mess up the clustering?**

Let me walk you through it clearly and break your confusion step-by-step.

---

## 🎯 The Core of Your Question:

> Are the initial 3 centroids chosen randomly from the actual dataset?

✅ **YES — usually they are.**  
The most basic version of **K-Means** just:

- **Randomly picks 3 data points** from the dataset itself as the starting centroids.
    

So if your dataset has 100 fruits, K-Means might just say:

> “Hey, pick 3 fruits randomly and use _their feature values_ as the 3 starting centroids.”

---

## 😬 So… Can This Cause Problems?

### ✔️ YES — and you spotted it!

Imagine if the 3 randomly selected centroids happen to be:

- All **close together**, like 3 apples that are very similar.
    

Then:

- All the fruits will get assigned based on distance to those 3 similar points.
    
- That leads to **bad clustering** because it **doesn’t represent** the full structure of the data.
    

This can result in:

- Empty clusters
    
- Poor convergence
    
- Wrong grouping
    

So your confusion is **valid and actually important**.

---

## 🧠 So, How is This Usually Handled in Practice?

### 🛠️ Option 1: **Basic K-Means** (random pick)

- Just randomly picks K points.
    
- Fast, but can go wrong.
    

### 🧠 Option 2: **[[K-Means & K-Means++|K-Means++]]** (smart initialization)

This is the **smarter version** most people use in practice. (See [[K-Means & K-Means++]] for complete derivation).

K-Means++ tries to spread out the starting centroids.

Here’s how it works:

1. Pick the **first centroid randomly** from the data points.
    
2. For every other point, compute its distance from the **nearest already chosen centroid**.
    
3. Select the next centroid **with probability proportional to the square of that distance**.
    

➡️ So if a point is **far** from existing centroids, it's **more likely to be chosen** as the next centroid.

This spreads out centroids nicely and **avoids poor starting points**.

---

## 🔍 Example: Fruit Dataset

Assume your data (simplified for clarity):

| Fruit    | Color | Weight | Sweetness | Texture |
| -------- | ----- | ------ | --------- | ------- |
| Apple    | 1     | 180    | 6.8       | 1       |
| Banana   | 2     | 120    | 8.5       | 2       |
| Orange   | 3     | 150    | 6.5       | 3       |
| Apple 2  | 1     | 175    | 6.9       | 1       |
| Banana 2 | 2     | 118    | 8.4       | 2       |

### ❌ Bad Random Init (Basic K-Means):

Let’s say it randomly picks:

- Apple
    
- Apple 2
    
- Banana 2
    

Then the algorithm starts with **2 very similar Apple centroids** and one Banana.

=> Orange might be misclustered early on.

### ✅ Smart Init (K-Means++):

- First: picks Banana randomly.
    
- Then: calculates distances to Banana for all other points.
    
- Picks a point **far away** (like Orange or Apple) next.
    
- Ensures better initial diversity.
    

---

## ✅ Final Clarification

| Concept | Explanation |
| :--- | :--- |
| Random centroids in K-Means | Usually just picks K **random points** from the dataset. |
| Problem with that | They might be **too close** to each other — giving bad starting clusters. |
| Why that matters | The whole clustering process depends on **initial positions** of centroids. |
| Solution | Use **[[K-Means & K-Means++|K-Means++]]** for smarter, spaced-out starting centroids. |