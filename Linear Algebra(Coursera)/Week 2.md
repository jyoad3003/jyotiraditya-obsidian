Here are clear, concise notes from the lecture titled “Application of Changing Basis,” along with intuitive explanations. I’ve included relevant insights from the previous lecture on “Basis, Vector Space, and Linear Independence” only where they help in deepening understanding. The attached image has also been interpreted and integrated into the notes.

---

## 📌 Lecture Notes: Application of Changing Basis

### 1. 🎯 Core Idea

- Changing basis helps us re-express data using new coordinates aligned with the structure of the data (e.g., aligning with trends or key features).
    
- Particularly useful in data science & machine learning to simplify, interpret, and compress data.
    

---

### 2. 📈 Intuition Behind Changing Basis (based on image)

- Imagine a set of 2D data points (pink crosses) that roughly align along a straight line (shown in green in the image).
    
- We define a new basis:
    
    - One vector (let’s call it b₁) along the line (captures how far along the trend each point lies).
        
    - Another orthogonal vector (b₂), perpendicular to b₁ (captures how far each point deviates from the trend = noise).
        

✅ This transformation gives us two new features:

- 📏 How far along the line → signal/direction of variance.
    
- 🧭 How far off the line → noise/error/deviation.
    

---

### 3. 🧠 Why This Basis Change Matters

- Projection of points onto the line (as shown in the image with vertical arrows) lets us:
    
    - Compress the data: Retain only the coordinates along b₁.
        
    - Measure noise: The orthogonal distances (along b₂) show how well a line fits.
        

📌 In statistics: There’s debate whether to use vertical distance or orthogonal projection for this residual — the projection is more geometrically meaningful (and consistent with vector space intuition).

---

### 4. 💡 Key Insight:

- The distance from the line (b₂) represents "noise" — often unwanted, but:
    
    - It’s useful for understanding how good the fit is.
        
    - Smaller noise = better fit of the model (e.g., regression line, PCA axis).
        

---

### 5. ⚙️ Mathematical Structure

- The basis vectors b₁ and b₂ are orthogonal ⇒ dot product can be used for projection.
    
- This aligns with last lecture’s idea: orthonormal basis simplifies computations (no need for full matrix transformations).
    

📎 Recall from earlier:

> A basis is a set of linearly independent vectors that spans a space. Changing basis is re-expressing vectors in a new coordinate system defined by a different basis.

---

### 6. 🤖 In Machine Learning Context

- Think of pixel data of a face (very high-dimensional, unstructured).
    
- The goal: learn a transformation to a new basis representing meaningful features:
    
    - E.g., nose shape, skin hue, eye distance.
        

This is what deep learning models (like neural nets) implicitly do: learn basis vectors (filters/features) that best capture variation in the data and discard irrelevant details (noise).

---

### 7. ✅ Summary: Why Changing Basis Is Useful

- Helps reduce dimensionality (e.g., in PCA).
    
- Reveals structure in data.
    
- Separates signal from noise.
    
- Makes data processing (especially in ML) more efficient.
    

---

## 🖼️ Reference to the Image

- The green line represents the new b₁ basis vector (direction of maximal variance).
    
- Pink points = data.
    
- Green arrows show projection of data onto the new basis.
    
- Orthogonal distance from each point to the line = magnitude along b₂ = noise/error/residual.
    
