# Viva & Evaluation Notes: Reversible Color Encryption & Cinematic Color Grading

**Topic:** Reversible Color Encryption and Cinematic Color Grading via Linear Transformations  
**Foundational Alignment:** Lab Project 3 (*Matrix multiplication, inversion, and photo filters*)

---

## 1. Mathematical Foundations & Vectorization

### 1.1 Vectorized 3D Tensor Transformations
In standard color representations, an image $I$ is an array in $\mathbb{R}^{H \times W \times 3}$. Each pixel at coordinate $(i, j)$ is represented as an RGB vector:
$$\mathbf{p} = \begin{bmatrix} R \\ G \\ B \end{bmatrix} \in [0, 255]^3$$

The reference PDF implemented linear filtering using nested loops over height $m$ and width $n$:
```python
for i in range(m):
    for j in range(n):
        PixelColor = ImJPG[i, j, :].reshape(3, 1)
        ImJPG_Gray[i, j, :] = np.dot(Matrix, PixelColor).flatten()
```
For an image of resolution $400 \times 600$, this executes $240,000$ iterations in the Python interpreter.

#### Modern Vectorized Implementation
Using tensor algebra, if we treat each pixel vector as a row vector $\mathbf{p}^T = [R, G, B]$, the transformation $\mathbf{p}' = M \mathbf{p}$ translates into:
$$(\mathbf{p}')^T = (M \mathbf{p})^T = \mathbf{p}^T M^T$$
In NumPy, tensor contraction along the channel dimension is written simply as:
```python
transformed = np.matmul(img_float, matrix.T)  # or img_float @ matrix.T
```
This leverages low-level BLAS/SIMD operations to process all $720,000$ channel values concurrently with zero nested loops, delivering over a **$500\times$ speedup**.

---

## 2. Pipeline Part A: Cinematic Color Grading & Spectral Analysis

Cinematic color grading matrices shift chromatic distributions without altering overall scene exposure.

### 2.1 The Matrices
1. **Teal & Orange Grade ($M_{\text{teal}}$):**
   $$M_{\text{teal}} = \begin{bmatrix} 1.18 & -0.08 & -0.10 \\ -0.05 & 1.05 & 0.00 \\ -0.15 & 0.10 & 1.05 \end{bmatrix}$$
   - *Design rationale:* Enhances skin tones by warming red channels and cools shadows by accentuating blue/cyan while suppressing stray reds.
   - $\det(M_{\text{teal}}) \approx 1.2815$, Condition Number $\kappa \approx 1.32$.

2. **Warm Vintage Film ($M_{\text{vintage}}$):**
   $$M_{\text{vintage}} = \begin{bmatrix} 1.08 & 0.06 & -0.14 \\ 0.04 & 0.98 & -0.02 \\ -0.12 & 0.08 & 1.04 \end{bmatrix}$$
   - *Design rationale:* Emulates cross-processed film chemistry (positive film processed in negative developer).
   - $\det(M_{\text{vintage}}) \approx 1.0832$, Condition Number $\kappa \approx 1.34$.

### 2.2 Why Row Sums and Dominant Eigenvalue Near $1.0$ Matter
Consider a neutral gray pixel where $R = G = B = c$, represented by the eigenvector $\mathbf{v}_{\text{gray}} = c \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$:
$$\mathbf{p}' = M \mathbf{v}_{\text{gray}} = c \begin{bmatrix} \sum_{j} M_{1j} \\ \sum_{j} M_{2j} \\ \sum_{j} M_{3j} \end{bmatrix}$$

- If the row sums equal $1.0$, $\mathbf{v}_{\text{gray}}$ is an eigenvector corresponding to eigenvalue $\lambda = 1$:
  $$M \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = 1.0 \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$$
- **Physical Interpretation:** Neutral white, grays, and shadows maintain their luminance level.
- **Consequence of row sum $\gg 1$:** High-key pixels blow out past $255$ and clip to harsh pure white.
- **Consequence of row sum $\ll 1$:** Shadows crash into dark clipped black.
- **Consequence of dominant $|\lambda| \approx 1.0$:** Preserves the dynamic range of the color gamut without saturation collapse.

---

## 3. Pipeline Part B: Reversible Color Encryption vs. Singular Collapse

### 3.1 Reversible Color Encryption via Invertible Key Matrix $K$
A color transformation is lossless if and only if $K$ is non-singular ($\det(K) \neq 0$):
$$K = \begin{bmatrix} 0.65 & 0.85 & -0.45 \\ -0.50 & 0.75 & 0.60 \\ 0.80 & -0.30 & 0.70 \end{bmatrix}$$
- $\det(K) \approx 1.3662 \neq 0$.
- Condition number $\kappa(K) \approx 1.21$ (optimal numerical stability).

#### Encryption:
$$\mathbf{P}_{\text{cipher}} = K \mathbf{P}$$
Visual chromaticity is completely scrambled across channels.

#### Decryption:
Using `np.linalg.inv(K)`:
$$\mathbf{P}_{\text{recovered}} = K^{-1} \mathbf{P}_{\text{cipher}} = K^{-1} (K \mathbf{P}) = I \mathbf{P} = \mathbf{P}$$
- **Reconstruction Metrics:**
  - $\text{MSE} = 0.0000 \times 10^0$
  - $\text{PSNR} = \infty \text{ dB}$ (Exact 100% mathematical recovery)

---

### 3.2 The Singular Failure Case: Lab PDF Sepia Matrix
From page 29 and 37 of Project 3:
$$M_{\text{sepia}} = \begin{bmatrix} 0.393 & 0.769 & 0.189 \\ 0.349 & 0.686 & 0.168 \\ 0.272 & 0.534 & 0.131 \end{bmatrix}$$

#### Why Inversion Fails:
1. **Linear Dependence of Rows:**
   - Notice that Row 2 $\approx 0.89 \times \text{Row 1}$, and Row 3 $\approx 0.69 \times \text{Row 1}$.
   - All three row vectors are nearly co-linear in 3D color space.
2. **Determinant:**
   $$\det(M_{\text{sepia}}) = 1.21 \times 10^{-7} \approx 0$$
3. **Condition Number:**
   $$\kappa(M_{\text{sepia}}) = \frac{\sigma_{\max}}{\sigma_{\min}} \approx 16,658.89$$
4. **Information Projection:**
   The transformation collapses the 3D RGB color space into a 1D line (sepia luminance axis). The rank is effectively $1$ rather than $3$.
5. **Numerical Explosion:**
   Computing $M_{\text{sepia}}^{-1}$ involves dividing by $\det(M_{\text{sepia}})$, resulting in inverse entries exceeding $\pm 10^4$. Applying this inverse amplifies tiny floating-point rounding errors and 8-bit clipping into severe noise:
   - $\text{MSE} \approx 19,137.52$
   - $\text{PSNR} \approx 5.31 \text{ dB}$

---

## 4. Summary Table for Viva Voce

| Matrix Name | $\det(M)$ | Condition Number $\kappa(M)$ | Dominant Eigenvalues $|\lambda_i|$ | Reconstruction MSE | Linear Algebra Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encryption Key $K$** | $1.3662$ | $1.21$ | $1.15, 1.15, 1.03$ | **$0.000$** | **Well-Conditioned, Isomorphic, Lossless Recovery** |
| **Teal & Orange** | $1.2815$ | $1.32$ | $1.27, 1.00, 1.01$ | N/A (Creative) | **Stable Chromatic Rotation, Preserved Exposure** |
| **Warm Vintage** | $1.0832$ | $1.34$ | $1.19, 1.00, 0.91$ | N/A (Creative) | **Balanced Cross-Process, Row-Stochastic** |
| **Lab Sepia Matrix** | $1.21 \times 10^{-7}$ | **$16,658.89$** | $1.21, 0.001, 0.0001$ | **$19,137.52$** | **Near-Singular, Rank Collapse, Irreversible Loss** |

---

## 5. Generated Figures

1. **`figure1_cinematic_grading.png`**: Original image vs Teal & Orange and Warm Vintage filters with RGB channel histograms showing chromatic redistribution.
2. **`figure2_reversible_encryption.png`**: Original $\rightarrow$ Scrambled noise cipher $\rightarrow$ Decrypted recovered image via $K^{-1}$ ($\text{MSE} = 0$).
3. **`figure3_singularity_sepia_failure.png`**: Sepia filtered image $\rightarrow$ Failed reconstruction via $M_{\text{sepia}}^{-1}$ showing severe noise and channel clipping.
