"""
Interactive Linear Algebra Color Lab: Transformations, Invertibility & Encryption
==================================================================================
A modern, self-explanatory local Streamlit UI demonstrating:
1. RGB Vector Field Transformations using vectorized NumPy matrix multiplication.
2. Interactive 3x3 Matrix Grid with dynamic Eigenvalues, Determinants & Row-Sums.
3. Cinematic Color Grading presets & RGB Channel Histograms.
4. Lossless Color Encryption & Recovery via matrix inversion (K^{-1}).
5. Singular Matrix Failure & Information Collapse (Lab Sepia breakdown).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# Set page config with wide layout and modern title
st.set_page_config(
    page_title="Linear Algebra Color Lab",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, high-contrast dark/light mode compatibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
        padding-left: 16px;
        padding-right: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. CORE VECTORIZED LINEAR ALGEBRA ENGINE
# ==============================================================================

def apply_color_transformation(img: np.ndarray, matrix: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Applies 3x3 matrix M to (H, W, 3) image tensor using vectorized BLAS tensor multiplication.
    Identity: p' = M @ p <=> p'^T = p^T @ M^T.
    """
    img_float = img.astype(np.float64)
    # Tensor contraction along color dimension: img @ M^T
    transformed = np.matmul(img_float, matrix.T)
    if clip:
        return np.clip(transformed, 0.0, 255.0).astype(np.uint8)
    return transformed


def analyze_matrix(matrix: np.ndarray) -> Dict[str, Any]:
    """Computes determinant, condition number, eigenvalues, eigenvectors, and row sums."""
    det_val = float(np.linalg.det(matrix))
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    try:
        cond_num = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        cond_num = float("inf")
    row_sums = np.sum(matrix, axis=1)
    
    return {
        "det": det_val,
        "cond": cond_num,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "row_sums": row_sums,
        "is_invertible": abs(det_val) > 1e-4 and cond_num < 5000.0
    }


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float]:
    """Calculates Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)."""
    orig_f = original.astype(np.float64)
    rec_f = reconstructed.astype(np.float64)
    mse = float(np.mean((orig_f - rec_f) ** 2))
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    return mse, psnr



# ==============================================================================
# 2. DEFAULT BENCHMARK IMAGE CREATION
# ==============================================================================

@st.cache_data
def get_default_benchmark_image() -> np.ndarray:
    """Generates a rich, multi-spectrum test image with gradients, florals, and color patches."""
    height, width = 360, 540
    y, x = np.mgrid[0:height, 0:width]
    
    # Sunset sky to emerald-ocean gradient
    r = (255 * (1.0 - y / height) * 0.95 + 180 * (x / width) * 0.4).clip(0, 255)
    g = (160 * (y / height) * 0.85 + 100 * np.sin(x / 45.0) + 40).clip(0, 255)
    b = (240 * (y / height) * 0.9 + 90 * np.cos(y / 40.0)).clip(0, 255)
    
    img = np.dstack([r, g, b]).astype(np.uint8)
    
    # Color calibration test swatches on top
    swatches = [
        [225, 160, 130],  # Skin tone
        [240,  50,  50],  # Pure Red
        [ 40, 200,  80],  # Vivid Green
        [ 40, 110, 240],  # Rich Blue
        [250, 215,  40],  # Amber Gold
        [ 40, 210, 220],  # Cyan / Teal
        [220, 220, 220],  # Neutral Light Gray
        [128, 128, 128],  # Neutral Mid Gray
        [ 50,  50,  50],  # Dark Gray
    ]
    patch_size = 35
    for idx, col in enumerate(swatches):
        px = 15 + idx * 55
        py = 15
        if px + patch_size < width:
            img[py:py + patch_size, px:px + patch_size] = col
            
    return img


# ==============================================================================
# 3. PRESET MATRICES DICTIONARY
# ==============================================================================

PRESETS = {
    "Teal & Orange (Blockbuster Film Look)": np.array([
        [1.18, -0.08, -0.10],
        [-0.05, 1.05,  0.00],
        [-0.15,  0.10, 1.05]
    ], dtype=np.float64),
    
    "Warm Vintage Film (Cross-Processing)": np.array([
        [1.08,  0.06, -0.14],
        [0.04,  0.98, -0.02],
        [-0.12, 0.08,  1.04]
    ], dtype=np.float64),
    
    "Grayscale Average (Rank-1 Projection)": np.array([
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3]
    ], dtype=np.float64),
    
    "Red Channel Isolator": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64),
    
    "Green Channel Isolator": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64),
    
    "Blue Channel Isolator": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64),

    "Identity Matrix (Pass-Through)": np.eye(3, dtype=np.float64)
}

DEFAULT_KEY_MATRIX = np.array([
    [ 0.65,  0.85, -0.45],
    [-0.50,  0.75,  0.60],
    [ 0.80, -0.30,  0.70]
], dtype=np.float64)

SEPIA_MATRIX = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
], dtype=np.float64)


# ==============================================================================
# 4. STREAMLIT UI LAYOUT & INTERACTIONS
# ==============================================================================

# Header Section
st.markdown('<div class="main-header">Interactive Linear Algebra Color Lab</div>', unsafe_allow_html=True)
st.caption("Vectorized 3D Tensor Transformations, Spectral Analysis, and Reversible Matrix Cryptography")

with st.expander("📘 Mathematical Foundation: RGB Images as 3D Vector Fields", expanded=False):
    st.markdown(r"""
    Every colored pixel at coordinate $(x, y)$ in an image is a 3-dimensional vector in the RGB Euclidean space:
    $$\mathbf{p} = \begin{bmatrix} R \\ G \\ B \end{bmatrix} \in \mathbb{R}^3$$
    
    ### 1. Linear Transformation
    Any $3 \times 3$ matrix $M$ defines a linear map $T: \mathbb{R}^3 \to \mathbb{R}^3$:
    $$\mathbf{p}' = M \mathbf{p} = \begin{bmatrix} M_{11} & M_{12} & M_{13} \\ M_{21} & M_{22} & M_{23} \\ M_{31} & M_{32} & M_{33} \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$
    
    - **Vectorization Speedup:** Instead of running $O(H \times W)$ nested loops, we contract the entire $(H, W, 3)$ tensor using:
      $$I' = I \cdot M^T$$
      This processes millions of color channels simultaneously via BLAS/SIMD operations.
    
    ### 2. Invertibility & Lossless Recovery
    A transformation is completely reversible if and only if $\det(M) \neq 0$:
    $$\mathbf{p} = M^{-1} \mathbf{p}'$$
    
    ### 3. Numerical Stability & Condition Number
    The **condition number** $\kappa(M) = \|M\|_2 \cdot \|M^{-1}\|_2 = \frac{\sigma_{\max}}{\sigma_{\min}}$ measures sensitivity to perturbation.
    - If $\kappa(M) \approx 1$, numerical inversion is stable and noise-free.
    - If $\kappa(M) \gg 10^3$ (e.g. Sepia where $\det \approx 10^{-7}$), rows are nearly co-linear and inverting $M$ blows up into catastrophic noise.
    """)

# Sidebar: Image Selection
st.sidebar.header("📁 Image Configuration")
uploaded_file = st.sidebar.file_uploader("Upload an Image (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    current_img = np.array(img_pil)
    source_label = f"Uploaded: {uploaded_file.name}"
else:
    current_img = get_default_benchmark_image()
    source_label = "Default Multi-Spectrum Benchmark"

h, w, c = current_img.shape
st.sidebar.info(f"**Image Dimensions:** {w} × {h} px\n\n**Total Pixels:** {h*w:,}\n\n**Channels:** {c} (RGB)")
st.sidebar.markdown(f"**Current Source:** `{source_label}`")

# Tabs for Main Modules
tab_grading, tab_crypto, tab_singular = st.tabs([
    "🎬 1. Cinematic Color Grading & Matrix Grid",
    "🔐 2. Reversible Color Encryption & Recovery",
    "💥 3. Singular Matrix Failure Case (Sepia)"
])


# ==============================================================================
# TAB 1: CINEMATIC COLOR GRADING & INTERACTIVE MATRIX GRID
# ==============================================================================
with tab_grading:
    st.subheader("1. Cinematic Color Grading & Custom 3x3 Matrix Grid")
    st.markdown("Select a preset or customize the 3x3 transformation matrix entries in real time to observe eigenvalue and chromatic changes.")
    
    col_preset, col_reset = st.columns([3, 1])
    with col_preset:
        selected_preset = st.selectbox("Choose a Filter Preset:", list(PRESETS.keys()), index=0)
    
    # Session state for interactive matrix grid
    if "current_matrix" not in st.session_state or st.session_state.get("last_preset") != selected_preset:
        st.session_state.current_matrix = PRESETS[selected_preset].copy()
        st.session_state.last_preset = selected_preset
        
    with col_reset:
        st.write("")
        st.write("")
        if st.button("Reset Matrix to Preset"):
            st.session_state.current_matrix = PRESETS[selected_preset].copy()
            st.rerun()

    # Interactive Matrix Editing Grid
    st.markdown("##### Interactive $3 \\times 3$ Transformation Matrix $M$:")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    m_val = st.session_state.current_matrix.copy()
    
    with col_m1:
        st.caption("**Red Output Channel**")
        m_val[0, 0] = st.number_input("M[0,0] (R → R)", value=float(m_val[0, 0]), step=0.05, format="%.2f", key="m00")
        m_val[0, 1] = st.number_input("M[0,1] (G → R)", value=float(m_val[0, 1]), step=0.05, format="%.2f", key="m01")
        m_val[0, 2] = st.number_input("M[0,2] (B → R)", value=float(m_val[0, 2]), step=0.05, format="%.2f", key="m02")
        
    with col_m2:
        st.caption("**Green Output Channel**")
        m_val[1, 0] = st.number_input("M[1,0] (R → G)", value=float(m_val[1, 0]), step=0.05, format="%.2f", key="m10")
        m_val[1, 1] = st.number_input("M[1,1] (G → G)", value=float(m_val[1, 1]), step=0.05, format="%.2f", key="m11")
        m_val[1, 2] = st.number_input("M[1,2] (B → G)", value=float(m_val[1, 2]), step=0.05, format="%.2f", key="m12")
        
    with col_m3:
        st.caption("**Blue Output Channel**")
        m_val[2, 0] = st.number_input("M[2,0] (R → B)", value=float(m_val[2, 0]), step=0.05, format="%.2f", key="m20")
        m_val[2, 1] = st.number_input("M[2,1] (G → B)", value=float(m_val[2, 1]), step=0.05, format="%.2f", key="m21")
        m_val[2, 2] = st.number_input("M[2,2] (B → B)", value=float(m_val[2, 2]), step=0.05, format="%.2f", key="m22")

    st.session_state.current_matrix = m_val
    matrix_props = analyze_matrix(m_val)
    
    # Linear Algebra Diagnostics Card
    st.markdown("---")
    st.markdown("##### 🔬 Spectral Properties & Exposure Indicators:")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    
    with d_col1:
        st.metric("Determinant det(M)", f"{matrix_props['det']:.4f}")
    with d_col2:
        st.metric("Condition Number κ(M)", f"{matrix_props['cond']:.2f}" if matrix_props['cond'] < 1e5 else "∞ (Singular)")
    with d_col3:
        eig_str = ", ".join([f"{abs(e):.2f}" for e in matrix_props['eigenvalues']])
        st.metric("Eigenvalues |λ|", eig_str)
    with d_col4:
        r_sums = matrix_props['row_sums']
        avg_r_sum = float(np.mean(r_sums))
        if abs(avg_r_sum - 1.0) < 0.05:
            st.success(f"Row Sums ≈ 1.00 (Balanced Luminance)")
        elif avg_r_sum > 1.05:
            st.warning(f"Row Sums > 1.0 ({avg_r_sum:.2f}) - Highlight Clipping Risk")
        else:
            st.info(f"Row Sums < 1.0 ({avg_r_sum:.2f}) - Darkens Image")

    # Apply Transformation
    graded_img = apply_color_transformation(current_img, m_val, clip=True)

    # Visual Display: Side-by-Side Images
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(current_img, caption="Original RGB Image", width='stretch')
    with img_col2:
        st.image(graded_img, caption="Linearly Transformed Output Image (I' = I · M^T)", width='stretch')




# ==============================================================================
# TAB 2: REVERSIBLE COLOR ENCRYPTION & DECRYPTION
# ==============================================================================
with tab_crypto:
    st.subheader("2. Reversible Color Encryption & Information Recovery")
    st.markdown(r"""
    A color transformation acts as a visual cryptographic cipher if $K$ is invertible. 
    Because $\det(K) \neq 0$, the transformation is an **isomorphism** from $\mathbb{R}^3 \to \mathbb{R}^3$, 
    guaranteeing that every original color vector can be uniquely reconstructed by applying $K^{-1}$.
    """)

    key_mat = DEFAULT_KEY_MATRIX.copy()
    key_props = analyze_matrix(key_mat)
    
    st.markdown("##### Invertible Key Matrix $K$:")
    st.latex(r"""
    K = \begin{bmatrix} 
    0.65 & 0.85 & -0.45 \\ 
    -0.50 & 0.75 & 0.60 \\ 
    0.80 & -0.30 & 0.70 
    \end{bmatrix}, \quad
    \det(K) = """ + f"{key_props['det']:.4f}" + r""", \quad \kappa(K) = """ + f"{key_props['cond']:.2f}"
    )

    # Step 1: Scramble in full floating-point precision
    # Intermediate cipher tensor is in R^(H x W x 3)
    cipher_tensor = apply_color_transformation(current_img, key_mat, clip=False)
    # Displayable version of the scrambled noise
    display_scramble = np.clip(cipher_tensor, 0.0, 255.0).astype(np.uint8)

    # Step 2: Invert matrix K and recover
    K_inv = np.linalg.inv(key_mat)
    decrypted_float = apply_color_transformation(cipher_tensor, K_inv, clip=False)
    decrypted_img = np.clip(np.round(decrypted_float), 0.0, 255.0).astype(np.uint8)

    # Compute error metrics
    rec_mse, rec_psnr = compute_metrics(current_img, decrypted_img)

    # Display 3-Stage Pipeline
    c_img1, c_img2, c_img3 = st.columns(3)
    with c_img1:
        st.image(current_img, caption="1. Original Plaintext P", width='stretch')
    with c_img2:
        st.image(display_scramble, caption="2. Scrambled Cipher P' = K @ P", width='stretch')
    with c_img3:
        st.image(decrypted_img, caption="3. Recovered Image P_rec = K^(-1) @ P'", width='stretch')

    # Verification Banner
    st.markdown("---")
    v_col1, v_col2, v_col3 = st.columns([1, 1, 2])
    with v_col1:
        st.metric("Reconstruction MSE", f"{rec_mse:.4e}")
    with v_col2:
        st.metric("PSNR (Reconstruction Quality)", "∞ dB" if np.isinf(rec_psnr) else f"{rec_psnr:.2f} dB")
    with v_col3:
        st.success("✅ **Mathematical Theorem Confirmed:**\n\nSince $\\det(K) \\neq 0$ and the matrix is well-conditioned ($\kappa = 1.21$), information is **100% preserved** and fully recoverable with zero distortion.")

    with st.expander("🔍 Show Computed Inverse Matrix K^(-1)"):
        st.write("Computed Inverse Matrix $K^{-1}$ using `np.linalg.inv`:")
        st.write(K_inv)


# ==============================================================================
# TAB 3: SINGULAR MATRIX FAILURE CASE (SEPIA)
# ==============================================================================
with tab_singular:
    st.subheader("3. The Singular / Ill-Conditioned Failure Case (Sepia Matrix)")
    st.markdown(r"""
    The standard photo filter Sepia matrix is given by:
    $$M_{\text{sepia}} = \begin{bmatrix} 0.393 & 0.769 & 0.189 \\ 0.349 & 0.686 & 0.168 \\ 0.272 & 0.534 & 0.131 \end{bmatrix}$$
    """)

    sepia_props = analyze_matrix(SEPIA_MATRIX)

    st.markdown("##### Linear Dependency & Spectral Analysis:")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.error(f"**Determinant det(M):** `{sepia_props['det']:.8f}`\n\n(Critically near 0!)")
    with s_col2:
        st.error(f"**Condition Number κ(M):** `{sepia_props['cond']:,.2f}`\n\n(Extreme instability)")
    with s_col3:
        st.warning(f"**Row Dependency:**\n\nRow 2 ≈ 0.89 × Row 1\n\nRow 3 ≈ 0.69 × Row 1")

    # Apply Sepia Filter (Standard 8-bit quantized image pipeline)
    sepia_filtered = apply_color_transformation(current_img, SEPIA_MATRIX, clip=True)

    # Attempt Inversion
    sepia_inv = np.linalg.inv(SEPIA_MATRIX)
    failed_recovery = apply_color_transformation(sepia_filtered, sepia_inv, clip=True)

    # Metrics
    fail_mse, fail_psnr = compute_metrics(current_img, failed_recovery)

    # Visual Comparison
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.image(current_img, caption="1. Original Image", width='stretch')
    with f_col2:
        st.image(sepia_filtered, caption="2. Sepia Filtered (Lab PDF)", width='stretch')
    with f_col3:
        st.image(failed_recovery, caption="3. Attempted Recovery via inv(Sepia)", width='stretch')

    # Explanation Callout
    st.markdown("---")
    st.metric("Failed Inversion MSE", f"{fail_mse:,.2f}", delta="-High Information Loss", delta_color="inverse")
    
    st.error(r"""
    ### 💥 Why Reconstruction Fails (The Linear Algebra Breakdown):
    1. **Rank Collapse & Dimension Reduction:**  
       The row vectors of $M_{\text{sepia}}$ are almost co-linear. This maps the 3D RGB color cube onto a nearly 1-dimensional line. The true rank is effectively 1, destroying 2 dimensions of color information.
    2. **Catastrophic Sensitivity to Inversion:**  
       Because $\det(M) \approx 1.21 \times 10^{-7}$, computing $M^{-1}$ involves dividing by an infinitesimally small number, yielding elements exceeding $\pm 10^4$.
    3. **Amplification of 8-Bit Quantization Noise:**  
       When pixel values are clipped and rounded to 8-bit integers $[0, 255]$, that minuscule rounding error is magnified by $\kappa \approx 16,658$, exploding into uncontrollable visual artifacts and color noise.
    """)
