import streamlit as st
from utils.theme import apply_theme

st.set_page_config(page_title="Neural Network Explorer", page_icon="🧠", layout="wide")

apply_theme(
    primary    = "#1e88e5",
    secondary  = "#1565c0",
    background = "#f0f7ff",
    text       = "#0d47a1"
)

st.markdown("""
<style>
    div.stButton > button {
        font-size: 1.1rem;
        font-weight: 700;
        padding: 1.2rem 2rem;
        border-radius: 16px;
        width: 100%;
        margin-top: 0.5rem;
        transition: transform 0.15s;
    }
    div.stButton > button:hover { transform: scale(1.03); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.title("🧠 Neural Network Explorer")
st.markdown(f"<p style='font-size:1.1rem;'>A collection of neural network models built with PyTorch. Select a model below to explore predictions and training details.</p>", unsafe_allow_html=True)
st.divider()

# =============================================================================
# MODEL CARDS
# =============================================================================
col1, col2, col3 = st.columns(3)

card_style = """
<style>
.model-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 16px rgba(194,24,91,0.10);
    border: 2px solid #f8bbd0;
    height: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
    cursor: pointer;
}
.model-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(194,24,91,0.18);
}
.card-icon   { font-size: 2.8rem; margin-bottom: 0.5rem; }
.card-title  { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.8rem; }
.card-badge  {
    display: inline-block;
    background: #fce4ec;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    margin-bottom: 1rem;
}
.card-desc   { font-size: 0.95rem; color: #555; margin-bottom: 1.2rem; line-height: 1.6; }
.card-stats  { display: flex; gap: 1rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.stat-box {
    background: #fff0f5;
    border-radius: 10px;
    padding: 0.4rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
}
.card-footer { font-size: 0.8rem; color: #aaa; }
.coming-card {
    background: #fafafa;
    border: 2px dashed #f8bbd0;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    color: #ccc;
    height: 100%;
}
.coming-icon  { font-size: 2.5rem; margin-bottom: 0.8rem; }
.coming-title { font-size: 1.2rem; font-weight: 700; color: #ddd; }
div.stButton > button[disabled] {
    background: #fafafa !important;
    border: none !important;
    color: transparent !important;
    cursor: default !important;
    box-shadow: none !important;
}
</style>
"""
st.markdown(card_style, unsafe_allow_html=True)

with col1:
    st.markdown("""
    <div class="model-card">
        <div class="card-icon">🫀</div>
        <div class="card-title">Heart Disease Predictor</div>
        <div class="card-badge">Binary Classification</div>
        <div class="card-desc">
            Predicts the likelihood of heart disease from 13 clinical features
            including age, cholesterol, chest pain type, and ECG results.
        </div>
        <div class="card-stats">
            <div class="stat-box">🧠 MLP</div>
            <div class="stat-box">📊 303 samples</div>
            <div class="stat-box">✅ 89% Accuracy</div>
            <div class="stat-box">📈 AUC 0.946</div>
        </div>
        <div class="card-footer">UCI Heart Disease Dataset · PyTorch</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("Open Heart Disease Model →", key="heart", type="primary"):
        st.switch_page("pages/01_Heart_Disease.py")

with col2:
    st.markdown("""
    <div class="coming-card">
        <div class="coming-icon">🔬</div>
        <div class="coming-title">Coming Soon</div>
        <p style="color:#ccc; font-size:0.9rem; margin-top:0.5rem;">Next model in progress</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="coming-card">
        <div class="coming-icon">🔬</div>
        <div class="coming-title">Coming Soon</div>
        <p style="color:#ccc; font-size:0.9rem; margin-top:0.5rem;">Next model in progress</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Built with PyTorch + Streamlit · Models trained on publicly available datasets.")