import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

st.markdown("""
<style>
    div.stButton > button {
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.65rem 2rem;
        border-radius: 50px;
        width: 100%;
        margin-top: 0.5rem;
        transition: transform 0.1s;
    }
    div.stButton > button:hover { transform: scale(1.03); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
@st.cache_resource
def load_model():
    m = MLP(input_dim=18)
    m.load_state_dict(torch.load("ANN/models/heart_mlp.pth", map_location="cpu"))
    m.eval()
    return m

@st.cache_resource
def load_scaler():
    return joblib.load("ANN/models/scaler.pkl")

@st.cache_resource
def load_training_data():
    return joblib.load("ANN/plots/training_data.pkl")

model  = load_model()
scaler = load_scaler()
data   = load_training_data()

# =============================================================================
# HEADER
# =============================================================================
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("<p style='color:#ad1457; font-size:1.1rem;'>An MLP-based clinical decision support tool trained on the UCI Heart Disease dataset.</p>", unsafe_allow_html=True)
st.divider()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["Predict", "Model Details"])

# =============================================================================
# TAB 1: PREDICT
# =============================================================================
with tab1:
    st.subheader("Patient Data")
    st.markdown("Adjust the values below to match the patient's clinical profile.")

    col1, col2 = st.columns(2)

    with col1:
        age      = st.slider("Age", 20, 80, 50)
        sex      = st.selectbox("Sex", ["Male", "Female"])
        cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                       help="0=Typical Angina, 1=Atypical Angina, 2=Non-anginal, 3=Asymptomatic")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                       format_func=lambda x: "Yes" if x else "No")
        restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                       help="0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy")

    with col2:
        thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang    = st.selectbox("Exercise Induced Angina", [0, 1],
                       format_func=lambda x: "Yes" if x else "No")
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope    = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                       help="0=Upsloping, 1=Flat, 2=Downsloping")
        ca       = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal     = st.selectbox("Thal", [1, 2, 3],
                       help="1=Normal, 2=Fixed Defect, 3=Reversible Defect")

    def preprocess(age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal):
        sex_val = 1 if sex == "Male" else 0
        raw = pd.DataFrame([{
            'age': age, 'sex': sex_val, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'exang': exang,
            'oldpeak': oldpeak, 'ca': ca,
            'cp': cp, 'restecg': restecg, 'slope': slope,
            'thal': thal, 'thalach': thalach
        }])
        raw = pd.get_dummies(raw, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
        raw = raw.reindex(columns=scaler.feature_names_in_, fill_value=0)
        return scaler.transform(raw)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict", type="primary"):
        X_input = preprocess(age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal)
        tensor = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            prob = model(tensor).item()

        st.divider()
        st.subheader("Result")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Disease Probability", f"{prob:.1%}")
        with col_b:
            risk_label = "High Risk" if prob >= 0.7 else "Moderate Risk" if prob >= 0.5 else "Low Risk"
            st.metric("Risk Level", risk_label)
        with col_c:
            st.metric("Confidence", f"{max(prob, 1-prob):.1%}")

        st.progress(prob)

        if prob >= 0.5:
            st.markdown("**Prediction:** <span style='color:#e91e8c; font-weight:700; font-size:1.2rem'>⚠️ Disease Likely</span>", unsafe_allow_html=True)
            st.warning("This patient shows elevated risk indicators. Clinical follow-up recommended.")
        else:
            st.markdown("**Prediction:** <span style='color:#2e7d32; font-weight:700; font-size:1.2rem'>✅ Low Risk</span>", unsafe_allow_html=True)
            st.success("This patient shows low risk indicators based on the provided data.")

        st.caption("⚠️ This tool is for educational purposes only and is not a substitute for medical advice.")

# =============================================================================
# TAB 2: MODEL DETAILS
# =============================================================================
with tab2:
    st.subheader("Model Details")

    with st.expander("🧠 Architecture", expanded=True):
        st.markdown("""
        | Layer | Type | Details |
        |---|---|---|
        | Input | — | 18 features |
        | Hidden 1 | Linear → ReLU → Dropout | 128 neurons, dropout 0.4 |
        | Hidden 2 | Linear → ReLU → Dropout | 64 neurons, dropout 0.3 |
        | Hidden 3 | Linear → ReLU | 32 neurons |
        | Output | Linear → Sigmoid | 1 neuron (binary probability) |

        **Optimizer:** Adam · **Learning Rate:** 0.0005 · **Loss:** Binary Cross-Entropy · **Early Stopping Patience:** 15
        """)

    with st.expander("📁 Dataset", expanded=False):
        st.markdown("""
        **Source:** UCI Heart Disease Dataset (Cleveland subset) via
        <span style='color:#e91e8c; font-weight:700;'>ucimlrepo</span>

        | Property | Value |
        |---|---|
        | Total samples | 303 |
        | Features (raw) | 13 |
        | Features (after encoding) | 18 |
        | Train / Test split | 80% / 20% (stratified) |
        | Target | Binary — 0 = No Disease, 1 = Disease |

        Categorical features `cp`, `restecg`, `slope`, and `thal` were one-hot encoded.
        Missing values imputed with median (continuous) and mode (categorical).
        Features scaled with `StandardScaler` fit on training data only.
        """, unsafe_allow_html=True)

    with st.expander("📈 Performance Metrics", expanded=True):
        y_pred      = (np.array(data['y_pred_prob']) >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_pred_prob'])
        roc_auc_val = auc(fpr, tpr)
        acc         = np.mean(np.array(y_pred) == np.array(data['y_test']))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",    f"{acc:.1%}")
        c2.metric("AUC-ROC",     f"{roc_auc_val:.3f}")
        c3.metric("CV Mean AUC", "0.896")
        c4.metric("CV Std",      "±0.030")

    st.divider()

    plot_tab1, plot_tab2, plot_tab3 = st.tabs(["📉 Training Curves", "🔲 Confusion Matrix", "📐 ROC Curve"])

    with plot_tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#fff0f5')
        ax1.plot(data['train_losses'], color='#e91e8c', label='Train Loss')
        ax1.plot(data['val_losses'],   color='#880e4f', linestyle='--', label='Val Loss')
        ax1.set_title("Loss over Epochs", color='#880e4f', fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_facecolor('#fce4ec')
        ax1.legend()
        ax2.plot(data['train_accs'], color='#e91e8c', label='Train Acc')
        ax2.plot(data['val_accs'],   color='#880e4f', linestyle='--', label='Val Acc')
        ax2.set_title("Accuracy over Epochs", color='#880e4f', fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_facecolor('#fce4ec')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Dashed line = validation. Early stopping restored best weights before overfitting.")

    with plot_tab2:
        cm = confusion_matrix(data['y_test'], y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#fff0f5')
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_title("Confusion Matrix", color='#880e4f', fontweight='bold')
        ax.set_ylabel("Actual",    color='#880e4f')
        ax.set_xlabel("Predicted", color='#880e4f')
        plt.tight_layout()
        st.pyplot(fig)

        tn, fp, fn, tp = cm.ravel()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives",  tp)
        c2.metric("True Negatives",  tn)
        c3.metric("False Positives", fp, delta=f"-{fp} misclassified", delta_color="inverse")
        c4.metric("False Negatives", fn, delta=f"-{fn} missed cases",  delta_color="inverse")

    with plot_tab3:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#fff0f5')
        ax.set_facecolor('#fce4ec')
        ax.plot(fpr, tpr, color='#e91e8c', lw=2, label=f'AUC = {roc_auc_val:.3f}')
        ax.plot([0, 1], [0, 1], color='#f48fb1', linestyle='--', lw=1, label='Random')
        ax.set_xlabel("False Positive Rate", color='#880e4f')
        ax.set_ylabel("True Positive Rate",  color='#880e4f')
        ax.set_title("ROC Curve", color='#880e4f', fontweight='bold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("AUC > 0.9 indicates strong discriminative ability between disease and no disease.")