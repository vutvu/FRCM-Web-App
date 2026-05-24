import streamlit as st
import pandas as pd
import joblib

# =========================================
# 页面配置
# =========================================

st.set_page_config(
    page_title="FRCM Bond Strength Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# SCI高级CSS
# =========================================

st.markdown("""
<style>

/* ========= 全局 ========= */

html, body, [class*="css"] {
    font-family: "Times New Roman", serif;
    background-color: #F5F7FA;
    color: #111111;
}

/* 页面宽度 */

.main .block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* ========= 标题 ========= */

.title {
    font-size: 46px;
    font-weight: 700;
    color: #0D1B2A;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 22px;
    color: #5B6770;
    margin-bottom: 35px;
    line-height: 1.5;
}

/* ========= 卡片 ========= */

.card {
    background-color: white;
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #DCE3EA;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

/* ========= 输入标题 ========= */

.section-title {
    font-size: 32px;
    font-weight: 700;
    color: #102A43;
    margin-bottom: 25px;
}

/* ========= Label ========= */

label {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #243B53 !important;
}

/* ========= 输入框 ========= */

.stNumberInput input {
    font-size: 20px !important;
    height: 58px !important;
    border-radius: 12px !important;
    border: 1px solid #BCCCDC !important;
    background-color: #FCFCFD !important;
}

/* ========= 按钮 ========= */

.stButton > button {
    width: 100%;
    height: 68px;
    font-size: 24px;
    font-weight: 700;
    border-radius: 14px;
    background-color: #102A43;
    color: white;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #243B53;
    transform: translateY(-2px);
}

/* ========= 结果 ========= */

.result-card {
    background: linear-gradient(
        135deg,
        #102A43 0%,
        #243B53 100%
    );
    padding: 38px;
    border-radius: 18px;
    color: white;
    margin-top: 25px;
    text-align: center;
}

.result-title {
    font-size: 24px;
    margin-bottom: 15px;
    opacity: 0.9;
}

.result-value {
    font-size: 56px;
    font-weight: 800;
    letter-spacing: 1px;
}

/* ========= Footer ========= */

.footer {
    margin-top: 40px;
    text-align: center;
    color: #7B8794;
    font-size: 18px;
}

/* ========= 缩放适配 ========= */

@media screen and (max-width: 1200px) {

    .title {
        font-size: 42px;
    }

    p {
        font-size: 28px !important;
    }

    label {
        font-size: 24px !important;
    }

    .stNumberInput input {
        font-size: 24px !important;
        height: 58px !important;
    }

    .result-value {
        font-size: 50px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================================
# 模型加载
# =========================================

@st.cache_resource
def load_model():

    model = joblib.load("stacking_optimized_model.pkl")
    feature_names = joblib.load("stacking_feature_names.pkl")

    return model, feature_names

try:

    model, feature_names = load_model()
    model_ok = True

except Exception as e:

    model_ok = False
    st.error(f"Model loading failed: {e}")

# =========================================
# Header
# =========================================

st.markdown("""
<div class="title">
FRCM–Concrete Interface Bond Strength Prediction
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Machine learning-based prediction platform using
Stacking ensemble learning framework
</div>
""", unsafe_allow_html=True)

# =========================================
# 输入区域
# =========================================

st.markdown("""
<div class="card">
<div class="section-title">
Input Parameters
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:

    bc = st.number_input(
        "Concrete width bc (mm)",
        90.0, 160.0, 120.0
    )

    fck = st.number_input(
        "Concrete compressive strength fck (MPa)",
        10.0, 60.0, 30.0
    )

    tf = st.number_input(
        "Matrix thickness tf (mm)",
        0.0, 60.0, 10.0
    )

    bf = st.number_input(
        "Bond width bf (mm)",
        30.0, 160.0, 100.0
    )

    Lf = st.number_input(
        "Bond length Lf (mm)",
        40.0, 710.0, 200.0
    )

    ff = st.number_input(
        "Fiber tensile strength ff (MPa)",
        280.0, 5900.0, 2000.0
    )

with col2:

    Ef = st.number_input(
        "Fiber elastic modulus Ef (GPa)",
        15.0, 290.0, 100.0
    )

    fc = st.number_input(
        "Matrix compressive strength fc (MPa)",
        10.0, 80.0, 30.0
    )

    ft = st.number_input(
        "Matrix tensile strength ft (MPa)",
        0.0, 8.0, 2.0
    )

    etu = st.number_input(
        "Ultimate tensile strain εtu (‰)",
        0.0, 75.0, 10.0
    )

    Af = st.number_input(
        "Fiber cross-sectional area Af (mm²)",
        0.0, 150.0, 50.0
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# 按钮
# =========================================

predict = st.button("Predict Bond Strength")

# =========================================
# 预测
# =========================================

if predict and model_ok:

    try:

        input_df = pd.DataFrame([[
            bc, fck, tf, bf, Lf,
            ff, Ef, fc, ft, etu, Af
        ]], columns=[
            'bc', 'fck', 'tf', 'bf', 'Lf',
            'ff', 'Ef', 'fc', 'ft',
            'εtu(‰)', 'Af'
        ])

        input_df = input_df[feature_names]

        pred = float(model.predict(input_df)[0])

        pred = max(pred, 0)

        st.markdown(f"""
        <div class="result-card">

            <div class="result-title">
            Predicted Ultimate Bond Capacity
            </div>

            <div class="result-value">
            {pred:.2f} kN
            </div>

        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.error(f"Prediction failed: {e}")

# =========================================
# Footer
# =========================================

st.markdown("""
<div class="footer">
FRCM Interface Prediction System · SCI Visualization Style
</div>
""", unsafe_allow_html=True)