import streamlit as st
import pandas as pd
import joblib

# =============================
# 页面设置
# =============================

st.set_page_config(
    page_title="FRCM Bond Prediction",
    layout="wide"
)

# =============================
# 自定义CSS（SCI风格）
# =============================

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: "Times New Roman";
    background-color: #F7F7F7;
}

.main {
    background-color: #F7F7F7;
}

h1 {
    color: #0B0B0B;
    font-size: 38px !important;
    font-weight: 700 !important;
}

.subtitle {
    color: #6A7D7C;
    font-size: 16px;
    margin-bottom: 25px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.result-box {
    background-color: white;
    padding: 25px;
    border-radius: 10px;
    border: 1px solid #D9D9D9;
    margin-top: 20px;
}

.result-text {
    font-size: 30px;
    font-weight: bold;
    color: #8B0000;
}

.small-note {
    color: gray;
    font-size: 13px;
}

.stButton>button {
    background-color: black;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 30px;
    border: none;
}

.stButton>button:hover {
    background-color: #333333;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =============================
# 加载模型
# =============================

@st.cache_resource
def load_model():
    model = joblib.load("stacking_optimized_model.pkl")
    feature_names = joblib.load("stacking_feature_names.pkl")
    return model, feature_names

try:
    model, feature_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading failed: {e}")

# =============================
# 标题
# =============================

st.title("FRCM-Concrete Interface Bond Performance Prediction")

st.markdown("""
<div class='subtitle'>
Stacking ensemble model • MLP + SVR + LightGBM + XGBoost + CatBoost
</div>
""", unsafe_allow_html=True)

# =============================
# 参数范围
# =============================

ranges = {
    'bc': (90, 160),
    'fck': (10, 60),
    'tf': (0, 60),
    'bf': (30, 160),
    'Lf': (40, 710),
    'ff': (280, 5900),
    'Ef': (15, 290),
    'fc': (10, 80),
    'ft': (0, 8),
    'etu': (0, 75),
    'Af': (0, 150)
}

# =============================
# 输入区域
# =============================

st.markdown("## Input Parameters")

col1, col2 = st.columns(2)

with col1:

    bc = st.number_input(
        "Width of concrete block, bc (mm)",
        min_value=90.0,
        max_value=160.0,
        value=120.0
    )

    fck = st.number_input(
        "Substrate concrete compressive strength, fck (MPa)",
        min_value=10.0,
        max_value=60.0,
        value=30.0
    )

    tf = st.number_input(
        "Thickness of matrix, tf (mm)",
        min_value=0.0,
        max_value=60.0,
        value=10.0
    )

    bf = st.number_input(
        "Bonding width, bf (mm)",
        min_value=30.0,
        max_value=160.0,
        value=100.0
    )

    Lf = st.number_input(
        "Bonding length, Lf (mm)",
        min_value=40.0,
        max_value=710.0,
        value=200.0
    )

    ff = st.number_input(
        "Fiber tensile strength, ff (MPa)",
        min_value=280.0,
        max_value=5900.0,
        value=2000.0
    )

with col2:

    Ef = st.number_input(
        "Elastic modulus of fiber, Ef (GPa)",
        min_value=15.0,
        max_value=290.0,
        value=100.0
    )

    fc = st.number_input(
        "Matrix compressive strength, fc (MPa)",
        min_value=10.0,
        max_value=80.0,
        value=30.0
    )

    ft = st.number_input(
        "Matrix tensile strength, ft (MPa)",
        min_value=0.0,
        max_value=8.0,
        value=2.0
    )

    etu = st.number_input(
        "Matrix ultimate tensile strain, εtu (‰)",
        min_value=0.0,
        max_value=75.0,
        value=10.0
    )

    Af = st.number_input(
        "Fiber cross-sectional area, Af (mm²)",
        min_value=0.0,
        max_value=150.0,
        value=50.0
    )

# =============================
# 按钮
# =============================

st.markdown("")

predict_btn = st.button("Predict")

# =============================
# 预测
# =============================

if predict_btn:

    if not model_loaded:
        st.error("Model not loaded.")
    else:

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

            prediction = float(model.predict(input_df)[0])

            prediction = max(prediction, 0)

            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:20px;">
                    Prediction ultimate bearing capacity Pu
                </div>

                <div class="result-text">
                    {prediction:.2f} kN
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =============================
# Footer
# =============================

st.markdown("---")

st.markdown("""
<div class='small-note'>
FRCM Interface Bond Strength Prediction System
</div>
""", unsafe_allow_html=True)