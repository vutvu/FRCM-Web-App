import streamlit as st
import pandas as pd
import joblib

# 加载模型
model = joblib.load("stacking_optimized_model.pkl")
feature_names = joblib.load("stacking_feature_names.pkl")

st.title("FRCM Bond Strength Prediction")

# 输入参数
bc = st.number_input("bc", 90.0, 160.0)
fck = st.number_input("fck", 10.0, 60.0)
tf = st.number_input("tf", 0.0, 60.0)
bf = st.number_input("bf", 30.0, 160.0)
Lf = st.number_input("Lf", 40.0, 710.0)
ff = st.number_input("ff", 280.0, 5900.0)
Ef = st.number_input("Ef", 15.0, 290.0)
fc = st.number_input("fc", 10.0, 80.0)
ft = st.number_input("ft", 0.0, 8.0)
etu = st.number_input("εtu", 0.0, 75.0)
Af = st.number_input("Af", 0.0, 150.0)

if st.button("Predict"):

    data = pd.DataFrame([[
        bc, fck, tf, bf, Lf,
        ff, Ef, fc, ft, etu, Af
    ]], columns=[
        'bc', 'fck', 'tf', 'bf', 'Lf',
        'ff', 'Ef', 'fc', 'ft', 'εtu(‰)', 'Af'
    ])

    data = data[feature_names]

    pred = model.predict(data)[0]

    st.success(f"Prediction Result: {pred:.2f} kN")