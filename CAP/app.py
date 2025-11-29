import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("lung_cancer_model.pkl")  # Keep the original model file
scaler = joblib.load("scaler.pkl")

# Streamlit page setup
st.set_page_config(page_title="Cancer Prediction", page_icon="🧬", layout="wide")
st.title("🧬 Cancer Prediction App")
st.write("Enter patient details below to check cancer risk and view model explanation.")

# Input fields
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        level = st.selectbox("Level", ["Low", "Medium", "High"])
        alcohol = st.number_input("Alcohol use", 0.0, 10.0, 1.0)
        obesity = st.number_input("Obesity", 0.0, 10.0, 1.0)
        smoking = st.number_input("Smoking", 0.0, 10.0, 1.0)
    with col2:
        coughing = st.number_input("Coughing of Blood", 0.0, 10.0, 0.0)
        weight_loss = st.number_input("Weight Loss", 0.0, 10.0, 1.0)
        fatigue = st.number_input("Fatigue", 0.0, 10.0, 1.0)
        chest_pain = st.number_input("Chest Pain", 0.0, 10.0, 1.0)
        short_breath = st.number_input("Shortness of Breath", 0.0, 10.0, 1.0)
        swallowing = st.number_input("Swallowing Difficulty", 0.0, 10.0, 1.0)

    submitted = st.form_submit_button("🔍 Predict Cancer Risk")

if submitted:
    # Encode categorical values
    gender = 0 if gender == "Male" else 1
    level_map = {"Low": 0, "Medium": 1, "High": 2}
    level = level_map[level]

    # Create DataFrame for available inputs
    input_df = pd.DataFrame([[
        age, gender, level, alcohol, obesity, smoking, coughing, weight_loss,
        fatigue, chest_pain, short_breath, swallowing
    ]], columns=[
        'Age', 'Gender', 'Level', 'Alcohol use', 'Obesity', 'Smoking',
        'Coughing of Blood', 'Weight Loss', 'Fatigue', 'Chest Pain',
        'Shortness of Breath', 'Swallowing Difficulty'
    ])

    # --- Ensure same columns as training ---
    feature_columns = joblib.load("feature_columns.pkl")

    # Add any missing columns with default = 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training order
    input_df = input_df[feature_columns]

    # --- Scale and predict ---
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    st.subheader("🔎 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High risk of Cancer (Probability: {proba*100:.2f}%)")
    else:
        st.success(f"✅ Low risk of Cancer (Probability: {proba*100:.2f}%)")

    # --- SHAP Explainability ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    # st.subheader("🧠 Model Explainability")
    # fig, ax = plt.subplots(figsize=(8, 4))
    # shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    # # st.pyplot(fig)

    # # GradCAM-like heatmap
    # mean_abs = np.abs(shap_values).flatten()
    # norm = mean_abs / np.max(mean_abs)
    # fig, ax = plt.subplots(figsize=(10, 1))
    # sns.heatmap([norm], cmap="coolwarm", annot=True, xticklabels=input_df.columns, cbar=False)
    # plt.title("Feature Influence Heatmap")
    # st.pyplot(fig)
# 


# ----------------------------------------------------------------------------------------------------------------#
# streamlit run app.py
