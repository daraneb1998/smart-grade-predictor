import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OrdinalEncoder

# --- App Configuration ---
st.set_page_config(page_title="🎓 Grade Intelligence Dashboard", layout="wide")

# --- Sidebar: Resampling Option ---
st.sidebar.title("⚙️ Settings")
resampling_option = st.sidebar.radio(
    "Resampling Strategy",
    ("Without Resampling", "With Resampling"),
    index=0
)
st.sidebar.info("Select whether to use the model trained with resampling or without.")

# --- Load models based on selection ---
if resampling_option == "With Resampling":
    model = joblib.load("logistic_regression_model_with_resampling.joblib")
    scaler = joblib.load("feature_scaler_with_resampling.joblib")
    selected_features = joblib.load("selected_features.joblib")
else:
    model = joblib.load("logistic_regression_model_without_resampling.joblib")
    scaler = joblib.load("feature_scaler_without_resampling.joblib")
    selected_features = joblib.load("selected_features.joblib")

# --- OrdinalEncoder for grades: F=0, D=1, C=2, B=3, A=4 ---
grade_order = [["F", "D", "C", "B", "A"]]
grade_encoder = OrdinalEncoder(categories=grade_order)
grade_encoder.fit(np.array(grade_order[0]).reshape(-1,1))

# --- Main Tab: Prediction Interface ---
st.title("🎓 Smart Grade Predictor")
st.write("Enter student performance details below to predict their **final grade**.")

with st.form("grade_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        final_score = st.number_input("Final Exam Score (0-100)", min_value=0.0, max_value=100.0, value=0.0)
        projects_score = st.number_input("Project Score (0-100)", min_value=0.0, max_value=100.0, value=0.0)
    with c2:
        midterm_score = st.number_input("Midterm Score (0-100)", min_value=0.0, max_value=100.0, value=0.0)
        assignments_avg = st.number_input("Assignments Average (0-100)", min_value=0.0, max_value=100.0, value=0.0)
    with c3:
        participation_score = st.number_input("Participation Score (0-10)", min_value=0.0, max_value=10.0, value=0.0)
    
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    try:
        # Prepare input
        X_input = np.array([[final_score, projects_score, midterm_score, assignments_avg, participation_score]])
        X_input_scaled = scaler.transform(X_input)
        
        # Predict grade
        y_pred_encoded = model.predict(X_input_scaled)
        predicted_grade = grade_encoder.inverse_transform(y_pred_encoded.reshape(-1,1))[0][0]
        
        # Predict confidence
        y_proba = model.predict_proba(X_input_scaled)[0]
        pred_index = int(y_pred_encoded[0])
        confidence = y_proba[pred_index] * 100

        st.success(f"🎯 **Predicted Grade:** {predicted_grade} ({confidence:.2f}% confidence)")

        # Feature importance
        st.subheader("Feature Influence")
        feature_importance = abs(model.coef_).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title("Feature Influence on Grade Prediction")
        st.pyplot(fig)

        # AI Insights
        st.subheader("📘 AI Insights")
        st.markdown("""
            - **Final Exam** and **Projects** tend to have the strongest impact on grade outcomes.  
            - Balanced performance across **Assignments** and **Participation** improves consistency.  
            - Lower midterm scores can be compensated by high final or project performance.
        """)

    except Exception as e:
        st.error(f"❌ Error: {e}")
