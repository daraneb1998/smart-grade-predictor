# 🎓 Grade Intelligence Dashboard

This is a Streamlit-based web application that predicts a student's final grade using a trained logistic regression model. The app allows users to input performance metrics and receive a grade prediction along with confidence levels, feature influence, and actionable suggestions.

---

## 🚀 Features

- **Interactive Grade Prediction**  
  Input scores for exams, projects, assignments, and participation to get a predicted grade (A–F).

- **Model Selection**  
  Choose between two strategies for handling imbalanced data:
  - `Class Weight Balanced`
  - `Resampling`

- **Confidence Score**  
  View the model's confidence in its prediction.

- **Feature Importance Visualization**  
  See which features most influence the predicted grade.

- **AI-Powered Suggestions**  
  Get personalized advice based on the predicted grade.

---

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

You can install them using:

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 📁 File Structure

```
├── models/
│   ├── logistic_regression_model_with_resampling.joblib
│   ├── logistic_regression_model_with_cw.joblib
│   ├── feature_scaler_with_resampling.joblib
│   ├── feature_scaler_with_cw.joblib
│   └── selected_features.joblib
├── app.py
└── README.md
```

---

## 🧠 How It Works

1. The user selects a model strategy from the sidebar.
2. The app loads the corresponding trained model and scaler.
3. User inputs performance scores.
4. The model predicts the grade and shows:
   - Confidence level
   - Feature influence
   - Suggestions for improvement

Grades are treated as **ordinal categories**:  
`F < D < C < B < A`

---

## ▶️ Run the App

To launch the app locally, run:

```bash
streamlit run app.py
```

---

## 📌 Notes

- Models are pre-trained and stored in the `models/` directory.
- The app uses `OrdinalEncoder` to handle grade labels.
- Feature scaling is applied before prediction to ensure model accuracy.

---