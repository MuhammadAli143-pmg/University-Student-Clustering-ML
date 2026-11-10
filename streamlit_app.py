import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="University Student Cluster Predictor", page_icon="🎓")
st.title("University Student Cluster Predictor")
st.write("Enter academic information to predict the student's K-Means cluster.")

model = joblib.load("student_cluster_classifier.joblib")
with open("schema.json", "r") as file:
    schema = json.load(file)

with st.form("student_form"):
    age = st.number_input("Age", min_value=16, max_value=60, value=21)
    semester = st.number_input("Semester", min_value=1, max_value=16, value=6)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.25, step=0.01)
    marks = st.number_input("Marks", min_value=0.0, max_value=100.0, value=82.0, step=0.5)
    attendance_pct = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=88.0, step=0.5)
    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    payload = pd.DataFrame([{
        "age": age, "semester": semester, "cgpa": cgpa,
        "marks": marks, "attendance_pct": attendance_pct
    }])
    try:
        for col in schema["columns"]:
            if col not in payload.columns:
                raise ValueError(f"Missing required feature: {col}")
            if not pd.api.types.is_numeric_dtype(payload[col]):
                raise TypeError(f"{col} must be numeric.")
            limits = schema["domain"][col]
            if not payload[col].between(limits["min"], limits["max"]).all():
                raise ValueError(f"{col} must be between {limits['min']} and {limits['max']}.")
        prediction = int(model.predict(payload[schema["columns"]])[0])
        names = {
            0: "Cluster 0",
            1: "Cluster 1",
            2: "Cluster 2"
        }
        st.success(f"Predicted student segment: {names.get(prediction, str(prediction))}")
        st.write("This is an academic segmentation result, not a formal student assessment.")
    except (ValueError, TypeError) as error:
        st.error(str(error))
