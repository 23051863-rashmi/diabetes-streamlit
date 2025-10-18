# Auto-generated Streamlit app skeleton from notebook: DiabetesPrediction.ipynb
# You MUST edit load_model() and run_inference() below to match logic from your notebook.

import streamlit as st
import time
import os
import numpy
import pandas
import sklearn
import google
import nbformat
import os
import pathlib
import streamlit
import time
import traceback

st.set_page_config(page_title="Converted Streamlit App", layout="wide")
@st.cache_resource(show_spinner=False)
def load_model():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn import svm

    # Load the diabetes dataset
    df = pd.read_csv("diabetes.csv")

    # Separate features and labels
    X = df.drop(columns="Outcome", axis=1)
    Y = df["Outcome"]

    # Split the data (same as your notebook)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Train the SVM classifier
    classifier = svm.SVC(kernel="linear")
    classifier.fit(scaler.transform(X_train), Y_train)

    # Return both model and scaler for later use
    return classifier, scaler


def run_inference(model_and_scaler, input_text):
    import numpy as np
    classifier, scaler = model_and_scaler

    # Expect comma-separated input
    try:
        input_values = [float(x) for x in input_text.split(",")]
        if len(input_values) != 8:
            return {"error": "Please enter exactly 8 numeric values separated by commas."}
    except ValueError:
        return {"error": "Invalid input — please enter numeric values separated by commas."}

    # Prepare and scale
    input_array = np.asarray(input_values).reshape(1, -1)
    std_data = scaler.transform(input_array)

    # Predict
    prediction = classifier.predict(std_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {"Prediction": result}



model = load_model()


st.title("Diabetes Prediction App 🩺 - GROUP 3(DMDW)")

st.subheader("Enter Patient Details:")

# Arrange inputs in a clean 2-column layout (start blank by using value=None)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=None, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=None, step=1)
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=None, step=1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=None, format="%.3f")

with col2:
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=None, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=None, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=None, format="%.1f")
    age = st.number_input("Age", min_value=1, max_value=120, value=None, step=1)

# Only build the inference input string when all values are provided
if None not in (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    text = f"{pregnancies},{glucose},{blood_pressure},{skin_thickness},{insulin},{bmi},{dpf},{age}"
else:
    text = ""  # empty when missing inputs

import traceback

if st.button("Run"):
    # Check inputs present
    if text == "":
        st.warning("Please fill in all 8 input fields before running the model.")
    else:
        t0 = time.time()
        try:
            res = run_inference(model, text)
        except Exception as exc:
            st.error(f"Error while running inference: {exc}")
            st.text(traceback.format_exc())
            res = None
        finally:
            t1 = time.time()
            st.info(f"Done — {t1 - t0:.2f}s")

        if res is not None:
            st.write(res)





# show some sample data / debug
if st.sidebar.checkbox("Show raw notebook snippets"):
    st.subheader("Extracted notebook snippets (for you to paste in load_model/run_inference)")
    with st.expander("Notebook snippets"):
        import pathlib
        filepath = pathlib.Path("notebook_helpers.txt")
        if filepath.exists():
            st.code(filepath.read_text(), language="python")
        else:
            st.write("No snippets found.")
