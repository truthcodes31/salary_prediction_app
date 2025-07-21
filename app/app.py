# app/app.py

import streamlit as st
import pandas as pd
import joblib
# os import is no longer needed if paths are hardcoded

# --- 1. Set Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# --- 2. Construct absolute paths to the model files ---
# These paths are directly taken from the output of your train_model.py script.
# IMPORTANT: If you move your project, you will need to update these paths manually.
pipeline_path = r"C:\Users\shand\OneDrive\Desktop\salary_prediction_app\models\salary_predictor_pipeline.pkl"
expected_features_path = r"C:\Users\shand\OneDrive\Desktop\salary_prediction_app\models\expected_features.pkl"
app_job_titles_path = r"C:\Users\shand\OneDrive\Desktop\salary_prediction_app\models\app_job_titles.pkl"


# You can keep this for initial debug, but it will appear on the page.
# st.write("Streamlit is running!")

try:
    # Load models using the absolute paths
    model_pipeline = joblib.load(pipeline_path)
    expected_features = joblib.load(expected_features_path)
    app_job_titles = joblib.load(app_job_titles_path)
    # st.write("Model files loaded successfully.") # Confirm loading, can remove later
except FileNotFoundError:
    st.error(f"Error: Model files not found. Attempted to load from:")
    st.error(f"- Pipeline: {pipeline_path}")
    st.error(f"- Expected Features: {expected_features_path}")
    st.error(f"- App Job Titles: {app_job_titles_path}")
    st.error("Please ensure 'python scripts/train_model.py' was run successfully and these files exist at the specified paths.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading model files: {e}")
    st.stop()


st.title("💰 Salary Prediction App")
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="big-font">Welcome! This application estimates an individual\'s salary based on their age, gender, education, years of experience, and job title. </p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Please fill in the details below to get a prediction.</p>', unsafe_allow_html=True)

st.markdown("---")

st.subheader("Employee Details")

age = st.slider(
    "Age",
    min_value=18,
    max_value=65,
    value=30,
    step=1,
    help="Drag the slider to select the individual's age."
)

gender = st.selectbox(
    "Gender",
    ("Male", "Female"),
    help="Select the individual's gender."
)

education_level = st.selectbox(
    "Education Level",
    ("Bachelor's", "Master's", "PhD"),
    help="Select the highest level of education attained."
)

years_experience = st.slider(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=5,
    step=1,
    help="Drag the slider to select the number of years of professional experience."
)

job_title_grouped = st.selectbox(
    "Job Title",
    app_job_titles,
    help="Select the individual's job title. Infrequent titles are grouped under 'Other Job Title'."
)

input_df = pd.DataFrame([[age, gender, education_level, years_experience, job_title_grouped]],
                        columns=['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title Grouped'])

st.markdown("---")

if st.button("Predict Salary", help="Click to get the estimated salary."):
    with st.spinner('Calculating salary...'):
        try:
            predicted_salary = model_pipeline.predict(input_df)[0]

            st.success(f"**Estimated Salary:** ₹{predicted_salary:,.2f}")
            st.info("Please note: This is an estimation based on the trained model and provided inputs. "
                    "Actual salaries may vary.")
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")
            st.warning("Ensure your input features match the format the model expects.")

st.markdown("---")

with st.expander("About This App"):
    st.write("""
        This web application uses a Linear Regression model trained on a dataset of salary information.
        The model considers 'Age', 'Gender', 'Education Level', 'Years of Experience', and 'Job Title' as factors
        to predict an individual's approximate salary.
        \n**Model Training:**
        The model was trained using `scikit-learn` in Python. It employs:
        - `StandardScaler` for numerical features (Age, Years of Experience).
        - `OneHotEncoder` for categorical features (Gender, Education Level, Job Title Grouped).
        - A `LinearRegression` algorithm.
        \n**Data Source:**
        The predictions are based on the patterns learned from the 'Salary Data.csv' dataset.
    """)

with st.expander("How to Use"):
    st.write("""
        1. Adjust the 'Age' and 'Years of Experience' sliders to the desired values.
        2. Select the 'Gender', 'Education Level', and 'Job Title' from the dropdown menus.
        3. Click the 'Predict Salary' button to see the estimated salary.
    """)

st.markdown("---")
st.markdown("Developed by [SATYA PRAKASH SHANDILYA](https://github.com/truthcodes31)")
