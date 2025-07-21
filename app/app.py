# app/app.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. Set Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# --- Define the path to the data file relative to the app.py script ---
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Salary Data.csv')

# --- Use Streamlit's caching decorator for resource-heavy operations ---
@st.cache_resource
def load_and_train_model():
    """
    Loads data, trains the model, and returns the trained pipeline and app job titles.
    This function will be cached by Streamlit to run only once.
    """
    # st.write("Loading data and training model... (This runs only once per deployment)") # REMOVED/COMMENTED OUT

    # Load the dataset
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        # st.write(f"Dataset '{DATA_FILE_PATH}' loaded successfully. Shape: {df.shape}") # REMOVED/COMMENTED OUT
    except FileNotFoundError:
        st.error(f"ERROR: Data file not found at '{DATA_FILE_PATH}'. Please ensure 'Salary Data.csv' is in the 'data/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"ERROR: An unexpected error occurred while loading data: {e}.")
        st.stop()

    # Data Cleaning and Type Conversion
    df_cleaned = df.dropna().copy()
    # st.write(f"Data cleaning: Original rows: {len(df)}, Rows after dropping NaNs: {len(df_cleaned)}") # REMOVED/COMMENTED OUT
    df_cleaned['Age'] = df_cleaned['Age'].astype(int)
    df_cleaned['Years of Experience'] = df_cleaned['Years of Experience'].astype(int)
    df_cleaned['Salary'] = df_cleaned['Salary'].astype(int)
    # st.write("Data cleaning: Numerical columns converted to integer type.") # REMOVED/COMMENTED OUT

    # Handle 'Job Title' High Cardinality
    JOB_TITLE_FREQ_THRESHOLD = 5
    job_title_counts = df_cleaned['Job Title'].value_counts()
    job_titles_to_keep = job_title_counts[job_title_counts >= JOB_TITLE_FREQ_THRESHOLD].index.tolist()
    df_cleaned['Job Title Grouped'] = df_cleaned['Job Title'].apply(
        lambda x: x if x in job_titles_to_keep else 'Other Job Title'
    )
    # st.write(f"Job titles grouped. Original unique: {df['Job Title'].nunique()}, Grouped unique: {df_cleaned['Job Title Grouped'].nunique()}") # REMOVED/COMMENTED OUT

    app_job_titles = sorted(df_cleaned['Job Title Grouped'].unique().tolist())

    # Define Features (X) and Target (y)
    X = df_cleaned[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title Grouped']]
    y = df_cleaned['Salary']
    # st.write(f"Features (X) and target (y) defined. Features used: {X.columns.tolist()}") # REMOVED/COMMENTED OUT

    # Identify Numerical and Categorical Columns for Preprocessing
    numerical_cols = ['Age', 'Years of Experience']
    categorical_cols = ['Gender', 'Education Level', 'Job Title Grouped']
    # st.write(f"Numerical columns: {numerical_cols}, Categorical columns: {categorical_cols}") # REMOVED/COMMENTED OUT

    # Create a ColumnTransformer for Preprocessing
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    # st.write("ColumnTransformer (preprocessor) created.") # REMOVED/COMMENTED OUT

    # Create a Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    # st.write("Model pipeline created.") # REMOVED/COMMENTED OUT

    # Split the Data into Training and Testing Sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # st.write(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.") # REMOVED/COMMENTED OUT

    # Train the Model (Fit the Pipeline)
    model_pipeline.fit(X_train, y_train)
    # st.write("Model pipeline trained successfully.") # REMOVED/COMMENTED OUT

    # Evaluate the Model
    score = model_pipeline.score(X_test, y_test)
    # st.write(f"Model R-squared on test set: {score:.2f}") # REMOVED/COMMENTED OUT

    return model_pipeline, app_job_titles

# --- Load and Train Model (this call will be cached) ---
model_pipeline, app_job_titles = load_and_train_model()

# These debug lines can also be removed now
# st.write("Model pipeline loaded/trained successfully.")
# st.write(f"App Job Titles (first 5): {app_job_titles[:5]}...")
# st.write(f"Total App Job Titles: {len(app_job_titles)}")


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
