# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

print("--- TRAIN_MODEL.PY SCRIPT STARTING (MAX DEBUG) ---")
print(f"Current working directory (os.getcwd()): {os.getcwd()}")
print(f"Script absolute path (__file__): {os.path.abspath(__file__)}")
print(f"Script directory (os.path.dirname(__file__)): {os.path.dirname(__file__)}")

# --- 2. Load the dataset ---
# Path for data file, relative to the project root (where the 'command' is run from)
data_file_relative_path = 'data/Salary Data.csv'
data_file_full_path = os.path.join(os.getcwd(), data_file_relative_path)

print(f"\n--- Data Loading ---")
print(f"Attempting to load data from relative path: '{data_file_relative_path}'")
print(f"Full absolute path for data file: '{data_file_full_path}'")
print(f"Does data file exist at full path? {os.path.exists(data_file_full_path)}")

try:
    df = pd.read_csv(data_file_relative_path)
    print(f"SUCCESS: Dataset '{data_file_relative_path}' loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Data file NOT FOUND at '{data_file_full_path}'. Exiting train_model.py.")
    exit() # Exit the script if the file is not found
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading data: {e}. Exiting train_model.py.")
    exit()

# --- 3. Data Cleaning and Type Conversion ---
df_cleaned = df.dropna().copy()
print(f"Data cleaning: Original rows: {len(df)}, Rows after dropping NaNs: {len(df_cleaned)}")

df_cleaned['Age'] = df_cleaned['Age'].astype(int)
df_cleaned['Years of Experience'] = df_cleaned['Years of Experience'].astype(int)
df_cleaned['Salary'] = df_cleaned['Salary'].astype(int)
print("Data cleaning: Numerical columns converted to integer type.")

# --- 4. Handle 'Job Title' High Cardinality ---
JOB_TITLE_FREQ_THRESHOLD = 5
job_title_counts = df_cleaned['Job Title'].value_counts()
job_titles_to_keep = job_title_counts[job_title_counts >= JOB_TITLE_FREQ_THRESHOLD].index.tolist()
df_cleaned['Job Title Grouped'] = df_cleaned['Job Title'].apply(
    lambda x: x if x in job_titles_to_keep else 'Other Job Title'
)
print(f"Job titles grouped. Original unique: {df['Job Title'].nunique()}, Grouped unique: {df_cleaned['Job Title Grouped'].nunique()}")

app_job_titles = sorted(df_cleaned['Job Title Grouped'].unique().tolist())

# --- 5. Define Features (X) and Target (y) ---
X = df_cleaned[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title Grouped']]
y = df_cleaned['Salary']
print(f"Features (X) and target (y) defined. Features used: {X.columns.tolist()}")

# --- 6. Identify Numerical and Categorical Columns for Preprocessing ---
numerical_cols = ['Age', 'Years of Experience']
categorical_cols = ['Gender', 'Education Level', 'Job Title Grouped']
print(f"Numerical columns: {numerical_cols}, Categorical columns: {categorical_cols}")

# --- 7. Create a ColumnTransformer for Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
print("ColumnTransformer (preprocessor) created.")

# --- 8. Create a Pipeline ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
print("Model pipeline created.")

# --- 9. Split the Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

# --- 10. Train the Model (Fit the Pipeline) ---
model_pipeline.fit(X_train, y_train)
print("Model pipeline trained successfully.")

# --- 11. Evaluate the Model ---
score = model_pipeline.score(X_test, y_test)
print(f"Model R-squared on test set: {score:.2f}")

# --- 12. Create the 'models' directory inside 'app/' ---
# This path is relative to the project root, and targets the 'app/models/' location.
models_dir_relative_to_root = 'app/models'
models_dir_full_path = os.path.join(os.getcwd(), models_dir_relative_to_root)

print(f"\n--- Saving Models ---")
print(f"Target directory for models (relative to root): '{models_dir_relative_to_root}'")
print(f"Full absolute path for models directory: '{models_dir_full_path}'")

if not os.path.exists(models_dir_relative_to_root):
    try:
        os.makedirs(models_dir_relative_to_root)
        print(f"SUCCESS: Created directory: {models_dir_full_path}")
    except Exception as e:
        print(f"ERROR: Failed to create directory '{models_dir_full_path}': {e}. Exiting train_model.py.")
        exit()
else:
    print(f"Directory '{models_dir_full_path}' already exists.")

# --- 13. Save the trained Pipeline ---
pipeline_save_relative_path = os.path.join(models_dir_relative_to_root, 'salary_predictor_pipeline.pkl')
pipeline_save_full_path = os.path.join(os.getcwd(), pipeline_save_relative_path)
try:
    joblib.dump(model_pipeline, pipeline_save_relative_path)
    print(f"SUCCESS: Model pipeline saved to: {pipeline_save_full_path}")
except Exception as e:
    print(f"ERROR: Failed to save pipeline to '{pipeline_save_full_path}': {e}. Exiting train_model.py.")
    exit()

# --- 14. Save the list of Expected Feature Names ---
ohe_transformer = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_cols)
all_feature_names = list(numerical_cols) + list(ohe_feature_names)

expected_features_save_relative_path = os.path.join(models_dir_relative_to_root, 'expected_features.pkl')
expected_features_save_full_path = os.path.join(os.getcwd(), expected_features_save_relative_path)
try:
    joblib.dump(all_feature_names, expected_features_save_relative_path)
    print(f"SUCCESS: Expected feature names saved to: {expected_features_save_full_path}")
except Exception as e:
    print(f"ERROR: Failed to save expected features to '{expected_features_save_full_path}': {e}. Exiting train_model.py.")
    exit()

# --- 15. Save the list of Job Titles for the App's Dropdown ---
app_job_titles_save_relative_path = os.path.join(models_dir_relative_to_root, 'app_job_titles.pkl')
app_job_titles_save_full_path = os.path.join(os.getcwd(), app_job_titles_save_relative_path)
try:
    joblib.dump(app_job_titles, app_job_titles_save_relative_path)
    print(f"SUCCESS: App job titles list saved to: {app_job_titles_save_full_path}")
except Exception as e:
    print(f"ERROR: Failed to save app job titles to '{app_job_titles_save_full_path}': {e}. Exiting train_model.py.")
    exit()

print("\n--- TRAIN_MODEL.PY SCRIPT COMPLETED SUCCESSFULLY! ---")
