# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# --- 2. Load the dataset ---
# Path is now relative to the project root, assuming the script is run from there.
# (Which it is, via the 'command' in .streamlit/config.toml)
try:
    data_file_path = 'data/Salary Data.csv' # Simplified path
    df = pd.read_csv(data_file_path)
    print(f"Dataset '{data_file_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{data_file_path}' not found. Please ensure it's in the 'data/' directory relative to the project root.")
    exit()

# --- 3. Data Cleaning and Type Conversion ---
df_cleaned = df.dropna().copy()
print(f"Original rows: {len(df)}, Rows after dropping NaNs: {len(df_cleaned)}")

df_cleaned['Age'] = df_cleaned['Age'].astype(int)
df_cleaned['Years of Experience'] = df_cleaned['Years of Experience'].astype(int)
df_cleaned['Salary'] = df_cleaned['Salary'].astype(int)
print("Numerical columns converted to integer type.")

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
print("Features (X) and target (y) defined, including 'Job Title Grouped'.")
print(f"Features used: {X.columns.tolist()}")

# --- 6. Identify Numerical and Categorical Columns for Preprocessing ---
numerical_cols = ['Age', 'Years of Experience']
categorical_cols = ['Gender', 'Education Level', 'Job Title Grouped']
print(f"Numerical columns identified: {numerical_cols}")
print(f"Categorical columns identified: {categorical_cols}")

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

# --- 12. Create the 'models' directory if it doesn't exist ---
# This path is now directly relative to the project root.
models_dir = 'models' # Simplified path
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")
else:
    print(f"Directory '{models_dir}' already exists.")

# --- 13. Save the trained Pipeline ---
pipeline_save_path = os.path.join(models_dir, 'salary_predictor_pipeline.pkl')
joblib.dump(model_pipeline, pipeline_save_path)
print(f"Model pipeline saved as: {pipeline_save_path}")

# --- 14. Save the list of Expected Feature Names ---
ohe_transformer = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_cols)
all_feature_names = list(numerical_cols) + list(ohe_feature_names)

expected_features_path = os.path.join(models_dir, 'expected_features.pkl')
joblib.dump(all_feature_names, expected_features_path)
print(f"Expected feature names saved as: {expected_features_path}")

# --- 15. Save the list of Job Titles for the App's Dropdown ---
app_job_titles_path = os.path.join(models_dir, 'app_job_titles.pkl')
joblib.dump(app_job_titles, app_job_titles_path)
print(f"App job titles list saved as: {app_job_titles_path}")

print("\nModel training and saving process completed successfully!")
