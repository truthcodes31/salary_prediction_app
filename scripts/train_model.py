# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# --- Get the absolute path to the project root ---
# This gets the directory of the current script (scripts/train_model.py)
script_dir = os.path.dirname(__file__)
# Go up one level from 'scripts/' to the project root 'salary_prediction_app/'
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# --- Construct the absolute path to the CSV file ---
csv_file_path = os.path.join(project_root, 'data', 'Salary Data.csv')

# --- 2. Load the dataset ---
try:
    df = pd.read_csv(csv_file_path) # Use the absolute path here
    print(f"Dataset '{csv_file_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset not found at '{csv_file_path}'. Please ensure it's in the 'data/' directory within the project root.")
    exit()

df_cleaned = df.dropna().copy()
print(f"Original rows: {len(df)}, Rows after dropping NaNs: {len(df_cleaned)}")

df_cleaned['Age'] = df_cleaned['Age'].astype(int)
df_cleaned['Years of Experience'] = df_cleaned['Years of Experience'].astype(int)
df_cleaned['Salary'] = df_cleaned['Salary'].astype(int)
print("Numerical columns converted to integer type.")

JOB_TITLE_FREQ_THRESHOLD = 5

job_title_counts = df_cleaned['Job Title'].value_counts()
job_titles_to_keep = job_title_counts[job_title_counts >= JOB_TITLE_FREQ_THRESHOLD].index.tolist()
df_cleaned['Job Title Grouped'] = df_cleaned['Job Title'].apply(
    lambda x: x if x in job_titles_to_keep else 'Other Job Title'
)
print(f"Job titles grouped. Original unique: {df['Job Title'].nunique()}, Grouped unique: {df_cleaned['Job Title Grouped'].nunique()}")

app_job_titles = sorted(df_cleaned['Job Title Grouped'].unique().tolist())

X = df_cleaned[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title Grouped']]
y = df_cleaned['Salary']
print("Features (X) and target (y) defined, including 'Job Title Grouped'.")
print(f"Features used: {X.columns.tolist()}")

numerical_cols = ['Age', 'Years of Experience']
categorical_cols = ['Gender', 'Education Level', 'Job Title Grouped']
print(f"Numerical columns identified: {numerical_cols}")
print(f"Categorical columns identified: {categorical_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
print("ColumnTransformer (preprocessor) created.")

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
print("Model pipeline created.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

model_pipeline.fit(X_train, y_train)
print("Model pipeline trained successfully.")

score = model_pipeline.score(X_test, y_test)
print(f"Model R-squared on test set: {score:.2f}")

# --- Create the 'models' directory if it doesn't exist ---
# These paths are still relative to the project root, so they remain the same
models_dir = os.path.join(project_root, 'models') # Use project_root here
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")
else:
    print(f"Directory '{models_dir}' already exists.")

pipeline_save_path = os.path.join(models_dir, 'salary_predictor_pipeline.pkl')
joblib.dump(model_pipeline, pipeline_save_path)
print(f"Model pipeline saved as: {pipeline_save_path}")

ohe_transformer = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']

ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_cols)

all_feature_names = list(numerical_cols) + list(ohe_feature_names)

expected_features_path = os.path.join(models_dir, 'expected_features.pkl')
joblib.dump(all_feature_names, expected_features_path)
print(f"Expected feature names saved as: {expected_features_path}")

app_job_titles_path = os.path.join(models_dir, 'app_job_titles.pkl')
joblib.dump(app_job_titles, app_job_titles_path)
print(f"App job titles list saved as: {app_job_titles_path}")

print("\nModel training and saving process completed successfully!")
