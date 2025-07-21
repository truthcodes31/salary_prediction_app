# scripts/train_model.py

# --- 1. Import necessary libraries ---
import pandas as pd                  # For data manipulation and analysis
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression    # The Linear Regression model
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For data preprocessing
from sklearn.compose import ColumnTransformer        # For applying different transformations to different columns
from sklearn.pipeline import Pipeline                # For chaining preprocessing and modeling steps
import joblib                        # For saving and loading Python objects (our model and features)
import os                            # For interacting with the operating system (e.g., creating directories)

# --- 2. Load the dataset ---
# The path is relative to where this script (train_model.py) is located.
# Since train_model.py is in 'scripts/', and 'Salary Data.csv' is in 'data/',
# we go up one level (..) to 'salary_prediction_app/' and then into 'data/'.
try:
    df = pd.read_csv('../data/Salary Data.csv')
    print("Dataset 'Salary Data.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'Salary Data.csv' not found. Please ensure it's in the '../data/' directory.")
    exit() # Exit the script if the file is not found

# --- 3. Data Cleaning and Type Conversion ---
# Drop rows with any missing values. This is a simple handling strategy.
df_cleaned = df.dropna().copy()
print(f"Original rows: {len(df)}, Rows after dropping NaNs: {len(df_cleaned)}")

# Convert numerical columns to integer type.
df_cleaned['Age'] = df_cleaned['Age'].astype(int)
df_cleaned['Years of Experience'] = df_cleaned['Years of Experience'].astype(int)
df_cleaned['Salary'] = df_cleaned['Salary'].astype(int)
print("Numerical columns converted to integer type.")

# --- 4. Handle 'Job Title' High Cardinality ---
# Determine a threshold for "frequent" job titles.
# Job titles appearing less than this threshold will be grouped into 'Other Job Title'.
JOB_TITLE_FREQ_THRESHOLD = 5 # Example: keep job titles that appear at least 5 times

# Get value counts of Job Titles
job_title_counts = df_cleaned['Job Title'].value_counts()

# Identify job titles to keep (those above the frequency threshold)
job_titles_to_keep = job_title_counts[job_title_counts >= JOB_TITLE_FREQ_THRESHOLD].index.tolist()

# Replace infrequent job titles with 'Other Job Title'
df_cleaned['Job Title Grouped'] = df_cleaned['Job Title'].apply(
    lambda x: x if x in job_titles_to_keep else 'Other Job Title'
)
print(f"Job titles grouped. Original unique: {df['Job Title'].nunique()}, Grouped unique: {df_cleaned['Job Title Grouped'].nunique()}")

# Save the list of unique grouped job titles for the app's dropdown
# This list will include 'Other Job Title' if it was created.
app_job_titles = sorted(df_cleaned['Job Title Grouped'].unique().tolist())

# --- 5. Define Features (X) and Target (y) ---
# Now we include the new 'Job Title Grouped' column
X = df_cleaned[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title Grouped']]
y = df_cleaned['Salary']
print("Features (X) and target (y) defined, including 'Job Title Grouped'.")
print(f"Features used: {X.columns.tolist()}")

# --- 6. Identify Numerical and Categorical Columns for Preprocessing ---
numerical_cols = ['Age', 'Years of Experience']
# Add 'Job Title Grouped' to categorical columns
categorical_cols = ['Gender', 'Education Level', 'Job Title Grouped']
print(f"Numerical columns identified: {numerical_cols}")
print(f"Categorical columns identified: {categorical_cols}")

# --- 7. Create a ColumnTransformer for Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        # handle_unknown='ignore' is crucial here, especially for 'Other Job Title'
        # if a new, truly unseen job title appears in the app, it will be encoded as zeros.
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
# This path is relative to where this script is run from (project root).
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")
else:
    print(f"Directory '{models_dir}' already exists.")

# --- 13. Save the trained Pipeline ---
# Save the entire pipeline (preprocessor + model) so it can be loaded later by the app.
pipeline_save_path = os.path.join(models_dir, 'salary_predictor_pipeline.pkl')
joblib.dump(model_pipeline, pipeline_save_path)
print(f"Model pipeline saved as: {pipeline_save_path}")

# --- 14. Save the list of Expected Feature Names ---
# This is crucial for ensuring consistency between training and prediction.
# It captures the exact order and names of features (especially after one-hot encoding).

# Access the fitted OneHotEncoder from the pipeline's preprocessor
ohe_transformer = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']

# Get the feature names generated by the OneHotEncoder for the categorical columns
ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_cols)

# Combine numerical feature names with the one-hot encoded feature names.
all_feature_names = list(numerical_cols) + list(ohe_feature_names)

# Save this combined list of feature names.
expected_features_path = os.path.join(models_dir, 'expected_features.pkl')
joblib.dump(all_feature_names, expected_features_path)
print(f"Expected feature names saved as: {expected_features_path}")

# --- 15. Save the list of Job Titles for the App's Dropdown ---
# This specific list is needed by app.py to populate the selectbox for Job Title.
app_job_titles_path = os.path.join(models_dir, 'app_job_titles.pkl')
joblib.dump(app_job_titles, app_job_titles_path)
print(f"App job titles list saved as: {app_job_titles_path}")

print("\nModel training and saving process completed successfully!")
