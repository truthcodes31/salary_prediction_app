💰 Salary Prediction Web App
This is a simple web application built with Streamlit and Python that predicts an individual's salary based on several key factors using a Linear Regression machine learning model.

✨ Features
Interactive Input: User-friendly sliders and dropdowns for inputting age, gender, education level, years of experience, and job title.

Machine Learning Powered: Utilizes a Linear Regression model trained on a real-world salary dataset.

Data Preprocessing Pipeline: Incorporates StandardScaler for numerical features and OneHotEncoder for categorical features (including intelligent grouping of infrequent job titles).

Instant Predictions: Provides immediate salary estimations upon clicking the "Predict Salary" button.

Clear UI: Organized layout with sections for input, prediction, and information about the app.

📊 Dataset
The application uses a Salary Data.csv dataset (expected to be in the data/ directory) which contains information such as:

Age

Gender

Education Level

Job Title

Years of Experience

Salary (the target variable)

Note: For Job Title, infrequent entries are grouped into an 'Other Job Title' category to manage high cardinality and improve model performance.

📁 Project Structure
salary_prediction_app/
├── data/
│   └── Salary Data.csv
├── models/
│   ├── salary_predictor_pipeline.pkl  # Trained ML pipeline (preprocessor + model)
│   ├── expected_features.pkl          # List of feature names model expects
│   └── app_job_titles.pkl             # List of grouped job titles for app dropdown
├── app/
│   ├── app.py                         # Streamlit web application code
│   └── requirements.txt               # Python dependencies for the app
├── scripts/
│   └── train_model.py                 # Script to train the ML model and save artifacts
├── .gitignore                         # Specifies files/folders to ignore in Git
└── README.md                          # This README file

🚀 Setup and Usage
Follow these steps to get the application running on your local machine.

Prerequisites
Python 3.8+ (recommended)

pip (Python package installer)

1. Clone the Repository (or create the structure)
If you have this project as a Git repository, clone it:

git clone https://github.com/your-repo-link/salary_prediction_app.git
cd salary_prediction_app

Otherwise, ensure you have created the file structure as described in the "Project Structure" section and placed Salary Data.csv in the data/ folder.

2. Create a Virtual Environment (Recommended)
It's good practice to use a virtual environment to manage dependencies.

python -m venv venv

3. Activate the Virtual Environment
On Windows:

.\venv\Scripts\activate

On macOS / Linux:

source venv/bin/activate

4. Install Dependencies
Navigate to the project root directory and install the required Python packages:

pip install -r app/requirements.txt

5. Train the Machine Learning Model
This script will load your dataset, preprocess it, train the Linear Regression model, and save the necessary .pkl files into the models/ directory.

Important: Ensure Salary Data.csv is in the data/ directory before running this.

python scripts/train_model.py

You should see output confirming the successful training and saving of the model files.

6. Run the Streamlit Web App
Once the model files are generated, you can launch the Streamlit application.

streamlit run app/app.py

This command will open the web application in your default browser (usually at http://localhost:8501).

⚙️ How it Works (Technical Overview)
train_model.py:

Loads Salary Data.csv.

Cleans data (drops NaNs, converts types).

Groups infrequent Job Title entries into 'Other Job Title'.

Defines features (Age, Gender, Education Level, Years of Experience, Job Title Grouped) and target (Salary).

Creates a scikit-learn Pipeline that includes:

ColumnTransformer: Applies StandardScaler to numerical features and OneHotEncoder (with handle_unknown='ignore') to categorical features.

LinearRegression: The core prediction model.

Trains the pipeline on the prepared data.

Saves the trained Pipeline (salary_predictor_pipeline.pkl), the list of expected feature names (expected_features.pkl), and the list of grouped job titles for the app's dropdown (app_job_titles.pkl) into the models/ directory.

app.py:

Loads the saved salary_predictor_pipeline.pkl and app_job_titles.pkl files.

Uses Streamlit widgets (st.slider, st.selectbox, st.button) to create an interactive user interface.

Collects user inputs and forms a Pandas DataFrame with the correct column names (Age, Gender, Education Level, Years of Experience, Job Title Grouped).

When the "Predict Salary" button is clicked, it passes this input_df to the loaded model_pipeline.predict(). The pipeline automatically handles the necessary preprocessing (scaling and one-hot encoding) before making the prediction.

Displays the predicted salary to the user.

Developed by SATYA PRAKASH SHANDILYA