# File paths
ORIGINAL_DATA_FILE_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ENCODERS_PATH = "artifacts/encoders/onehot_encoders.pkl"
STANDARD_SCALERS_PATH = "artifacts/scalers/standard_scalers.pkl"
MINMAX_SCALERS_PATH = "artifacts/scalers/normalisation_scalers.pkl"
PREPROCESSED_TRAIN_DATA_PATH = "artifacts/preprocessed_data/cleaned_train_telco_dataset.csv"
PREPROCESSED_INFERENCE_DATA_PATH = "artifacts/preprocessed_data/cleaned_inference_telco_dataset.csv"
TEMPORARY_API_INITIAL_DATASET_PATH = "artifacts/temporary_telco_api_initial_dataset.csv"
TEMPORARY_API_PROCESSED_DATASET_PATH = "artifacts/temporary_telco_api_processed_dataset.csv"
TRAINED_MODEL_PATH = "artifacts/trained_model_for_inference_pipeline_to_first_use_to_test.pkl"
API_MODEL_PATH = "artifacts/api_model.pkl"

# MLflow model settings
MODEL_NAME = "Churn_Prediction_Model"
MODEL_STAGE = "Staging"
ML_FLOW_TRACKING_URI = "http://127.0.0.1:5102"

# API

# Column names for inference_pipeline to create dataframe in API
ORIGINAL_DATAFRAME_COLUMN_NAMES = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]


# Binary mapping for Yes/No and Gender columns
YES_NO_DICT = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

# Columns to apply binary encoding
COLUMNS_TO_ENCODE_YES_NO = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

# Mapping for contract types
CONTRACT_MAPPING = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}

# One-hot encoding columns
COLUMNS_TO_ENCODE = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaymentMethod'
]

# Columns that should be converted to float
COLUMNS_TO_FLOAT = ['TotalCharges', 'customerID_number']

# Feature scaling categories
BELL_CURVE_TYPE_COLUMNS = ['MonthlyCharges']
NOT_BELL_CURVE_TYPE_COLUMNS = ['tenure', 'TotalCharges', 'customerID_number', 'Contract_encoded']

# Best hyperparameters for the Logistic Regression model
LR_BEST_PARAMS = {
    'penalty': 'l1',
    'C': 0.09967870056144569,
    'solver': 'liblinear',
    'max_iter': 143,
    'fit_intercept': False
}

# Test split ratio
TEST_SIZE = 0.33
RANDOM_STATE = 42

PORT_NUMBER_REST_API = 8000
LOCALHOST_NUMBER = "127.0.0.1"
REST_API_BASE_URL = f"http://{LOCALHOST_NUMBER}:{PORT_NUMBER_REST_API}/"

# AirFlow constants
COMPARISON_FOLDER_PATH = "airflow_data_comparison_upload"
AIRFLOW_USER = "admin"
AIRFLOW_PASSWORD = "admin"
AIRFLOW_PORT = 8080
AIRFLOW_BASE_URL = f"http://{LOCALHOST_NUMBER}:{AIRFLOW_PORT}/"
MY_EMAIL_ADDRESS_TO_SEND_NOTIFICATOIN = "dbobo4@gmail.com"