import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import constants
from flask import Flask, jsonify
import mlflow
from mlflow.artifacts import download_artifacts
import tempfile
import pickle
import mlflow

class MLModel:
#---------------CONSTRUCTOR---------------
    def __init__(self, client):
        self.client = client
        self.onehot_encoders = {}
        self.standard_scalers = []
        self.minmax_scalers = []
        self.model = None # Initially None
        self.create_required_folders()
        # load recent staging model
        self.load_staging_model()
        self.create_required_folders()
        
    def load_staging_model(self):
        """Loads the staging model and downloads artifacts from MLFLOW."""
        try:
            latest_staging = None
            for reg_model in self.client.search_registered_models():
                for v in reg_model.latest_versions:
                    if v.current_stage == "Staging":
                        latest_staging = v
                        break
                if latest_staging:
                    break

            if not latest_staging:
                print("No staging model found.")
                return

            # modell betöltése
            self.model = mlflow.sklearn.load_model(latest_staging.source)
            print("Staging model loaded.")

            # artifactok URI-je
            artifact_root = latest_staging.source.rpartition('/')[0]
            self.load_artifacts(artifact_root)
        except Exception as e:
            print(f"Error loading staging model: {e}")
            
    def load_artifacts(self, artifact_uri):
        """Downloads and loads the pickle artifacts from MLFLOW"""
        try:
            # one-hot encoders
            path = download_artifacts(f"{artifact_uri}/preprocessing/onehot_encoders.pkl")
            with open(path, "rb") as f:
                self.onehot_encoders = pickle.load(f)

            # standard scalers
            path = download_artifacts(f"{artifact_uri}/preprocessing/standard_scalers.pkl")
            with open(path, "rb") as f:
                self.standard_scalers = pickle.load(f)

            # minmax scalers
            path = download_artifacts(f"{artifact_uri}/preprocessing/minmax_scalers.pkl")
            with open(path, "rb") as f:
                self.minmax_scalers = pickle.load(f)

            print("Artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def create_required_folders(self):
        """ Creates required directories if they do not exist """
        required_folders = [
            os.path.dirname(constants.ENCODERS_PATH),
            os.path.dirname(constants.STANDARD_SCALERS_PATH),
            os.path.dirname(constants.MINMAX_SCALERS_PATH),
        ]
        for folder in required_folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"📁 Created folder: {folder}")

#---------------METHODS---------------

    @staticmethod
    def save_pickle(obj, path):
        """ Saves an object as a pickle file """
        with open(path, 'wb') as file_destination:
            pickle.dump(obj, file_destination)
        print(f"Saved: {path}")

    @staticmethod
    def load_data(path):
        """ Loads data from the specified CSV file path """
        return pd.read_csv(path)

    @staticmethod
    def clean_data(df):
        """ Cleans data and handles missing values """
        df = df.apply(lambda col: col.apply(lambda v: np.nan if isinstance(v, str) and v.strip() == '' else v))
        df['TotalCharges'] = df['TotalCharges'].apply(lambda x: float(x) if pd.notna(x) else x)
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

        return df

    
    # (MAY) NEED TO DEAL WITH THE SIMPLE INPUT FROM THE API TO BE PREPROCESSED CORRECTLY FOR FURTHER PROCESSING
    @staticmethod
    def clean_data_for_inference(sample_data_input_list):
        """ Cleans data and handles missing values for inference input """
        sample_data = pd.DataFrame(sample_data_input_list, columns=constants.ORIGINAL_DATAFRAME_COLUMN_NAMES)
        sample_data = sample_data.map(lambda v: np.nan if isinstance(v, str) and v.strip() == '' else v)
        if isinstance(sample_data['TotalCharges'], pd.Series):
            sample_data['TotalCharges'] = sample_data['TotalCharges'].apply(lambda x: float(x) if pd.notna(x) else x)
        else:
            sample_data['TotalCharges'] = float(sample_data['TotalCharges']) if pd.notna(sample_data['TotalCharges']) else sample_data['TotalCharges']
        mean_value = sample_data['TotalCharges'].mean()
        if pd.isna(mean_value):
            mean_value = 0
        sample_data['TotalCharges'] = sample_data['TotalCharges'].fillna(mean_value)

        return sample_data


    @staticmethod
    def split_customer_id(df):
        """Splits the customerID column into numeric and character parts."""

        if 'customerID' not in df.columns or df['customerID'].isna().all():
            return df, []

        split_values = df['customerID'].astype(str).str.split('-', expand=True)

        if split_values.shape[1] != 2:
            df['customerID_number'] = df['customerID']
            df['customerID_text'] = 'missing'
        else:
            df[['customerID_number', 'customerID_text']] = split_values

        max_length = df['customerID_text'].str.len().max()
        customer_id_text_column_names = [f'customerID_text_char_{i+1}' for i in range(max_length)]

        for i in range(max_length):
            df[f'customerID_text_char_{i+1}'] = df['customerID_text'].apply(
                lambda x: x[i] if isinstance(x, str) and i < len(x) else 'missing'
            )

        df = df.drop(['customerID', 'customerID_text'], axis=1)

        return df, customer_id_text_column_names


    @staticmethod
    def encode_yes_no_columns(df):
        """ Converts specified columns from 'Yes/No' or 'Male/Female' to binary format using constants.YES_NO_DICT """
        for column in constants.COLUMNS_TO_ENCODE_YES_NO:
            if column in df.columns:
                df[f'{column}_encoded'] = df[column].map(constants.YES_NO_DICT)
                df.drop(columns=[column], inplace=True)
        return df

    @staticmethod
    def encode_contract_column(df):
        """ Ordinal encoding for the 'Contract' column """
        df['Contract_encoded'] = df['Contract'].map(constants.CONTRACT_MAPPING)
        df = df.drop('Contract', axis=1)
        return df
    
    @staticmethod
    def convert_columns_to_float(df):
        """ Converts specified columns to float type """
        df[constants.COLUMNS_TO_FLOAT] = df[constants.COLUMNS_TO_FLOAT].astype(float)
        return df

    def create_one_hot_coded_columns(self, df, columns_to_encode):
        """ Applies one-hot encoding to specified columns """
        for column in columns_to_encode:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(df[[column]])
            encoded_column_names = encoder.get_feature_names_out([column])
            encoded_dataframe = pd.DataFrame(encoded_data, columns=encoded_column_names, index=df.index)
            df = pd.concat([df.drop(columns=[column]), encoded_dataframe], axis=1)
            self.onehot_encoders[column] = encoder
        return df

    def apply_standard_scaling(self, df, columns):
        """ Applies StandardScaler (for Gaussian-distributed data) """
        for column in columns:
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[[column]])
            self.standard_scalers.append(scaler)
        return df

    def apply_minmax_scaling(self, df, columns):
        """ Applies MinMaxScaler (normalization between 0.01 and 1) """
        for column in columns:
            scaler = MinMaxScaler(feature_range=(0.01, 1))
            df[column] = scaler.fit_transform(df[[column]])
            self.minmax_scalers.append(scaler)
        return df
    
    def encode_features(self, sample_data, customer_id_text_column_names):
        """Encodes categorical features with the encoders loaded from MLflow."""
        columns_to_encode = constants.COLUMNS_TO_ENCODE + customer_id_text_column_names

        encoders = self.onehot_encoders

        for col in columns_to_encode:
            enc = encoders[col]
            # Notice here now (in from 'preprocessing_pipeline_inference') we use not fit_transform, but simple transform
            # because we use what we have created (will create in production ready code)
            new = enc.transform(sample_data[[col]])
            new_cols = enc.get_feature_names_out([col])
            sample_data = pd.concat(
                [sample_data.drop(columns=[col]),
                pd.DataFrame(new, columns=new_cols, index=sample_data.index)],
                axis=1
            )
        return sample_data

    def scale_features(self, sample_data):
        """Scale numeric features with the scalers loaded from MLflow."""
        for col, scaler in zip(constants.BELL_CURVE_TYPE_COLUMNS, self.standard_scalers):
            sample_data[col] = scaler.transform(sample_data[[col]])

        # MinMaxScaler a többire
        for col, scaler in zip(constants.NOT_BELL_CURVE_TYPE_COLUMNS, self.minmax_scalers):
            sample_data[col] = scaler.transform(sample_data[[col]])

        return sample_data
    
    def train_and_save_model(self, df):
        """Train the Logistic Regression model and save it to the instance."""
        X = df.drop(columns="Churn_encoded")
        y = df["Churn_encoded"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

        lr_model = LogisticRegression(**constants.LR_BEST_PARAMS)

        nan_rows = df[df.isna().any(axis=1)]
        print(f'HERE ARE THE NAN ROWS (train and save model): {nan_rows}')
        lr_model.fit(X_train, y_train)

        self.model = lr_model
        train_accuracy, test_accuracy = self.get_accuracy(X_train, X_test, y_train, y_test)

        return train_accuracy, test_accuracy, lr_model

    
    def predict_mine(self, infer_array):
        """Predicts the outcome based on the RAW input data row."""
        try:
            df = self.preprocessing_pipeline_inference(infer_array)
            df.drop('Churn_encoded', axis=1, inplace=True)

            y_pred = self.model.predict(df)

            return int(y_pred)

        except Exception as e:
            return jsonify({'message': 'Internal Server Error. ',
                        'error': str(e)}), 500
    
    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    def get_accuracy(self, X_train, X_test, y_train, y_test):
        """Calculate and print the accuracy of the model on both the training and test data sets. """
        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy
    
    # For testing in test_of_train_and_inference_pipeline
    def get_accuracy_full(self, X, y):
        """Calculate and print the overall accuracy of the model using a data set. """
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return accuracy


#---------------PREPROCESSING PIPELINES---------------
    
    def preprocessing_pipeline(self, df):
        """ Performs full preprocessing pipeline on input dataframe and saves encoders/scalers """
        df = self.clean_data(df)
        df, customer_id_text_column_names = self.split_customer_id(df)
        df = self.encode_yes_no_columns(df)
        df = self.encode_contract_column(df)
        df = self.create_one_hot_coded_columns(df, constants.COLUMNS_TO_ENCODE + customer_id_text_column_names)
        df = self.convert_columns_to_float(df)
        df = self.apply_standard_scaling(df, constants.BELL_CURVE_TYPE_COLUMNS)
        df = self.apply_minmax_scaling(df, constants.NOT_BELL_CURVE_TYPE_COLUMNS)
        
        # Save and log preprocessing artifacts (encoders and scalers) to MLflow.
        # For each object:
        #   1) A temporary directory is created to avoid polluting the local project folder.
        #   2) The object is saved as a pickle (.pkl) file inside that temporary folder.
        #   3) The pickle file is logged to MLflow under the 'preprocessing' subdirectory in the artifact store.
        #
        # This avoids overwriting local files and keeps the workspace clean.
        # Final MLflow artifact structure:
        # artifacts/
        # └── preprocessing/
        #     ├── onehot_encoders.pkl
        #     ├── standard_scalers.pkl
        #     └── minmax_scalers.pkl
        def log_pickle_to_mlflow(obj, filename, artifact_subdir="preprocessing"):
            # 1) temporary folder
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, filename)
                # 2) pickle  
                with open(tmp_path, "wb") as f:
                    pickle.dump(obj, f)
                # 3) log into ml flow
                mlflow.log_artifact(tmp_path, artifact_path=artifact_subdir)

        log_pickle_to_mlflow(self.onehot_encoders,  "onehot_encoders.pkl")
        log_pickle_to_mlflow(self.standard_scalers, "standard_scalers.pkl")
        log_pickle_to_mlflow(self.minmax_scalers,   "minmax_scalers.pkl")

        return df

    def preprocessing_pipeline_inference(self, sample_data):
        """Performs the full preprocessing pipeline for inference."""
        
        sample_data = self.clean_data_for_inference(sample_data)
        sample_data, customer_id_text_column_names = self.split_customer_id(sample_data)
        sample_data = self.encode_yes_no_columns(sample_data)
        sample_data = self.encode_contract_column(sample_data)
        sample_data = self.encode_features(sample_data, customer_id_text_column_names)
        sample_data = self.convert_columns_to_float(sample_data)
        sample_data = self.scale_features(sample_data)

        return sample_data