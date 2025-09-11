import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from pathlib import Path
import constants

#----------FOR MAINLY TRAIN_PIPELINE----------

def load_data(path):
    """Loads data from the specified CSV file path"""
    return pd.read_csv(path)

def clean_data(df):
    """ Cleans data and handles missing values """
    df = df.apply(lambda col: col.apply(lambda v: np.nan if isinstance(v, str) and v.strip() == '' else v))
    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: float(x) if pd.notna(x) else x)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    return df


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



def encode_yes_no_columns(df):
    """ Converts specified columns from 'Yes/No' or 'Male/Female' to binary format using constants.YES_NO_DICT """
    for column in constants.COLUMNS_TO_ENCODE_YES_NO:
        if column in df.columns:
            df[f'{column}_encoded'] = df[column].map(constants.YES_NO_DICT)
            df.drop(columns=[column], inplace=True)
    return df


def encode_contract_column(df):
    """ Ordinal encoding for the 'Contract' column """
    df['Contract_encoded'] = df['Contract'].map(constants.CONTRACT_MAPPING)
    df = df.drop('Contract', axis=1)
    return df

def create_one_hot_coded_columns(df, columns_to_encode):
    """ Applies one-hot encoding to specified columns """
    onehot_encoders = {}
    for column in columns_to_encode:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_data = encoder.fit_transform(df[[column]])
        encoded_column_names = encoder.get_feature_names_out([column])
        encoded_dataframe = pd.DataFrame(encoded_data, columns=encoded_column_names, index=df.index)
        df = pd.concat([df.drop(columns=[column]), encoded_dataframe], axis=1)
        onehot_encoders[column] = encoder
    return df, onehot_encoders

def convert_columns_to_float(df):
    """ Converts specified columns to float type """
    df[constants.COLUMNS_TO_FLOAT] = df[constants.COLUMNS_TO_FLOAT].astype(float)
    return df

def apply_standard_scaling(df, columns):
    """ Applies StandardScaler (for Gaussian-distributed data) """
    scalers = []
    for column in columns:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
        scalers.append(scaler)
    return df, scalers

def apply_minmax_scaling(df, columns):
    """ Applies MinMaxScaler (normalization between 0.01 and 1) """
    scalers = []
    for column in columns:
        scaler = MinMaxScaler(feature_range=(0.01, 1))
        df[column] = scaler.fit_transform(df[[column]])
        scalers.append(scaler)
    return df, scalers

def create_folder(folder_path):
    """ Creates the specified folder if it does not exist """
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path):
    """ Saves an object as a pickle file """
    with open(path, 'wb') as file_destination:
        pickle.dump(obj, file_destination)

def load_pickle(path):
    """ Loads an object from a pickle file """
    with open(path, 'rb') as file_source:
        return pickle.load(file_source)


#----------FOR MAINLY INFERENCE_PIPELINE----------


def get_sample_data(df):
    """Extracts a sample from the dataset and converts it to a DataFrame."""
    sample_data = df.iloc[0]
    return pd.DataFrame([sample_data])

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


def encode_features(sample_data, customer_id_text_column_names):
    """Encodes categorical features using pre-trained encoders."""
    sample_data = encode_yes_no_columns(sample_data)
    sample_data = encode_contract_column(sample_data)
    
    columns_to_encode = constants.COLUMNS_TO_ENCODE + customer_id_text_column_names
    with open(constants.ENCODERS_PATH, 'rb') as file:
        encoders = pickle.load(file)
    
    for column_name in columns_to_encode:
        current_encoder = encoders[column_name]
        encoded_data = current_encoder.transform(sample_data[[column_name]])
        encoded_dataframe = pd.DataFrame(encoded_data, 
                                         columns=current_encoder.get_feature_names_out([column_name]), 
                                         index=sample_data.index)
        sample_data = pd.concat([sample_data.drop(columns=[column_name]), encoded_dataframe], axis=1)
    
    return convert_columns_to_float(sample_data)


def scale_features(sample_data):
    """Applies standard and min-max scaling to the dataset."""
    
    # Apply standard scaling
    with open(constants.STANDARD_SCALERS_PATH, 'rb') as file:
        scalers = pickle.load(file)
    for column_name, scaler in zip(constants.BELL_CURVE_TYPE_COLUMNS, scalers):
        sample_data[column_name] = scaler.transform(sample_data[[column_name]])

    # Apply min-max scaling
    with open(constants.MINMAX_SCALERS_PATH, 'rb') as file:
        scalers = pickle.load(file)
    for column_name, scaler in zip(constants.NOT_BELL_CURVE_TYPE_COLUMNS, scalers):
        sample_data[column_name] = scaler.transform(sample_data[[column_name]])

    return sample_data


def load_model(model_path):
    """Loads the trained model if it exists."""
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error when loading the file: {e}")
    else:
        print(f"Warning! File not found -> {model_path}")
    return None