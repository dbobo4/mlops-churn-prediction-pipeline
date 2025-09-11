import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Run the test from the terminal with this: pytest tests/test_of_train_and_inference_pipeline.py

# Beside pytest: the pip install importlib_metadata need to be!
# This is useful when you want to dynamically check versions, entry points, or other metadata.

# The filename follows the pytest convention!!!
# The file name is test_of_train_and_inference_pipeline.py!!!
# Pytest automatically detects and runs any file matching test_*.py or *_test.py!!!
# If the file were named train_pipeline.py, pytest would not run it automatically!!!

# Add the parent directory to sys.path so we can import MLModel
current_file_path = Path(__file__).resolve()  # Get the path of the current file
parent_directory = current_file_path.parent.parent  # Get the parent directory (two levels up)
sys.path.append(str(parent_directory))  # Add the parent directory to sys.path

# Now we can import our MLModel
from MLModel import MLModel
import constants

def test_prediction_accuracy_calculation():

    print("🔵 Creating MLModel instance...")
    mlmodel_object = MLModel()

    print("✅ Model instance created, loading dataset...")
    data_path = constants.ORIGINAL_DATA_FILE_PATH
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")

    # Training pipeline
    print("🔵 Running preprocessing pipeline...")
    df_preprocessed = mlmodel_object.preprocessing_pipeline(df)
    print("✅ Preprocessing pipeline completed!")

    y_expected = df_preprocessed['Churn_encoded']
    print("🔵 Running accuracy test on training pipeline...")
    accuracy_train_pipeline_full = mlmodel_object.get_accuracy_full(df_preprocessed.drop(columns='Churn_encoded'), y_expected) # Here there is no Curn_encoded
    accuracy_train_pipeline_full = np.round(accuracy_train_pipeline_full, 2)
    print(f"✅ Training accuracy computed: {accuracy_train_pipeline_full}")

    # Inference pipeline
    print("🔵 Running inference pipeline...")
    preprocessed_list = []

    for idx, (index, row) in enumerate(df.iterrows()):
        if idx % 100 == 0:
            print(f"🔄 Processing row {idx}...")

        row_df = row.to_frame().T  # Convert it into a single-row DataFrame
        preprocessed_list.append(mlmodel_object.preprocessing_pipeline_inference(row_df))
    
    print("✅ Inference pipeline completed!")

    df_preprocessed = pd.concat(preprocessed_list)
    print("🔵 Running accuracy test on inference pipeline...")
    accuracy_inference_pipeline_full = mlmodel_object.get_accuracy_full(df_preprocessed.drop(columns='Churn_encoded'), y_expected)
    accuracy_inference_pipeline_full = np.round(accuracy_inference_pipeline_full, 2)
    print(f"✅ Inference accuracy computed: {accuracy_inference_pipeline_full}")

    assert accuracy_train_pipeline_full == accuracy_inference_pipeline_full, 'Inference prediction accuracy is not as expected'
