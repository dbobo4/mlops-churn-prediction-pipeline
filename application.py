from flask import Flask, request
from flask_restx import Api, Resource, fields  # When pip installed use: pip install flask-restx
from werkzeug.datastructures import FileStorage
import os
import pandas as pd
from MLModel import MLModel
import constants
from datetime import datetime
from pathlib import Path
import requests

# MLflow imports
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

import tempfile
from report_generator import create_report

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(constants.ML_FLOW_TRACKING_URI)

# ▶ DOCKER MODE (recommended) — one container starts everything
# ------------------------------------------------------------
# Prerequisite:
#   Make sure Docker Desktop (or the Docker daemon) is running on your machine.
#
# How to run (from project root, thanks to docker-compose.yml config):
#   docker compose up -d --build
#   # or force clean build without cache (if it fails, just try again — sometimes Docker is naughty):
#   docker compose build --no-cache && docker compose up -d

# The docker-compose.yml file defines:
#   - ports (5102, 8080, 8000, 8501)
#   - env_file (.env.smtp with sensitive configs)
#   - restart policy
#   - image/container naming
#
# Access from host:
#   - API (Swagger): http://localhost:8000
#   - MLflow UI:    http://localhost:5102
#   - Airflow UI:   http://localhost:8080
#   - Dashboard:    http://localhost:8501


# ▶ MANUAL MODE — First-time setup
# ------------------------------------------------------------
# 1) MLflow server (creates mlruns/ folder + sqlite db on first run):
# mlflow server \
#   --backend-store-uri sqlite:///mlflow.db \
#   --default-artifact-root file:///%CD%/mlruns \
#   --host 127.0.0.1 --port 5102
#   → This will create 'mlruns/' in the current folder to store artifacts.

# 2) Airflow init (only once):
# airflow db init
# airflow users create \
#   --username admin --password admin \
#   --firstname Admin --lastname User \
#   --role Admin --email admin@example.com

# 3) Flask API (production-like):
#   Windows: waitress-serve --listen=127.0.0.1:8000 application:app

# 4) Streamlit Dashboard:
# streamlit run dashboard.py
#   (First call /model/evaluate_batch via API to generate report.html)


# ▶ MANUAL MODE — Daily start (after init is done once)
# ------------------------------------------------------------
# 1) MLflow server:
# mlflow server \
#   --backend-store-uri sqlite:///mlflow.db \
#   --default-artifact-root file:///%CD%/mlruns \
#   --host 127.0.0.1 --port 5102

# 2) Airflow:
# airflow webserver --port 8080
# airflow scheduler

# 3) Flask API:
# waitress-serve --listen=127.0.0.1:8000 application:app

# 4) Streamlit:
# streamlit run dashboard.py


# Set default experiment
experiment_name = "new_clean_experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Initialize MLflow client and model object
client = MlflowClient()

# Attempt to load the latest "Staging" model version if it exists
try:
    mlmdel_object = MLModel(client=client)
    if mlmdel_object.model is None:
        print("⚠️  Warning: No 'Staging' model found. Training is still possible.")
except Exception as e:
    print(f"⚠️  Warning: Could not load 'Staging' model. Training is still possible. Error: {e}")
    mlmdel_object = MLModel(client=client)

# Rest API creation
app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation')

predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.String, required=True, description='A row of data for inference/prediction')
})

file_upload = api.parser()
file_upload.add_argument('file', location='files', type=FileStorage, required=True, help='CSV file for training')

# Defines a namespace for grouping related endpoints under a common URL prefix.
# All routes added to this namespace will be accessible under /model/... in Swagger UI.
# Helps organize the API structure and documentation logically (e.g., /model/train, /model/predict).
ns = api.namespace('model', description='Model operations')

@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Invalid file type'}, 400
        
        temporary_data_path = constants.TEMPORARY_API_INITIAL_DATASET_PATH
        uploaded_file.save(temporary_data_path)

        try:
            # Start MLflow run
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = constants.MODEL_NAME  # Name for model registry
            # MLflow starts to registry this RUN (every run will be seen on the ui)
            with mlflow.start_run(run_name=run_name) as run:
                # Load data
                df_raw = pd.read_csv(temporary_data_path)
                # Prepare input example and signature
                input_example = df_raw.iloc[:1]
                signature = mlflow.models.infer_signature(
                    df_raw.drop(columns=['Churn']),
                    df_raw['Churn']
                )
                # Preprocess
                df_pre = mlmdel_object.preprocessing_pipeline(df_raw)

                # Save preprocessed dataset temporarily using tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_pre:
                    df_pre.to_csv(tmp_pre.name, index=False)
                    preprocessed_temp_path = tmp_pre.name

                # Log raw and preprocessed datasets as MLflow artifacts
                mlflow.log_artifact(temporary_data_path, artifact_path="datasets/raw")

                # Rename the preprocessed dataset to a fixed filename before logging.
                # This ensures we can reliably retrieve it later using a known filename
                # (e.g., during batch evaluation), instead of relying on a random temp file name.
                fixed_pre_path = os.path.join(os.path.dirname(preprocessed_temp_path), "preprocessed_dataset.csv")
                df_pre.to_csv(fixed_pre_path, index=False)
                mlflow.log_artifact(fixed_pre_path, artifact_path="datasets/preprocessed")

                # Train model
                train_accuracy, test_accuracy, trained_model = mlmdel_object.train_and_save_model(df_pre)

                # log metrics
                mlflow.log_metric("train_accuracy", float(train_accuracy))
                mlflow.log_metric("test_accuracy",  float(test_accuracy))

                # log model
                mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature
                )
                
                # registration
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

                # We intentionally skip setting the model version to a specific stage (e.g., 'Staging') here,
                # because stage transitions are now handled by the Airflow DAG.
                # This ensures that all promotion logic — including comparisons of accuracy,
                # version transitions, and archiving — is centralized in one place (the DAG),
                # avoiding conflicts or duplicated logic between the API and the scheduler.
                # As a result, the initially trained model will remain in the 'None' stage
                # until the DAG evaluates and explicitly promotes it if appropriate.
                # The logic for evaluating and promoting the model version is implemented in the Airflow DAG,
                # specifically in the `train_model_dag_without_notification.py` file located at
                # `C:\Users\ActualUser\airflow\dags`.
                # You can refer to that file to see how accuracy comparison, archiving, and staging are handled.

                    # client.transition_model_version_stage(
                    #     name=model_name,
                    #     version=registered_model.version,
                    #     stage=constants.MODEL_STAGE
                    # )

            # Save model locally for tests
            mlmdel_object.save_model(trained_model, constants.API_MODEL_PATH)
            # Save processed dataset snapshot
            df_pre.to_csv(constants.TEMPORARY_API_PROCESSED_DATASET_PATH, index=False)
            # Clean up
            os.remove(temporary_data_path)
            os.remove(preprocessed_temp_path)

            return {
                "message": "Model trained and registered successfully",
                "train_accuracy": float(train_accuracy),
                "test_accuracy":  float(test_accuracy),
                
                # Returning the model version so that the Airflow DAG can decide whether to promote it,
                # compare it to the current staging version, and handle version transitions.
                "model_version":  registered_model.version,
                
                # Returning the run ID can be useful for logging, traceability, or fetching run-specific metrics later.
                "run_id":         run.info.run_id
            }, 200


        except MlflowException as mfe:
            return {'message': 'MLflow Error', 'error': str(mfe)}, 500
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500
        
@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)# This part is responsible to validate data from SwaggerUI
    def post(self): # This part is pure Flask (so it can handle the NOT String format from JSON)
        try:
            data = request.get_json()
           # this was defined upper in 'inference_row': fields.List(fields.Raw, required=True, description='A row of data for inference/prediction')
            if 'inference_row' not in data:
                return {'error': 'No inference_row found'}, 400
            infer_array = data['inference_row']

            if mlmdel_object.model is None:
                return {'error': "No staging model is loaded. Train a model first."}, 400

            # Start MLflow inference run
            run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                # Preprocess input for inference
                # ([infer_array]) -> the list type works because the pipeline expects a 2D list (list of one row)
                # It internally converts it to a DataFrame with correct structure and types
                # inference input for swagger is the following to test:
                # {
                #   "inference_row": [
                #     "7590-VHVEG", "Female", "0", "Yes", "No", "1", "No", "No phone service", "DSL", "No",
                #     "Yes", "No", "No", "No", "No", "Month-to-month", "Yes", "Electronic check", "29.85", "29.85", "No"
                #   ]
                # }
                df_inf = mlmdel_object.preprocessing_pipeline_inference([infer_array])
                # Drop target if present
                if 'Churn_encoded' in df_inf.columns:
                 # WE NEED TO MANUALLY DELETE THE y TARGET HERE, BECAUSE upper the mlmdel_object.train_and_save_model(df) and inside that the fit() method
                 # IS FITTING WITH THE STANDARD METHOD (X (WITHOUT TARGET), y), BUT HERE THE API SENDS ([]) THE WHOLE DATA AND THE preprocessing_pipeline_inference
                 # DOES NOT DEAL WITH IT, THAT JUST PREPROCESSING THE DATA, BUT DOES NOT HANDLE THE FITTING, SO WE HAVE TO DO IT RIGHT HERE
                    df_inf = df_inf.drop(columns=['Churn_encoded'])
                # Check for NaNs
                nan_rows = df_inf[df_inf.isna().any(axis=1)]
                if not nan_rows.empty:
                    print(f'HERE ARE THE NAN ROWS (Predict): {nan_rows}')
                # Predict
                y_pred = mlmdel_object.model.predict(df_inf)

                # Log inference
                mlflow.log_param("inference_input", infer_array)
                mlflow.log_param("inference_output", y_pred.tolist() if hasattr(y_pred, 'tolist') else int(y_pred))

            return {'message': 'Inference Successful', 'prediction': int(y_pred)}, 200

        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500
    
@ns.route('/evaluate_batch')
class EvaluateBatch(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']

        if os.path.splitext(uploaded_file.filename)[1].lower() != '.csv':
            return {'error': 'Invalid file type, only .csv allowed'}, 400

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            uploaded_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load and preprocess current (uploaded) dataset
            df_raw = pd.read_csv(tmp_path)
            df_current = mlmdel_object.preprocessing_pipeline(df_raw)
            os.remove(tmp_path)

            # Load latest model version in the specified stage from MLflow
            latest_model = client.get_latest_versions(name=constants.MODEL_NAME, stages=[constants.MODEL_STAGE])[0]
            run_id = latest_model.run_id

            # Download the preprocessed training dataset from MLflow artifacts
            artifact_dir = client.download_artifacts(run_id, path="datasets/preprocessed")
            reference_path = os.path.join(artifact_dir, "preprocessed_dataset.csv")

            # Load reference dataset
            df_reference = pd.read_csv(reference_path)

            # Create report comparing reference vs current
            create_report(df_current=df_current, df_reference=df_reference)

            return {
                'message': 'Data evaluation completed successfully and is available on the dashboard.'
            }, 200

        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500

@ns.route('/collect_csv_data_for_airflow')
class CollectCsvDataForAirflow(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']

        if Path(uploaded_file.filename).suffix.lower() != '.csv':
            return {'error': 'Invalid file type, only .csv allowed'}, 400

        dest_path = Path(constants.COMPARISON_FOLDER_PATH) / uploaded_file.filename
        uploaded_file.save(dest_path)

        return {'message': 'CSV collected for Airflow', 'filename': uploaded_file.filename}, 200
    
# ─── Add the new Airflow namespace and endpoint ────────────────────────────────────
airflow_ns = api.namespace('airflow', description='Airflow DAG control')

@airflow_ns.route('/trigger_dag')
class TriggerDag(Resource):
    def post(self):
        """
        Trigger an Airflow DAG run via the REST API, passing along the
        filename so the DAG knows which CSV to train on.
        Expects JSON: {"dag_id": "...", "filename": "..."}
        """
        data = request.get_json() or {}
        dag_id = data.get('dag_id')
        fname  = data.get('filename')
        logger.info(f"POST /airflow/trigger_dag received: dag_id={dag_id}, filename={fname}")

        if not dag_id or not fname:
            logger.warning("Missing dag_id or filename in request")
            return {'error': 'Both dag_id and filename are required'}, 400

        airflow_url = f"{constants.AIRFLOW_BASE_URL}api/v1/dags/{dag_id}/dagRuns"
        payload = {'conf': {'filename': fname}}
        try:
            logger.info(f"Triggering DAG at {airflow_url} with payload {payload}")
            resp = requests.post(
                airflow_url,
                auth=(constants.AIRFLOW_USER, constants.AIRFLOW_PASSWORD),
                json=payload
            )
            resp.raise_for_status()
            logger.info(f"DAG {dag_id} triggered successfully: {resp.json()}")
            return {'message': f'DAG {dag_id} triggered', 'run_info': resp.json()}, 200

        except requests.HTTPError as he:
            logger.error(f"HTTP error when triggering DAG {dag_id}: {he.response.text}", exc_info=True)
            return {'error': 'Failed to trigger DAG', 'details': he.response.text}, he.response.status_code
        except Exception as e:
            logger.error(f"Unexpected error when triggering DAG {dag_id}: {e}", exc_info=True)
            return {'error': 'Internal server error'}, 500
    
if __name__ == '__main__':
    # But ON WINDOWS gunicorn is not compatible so use instead waitress (pip install waitress) and run the following code: waitress-serve --listen=127.0.0.1:8000 application:app
    # To kill the API just Ctrl + C in terminal
    # This will run the SWAGGER UI!
    # where from app:app the first one is the name of this .py file and the second one is the flask application name (here it is 'app')
    app.run(host=constants.LOCALHOST_NUMBER, port=constants.PORT_NUMBER_REST_API, debug=False)
