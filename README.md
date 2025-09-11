# Automated MLOps Pipeline for Churn Prediction

This repository showcases a fully automated **end-to-end MLOps pipeline** built around the  
*Telco Customer Churn* dataset.  

The project demonstrates how modern MLOps tools can be combined into a single, reproducible system,  
covering the entire lifecycle from **data preprocessing, model training, and experiment tracking**  
to **deployment, monitoring, and reporting**.  

All components run inside Docker, providing a portable and production-like environment.  

🔧 **Key technologies integrated:**
- **MLflow** – experiment tracking, model registry, and artifact management  
- **Apache Airflow** – workflow orchestration and automation (DAG-based)  
- **Flask REST API** – serving models and exposing training/inference endpoints (with Swagger UI)  
- **Streamlit Dashboard** – real-time monitoring of data drift, quality checks, and model stability  

This pipeline simulates an industrial MLOps setup where **new data ingestion automatically triggers retraining**,  
performance validation, staging/archiving of models, and **email alerts** if degradations occur.  


## 📂 Project Structure

MLops-Churn-Prediction-Pipeline/
│
├── airflow_data_comparison_upload/          Upload folder watched by watchdog & Airflow
├── dags/                                    Airflow DAG definitions
│   └── monitor_csv_folder_for_training.py
├── reports/                                 Generated Evidently reports (HTML + JSON)
├── tests/                                   Unit tests for training/inference pipelines
│
├── .env.smtp                                SMTP + API auth (ignored in Git)
├── .gitignore                               Ignore rules (env, logs, artifacts)
├── application.py                           Flask REST API (Swagger UI)
├── constants.py                             Centralized constants & paths
├── dashboard.py                             Streamlit dashboard
├── data_understanding_and_data_handling.ipynb   Data exploration + preprocessing
├── docker-compose.yml                       Multi-service startup
├── dockerfile                               Container build definition
├── environment.yml                          Conda environment
├── inference_pipeline.ipynb                 Inference pipeline demo
├── MLModel.py                               ML model class (train, predict, save/load)
├── prediction_request_from_python.ipynb     Example Python client request
├── README.md                                Project documentation
├── report_generator.py                      Evidently report generation
├── train_pipeline.ipynb                     Training pipeline demo
├── utils.py                                 Utility functions
├── watchdog_csv_uploader.py                 Watches upload folder, triggers Airflow DAG
│
├── WA_Fn-UseC_-Telco-Customer-Churn_20.csv  Sample dataset (20%)
├── WA_Fn-UseC_-Telco-Customer-Churn_30.csv  Sample dataset (30%)
└── WA_Fn-UseC_-Telco-Customer-Churn_50.csv  Sample dataset (50%)


## 📑 File Descriptions

### 1. data_understanding_and_data_handling.ipynb

First, of course, the `data_understanding_and_data_handling.ipynb` must be created in order to basically solve the task: feature engineering, model optimization, fine tuning, and setting up the final model parameters.

### 2. utils.py / constants.py

It is very important that from the very beginning we externalize as much as possible into `constants.py`, so that later, if something needs to be modified, it only has to be done in one place. It is naturally advantageous to hardcode here (to avoid surprises later in the pipelines — training and inference). For example, it should not happen that the code does not process columns the way we want, because when it works with the full dataframe it saves all columns, but when it works with just a single row it may save different ones, leading to incorrect transformations later on. To prevent this, it is worth hardcoding the column names we identified during the `data_understanding_and_data_handling` process into this file, and then referencing them in the appropriate parts of the code.

### 3. train_pipeline.ipynb

After this, the `train_pipeline.ipynb` must be created. This is a notebook only as a supporting tool. The reason we need this notebook is that here we must create the code that will later be contained in `MLModel.py`. The point is that this notebook contains, without any data science exploration, only those codes that were in the data understanding and handling phase — the parts that made the data learnable for the model, working with the entire dataframe. Of course, the shorter and clearer it is, the better. Therefore, it is recommended to move the methods used here into a `utils` module, so they can also be reused later in the inference phase.

### 4. inference_pipeline.ipynb

After that, it is worth creating the `inference_pipeline.ipynb` following the pattern of the training pipeline. In this case, however, the code must be designed so that the model can still make predictions in the end, BUT IN THIS CASE NOT FOR THE ENTIRE DATAFRAME, ONLY FOR A SINGLE ROW from it — meaning just one piece of data (later coming from the REST API). The point is that the code of the training pipeline must be transformed here so that it does not convert the entire dataframe into a model-learnable form, but only a single incoming row.

Because of this, it is very important to pay attention from the beginning to how we start the data transformation — the input must be of the same type as what will later come from the API. During processing we must also be careful to avoid differences compared to the original handling (e.g., NaN values appearing because we used code that worked fine on the full dataframe but does not work properly if it only receives one row, causing parameterization errors, etc.).

In the end, for example, in the case of binary classification, it should naturally output either `1` or `0`. Of course, any methods extracted into `utils` can also be reused here, but we must keep in mind that the training pipeline methods are optimized for the entire dataframe.

### 5. MLModel.py

In this part, we create the class from which we will be able to perform everything we did in both the training and inference phases. Therefore, this class must contain both `preprocessing_pipeline` and `preprocessing_pipeline_inference`, which are necessary for transformation. Here we must be careful about what kind of data they will receive and how the transformation should begin.

These two methods do nothing more than receive the data (either a single row for inference, or the full dataframe for training) and transform it into a form the model can process. Naturally, the transformation methods themselves (which were created in `utils.py` for the pipelines) must also be integrated here, so that they can later be called directly, either as static methods or via class instances.

### 6. tests/test_of_train_and_inference_pipeline.py

The `test_of_train_and_inference_pipeline.py` located in the `tests/` folder is a pytest-based test that checks whether the full training and inference pipelines produce the same result. The script loads the `MLModel` class, runs both batch-based training and row-by-row inference processing, and then compares the accuracy values. If there is any discrepancy, the test fails, ensuring that the model’s training and prediction processes work consistently.

### 7. application.py

A Flask + Flask-RESTx based REST API that provides a Swagger UI for testing the endpoints. Through these endpoints you can train a model (`/model/train` – via MLflow), run predictions, generate reports with Evidently (`/model/evaluate_batch`), or trigger an Airflow DAG (`/airflow/trigger_dag`).

For testing, Swagger UI was used manually, but in a real production system the endpoints would typically receive data from automated sources.

This file contains the REST API endpoints. It is important to note that Flask (the web framework handling the endpoints) can only function if it runs behind a server, such as **waitress** on Windows or **gunicorn** on Unix/Mac systems. From here, we can call the methods defined in the `MLModel` class. This layer represents a higher level of abstraction, since it is where we give the commands that tell the system what to do with the data.

### 8. prediction_request_from_python.ipynb

The `prediction_request_from_python.ipynb` file is a simple Jupyter notebook that demonstrates how to call the REST API’s `/model/predict` endpoint directly from Python. It uses the `requests` library to send an HTTP POST request, passing a sample input row (`inference_row`) in JSON format to the API.

For the code to run, the API (e.g., started with waitress) must be actively running in the background. The notebook prints out the status code and the prediction returned by the server. This file is therefore a convenient tool for quickly testing the prediction endpoint and verifying that the REST API is functioning correctly.

### 9. Airflow DAG – dags/monitor_csv_folder_for_training.py

Inside the `dags/` folder there is a single Airflow DAG: `monitor_csv_folder_for_training.py`. This DAG monitors for new CSV files, trains a new model via the REST API, compares it against the current staging model in MLflow, and sends an email notification if a performance drop is detected. The `EmailOperator` SMTP settings are loaded from the `.env.smtp` file.

### 10. dashboard.py

A Streamlit-based visualization interface that displays the reports generated by the `/model/evaluate_batch` endpoint. Separate tabs show the preview of training/evaluation data, the drift report, quality tests, and the stability report. Since Evidently and Streamlit are not compatible within a single file, the report generation was separated into its own module.

The dashboard is launched with the command `streamlit run dashboard.py`, which automatically starts a Tornado-based web server (by default on port 8501), loads the `dashboard.py` file, executes the `st.*` commands inside it, and displays the output in the browser. It is important to note that there is no explicit server startup code inside `dashboard.py` — the server lifecycle is handled entirely by the Streamlit framework. In Docker, the `Dockerfile` and `docker-compose.yml` ensure that this server is also accessible from the host machine at `http://localhost:8501`.

It is essential to note that the dashboard can only function if the DAG has already been triggered at least once, producing a staging model together with its reference dataset. If the DAG has never been triggered, there is no staging model (set to `None`), which means no reference dataset exists — therefore, the dashboard cannot display anything. In this case, the correct workflow is to first upload a CSV file to the `/collect_csv_data_for_airflow` endpoint (as if, in an industrial environment, new data arrived from an external server). Afterwards, the `watchdog_csv_uploader.py` monitoring script automatically detects the upload, triggers the DAG, and creates the first staging model. Only then does the dashboard become usable (which is also the cleaner business approach). In the Swagger UI, the endpoint is visible but cannot be meaningfully used until this process has occurred.

### 11. report_generator.py

This file generates the Evidently HTML reports: drift, quality, and suite. It also saves JSON previews that are consumed by the dashboard. The dashboard and report generation are separated into different files because Streamlit is not compatible with Evidently HTML rendering within the same process.

The data is not loaded directly inside this module; instead, the `create_report(df_current, df_reference)` function receives it as parameters: `df_reference` is the model’s training/preprocessed dataset (downloaded from MLflow), while `df_current` is the newly uploaded CSV. This function is called by the `/model/evaluate_batch` endpoint in `application.py`, ensuring that the reports are always based on a comparison between the reference and current datasets.

It is important to emphasize that the `/evaluate_batch` endpoint does **not** train a new model — it only generates a report comparing the new data with the reference data from the current staging model.

### 12. watchdog_csv_uploader.py

This script monitors the `airflow_data_comparison_upload` folder, and whenever a new CSV appears, it automatically calls the `/airflow/trigger_dag` endpoint. This simulates how the system would immediately react to newly incoming data — as if it were coming from a database. As a result, retraining and evaluation happen in real time.

### 13. environment.yml

Contains all project dependencies: scikit-learn, xgboost, mlflow, streamlit, flask, airflow, evidently, etc.
Based on this file, the Conda environment (`airflow_mlops`) is built inside Docker.

### 14. Dockerfile

Step by step builds the container starting from an Anaconda base image. It prepares the necessary directories, copies in the source code, installs dependencies based on the Conda environment, and finally launches all components: Airflow, MLflow, Flask API, Streamlit, and the watchdog script.

- **Base image:** Starts from the Linux-based `continuumio/anaconda3` image.
- **Working directory:** Sets `/app` as the working directory.
- **Directory preparation:** Creates the MLflow (`/app/mlruns`) and Airflow (`/app/airflow/dags`, `logs`, `plugins`) directories, and assigns proper permissions.
- **Copying source code & configs:** Copies Python files (`application.py`, `MLModel.py`, `constants.py`, etc.), DAGs, the `environment.yml`, and the CSV upload folder into the container. The `.env.smtp` file is intentionally excluded, since it must be provided at runtime.
- **Environment installation:** Adds the `conda-forge` channel and creates the `airflow_mlops` Conda environment from `environment.yml`.
- **Default environment setup:** Updates the PATH so that the `airflow_mlops` environment is active by default.
- **Environment variables:** Configures `AIRFLOW_HOME`, `MLFLOW_TRACKING_URI`, `ARTIFACT_PATH`, and sets `PYTHONPATH` to `/app`.
- **Port exposure:** Opens ports `5102` (MLflow), `8080` (Airflow UI), `8000` (Flask API), and `8501` (Streamlit Dashboard).
- **Airflow initialization:** During the build, runs `airflow db init` and creates an admin user (`admin/admin`).
- **Startup services:** When the container starts, a bash command launches all required services in parallel:
  - MLflow Tracking Server → [http://0.0.0.0:5102](http://0.0.0.0:5102)
  - Airflow Scheduler + Webserver → [http://0.0.0.0:8080](http://0.0.0.0:8080)
  - Flask API (via waitress) → [http://0.0.0.0:8000](http://0.0.0.0:8000)
  - Streamlit Dashboard → [http://0.0.0.0:8501](http://0.0.0.0:8501)
  - Watchdog script → monitors CSV uploads and triggers DAG execution


### 15. docker-compose.yml

This file defines how the entire system should be built and run inside a container. It loads the `.env.smtp` file, exposes the required ports (Airflow, MLflow, API, and dashboard), and builds the image based on the local Dockerfile.

### 16. .env.smtp

This file contains the environment variables required for Airflow email sending and REST API authentication.In Docker mode, it is referenced in `docker-compose.yml` under the `env_file` key, so the values are automatically loaded when the container starts, overriding the default `airflow.cfg` settings.

- It enables Airflow to send emails via SMTP (e.g., if a newly trained model performs worse).
- It enables REST API access using Basic Auth.
- The `.env.smtp` file must be excluded in `.gitignore`, since it contains sensitive data (SMTP user/password, API auth backend).
- It is **not** copied into the Dockerfile; only `docker-compose.yml` references it.
- This means its contents never become part of the image itself, only injected into the runtime container environment.

This separation ensures security by keeping source code and configuration apart.

### 17. Test CSV files (e.g., WA_Fn-UseC_-Telco-Customer-Churn_20.csv)

The number at the end of each filename (20, 30, 50) indicates what percentage of the original dataset it contains. These files are used to test how the pipeline behaves with datasets of different sizes, especially to validate email notification functionality when comparing models.

## Docker Startup


1. Make sure Docker Desktop (or the Docker daemon) is running on your machine.
2. Start the system from the project root (using `docker-compose.yml`):

   ```bash
   docker compose up -d --build
   ```

For a clean rebuild (or if the previous run failed with an error), use:

```
docker compose build --no-cache && docker compose up -d
```

If the command fails, simply retry — sometimes Docker needs a second attempt.


* The `docker-compose.yml` file defines:

  * Ports: `5102` (MLflow), `8080` (Airflow UI), `8000` (Flask API), `8501` (Streamlit Dashboard)
  * `env_file`: `.env.smtp` for sensitive configs
  * Restart policy
  * Image and container names
* Access from the host machine:

  * API (Swagger): http://localhost:8000
  * MLflow UI: http://localhost:5102
  * Airflow UI: http://localhost:8080
  * Dashboard: http://localhost:8501

## Pipeline Usage Steps After Docker Startup

* Once the Docker container has successfully started, open the Swagger UI:
  [http://localhost:8000](http://localhost:8000)
* In Swagger, use the `/model/train` endpoint to train a model.
  For example, upload the file: `WA_Fn-UseC_-Telco-Customer-Churn_50.csv`.
* Next, open the MLflow UI:
  [http://localhost:5102](http://localhost:5102)
* Here you can verify that the experiment and model version were correctly logged.
* In Swagger, test predictions via the `/model/predict` endpoint.
  For example, provide the following valid JSON input:

```json
{
  "inference_row": [
    "7590-VHVEG", "Female", "0", "Yes", "No", "1", "No", "No phone service",
    "DSL", "No", "Yes", "No", "No", "No", "No", "Month-to-month", "Yes",
    "Electronic check", "29.85", "29.85", "No"
  ]
}
```

* This format matches what is validated in the test file
  (tests/test_of_train_and_inference_pipeline.py), ensuring it works correctly.
* According to the business logic, the DAG always decides staging promotion.
  Therefore, the very first trained model remains in the None stage by default.
  Because of this, Airflow must be running to enable the dashboard.

* Upload another CSV to the /model/collect_csv_data_for_airflow endpoint,
  e.g., WA_Fn-UseC_-Telco-Customer-Churn_30.csv.
* Then open the Airflow UI:
  http://localhost:8080
  (Username: admin, Password: admin)

* In the Airflow UI, activate the monitor_csv_folder_for_training DAG
  using the toggle switch next to its name.
  (Sometimes the page needs refreshing, or the data must be re-uploaded,
  to ensure the DAG completes successfully.)
* After the DAG run, you’ll see that all tasks finished.
  If the new model performs worse than the previous one,
  you may also receive an email notification.

* At this point, the /model/evaluate_batch endpoint is available.
  You can upload the same file (e.g., WA_Fn-UseC_-Telco-Customer-Churn_30.csv)
  to generate reports comparing the new data against the staging model’s reference dataset.
  (This process may take a few minutes.)
* Open the Streamlit Dashboard:
  http://localhost:8501

The Dashboard provides four views: Inspect (preview of training and evaluation data), Drift Report, Quality Tests, Suite Report

These views show how the new data differs from the training data of the staging model.


## Pipeline Execution Screenshots

Below are example screenshots captured during the execution of the pipeline components.  
All images are located in the `picture/` folder.

### 1. Swagger UI
![Swagger UI](picture/1_api_swagger_ui.jpg)

### 2. Training Success
![Training Success](picture/2_train_success.jpg)

### 3. MLflow Experiment Log Success
![MLflow Experiment Log Success](picture/3_mlflow_experiment_log_success.jpg)

### 4. Prediction Inference Success
![Prediction Inference Success](picture/4_predict_inference_success.jpg)

### 5. Airflow DAG Presence
![Airflow DAG Presence](picture/5_airflow_dag_presence.jpg)

### 6. DAG Run Success
![DAG Run Success](picture/6_dag_run_success.jpg)

### 7. DAG Graph Success
![DAG Graph Success](picture/7_dag_graph_success.jpg)

### 8. Evaluate Batch Success
![Evaluate Batch Success](picture/8_evaluate_batch_success.jpg)

### 9. Dashboard Inspect Tab
![Dashboard Inspect](picture/9_dashboard_inspect.jpg)

### 10. Dashboard Drift Report
![Dashboard Drift Report](picture/10_dashboard_drift_report.jpg)

### 11. Dashboard Quality Tests
![Dashboard Quality Tests](picture/11_dashboard_quality_tests.jpg)

### 12. Dashboard Suite Report
![Dashboard Suite Report](picture/12_dashboard_suite_report.jpg)

### 13. Push Notification Message
![Push Notification Message](picture/13_push_message.jpg)




👨‍💻 Made by Adam K.