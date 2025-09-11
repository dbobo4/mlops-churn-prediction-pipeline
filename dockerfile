# 1) Base image: Linux-based Anaconda3
FROM continuumio/anaconda3

# 2) Set the working directory inside the container
WORKDIR /app

# 3) Create required directories and set permissions
RUN mkdir -p /app/mlruns \
    /app/airflow/dags \
    /app/airflow/logs \
    /app/airflow/plugins && \
    chmod -R 777 /app/mlruns /app/airflow

# 4) Copy environment specification and application source code
COPY environment.yml /app/
COPY application.py constants.py MLModel.py utils.py report_generator.py dashboard.py watchdog_csv_uploader.py /app/

# 5) Copy all your DAG files into the Airflow DAGs folder
COPY dags/ /app/airflow/dags/

# 6) Copy the directory where uploaded CSVs are saved
COPY airflow_data_comparison_upload/ /app/airflow_data_comparison_upload/

# ─── the .env.smtp file is already loaded at runtime, do not copy it here! ──────────────────────

# 7) Add conda-forge channel and create the Conda environment
RUN conda config --add channels conda-forge && \
    conda env create -f environment.yml

# 8) Ensure the created conda environment is activated by default
ENV PATH="/opt/conda/envs/airflow_mlops/bin:$PATH"

# 9) Make /app visible to all DAG imports
ENV PYTHONPATH="/app"

# 10) Define core service environment variables
ENV AIRFLOW_HOME="/app/airflow"
ENV MLFLOW_TRACKING_URI="http://0.0.0.0:5102"
ENV ARTIFACT_PATH="file:///app/mlruns"

# 11) Expose the ports used by MLflow Tracking Server, Flask REST API, Streamlit Dashboard, and Airflow Web UI
EXPOSE 5102 8000 8501 8080

# 12) Initialize the Airflow metadata database at build time
RUN bash -c "source activate airflow_mlops && airflow db init"

# 13) Create the Airflow admin user
RUN bash -c "\
    source activate airflow_mlops && \
    airflow users create \
      --username admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email admin@example.com \
      --password admin"

# 14) Start all services on container startup:
#     - Activate Conda
#     - Start MLflow, Airflow, Flask, Streamlit, and watchdog
CMD ["bash","-c","\
    source activate airflow_mlops && \
    mlflow server --host 0.0.0.0 --port 5102 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ${ARTIFACT_PATH} & \
    airflow scheduler & \
    airflow webserver --host 0.0.0.0 --port 8080 & \
    waitress-serve --listen=0.0.0.0:8000 application:app & \
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 & \
    python watchdog_csv_uploader.py \
"]
