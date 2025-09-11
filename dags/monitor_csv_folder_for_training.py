import logging
import os
import requests

import mlflow
from mlflow import MlflowClient

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta

import constants

# ─── Logging configuration ────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── MLflow configuration ─────────────────────────────────────────────────────────
mlflow.set_tracking_uri(constants.ML_FLOW_TRACKING_URI)
client = MlflowClient()

# ─── Failure callback for EmailOperator ───────────────────────────────────────────
def _email_failure_callback(context):
    """
    This callback runs if the EmailOperator (or any task) raises an exception.
    It logs the full exception stacktrace so you can diagnose what went wrong.
    """
    task_id = context['task_instance'].task_id
    exception = context.get('exception')
    logger = logging.getLogger("airflow.task")
    logger.error(f"💣 Task {task_id} FAILED! Exception: {exception}", exc_info=True)

# ─── Helper functions ─────────────────────────────────────────────────────────────

def _get_csv_from_conf(**ctx):
    filename = ctx["dag_run"].conf.get("filename")
    logger.info(f"Retrieving filename from dag_run.conf: {filename!r}")
    if not filename:
        logger.error("No 'filename' provided in dag_run.conf")
        raise ValueError("No 'filename' provided in dag_run.conf")

    path = os.path.join(constants.COMPARISON_FOLDER_PATH, filename)
    logger.info(f"Constructed CSV path: {path}")
    if not os.path.exists(path):
        logger.error(f"CSV not found at expected path: {path}")
        raise FileNotFoundError(f"CSV not found at expected path: {path}")

    logger.info(f"Found CSV file: {path}")
    return path

def _train_and_compare(**ctx):
    ti = ctx["ti"]
    csv_path = ti.xcom_pull(task_ids="get_csv_from_conf")
    logger.info(f"Pulled CSV path from XCom: {csv_path}")

    api_url = f"{constants.REST_API_BASE_URL}model/train"
    logger.info(f"Sending POST to train endpoint: {api_url} with file {os.path.basename(csv_path)}")
    with open(csv_path, "rb") as f:
        resp = requests.post(api_url, files={"file": (os.path.basename(csv_path), f)})
    resp.raise_for_status()
    data = resp.json()
    new_acc = data.get("test_accuracy")
    new_version = data.get("model_version")
    logger.info(f"New model version {new_version} with test_accuracy={new_acc}")

    # compare to current Staging model
    old_versions = client.get_latest_versions(constants.MODEL_NAME, stages=["Staging"])
    old_acc = None
    if old_versions:
        old_ver = old_versions[0]
        old_acc = client.get_metric_history(old_ver.run_id, "test_accuracy")[-1].value
        logger.info(f"Current Staging version {old_ver.version} has test_accuracy={old_acc}")
        if new_acc >= old_acc:
            logger.info(f"New accuracy >= old accuracy, archiving version {old_ver.version}")
            client.transition_model_version_stage(
                name=constants.MODEL_NAME,
                version=old_ver.version,
                stage="Archived"
            )
        else:
            logger.info("New accuracy is lower than old; will trigger notification")

    # promote the new model
    logger.info(f"Promoting new version {new_version} to Staging")
    client.transition_model_version_stage(
        name=constants.MODEL_NAME,
        version=new_version,
        stage="Staging"
    )
    logger.info("Model promotion complete")

    # push metrics for branching
    ti.xcom_push(key="new_acc", value=new_acc)
    ti.xcom_push(key="old_acc", value=old_acc)

def _branch_decision(**ctx):
    ti = ctx["ti"]
    new_acc = ti.xcom_pull(task_ids="train_and_compare_model", key="new_acc")
    old_acc = ti.xcom_pull(task_ids="train_and_compare_model", key="old_acc")
    if old_acc is not None and new_acc < old_acc:
        return "send_notification"
    return "skip_notification"

# ─── DAG definition ───────────────────────────────────────────────────────────────
with DAG(
    dag_id="monitor_csv_folder_for_training",
    start_date=datetime(2025, 6, 20),
    schedule_interval=None,      # only triggered via API
    catchup=False,
    max_active_runs=1,
) as dag:

    get_csv = PythonOperator(
        task_id="get_csv_from_conf",
        python_callable=_get_csv_from_conf,
        provide_context=True,
    )

    train_and_compare = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=_train_and_compare,
        provide_context=True,
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    branch = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=_branch_decision,
        provide_context=True,
    )

    send_notification = EmailOperator(
        task_id="send_notification",
        to=[constants.MY_EMAIL_ADDRESS_TO_SEND_NOTIFICATOIN],
        subject="⚠️ Model Accuracy Alert",
        html_content="""
            <p>The newly trained model (version {{ ti.xcom_pull(task_ids='train_and_compare_model', key='new_acc') }})
            performed worse than the current staging model (accuracy {{ ti.xcom_pull(task_ids='train_and_compare_model', key='old_acc') }}).</p>
            <p>The new version was still promoted, but you may want to investigate.</p>
        """,
        on_failure_callback=_email_failure_callback,  # log any SMTP/email errors here
    )

    skip_notification = PythonOperator(
        task_id="skip_notification",
        python_callable=lambda: logger.info("No notification needed."),
    )

    # ─── Task dependencies ─────────────────────────────────────────────────────────
    get_csv >> train_and_compare >> branch >> [send_notification, skip_notification]
