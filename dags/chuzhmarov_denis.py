import mlflow
import os

from airflow.models import DAG
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal

import io
import json
import logging
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


# Настройки

BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    "owner": "Denis Chuzhmarov",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET = "MedHouseVal"

NAME = "DenisChuzhmarov"
BUCKET = Variable.get("S3_BUCKET")
S3_CONN_ID = "s3_connection"

model_names = ["random_forest", "linear_regression", "decision_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


# Полезные функции

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        val = Variable.get(key, default_var=None)
        if val:
            os.environ[key] = val

def load_data_to_s3(df: pd.DataFrame, path: str):
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=path,
        bucket_name=BUCKET,
        replace=True
    )

def dump_object_to_s3(obj, path: str):
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=path,
        bucket_name=BUCKET,
        replace=True
    )

def read_object_from_s3(path: str):
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)
    local_path = s3_hook.download_file(key=path, bucket_name=BUCKET)
    with open(local_path, "rb") as f:
        obj = pickle.load(f)
    return obj


# Пайплайн

def init() -> Dict[str, Any]:
    configure_mlflow()
    experiment_name = "DenisChuzhmarov"  # Имя эксперимента
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # Стартуем родительский run
    with mlflow.start_run(run_name="@DenisChuzhmarov_hw3") as run:
        _LOG.info(f"Started parent run: {run.info.run_id}")

        metrics = {
            "experiment_id": experiment_id,
            "parent_run_id": run.info.run_id,
            "init_time": str(datetime.now())
        }
    return metrics


def get_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")

    start_time = datetime.now()
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing.data, housing.target.rename(TARGET)], axis=1)
    end_time = datetime.now()

    path = f"{NAME}/shared_data/raw_data.pkl"
    load_data_to_s3(data, path)

    metrics["get_data_start"] = str(start_time)
    metrics["get_data_end"] = str(end_time)
    metrics["dataset_size"] = data.shape
    _LOG.info(f"Data saved to s3.")
    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="get_data")

    start_time = datetime.now()
    raw_path = f"{NAME}/shared_data/raw_data.pkl"
    data = read_object_from_s3(raw_path)

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    end_time = datetime.now()

    dump_object_to_s3(X_train_scaled, f"{NAME}/shared_data/X_train.pkl")
    dump_object_to_s3(X_test_scaled, f"{NAME}/shared_data/X_test.pkl")
    dump_object_to_s3(y_train, f"{NAME}/shared_data/y_train.pkl")
    dump_object_to_s3(y_test, f"{NAME}/shared_data/y_test.pkl")

    metrics["prepare_data_start"] = str(start_time)
    metrics["prepare_data_end"] = str(end_time)
    metrics["features"] = FEATURES

    _LOG.info("Data prepared and saved to S3.")
    return metrics


def train_model(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    model_key = kwargs["model_key"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    experiment_id = metrics["experiment_id"]
    parent_run_id = metrics["parent_run_id"]

    configure_mlflow()
    _LOG.info(f"Starting nested run for model={model_key}. Parent run_id={parent_run_id}")

    X_train = read_object_from_s3(f"{NAME}/shared_data/X_train.pkl")
    X_test = read_object_from_s3(f"{NAME}/shared_data/X_test.pkl")
    y_train = read_object_from_s3(f"{NAME}/shared_data/y_train.pkl")
    y_test = read_object_from_s3(f"{NAME}/shared_data/y_test.pkl")

    start_time = datetime.now()

    with mlflow.start_run(experiment_id=experiment_id, run_id=parent_run_id, nested=True) as child_run:
        model = models[model_key]
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        # _LOG.info("NaN в X_test:", np.isnan(X_test).sum())
        # _LOG.info("NaN в y_test:", np.isnan(y_test).sum())
        # _LOG.info("NaN в predictions:", np.isnan(prediction).sum())
        # _LOG.info("Размерность X_test:", X_test.shape)
        # _LOG.info("Размерность y_test:", y_test.shape)
        # _LOG.info("Размерность predictions:", prediction.shape)

        X_test = pd.DataFrame(X_test, columns=FEATURES).reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        prediction = pd.Series(prediction).reset_index(drop=True)

        eval_df = pd.DataFrame(X_test, columns=FEATURES).copy()
        eval_df["target"] = y_test
        eval_df["prediction"] = prediction


        mlflow.evaluate(
            data=eval_df,
            targets="target",
            predictions="prediction",
            model_type="regressor",
            evaluators=["default"]
        )
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_key,
            )
        except Exception as e:
            _LOG.error(f"Произошла ошибка с подключением mlflow к s3: {e}")

    end_time = datetime.now()

    metrics["model_name"] = model_key
    metrics["train_start_time"] = str(start_time)
    metrics["train_end_time"] = str(end_time)

    _LOG.info(f"Model trained, logged and saved")
    return metrics


def save_results(**kwargs) -> None:
    ti = kwargs["ti"]
    rf_metrics = ti.xcom_pull(key="return_value", task_ids="train_rf")
    lr_metrics = ti.xcom_pull(key="return_value", task_ids="train_lr")
    dt_metrics = ti.xcom_pull(key="return_value", task_ids="train_dt")

    for metrics in [rf_metrics, lr_metrics, dt_metrics]:
        model_name = metrics["model_name"]
        results_path = f"{NAME}/{model_name}/results/metrics.json"

        s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)
        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics, indent=2).encode())
        buffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=buffer,
            key=results_path,
            bucket_name=BUCKET,
            replace=True
        )
    _LOG.info("All metrics saved to S3.")


# DAG

with DAG(
    dag_id="denis_chuzhmarov",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"]
) as dag:

    task_init = PythonOperator(
        task_id="init",
        python_callable=init,
        provide_context=True
    )

    task_get_data = PythonOperator(
        task_id="get_data",
        python_callable=get_data,
        provide_context=True
    )

    task_prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
        provide_context=True
    )

    # Три задачи обучения
    train_rf = PythonOperator(
        task_id="train_rf",
        python_callable=train_model,
        op_kwargs={"model_key": "random_forest"},
        provide_context=True
    )
    train_lr = PythonOperator(
        task_id="train_lr",
        python_callable=train_model,
        op_kwargs={"model_key": "linear_regression"},
        provide_context=True
    )
    train_dt = PythonOperator(
        task_id="train_dt",
        python_callable=train_model,
        op_kwargs={"model_key": "decision_tree"},
        provide_context=True
    )

    task_save_results = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
        provide_context=True
    )

    task_init >> task_get_data >> task_prepare_data
    task_prepare_data >> [train_rf, train_lr, train_dt] >> task_save_results
