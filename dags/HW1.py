import io
import json
import logging
import pickle
import pandas as pd

from datetime import datetime, timedelta
from typing import Any, Dict
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler

from airflow import DAG
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "DenisChuzhmarov"
BUCKET = Variable.get("S3_BUCKET")
S3_CONN_ID = "s3_connection"

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Denis Chuzhmarov",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}


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


def init_task(model_name: str, **kwargs) -> Dict[str, Any]:
    metrics = {
        "init_time": str(datetime.now()),
        "model_name": model_name
    }
    _LOG.info("Train pipeline started.")
    return metrics

def get_data_task(model_name: str, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(key="return_value", task_ids="init")

    start_time = datetime.now()
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing.data, housing.target.rename(TARGET)], axis=1)
    end_time = datetime.now()

    metrics["get_data_start_time"] = str(start_time)
    metrics["get_data_end_time"] = str(end_time)
    metrics["dataset_size"] = data.shape

    path = f"{NAME}/{model_name}/datasets/raw_data.pkl"
    load_data_to_s3(data, path)
    _LOG.info(f"Data saved to s3.")
    return metrics

def prepare_data_task(model_name: str, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(key="return_value", task_ids="get_data")
    
    start_time = datetime.now()
    raw_path = f"{NAME}/{model_name}/datasets/raw_data.pkl"
    data = read_object_from_s3(raw_path)

    # train-test split
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    end_time = datetime.now()

    # Метрики
    metrics["prepare_data_start_time"] = str(start_time)
    metrics["prepare_data_end_time"] = str(end_time)
    metrics["features"] = FEATURES

    # Сохраняем на S3
    dump_object_to_s3(X_train_scaled, f"{NAME}/{model_name}/datasets/X_train.pkl")
    dump_object_to_s3(X_test_scaled, f"{NAME}/{model_name}/datasets/X_test.pkl")
    dump_object_to_s3(y_train, f"{NAME}/{model_name}/datasets/y_train.pkl")
    dump_object_to_s3(y_test, f"{NAME}/{model_name}/datasets/y_test.pkl")

    _LOG.info(f"Data prepared.")
    return metrics

def train_model_task(model_name: str, ml_model, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(key="return_value", task_ids="prepare_data")

    start_time = datetime.now()
    X_train = read_object_from_s3(f"{NAME}/{model_name}/datasets/X_train.pkl")
    X_test = read_object_from_s3(f"{NAME}/{model_name}/datasets/X_test.pkl")
    y_train = read_object_from_s3(f"{NAME}/{model_name}/datasets/y_train.pkl")
    y_test = read_object_from_s3(f"{NAME}/{model_name}/datasets/y_test.pkl")

    ml_model.fit(X_train, y_train)
    preds = ml_model.predict(X_test)

    end_time = datetime.now()

    # Метрики
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = median_absolute_error(y_test, preds)

    metrics["train_start_time"] = str(start_time)
    metrics["train_end_time"] = str(end_time)
    metrics["r2_score"] = r2
    metrics["rmse"] = rmse
    metrics["mae"] = mae

    _LOG.info("Model trained.")
    return metrics

def save_results(model_name: str, **kwargs) -> None:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(key="return_value", task_ids=f"train_model")

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
    _LOG.info("Metrics saved to S3.")


with DAG(
    dag_id="denis_chuzhmarov_linear_regression",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"]
) as dag1:
    
    init_linreg = PythonOperator(
        task_id="init",
        python_callable=init_task,
        op_kwargs={"model_name": "linreg"},
        provide_context=True
    )

    get_data_linreg = PythonOperator(
        task_id="get_data",
        python_callable=get_data_task,
        op_kwargs={"model_name": "linreg"},
        provide_context=True
    )

    prepare_data_linreg = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data_task,
        op_kwargs={"model_name": "linreg"},
        provide_context=True
    )

    train_linreg = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
        op_kwargs={
            "model_name": "linreg",
            "ml_model": LinearRegression()
        },
        provide_context=True
    )

    save_results_linreg = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
        op_kwargs={"model_name": "linreg"},
        provide_context=True
    )

    init_linreg >> get_data_linreg >> prepare_data_linreg >> train_linreg >> save_results_linreg


with DAG(
    dag_id="denis_chuzhmarov_decision_tree",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"]
) as dag2:

    init_tree = PythonOperator(
        task_id="init",
        python_callable=init_task,
        op_kwargs={"model_name": "tree"},
        provide_context=True
    )

    get_data_tree = PythonOperator(
        task_id="get_data",
        python_callable=get_data_task,
        op_kwargs={"model_name": "tree"},
        provide_context=True
    )

    prepare_data_tree = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data_task,
        op_kwargs={"model_name": "tree"},
        provide_context=True
    )

    train_tree = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
        op_kwargs={
            "model_name": "tree",
            "ml_model": DecisionTreeRegressor(random_state=42)
        },
        provide_context=True
    )

    save_results_tree = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
        op_kwargs={"model_name": "tree"},
        provide_context=True
    )

    init_tree >> get_data_tree >> prepare_data_tree >> train_tree >> save_results_tree


with DAG(
    dag_id="denis_chuzhmarov_random_forest",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"]
) as dag3:

    init_forest = PythonOperator(
        task_id="init",
        python_callable=init_task,
        op_kwargs={"model_name": "forest"},
        provide_context=True
    )

    get_data_forest = PythonOperator(
        task_id="get_data",
        python_callable=get_data_task,
        op_kwargs={"model_name": "forest"},
        provide_context=True
    )

    prepare_data_forest = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data_task,
        op_kwargs={"model_name": "forest"},
        provide_context=True
    )

    train_forest = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
        op_kwargs={
            "model_name": "forest",
            "ml_model": RandomForestRegressor(random_state=42)
        },
        provide_context=True
    )

    save_results_forest = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
        op_kwargs={"model_name": "forest"},
        provide_context=True
    )

    init_forest >> get_data_forest >> prepare_data_forest >> train_forest >> save_results_forest
