from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

def launch_fastapi():
    from subprocess import Popen
    Popen(["uvicorn", "src.fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"])

def launch_dash():
    from subprocess import Popen
    Popen(["python3.10", "/app/dashboard.py"])

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    'data_pipeline',
    default_args=default_args,
    description='RecSys App Complete Workflow',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    download_data = BashOperator(
        task_id='download_data',
        bash_command='make download-data',
    )
    
    load_data = BashOperator(
        task_id='load_data',
        bash_command='make load-data',
    )
    
    sentiment_analysis = BashOperator(
        task_id='sentiment_analysis',
        bash_command='make sentiment',
    )
    
    launch_mlflow = BashOperator(
        task_id='launch_mlflow',
        bash_command='mlflow server --host 0.0.0.0 --port 5000',
    )
    
    launch_fastapi_task = PythonOperator(
        task_id='launch_fastapi',
        python_callable=launch_fastapi,
    )
    
    launch_dash_task = PythonOperator(
        task_id='launch_dash',
        python_callable=launch_dash,
    )
    
    download_data >> load_data >> sentiment_analysis
    sentiment_analysis >> [launch_mlflow, launch_fastapi_task, launch_dash_task]
