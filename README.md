# Recommendation System for Amazon Products with enhanced BERT classifier

## Description
We download Amazon product reviews data from https://amazon-reviews-2023.github.io/, load them to a postgres database and develop a recommender system.
- embed reviews using a pre-trained embedding BERT or TF-IDF method
- build a sentiment classifier (compare various models amongst random forests, gradient boosting and BERT)
- create an interactive dashboard for data visualization and model performance analysis
- implement the recommendation system
- wrap up the app using FastAPI routes and dockerinzing it

## People
- Luca Nyckees
- Silvio Innocenti-Malini

## Pipeline Snapshot
<img src="https://github.com/LucaNyckees/recommendation_system/blob/main/images/rs_pipeline.png?raw=true" width="700">


## Basic structure
```
├── LICENSE
|
├── config files (.env, .ini, ...)
|
├── README.md
│
├── docs/               
│
├── requirements.txt  
|
├── __main__.py
│
├── src/                
|     ├── __init__.py
|     └── _version.py
|
└── tests/
```

## MLflow Tracking Server Service

> The MLflow tracking server can be started with a service.

**Setup Instructions**

Let's say your project directory looks like `ROOT_DIR := PRE_ROOT_DIR / recommendation_system`.
Go to your **`ROOT_DIR` directory** and do the following.

1. Create a file called `mlflow.service` by copying the content of `mlflow.service.sample` and fill the dots `...` (twice) with adequate info about your folder structure.

2. Copy the service file to `/etc/systemd/system/` with

    ```sh
    sudo cp mlflow.service /etc/systemd/system/.
    ```
3. Reload the services with
    ```sh
    sudo systemctl daemon-reload
    ```
4. Now the service can be started with
    ```sh
    sudo systemctl start mlflow.service
    ```
5. Make sure the service is started on boot by enabling it with
    ```sh
    sudo systemctl enable mlflow.service
    ```
6. Make sure everything works fine by checking logs with
    ```sh
    journalctl -ru mlflow.service
    ```
NB. The command at step 4 has variants with `status/stop/restart` replacing `start`.

## Wrapped up worflow with Docker
First, make sure you have **Docker** installed on your machine. If you wish to make all the steps yourself, without using Docker, you can go to the next section.

**Run the command `make`**, which will basically execute
- `docker compose build`
- `docker compose up`

From those commands, three Docker images shall be created and started.
- `db`: an image for the postgres database
- `migrator`: an image with alembic utilities enabling database versioning
- `app`: an image with python packages and code for running the FastAPI application and the Dash dashboard
You should see the following appear in your terminal.
```
[+] Running 4/4
 ✔ Network recommendation_system_default       Created                                                                                                                   
 ✔ Container recommendation_system-db-1        Healthy                                                                                                                    
 ✔ Container recommendation_system-migrator-1  Exited                                                                                                                    
 ✔ Container recommendation_system-app-1       Started
```
Once this is done, you can access the Dash application by visiting http://0.0.0.0:8050/.

## Step-by-step workflow

Let's say your project directory looks like `ROOT_DIR := PRE_ROOT_DIR / recommendation_system`.

1. Setup a Postgres database and specify your associated credentials in the `config.toml` file.

2. Create a virtual environment named `venv` and install dependencies and requirements in it
```
make venv
```
3. Activate your virtual environment.
```
source venv/bin/activate
```
4. Download Amazon datasets into your project (run from `PRE_ROOT_DIR`).
```
python recommendation_system download datasets
```
5. Load downloaded datasets to your Postgres database (run from `PRE_ROOT_DIR`).
```
python recommendation_system load datasets
```
6. Launch the MLFlow service by following the instructions of the section **MLflow Tracking Server Service** above.

7. Launch the FastAPI application - it creates routes that will be called by the dashboard (run from `ROOT_DIR`).
```
uvicorn src.fastapi_app.main:app
```
8. Launch the Dash dashboard have check the result at http://localhost:8000 in your favorite browser (run from `ROOT_DIR`).
```
python src/dashboard/dashboard.py
```
